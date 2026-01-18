import torch
import torchvision.transforms as transforms
import logging
from ema import EMA
import math
from dataclasses import asdict
from pathlib import Path


class TrainModel:
    def __init__(self, model, train_loader, test_loader, device, cfg):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.cfg = cfg
        self.global_step = 0

        # AMP toggle
        use_amp = getattr(self.cfg, "use_amp", True)
        self.amp_enabled = (device.type == "cuda") and use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        crop_size = getattr(self.cfg, "crop_size", 224)
        self.fivecrop = transforms.FiveCrop(crop_size)

        # ---- logger ----
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Ensure models directory exists
        Path("../models").mkdir(parents=True, exist_ok=True)



    @torch.no_grad()
    def evaluate_flip_tta(self, ema = False):
        """Horizontal flip TTA - fast."""

        if ema:
            self.logger.info("Evaluating with EMA weights.")
            ema.apply_to(self.model)

        self.model.eval()
        correct, total = 0, 0

        for imgs, labels in self.test_loader:
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                logits = (self.model(imgs) + self.model(torch.flip(imgs, dims=[3]))) / 2.0

            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        if ema:
            ema.restore(self.model)
        
        return correct / max(1, total)

    @torch.no_grad()
    def evaluate_tta10(self):
        """5-crop + flip = 10-view TTA - slower, usually a bit better."""
        self.model.eval()
        correct, total = 0, 0

        for i, (imgs, labels) in enumerate(self.test_loader):
            if i % 10 == 0:
                self.logger.info("TTA10 eval batch %d/%d", i, len(self.test_loader))

            # imgs stays on CPU until we build the 10-view batch, then we move once
            labels = labels.to(self.device, non_blocking=True)

            crops = self.fivecrop(imgs)  # tuple length 5, each [B,C,H,W]
            views = []
            for c in crops:
                views.append(c)
                views.append(torch.flip(c, dims=[3]))

            v = torch.cat(views, dim=0).to(self.device, non_blocking=True)  # [10B,C,H,W]

            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                logits = self.model(v)  # [10B,num_classes]

            logits = logits.view(10, labels.size(0), -1).mean(dim=0)  # [B,num_classes]
            preds = logits.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return correct / max(1, total)
    
    def save_model(self, path):
        """Saves the model state dict to the specified path."""
        torch.save({"model": self.model.state_dict(), "config": asdict(self.cfg)}, path)

    def load_model(self, path):
        """Loads the model state dict from the specified path."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])

    def make_optimizer(self, lr, wd):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    def cosine_lr(self, step, total_steps, lr_max, lr_min):
        t = step / max(1, total_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t))

    def train_epoch(self, train_loader, criterion, optimizer, ema, lr_max, total_steps):
        """Trains the model for one epoch. Returns (loss_epoch, train_acc)."""
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # cosine LR schedule per step
            lr = self.cosine_lr(
                step=self.global_step,
                total_steps=total_steps,
                lr_max=lr_max,
                lr_min=lr_max * 0.02,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                outputs = self.model(imgs)
                loss = criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            # EMA update every step
            if ema is not None:
                ema.update(self.model)

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            self.global_step += 1

        loss_epoch = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        return loss_epoch, train_acc

    def train_stage(self, stage_name, train_loader, epochs, lr_max, wd, criterion):
        self.set_stage(stage_name)

        self.logger.info("Starting stage '%s' (%d epochs)", stage_name, epochs)

        optimizer = self.make_optimizer(lr_max, wd)
        ema = EMA(self.model, decay=self.cfg.ema_decay)  # or None if you want it optional

        total_steps = epochs * len(train_loader)
        best = 0.0

        for epoch in range(1, epochs + 1):
            loss_epoch, train_acc = self.train_epoch(
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                ema=ema,
                lr_max=lr_max,
                total_steps=total_steps,
            )

            # Eval using EMA weights (fast flip TTA)
            acc = self.evaluate_flip_tta(ema=True)

            if acc > best:
                best = acc
                self.logger.info(
                    "New best model for stage '%s': acc=%.4f → saving checkpoint",
                    stage_name,
                    best,
                )
                self.save_model(f"../models/{stage_name}_best_model.pth")

                # Also save EMA weights version
                ema.apply_to(self.model)
                self.save_model(f"../models/{stage_name}_with_ema_best_model.pth")
                ema.restore(self.model)

            
            self.logger.info(
                "[%s %02d/%d] loss=%.4f train_acc=%.4f test_acc(flipTTA)=%.4f best=%.4f",
                stage_name,
                epoch,
                epochs,
                loss_epoch,
                train_acc,
                acc,
                best,
            )

        return best

    def _freeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def _unfreeze(self, module):
        for p in module.parameters():
            p.requires_grad = True

    def set_stage(self, stage_name):
        self._freeze_all()
        self._unfreeze(self.model.fc)

        if stage_name in ("layer4", "layer3", "polish"):
            self._unfreeze(self.model.layer4)

        if stage_name in ("layer3", "polish"):
            self._unfreeze(self.model.layer3)






            



        