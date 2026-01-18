from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

@dataclass
class Config:

    # Training configuration
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

    batch_size_train: int = 64
    batch_size_test: int = 128
    batch_size_test_tta: int = 16
    num_workers: int = 4

    epochs_stages: Tuple[int, int, int, int] = (5, 8, 8, 10)
    ema_decay: float = 0.999

    use_amp: bool = True
    crop_size: int = 224

    @property
    def stages(self) -> List[Dict[str, Any]]:
        e0, e1, e2, e3 = self.epochs_stages
        return [
            dict(name="head",   epochs=e0, lr_max=2e-3, wd=0.02,  label_smoothing=0.10, use_class_weights=True),
            dict(name="layer4", epochs=e1, lr_max=5e-4, wd=0.03,  label_smoothing=0.10, use_class_weights=True),
            dict(name="layer3", epochs=e2, lr_max=2e-4, wd=0.05,  label_smoothing=0.08, use_class_weights=True),
            dict(name="polish", epochs=e3, lr_max=8e-5, wd=0.015, label_smoothing=0.02, use_class_weights=False),
        ]
    
    # Logging configuration
    log_level: int = 20  # INFO
    log_file: str = "training.log"
