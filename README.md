# Automated_Material_Recognition

Need to check requirements install tbh
Note: Where you should put the data by default


# TODO

Review perhaps excessive logging - Dataloader

Could add ensemble - only once the codebase is clean

Setup logger in main, and pass this into components

torch.save({
    "model": model.state_dict(),
    "config": cfg.as_dict(),
}, "checkpoint.pt")


Kimmins - Thoughts on oversampling the challenging images?

TensorBoard - use for showing model as it trains
Visualing Augs