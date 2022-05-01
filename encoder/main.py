from data import build_data
from model import build_model
from configs import build_config

from pytorch_lightning import Trainer, seed_everything

if __name__ == "__main__":
    seed_everything(42, workers=True)
    cfg = build_config()
    data = build_data(cfg)
    model = build_model(cfg)
    trainer = Trainer(
        gpus=1, 
        accelerator="gpu",
        benchmark=False, 
        deterministic=True
    )
    trainer.test(model, datamodule=data)