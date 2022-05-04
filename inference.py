from data import build_data
from model import build_model
from configs import build_config

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

if __name__ == "__main__":
    seed_everything(0, workers=True)
    cfg = build_config()
    dataset = build_data(cfg)
    model = build_model(cfg)
    trainer = Trainer(
        gpus=cfg.NUM_GPUS, 
        accelerator="gpu",
        strategy=DDPPlugin(find_unused_parameters=False),
        callbacks=[
            LearningRateMonitor(logging_interval='step'), 
            ModelCheckpoint()
        ],
        benchmark=False, 
        deterministic=True,
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        default_root_dir=cfg.OUTPUT_DIR,
        check_val_every_n_epoch=cfg.CHECK_VAL_EVERY_N_EPOCH,
        num_sanity_val_steps=0,
    )
    trainer.validate(model, datamodule=dataset, 
        ckpt_path=cfg.CKPT if hasattr(cfg, "CKPT") else None)