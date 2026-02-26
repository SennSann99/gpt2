# main.py (recommended)
import tiktoken
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from pl_src.config import ModelConfig, TrainingConfig
from pl_src.model import GPTLightning
from pl_src.datamodule import GPTDataModule

def main():
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()

    tokenizer = tiktoken.get_encoding("gpt2")
    datamodule = GPTDataModule(train_cfg, tokenizer, model_cfg)

    model = GPTLightning(model_cfg)

    logger = TensorBoardLogger("tb_logs", name="gpt_experiment")
    ckpt = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=3, mode="min")

    trainer = pl.Trainer(
        max_epochs=train_cfg.num_epochs,
        accelerator="auto",
        devices="auto",
        precision=16 if pl.utilities.device_parser.determine_root_gpu_device() is not None else 32,
        logger=logger,
        callbacks=[ckpt],
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()