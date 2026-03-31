from lightning.pytorch.cli import LightningCLI
from genie.dataset import LightningLAMDataModule
from genie.model import DINO_LAM

cli = LightningCLI(
    DINO_LAM,
    LightningLAMDataModule,
    seed_everything_default=42,
)
