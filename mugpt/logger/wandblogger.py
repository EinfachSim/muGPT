import wandb
from .base import BaseLogger


class WandBLogger(BaseLogger):
    def __init__(self, project: str, config: dict = None, run_name: str = None):
        self.run = wandb.init(
            project=project,
            config=config,
            name=run_name,
        )

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.run.log(metrics, step=step)

    def close(self) -> None:
        self.run.finish()