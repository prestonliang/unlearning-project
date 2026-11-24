from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: str = "full"
    target_class: int | None = None
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 42
    data_dir: str = "./data"
    output_dir: str = "./"
    num_workers: int = 2

    def to_dict(self) -> dict:
        return asdict(self)
