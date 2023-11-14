from dataclasses import dataclass


@dataclass
class DatasetConfig:
    root_dir: str = "",
    target_length: int = 2048,
    hflip: bool = False,
    vflip: bool = False,
    flip_rate: float = 0.9,
    randshift: bool = False,
    seed: int = 42,
    sep_tok: bool = False,
    normalize: bool = False,
    num_classes: int = 1,
    flat: bool = False
