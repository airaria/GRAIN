from dataclasses import asdict
import torch
import json
import logging
from typing import Union, Optional
from dataclasses import dataclass, asdict

from ..configurations import Config
logger = logging.getLogger(__name__)


@dataclass
class FineGrainedPruningConfig(Config):
    """
    Configurations for transformer pruning.
    """

    target_QK_head_size: Optional[int] = None
    target_VO_head_size: Optional[int] = None
    pruning_method : str = 'masks'
    n_iters : Optional[int] = 1
    multiple_of : int = 1
    use_logits : bool = False
    config_class: str = "FineGrainedPruningConfig"
    def __post_init__(self):
        assert self.pruning_method in ('masks','iterative'), "Unrecgonized pruning method"