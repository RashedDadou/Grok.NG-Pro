# layer_engine.py
from abc import ABC, abstractmethod
from typing import Tuple, Any
from Image_generation import GenerationResult  # تأكد من المسار الصحيح

class LayerEngine(ABC):
    @abstractmethod
    def generate_layer(
        self,
        prompt: str,
        target_size: Tuple[int, int] = (1024, 1024),
        is_video: bool = False,
        as_layer: bool = True,
        force_refresh: bool = False,
        **kwargs: Any
    ) -> GenerationResult:
        pass