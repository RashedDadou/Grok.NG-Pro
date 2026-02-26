# generation_result.py

from dataclasses import dataclass, field
from typing import Dict, Optional, Any

@dataclass
class GenerationResult:
    success: bool = False
    message: str = ""
    total_time: float = 0.0
    layer: Optional[Any] = None          # أو PlaneLayer لاحقًا
    output_data: Dict[str, Any] = field(default_factory=dict)
    specialization: str = "environment"
    is_video: bool = False
    stage_times: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.stage_times is None:
            self.stage_times = {}