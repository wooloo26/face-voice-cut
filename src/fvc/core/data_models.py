from dataclasses import dataclass, field
from typing import Any, List, NamedTuple, Optional, Tuple


class FrameMatchResult(NamedTuple):
    frame_count: int
    similarity: float
    has_match: bool
    time_sec: float


class ClipSegment(NamedTuple):
    """Represents a video clip segment with frame and time information."""

    start_frame: int
    end_frame: int
    start_time: float
    end_time: float


@dataclass
class ClipExtractionState:
    """Maintains state during video clip extraction process."""

    extracted_clips: List[ClipSegment] = field(default_factory=list)
    current_segment_start: Optional[int] = None
    consecutive_missing: int = 0
    frame_count: int = 0
    processed_frames: int = 0
    clip_counter: int = 1
    frame_rate: float = 0.0
    interval: int = 1
    recent_frames: List[Tuple[int, Any]] = field(default_factory=list)
