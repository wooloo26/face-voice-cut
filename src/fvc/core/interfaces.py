import cv2
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any

from .data_models import ClipSegment, ClipExtractionState, FrameMatchResult


class IFrameMatcher(ABC):
    """Interface for frame matching functionality."""

    @abstractmethod
    def match(
        self, frame: cv2.typing.MatLike, frame_count: int, frame_rate: float
    ) -> FrameMatchResult:
        """Check if frame matches target criteria."""
        pass

    @abstractmethod
    def get_embeddings_size(self) -> int:
        """Get the size of the embeddings."""
        pass


class IVideoFileProcessor(ABC):
    """Interface for processing video files."""

    @abstractmethod
    def get_video_files(self, folder_path: str) -> List[Path]:
        """Get all video files from a folder."""
        pass

    @abstractmethod
    def get_video_info(self, cap: cv2.VideoCapture) -> Dict[str, float]:
        """Extract video information."""
        pass


class IClipExtractor(ABC):
    """Interface for extracting video clips."""

    @abstractmethod
    def extract_clip(
        self,
        video_path: str,
        segment: ClipSegment,
        output_path: Path,
        video_title: str,
        clip_number: int,
    ) -> bool:
        """Extract a single video clip."""
        pass


class ISegmentAnalyzer(ABC):
    """Interface for analyzing segments during extraction."""

    @abstractmethod
    def find_segment_start(self, current_frame: int, state: ClipExtractionState) -> int:
        """Find the actual start of a segment."""
        pass

    @abstractmethod
    def find_segment_end(self, current_frame: int, state: ClipExtractionState) -> int:
        """Find the actual end of a segment."""
        pass


class IProgressLogger(ABC):
    """Interface for logging progress during extraction."""

    @abstractmethod
    def log_video_specs(self, video_info: Dict[str, float], settings: Dict[str, Any]) -> None:
        """Log video specifications."""
        pass

    @abstractmethod
    def log_progress(self, state: ClipExtractionState, frame_rate: float) -> None:
        """Log extraction progress."""
        pass

    @abstractmethod
    def log_batch_results(self, stats: Dict[str, int], output_folder: str) -> None:
        """Log batch processing results."""
        pass


class IStateManager(ABC):
    """Interface for managing extraction state."""

    @abstractmethod
    def handle_match_found(
        self, state: ClipExtractionState, match_result: FrameMatchResult
    ) -> None:
        """Handle when a match is found."""
        pass

    @abstractmethod
    def handle_no_match(
        self,
        state: ClipExtractionState,
        frame_rate: float,
        interval: int,
        video_path: str,
        output_path: Path,
        video_title: str,
    ) -> None:
        """Handle when no match is found."""
        pass

    @abstractmethod
    def finalize_segment(
        self,
        state: ClipExtractionState,
        frame_rate: float,
        video_path: str,
        output_path: Path,
        video_title: str,
    ) -> None:
        """Finalize the last segment."""
        pass
