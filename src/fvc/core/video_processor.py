import cv2
from pathlib import Path
from typing import List, Dict

from fvc.tools.logger import logger
from .interfaces import IVideoFileProcessor


class VideoFileProcessor(IVideoFileProcessor):
    """Handles video file discovery and information extraction."""

    def __init__(self, video_extensions: tuple[str, ...]):
        self.video_extensions = video_extensions

    def get_video_files(self, folder_path: str) -> List[Path]:
        """Get all video files from the specified folder."""
        videos_path = Path(folder_path)
        if not videos_path.exists():
            raise FileNotFoundError(f"Video folder not found: {folder_path}")

        video_files: List[Path] = []
        for ext in self.video_extensions:
            for file_path in videos_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == ext.lower():
                    video_files.append(file_path)

        if not video_files:
            logger.warning(f"No video files found in directory: {folder_path}")
            return []

        logger.info(f"Found {len(video_files)} video file(s) to process for clip extraction")
        return video_files

    def get_video_info(self, cap: cv2.VideoCapture) -> Dict[str, float]:
        """Extract video information from OpenCV VideoCapture."""
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / frame_rate if frame_rate > 0 else 0

        return {"frame_rate": frame_rate, "total_frames": total_frames, "duration": duration}
