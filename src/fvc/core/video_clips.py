import cv2
from pathlib import Path
from typing import List, Optional, Tuple

from fvc.core.frame_matcher import FrameMatcher
from fvc.tools.config import Config
from fvc.tools.logger import logger
from .data_models import ClipSegment, ClipExtractionState
from .interfaces import (
    IFrameMatcher,
    IVideoFileProcessor,
    IClipExtractor,
    ISegmentAnalyzer,
    IProgressLogger,
    IStateManager,
)
from .video_processor import VideoFileProcessor
from .clip_extractor import MoviePyClipExtractor
from .segment_analyzer import BackwardCheckSegmentAnalyzer, SimpleSegmentAnalyzer
from .progress_logger import StandardProgressLogger
from .state_manager import ClipExtractionStateManager


class VideoClipExtractor:
    """Extracts video clips based on face recognition matching."""

    # Default video extensions to process
    DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm")

    def __init__(
        self,
        target_face_embeddings_filename: str,
        frames_per_second: int = 1,
        use_gpu: bool = True,
        max_consecutive_missing_frames: Optional[int] = None,
        min_clip_duration: Optional[float] = None,
        start_time: int = 0,
        enable_backward_check: bool = True,
        video_extensions: Optional[Tuple[str, ...]] = None,
        # Dependency injection parameters
        frame_matcher: Optional[IFrameMatcher] = None,
        video_processor: Optional[IVideoFileProcessor] = None,
        clip_extractor: Optional[IClipExtractor] = None,
        segment_analyzer: Optional[ISegmentAnalyzer] = None,
        progress_logger: Optional[IProgressLogger] = None,
        state_manager: Optional[IStateManager] = None,
    ):
        """
        Initialize the VideoClipExtractor with dependency injection.

        Args:
            target_face_embeddings_filename: Path to face embeddings file
            frames_per_second: Number of frames to process per second
            use_gpu: Whether to use GPU acceleration
            max_consecutive_missing_frames: Max frames without match before ending segment
            min_clip_duration: Minimum duration for extracted clips
            start_time: Start time in seconds for processing
            enable_backward_check: Whether to check previous frames for better boundaries
            video_extensions: Supported video file extensions
            frame_matcher: Injectable frame matcher (optional)
            video_processor: Injectable video processor (optional)
            clip_extractor: Injectable clip extractor (optional)
            segment_analyzer: Injectable segment analyzer (optional)
            progress_logger: Injectable progress logger (optional)
            state_manager: Injectable state manager (optional)
        """
        # Core settings
        self.target_face_embeddings_filename = target_face_embeddings_filename
        self.frames_per_second = frames_per_second
        self.use_gpu = use_gpu
        self.max_consecutive_missing_frames = (
            max_consecutive_missing_frames or Config.MAX_CONSECUTIVE_MISSING_FRAMES
        )
        self.min_clip_duration = min_clip_duration or Config.MIN_CLIP_DURATION
        self.start_time = start_time
        self.enable_backward_check = enable_backward_check
        self.video_extensions = video_extensions or self.DEFAULT_VIDEO_EXTENSIONS

        # Dependency injection with default implementations
        self.frame_matcher = frame_matcher or FrameMatcher(
            self.target_face_embeddings_filename, self.use_gpu
        )
        self.video_processor = video_processor or VideoFileProcessor(self.video_extensions)
        self.clip_extractor = clip_extractor or MoviePyClipExtractor()

        if enable_backward_check:
            self.segment_analyzer = segment_analyzer or BackwardCheckSegmentAnalyzer(
                self.frame_matcher, enable_backward_check
            )
        else:
            self.segment_analyzer = segment_analyzer or SimpleSegmentAnalyzer()

        self.progress_logger = progress_logger or StandardProgressLogger()
        self.state_manager = state_manager or ClipExtractionStateManager(
            self.segment_analyzer,
            self.clip_extractor,
            self.min_clip_duration,
            self.max_consecutive_missing_frames,
        )

    def _process_video_files(self, video_files: List[Path], output_path: Path) -> dict[str, int]:
        """Process multiple video files and return statistics."""
        successful_count = 0
        failed_count = 0
        total_clips_extracted = 0

        for i, video_file in enumerate(video_files, 1):
            logger.info(f"Processing video [{i}/{len(video_files)}]: {video_file.name}")

            try:
                clips_count = self._process_single_video(video_file, output_path)
                if clips_count >= 0:
                    successful_count += 1
                    total_clips_extracted += clips_count
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                logger.error(f"Failed to process '{video_file.name}': {e}", exc_info=True)

        return {
            "successful": successful_count,
            "failed": failed_count,
            "total_clips": total_clips_extracted,
        }

    def _process_single_video(self, video_file: Path, output_path: Path) -> int:
        """Process a single video file and return the number of clips extracted."""
        video_title = video_file.stem
        video_output_folder = output_path / video_title

        if video_output_folder.exists():
            existing_clips = list(video_output_folder.glob("*_clip_*.mp4"))
            if existing_clips:
                logger.info(
                    f"Video '{video_file.name}' already processed "
                    f"({len(existing_clips)} clips found) - skipping"
                )
                return len(existing_clips)

        extracted_clips = self.extract_clips_from_video(str(video_file), str(video_output_folder))
        logger.info(
            f"Successfully processed: {video_file.name} ({len(extracted_clips)} clips extracted)"
        )
        return len(extracted_clips)

    def extract_clips_from_video(self, video_path: str, output_folder: str) -> List[ClipSegment]:
        """
        Extract clips from a single video file, using a temp file with only video/audio streams.

        Args:
            video_path: Path to the video file
            output_folder: Path to output folder for extracted clips

        Returns:
            List of extracted clip segments
        """
        import subprocess
        import tempfile

        if self.frame_matcher.get_embeddings_size() == 0:
            logger.warning("No face embeddings found in reference file - exiting video processing")
            return []

        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        video_title = Path(video_path).stem
        logger.info(f"Starting video clip extraction: {video_title}")

        temp_dir = tempfile.gettempdir()
        temp_video_path = str(Path(temp_dir) / f"{video_title}_va_temp.mp4")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-map",
            "0:v?",
            "-map",
            "0:a?",
            "-c",
            "copy",
            temp_video_path,
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Unable to open temp video file: '{temp_video_path}'")

            try:
                video_info = self.video_processor.get_video_info(cap)
                settings = {
                    "max_consecutive_missing_frames": self.max_consecutive_missing_frames,
                    "min_clip_duration": self.min_clip_duration,
                }
                self.progress_logger.log_video_specs(video_info, settings)

                interval, frame_count = self._prepare_video_processing(
                    cap, video_info["frame_rate"]
                )

                extracted_clips = self._identify_and_extract_clips_realtime(
                    cap,
                    video_info["frame_rate"],
                    interval,
                    frame_count,
                    temp_video_path,
                    output_folder,
                    video_title,
                )

                logger.info(
                    f"Video clip extraction completed! Extracted {len(extracted_clips)} clips"
                )
                return extracted_clips

            finally:
                cap.release()
        finally:
            try:
                if Path(temp_video_path).exists():
                    Path(temp_video_path).unlink()
            except Exception as e:
                logger.warning(f"Failed to remove temp video file: {temp_video_path} ({e})")

    def _prepare_video_processing(
        self, cap: cv2.VideoCapture, frame_rate: float
    ) -> Tuple[int, int]:
        """Prepare video processing by setting interval and start position."""
        interval = max(1, int(frame_rate / self.frames_per_second))
        frame_count = 0

        if self.start_time != 0:
            frame_count = int(self.start_time * frame_rate)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        return interval, frame_count

    def extract_clips_from_folder(self, videos_folder: str, output_folder: str) -> None:
        """
        Extract clips from all video files in a folder.

        Args:
            videos_folder: Path to folder containing video files
            output_folder: Path to output folder for extracted clips
        """
        video_files = self.video_processor.get_video_files(videos_folder)
        if not video_files:
            return

        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        stats = self._process_video_files(video_files, output_path)
        self.progress_logger.log_batch_results(stats, output_folder)

    def _identify_and_extract_clips_realtime(
        self,
        cap: cv2.VideoCapture,
        frame_rate: float,
        interval: int,
        start_frame: int,
        video_path: str,
        output_folder: str,
        video_title: str,
    ) -> List[ClipSegment]:
        """Identify and extract clips in real-time while processing video frames."""
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        state = ClipExtractionState(
            frame_rate=frame_rate, interval=interval, frame_count=start_frame
        )

        logger.info("Analyzing frames and extracting clips in real-time...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if self.enable_backward_check:
                if len(state.recent_frames) >= interval:
                    state.recent_frames.pop(0)
                state.recent_frames.append((state.frame_count, frame.copy()))

            if state.frame_count % interval == 0:
                self._process_frame(
                    frame, state, frame_rate, interval, video_path, output_path, video_title
                )

            state.frame_count += 1

        self.state_manager.finalize_segment(state, frame_rate, video_path, output_path, video_title)

        logger.info(
            f"Real-time extraction completed! Extracted {len(state.extracted_clips)} clips "
            f"from {state.processed_frames:,} processed frames"
        )
        return state.extracted_clips

    def _process_frame(
        self,
        frame: cv2.typing.MatLike,
        state: ClipExtractionState,
        frame_rate: float,
        interval: int,
        video_path: str,
        output_path: Path,
        video_title: str,
    ) -> None:
        """Process a single frame and update extraction state."""
        match_result = self.frame_matcher.match(frame, state.frame_count, frame_rate)
        state.processed_frames += 1

        if match_result.has_match:
            self.state_manager.handle_match_found(state, match_result)
        else:
            self.state_manager.handle_no_match(
                state, frame_rate, interval, video_path, output_path, video_title
            )

        self.progress_logger.log_progress(state, frame_rate)
