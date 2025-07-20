from typing import Dict, Any

from fvc.tools.logger import logger
from .interfaces import IProgressLogger
from .data_models import ClipExtractionState


def _format_time(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class StandardProgressLogger(IProgressLogger):
    """Standard implementation of progress logging."""

    def log_video_specs(self, video_info: Dict[str, float], settings: Dict[str, Any]) -> None:
        """Log video specifications and processing settings."""
        logger.info(
            f"Video specs - FPS: {video_info['frame_rate']:.2f} | "
            f"Total frames: {video_info['total_frames']:,} | "
            f"Duration: {video_info['duration']:.2f}s"
        )
        logger.info(
            f"Tolerance settings - Max missing frames: {settings['max_consecutive_missing_frames']} | "
            f"Min clip duration: {settings['min_clip_duration']}s"
        )

    def log_progress(self, state: ClipExtractionState, frame_rate: float) -> None:
        """Log progress information at regular intervals."""
        if state.processed_frames % 100 == 0:
            progress_time = state.frame_count / frame_rate
            formatted_time = _format_time(progress_time)
            current_segment_status = "Active" if state.current_segment_start else "None"
            logger.info(
                f"Progress: Time: {formatted_time} | "
                f"Current segment: {current_segment_status} | "
                f"Clips extracted: {len(state.extracted_clips)}"
            )

    def log_batch_results(self, stats: Dict[str, int], output_folder: str) -> None:
        """Log the results of batch processing."""
        logger.info("Batch clip extraction completed!")
        logger.info(f"Successfully processed: {stats['successful']} videos")
        logger.info(f"Failed to process: {stats['failed']} videos")
        logger.info(f"Total clips extracted: {stats['total_clips']}")
        logger.info(f"Output directory: {output_folder}")
