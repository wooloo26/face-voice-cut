from fvc.tools.logger import logger
from .interfaces import ISegmentAnalyzer, IFrameMatcher
from .data_models import ClipExtractionState


class BackwardCheckSegmentAnalyzer(ISegmentAnalyzer):
    """Analyzes segments with backward checking for better boundaries."""

    def __init__(self, frame_matcher: IFrameMatcher, enable_backward_check: bool = True):
        self.frame_matcher = frame_matcher
        self.enable_backward_check = enable_backward_check

    def find_segment_start(self, current_frame: int, state: ClipExtractionState) -> int:
        """Find the actual start frame by checking previous frames."""
        if not self.enable_backward_check:
            return current_frame

        recent_frames = state.recent_frames
        if not recent_frames:
            return current_frame

        frame_rate = state.frame_rate
        logger.info(
            f"Checking {len(recent_frames)} previous frames to find actual segment start..."
        )

        earliest_match_frame = current_frame

        for frame_num, frame in reversed(recent_frames):
            if frame_num >= current_frame:
                continue

            try:
                match_result = self.frame_matcher.match(frame, frame_num, frame_rate)

                if match_result.has_match:
                    earliest_match_frame = frame_num
                    logger.debug(
                        f"Found matching frame at {frame_num} "
                        f"(time: {match_result.time_sec:.2f}s, similarity: {match_result.similarity:.3f})"
                    )
                else:
                    break

            except Exception as e:
                logger.warning(f"Error checking frame {frame_num}: {e}")
                break

        if earliest_match_frame != current_frame:
            time_diff = (current_frame - earliest_match_frame) / frame_rate
            logger.info(
                f"Found earlier start frame: {earliest_match_frame} (saved {time_diff:.2f}s)"
            )

        return earliest_match_frame

    def find_segment_end(self, current_frame: int, state: ClipExtractionState) -> int:
        """Find the actual end frame by checking recent frames."""
        if not self.enable_backward_check:
            return current_frame

        recent_frames = state.recent_frames
        if not recent_frames:
            return current_frame

        frame_rate = state.frame_rate
        logger.info("Checking recent frames to find actual segment end...")

        latest_match_frame = current_frame

        for frame_num, frame in recent_frames:
            if frame_num >= current_frame:
                continue

            try:
                match_result = self.frame_matcher.match(frame, frame_num, frame_rate)

                if match_result.has_match:
                    latest_match_frame = frame_num
                    logger.debug(
                        f"Found matching frame at {frame_num} "
                        f"(time: {match_result.time_sec:.2f}s, similarity: {match_result.similarity:.3f})"
                    )

            except Exception as e:
                logger.warning(f"Error checking frame {frame_num}: {e}")
                continue

        if latest_match_frame != current_frame:
            time_diff = (current_frame - latest_match_frame) / frame_rate
            logger.info(
                f"Found actual end frame: {latest_match_frame} (trimmed {time_diff:.2f}s from end)"
            )

        return latest_match_frame


class SimpleSegmentAnalyzer(ISegmentAnalyzer):
    """Simple segment analyzer without backward checking."""

    def find_segment_start(self, current_frame: int, state: ClipExtractionState) -> int:
        """Return the current frame as segment start."""
        return current_frame

    def find_segment_end(self, current_frame: int, state: ClipExtractionState) -> int:
        """Return the current frame as segment end."""
        return current_frame
