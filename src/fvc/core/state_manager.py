from pathlib import Path

from fvc.tools.logger import logger
from .interfaces import IStateManager, ISegmentAnalyzer, IClipExtractor
from .data_models import ClipExtractionState, ClipSegment, FrameMatchResult


class ClipExtractionStateManager(IStateManager):
    """Manages state during clip extraction process."""

    def __init__(
        self,
        segment_analyzer: ISegmentAnalyzer,
        clip_extractor: IClipExtractor,
        min_clip_duration: float,
        max_consecutive_missing_frames: int,
    ):
        self.segment_analyzer = segment_analyzer
        self.clip_extractor = clip_extractor
        self.min_clip_duration = min_clip_duration
        self.max_consecutive_missing_frames = max_consecutive_missing_frames

    def handle_match_found(
        self, state: ClipExtractionState, match_result: FrameMatchResult
    ) -> None:
        """Handle when a face match is found in a frame."""
        state.consecutive_missing = 0

        if state.current_segment_start is None:
            actual_start_frame = self.segment_analyzer.find_segment_start(state.frame_count, state)
            state.current_segment_start = actual_start_frame

            actual_start_time = actual_start_frame / state.frame_rate
            if actual_start_frame != state.frame_count:
                logger.info(
                    f"Started new clip segment at {actual_start_time:.2f}s "
                    f"(found earlier start, original match at {match_result.time_sec:.2f}s, "
                    f"similarity: {match_result.similarity:.3f})"
                )
            else:
                logger.info(
                    f"Started new clip segment at {actual_start_time:.2f}s "
                    f"(similarity: {match_result.similarity:.3f})"
                )

    def handle_no_match(
        self,
        state: ClipExtractionState,
        frame_rate: float,
        interval: int,
        video_path: str,
        output_path: Path,
        video_title: str,
    ) -> None:
        """Handle when no face match is found in a frame."""
        state.consecutive_missing += 1

        if (
            state.current_segment_start is not None
            and state.consecutive_missing >= self.max_consecutive_missing_frames
        ):
            self._end_current_segment(
                state, frame_rate, interval, video_path, output_path, video_title
            )

    def finalize_segment(
        self,
        state: ClipExtractionState,
        frame_rate: float,
        video_path: str,
        output_path: Path,
        video_title: str,
    ) -> None:
        """Finalize and extract the last segment if it exists."""
        if state.current_segment_start is None:
            logger.warning("No current segment to finalize, skipping.")
            return

        actual_end_frame = self.segment_analyzer.find_segment_end(state.frame_count, state)

        start_time = state.current_segment_start / frame_rate
        end_time = actual_end_frame / frame_rate
        duration = end_time - start_time

        if duration >= self.min_clip_duration:
            segment = ClipSegment(
                start_frame=state.current_segment_start,
                end_frame=actual_end_frame,
                start_time=start_time,
                end_time=end_time,
            )

            if self.clip_extractor.extract_clip(
                video_path, segment, output_path, video_title, state.clip_counter
            ):
                state.extracted_clips.append(segment)
                if actual_end_frame != state.frame_count:
                    logger.info(
                        f"Final clip segment extracted: {start_time:.2f}s - {end_time:.2f}s "
                        f"(found actual end, duration: {duration:.2f}s)"
                    )
                else:
                    logger.info(
                        f"Final clip segment extracted: {start_time:.2f}s - {end_time:.2f}s "
                        f"(duration: {duration:.2f}s)"
                    )
            else:
                logger.warning(
                    f"Failed to extract final clip segment: {start_time:.2f}s - {end_time:.2f}s"
                )

    def _end_current_segment(
        self,
        state: ClipExtractionState,
        frame_rate: float,
        interval: int,
        video_path: str,
        output_path: Path,
        video_title: str,
    ) -> None:
        """End the current segment and extract the clip if it meets criteria."""
        base_segment_end = state.frame_count - (state.consecutive_missing * interval)
        actual_segment_end = self.segment_analyzer.find_segment_end(base_segment_end, state)

        if state.current_segment_start is None:
            logger.warning("No current segment to end, skipping.")
            return

        start_time = state.current_segment_start / frame_rate
        end_time = actual_segment_end / frame_rate
        duration = end_time - start_time

        if duration >= self.min_clip_duration:
            segment = ClipSegment(
                start_frame=state.current_segment_start,
                end_frame=actual_segment_end,
                start_time=start_time,
                end_time=end_time,
            )

            if self.clip_extractor.extract_clip(
                video_path, segment, output_path, video_title, state.clip_counter
            ):
                state.extracted_clips.append(segment)
                state.clip_counter += 1
                if actual_segment_end != base_segment_end:
                    logger.info(
                        f"Ended and extracted clip segment: {start_time:.2f}s - {end_time:.2f}s "
                        f"(found actual end, duration: {duration:.2f}s)"
                    )
                else:
                    logger.info(
                        f"Ended and extracted clip segment: {start_time:.2f}s - {end_time:.2f}s "
                        f"(duration: {duration:.2f}s)"
                    )
            else:
                logger.warning(
                    f"Failed to extract clip segment: {start_time:.2f}s - {end_time:.2f}s"
                )
        else:
            logger.info(
                f"Discarded short segment: {start_time:.2f}s - {end_time:.2f}s "
                f"(duration: {duration:.2f}s < {self.min_clip_duration:.2f}s)"
            )

        state.current_segment_start = None
        state.consecutive_missing = 0
