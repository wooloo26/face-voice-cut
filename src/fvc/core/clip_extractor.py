import contextlib
import os
from pathlib import Path
from moviepy import VideoFileClip

from fvc.tools.logger import logger
from .interfaces import IClipExtractor
from .data_models import ClipSegment


class MoviePyClipExtractor(IClipExtractor):
    """Extracts video clips using MoviePy library."""

    def extract_clip(
        self,
        video_path: str,
        segment: ClipSegment,
        output_path: Path,
        video_title: str,
        clip_number: int,
    ) -> bool:
        """Extract a single video clip using MoviePy."""
        output_filename = (
            f"{video_title}_clip_{clip_number:03d}_"
            f"{segment.start_time:.1f}s-{segment.end_time:.1f}s.mp4"
        )
        output_filepath = output_path / output_filename

        with (
            contextlib.redirect_stdout(open(os.devnull, "w")),
            contextlib.redirect_stderr(open(os.devnull, "w")),
        ):
            try:
                logger.info(
                    f"Extracting clip {clip_number}: {segment.start_time:.1f}s - {segment.end_time:.1f}s"
                )

                clip = VideoFileClip(video_path).subclipped(segment.start_time, segment.end_time)
                clip.write_videofile(str(output_filepath))
                clip.close()

                logger.info(f"Successfully extracted: {output_filename}")
                return True
            except Exception as e:
                logger.error(f"Error extracting clip {clip_number}: {e}")
                return False
