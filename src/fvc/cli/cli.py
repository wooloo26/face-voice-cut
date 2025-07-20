import click

from fvc.core import extract_face_frames_from_folder, VideoClipExtractor, generate_face_embeddings
from fvc.tools.config import Config


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option(
    "--image-folder",
    "-i",
    default=Config.FACE_ORIGINAL_IMAGE_FOLDER,
    type=click.Path(exists=True),
    help=f"Path to the folder containing original images (default: '{Config.FACE_ORIGINAL_IMAGE_FOLDER}')",
)
@click.option(
    "--output-filename",
    "-o",
    default=Config.FACE_EMBEDDINGS_FILENAME,
    type=click.Path(),
    help=f"Path to the face embeddings file (default: '{Config.FACE_EMBEDDINGS_FILENAME}')",
)
@click.option(
    "--use-gpu/--no-use-gpu",
    default=Config.USE_GPU,
    help=f"Use GPU acceleration (default: {Config.USE_GPU})",
)
def generate(image_folder: str, output_filename: str, use_gpu: bool) -> None:
    try:
        generate_face_embeddings(
            image_folder=image_folder,
            output_filename=output_filename,
            use_gpu=use_gpu,
        )
    except Exception as e:
        raise click.ClickException(str(e))


@cli.command()
@click.argument("videos_folder", type=click.Path(exists=True))
@click.option(
    "--face-embeddings",
    default=Config.FACE_EMBEDDINGS_FILENAME,
    type=click.Path(exists=True),
    help=f"Path to the face embeddings file (default: '{Config.FACE_EMBEDDINGS_FILENAME}')",
)
@click.option(
    "--fps",
    default=Config.FRAME_PER_SECOND,
    type=int,
    help="Number of frames per second to process (default: 1)",
)
@click.option(
    "--output-folder",
    "-o",
    default=Config.FACE_FRAMES_OUTPUT_FOLDER,
    type=click.Path(),
    help=f"Output directory for extracted frames (default: '{Config.FACE_FRAMES_OUTPUT_FOLDER}')",
)
@click.option(
    "--start-time",
    default=Config.START_TIME,
    type=int,
    help="Start time in seconds (default: 0)",
)
@click.option(
    "--use-gpu/--no-use-gpu",
    default=Config.USE_GPU,
    help=f"Use GPU acceleration (default: {Config.USE_GPU})",
)
def extract(
    videos_folder: str,
    face_embeddings: str,
    fps: int,
    output_folder: str,
    start_time: int,
    use_gpu: bool,
) -> None:
    """Batch process all videos in a folder to extract face frames."""
    try:
        extract_face_frames_from_folder(
            videos_folder=videos_folder,
            target_face_embeddings_filename=face_embeddings,
            frames_per_second=fps,
            output_folder=output_folder,
            start_time=start_time,
            use_gpu=use_gpu,
        )
    except Exception as e:
        raise click.ClickException(str(e))


@cli.command()
@click.argument("videos_folder", type=click.Path(exists=True))
@click.option(
    "--face-embeddings",
    default=Config.FACE_EMBEDDINGS_FILENAME,
    type=click.Path(exists=True),
    help=f"Path to the face embeddings file (default: '{Config.FACE_EMBEDDINGS_FILENAME}')",
)
@click.option(
    "--fps",
    default=Config.CLIP_FRAME_PER_SECOND,
    type=int,
    help="Number of frames per second to process (default: 1)",
)
@click.option(
    "--output-folder",
    "-o",
    default=Config.VIDEO_CLIPS_OUTPUT_FOLDER,
    type=click.Path(),
    help=f"Output directory for extracted video clips (default: '{Config.VIDEO_CLIPS_OUTPUT_FOLDER}')",
)
@click.option(
    "--start-time",
    default=Config.START_TIME,
    type=int,
    help="Start time in seconds (default: 0)",
)
@click.option(
    "--use-gpu/--no-use-gpu",
    default=Config.USE_GPU,
    help=f"Use GPU acceleration (default: {Config.USE_GPU})",
)
@click.option(
    "--max-missing-frames",
    default=Config.MAX_CONSECUTIVE_MISSING_FRAMES,
    type=int,
    help=f"Maximum consecutive missing frames before ending a clip (default: {Config.MAX_CONSECUTIVE_MISSING_FRAMES})",
)
@click.option(
    "--min-clip-duration",
    default=Config.MIN_CLIP_DURATION,
    type=float,
    help=f"Minimum clip duration in seconds (default: {Config.MIN_CLIP_DURATION})",
)
def clip(
    videos_folder: str,
    face_embeddings: str,
    fps: int,
    output_folder: str,
    start_time: int,
    use_gpu: bool,
    max_missing_frames: int,
    min_clip_duration: float,
) -> None:
    """Extract video clips containing target faces with frame matching tolerance."""
    try:
        extractor = VideoClipExtractor(
            target_face_embeddings_filename=face_embeddings,
            frames_per_second=fps,
            use_gpu=use_gpu,
            max_consecutive_missing_frames=max_missing_frames,
            min_clip_duration=min_clip_duration,
            start_time=start_time,
            video_extensions=(".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm"),
        )
        extractor.extract_clips_from_folder(videos_folder, output_folder)
    except Exception as e:
        raise click.ClickException(str(e))
