import os
import cv2
import torch
import numpy as np
import contextlib

from PIL import Image
from typing import TypedDict
from pathlib import Path
from insightface.app import FaceAnalysis

from fvc.tools.logger import logger
from fvc.tools.config import Config


class FaceFeatures(TypedDict):
    tensor: torch.Tensor | None
    embeddings: np.ndarray


def extract_face_frames_from_folder(
    videos_folder: str,
    target_face_embeddings_filename: str,
    frames_per_second: int,
    output_folder: str,
    start_time: int,
    use_gpu: bool,
    video_extensions: tuple[str, ...] = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm"),
) -> None:
    """
    Batch process all video files in a folder to extract frames containing specified faces.

    Args:
        videos_folder (str): Path to the folder containing video files
        target_face_embeddings_filename (str): Path to the face feature file
        frames_per_second (int): Number of frames to process per second, defaults to 1
        output_folder (str): Output folder path
        start_time (int): Start processing time in seconds, defaults to 0
        use_gpu (bool): Whether to use GPU acceleration, defaults to True
        video_extensions (tuple): Supported video file extensions

    Returns:
        None

    Raises:
        FileNotFoundError: If the video folder does not exist
        Exception: Other errors during processing
    """
    videos_path = Path(videos_folder)
    if not videos_path.exists():
        raise FileNotFoundError(f"Video folder not found: {videos_folder}")

    video_files: list[Path] = []
    for ext in video_extensions:
        for file_path in videos_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() == ext.lower():
                video_files.append(file_path)

    if not video_files:
        logger.warning(f"No video files found in directory: {videos_folder}")
        return

    logger.info(f"Found {len(video_files)} video file(s) to process")

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        face_analysis = initialize_face_analysis(use_gpu)
        face_features = load_face_features(target_face_embeddings_filename, use_gpu)

        if face_features["embeddings"].size == 0:
            logger.warning("No face embeddings found in reference file - exiting batch processing")
            return
    except Exception as e:
        logger.error(f"Failed to initialize face analysis system: {e}")
        raise

    successful_count = 0
    failed_count = 0

    for i, video_file in enumerate(video_files, 1):
        logger.info(f"Processing video [{i}/{len(video_files)}]: {video_file.name}")

        try:
            video_title = video_file.stem
            existing_frames = list(output_path.glob(f"{video_title}_*m*s_sim_*.jpg"))

            if existing_frames:
                logger.info(
                    f"Video '{video_file.name}' already processed ({len(existing_frames)} frames found) - skipping"
                )
                successful_count += 1
                continue

            _process_single_video_with_shared_analysis(
                video_path=str(video_file),
                face_analysis=face_analysis,
                face_features=face_features,
                frames_per_second=frames_per_second,
                output_folder=output_folder,
                start_time=start_time,
            )

            successful_count += 1
            logger.info(f"Successfully processed: {video_file.name}")

        except Exception as e:
            failed_count += 1
            logger.error(f"Failed to process '{video_file.name}': {e}", exc_info=True)
            continue

    logger.info("Batch processing completed!")
    logger.info(f"Successfully processed: {successful_count} videos")
    logger.info(f"Failed to process: {failed_count} videos")
    logger.info(f"Output directory: {output_folder}")


def _process_single_video_with_shared_analysis(
    video_path: str,
    face_analysis: FaceAnalysis,
    face_features: FaceFeatures,
    frames_per_second: int,
    output_folder: str,
    start_time: int,
) -> None:
    """
    Process a single video file using an already initialized face analysis instance.

    Args:
        video_path (str): Video file path
        face_analysis (FaceAnalysis): Already initialized face analysis instance
        face_features (FaceFeatures): Already loaded face features
        frames_per_second (int): Number of frames to process per second
        output_folder (str): Output folder path
        start_time (int): Start processing time in seconds
    """
    video_title = Path(video_path).stem
    logger.info(f"Starting video analysis: {video_title}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: '{video_path}'.")

    try:
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / frame_rate if frame_rate > 0 else 0

        logger.info(
            f"Video specs - FPS: {frame_rate:.2f} | Total frames: {total_frames:,} | Duration: {duration:.2f}s"
        )

        interval = max(1, int(frame_rate / frames_per_second))
        frame_count = 0
        processed_frames = 0

        if start_time != 0:
            frame_count = int(start_time * frame_rate)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                save_similar_face_frame(
                    frame=frame,
                    frame_count=frame_count,
                    frame_rate=frame_rate,
                    video_title=video_title,
                    output_folder=output_folder,
                    face_analysis=face_analysis,
                    face_features=face_features,
                )
                processed_frames += 1

            frame_count += 1

            if frame_count % 1000 == 0:
                progress_percent = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                logger.info(
                    f"Progress: {frame_count:,}/{total_frames:,} frames ({progress_percent:.1f}%)"
                )

        logger.info(
            f"Video analysis completed! Processed {processed_frames:,} frames from {frame_count:,} total frames"
        )

    finally:
        cap.release()


def extract_face_frames(
    video_path: str,
    target_face_embeddings_filename: str,
    frames_per_second: int,
    output_folder: str,
    start_time: int,
    use_gpu: bool,
) -> None:
    """
    Extracts frames from a video containing faces similar to the provided face embeddings.
    Args:
        video_path (str): Path to the input video file.
        target_face_embeddings_filename (str): Path to the file containing face embeddings for comparison.
        frames_per_second (int): Number of frames per second to process.
        output_folder (str): Directory where the extracted frames will be saved.
        start_time (int): Start time in seconds to begin processing the video.
        use_gpu (bool): Whether to use GPU for face analysis.
    Returns:
        None
    Raises:
        RuntimeError: If the video file cannot be opened.
        Exception: For any other errors encountered during processing.
    Notes:
        - The function initializes face analysis and loads face embeddings for comparison.
        - Frames are saved only if they contain faces similar to the provided embeddings.
        - The output frames are saved in the specified output folder, organized by video title.
    """
    try:
        face_analysis = initialize_face_analysis(use_gpu)
        face_features = load_face_features(target_face_embeddings_filename, use_gpu)

        if face_features["embeddings"].size == 0:
            logger.warning("No face embeddings found in reference file - exiting video processing")
            return None

        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        video_title = Path(video_path).stem
        logger.info(f"Starting video analysis: {video_title}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video file: '{video_path}'.")

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(frame_rate / frames_per_second))
        frame_count = 0

        if start_time != 0:
            frame_count = int(start_time * frame_rate)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                save_similar_face_frame(
                    frame=frame,
                    frame_count=frame_count,
                    frame_rate=frame_rate,
                    video_title=video_title,
                    output_folder=output_folder,
                    face_analysis=face_analysis,
                    face_features=face_features,
                )

            frame_count += 1
    except Exception:
        logger.error(f"Error processing video '{video_path}'", exc_info=True)
        raise

    logger.info(f"Processing completed! Output directory: {output_folder}")
    cap.release()


def initialize_face_analysis(use_gpu: bool) -> FaceAnalysis:
    providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]

    with (
        contextlib.redirect_stdout(open(os.devnull, "w")),
        contextlib.redirect_stderr(open(os.devnull, "w")),
    ):
        try:
            face_analysis = FaceAnalysis(name=Config.MODEL_ZOO, providers=providers)
            ctx_id = 0 if use_gpu else -1
            face_analysis.prepare(ctx_id, det_size=(640, 640))
        except Exception as e:
            logger.error(f"Failed to initialize face analysis: {e}")
            raise

    return face_analysis


def load_face_features(filename: str, use_gpu: bool) -> FaceFeatures:
    try:
        data = np.load(filename)
        face_embeddings: np.ndarray = data["face_embeddings"]
        logger.info(f"Successfully loaded {len(face_embeddings)} face embeddings from '{filename}'")

        face_features: FaceFeatures = {"tensor": None, "embeddings": face_embeddings}

        if use_gpu:
            face_features["tensor"] = torch.tensor(face_embeddings, device="cuda").float()

        return face_features
    except Exception:
        logger.error(f"Failed to load face embeddings from '{filename}'", exc_info=True)
        raise


def save_similar_face_frame(
    frame: cv2.typing.MatLike,
    frame_count: int,
    frame_rate: float,
    video_title: str,
    output_folder: str,
    face_analysis: FaceAnalysis,
    face_features: FaceFeatures,
) -> None:
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        time_sec = frame_count / frame_rate
        minutes, seconds = divmod(int(time_sec), 60)
        frame_face_embeddings = extract_frame_face_embeddings(face_analysis, frame_rgb)
        if len(frame_face_embeddings) == 0:
            return

        if face_features["tensor"] is not None:
            similarity = (
                calculate_max_similarity_torch(face_features["tensor"], frame_face_embeddings)
                or 0.0
            )
        else:
            similarity = (
                calculate_max_similarity(face_features["embeddings"], frame_face_embeddings) or 0.0
            )

        if similarity < Config.MIN_SIMILARITY:
            return

        frame_filename = os.path.join(
            output_folder, f"{video_title}_{minutes}m{seconds:02d}s_sim_{similarity:.3f}.jpg"
        )
        Image.fromarray(frame_rgb).save(frame_filename)

        logger.info(
            f"Frame {frame_count} saved - Time: {minutes}m{seconds:02d}s | Similarity: {similarity:.3f}"
        )
    except Exception:
        logger.warning(
            f"Error processing frame {frame_count} at {frame_count / frame_rate:.2f}s",
            exc_info=True,
        )


def extract_frame_face_embeddings(
    face_analysis: FaceAnalysis, image: cv2.typing.MatLike
) -> list[np.ndarray]:
    try:
        faces = face_analysis.get(image)
        return [face.embedding for face in faces] if faces else []
    except Exception:
        logger.warning("Error extracting face embeddings from frame", exc_info=True)
        return []


def calculate_max_similarity_torch(
    face_tensor: torch.Tensor, frame_face_embeddings: list[np.ndarray]
) -> float:
    stacked_frame_face_embeddings = np.vstack(frame_face_embeddings)

    current_tensor = torch.tensor(stacked_frame_face_embeddings, device="cuda").float()
    similarities = torch.cosine_similarity(
        face_tensor.unsqueeze(1), current_tensor.unsqueeze(0), dim=2
    )
    return float(similarities.max().cpu().item())


def calculate_max_similarity(
    face_embeddings: np.ndarray, frame_face_embeddings: list[np.ndarray]
) -> float:
    stacked_frame_face_embeddings = np.vstack(frame_face_embeddings)
    similarities = calculate_cosine_similarity_matrix(
        face_embeddings, stacked_frame_face_embeddings
    )
    return float(np.max(similarities))


def calculate_cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    norm_A = np.linalg.norm(A, axis=1, keepdims=True)
    norm_B = np.linalg.norm(B, axis=1, keepdims=True)
    return np.dot((A / norm_A), (B / norm_B).T)
