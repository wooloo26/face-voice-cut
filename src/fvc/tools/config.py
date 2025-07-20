import torch

from typing import ClassVar
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    USE_GPU: ClassVar[bool] = torch.cuda.is_available()
    MODEL_ZOO: ClassVar[str] = "buffalo_l"

    # Generate face embeddings
    FACE_ORIGINAL_IMAGE_FOLDER: ClassVar[str] = "face_original_images"
    FACE_EMBEDDINGS_FILENAME: ClassVar[str] = "face_embeddings.npz"

    # Extract face frames from video
    FRAME_PER_SECOND: ClassVar[int] = 2
    FACE_FRAMES_OUTPUT_FOLDER: ClassVar[str] = "output_face_frames"
    START_TIME: ClassVar[int] = 0
    MIN_SIMILARITY: ClassVar[float] = 0.58

    # Video clipping with frame tolerance
    CLIP_FRAME_PER_SECOND: ClassVar[int] = 2
    MAX_CONSECUTIVE_MISSING_FRAMES: ClassVar[int] = 20
    MIN_CLIP_DURATION: ClassVar[float] = 1.0
    VIDEO_CLIPS_OUTPUT_FOLDER: ClassVar[str] = "output_video_clips"
