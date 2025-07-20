import cv2

from fvc.tools.config import Config
from fvc.tools.logger import logger

from .data_models import FrameMatchResult
from .interfaces import IFrameMatcher
from .face_frames import (
    initialize_face_analysis,
    load_face_features,
    extract_frame_face_embeddings,
    calculate_max_similarity_torch,
    calculate_max_similarity,
)


class FrameMatcher(IFrameMatcher):
    def __init__(self, embeddings_file: str, use_gpu: bool = True):
        self.face_analysis = initialize_face_analysis(use_gpu)
        self.face_features = load_face_features(embeddings_file, use_gpu)
        if self.face_features["embeddings"].size == 0:
            logger.warning("No face embeddings found in reference file")
        self.tensor = self.face_features.get("tensor")
        self.embeddings = self.face_features["embeddings"]

    def match(
        self, frame: cv2.typing.MatLike, frame_count: int, frame_rate: float
    ) -> FrameMatchResult:
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            time_sec = frame_count / frame_rate
            frame_emb = extract_frame_face_embeddings(self.face_analysis, frame_rgb)
            if len(frame_emb) == 0:
                return FrameMatchResult(frame_count, 0.0, False, time_sec)
            if self.tensor is not None:
                sim = calculate_max_similarity_torch(self.tensor, frame_emb) or 0.0
            else:
                sim = calculate_max_similarity(self.embeddings, frame_emb) or 0.0
            has_match = sim >= Config.MIN_SIMILARITY
            return FrameMatchResult(frame_count, sim, has_match, time_sec)
        except Exception:
            logger.warning(
                f"Error matching frame at {frame_count / frame_rate:.2f}s", exc_info=True
            )
            return FrameMatchResult(frame_count, 0.0, False, frame_count / frame_rate)

    def get_embeddings_size(self) -> int:
        """Get the size of the embeddings."""
        return self.embeddings.size
