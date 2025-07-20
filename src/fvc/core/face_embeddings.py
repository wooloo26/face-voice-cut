import os
import cv2
import numpy as np

from PIL import Image
from tqdm import tqdm
from insightface.app import FaceAnalysis

from fvc.tools.logger import logger
from fvc.tools.config import Config

MAX_IMAGE_DIMENSION = 2000


def generate_face_embeddings(image_folder: str, output_filename: str, use_gpu: bool) -> None:
    """
    Generates face embeddings for all valid image files in a specified folder and saves them to an output file.
    Args:
        image_folder (str): The path to the folder containing the images to process.
        output_filename (str): The path to the file where the generated face embeddings will be saved.
        use_gpu (bool): Whether to use GPU for face analysis
    Returns:
        None
    Raises:
        FileNotFoundError: If the specified image folder does not exist.
        ValueError: If no valid image files are found in the folder.
    Notes:
        - Valid image file extensions are '.jpg', '.jpeg', and '.png'.
        - The function uses a face analysis tool to extract embeddings for each image.
        - Progress is displayed using a progress bar.
    """
    face_analysis = initialize_face_analysis(use_gpu)

    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]

    logger.info(
        f"Found {len(image_files)} image(s) in folder '{image_folder}' - starting face embedding extraction"
    )

    face_embeddings: list[np.ndarray] = []
    for filename in tqdm(image_files, desc="Processing images", unit="img"):
        image_path = os.path.join(image_folder, filename)
        embedding = extract_face_embedding(image_path, face_analysis)
        if embedding is not None:
            face_embeddings.append(embedding)

    save_face_embeddings(face_embeddings, output_filename)


def initialize_face_analysis(use_gpu: bool) -> FaceAnalysis:
    face_analysis = FaceAnalysis(name=Config.MODEL_ZOO)
    ctx_id = 0 if use_gpu else -1
    face_analysis.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=0.5)
    return face_analysis


def extract_face_embedding(image_path: str, face_analysis: FaceAnalysis) -> np.ndarray | None:
    image = load_and_preprocess_image(image_path)
    if image is None:
        return None
    return get_largest_face_embedding(image, face_analysis)


def load_and_preprocess_image(image_path: str) -> cv2.typing.MatLike | None:
    try:
        image_file = Image.open(image_path)
        mat: cv2.typing.MatLike
        if image_file.mode != "RGB":
            mat = np.array(image_file.convert("RGB"))
        else:
            mat = np.array(image_file)

        return cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    except Exception:
        logger.warning(f"Error loading or preprocessing image: {image_path}", exc_info=True)
        return None


def get_largest_face_embedding(
    image: cv2.typing.MatLike, face_analysis: FaceAnalysis
) -> np.ndarray | None:
    try:
        resized_image = resize_if_image_oversized(image)
        faces = face_analysis.get(resized_image)

        if not faces:
            return None

        if len(faces) > 1:
            face_areas = [face.bbox[2] * face.bbox[3] for face in faces]
            return faces[np.argmax(face_areas)].embedding

        return faces[0].embedding
    except Exception:
        logger.warning("Error during face analysis - skipping image", exc_info=True)
        return None


def resize_if_image_oversized(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    h, w = image.shape[:2]
    if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
        scale = min(MAX_IMAGE_DIMENSION / h, MAX_IMAGE_DIMENSION / w)
        return cv2.resize(image, (0, 0), fx=scale, fy=scale)
    return image


def save_face_embeddings(face_embeddings: list[np.ndarray], filename: str) -> None:
    if not face_embeddings:
        logger.warning("No valid face embeddings were generated from the images")
        return

    try:
        np.savez(filename, face_embeddings=np.array(face_embeddings))
        logger.info(f"Successfully generated face embeddings for {len(face_embeddings)} image(s)")
        logger.info(f"Face embeddings saved to: {filename}")
    except Exception:
        logger.error("Failed to save face embeddings file", exc_info=True)
        raise
