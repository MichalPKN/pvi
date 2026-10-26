import difflib
import functools
import json

import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity

from tests import PVITestCase

DATA_DIR = "data/10_cze_id"
IMAGE_NAME = "KA10_01"
OCR_IMAGES = ("TS10_01", "HA10_06")
TEST_IMAGES = ("CA10_01", "HA10_06", "HS10_12", "KA10_01", "KS10_02", "PA10_30", "TA10_01", "TS10_01")
SURNAME, NAME = "SPECIMEN", "VZOR"
SYNTHETIC_CARDS = (("HOLUB", "PETR"), ("FIALA", "JANA"), ("SOUKUP", "MARTIN"), ("MAREK", "EVA"))
SYNTHETIC_SCENES = ((0, "HA10_06"), (2, "PA10_30"))
MAX_PHOTO_AREA = 2.0


def load_rgb(path: str) -> np.ndarray:
    return cv.cvtColor(cv.imread(PVITestCase.rpath(path)), cv.COLOR_BGR2RGB)


@functools.cache
def load_reference(name: str) -> np.ndarray:
    return load_rgb(f"tests/data/id_recognition-{name}.png")


def load_quad(name: str) -> np.ndarray:
    with open(PVITestCase.rpath(f"{DATA_DIR}/{name}.json")) as f:
        return np.array(json.load(f)["quad"], dtype=np.float64)


def to_gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image * 255 if image.max() <= 1 else image, 0, 255).astype(np.uint8)
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY) if image.ndim == 3 else image


def card_similarity(card: np.ndarray, reference: np.ndarray, scale: float = 0.25) -> float:
    """SSIM of the grayscale cards downscaled to tolerate the misalignment of a few pixels."""
    size = (round(reference.shape[1] * scale), round(reference.shape[0] * scale))
    a, b = (cv.resize(to_gray(c), size, interpolation=cv.INTER_AREA) for c in (card, reference))
    return float(structural_similarity(a, b, data_range=255))


def field_score(roi: np.ndarray, field: np.ndarray) -> float:
    """Normalized correlation of the field and the best matching part of the ROI, -1 if the ROI is too small."""
    roi, field = to_gray(roi), to_gray(field)
    if roi.shape[0] < field.shape[0] or roi.shape[1] < field.shape[1]:
        return -1.0
    return float(cv.matchTemplate(roi, field, cv.TM_CCOEFF_NORMED).max())


def normalize_text(text: str) -> str:
    return " ".join(str(text).upper().split())


def text_errors(text: str, expected: str) -> int:
    """Number of differing characters: the length of the longer text minus the number of matching characters."""
    a, b = normalize_text(text), normalize_text(expected)
    matching = sum(block.size for block in difflib.SequenceMatcher(None, a, b).get_matching_blocks())
    return max(len(a), len(b)) - matching


def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.float32(a), np.float32(b)
    intersection, _ = cv.intersectConvexConvex(a, b)
    return float(intersection / (cv.contourArea(a) + cv.contourArea(b) - intersection))


class TestDetection(PVITestCase):
    min_iou = 0.9

    def test_shape(self):
        corners = np.asarray(self.params["corners"], dtype=np.float64)
        self.assertEqual(corners.shape, (4, 2), msg="Corners must be an array (4, 2) of x, y")

    def test_iou(self):
        corners = np.asarray(self.params["corners"], dtype=np.float64).reshape(4, 2)
        iou = quad_iou(corners, load_quad(IMAGE_NAME))
        print(f"IoU: {iou:.3f}")
        self.assertGreaterEqual(
            iou,
            self.min_iou,
            msg=f"IoU {iou:.3f} of the detected and the annotated card is less than {self.min_iou}. Are the corners "
            "in the pixels of the scene image?",
        )

    def test_corner_order(self):
        # Each detected corner must be the closest one to the annotated corner of the same index
        corners = np.asarray(self.params["corners"], dtype=np.float64).reshape(4, 2)
        quad = load_quad(IMAGE_NAME)
        distances = np.linalg.norm(corners[:, None] - quad[None], axis=2)
        self.assertTrue(
            (distances.argmin(axis=1) == np.arange(4)).all(),
            msg=f"Corners {np.round(corners).tolist()} are not in the order top-left, top-right, bottom-right, "
            f"bottom-left of the card annotated as {quad.tolist()}",
        )


class TestRectification(PVITestCase):
    min_ssim = 0.6

    def test_shape(self):
        card, reference = np.asarray(self.params["card"]), load_reference(f"card_{IMAGE_NAME}")
        self.assertEqual(
            card.shape, reference.shape, msg=f"Rectified card must be an RGB image of shape {reference.shape}"
        )

    def test_similarity(self):
        card, reference = np.asarray(self.params["card"]), load_reference(f"card_{IMAGE_NAME}")
        self.assertEqual(
            card.shape[:2], reference.shape[:2], msg=f"Rectified card must have the shape {reference.shape}"
        )
        ssim = card_similarity(card, reference)
        print(f"SSIM: {ssim:.3f}")
        self.assertGreaterEqual(
            ssim,
            self.min_ssim,
            msg=f"Rectified card differs from the reference (SSIM {ssim:.3f} < {self.min_ssim}). Is the card upright and "
            "are the corners mapped to the corners of the rectified card in the right order?",
        )


class ROITestCase(PVITestCase):
    # Minimal correlation of the field and the ROI and maximal correlation of the other text field and the ROI
    min_score = 0.8
    max_other_score = 0.85

    def check_photo(self, photo: np.ndarray, field: np.ndarray, label: str = "") -> None:
        photo = np.asarray(photo)
        area = photo.shape[0] * photo.shape[1] / (field.shape[0] * field.shape[1])
        self.assertLessEqual(
            area,
            MAX_PHOTO_AREA,
            msg=f"{label}Photo of shape {photo.shape[:2]} is too large, crop only the photo of the face",
        )
        score = field_score(photo, field)
        print(f"{label}photo: score {score:.3f}")
        self.assertGreaterEqual(
            score,
            self.min_score,
            msg=f"{label}Photo of the face not found in the ROI (score {score:.3f} < {self.min_score}). Is the ROI "
            "cropped from the right position of the rectified card and not resized?",
        )

    def check_text(self, roi: np.ndarray, field: np.ndarray, other: np.ndarray, key: str, label: str = "") -> None:
        score, other_score = field_score(roi, field), field_score(roi, other)
        print(f"{label}{key}: score {score:.3f}, score of the other text {other_score:.3f}")
        self.assertGreaterEqual(
            score,
            self.min_score,
            msg=f"{label}Text of the {key} not found in the ROI (score {score:.3f} < {self.min_score}). Does the ROI "
            "contain the whole text, is it cropped from the right position of the rectified card and not resized?",
        )
        self.assertLess(
            other_score,
            self.max_other_score,
            msg=f"{label}ROI of the {key} contains the other text too (score {other_score:.3f} >= "
            f"{self.max_other_score}). Are the surname and the given name swapped?",
        )


class TestROIs(ROITestCase):
    def test_keys(self):
        self.assertEqual(
            set(self.params["rois"]), {"photo", "surname", "name"}, msg="ROIs must have the keys photo, surname, name"
        )

    def test_photo(self):
        self.check_photo(self.params["rois"]["photo"], load_reference(f"field_{IMAGE_NAME}_photo"))

    def test_surname(self):
        surname, name = load_reference(f"field_{IMAGE_NAME}_surname"), load_reference(f"field_{IMAGE_NAME}_name")
        self.check_text(self.params["rois"]["surname"], surname, name, "surname")

    def test_name(self):
        surname, name = load_reference(f"field_{IMAGE_NAME}_surname"), load_reference(f"field_{IMAGE_NAME}_name")
        self.check_text(self.params["rois"]["name"], name, surname, "given name")


def ocr_cases() -> list[tuple[np.ndarray, str]]:
    """Reference ROIs of the specimen card and of the synthetic cards with the expected texts."""
    cases = []
    for name in (IMAGE_NAME, *OCR_IMAGES):
        cases += [(load_reference(f"roi_{name}_surname"), SURNAME), (load_reference(f"roi_{name}_name"), NAME)]
    for index, (surname, name) in enumerate(SYNTHETIC_CARDS):
        cases += [
            (load_reference(f"roi_synthetic_{index}_surname"), surname),
            (load_reference(f"roi_synthetic_{index}_name"), name),
        ]
    return cases


class TestOCR(PVITestCase):
    max_errors = 1

    def test_ocr(self):
        for roi, expected in ocr_cases():
            text = self.params["ocr_fn"](roi.copy())
            self.assertIsInstance(text, str, msg="OCR must return a string")
            errors = text_errors(text, expected)
            print(f"{expected!r} -> {text!r}: {errors} errors")
            with self.subTest(text=expected):
                self.assertLessEqual(
                    errors,
                    self.max_errors,
                    msg=f"Text {text!r} differs from {expected!r} in {errors} characters (max. {self.max_errors})",
                )


class TestOCRExact(TestOCR):
    max_errors = 0


@functools.cache
def run_ocr_cze_id(ocr_cze_id_fn) -> list[tuple[str, np.ndarray, str, str, tuple]]:
    """Apply the function to the test images and the synthetic scenes, return (label, photo field, name, surname,
    result)."""
    cases = [
        (name, load_rgb(f"{DATA_DIR}/{name}.jpg"), load_reference(f"field_{name}_photo"), NAME, SURNAME)
        for name in TEST_IMAGES
    ]
    for index, _ in SYNTHETIC_SCENES:
        surname, name = SYNTHETIC_CARDS[index]
        cases.append(
            (
                f"synthetic {surname} {name}",
                load_rgb(f"tests/data/id_recognition-scene_synthetic_{index}.jpg"),
                load_reference(f"field_synthetic_{index}_photo"),
                name,
                surname,
            )
        )
    return [(label, field, name, surname, ocr_cze_id_fn(image)) for label, image, field, name, surname in cases]


class OcrCzeIdTestCase(ROITestCase):
    def results(self):
        results = run_ocr_cze_id(self.params["ocr_cze_id_fn"])
        for label, _, _, _, result in results:
            self.assertEqual(len(result), 3, msg=f"{label}: ocr_cze_id must return a triple (photo, name, surname)")
        return results


class TestOcrCzeIdPhotos(OcrCzeIdTestCase):
    def test_photos(self):
        for label, field, _, _, (photo, _, _) in self.results():
            with self.subTest(image=label):
                self.check_photo(photo, field, label=f"{label}: ")


class TestOcrCzeIdTexts(OcrCzeIdTestCase):
    # Maximal number of differing characters of a text and minimal fraction of the texts with at most so many errors,
    # separately for the real images and for the synthetic scenes (otherwise constant SPECIMEN and VZOR would pass)
    max_errors = 1
    min_fraction = 0.8

    def test_texts(self):
        correct = {"real images": [], "synthetic scenes": []}
        for label, _, name, surname, (_, name_read, surname_read) in self.results():
            group = correct["synthetic scenes" if label.startswith("synthetic") else "real images"]
            for text, expected in ((surname_read, surname), (name_read, name)):
                self.assertIsInstance(text, str, msg=f"{label}: the name and the surname must be strings")
                errors = text_errors(text, expected)
                group.append(errors <= self.max_errors)
                print(f"{label}: {expected!r} -> {text!r}: {errors} errors")
        for group, values in correct.items():
            print(f"{group}: texts with at most {self.max_errors} errors: {sum(values)}/{len(values)}")
        for group, values in correct.items():
            self.assertGreaterEqual(
                float(np.mean(values)),
                self.min_fraction,
                msg=f"{group}: only {sum(values)} of {len(values)} texts have at most {self.max_errors} errors "
                f"(required {self.min_fraction:.0%}). Are the name and the surname in the right order?",
            )


class TestOcrCzeIdExact(TestOcrCzeIdTexts):
    max_errors = 0
    min_fraction = 1.0
