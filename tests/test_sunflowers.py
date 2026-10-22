import os
import tempfile

import cv2 as cv
import numpy as np

from tests import PVITestCase

NUM_IMAGES = 4

# Image shapes (height, width) of data/sunflowers{1..4}.jpg
IMAGE_SHAPES = {1: (424, 640), 2: (638, 640), 3: (336, 640), 4: (775, 640)}

# Number of annotated sunflowers in data/sunflowers{1..4}.txt. The annotation is not exhaustive: partially visible and
# distant flowers are often missing, hence the counts are only approximate.
GT_COUNTS = {1: 7, 2: 30, 3: 24, 4: 8}


class TestLoadData(PVITestCase):
    def test_images(self):
        images = self.params["images"]
        self.assertEqual(len(images), NUM_IMAGES, msg=f"Expected {NUM_IMAGES} images")
        for index, image in enumerate(images, start=1):
            self.assertIsInstance(image, np.ndarray, msg=f"Image {index} must be a numpy array")
            self.assertEqual(image.shape, (*IMAGE_SHAPES[index], 3), msg=f"Image {index} has unexpected shape")
            self.assertEqual(image.dtype, np.uint8, msg=f"Image {index} must be of type uint8")

    def test_channel_order(self):
        # The sky in the first image is blue, so swapped channels show up as red > blue
        sky = self.params["images"][0][:100, 300:600].reshape(-1, 3).mean(axis=0)
        self.assertGreater(sky[2], sky[0], msg="The sky must have more blue than red. Is the image still in BGR order?")

    def test_load_boxes(self):
        # Synthetic labels of an image 200 px wide and 100 px high, the file with a single line is a corner case
        shape = (100, 200, 3)
        for name, labels, expected in (
            ("two boxes", "0 0.5 0.5 0.2 0.4\n0 0.1 0.25 0.2 0.1\n", [(80, 30, 120, 70), (0, 20, 40, 30)]),
            ("single box", "0 0.75 0.6 0.1 0.6\n", [(140, 30, 160, 90)]),
        ):
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmp_dir:
                path = os.path.join(tmp_dir, "labels.txt")
                with open(path, "w") as f:
                    f.write(labels)
                self.assertArraysClose(
                    np.asarray(self.params["load_boxes_fn"](path, shape), dtype=np.float64),
                    np.array(expected, dtype=np.float64),
                    rtol=0,
                    atol=1e-6,
                    msg=f"Case '{name}': boxes converted from the YOLO labels {labels!r} of an image of shape {shape} "
                    "are incorrect (expected an array (N, 4) of pixels x1, y1, x2, y2)",
                )

    def test_boxes_true(self):
        boxes_true = self.params["boxes_true"]
        self.assertEqual(len(boxes_true), NUM_IMAGES, msg=f"Expected annotations of {NUM_IMAGES} images")
        for index, boxes in enumerate(boxes_true, start=1):
            self.assertEqual(
                np.shape(boxes),
                (GT_COUNTS[index], 4),
                msg=f"Image {index}: expected {GT_COUNTS[index]} boxes (x1, y1, x2, y2), got shape {np.shape(boxes)}",
            )


class TestHueHistogram(PVITestCase):
    # Synthetic image of 20 pixels: 1 gray, 2 orange, 3 yellow, 4 green, 5 blue and 5 violet with the known hues
    COLORS = ((128, 128, 128), (255, 130, 0), (255, 205, 0), (0, 255, 80), (0, 125, 255), (105, 0, 255))
    HUES = (0.0, 0.085, 0.134, 0.3856, 0.585, 0.7353)
    COUNTS = (1, 2, 3, 4, 5, 5)
    min_bins = 10
    # Hue computed by OpenCV is rounded to 2 degrees, the bin edges closer than this to a hue are not checked
    hue_tolerance = 0.01

    def test_histogram(self):
        image = np.repeat(np.array(self.COLORS, dtype=np.uint8), self.COUNTS, axis=0).reshape(4, 5, 3)
        hist = np.asarray(self.params["hist_fn"](image), dtype=np.float64).ravel()
        self.assertGreaterEqual(hist.size, self.min_bins, msg=f"Histogram must have at least {self.min_bins} bins")
        self.assertAlmostEqual(float(hist.sum()), 1.0, places=6, msg="Histogram must be normalized to the unit sum")
        # The fraction of the pixels below each inner bin edge must agree with the cumulative sum of the histogram
        edges, hues = np.linspace(0, 1, hist.size + 1)[1:-1, None], np.array(self.HUES)
        expected = ((hues < edges) * np.array(self.COUNTS)).sum(axis=1) / sum(self.COUNTS)
        checked = np.abs(edges - hues).min(axis=1) > self.hue_tolerance
        self.assertArraysClose(
            np.cumsum(hist)[:-1][checked],
            expected[checked],
            rtol=0,
            atol=1e-6,
            msg=f"Histogram of an image with {self.COUNTS} pixels of hues {self.HUES} is incorrect. "
            "Do the bins cover the hue range 0..1 (0..179 in OpenCV) uniformly?",
        )


class TestDetectSunflowers(PVITestCase):
    # Synthetic image of grass and sky with dark discs (y, x, radius), which are blobs, but not sunflowers, and with
    # the template resized to squares (y, x, size). The center of the flower has the radius of about size / 4.
    DISCS = ((60, 60, 12), (200, 110, 40), (80, 290, 25), (200, 330, 55))
    FLOWERS = ((40, 150, 50), (230, 230, 100), (90, 380, 70))

    def test_detect_sunflowers(self):
        image = np.full((300, 420, 3), (40, 140, 40), dtype=np.uint8)
        image[:, 210:] = (90, 150, 230)
        for y, x, radius in self.DISCS:
            cv.circle(image, (x, y), radius, (70, 40, 20), thickness=-1)
        template = cv.cvtColor(cv.imread(self.rpath("data/sunflower_template.jpg")), cv.COLOR_BGR2RGB)
        for y, x, size in self.FLOWERS:
            image[y - size // 2 : y + size // 2, x - size // 2 : x + size // 2] = cv.resize(template, (size, size))
        detections = np.asarray(self.params["detect_fn"](image), dtype=np.float64).reshape(-1, 3)
        print(f"Detections (x, y, sigma):\n{np.round(detections, 1)}")
        for y, x, radius in self.DISCS:
            self.assertFalse(
                (np.hypot(detections[:, 0] - x, detections[:, 1] - y) <= radius).any(),
                msg=f"Dark disc of radius {radius} at (x, y) = ({x}, {y}) is a blob, but not a sunflower. "
                "Are the blobs filtered by the color of their surroundings?",
            )
        for y, x, size in self.FLOWERS:
            found = (np.hypot(detections[:, 0] - x, detections[:, 1] - y) <= size / 4) & (
                (detections[:, 2] >= size / 10) & (detections[:, 2] <= size / 2)
            )
            self.assertTrue(
                found.any(),
                msg=f"No detection (x, y, sigma) close to ({x}, {y}, {size / 5:.0f}) of the sunflower {size} px wide "
                f"centered at (x, y) = ({x}, {y}). Is the range of sigmas sufficient and are the coordinates in the "
                "order (x, y, sigma)?",
            )


class EvaluatingTestCase(PVITestCase):
    def evaluate(self, blobs, boxes_true) -> tuple[int, int, int]:
        result = tuple(self.params["evaluate_fn"](blobs, boxes_true))
        self.assertEqual(len(result), 3, msg="The evaluation function must return a triple (TP, FP, FN)")
        return tuple(int(v) for v in result)


# Synthetic data for the tests of the evaluation functions: reference boxes (x1, y1, x2, y2) and
# cases of detected blobs (x, y, sigma) with the expected (TP, FP, FN). For the evaluation by IoU, the blobs are
# represented by squares with the side 4 * sigma.
EVAL_boxes_true = [(100, 100, 140, 140), (300, 200, 340, 240), (500, 50, 560, 110)]

# Cases with the same result for the evaluation by centers and for the evaluation by IoU with unique assignment
EVAL_COMMON_CASES = {
    "perfect": ([(120, 120, 10), (320, 220, 10), (530, 80, 15)], (3, 0, 0)),
    "slightly shifted": ([(125, 115, 10), (315, 225, 9), (535, 85, 14)], (3, 0, 0)),
    "missed one": ([(120, 120, 10), (530, 80, 15)], (2, 0, 1)),
    "one false": ([(120, 120, 10), (320, 220, 10), (530, 80, 15), (400, 400, 10)], (3, 1, 0)),
    "all false": ([(10, 10, 10), (700, 700, 10)], (0, 2, 3)),
    "empty": ([], (0, 0, 3)),
}


class EvaluationTestCase(EvaluatingTestCase):
    def check_cases(self, cases: dict) -> None:
        for name, (blobs, expected) in cases.items():
            with self.subTest(case=name):
                result = self.evaluate(blobs, EVAL_boxes_true)
                self.assertEqual(
                    result, expected, msg=f"Case '{name}': got (TP, FP, FN) = {result}, expected {expected}"
                )

    def test_common_cases(self):
        self.check_cases(EVAL_COMMON_CASES)


class TestEvaluateBlobs(EvaluationTestCase):
    def test_center_rules(self):
        self.check_cases(
            {
                # a blob is a hit whenever its center is inside a box, whatever its sigma is
                "small blob inside": ([(120, 120, 2)], (1, 0, 2)),
                "poor detection first": ([(110, 110, 10), (120, 120, 10)], (2, 0, 2)),
                # several blobs inside the same box are all true positives and do not matter for the false negatives
                "duplicate hits": ([(118, 118, 10), (122, 122, 10), (320, 220, 10)], (3, 0, 1)),
            }
        )


class TestBoxIoU(PVITestCase):
    def test_box_iou(self):
        box_iou_fn = self.params["box_iou_fn"]
        for name, a, b, expected in (
            ("identical boxes", (0, 0, 10, 10), (0, 0, 10, 10), 1.0),
            ("disjoint boxes", (0, 0, 10, 10), (20, 20, 30, 30), 0.0),
            ("boxes touching by an edge", (0, 0, 10, 10), (10, 0, 20, 10), 0.0),
            ("half overlap", (0, 0, 10, 10), (5, 0, 15, 10), 1 / 3),
            ("nested box", (0, 0, 10, 10), (2, 2, 8, 8), 0.36),
            ("non-square boxes", (0, 0, 20, 10), (10, 5, 30, 15), 50 / 350),
        ):
            with self.subTest(case=name):
                for first, second in ((a, b), (b, a)):
                    self.assertAlmostEqual(
                        float(box_iou_fn(first, second)),
                        expected,
                        places=6,
                        msg=f"IoU of {first} and {second} ({name}) should be {expected:.4f}",
                    )


class TestEvaluateBlobsIoU(EvaluationTestCase):
    def test_iou_rules(self):
        self.check_cases(
            {
                # the square with the side 4 * sigma = 8 is much smaller than the reference box, IoU is below 0.5
                "small blob inside": ([(120, 120, 2)], (0, 1, 3)),
                # each reference box can be matched only once, the second detection of the box is a false positive
                "duplicate hits": ([(118, 118, 10), (122, 122, 10), (320, 220, 10)], (2, 1, 1)),
                # a detection with IoU below the threshold is a false positive and does not occupy the box
                "poor detection first": ([(110, 110, 10), (120, 120, 10)], (1, 1, 2)),
            }
        )


class TestDetect(EvaluatingTestCase):
    index = 0
    # Minimal simplified F1 score 2 * TP / (2 * TP + FP + FN) of the detected blobs
    min_f1 = 0.0

    def test_positions(self):
        blobs = self.params["detections"][self.index - 1]
        boxes_true = self.params["boxes_true"][self.index - 1]
        tp, fp, fn = self.evaluate(blobs, boxes_true)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        print(f"Detected {len(blobs)} sunflowers, annotated {len(boxes_true)}: TP={tp}, FP={fp}, FN={fn}, F1={f1:.3f}")
        self.assertGreaterEqual(f1, self.min_f1, msg=f"Image {self.index}: F1 score {f1:.3f} < {self.min_f1}")


class TestDetectImage1(TestDetect):
    index = 1
    min_f1 = 0.55


class TestDetectImage2(TestDetect):
    index = 2
    min_f1 = 0.55


class TestDetectImage3(TestDetect):
    index = 3
    min_f1 = 0.7


class TestDetectImage4(TestDetect):
    index = 4
    min_f1 = 0.75


class TestDetectionIoU(EvaluatingTestCase):
    # Minimal precision and recall of the detected blobs at IoU >= 0.5 with unique assignment of the boxes,
    # computed over all images together
    min_precision = 0.6
    min_recall = 0.6

    def test_precision_recall(self):
        tp, fp, fn = np.sum(
            [self.evaluate(blobs, gt) for blobs, gt in zip(self.params["detections"], self.params["boxes_true"])], axis=0
        )
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        print(f"Precision: {tp}/{tp + fp} = {precision:.3f}, recall: {tp}/{tp + fn} = {recall:.3f}")
        self.assertGreaterEqual(precision, self.min_precision, msg=f"Precision {precision:.3f} < {self.min_precision}")
        self.assertGreaterEqual(recall, self.min_recall, msg=f"Recall {recall:.3f} < {self.min_recall}")
