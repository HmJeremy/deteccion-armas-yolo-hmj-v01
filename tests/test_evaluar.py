import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deteccion_armas.evaluar import intersection_over_union  # noqa: E402


def test_iou_identical_boxes_is_one():
    box = (10, 10, 50, 50)
    assert intersection_over_union(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    box_a = (0, 0, 10, 10)
    box_b = (100, 100, 110, 110)
    assert intersection_over_union(box_a, box_b) == 0.0


def test_iou_partial_overlap():
    box_a = (0, 0, 10, 10)   # area 100
    box_b = (5, 0, 15, 10)   # area 100, overlap 5x10=50
    # union = 100 + 100 - 50 = 150 -> iou = 50/150
    assert abs(intersection_over_union(box_a, box_b) - (50 / 150)) < 1e-9


def test_iou_is_symmetric():
    box_a = (0, 0, 20, 20)
    box_b = (10, 10, 30, 30)
    assert intersection_over_union(box_a, box_b) == intersection_over_union(box_b, box_a)
