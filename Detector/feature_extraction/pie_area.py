import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __str__(self):
        return f"({self.x}, {self.y})"


def get_minibox_area(x, y, r):
    """Assuming the center of the circle is at point (0,0) - top left."""
    if x <= 0 or y <= 0:
        return 0
    if np.sqrt(x ** 2 + y ** 2) < r:
        return 0
    sqrt = np.sqrt(r ** 2 - y ** 2) if r ** 2 - y ** 2 > 0 else 0
    q1 = Point(sqrt, y)
    sqrt = np.sqrt(r ** 2 - x ** 2) if r ** 2 - x ** 2 > 0 else 0
    q2 = Point(x, sqrt)
    return np.arctan2(q1.y, q1.x) - np.arctan2(q2.y, q2.x)


def get_box_area(p1, p2, r, center):
    # to R^2
    n_p1 = Point(-(center.x - p1.x), (center.y - p1.y))
    n_p2 = Point((p2.x - center.x), -(p2.y - center.y))

    boxes_xy = [(abs(n_p2.x), abs(n_p2.y)),
                (abs(n_p1.y), abs(n_p2.x)),
                (abs(n_p1.x), abs(n_p1.y)),
                (abs(n_p2.y), abs(n_p1.x))]

    return sum([get_minibox_area(box[0], box[1], r) for box in boxes_xy])


def num_sides_intersections(box1, box2):
    sides = [np.isclose(box1[0].x, box2[0].x, atol=4),
             np.isclose(box1[1].x, box2[1].x, atol=4),
             np.isclose(box1[0].y, box2[0].y, atol=4),
             np.isclose(box1[1].y, box2[1].y, atol=4),
             ]

    return sum(sides)


def boxes_to_points(boxes):
    point_boxes = []
    for i in range(len(boxes)):
        point_boxes.append([Point(boxes[i][0], boxes[i][1]), Point(boxes[i][2], boxes[i][3])])
    return point_boxes


def get_pie_areas(detector_bboxes):
    PREAVIEW_AREA_TRESHOLD = 450
    vis_boxes = boxes_to_points(detector_bboxes)

    boxes = [box for box in vis_boxes if ((box[1].x - box[0].x) * (box[1].y - box[0].y)) >= PREAVIEW_AREA_TRESHOLD]
    if len(boxes) == 0:
        return [None] * len(detector_bboxes), None, None
    pie_box_p1 = Point(min([box[0].x for box in boxes]), min([box[0].y for box in boxes]))
    pie_box_p2 = Point(max([box[1].x for box in boxes]), max([box[1].y for box in boxes]))
    r = max(pie_box_p2.x - pie_box_p1.x, pie_box_p2.y - pie_box_p1.y) / 2
    center = Point((pie_box_p1.x + pie_box_p2.x) / 2, (pie_box_p1.y + pie_box_p2.y) / 2)

    biggest = sorted(boxes, key=lambda b: num_sides_intersections((pie_box_p1, pie_box_p2), b))[-1]
    angles = [get_box_area(box[0], box[1], r, center) for box in boxes]

    with_nones = []
    for i in range(len(boxes)):
        if boxes[i] == biggest:
            angles[i] = 2 * np.pi - sum(angles[:i] + angles[i + 1:])
    i = 0
    for j in range(len(vis_boxes)):
        if ((vis_boxes[j][1].x - vis_boxes[j][0].x) * (
                vis_boxes[j][1].y - vis_boxes[j][0].y)) >= PREAVIEW_AREA_TRESHOLD:
            with_nones.append(angles[i])
            i += 1
        else:
            with_nones.append(None)

    return with_nones, center, r
