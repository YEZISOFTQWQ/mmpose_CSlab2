"""OpenCV renderers shared by real-time and batch pose inference."""

import cv2
import numpy as np


EDGES = ((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7),
         (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
         (8, 14), (14, 15), (15, 16))


def draw_2d_annotations(frame, predictions, show_keypoints=True, show_bbox=True):
    """Draw optional H36M-17 joints and detector boxes on a BGR image."""
    image = frame.copy()
    for number, prediction in enumerate(predictions, 1):
        if show_bbox:
            x1, y1, x2, y2 = prediction.bbox.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (77, 220, 135), 2)
            cv2.putText(image, f'person {number}: {prediction.score:.2f}',
                        (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .6,
                        (77, 220, 135), 2)
        if show_keypoints:
            points = prediction.keypoints_2d.astype(int)
            for start, end in EDGES:
                cv2.line(image, tuple(points[start]), tuple(points[end]),
                         (255, 185, 85), 2)
            for x, y in points:
                cv2.circle(image, (x, y), 4, (71, 190, 255), -1)
    return image


def render_3d_panel(predictions, height, width=None):
    """Render the largest H36M camera-coordinate pose to an efficient panel."""
    width = width or max(360, int(height * .72))
    panel = np.full((height, width, 3), (250, 250, 250), dtype=np.uint8)
    cv2.putText(panel, '3D pose | camera view', (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, .62, (35, 35, 35), 2)
    cv2.putText(panel, 'X: image horizontal | -Y: image up | Z: depth', (16, 55),
                cv2.FONT_HERSHEY_SIMPLEX, .42, (90, 90, 90), 1)
    if not predictions:
        cv2.putText(panel, 'No person detected', (width // 2 - 90, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (65, 65, 65), 2)
        return panel

    points = np.asarray(predictions[0].keypoints_3d, dtype=float)
    points = points - points[0]
    # Human3.6M camera coordinates: X follows image horizontal, Y follows image
    # down, and Z is depth. Therefore X/-Y gives the source-image orientation;
    # a small Z offset reveals depth without turning the skeleton sideways.
    projected = np.column_stack((points[:, 0] + .22 * points[:, 2],
                                 -points[:, 1] - .10 * points[:, 2]))
    extent = max(np.ptp(projected, axis=0).max(), .1)
    scale = min(width * .70, height * .64) / extent
    xy = projected * scale
    xy[:, 0] += width / 2
    xy[:, 1] = height * .60 - xy[:, 1]
    depth = points[:, 2]
    depth_norm = (depth - depth.min()) / max(np.ptp(depth), 1e-6)

    # Draw an axis triad in the same camera-coordinate convention. It is
    # anchored below the pose so it stays readable even when the body overlaps.
    axis_origin = np.array([58, height - 48], dtype=float)
    axis_size = min(width, height) * .11
    axis_vectors = {
        'X': (np.array([axis_size, 0]), (55, 55, 220)),
        '-Y': (np.array([0, -axis_size]), (70, 165, 70)),
        'Z': (np.array([axis_size * .45, axis_size * .28]), (220, 120, 35)),
    }
    for label, (vector, color) in axis_vectors.items():
        endpoint = axis_origin + vector
        cv2.arrowedLine(panel, tuple(axis_origin.astype(int)), tuple(endpoint.astype(int)),
                        color, 2, cv2.LINE_AA, tipLength=.16)
        cv2.putText(panel, label, tuple((endpoint + (4, -3)).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, color, 2, cv2.LINE_AA)
    cv2.circle(panel, tuple(axis_origin.astype(int)), 3, (40, 40, 40), -1, cv2.LINE_AA)

    for start, end in EDGES:
        segment_depth = float((depth_norm[start] + depth_norm[end]) / 2)
        color = (int(235 - 145 * segment_depth), int(110 + 110 * segment_depth),
                 int(80 + 160 * segment_depth))
        cv2.line(panel, tuple(xy[start].astype(int)), tuple(xy[end].astype(int)),
                 color, 3, cv2.LINE_AA)
    for point, depth_value in zip(xy, depth_norm):
        color = (int(235 - 145 * depth_value), int(110 + 110 * depth_value),
                 int(80 + 160 * depth_value))
        cv2.circle(panel, tuple(point.astype(int)), 6, color, -1, cv2.LINE_AA)
        cv2.circle(panel, tuple(point.astype(int)), 6, (65, 65, 65), 1, cv2.LINE_AA)
    return panel


def render_prediction(frame, predictions, show_keypoints=True, show_bbox=True,
                      show_3d=True):
    """Compose selectable 2D annotations and optional 3D panel."""
    if show_keypoints or show_bbox:
        output = draw_2d_annotations(frame, predictions, show_keypoints, show_bbox)
    else:
        output = frame.copy()
    if not show_3d:
        return output
    panel = render_3d_panel(predictions, output.shape[0])
    return np.hstack((output, panel))
