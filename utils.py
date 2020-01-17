import cv2 as cv
import numpy as np

from align_faces import get_reference_facial_points, warp_and_crop_face
from retinaface.detector import detector


def align_face(raw, facial5points):
    # raw = cv.imread(img_fn, True)  # BGR
    facial5points = np.reshape(facial5points, (2, 5))

    crop_size = (112, 112)

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    output_size = (112, 112)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    # dst_img = warp_and_crop_face(raw, facial5points)
    dst_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
    return dst_img


def get_face_attributes(full_path):
    try:
        img = cv.imread(full_path)
        bounding_boxes, landmarks = detector.detect_faces(img)

        if len(landmarks) > 0:
            landmarks = [int(round(x)) for x in landmarks[0]]
            return True, landmarks

    except KeyboardInterrupt:
        raise
    except:
        pass
    return False, None


def select_significant_face(bboxes):
    best_index = -1
    best_rank = float('-inf')
    for i, b in enumerate(bboxes):
        bbox_w, bbox_h = b[2] - b[0], b[3] - b[1]
        area = bbox_w * bbox_h
        score = b[4]
        rank = score * area
        if rank > best_rank:
            best_rank = rank
            best_index = i

    return best_index


def get_central_face_attributes(full_path):
    try:
        img = cv.imread(full_path)
        bboxes, landmarks = detector.detect_faces(img)

        if len(landmarks) > 0:
            i = select_significant_face(bboxes)
            return True, [bboxes[i]], [landmarks[i]]

    except KeyboardInterrupt:
        raise
    except ValueError:
        pass
    except IOError:
        pass
    return False, None, None


def get_all_face_attributes(full_path):
    img = cv.imread(full_path)
    bounding_boxes, landmarks = detector.detect_faces(img)
    return bounding_boxes, landmarks


def draw_bboxes(img, bounding_boxes, facial_landmarks=[]):
    for b in bounding_boxes:
        cv.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 1)

    for p in facial_landmarks:
        for i in range(5):
            cv.circle(img, (int(p[i]), int(p[i + 5])), 1, (0, 255, 0), -1)

        break  # only first

    return img
