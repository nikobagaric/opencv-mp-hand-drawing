from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
from enums import Mode


class HandLandmarker:
    def __init__(self, model_path="hand_landmarker.task", num_hands=1):
        base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
        )
        self.tip_ids = [4, 8, 12, 16, 20]
        self.detector = vision.HandLandmarker.create_from_options(self.options)
        self.landmark_list = []
        self.activity_mode = Mode.HOVERING

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        MARGIN = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]
            coordinate_list = []

            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in hand_landmarks
                ]
            )

            for landmark in hand_landmarks:
                coordinate_list.append([landmark.x, landmark.y, landmark.z])
            self.landmark_list = coordinate_list

            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style(),
            )

            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            cv2.putText(
                annotated_image,
                self.activity_mode.name,
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                HANDEDNESS_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return annotated_image

    def draw_right_index_finger_on_image(rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            if handedness[0].category_name == "Right":
                right_index_landmarks = hand_landmarks[5:9]  # Landmarks 5, 6, 7, 8

                for i in range(len(right_index_landmarks) - 1):
                    start = right_index_landmarks[i]
                    end = right_index_landmarks[i + 1]

                    height, width, _ = annotated_image.shape
                    start_point = (int(start.x * width), int(start.y * height))
                    end_point = (int(end.x * width), int(end.y * height))

                    cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)

                for landmark in right_index_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(annotated_image, (x, y), 5, (255, 0, 0), -1)

        return annotated_image

    def euclidean_distance(self, point_a, point_b) -> float:
        return (
            (point_a[0] - point_b[0]) ** 2
            + (point_a[1] - point_b[1]) ** 2
            + (point_a[2] - point_b[2]) ** 2
        ) ** (1 / 2)

    def are_fingers_up(self, finger_list: list):
        """
        finger_list => value list 0-4
        """
        ret = True
        for finger in finger_list:
            if (
                self.landmark_list[self.tip_ids[finger]][1]
                < self.landmark_list[self.tip_ids[finger] - 1][1]
            ):
                ret = True
            else:
                ret = False
        return ret

    def get_pixel_coordinates(self, landmark_index, image_width, image_height):
        if not self.landmark_list or landmark_index >= len(self.landmark_list):
            return None 

        landmark = self.landmark_list[landmark_index]
        normalized_x, normalized_y = landmark[0], landmark[1]

        # Map normalized coordinates to pixel coordinates
        pixel_x = int(normalized_x * image_width)
        pixel_y = int(normalized_y * image_height)

        return (pixel_x, pixel_y)

    def check_mode(self):
        if self.landmark_list:
            if self.euclidean_distance(
                self.landmark_list[8], self.landmark_list[12]
            ) <= 0.1 and self.are_fingers_up([1, 2]):
                self.activity_mode = Mode.SELECTING
            elif self.are_fingers_up([1]):
                self.activity_mode = Mode.DRAWING
            else:
                self.activity_mode = Mode.HOVERING
