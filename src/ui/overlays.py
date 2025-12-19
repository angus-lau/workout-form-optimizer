import cv2
import numpy as np

class Overlays:
    """
    Class for drawing pose overlays on video frame
    """

    def __init__(self):
        self.joint_radius: int = 5
        self.joint_color: tuple[int, int, int] = (0, 255, 0)
        self.joint_thickness: int = -1

        self.skeleton_color: tuple[int, int, int] = (255, 0, 0)
        self.skeleton_thickness: int = 2

        self.text_color: tuple[int, int, int] = (225, 225, 225)
        self.text_scale: float = 1.0
        self.text_thickness: int = 1
        self.text_font: int = cv2.FONT_HERSHEY_PLAIN

        self.connections: list[tuple[str, str]] = [
            # Upper body
            ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
            ("LEFT_SHOULDER", "LEFT_ELBOW"),
            ("LEFT_ELBOW", "LEFT_WRIST"),
            ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
            ("RIGHT_ELBOW", "RIGHT_WRIST"),

            # Torso
            ("LEFT_SHOULDER", "LEFT_HIP"),
            ("RIGHT_SHOULDER", "RIGHT_HIP"),
            ("LEFT_HIP", "RIGHT_HIP"),

            # Legs
            ("LEFT_HIP", "LEFT_KNEE"),
            ("LEFT_KNEE", "LEFT_ANKLE"),
            ("RIGHT_HIP", "RIGHT_KNEE"),
            ("RIGHT_KNEE", "RIGHT_ANKLE"),

            # Feet
            ("LEFT_ANKLE", "LEFT_HEEL"),
            ("LEFT_HEEL", "LEFT_FOOT_INDEX"),
            ("RIGHT_ANKLE", "RIGHT_HEEL"),
            ("RIGHT_HEEL", "RIGHT_FOOT_INDEX"),

            # Face
            ("NOSE", "LEFT_EYE"),
            ("NOSE", "RIGHT_EYE"),
            ("LEFT_EYE", "LEFT_EAR"),
            ("RIGHT_EYE", "RIGHT_EAR"),
        ]
        
    def draw_skeleton(self, frame: np.ndarray, joints: dict[str, tuple[int, int]], angles: dict[str, float] | None = None) -> None:
        """
        Draw a full pose skeleton, including joints and angle labels.

        Parameters:
            frame:
                The image to draw on
            joints:
                A dictionary mapping joint names to pixel coordinates (x, y) on the frame.
            angles:
                Optinoal dictionary mapping joint names to angle values in degrees.
        """

        self.draw_connections(frame, joints)

        if angles is None:
            angles = {}

        for key, coord in joints.items():
            cv2.circle(frame, coord, self.joint_radius, self.joint_color, self.joint_thickness)

            if angles.get(key) is not None:
                x = coord[0] + 10
                y = coord[1]
                self.draw_angle(frame, angles.get(key), (x, y)) 

    def draw_connections(self, frame: np.ndarray, joints: dict[str, tuple[int, int]]) -> None:
        """
        Draw all skeleton lines between connected joints. Helper
        
        Parameters:
            frame:
                The image to draw on
            joints:
                A dictionary mapping joint names to pixel coordinates (x, y) on the frame.
        """

        for i in self.connections:
            pt1 = joints.get(i[0])
            pt2 = joints.get(i[1])
            if pt1 is None or pt2 is None:
                continue
            cv2.line(frame, pt1, pt2, self.skeleton_color, self.skeleton_thickness) 

    def draw_angle(self, frame: np.ndarray, angle: float, coord: tuple[int, int]) -> None:
        """
        Draw an angle value at coordinate.

        Parameters:
            frame:
                The image to draw on
            angle:
                The angle in degrees to display
            coord:
                The (x, y) pixel coordinate
        """
        if angle is None:
            return
        text = str(int(round(angle)))
        cv2.putText(frame, text, coord, self.text_font, self.text_scale, self.text_color, self.text_thickness)


    def feedback_text(self, frame: np.ndarray, msg: str) -> None:
        """
        Draw feedback message centered near the bottom of the frame.

        Parameters:
            frame:
                The image to draw on
            msg:
                The message string to display
        """
        height, width, _ = frame.shape
        (text_width, _), _ = cv2.getTextSize(msg, self.text_font, self.text_scale, self.text_thickness)
        centerX = int((width - text_width) // 2)

        y = int(height - height/10)
        cv2.putText(frame, msg, (centerX, y), self.text_font, self.text_scale, self.text_color, self.text_thickness)
        