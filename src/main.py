from __future__ import annotations

import cv2

from src.preprocessing import FramePreprocessor
from src.detection import HumanDetector, TrackingManager


def main() -> None:
    video_path = "data/test_video.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la vidéo.")
        return

    preprocessor = FramePreprocessor()
    detector = HumanDetector()
    tracker = TrackingManager()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        clean_frame, gray_frame = preprocessor.get_clean_frame(frame)
        detections = detector.detect(gray_frame)
        tracked_persons = tracker.update(detections)

        display_frame = clean_frame.copy()

        for person in tracked_persons:
            x, y, w, h = person.bbox
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                display_frame,
                f"ID {person.person_id}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imshow("SafeSwim - Tracking", display_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()