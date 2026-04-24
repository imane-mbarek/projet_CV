from __future__ import annotations

import cv2

from src.preprocessing import FramePreprocessor


def main() -> None:
    video_path = "data/test_video.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la vidéo.")
        return

    preprocessor = FramePreprocessor()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        clean_frame, gray_frame = preprocessor.get_clean_frame(frame)

        cv2.imshow("Original", frame)
        cv2.imshow("Gray Frame", gray_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()