"""
main.py – Entry point for Face Recognition & Attendance System
"""

import sys
import traceback


def main():
    try:
        # Simple dependency check (safer method)
        missing = []

        try:
            import cv2
        except ImportError:
            missing.append("opencv-python")

        try:
            import face_recognition
        except ImportError:
            missing.append("face_recognition")

        try:
            import numpy
        except ImportError:
            missing.append("numpy")

        if missing:
            raise ImportError(
                "Missing required package(s):\n\n"
                + "\n".join(f"  pip install {p}" for p in missing)
            )

        from gui import FaceRecognitionGUI
        app = FaceRecognitionGUI()
        app.run()

    except Exception as exc:
        msg = (
            f"The application failed to start.\n\n"
            f"Error: {exc}\n\n"
            "Ensure dependencies are installed:\n"
            "  pip install face_recognition opencv-python numpy pillow"
        )
        print(msg, file=sys.stderr)
        traceback.print_exc()

        try:
            import tkinter.messagebox as mb
            mb.showerror("Startup Error", msg)
        except Exception:
            pass

        sys.exit(1)


if __name__ == "__main__":
    main()