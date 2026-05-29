# AFRAS – Automated Facial Recognition Attendance System

AFRAS is a desktop-based attendance management system that uses facial recognition technology to automatically detect and mark attendance in real time.

---

## Features

- Real-time face recognition
- Face registration using webcam or image
- Automatic attendance marking
- CSV attendance export
- Duplicate attendance prevention
- Modern Tkinter GUI
- Adjustable recognition settings

---

## Technologies Used

- Python
- OpenCV
- face_recognition (dlib)
- NumPy
- Tkinter
- Pillow

---

## Project Structure

```text
AFRAS/
│
├── face_system.py
├── gui.py
├── main.py
├── Launch_AFRAS.bat
├── requirements.txt
├── README.md
```

---

## Installation

Create Conda environment:

```bash
conda create -n face python=3.10
conda activate face
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running AFRAS

Double click:

```text
Launch_AFRAS.bat
```

OR run manually:

```bash
python main.py
```

---

## Current Features

- Webcam-based face registration
- Real-time recognition using HOG model
- Attendance logging with date and time
- CSV export support
- Theme toggle support
- Recognition settings control

---

## Future Enhancements

- Database integration
- Cloud synchronization
- Multi-camera support
- Mobile application
- Advanced CNN recognition
- Liveness detection

---

## Author

- Devansh Singh Bais
