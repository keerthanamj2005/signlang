# SignLang – Real-Time Sign Language Detection

A real-time Sign Language Detection system built using MediaPipe, OpenCV, and a Random Forest Classifier. The application captures hand landmarks through a webcam, extracts features using MediaPipe, and predicts sign language gestures in real time.

## Features

* Real-time hand tracking using MediaPipe
* Hand landmark extraction and preprocessing
* Random Forest based gesture classification
* Live gesture prediction through webcam
* Modular project structure for training and inference

## Tech Stack

* Python
* OpenCV
* MediaPipe
* Scikit-learn
* NumPy

## Project Structure

* `src/` – Data collection, preprocessing, training, and inference scripts
* `mediapipe_utils/` – Hand detection and landmark extraction utilities
* `models/` – Trained machine learning models
* `dataset/` – Training dataset
* `gui/` – User interface components
* `main.py` – Application entry point

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python main.py
```

## Future Enhancements

* Support for full sentence recognition
* Deep learning based gesture classification
* Text-to-speech integration
* Expanded gesture vocabulary
