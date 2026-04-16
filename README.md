\# DeepShield AI



DeepShield AI is a deep learning system designed to detect deepfake videos using facial frame analysis.



The system analyzes facial frames extracted from videos and classifies them as \*\*REAL\*\* or \*\*FAKE\*\* using a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks.



---



\## Project Motivation



Deepfake technology has rapidly advanced and can be used to create highly realistic manipulated videos.



Such videos can be misused for:



\* misinformation

\* identity fraud

\* fake news

\* digital impersonation



DeepShield AI aims to detect such manipulated content by analyzing facial inconsistencies in video frames.



---



\## System Architecture



The DeepShield AI pipeline follows these steps:



Video Upload

↓

Frame Extraction (OpenCV)

↓

Face Detection (MediaPipe)

↓

Face Cropping

↓

CNN Feature Extraction (ResNet)

↓

Sequence Formation

↓

LSTM Temporal Modeling

↓

Classification (REAL / FAKE)



---



\## Tech Stack



Programming Language:

Python



Deep Learning Framework:

PyTorch



Computer Vision:

OpenCV

MediaPipe



Data Processing:

NumPy

Pandas

Scikit-learn



Backend:

FastAPI



Frontend:

HTML

CSS

JavaScript



---



\## Project Structure



DeepShield-AI/



dataset/

preprocessing/

models/

training/

inference/

backend/

frontend/

notebooks/



---



\## Development Roadmap



Project setup and environment configuration

Dataset preparation

Video frame extraction

Face detection and cropping

CNN feature extraction

LSTM temporal modeling

Model training

Model evaluation

Backend API development

Frontend interface



---



\## Author



Prasad Manolkar



This project is being developed as part of a machine learning portfolio to demonstrate skills in deep learning, computer vision, and full-stack AI system development.

Day 8 Progress: Implemented image-based deepfake classifier and video prediction pipeline.

