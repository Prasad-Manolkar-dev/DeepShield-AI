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



Day 1 – Project setup and environment configuration

Day 2 – Dataset preparation

Day 3 – Video frame extraction

Day 4 – Face detection and cropping

Day 5 – CNN feature extraction

Day 6 – LSTM temporal modeling

Day 7 – Model training

Day 8 – Model evaluation

Day 9 – Backend API development

Day 10 – Frontend interface



---



\## Author



Prasad Manolkar



This project is being developed as part of a machine learning portfolio to demonstrate skills in deep learning, computer vision, and full-stack AI system development.



