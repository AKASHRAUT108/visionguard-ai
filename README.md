# VisionGuard AI — Multimodal Defect Detection & Root Cause Intelligence

[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-24.0-blue)](https://www.docker.com/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**VisionGuard AI** is an end‑to‑end manufacturing quality control system that goes beyond simple defect detection. It combines computer vision, NLP, and predictive analytics to **detect defects**, **explain their root causes**, and **forecast future failures**—all delivered through a clean web interface.

---

## 🚀 Features

- **Defect Detection & Segmentation**  
  Uses a fine‑tuned EfficientNet‑B4 classifier with U‑Net / SAM segmentation to pinpoint defects and visualize them with Grad‑CAM heatmaps.

- **Root Cause Analysis**  
  Employs a BERT‑based zero‑shot classifier to infer probable causes from maintenance logs (Machine Calibration, Raw Material Defect, etc.).

- **Multimodal Fusion**  
  Integrates image and text embeddings via OpenAI CLIP to align visual defects with textual descriptions, improving explanation accuracy.

- **RAG‑Powered Chatbot**  
  Leverages LangChain + FAISS + Mistral 7B to answer natural‑language questions about corrective actions, pulling from a knowledge base of ISO standards and manuals.

- **Failure Prediction**  
  LSTM model trained on sensor time‑series data (SECOM dataset) predicts the probability of failure in the next production batch.

- **Interactive Dashboard**  
  Simple HTML/CSS/JS frontend that communicates with a FastAPI backend. Upload images, add sensor notes, and receive a comprehensive analysis.

- **Containerized Deployment**  
  Dockerized and ready to deploy on HuggingFace Spaces (or any cloud platform) with a single command.

---

## 🧰 Tech Stack

| Area                  | Technologies                                                                 |
|-----------------------|------------------------------------------------------------------------------|
| Computer Vision       | EfficientNet, U‑Net, SAM, Grad‑CAM, OpenCV                                   |
| NLP                   | BERT, Transformers, CLIP, Zero‑Shot Classification                           |
| RAG & Search          | LangChain, FAISS, Mistral 7B (via HuggingFace), Sentence‑Transformers        |
| Time Series           | LSTM, Scikit‑learn, Pandas                                                   |
| Backend               | FastAPI, Uvicorn, Python‑multipart                                           |
| Frontend              | HTML5, CSS3, JavaScript (vanilla)                                            |
| Deployment            | Docker, Docker Compose, GitHub Actions (CI/CD), HuggingFace Spaces           |
| Version Control       | Git, GitHub                                                                  |

---

## 📁 Project Structure
visionguard-ai/
├── backend/
│ ├── app/
│ │ ├── main.py # FastAPI application
│ │ ├── models/ # All ML models
│ │ │ ├── classifier.py
│ │ │ ├── segmenter.py
│ │ │ ├── clip_fusion.py
│ │ │ ├── root_cause_bert.py
│ │ │ ├── lstm_predictor.py
│ │ │ └── rag_engine.py
│ │ ├── utils/ # Helper functions (gradcam, PDF report)
│ │ └── static/ # Frontend assets
│ │ ├── index.html
│ │ ├── style.css
│ │ └── script.js
│ └── requirements.txt
├── data/ # Sample images / datasets (optional)
├── notebooks/ # Training experiments (optional)
├── scripts/ # Utility scripts
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md


---

## 🖥️ Live Demo

🚀 **Try it yourself:** [https://huggingface.co/spaces/AKASHRAUT108/visionguard-ai](https://huggingface.co/spaces/AKASHRAUT108/visionguard-ai)  
*(This link will work after deployment – update it once your Space is live.)*

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.9+
- Git
- Docker (optional, for containerized run)

### 1. Clone the Repository
```bash
git clone https://github.com/AKASHRAUT108/visionguard-ai.git
cd visionguard-ai