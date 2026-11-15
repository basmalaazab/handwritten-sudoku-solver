# Deployment Guide — Handwritten Sudoku Reader & Solver

This guide outlines how to run and deploy the interactive Streamlit Web Application locally, with Docker, or to cloud hosting providers.

---

## 1. Local Run (Fastest)

The application is already installed and running locally.

### On Windows
You can run either:
- Double-click `run_app.bat` in the project root folder.
- Or run PowerShell script:
  ```powershell
  .\run_app.ps1
  ```
- Or run manually:
  ```powershell
  cd p7
  python -m streamlit run app.py
  ```

Access the UI at: **[http://localhost:8501](http://localhost:8501)**

---

## 2. Docker Container Deployment

A production-ready `Dockerfile` and `docker-compose.yml` are included.

### Using Docker Compose
```bash
cd p7
docker compose up --build
```

### Using Docker CLI
```bash
cd p7
docker build -t handwritten-sudoku-solver .
docker run -d -p 8501:8501 --name sudoku-solver handwritten-sudoku-solver
```

Open **[http://localhost:8501](http://localhost:8501)**.

---

## 3. Cloud Deployment

### Option A: Hugging Face Spaces (Free ML Hosting)
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Select **Streamlit** as the SDK.
3. Push the files (`app.py`, `p7/`, `data/`, `results/`, `requirements.txt`).
4. Hugging Face will automatically detect `app.py` and deploy it with a public HTTPS URL.

### Option B: Render / Railway / Fly.io
1. Connect your GitHub repository to [Render](https://render.com) or [Railway](https://railway.app).
2. Choose **Web Service** with Docker or Python environment.
3. Set start command:
   ```bash
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```
4. Set port to `8501`.
