# End-to-End MLOps Pipeline for Employee Attrition Prediction

![CI/CD Pipeline](https://github.com/Placeholder/EmployeeAttrition/actions/workflows/main.yml/badge.svg)

## Overview
This project operationalizes a classification model predicting employee attrition using the **IBM HR Analytics Employee Attrition dataset**.
It covers the full MLOps lifecycle from data versioning to automated deployment.

## Key Technologies & Architecture
1. **Data Versioning (DVC)**: The raw dataset is tracked using DVC with an Amazon S3 remote.
2. **Experiment Tracking (MLflow)**: Multiple experiments are tracked locally, comparing F1 score and accuracy of Logistic Regression, Random Forest, and XGBoost models. Best performing models are registered in the MLflow model registry.
3. **Model Serving (Flask & Docker)**: The best registered model is serialized and packaged into a Docker container serving a `/predict` REST API.
4. **CI/CD Automation (GitHub Actions)**: Every push to `main` triggers a complete CI/CD pipeline which lints code (`flake8`), runs unit tests (`pytest`), builds the Docker image, and pushes it to Docker Hub.
5. **Workflow Orchestration (Airflow)**: An Apache Airflow DAG schedules retraining on a weekly interval by pulling the latest DVC data and re-running the MLflow experiment pipeline.

## Project Structure
```
.
├── airflow/
│   └── dags/
│       └── training_dag.py        # Airflow DAG for retraining
├── data/                          # Data directory tracked by DVC (ignores contents in Git)
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv 
├── src/
│   ├── app.py                     # Flask API for inference
│   └── train.py                   # Model training and MLflow tracking script
├── tests/
│   └── test_api.py                # Unit tests for the Flask API
├── .github/workflows/main.yml     # CI/CD action
├── Dockerfile                     # Container definition
├── requirements.txt               # Dependencies
└── dvc.yaml                       # DVC pipelines
```

## How to Run Locally

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Pull Data
```bash
dvc pull
```

### 3. Run Experiments
```bash
python src/train.py
```
This will train the models and register the best one into MLflow. To see the dashboard:
```bash
mlflow ui
```

### 4. Run API locally
```bash
python src/app.py
```
Send a request to the API:
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '[{"Age": 30, "DailyRate": 1000}]'
```

### 5. Build Docker Image
```bash
docker build -t employee-attrition-model .
docker run -p 5000:5000 employee-attrition-model
```

## Automations & Orchestration
* **Airflow**: To run Airflow, start your airflow webserver and scheduler pointing to `airflow/` directory.
* **CI/CD**: Uses the `.github/workflows/main.yml`.

## Future Improvements
* Integrating `Great Expectations` to perform automated data quality checks before the DAG training.
* Deploying `Prometheus` and `Grafana` to monitor API response times and track prediction distributions to capture drift.
