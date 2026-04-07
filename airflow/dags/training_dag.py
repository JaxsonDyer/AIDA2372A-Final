from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'employee_attrition_training_pipeline',
    default_args=default_args,
    description='A DAG to retrain the employee attrition model',
    schedule_interval=timedelta(days=7),
    catchup=False
)

# 1. Pull data with DVC
pull_data = BashOperator(
    task_id='pull_dvc_data',
    bash_command='cd /home/jaxsond/Documents/AIDA2372A_Final && dvc pull',
    dag=dag,
)

# 2. Train and log experiments with MLflow (train.py contains logic for this and next steps)
train_model = BashOperator(
    task_id='run_training_and_mlflow',
    bash_command='cd /home/jaxsond/Documents/AIDA2372A_Final && source .venv/bin/activate && python src/train.py',
    dag=dag,
)

pull_data >> train_model
