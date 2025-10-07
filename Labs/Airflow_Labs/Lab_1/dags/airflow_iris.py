# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.lab_iris import load_data, preprocess_data, train_model, test_model
from airflow import configuration as conf

# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')

# Define default arguments for your DAG
default_args = {
    'owner': 'your_name',
    'start_date': datetime(2025, 1, 15),
    'retries': 0,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5),  # Delay before retries
}

# Create a DAG instance named 'Airflow_Iris' with the defined default arguments
dag = DAG(
    'Airflow_Iris',
    default_args=default_args,
    description='DAG example for Iris dataset ML workflow',
    schedule_interval=None,  # Set to None for manual triggering
    catchup=False,
)

# Define PythonOperators for each function

# Task 1: Load data (Iris dataset)
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)

# Task 2: Preprocess the data
preprocess_data_task = PythonOperator(
    task_id='preprocess_data_task',
    python_callable=preprocess_data,
    op_args=[load_data_task.output],
    dag=dag,
)

# Task 3: Train the model
train_model_task = PythonOperator(
    task_id='train_model_task',
    python_callable=train_model,
    op_args=[preprocess_data_task.output],
    dag=dag,
)

# Task 4: Test the model
test_model_task = PythonOperator(
    task_id='test_model_task',
    python_callable=test_model,
    op_args=[train_model_task.output],
    dag=dag,
)

# Set task dependencies
load_data_task >> preprocess_data_task >> train_model_task >> test_model_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.cli()
