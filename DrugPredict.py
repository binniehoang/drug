import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import multiprocessing
import time
import os
import portalocker
import csv
import queue
import threading
# import signal library
import signal
import sys
import atexit
# Initialize global variables
conn = None
task_queue = queue.Queue()


# Function to perform data preprocessing
def preprocess_data(df, lock):
    print("Data preprocessing started...")
    with lock:
        df = df.dropna()
        df = df.drop('State', axis=1)
        scaler = StandardScaler()
        X = df.drop('Rates.Illicit Drugs.Cocaine Used Past Year.18-25', axis=1)
        y = df['Rates.Illicit Drugs.Cocaine Used Past Year.18-25']
        X = scaler.fit_transform(X)
    print("Data preprocessing completed.")
    return X, y


# Function to train a linear regression model
def train_model(X_train, y_train, lock):
    print("Model training started...")
    with lock:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
    print("Model training completed.")
    return lr


# Function to perform model evaluation
def evaluate_model(model, X_test, y_test, lock):
    print("Model evaluation started...")
    with lock:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        plt.scatter(y_test, y_pred, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        plt.xlabel('Observed')
        plt.ylabel('Predicted')
        plt.title('Observed vs Predicted values')
        plt.show()
    print("Model evaluation completed.")


def write_to_csv(filename, data):
    lockfile = 'predicted_data.csv' + ".lock"
    with open(lockfile, 'w') as lock:
        portalocker.lock(lock, portalocker.LOCK_EX)
        try:
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data)
        finally:
            portalocker.unlock(lock)

# Function to handle process for data preprocessing and model training
def process_data(df, lock):
    X, y = preprocess_data(df, lock)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
    model = train_model(X_train, y_train, lock)
    evaluate_model(model, X_test, y_test, lock)

    write_to_csv('predicted_data.csv', [y_test, model.predict(X_test)])


def worker():
    while True:
        task = task_queue.get()
        if task is None:
            break
        process_data(*task)
        task_queue.task_done()

# Function to handle signals
def signal_handler(sig, frame):
    print('Termination signal received. Cleaning up...')
    close_database_connection()
    sys.exit(0)

# Function to handle exit
def exit_handler():
    print('Exiting program. Cleaning up...')
    close_database_connection()

# Function to close database connection
def close_database_connection():
    global conn
    if conn is not None:
        conn.close()
        print("Database connection closed.")

if __name__ == "__main__":
    # Set signal handler for termination signal
    signal.signal(signal.SIGINT, signal_handler)
    # Set exit handler
    atexit.register(exit_handler)

    # Data collection and storage
    print("Data collection and storage started...")
    start_time = time.time()
    df = pd.read_csv("drugs.csv")
    conn = sqlite3.connect('Drug_data.db')
    df.to_sql('drug', conn, if_exists='replace', index=False)
    df = pd.read_sql_query("SELECT * FROM drug", conn)
    end_time = time.time()
    print("Data collection and storage completed.")
    print(f"Time taken for data collection and storage: {end_time - start_time} seconds")

    # Create a lock for synchronization
    lock = multiprocessing.Lock()

    # Create a thread for each task
    num_worker_threads = 2
    threads = []
    results = []
    for i in range(num_worker_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    # Add tasks to the queue
    task_queue.put((df, lock))

    # Wait for all tasks to be completed
    task_queue.join()

    # Stop the worker threads
    for i in range(num_worker_threads):
        task_queue.put(None)
    for t in threads:
        t.join() # Collect results

    # Create a process for data processing and model training
    print("Starting data processing and model training process...")
    process_start_time = time.time()
    data_process = multiprocessing.Process(target=process_data, args=(df, lock))
    data_process.start()

    # Continue with other tasks in the main process
    # Example: Handling user authentication, web requests, etc.
    print("Main process continues...")

    # Wait for the data processing process to finish
    data_process.join()
    process_end_time = time.time()
    print("Data processing and model training process completed.")
    print(f"Total time taken for data processing and model training: {process_end_time - process_start_time} seconds")

    # Close database connection
    close_database_connection()
