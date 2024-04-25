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

# importing tkinter for graphical user interface
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

conn = None
task_queue = queue.Queue()


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Drug Analysis GUI")
        self.geometry("1000x600")

        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

        self.create_widgets()

    def create_widgets(self):
        self.label_file = tk.Label(self, text="Select CSV file:")
        self.label_file.pack()

        self.button_browse = tk.Button(self, text="Browse", command=self.browse_file)
        self.button_browse.pack()

        self.label_feature = tk.Label(self, text="Select feature for preprocessing:")
        self.label_feature.pack()

        self.selected_feature = tk.StringVar(self)
        self.dropdown_feature = tk.OptionMenu(self, self.selected_feature, "")
        self.dropdown_feature.pack()

        self.label_target = tk.Label(self, text="Select target feature:")
        self.label_target.pack()

        self.selected_target = tk.StringVar(self)
        self.dropdown_target = tk.OptionMenu(self, self.selected_target, "")
        self.dropdown_target.pack()

        self.button_preprocess = tk.Button(self, text="Preprocess Data", command=self.preprocess_data)
        self.button_preprocess.pack()


        self.button_plot_feature = tk.Button(self, text="Plot Feature", command=self.plot_feature)
        self.button_plot_feature.pack()

        self.output_text = tk.Text(self, height=10, width=60)
        self.output_text.pack()

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            self.update_dropdown_features()
            self.output_text.insert(tk.END, f"Loaded CSV file: {file_path}\n")
        else:
            self.output_text.insert(tk.END, "Error: No file selected.\n")

    def update_dropdown_features(self):
        if self.df is not None:
            features = list(self.df.columns)
            self.dropdown_feature.destroy()
            self.selected_feature.set("")  # Reset selected feature
            self.dropdown_feature = tk.OptionMenu(self, self.selected_feature, *features)
            self.dropdown_feature.pack()

            self.dropdown_target.destroy()
            self.selected_target.set("")  # Reset selected target feature
            self.dropdown_target = tk.OptionMenu(self, self.selected_target, *features)
            self.dropdown_target.pack()

    def preprocess_data(self):
        if self.df is None:
            messagebox.showerror("Error", "No CSV file loaded.")
            return

        feature = self.selected_feature.get()
        target = self.selected_target.get()
        if not feature or not target:
            messagebox.showerror("Error", "Please select both feature and target feature.")
            return

        self.output_text.insert(tk.END, f"Preprocessing data with feature: {feature} and target feature: {target}\n")
        try:
            # Drop missing values
            self.df = self.df.dropna()

            # Split the data into input features (X) and target variable (y)
            X = self.df.drop(columns=[target], axis=1)  # Exclude target variable
            y = self.df[target]

            # Split the data into training and testing sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3,
                                                                                    random_state=50)

            # Standardize selected feature column only
            scaler = StandardScaler()
            self.X_train[feature] = scaler.fit_transform(self.X_train[[feature]])
            self.X_test[feature] = scaler.transform(self.X_test[[feature]])

            self.output_text.insert(tk.END, "Data preprocessing completed.\n")
        except Exception as e:
            messagebox.showerror("Error", f"Data preprocessing error: {e}")

    def train_model(self):
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            messagebox.showerror("Error", "Data not preprocessed yet.")
            return

        try:
            self.model = LinearRegression()
            self.model.fit(self.X_train, self.y_train)
            self.output_text.insert(tk.END, "Model training completed.\n")
        except Exception as e:
            messagebox.showerror("Error", f"Model training error: {e}")

    def plot_feature(self):
        if self.df is None:
            messagebox.showerror("Error", "No CSV file loaded.")
            return

        feature = self.selected_feature.get()
        if not feature:
            messagebox.showerror("Error", "No feature selected for plotting.")
            return

        if feature not in self.df.columns:
            messagebox.showerror("Error", "Invalid feature selected.")
            return

        try:
            # Plot the selected feature against the target variable
            plt.scatter(self.X_train[feature], self.y_train, color='blue', label='Training Data')
            plt.scatter(self.X_test[feature], self.y_test, color='red', label='Testing Data')

            # Fit a linear regression model to the data
            model = LinearRegression()
            model.fit(self.X_train[[feature]], self.y_train)

            # Plot the regression line
            plt.plot(self.X_train[feature], model.predict(self.X_train[[feature]]), color='green',
                     label='Regression Line')

            plt.xlabel(feature)
            plt.ylabel('Target')
            plt.title(f'{feature} vs Target')
            plt.legend()
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Plotting error: {e}")





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






if __name__ == "__main__":

    # Close database connection
    app = Application()
    app.mainloop()

