import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import multiprocessing


import portalocker
import csv
import queue

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
        self.geometry("1000x1000")

        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.num_cores = multiprocessing.cpu_count()
        self.create_widgets()

    def write_predicted_data_to_csv(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not trained yet.")
            return

        if self.X_test is None:
            messagebox.showerror("Error", "No test data available.")
            return

        try:
            # Predict on test data
            predicted_values = self.model.predict(self.X_test)

            # Ask the user to select a file to save the predicted data
            filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
            if not filename:
                return
            lockfile = filename + ".lock"
            with open(lockfile, 'w') as lock:
                portalocker.lock(lock, portalocker.LOCK_EX)
                try:
                    # Write predicted data to CSV file
                    with open(filename, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['Predicted Value'])
                        writer.writerows(
                            [[value] for value in predicted_values])  # Convert each value into a list before writing
                    self.output_text.insert(tk.END, f"Predicted data written to {filename}\n")
                finally:
                    portalocker.unlock(lock)

        except Exception as e:
            messagebox.showerror("Error", f"Error writing predicted data: {e}")

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

        self.label_write_data = tk.Label(self, text="Write Predicted Data:")
        self.label_write_data.pack()

        self.button_write_data = tk.Button(self, text="Write Predicted Data to CSV",
                                           command=self.write_predicted_data_to_csv)
        self.button_write_data.pack()

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

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.destroy()


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
            self.df = self.df.drop(columns=['State'])
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
            # with multiprocessing.Pool(processes=self.num_cores):
            #     self.model.fit(self.X_train, self.y_train)
            self.output_text.insert(tk.END, "Model training completed.\n")
        except Exception as e:
            messagebox.showerror("Error", f"Model training error: {e}")

    def plot_feature(self):
        if self.df is None:
            messagebox.showerror("Error", "No CSV file loaded.")
            return

        feature = self.selected_feature.get()
        target = self.selected_target.get()
        if not feature or not target:
            messagebox.showerror("Error", "Please select both feature and target feature.")
            return

        if feature not in self.df.columns or target not in self.df.columns:
            messagebox.showerror("Error", "Invalid feature or target feature selected.")
            return

        try:
            # Preprocess data
            self.preprocess_data()

            # Train the model
            self.train_model()

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
            plt.ylabel(target)
            plt.title(f'{feature} vs {target}')
            plt.legend()
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Plotting error: {e}")




if __name__ == "__main__":


    app = Application()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()



