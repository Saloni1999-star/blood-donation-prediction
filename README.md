Blood Donation Prediction 


Blood Donation Prediction Project

Overview

This project aims to predict whether an individual will donate blood, based on historical donation data. By leveraging machine learning models, we seek to identify key factors influencing donation behavior, which can assist blood banks in optimizing their recruitment strategies and improving donor engagement.

Dataset

The project utilizes the transfusion.csv dataset, which contains information about blood donors, including:

Recency (months): Months since last donation.

Frequency (times): Total number of donations.

Monetary (c.c. blood): Total blood donated in c.c.

Time (months): Months since first donation.

Target (whether he/she donated blood in March 2007): Binary variable indicating if the donor donated blood in March 2007 (1 = Yes, 0 = No).

The dataset is located in the dataset M/ directory within this repository.


Project Workflow

The project follows a standard machine learning pipeline:

Data Loading and Initial Exploration:

The transfusion.csv dataset was loaded using pandas.

Initial data checks (e.g., data.head(), data.info(), data.describe(), data.isnull().sum()) were performed to understand its structure, identify missing values, and check data types.

Column names were reviewed for clarity and consistency.

Data Preprocessing:

Features (X) and the target variable (y) were separated from the dataset.

The data was split into training and testing sets (80% training, 20% testing) to ensure robust model evaluation.
Model Training - Logistic Regression:

A Logistic Regression model, a common baseline for binary classification, was trained on the preprocessed training data.
Automated Machine Learning (AutoML) - TPOT:


TPOT (Tree-based Pipeline Optimization Tool) was used to automatically discover the best machine learning pipeline (including data preprocessing, feature selection, and model selection) for the dataset. This helps in exploring a wide range of models and hyperparameters efficiently.


Model Evaluation:

Both the Logistic Regression model and the best model found by TPOT were evaluated using key classification metrics:
Accuracy: Overall proportion of correct predictions.

Confusion Matrix: Breakdown of True Positives, True Negatives, False Positives, and False Negatives.

Classification Report: Provides Precision, Recall, and F1-score for each class (donated/not donated).

A Confusion Matrix heatmap was generated for visualization.
Model Interpretation:

The coefficients of the Logistic Regression model were analyzed to understand the impact of each feature on the prediction of blood donation. 

A bar plot of coefficients was generated for visual interpretation.

The structure of the best pipeline identified by TPOT was inspected for insights into automated model selection.

Mean-Variance-Standard Deviation CalculatorThis project implements a Python function that calculates various statistical properties (mean, variance, standard deviation, 

max, min, and sum) for a 3x3 matrix derived from a list of nine numbers. 

The calculations are performed for the rows, columns, and the flattened matrix.FeaturesThe calculate function takes a list of 9 numbers and returns a dictionary containing the following statistics:Mean: Calculated for columns, rows, and the flattened matrix.Variance: Calculated for columns, rows, and the flattened matrix.Standard Deviation: 

Calculated for columns, rows, and the flattened matrix.Max: The maximum value for columns, rows, and the flattened matrix.

Min: The minimum value for columns, rows, and the flattened matrix.Sum: The sum of values for columns, rows, and the flattened matrix.If the input list does not contain exactly nine numbers, a ValueError is raised.

SetupTo run this project, you'll need Python and the NumPy library installed.

PrerequisitesPython 3.xpip (Python package installer)InstallationClone the repository (if applicable) or create the project files:If you're working in a Codespace, you likely already have the files. 

Otherwise, ensure you have the following files in your project directory:mean_var_std.pymain.pytest_module.py (optional, for running tests)Install NumPy:Open your terminal or command prompt and run:pip install numpy

Usagemean_var_std.pyThis file contains the core calculate function.import numpy as np

def calculate(list_of_nine_numbers):
    """
    Calculates mean, variance, standard deviation, max, min, and sum
    for rows, columns, and the flattened 3x3 NumPy array created from
    the input list.

    Args:
        list_of_nine_numbers: A list containing exactly nine numbers.

    Returns:
        A dictionary containing the calculated statistics.

    Raises:
        ValueError: If the input list does not contain nine numbers.
    """
    if len(list_of_nine_numbers) != 9:
        raise ValueError("List must contain nine numbers.")

    np_array = np.array(list_of_nine_numbers).reshape(3, 3)

    mean_columns = np_array.mean(axis=0).tolist()
    mean_rows = np_array.mean(axis=1).tolist()
    mean_flat = np_array.mean()

    var_columns = np_array.var(axis=0).tolist()
    var_rows = np_array.var(axis=1).tolist()
    var_flat = np_array.var()

    std_columns = np_array.std(axis=0).tolist()
    std_rows = np_array.std(axis=1).tolist()
    std_flat = np_array.std()

    max_columns = np_array.max(axis=0).tolist()
    max_rows = np_array.max(axis=1).tolist()
    max_flat = np_array.max()

    min_columns = np_array.min(axis=0).tolist()
    min_rows = np_array.min(axis=1).tolist()
    min_flat = np_array.min()

    sum_columns = np_array.sum(axis=0).tolist()
    sum_rows = np_array.sum(axis=1).tolist()
    sum_flat = np_array.sum()

    calculations = {
        'mean': [mean_columns, mean_rows, mean_flat],
        'variance': [var_columns, var_rows, var_flat],
        'standard deviation': [std_columns, std_rows, std_flat],
        'max': [max_columns, max_rows, max_flat],
        'min': [min_columns, min_rows, min_flat],
        'sum': [sum_columns, sum_rows, sum_flat]
    }

    return calculations
main.pyThis file serves as an entry point to demonstrate the calculate function and can be used to run your unit tests.import mean_var_std
from unittest import main

# Example usage of your calculate function
try:
    result = mean_var_std.calculate([0,1,2,3,4,5,6,7,8])
    print("Calculations for [0,1,2,3,4,5,6,7,8]:")
    print(result)
except ValueError as e:
    print(f"Error: {e}")

# Run unit tests automatically (if test_module.py exists)
# main(module='test_module', exit=False)
To run the main.py script, navigate to your project directory in the terminal and execute:python main.py
You will see the calculated statistics printed to the console.TestingThe project includes a test_module.py file for unit testing the calculate function.test_module.py (Example)import unittest
import mean_var_std # Import your module

class UnitTests(unittest.TestCase):
    def test_calculate_function_valid_input(self):
        # Test case for valid input
        input_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        expected_mean_flat = 4.0
        result = mean_var_std.calculate(input_list)
        self.assertAlmostEqual(result['mean'][2], expected_mean_flat)
        # Add more specific assertions for rows, columns, and other stats

    def test_calculate_function_invalid_input_length(self):
        # Test case for incorrect input length
        with self.assertRaises(ValueError) as cm:
            mean_var_std.calculate([1, 2, 3])
        self.assertEqual(str(cm.exception), "List must contain nine numbers.")

    def test_calculate_function_another_valid_input(self):
        input_list = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        result = mean_var_std.calculate(input_list)
        self.assertAlmostEqual(result['mean'][2], 5.0)
        self.assertEqual(result['sum'][0], [18, 15, 12]) # Sum of columns [9+6+3, 8+5+2, 7+4+1]
        self.assertEqual(result['sum'][1], [24, 15, 6]) # Sum of rows [9+8+7, 6+5+4, 3+2+1]


if __name__ == '__main__':
  
    unittest.main()

To run the unit tests, ensure the main(module='test_module', exit=False) line in main.py is uncommented, or simply run test_module.py directly:python -m unittest 

test_module.py

Model Saving:

The trained Logistic Regression model was saved using joblib for future use and potential deployment, preventing the need for retraining.
Key Results and Findings

Logistic Regression Performance:

Accuracy: 0.66

Confusion Matrix:
[[72  41]
 [10  27]]

Classification Report (for class 1 - donated):

Precision: 0.39

Recall: 0.73

F1-Score: 0.51

Interpretation: The Logistic Regression model, now using class_weight='balanced', achieved an overall accuracy of 66%. Critically, by accounting for class imbalance, its ability to identify actual donors (class 1) significantly improved:

A Precision of 0.39 means that when the model predicted someone would donate, it was correct about 39% of the time (41 false positives).

A Recall of 0.73 means the model successfully identified 73% of the actual donors in the test set (only 10 actual donors were missed). This is a substantial improvement over the previous recall of 0.16, indicating better identification of the minority class.

The F1-Score of 0.51 reflects a more balanced performance between precision and recall for the positive class compared to before.


In summary, while there's still room for improvement in reducing false positives, this model is much more effective at identifying potential blood donors, which is crucial for blood bank recruitment strategies.
TPOT Best Model Performance:

Accuracy: 0.70

Confusion Matrix:
[[105   8]
[ 30   7]]

Classification Report (for class 1 - donated):

Precision: 0.47

Recall: 0.19

F1-Score: 0.27

Best Pipeline Found by TPOT: Pipeline(steps=[('RobustScaler', RobustScaler()), ('MLPClassifier', MLPClassifier(alpha=0.0001, learning_rate_init=0.01))])
Interpretation: The TPOT classifier achieved an overall accuracy of 70%, which is lower than the Logistic Regression model's 66% accuracy. Looking at the metrics for class 
1 (donors), TPOT's performance, while not completely zero like before, still heavily favors the majority class:

Precision of 0.47 indicates that when TPOT predicted a donation, it was correct about 47% of the time.

Recall of 0.19 shows that the model still only identified 19% of the actual donors in the test set, missing a large majority (30 false negatives).

The F1-Score of 0.27 reflects this imbalance in performance.

TPOT selected a pipeline consisting of RobustScaler for preprocessing and an MLPClassifier (Multi-layer Perceptron, a type of neural network). Despite using f1_weighted scoring, TPOT still struggled to find a pipeline that robustly handles the class imbalance and effectively predicts the minority class. This suggests that further manual intervention or more aggressive imbalance handling techniques might be necessary for TPOT to find a truly optimized solution for this problem.


Most Important Features (from Logistic Regression Coefficients):

Frequency (times): Positive coefficient (0.30), suggesting that individuals who donate more frequently are significantly more likely to donate again. This is the most influential positive predictor.

Recency (months): Negative coefficient (-0.05), indicating that surprisingly, the longer since the last donation (higher recency value), the less likely the individual is to donate in March 2007. More recent donors (lower recency value) are therefore more likely.

Monetary (c.c. blood): Small positive coefficient (0.01), implying a very slight positive effect; greater total blood donated has a minor association with increased likelihood of donating again.

Time (months): Positive coefficient (0.10), suggesting that individuals with a longer history since their first donation are slightly more likely to donate again.
Conclusion and Future Work

Conclusion: Both the Logistic Regression and TPOT models were applied to predict blood donation. Initially, both models struggled significantly with identifying actual donors due to class imbalance. However, by implementing class_weight='balanced' for Logistic Regression, its ability to recall actual donors substantially improved (recall of 0.73), making it a more useful tool for blood banks, despite a lower precision. TPOT, even with f1_weighted scoring, still found a pipeline that largely favors the majority class, indicating the persistent challenge of class imbalance. The analysis of Logistic Regression coefficients highlighted Frequency and Recency as key factors influencing donor behavior. This project demonstrates the pipeline for blood donation prediction and highlights the critical importance of addressing class imbalance in such real-world scenarios to build truly effective models for the minority class.

Future Work:

Deepen Class Imbalance Handling: This is the most critical next step. Experiment further with techniques such as:

Oversampling: More advanced SMOTE variants (e.g., Borderline-SMOTE, SVMSMOTE) using imbalanced-learn.
Ensemble Methods for Imbalanced Data: Explore algorithms like BalancedRandomForestClassifier or EasyEnsembleClassifier from imbalanced-learn.

Explore Other Algorithms: Experiment with other classification algorithms known to perform well on imbalanced data or with different characteristics, such as XGBoost, LightGBM, or Support Vector Machines, with thorough hyperparameter tuning.
Feature Engineering: Create new, more informative features from existing ones (e.g., donor consistency, average donation interval, donation rate).

Advanced Evaluation Metrics: Beyond accuracy, focus on metrics like AUC-ROC, Precision-Recall Curve, and Cohen's Kappa, which are more robust for imbalanced datasets and provide a more complete picture of model performance.

Model Deployment: Develop a simple web application (e.g., using Flask or Streamlit) to deploy the trained model for interactive predictions, allowing blood banks to input donor data and get real-time predictions.

How to Run This Project
To run this project, you can use GitHub Codespaces (recommended) or set it up locally:

Using GitHub Codespaces (Recommended)

Open in Codespaces: Navigate to this repository on GitHub and click the green "Code" button, then select "Open with Codespaces". A cloud-based development environment will launch.

Open main.ipynb: Once the Codespace is ready, open the main.ipynb file in the file explorer.

Install Dependencies: Open the terminal (Terminal > New Terminal) and install any missing libraries (e.g., pip install tpot imbalanced-learn matplotlib seaborn).

Run Cells: Execute each cell in the main.ipynb notebook sequentially.
Running Locally

Clone the Repository:
git clone [https://github.com/Saloni1999-star/blood-donation-prediction.git](https://github.com/Saloni1999-star/blood-donation-prediction.git)
cd blood-donation-prediction
Create a Virtual Environment (Recommended):
python -m venv venv
# On Windows:
# .\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
Install Dependencies:
pip install pandas numpy scikit-learn tpot matplotlib seaborn imbalanced-learn joblib
Run Jupyter Notebook:
jupyter notebook
This will open a browser window. Navigate to main.ipynb and open it.
Run Cells: Execute each cell in the notebook sequentially.
