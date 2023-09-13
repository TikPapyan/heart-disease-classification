# Heart Disease Classification
This project involves the classification of heart disease using a machine learning approach.

## Project Description
This project involves the classification of heart disease using a machine learning approach. The dataset was preprocessed and various classification algorithms were applied, including Support Vector Machine (SVM), Logistic Regression (both with sklearn and implemented from scratch), and Random Forest. The project also includes data visualizations and metrics to evaluate the performance of these models.

## Dataset
The dataset used in this project contains information related to heart disease diagnosis. The preprocessing of the dataset involved one-hot encoding categorical features, scaling numerical features, and splitting the data into training and testing sets. The preprocessed dataset is saved as 'data/heart_preprocessed.csv'.

## Preprocessing
The data preprocessing steps are as follows:

 - One-hot encoding of categorical columns ('Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope').
 - Scaling numerical columns ('Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak') using Min-Max scaling.
 - Combining one-hot encoded categorical and scaled numerical features.
 - Splitting the dataset into training and testing sets (75% train, 25% test).

## Logistic Regression
 - Implemented from scratch with custom code.
 - Utilized the Logistic Regression model from scikit-learn for comparison.
 - Classification report and visualizations generated for evaluation.

## Support Vector Machine (SVM)
 - SVM model trained using scikit-learn.
 - Classification report and visualizations generated for evaluation.

## Random Forest
 - Random Forest model trained using scikit-learn.
 - Classification report and visualizations generated for evaluation.

## Metrics and Visualizations
Classification reports are generated for each model, providing metrics such as accuracy, precision, recall, and F1-score.
Visualizations include categorical and continuous feature plots, confusion matrices, and ROC curves.

## Usage
Clone the project repository to your local machine.
Ensure you have the required libraries installed (numpy, pandas, scikit-learn, matplotlib).
Run the **main()** function in the **main.py** file to execute the entire project.
Metrics and visualizations will be saved in a folder named 'metrics_dir'.

## Project Structure
**data/:** Contains the raw and preprocessed datasets.
**model_training.py:** Contains functions for training SVM, Logistic Regression, and Random Forest models.
**logistic_regression.py:** Contains code for Logistic Regression implemented from scratch.
**data_wrangling.py:** Functions for loading and preprocessing data.
**visualization.py:** Functions for generating visualizations.
**main.py:** Main script to execute the project.

## Results
The project results include classification reports and visualizations for each model, which can be found in the 'metrics_dir' folder.
