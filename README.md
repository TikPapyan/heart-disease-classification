# Heart Disease Classification
This project focuses on the classification of heart disease using various machine learning algorithms. It demonstrates preprocessing of a heart disease dataset, model training, evaluation, visualization of results, and integrates Weights & Biases (Wandb) for experiment tracking and visualization.

## Project Description
The heart disease classification project employs machine learning techniques to predict the presence of heart disease based on various health indicators. The project uses algorithms like Support Vector Machine (SVM), Logistic Regression, and Random Forest. It features both a custom implementation of Logistic Regression and scikit-learn's implementation for comparison. Comprehensive data visualizations, evaluation metrics, and Wandb integration are key components of this project.

## Dataset
The dataset includes various heart health indicators and a target variable indicating the presence of heart disease. Key preprocessing steps include:
 - One-hot encoding for categorical variables.
 - Min-Max scaling for numerical variables.
 - Splitting into training and testing sets (75% train, 25% test).

## Models
The project includes the following models:
 - **Logistic Regression**: Custom implementation and scikit-learn's implementation.
 - **Support Vector Machine (SVM)**: Implemented using scikit-learn.
 - **Random Forest**: Implemented using scikit-learn.

## Metrics and Visualizations
Each model is evaluated using classification reports (accuracy, precision, recall, F1-score) and visualizations (feature plots, confusion matrices, ROC curves). Wandb is used for tracking experiments, logging metrics, and visualizing results.

## New Features
 - **Weights & Biases (Wandb) Integration**: For tracking experiments, logging metrics, and visualizing results.
 - **Command-Line Interface**: The project now supports command-line arguments for flexible execution of different parts, including model training and visualizations.
 - **Visualizations for Categorical and Continuous Variables**: Enhanced with options to visualize specific variables.

## Usage
1. Clone the repository.
2. Install required libraries.
3. Run `main.py` with desired flags and enable Wandb logging:

```
python main.py --svm --use-wandb
python main.py --logistic_regression --use-wandb
python main.py --random_forest --use-wandb
python main.py --categorical_visualization --categorical_columns <column_names> --use-wandb
python main.py --continuous_visualization --continuous_columns <column_names> --use-wandb   
```


4. View generated metrics and visualizations in the 'metrics' folder and on the Wandb dashboard.

## Project Structure
 - data/: Raw and preprocessed datasets.
 - model_training.py: Training functions for SVM, Logistic Regression, and Random Forest.
 - logistic_regression.py: Custom Logistic Regression implementation.
 - data_wrangling.py: Data loading and preprocessing functions.
 - visualization.py: Visualization functions with Wandb integration.
 - main.py: Main script with command-line interface.

## Results
Results, including classification reports and visualizations for each model, are stored in the 'metrics' folder and can also be viewed on the Wandb dashboard. Detailed results are dependent on the specific models and visualizations executed.
