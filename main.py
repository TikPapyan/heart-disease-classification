import os
import matplotlib.pyplot as plt
import numpy as np
from data_wrangling import load_data, preprocess_data
from model_training import train_random_forest, train_svm, train_logistic_regression
from visualization import categorical_visualizations, continuous_visualizations, plot_classification_report
from logistic_regression import logistic_regression, lr_classification_report


def create_metrics_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def describe_data(data):
    print("(Rows, columns): " + str(data.shape))
    data.columns
    print(data.nunique(axis=0))
    print(data.describe())
    print(print(data.isna().sum()))
    print(data['HeartDisease'].value_counts())


def main():
    metrics_dir = "metrics_dir"
    file_path = 'data/heart_initial.csv'
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)

    create_metrics_folder(metrics_dir)    

    metrics_scratch, y_pred_scratch, weights = logistic_regression(X_train, y_train, X_test, y_test)
    lr_classification_report(metrics_scratch)

    best_model_lr, report_lr = train_logistic_regression(X_train, y_train, X_test, y_test)
    print(f"Logistic Regression SciKit Results: \n \n {report_lr}")
    plot_classification_report(report_lr)
    plt.show()
    plt.savefig(f'{metrics_dir}/test_plot_plot_lr.png', dpi=200, format='png', bbox_inches='tight')
    plt.close()

    best_model_svm, report_svm = train_svm(X_train, y_train, X_test, y_test)
    print(f"SVM Results: \n \n {report_svm}")
    plot_classification_report(report_svm)
    plt.show()
    plt.savefig(f'{metrics_dir}/test_plot_plot_svm.png', dpi=200, format='png', bbox_inches='tight')
    plt.close()

    best_model_rf, report_rf = train_random_forest(X_train, y_train, X_test, y_test)
    print(f"Random Forest Results: \n \n {report_rf}")
    plot_classification_report(report_rf)
    plt.show()
    plt.savefig(f'{metrics_dir}/test_plot_plot_rf.png', dpi=200, format='png', bbox_inches='tight')
    plt.close()

    # Categorical visualizations to be fixed
    all_categorical_columns = ["Sex_F", "Sex_M", "ChestPainType_ASY", "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA", "RestingECG_LVH", "RestingECG_Normal", "RestingECG_ST", "ExerciseAngina_N", "ExerciseAngina_Y", "ST_Slope_Down", "ST_Slope_Flat", "ST_Slope_Up", "FastingBS", "HeartDisease"]
    categorical_columns = ["Sex_F", "Sex_M"]
    for column in categorical_columns:
        categorical_visualizations(X_test.assign(HeartDisease=y_test), column)
        plt.show()
        plt.savefig(f'{metrics_dir}/{column}.png', dpi=200, format='png', bbox_inches='tight')
        plt.close()

    all_continuous_columns = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    continuous_columns = ["Cholesterol"]
    for column in continuous_columns:
        continuous_visualizations(X_test.assign(HeartDisease=y_test), column)
        plt.show()
        plt.savefig(f'{metrics_dir}/{column}.png', dpi=200, format='png', bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()