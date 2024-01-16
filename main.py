import argparse
import os

import matplotlib.pyplot as plt

from data_wrangling import load_data, preprocess_data
from model_training import train_random_forest, train_svm, train_logistic_regression
from visualization import categorical_visualizations, continuous_visualizations, plot_classification_report
from logistic_regression import logistic_regression, lr_classification_report


def pars_arguments():
    parser = argparse.ArgumentParser(description="Run Heart Disease Classification Models")
    parser.add_argument('--logistic_regression', action='store_true', help='Run custom logistic regression')
    parser.add_argument('--sklearn_logistic_regression', action='store_true', help='Run scikit-learn logistic regression')
    parser.add_argument('--svm', action='store_true', help='Run SVM model')
    parser.add_argument('--random_forest', action='store_true', help='Run Random Forest model')
    parser.add_argument('--categorical_visualization', action='store_true', help='Run categorical visualizations')
    parser.add_argument('--continuous_visualization', action='store_true', help='Run continuous visualizations')

    parser.add_argument('--categorical_columns', nargs='+', help='Specify categorical columns for visualization')
    parser.add_argument('--continuous_columns', nargs='+', help='Specify continuous columns for visualization')

    return parser.parse_args()


def create_metrics_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def main(args):
    metrics_dir = "metrics"
    file_path = 'data/heart_initial.csv'
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)

    create_metrics_folder(metrics_dir)

    if args.logistic_regression:
        metrics_scratch, y_pred_scratch, weights = logistic_regression(X_train, y_train, X_test, y_test)
        lr_classification_report(metrics_scratch)

    if args.sklearn_logistic_regression:
        best_model_lr, report_lr = train_logistic_regression(X_train, y_train, X_test, y_test)
        print(f"Logistic Regression SciKit Results: \n \n {report_lr}")
        plot_classification_report(report_lr)
        plt.savefig(f'{metrics_dir}/logistic_regression_plot.png', dpi=200, format='png', bbox_inches='tight')
        plt.close()

    if args.svm:
        best_model_svm, report_svm = train_svm(X_train, y_train, X_test, y_test)
        print(f"SVM Results: \n \n {report_svm}")
        plot_classification_report(report_svm)
        plt.savefig(f'{metrics_dir}/svm_plot.png', dpi=200, format='png', bbox_inches='tight')
        plt.close()

    if args.random_forest:
        best_model_rf, report_rf = train_random_forest(X_train, y_train, X_test, y_test)
        print(f"Random Forest Results: \n \n {report_rf}")
        plot_classification_report(report_rf)
        plt.savefig(f'{metrics_dir}/random_forest_plot.png', dpi=200, format='png', bbox_inches='tight')
        plt.close()
    
    if args.categorical_visualization:
        categorical_columns = args.categorical_columns if args.categorical_columns else ['Sex_F', 'Sex_M']  # Example default columns
        for column in categorical_columns:
            categorical_visualizations(X_test.assign(HeartDisease=y_test), column)
            plt.savefig(f'{metrics_dir}/categorical_{column}.png', dpi=200, format='png', bbox_inches='tight')
            plt.close()

    if args.continuous_visualization:
        continuous_columns = args.continuous_columns if args.continuous_columns else ['Cholesterol']  # Example default column
        for column in continuous_columns:
            continuous_visualizations(X_test.assign(HeartDisease=y_test), column)
            plt.savefig(f'{metrics_dir}/continuous_{column}.png', dpi=200, format='png', bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    main(pars_arguments())
