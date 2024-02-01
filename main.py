import wandb
import os
import argparse

import matplotlib.pyplot as plt

from data_wrangling import load_data, preprocess_data
from model_training import train_random_forest, train_svm, train_logistic_regression
from visualization import categorical_visualizations, continuous_visualizations, plot_classification_report
from logistic_regression import logistic_regression, lr_classification_report


def pars_arguments():
    parser = argparse.ArgumentParser(description="Run Heart Disease Classification Models")
    parser.add_argument('--lr', action='store_true', help='Run custom logistic regression')
    parser.add_argument('--sklearn_lr', action='store_true', help='Run scikit-learn logistic regression')
    parser.add_argument('--svm', action='store_true', help='Run SVM model')
    parser.add_argument('--rf', action='store_true', help='Run Random Forest model')
    parser.add_argument('--categorical_visualization', action='store_true', help='Run categorical visualizations')
    parser.add_argument('--continuous_visualization', action='store_true', help='Run continuous visualizations')

    parser.add_argument('--categorical_columns', nargs='+', help='Specify categorical columns for visualization')
    parser.add_argument('--continuous_columns', nargs='+', help='Specify continuous columns for visualization')
    
    parser.add_argument('--use_wandb', action='store_true', help='Enable Wandb logging.')

    return parser.parse_args()


def create_metrics_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_models_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def main(args):
    metrics_dir = "metrics"
    models_dir = 'models'
    file_path = 'data/heart_initial.csv'
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)

    create_metrics_folder(metrics_dir)
    create_models_folder(models_dir)

    if args.use_wandb:
        print("Running with wandb configuration")
        wandb.login()
        wandb.init(project='Heart Disease Classification', entity='tigran-papyan',
                   config={"dataset_name": "heart disease initial CSV"})

    if args.lr:
        metrics_scratch, weights = logistic_regression(X_train, y_train, X_test, y_test, args.use_wandb)
        lr_classification_report(metrics_scratch)

        if args.use_wandb:
            wandb.log({"weights": weights.tolist()})
            wandb.log(metrics_scratch)

    if args.sklearn_lr:
        report_lr = train_logistic_regression(X_train, y_train, X_test, y_test, models_dir, args.use_wandb)
        print(f"Logistic Regression SciKit Results: \n \n {report_lr}")

        plot_classification_report(report_lr)
        plot_path = f'{metrics_dir}/lr_plot.png'
        plt.savefig(plot_path, dpi=200, format='png', bbox_inches='tight')
        plt.close()

        if args.use_wandb:
            wandb.log({"lr_classification_report_plot": wandb.Image(plot_path)})
            
    if args.svm:
        report_svm = train_svm(X_train, y_train, X_test, y_test, models_dir, args.use_wandb)
        print(f"SVM Results: \n\n {report_svm}")

        plot_classification_report(report_svm)
        plot_path = f'{metrics_dir}/svm_plot.png'
        plt.savefig(plot_path, dpi=200, format='png', bbox_inches='tight')
        plt.close()

        if args.use_wandb:
            wandb.log({"svm_classification_report_plot": wandb.Image(plot_path)})

    if args.rf:
        report_rf = train_random_forest(X_train, y_train, X_test, y_test, models_dir, args.use_wandb)
        print(f"Random Forest Results: \n \n {report_rf}")

        plot_classification_report(report_rf)
        plot_path = f'{metrics_dir}/rf_plot.png'
        plt.savefig(plot_path, dpi=200, format='png', bbox_inches='tight')
        plt.close()

        if args.use_wandb:
            wandb.log({"rf_classification_report_plot": wandb.Image(plot_path)})
    
    if args.continuous_visualization:
        continuous_columns = ['Cholesterol']
        for column in continuous_columns:
            continuous_visualizations(X_test.assign(HeartDisease=y_test), column, args.use_wandb, metrics_dir)

    if args.categorical_visualization:
        categorical_columns = ['Sex_F', 'Sex_M']
        for column in categorical_columns:
            categorical_visualizations(X_test.assign(HeartDisease=y_test), column, args.use_wandb, metrics_dir)

    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main(pars_arguments())
