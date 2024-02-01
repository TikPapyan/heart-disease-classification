import wandb

from joblib import dump

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def train_logistic_regression(X_train, y_train, X_test, y_test, models_dir, use_wandb=False):
    model_lr = LogisticRegression()
    model_lr.fit(X_train, y_train)

    param_grid_lr = {
        'penalty': ['l2'],
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [100, 200, 300]
    }

    grid_search_lr = GridSearchCV(
        estimator=model_lr,
        param_grid=param_grid_lr,
        scoring='accuracy',
        cv=3
    )
    
    grid_search_lr.fit(X_train, y_train)

    best_model_lr = grid_search_lr.best_estimator_
    y_pred_lr = best_model_lr.predict(X_test)
    report_lr = classification_report(y_test, y_pred_lr)

    best_model_path = f"{models_dir}/best_model_lr.joblib"
    dump(grid_search_lr.best_estimator_, best_model_path)

    if use_wandb:
        artifact = wandb.Artifact('best_model_lr', type='model')
        artifact.add_file(best_model_path)
        wandb.log_artifact(artifact)

    return report_lr


def train_svm(X_train, y_train, X_test, y_test, models_dir, use_wandb=False):
    model_svm = SVC()
    model_svm.fit(X_train, y_train)
    
    param_grid_svm = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3, 4]
    }

    grid_search_svm = GridSearchCV(
        estimator=model_svm,
        param_grid=param_grid_svm,
        scoring='accuracy',
        cv=3
    )

    grid_search_svm.fit(X_train, y_train)

    best_model_svm = grid_search_svm.best_estimator_
    y_pred_svm = best_model_svm.predict(X_test)
    report_svm = classification_report(y_test, y_pred_svm)

    best_model_path = f"{models_dir}/best_model_svm.joblib"
    dump(grid_search_svm.best_estimator_, best_model_path)

    if use_wandb:
        artifact = wandb.Artifact('best_model_svm', type='model')
        artifact.add_file(best_model_path)
        wandb.log_artifact(artifact)

    return report_svm


def train_random_forest(X_train, y_train, X_test, y_test, models_dir, use_wandb=False):
    model_rf = RandomForestClassifier(random_state=0)
    model_rf.fit(X_train, y_train)
    
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search_rf = GridSearchCV(
        estimator=model_rf,
        param_grid=param_grid_rf,
        scoring='accuracy',
        cv=3
    )

    grid_search_rf.fit(X_train, y_train)

    best_model_rf = grid_search_rf.best_estimator_
    y_pred_rf = best_model_rf.predict(X_test)
    report_rf = classification_report(y_test, y_pred_rf)
    
    best_model_path = f"{models_dir}/best_model_rf.joblib"
    dump(grid_search_rf.best_estimator_, best_model_path)

    if use_wandb:
        artifact = wandb.Artifact('best_model_rf', type='model')
        artifact.add_file(best_model_path)
        wandb.log_artifact(artifact)

    return report_rf

