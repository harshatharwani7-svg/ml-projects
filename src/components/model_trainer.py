import os
import sys
from dataclasses import dataclass
import traceback
import numpy as np

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.datasets import make_regression

# --------------------- CUSTOM EXCEPTION ---------------------
class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_detail = error_detail

    def __str__(self):
        return f"{self.args[0]} \nDetail: {self.error_detail}"


# --------------------- LOGGER ---------------------
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --------------------- UTILITY FUNCTIONS ---------------------
def save_object(file_path, obj):
    """Save model object to a file using pickle"""
    import pickle
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """Evaluate multiple regression models and return R2 scores"""
    report = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        report[name] = score
    return report


# --------------------- CONFIG AND TRAINER ---------------------
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path=None):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define models
            models = {
                "RandomForest": RandomForestRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "LinearRegression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(),
                "KNN": KNeighborsRegressor()
            }

            # Evaluate models
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            # Find best model
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with score {best_model_score}")

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(
                error_message=f"Error in initiate_model_trainer: {e}",
                error_detail=traceback.format_exc()
            )


# --------------------- EXAMPLE USAGE ---------------------
if __name__ == "__main__":
    # Generate synthetic regression data (learnable)
    X_train, y_train = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
    X_test, y_test = make_regression(n_samples=20, n_features=4, noise=0.1, random_state=43)

    # Combine features + target for trainer
    train_arr = np.c_[X_train, y_train]
    test_arr = np.c_[X_test, y_test]

    # Train models
    model_trainer = ModelTrainer()
    best_model_name, best_model_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Best Model: {best_model_name}, Score: {best_model_score}")
