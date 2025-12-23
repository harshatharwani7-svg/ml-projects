
# src/utils.py
import os
import sys
import dill
from src.exception import CustomException
from src.logger import logging

def save_object(file_path: str, obj) -> None:
    """
    Serialize and save a Python object to the given file path using dill.
    Creates the parent directory if it does not exist.
    """
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object saved to: {file_path}")

    except Exception as e:
        # Log the stack trace and re-raise as your custom exception
        logging.exception(f"Failed to save object to {file_path}")
        raise CustomException(e, sys)
