import os
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    test_size: float = 0.2
    random_state: int = 42


class DataIngestion:
    def __init__(self, override_source=None):
        self.config = DataIngestionConfig()
        self.root = _project_root()
        self.override_source = override_source

    def run(self):
        try:
            source_csv = Path(self.override_source)
            if not source_csv.is_absolute():
                source_csv = self.root / source_csv

            df = pd.read_csv(source_csv)

            artifacts_dir = self.root / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)

            df.to_csv(artifacts_dir / "data.csv", index=False)

            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )

            train_df.to_csv(artifacts_dir / "train.csv", index=False)
            test_df.to_csv(artifacts_dir / "test.csv", index=False)

            logging.info("Data ingestion completed")

            return (
                artifacts_dir / "train.csv",
                artifacts_dir / "test.csv"
            )

        except Exception as e:
            raise CustomException(e, sys)


def initiate_data_ingestion(csv_path):
    return DataIngestion(csv_path).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    train_path, test_path = initiate_data_ingestion(args.csv)
    print(train_path, test_path)
