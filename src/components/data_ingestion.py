# src/components/data_ingestion.py

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
    """
    Resolve the project root.
    Assumes this file is at: <root>/src/components/data_ingestion.py
    """
    return Path(__file__).resolve().parents[2]


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    source_csv_path: str = os.path.join("notebook", "stud.csv")
    test_size: float = 0.2
    random_state: int = 42
    shuffle: bool = True


class DataIngestion:
    def __init__(self, config: DataIngestionConfig | None = None, override_source: str | None = None):
        self.config = config or DataIngestionConfig()
        self.root = _project_root()
        self.override_source = override_source

    def _abs(self, path: str | Path) -> Path:
        return self.root / Path(path)

    def _resolve_source_csv(self) -> Path:
        if self.override_source:
            p = Path(self.override_source)
            if not p.is_absolute():
                p = self.root / p
            if p.exists():
                return p

        candidates = [
            self._abs(self.config.source_csv_path),
            self._abs("data/stud.csv"),
            self._abs("dataset/stud.csv"),
            self._abs("artifacts/data.csv"),
        ]

        for c in candidates:
            if c.exists():
                return c

        raise FileNotFoundError("Source CSV not found. Provide --csv or place stud.csv correctly.")

    def run(self):
        try:
            raw_path = self._abs(self.config.raw_data_path)
            train_path = self._abs(self.config.train_data_path)
            test_path = self._abs(self.config.test_data_path)

            raw_path.parent.mkdir(parents=True, exist_ok=True)

            source_csv = self._resolve_source_csv()
            logging.info(f"Reading data from {source_csv}")

            df = pd.read_csv(source_csv)
            df.to_csv(raw_path, index=False)

            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                shuffle=self.config.shuffle,
            )

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            logging.info("Data ingestion completed successfully")
            return train_path, test_path

        except Exception as e:
            raise CustomException(e, sys)


def initiate_data_ingestion(override_source: str | None = None):
    ingestion = DataIngestion(override_source=override_source)
    return ingestion.run()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="Path to source CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_path, test_path = initiate_data_ingestion(args.csv)
    print("Train CSV:", train_path)
    print("Test CSV :", test_path)
