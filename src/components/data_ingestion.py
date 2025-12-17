
import os
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# If you run with: python -m src.components.data_ingestion
from src.exception import CustomException
from src.logger import logging

# If you prefer relative imports instead, comment the two lines above and use:
# from ..exception import CustomException
# from ..logger import logging


# -------- Path helpers --------
def _project_root() -> Path:
    """
    Resolve the project root based on this file location.
    Assuming this file is at: <root>/src/components/data_ingestion.py
    parents[2] -> <root>
    """
    return Path(__file__).resolve().parents[2]


# -------- Config --------
@dataclass
class DataIngestionConfig:
    """
    Output paths are RELATIVE to the project root.
    Input (source) can be set via CLI (--csv) or by placing the file
    in one of the candidate locations.
    """
    # outputs
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str  = os.path.join('artifacts', 'test.csv')
    raw_data_path: str   = os.path.join('artifacts', 'data.csv')

    # default expected input location (checked if --csv is not provided)
    source_csv_path: str = os.path.join('notebook', 'stud.csv')

    # split params
    test_size: float   = 0.2
    random_state: int  = 42
    shuffle: bool      = True


# -------- Ingestion --------
class DataIngestion:
    def __init__(self, config: DataIngestionConfig | None = None, override_source: str | None = None):
        self.cfg = config or DataIngestionConfig()
        self.root = _project_root()
        self.override_source = override_source  # absolute path or relative to project root

    def _abs(self, rel_path: str | Path) -> Path:
        """Make an absolute path under the project root."""
        return self.root / Path(rel_path)

    def _resolve_source_csv(self) -> Path:
        """
        Resolve the source CSV by checking (in order):
        1) CLI override (--csv) (absolute or relative-to-root)
        2) <root>/notebook/stud.csv
        3) <root>/data/stud.csv
        4) <root>/dataset/stud.csv
        5) <root>/src/components/stud.csv
        6) <root>/artifacts/data.csv   (in case a raw copy already exists)
        """

        # 1) CLI override
        if self.override_source:
            p = Path(self.override_source)
            if not p.is_absolute():
                p = self.root / p
            if p.exists():
                logging.info(f"Using source CSV from CLI: {p}")
                return p
            else:
                logging.warning(f"--csv provided but file not found: {p}")

        # 2..6) Candidate locations
        candidates = [
            self._abs(self.cfg.source_csv_path),                 # <root>/notebook/stud.csv
            self._abs(Path("data") / "stud.csv"),                # <root>/data/stud.csv
            self._abs(Path("dataset") / "stud.csv"),             # <root>/dataset/stud.csv
            self._abs(Path("src") / "components" / "stud.csv"),  # <root>/src/components/stud.csv
            self._abs(self.cfg.raw_data_path),                   # <root>/artifacts/data.csv
        ]

        for c in candidates:
            if c.exists():
                logging.info(f"Found source CSV at: {c}")
                return c

        # None found: build helpful message
        msg = (
            "Source CSV not found in any of the expected locations.\n"
            "Please provide the CSV path via CLI or place it at one of the paths below.\n\n"
            "Expected locations:\n"
            f"  1) {self.root / self.cfg.source_csv_path}\n"
            f"  2) {self.root / 'data' / 'stud.csv'}\n"
            f"  3) {self.root / 'dataset' / 'stud.csv'}\n"
            f"  4) {self.root / 'src' / 'components' / 'stud.csv'}\n"
            f"  5) {self.root / self.cfg.raw_data_path}  (if a raw copy already exists)\n\n"
            "CLI usage:\n"
            r'  python -m src.components.data_ingestion --csv "C:\path\to\stud.csv"' "\n\n"
            "Diagnostics:\n"
            f"  CWD: {Path.cwd()}\n"
            f"  Script: {Path(__file__).resolve()}\n"
            f"  Project root: {self.root}\n"
        )
        raise FileNotFoundError(msg)

    def run(self) -> tuple[Path, Path]:
        """
        Reads the source CSV, saves a raw copy, splits into train/test, and writes them to artifacts.
        Returns:
            (train_data_path, test_data_path) as absolute Paths.
        """
        try:
            # Resolve outputs
            raw_csv  = self._abs(self.cfg.raw_data_path)
            train_csv = self._abs(self.cfg.train_data_path)
            test_csv  = self._abs(self.cfg.test_data_path)

            # Ensure artifacts directory exists
            artifacts_dir = raw_csv.parent
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Ensured artifacts directory: {artifacts_dir}")

            # Resolve source
            source_csv = self._resolve_source_csv()

            logging.info(f"Reading source CSV: {source_csv}")
            df = pd.read_csv(source_csv)

            # Save raw copy (overwrite to keep latest snapshot)
            logging.info(f"Saving raw data to: {raw_csv}")
            df.to_csv(raw_csv, index=False)

            # Train/test split
            logging.info(
                f"Splitting data (test_size={self.cfg.test_size}, "
                f"random_state={self.cfg.random_state}, shuffle={self.cfg.shuffle})"
            )
            train_df, test_df = train_test_split(
                df,
                test_size=self.cfg.test_size,
                random_state=self.cfg.random_state,
                shuffle=self.cfg.shuffle,
            )

            # Write out train/test
            logging.info(f"Writing train CSV: {train_csv}")
            train_df.to_csv(train_csv, index=False)

            logging.info(f"Writing test CSV: {test_csv}")
            test_df.to_csv(test_csv, index=False)

            logging.info("Data ingestion completed successfully.")
            return train_csv, test_csv

        except Exception as e:
            logging.exception("Data ingestion failed.")
            raise CustomException(e, sys) from e


# -------- Optional convenience function --------
def initiate_data_ingestion(override_source: str | None = None) -> tuple[Path, Path]:
    ingestion = DataIngestion(override_source=override_source)
    return ingestion.run()


# -------- Main entrypoint --------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data ingestion: read CSV, snapshot to artifacts, split to train/test."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to the source CSV (absolute or relative to project root). "
             "If not provided, script will try common locations under the project root."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        logging.info("Starting data ingestion (module entrypoint).")
        train_path, test_path = initiate_data_ingestion(override_source=args.csv)
        print(f"Train CSV: {train_path}")
        print(f"Test  CSV: {test_path}")
    except CustomException as ce:
        print(f"[DataIngestion] Failed: {ce}")
        sys.exit(1)

