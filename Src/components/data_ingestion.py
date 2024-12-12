import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger.logger import logging
from src.config import DATASET_PATH, ARTIFACTS_DIR


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion paths."""

    train_data_path: str = os.path.join(ARTIFACTS_DIR, "train.csv")
    test_data_path: str = os.path.join(ARTIFACTS_DIR, "test.csv")
    raw_data_path: str = os.path.join(ARTIFACTS_DIR, "data.csv")


class DataIngestion:
    """Handles data ingestion: reading data, splitting, and saving train/test sets."""

    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    def initiate_data_ingestion(self) -> tuple:
        """Reads raw data, splits it into train/test, and saves to CSV."""
        logging.info("Starting data ingestion process")
        try:
            os.makedirs(ARTIFACTS_DIR, exist_ok=True)
            if not DATASET_PATH.exists():
                raise FileNotFoundError(f"Dataset not found at path: {DATASET_PATH}")

            df = pd.read_csv(DATASET_PATH)
            df.to_csv(self.config.raw_data_path, index=False)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.config.train_data_path, index=False)
            test_set.to_csv(self.config.test_data_path, index=False)
            logging.info("Data ingestion completed and files saved.")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


# Main execution
if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()
