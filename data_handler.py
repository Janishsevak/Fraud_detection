import os
import pandas as pd
from logger import get_logger

logger = get_logger(__name__)

class DataHandler:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.df = None

    def load_data(self):
        """
        Loads and combines all .pkl files from the specified folder.
        """
        combined_data = []
        try:
            # Check if folder exists
            if not os.path.exists(self.folder_path):
                logger.error(f"Folder not found: {self.folder_path}")
                raise FileNotFoundError(f"Folder not found: {self.folder_path}")

            # Iterate over .pkl files in the folder
            for file_name in os.listdir(self.folder_path):
                file_path = os.path.join(self.folder_path, file_name)
                if file_name.endswith('.pkl'):
                    try:
                        logger.info(f"Loading file: {file_path}")
                        data = pd.read_pickle(file_path)
                        combined_data.append(data)
                    except Exception as e:
                        logger.error(f"Error loading file {file_path}: {e}")
                        continue

            # Combine data if any files were successfully loaded
            if combined_data:
                self.df = pd.concat(combined_data, ignore_index=True)
                self.df['TX_DATETIME'] = pd.to_datetime(self.df['TX_DATETIME'])
                logger.info("All data successfully loaded and combined.")
            else:
                logger.warning("No valid .pkl files found to load.")
        except Exception as e:
            logger.critical(f"Unexpected error during data loading: {e}")
        return self.df
