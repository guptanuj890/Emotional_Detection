import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import subprocess
import zipfile
import yaml

# Logging setup
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "data_ingestion.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Download dataset using Kaggle CLI
def download_kaggle_dataset(dataset_name: str, output_dir: str) -> str:
    """Download Kaggle dataset using the Kaggle CLI and extract it."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        subprocess.run([
            "kaggle", "datasets", "download", "-d", dataset_name, "-p", output_dir
        ], check=True)

        zip_files = [f for f in os.listdir(output_dir) if f.endswith('.zip')]
        if not zip_files:
            raise FileNotFoundError("No zip file found after download.")

        zip_path = os.path.join(output_dir, zip_files[0])
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        logger.debug("Downloaded and extracted dataset: %s", dataset_name)
        return output_dir
    except Exception as e:
        logger.error("Error downloading dataset from Kaggle: %s", e)
        raise

#Load Parameters
def load_params(params_path: str) -> dict:
    """Load parameters from YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

# Load CSV data
def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise

# Preprocess data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        logger.debug("Data preprocessing done.")
        return df
    except KeyError as e:
        logger.error("Missing column in the DataFrame: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during preprocessing: %s", e)
        raise

# Save processed data
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug("Train and test data saved to %s", raw_data_path)
    except Exception as e:
        logger.error("Unexpected error occurred while saving the data: %s", e)
        raise

# Main function
def main():
    try:
        params = load_params(params_path = 'params.yaml')
        test_size = params['data_ingestion']['test_size']
        
        data_dir = download_kaggle_dataset('nelgiriyewithana/emotions', './data')
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV file found after extracting dataset.")

        data_path = os.path.join(data_dir, csv_files[0])
        df = load_data(data_url=data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=111)
        save_data(train_data, test_data, data_path="./data")
    except Exception as e:
        logger.error("Failed to complete the data ingestion process: %s", e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
