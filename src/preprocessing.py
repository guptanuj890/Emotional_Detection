import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
import re
stopwords = stopwords.words('english')

log_dir = 'logs'
os.makedirs(log_dir, exist_ok = True)

logger = logging.getLogger('preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, removing special characters,
    tokenizing, removing stopwords and punctuation, and applying stemming.
    """

    ps = PorterStemmer()

    # Convert to lowercase
    text = text.lower()

    # Remove special characters using regex (keep alphanumerics and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Tokenize
    words = nltk.word_tokenize(text)

    # Remove stopwords and apply stemming
    filtered_words = [
        ps.stem(word) for word in words
        if word not in stopwords
    ]

    return " ".join(filtered_words)

def preprocess_df(df, text_column = 'text', target_column = 'target'):
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
    """
    try:
        logger.debug('Starting preprocessing for DataFrame')
        #Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')
        
        #remove duplicates rows
        df = df.drop_duplicates()
        logger.debug('Duplicates removed')
        
        #Apply text transformation to the specified text column
        df.loc[:, text_column] = df[text_column].apply (transform_text)
        logger.debug('Text column transformed')
        return df
    
    except KeyError as e:
        logger.error('column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise
    
def main(text_column = 'text', target_column = 'label'):
    """
        Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        #Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug("Data load properly.")
        
        #Transform the data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)
        
        #Store the data inside data/interim
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok = True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_process.csv"), index = False)
        test_processed_data.to_csv(os.path.join(data_path, "test_process.csv"), index = False)
        
        logger.debug('Processed data saved to %s', data_path)
        
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
        raise
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")
        
if __name__ == '__main__':
    main()