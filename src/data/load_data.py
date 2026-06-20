import pandas as pd
import os


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    if not file_path.endswith('.csv'):
        raise ValueError(f"Expected a .csv file, got: {file_path}")
    
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"An error occurred while loading the data: {e}")
    
    if data.empty:
        raise ValueError("The loaded CSV file is empty.")
    
    print(f"Data loaded successfully from '{file_path}'")
    print(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")
    
    return data


