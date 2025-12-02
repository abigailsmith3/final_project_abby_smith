import os
import pandas as pd
from .load_data import load_dataset


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by dropping rows with missing values."""
    tenureMed = df['Tenure'].median()
    df['Support Calls'] = df['Support Calls'].replace("none",0)
    df['Tenure'] = df['Tenure'].replace(np.nan,tenureMed)
    
    #change this so that the median of support calls is the same as the first one
    df['Support Calls'] = df['Support Calls'].fillna(int(df['Support Calls'].median()))
    df['Last Interaction'] = df['Last Interaction'].fillna(df['Last Interaction'].median())
    clean_df = df.fillna(0)

    
    return clean_df


if __name__ == "__main__":
    #cleaning data

    # Load the raw dataset
    rawData = load_dataset("data/raw/data.csv")
    # Clean the dataset
    cleanedData = clean_dataset(rawData)
    # Ensure the processed directory exists
    os.makedirs("data/processed", exist_ok=True)
    # Save the cleaned data
    processed_path_data = "data/processed/clean_data.csv"
    cleanedData.to_csv(processed_path_data, index=False)

    #cleaning test
    
    # Load the raw dataset
    rawTest = load_dataset("data/raw/test.csv")
    # Clean the dataset
    cleanedTest = clean_dataset(rawTest)
    # Ensure the processed directory exists
    os.makedirs("data/processed", exist_ok=True)
    # Save the cleaned data
    processed_path_test = "data/processed/clean_test.csv"
    cleanedTest.to_csv(processed_path_test, index=False)
    
    print(f"Cleaned data saved to {processed_path_data}")
    print(f"Cleaned test data saved to {processed_path_test}")

