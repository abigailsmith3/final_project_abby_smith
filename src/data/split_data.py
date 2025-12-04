import os
import pandas as pd
from sklearn.model_selection import train_test_split
from .load_data import load_dataset


def split_dataset(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.3,
    seed: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the DataFrame into train, validation and test sets."""
    if not abs(train_frac + val_frac - 1.0) < 1e-8:
        raise ValueError("Fractions must sum to 1.0")
        
#should I do x and y or just x like this?
    train_df, val_df = train_test_split(
        df, test_size=val_frac, random_state=seed
    )


    return train_df, val_df


if __name__ == "__main__":
    cleaned_path = "data/processed/clean_data.csv"
    df = load_dataset(cleaned_path)

    train, val = split_dataset(df)
#ask about my df_encoded stuff, how would I do this?
    
    os.makedirs("data/processed", exist_ok=True)
    train.to_csv("data/processed/card_transdata_train.csv", index=False)
    val.to_csv("data/processed/card_transdata_validation.csv", index=False)
    

    print("Train and validation have been saved to data/processed/")
