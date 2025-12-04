import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_eda(df: pd.DataFrame) -> None:
    """Create exploratory plots including bivariate views colored by fraud."""
    churned = clean_data[clean_data["Churn"]==1]
    unchurned = clean_data[clean_data["Churn"]==0]

    #churned vs unchurned counts
    sns.countplot(data=data,x="Churn")
    plt.title("churned vs unchruned counts")
    plt.show()
    
    #churned and unchurned last interaction
    
    sns.histplot(data=unchurned,x='Last Interaction')
    plt.title("unchurned last interaction")
    plt.show()
    
    sns.histplot(data=churned,x='Last Interaction')
    plt.title("churned last interaction")
    plt.show()
    
    #churned vs unchurned usage frequency
    sns.countplot(data=churned,x='Usage Frequency')
    plt.title("churned usage frequency")
    plt.show()

    sns.countplot(data=unchurned,x='Usage Frequency')
    plt.title("unchurned usage frequency")
    plt.show()

    #churned vs unchurned contract length
    sns.countplot(data=churned,x='Contract Length')
    plt.title("churned usage frequency")
    plt.show()
    
    sns.countplot(data=unchurned,x='Contract Length')
    plt.title("churned usage frequency")
    plt.show()

    #churned vs unchruned support calls
    sns.countplot(data=churned,x="Support Calls")
    plt.title("churned support calls")
    plt.show()
    
    sns.countplot(data=unchurned,x="Support Calls")
    plt.title("unchurned support calls")
    plt.show()

    #churned vs unchurned tenure
    sns.histplot(data=unchurned,x='Tenure',bins=10)
    plt.title("unchurned tenure")
    plt.show()

    sns.histplot(data=churned,x='Tenure',bins=10)
    plt.title("churned tenure")
    plt.show()

    #churned vs unchruned total spend
    sns.histplot(data=churned,x='Total Spend',bins=10)
    plt.title("churned total spend")
    plt.show()
    
    sns.histplot(data=unchurned,x='Total Spend',bins=10)
    plt.title("unchurned total spend")
    plt.show()

    #churned vs unchruned age
    sns.histplot(data=unchurned,x='Age',bins=10)
    plt.title("unchurned age")
    plt.show()
    
    sns.histplot(data=churned,x='Age',bins=10)
    plt.title("churned age")
    plt.show()
    
    #unchruned vs churned gender
    sns.countplot(data=unchurned,x='Gender')
    plt.title("unchurned Gender")
    plt.show()
    
    sns.countplot(data=churned,x='Gender')
    plt.title("churned Gender")
    plt.show()
    
    #unchurned vs churned sub type
    sns.countplot(data=unchurned,x='Subscription Type')
    plt.title("unchurned Sub type")
    plt.show()
    
    sns.countplot(data=churned,x='Subscription Type')
    plt.title("churned Sub type")
    plt.show()
    
if __name__ == "__main__":
    from src.data.load_data import load_dataset
    from src.data.preprocess import clean_dataset

    raw = load_dataset("data/raw/train.csv")
    clean = clean_dataset(raw)
    plot_eda(clean)
