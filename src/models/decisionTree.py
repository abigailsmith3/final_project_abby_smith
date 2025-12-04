import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree

def deision_tree(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTree:

  features = ['Total Spend','Support Calls','Contract Length_Monthly','Last Interaction','Age','Contract Length_Quarterly','Gender_Female']
  depth_limit = 10
  
  #predict only using one feature
  dt_model = DecisionTreeClassifier(criterion = 'entropy', max_depth = depth_limit)
  dt_model.fit(X_train[features],y_train)
  
  return dt_model

  

  
  
