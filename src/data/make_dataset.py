import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../../data/raw/creditcard.csv")
df.head()

df.describe()

df.info()

df.isnull().sum()

df.columns

# check class skewneww and the percentage of fraud and non-fraud transactions
df['Class'].value_counts(normalize=True)
df['Class'].value_counts()

#  plot the distribution of the Amount column and Time column
sns.histplot(df['Amount'], kde=True, bins=50)
sns.histplot(df['Time'], kde=True, bins=50)
plt.show()

# feature Amount is highly skewed, we can use log transformation to reduce the skewness
df["Amount_log"] = np.log(df["Amount"] + 0.00001)

# dropping redundant columns
df = df.drop(["Time", "Amount"], axis=1)

# save the preprocessed data to a new csv file
df.to_csv("../../data/interim/01_non_skew_data.csv", index=False)
