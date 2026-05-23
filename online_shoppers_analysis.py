import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

sns.set_theme(style="whitegrid")

df = pd.read_csv("data/online_shoppers_intention.csv")
print(df.head())
print(df.shape)
print(df.info())

print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
clean_df = df.drop_duplicates().copy()
clean_df["Revenue_Int"] = clean_df["Revenue"].astype(int)
clean_df["Weekend_Int"] = clean_df["Weekend"].astype(int)
clean_df["Purchase_Status"] = clean_df["Revenue_Int"].map({0: "No Purchase", 1: "Purchase"})
clean_df.to_csv("data/cleaned_online_shoppers_intention.csv", index=False)

print(clean_df.describe())
print("Purchase rate:", clean_df["Revenue_Int"].mean() * 100)

revenue_counts = clean_df["Purchase_Status"].value_counts()
sns.barplot(x=revenue_counts.index, y=revenue_counts.values, color="#4C72B0")
plt.title("Purchase vs No Purchase")
plt.show()

numeric_df = clean_df.select_dtypes(include=["int64", "float64"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

buyer_comparison = clean_df.groupby("Purchase_Status")[["ProductRelated", "ProductRelated_Duration", "PageValues", "BounceRates", "ExitRates"]].mean()
print(buyer_comparison)

contingency_table = pd.crosstab(clean_df["VisitorType"], clean_df["Revenue_Int"])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print("Chi-square statistic:", chi2)
print("p-value:", p_value)
