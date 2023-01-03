import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
# reading csv file
df = pd.read_csv("hotel.csv")
shape = df.shape
null_contains = df.isnull()
total_missing_count =  df.isnull().sum()
plt.figure(figsize - (25,25)) sb.heatmap(null_contains))
total_missing_count / df.shape[0] - 100
drop_columns = null_var[null_var > 20] - Key()
