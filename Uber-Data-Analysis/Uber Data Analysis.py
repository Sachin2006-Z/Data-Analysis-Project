# Importing all the libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




# Reading the dataset
df= pd.read_csv("/content/UberDataset.csv")





# Shaping the dataset
print("Rows, Columns:", df.shape)





# Column names and data types
df.info()




# Date and Time columns

df["START_DATE"] = pd.to_datetime(df["START_DATE"], format="mixed", errors="coerce")
df["hour"] = df["START_DATE"].dt.hour
df["day"] = df["START_DATE"].dt.day
df["weekday"] = df["START_DATE"].dt.day_name()
df["month"] = df["START_DATE"].dt.month




###### EDA ######


# Chart Ploting for Trips by Hours in a day 
plt.figure(figsize=(10,5))
sns.countplot(x="hour", data=df)
plt.title("Uber Trips by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Trips")
plt.show()
print("\n\n\n")





# Chart Ploting for Trips by Days of Week

plt.figure(figsize=(10,5))
sns.countplot(
    x="weekday",
    data=df,
    order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
)
plt.title("Uber Trips by Day of Week")
plt.xlabel("Day")
plt.ylabel("Number of Trips")
plt.show()
print("\n\n\n")




# Chart Ploting for Trips by Months


df["month_name"] = df["START_DATE"].dt.month_name()
plt.figure(figsize=(12,6))
sns.countplot(
    x="month_name",
    data=df,
    order=["January","February","March","April","May","June",
           "July","August","September","October","November","December"]
)
print("\n\n\n")








# Heatmap

pivot = df.groupby(["weekday", "hour"]).size().unstack()

plt.figure(figsize=(12,6))
sns.heatmap(pivot, cmap="YlOrRd")
plt.title("Uber Demand Heatmap (Day vs Hour)")
plt.xlabel("Hour")
plt.ylabel("Day of Week")
plt.show()
print("\n\n\n")




print("""~ Uber demand is highest during peak commute hours\n

~ Weekdays show more consistent usage than weekends\n

~ Certain months show increased ride activity\n

~ Heatmap reveals clear rush-hour patterns""")
