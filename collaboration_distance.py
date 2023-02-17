import pandas as pd
import plotly.express as px
import seaborn as sns
from ast import literal_eval
from geopy.distance import geodesic as GD
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import f_oneway

df = pd.read_csv("cs_old.csv")
df = df[df['distance'] > 0]
print("MEAN")
print(df[["type","distance"]].groupby("type").mean())
print("MEDIAN")
print(df[["type","distance"]].groupby("type").median())

unique_majors = df['type'].unique()
#for major in unique_majors:
    #stats.probplot(df[df['type'] == major]['distance'], dist="norm", plot=plt)
    #plt.title("Probability Plot - " +  major)
    #plt.show()

#df.boxplot(by ='type', column =['distance'], grid = False)
#plt.show()

ratio = df.groupby('type').std().max() / df.groupby('type').std().min()
print("RATIO")
print(ratio)

print("ANOVA")
print(f_oneway(df[df['type'] == "company"]['distance'], df[df['type'] == "education"]['distance'], df[df['type'] == "mixed"]['distance']))


print("PEARSON")
df["dist_trunc"] = round(df["distance"], 0)
new_df = (
    df.groupby(["type", "dist_trunc"])
    .size()
    .reset_index(name="count")
)
for major in unique_majors:
    print(major + " "+  str(pearsonr(new_df[new_df['type'] == major]['dist_trunc'], new_df[new_df['type'] == major]['count'])))

#sns.lmplot(x="dist_trunc", y="count", hue="type", data=new_df)
#plt.show()

"""new_df.groupby('type')['dist_trunc'].plot(kind='kde')
plt.legend(['Company', 'Education', 'Mixed'], title='Relationship')
plt.xlabel('Distance')
plt.show()"""
#ax = df.plot.hist(column=["distance"], by="type", figsize=(10, 8))
#plt.show()

#Test distance by year
"""year_df = df[df['year'] > 1980]
means = year_df.groupby(['year', 'type'])['distance'].mean().reset_index(name="mean")
sns.lineplot(data=means, x="year", y="mean", hue="type")
plt.show()"""

#No all from same place
separated_df = df[df['distance'] > 0]