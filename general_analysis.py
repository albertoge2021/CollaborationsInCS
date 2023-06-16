import pandas as pd
import plotly.express as px
import seaborn as sns
from ast import literal_eval
from geopy.distance import geodesic as GD
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import f_oneway
from scipy.stats import shapiro

# https://www.cwauthors.com/article/Four-common-statistical-test-ideas-to-share-with-your-academic-colleagues
# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

# Setup Data
df = pd.read_csv("cs_all_test.csv")
df = df[df["distance"] > 0]
df = df[df["year"] > 1980]

# Descriptive statistics
# print("Describe")
# print(df[["type","distance"]].groupby("type").describe())
"""
              count         mean          std       min         25%          50%          75%           max
type                                                                                                       
company      1432.0  4878.240003  4057.167163  0.070442  817.066474  4146.040111  8343.072184  18240.536255
education  316214.0  4387.616720  4693.693908  0.001219  460.648206  2024.854672  8188.014895  19965.472832
mixed       38968.0  4307.127391  4372.683918  0.043201  448.410212  2773.327000  8044.148213  19871.738704
"""

# Normality check
# It does not come from normal distribution

# Kruskal test - Statistical diference between groups
# print("Kruskal")
# print(stats.kruskal(df[df['type'] == "company"]['distance'], df[df['type'] == "education"]['distance'], df[df['type'] == "mixed"]['distance']))
"""
KruskalResult(statistic=32.343252344059835, pvalue=9.47877010340211e-08) - There are statistical diferences between groups
"""


# Pearson test and Spearman test- correlation coeficient
# print("PEARSON")
unique_majors = df["type"].unique()
df["dist_trunc"] = round(df["distance"], 0)
new_df = df.groupby(["type", "dist_trunc"]).size().reset_index(name="count")
# for major in unique_majors:
#    print(major + " "+  str(stats.pearsonr(new_df[new_df['type'] == major]['dist_trunc'], new_df[new_df['type'] == major]['count'])))
#    print(major + " "+  str(stats.spearmanr(new_df[new_df['type'] == major]['dist_trunc'], new_df[new_df['type'] == major]['count'])))
"""
education PearsonRResult(statistic=-0.36222157913146663, pvalue=0.0) - Low negative correlation, correlation coefficient is called statistically significant.
mixed PearsonRResult(statistic=-0.23949458179289307, pvalue=9.523685023092745e-123) - Low negative correlation, correlation coefficient is called statistically significant.
company PearsonRResult(statistic=-0.034043875141394196, pvalue=0.32408904498656904) - No association

education SignificanceResult(statistic=-0.6176221659117133, pvalue=0.0) - Strong negative correlation, correlation coefficient is called statistically significant.
mixed SignificanceResult(statistic=-0.4036757625569225, pvalue=0.0) - Strong negative correlation, correlation coefficient is called statistically significant.
company SignificanceResult(statistic=-0.06969926044398334, pvalue=0.0433077102410395) - No association
"""


# Plot regression
# sns.lmplot(
#    x="dist_trunc",
#    y="count",
#    hue="type",
#    data=new_df,
#    scatter=False,
# )
# plt.show()
"""
Photo in folder
"""


# Test distance by year
# means = df.groupby(['year', 'type'])['distance'].mean().reset_index(name="mean")
# sns.lineplot(data=means, x="year", y="mean", hue="type")
# plt.show()
"""
Photo in folder
"""

# Probability distribution
# for major in unique_majors:
#    stats.probplot(df[df['type'] == major]['distance'], dist="norm", plot=plt)
#    plt.title("Probability Plot - " +  major)
#    plt.show()

# Boxplot
# df.boxplot(by ='type', column =['distance'], grid = False)
# plt.show()
"""
Photo in folder
"""

# Denstity
# new_df.groupby('type')['dist_trunc'].plot(kind='kde')
# plt.legend(['Company', 'Education', 'Mixed'], title='Relationship')
# plt.xlabel('Distance')
# plt.show()
"""
Photo in folder
"""

# Probabilty
# sns.displot(new_df, x="dist_trunc", hue="type", stat="probability", common_norm=False)
# plt.xlabel('Distance')
# plt.show()
"""
Photo in folder
"""

# Histogram
# ax = df.plot.hist(column=["distance"], by="type", figsize=(10, 8))
# plt.show()
"""
Photo in folder
"""

"""sns.histplot(
    df, x="distance", y="citations",
    bins=30, pthresh=.05, pmax=.9,
)
plt.show()
"""
