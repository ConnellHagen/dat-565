import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("assignment_1/swedish_population_by_year_and_sex_1860-2022.csv")

df.loc[df["age"] == "110+", "age"] = "110"

age_0_to_14 =  df[df["age"].astype(int) <= 14].sum().drop(["age", "sex"], axis="index")
age_15_to_64 = df[(df["age"].astype(int) > 14) & (df["age"].astype(int) < 65)].sum().drop(["age", "sex"], axis="index")
age_65_plus =  df[df["age"].astype(int) >= 65].sum().drop(["age", "sex"], axis="index")

dep_ratio = np.array([[i, 0.0] for i in range(1860, 2023)])
dep_ratio = pd.DataFrame(dep_ratio)
dep_ratio.columns = ["year", "dep_ratio"]

total_pop = np.array([[i, 0.0] for i in range(1860, 2023)])
total_pop = pd.DataFrame(total_pop)
total_pop.columns = ["year", "population"]

frac_pop = np.array([[i, 0.0, 0.0, 0.0] for i in range(1860, 2023)])
frac_pop = pd.DataFrame(frac_pop)
frac_pop.columns = ["year", "0_14", "15_64", "65+"]

for i in range(1860, 2023):
    dep_ratio.loc[dep_ratio["year"] == i, "dep_ratio"] = 100 * (age_0_to_14.iloc[i - 1860] + age_65_plus.iloc[i - 1860]) / age_15_to_64.iloc[i - 1860] 
    total_pop.loc[total_pop["year"] == i, "population"] = age_0_to_14.iloc[i - 1860] + age_65_plus.iloc[i - 1860] + age_15_to_64.iloc[i - 1860]
    
    frac_pop.loc[frac_pop["year"] == i, "0_14"] = age_0_to_14.iloc[i - 1860] / total_pop.loc[total_pop["year"] == i, "population"]
    frac_pop.loc[frac_pop["year"] == i, "15_64"] = age_15_to_64.iloc[i - 1860] / total_pop.loc[total_pop["year"] == i, "population"]
    frac_pop.loc[frac_pop["year"] == i, "65+"] = age_65_plus.iloc[i - 1860] / total_pop.loc[total_pop["year"] == i, "population"]

plt.scatter([i for i in range(1860, 2023)], [dep_ratio.loc[dep_ratio["year"] == i, "dep_ratio"] for i in range(1860, 2023)])
    
plt.show()

plt.scatter([i for i in range(1860, 2023)], [frac_pop.loc[frac_pop["year"] == i, "0_14"] for i in range(1860, 2023)], color = "red", label = "0 - 14")
plt.scatter([i for i in range(1860, 2023)], [frac_pop.loc[frac_pop["year"] == i, "15_64"] for i in range(1860, 2023)], color = "green", label = "15 - 64")
plt.scatter([i for i in range(1860, 2023)], [frac_pop.loc[frac_pop["year"] == i, "65+"] for i in range(1860, 2023)], color = "blue", label = "65+")
plt.legend()

plt.show()
