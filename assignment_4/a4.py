import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

SEED = 12345678

df = pd.read_csv("assignment_4/life_expectancy.csv")
df = df.drop("Country", axis="columns")
df_train, df_test = train_test_split(df, random_state=SEED, test_size=0.25)

variables = list(df_train.columns.values)

LEB = "Life Expectancy at Birth, both sexes (years)"
HDE = "Human Development Index (value)"

pearson = df_train.corr(method='pearson', min_periods=1, numeric_only=False)
pearson_drop = pearson[LEB].drop(LEB)
# max 0.9186657942068257 Human Development Index (value),
# min -0.8646498454415058 Crude Birth Rate
# print(f"{max(pearson_drop)}, {min(pearson_drop)}")

df_train_hde = df_train[[HDE, LEB]]

linreg = LinearRegression()
linreg.fit(df_train_hde[[HDE]], df_train_hde[LEB])
pred = df_train_hde[HDE] * linreg.coef_ + linreg.intercept_

r2_train = r2_score(df_train_hde[LEB], pred)

predictions = linreg.predict(df_test[[HDE]])
# 13.730158926302014
mse = mean_squared_error(df_test[LEB], predictions)
correlation = df_test[[HDE, LEB]].corr(method='pearson', min_periods=1, numeric_only=False)
r2_test = r2_score(df_test[LEB], predictions)
# 0.918989
# print(correlation)

plt.scatter(df_test[HDE], predictions)
plt.show()

plt.scatter(df_train_hde[HDE], df_train_hde[LEB], s=1)
plt.plot(df_train_hde[HDE], pred, color="red")
plt.show()


MA = "Median Age, as of 1 July (years)"

spearman = df_train.corr(method='spearman', min_periods=1, numeric_only=False)
spearman_drop = spearman[LEB].drop(LEB)
print(f"{max(spearman_drop)}, {min(spearman_drop)}")

# no transform for comparison
df_train_nt_ma = df_train[[MA, LEB]]
linreg_nt_ma = LinearRegression()
linreg_nt_ma.fit(df_train_nt_ma[[MA]], df_train_nt_ma[LEB])
pred_nt_leb = df_train_nt_ma[MA] * linreg_nt_ma.coef_ + linreg_nt_ma.intercept_
# 0.799285
correlation_nt_ma = df_train_nt_ma[[MA, LEB]].corr(method='pearson', min_periods=1, numeric_only=False)

# with transform
df_train_ma = df_train[[MA, LEB]]
df_train_ma["LEB Transform"] = (np.pow(df_train_ma[LEB], 6))

df_test_ma = df_test[[MA, LEB]]
df_test_ma["LEB Transform"] = (np.pow(df_test_ma[LEB], 6))

linreg_ma = LinearRegression()
linreg_ma.fit(df_train_ma[[MA]], df_train_ma["LEB Transform"])
pred_leb = df_train_ma[MA] * linreg_ma.coef_ + linreg_ma.intercept_

# for 4: 0.7052068299250327
# for 5: 0.7103432261911896
# for 6: 0.7089855707748298
# for 7: 0.7023901539198063
r2_ma = r2_score(df_train_ma["LEB Transform"], pred_leb)

# 0.831497
correlation_ma = df_test_ma[[MA, "LEB Transform"]].corr(method='pearson', min_periods=1, numeric_only=False)
# print(correlation_ma)

plt.plot(df_train_ma[MA], pred_leb, color="red")
plt.scatter(df_train_ma[MA], df_train_ma["LEB Transform"], s=1)
plt.show()


print(pearson_drop)

for feature in variables:
    df_train[feature].fillna(value=df_train[feature].dropna().mean(), inplace=True)
    df_test[feature].fillna(value=df_test[feature].dropna().mean(), inplace=True)


# for s in subsets:
s = ["Crude Birth Rate (births per 1,000 population)"]
best_mse = 20

for v in variables:
    if v == "Crude Birth Rate (births per 1,000 population)":
        continue
    if v == HDE or v == LEB:
        continue
    s.append(v)

    ln = LinearRegression()
    ln.fit(df_train[[*s]], df_train[LEB])

    predictions = ln.predict(df_test[[*s]])
    mse = mean_squared_error(df_test[LEB], predictions)
    if mse < best_mse:
        best_mse = mse
    else:
        s.pop()



# the initial list was everything listed here,
# then we removed entries that were similar to each other (the ones with no number next to them)
# then we found the pearson correlation coefficients for each of the remaining variables, and removed all
# with coefficient c such that |c| < 0.5
newvars = [
    'Crude Birth Rate (births per 1,000 population)', # -0.86
    # 'Rate of Natural Change (per 1,000 population)',
    # 'Population Change (thousands)',
    # 'Population Growth Rate (percentage)', # -0.2853
    # 'Population Annual Doubling Time (years)',
    # 'Births (thousands)',
    # 'Births by women aged 15 to 19 (thousands)',
    'Total Fertility Rate (live births per woman)', # -0.83
    'Net Reproduction Rate (surviving daughters per woman)', # -0.78
    # 'Mean Age Childbearing (years)', # 0.0423
    # 'Sex Ratio at Birth (males per 100 female births)', # 0.407
    # 'Total Deaths (thousands)',
    'Crude Death Rate (deaths per 1,000 population)', # -0.54
    # 'Live births Surviving to Age 1 (thousands)',
    # 'Net Number of Migrants (thousands)',
    # 'Net Migration Rate (per 1,000 population)' # 0.13
]

ln = LinearRegression()
ln.fit(df_train[[*newvars]], df_train[LEB])

predictions = ln.predict(df_test[[*newvars]])
mse = mean_squared_error(df_test[LEB], predictions)
print(mse)
print(f"{s}: mse:{mse}, intercept: {ln.intercept_}, coeff: {ln.coef_}")

r2_test = r2_score(df_test[LEB], predictions)
df_test2 = pd.DataFrame({"leb": df_test[LEB], "pred": predictions})
pearson = df_test2.corr(method='pearson', min_periods=1, numeric_only=False)