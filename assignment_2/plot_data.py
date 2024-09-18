import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("assignment_2/housing_data.csv")
df_2022 = df.loc[df["Date Sold"].str.contains("2022")]

closing_price_data = {
    "minmum": df_2022["Closing Price"].min(),
    "maxmimum": df_2022["Closing Price"].max(),
    "first_q": df_2022["Closing Price"].quantile(0.25),
    "median": df_2022["Closing Price"].median(),
    "third_q": df_2022["Closing Price"].quantile(0.75)
}

plt.hist(df_2022["Closing Price"], bins = 11)
plt.title("House Closing Prices in Kungälv 2022")
plt.xlabel("Price in Tens of Millions of SEK")
plt.ylabel("Number of Houses Sold")
plt.show()


closing_prices = df_2022["Closing Price"]
boareas = df_2022["Living Area"].str.split("+").str[0].astype("float64")
rooms = df_2022["Rooms"]

plt.scatter(closing_prices, boareas, s = 20)
plt.title("Price and Boarea of Houses Sold in Kungälv in 2022")
plt.xlabel("Price in Tens of Millions of SEK")
plt.ylabel("Boarea in m²")
plt.show()

plt.scatter(closing_prices[(rooms <= 2)], boareas[(rooms <= 2)], s = 20, color = 'red', label = '0 - 2')
plt.scatter(closing_prices[(rooms <= 4) & (rooms > 2)], boareas[(rooms <= 4) & (rooms > 2)], s = 20, color = 'orange', label = '3 - 4')
plt.scatter(closing_prices[(rooms <= 6) & (rooms > 4)], boareas[(rooms <= 6) & (rooms > 4)], s = 20, color = '#FFD700', label = '5 - 6')
plt.scatter(closing_prices[(rooms <= 8) & (rooms > 6)], boareas[(rooms <= 8) & (rooms > 6)], s = 20, color = 'green', label = '7 - 8')
plt.scatter(closing_prices[(rooms >= 9)], boareas[(rooms >= 9)], s = 20, color = 'blue', label = '9+')
plt.title("Price and Boarea of Houses Sold in Kungälv in 2022")
plt.xlabel("Price in Tens of Millions of SEK")
plt.ylabel("Boarea in m²")
plt.legend(title = "Number of Rooms")
plt.show()
