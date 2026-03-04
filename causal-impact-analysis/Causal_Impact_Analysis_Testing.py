import pandas as pd
import numpy as np

# Configure dataframe printing
desired_width = 320                                                 # shows columns with X or fewer characters
pd.set_option("display.width", desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 10)                            # shows Y columns in the display
pd.set_option("display.max_rows", 20)                               # shows Z rows in the display
pd.set_option("display.min_rows", 10)                               # defines the minimum number of rows to show
pd.set_option("display.precision", 3)                               # displays numbers to 3 dps


store = pd.read_csv("cia_store.csv")
promo2_stores = store[store["Promo2"] == 1][["Store", "StoreType",
                                              "Assortment",
                                              "Promo2SinceWeek",
                                              "Promo2SinceYear",
                                              "PromoInterval"]]

promo2_stores = promo2_stores[promo2_stores["Promo2SinceYear"] == 2014]

print(promo2_stores.head(20))
print(f"\nTotal stores with Promo2 active: {len(promo2_stores)}")