import pandas as pd

# ==== File paths ====
backtest_file = "backtest_df.xlsx"
features_file = "feature_backtest.csv"
output_file = "backtest_df_updated.csv"

# ==== Read ====
df_backtest = pd.read_excel(backtest_file, sheet_name="Sheet1")
df_features = pd.read_csv(features_file)


# ---- Rename matching key columns ----
df_backtest.rename(columns={df_backtest.columns[1]: "date",
                            df_backtest.columns[2]: "symbol"}, inplace=True)
df_features.rename(columns={df_features.columns[7]: "date",
                            df_features.columns[6]: "symbol"}, inplace=True)




# # ---- Clean symbols ----
# df_backtest["symbol"] = df_backtest["symbol"].astype(str).str.strip().str.upper()
# df_features["symbol"] = df_features["symbol"].astype(str).str.strip().str.upper()

# # ---- Convert dates to datetime (dayfirst for CSV) ----
df_backtest["date"] = pd.to_datetime(df_backtest["date"], errors="coerce").dt.normalize()
df_features["date"] = pd.to_datetime(df_features["date"], errors="coerce", dayfirst=True).dt.normalize()

# print(df_backtest['date'],df_backtest['symbol'])
# print(df_features['date'],df_features['symbol'])
# # ---- Drop duplicates in features ----
# df_features = df_features.drop_duplicates(subset=["date", "symbol"], keep="last")

# # ---- Get columns I to BV (8 to 73 in 0-based indexing) ----
cols_to_copy = df_features.columns[8:73]

# # ---- Merge exact match ----
df_merged = pd.merge(
    df_backtest,
    df_features[["date", "symbol"] + list(cols_to_copy)],
    on=["date", "symbol"],
    how="left"
)

# # ---- Place features starting from col N (index 13 in pandas) ----
N_index = 13
df_final = pd.concat([df_merged.iloc[:, :N_index], df_merged[cols_to_copy]], axis=1)

# # ==== Save ====
df_final.to_csv(output_file, index=False)
print(f"✅ Updated file saved to {output_file}")
