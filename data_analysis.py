import os
import pandas as pd
from functools import reduce

'''merge the data into desired format'''

data_folder = './Data/EastHsinchu'
feature_dfs = {} 

for subfolder in os.listdir(data_folder):
    subfolder_path = os.path.join(data_folder, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    feature_name = subfolder
    monthly_frames = []

    for file_name in os.listdir(subfolder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(subfolder_path, file_name)
            print(f"Processing: {file_path}")

            month_label = file_name[7: 14]

            df = pd.read_csv(file_path, header=None)
            df_cleaned = df.drop(index=[0]).reset_index(drop=True)
            df_trimmed = df_cleaned.iloc[:-1, :-1]

            df_melted = df_trimmed.melt(id_vars=[0], var_name="Hour", value_name=feature_name)
            df_melted.columns = ["Date", "Hour", feature_name]
            df_melted["Month"] = month_label

            df_final = df_melted[["Month", "Date", "Hour", feature_name]]
            monthly_frames.append(df_final)

    combined_feature_df = pd.concat(monthly_frames, ignore_index=True)
    feature_dfs[feature_name] = combined_feature_df

merged_df = reduce(lambda left, right: pd.merge(left, right, on=["Month", "Date", "Hour"], how="outer"), feature_dfs.values())

# merged_df["Month-Date"] = str(month_label) + "-" + merged_df["Date"]
# merged_df.drop(columns=["Date"], inplace=True)

merged_df = merged_df.sort_values(by=["Month", "Date", "Hour"]).reset_index(drop=True)
merged_df.to_csv("master_EastHsinchu.csv", index=False)

'''split the final column'''

file = pd.read_csv('master_EastHsinchu.csv')

last_col = file.columns[-1]

split_data = file[last_col].astype(str).str.split(r'[,/]', n=1, expand=True)

file["WindSpeed"] = split_data[0]
file["WindDirection"] = split_data[1]

file.drop(columns=[last_col], inplace=True)

file['Date'] = pd.to_datetime(file['Month'] + '-' + file['Date'].astype(str).str.zfill(2))
file.drop(columns=['Month'], inplace=False).to_csv('merged_date_EastHsinchu.csv', index=False)

file.to_csv('Masters/Master.csv', index=False)