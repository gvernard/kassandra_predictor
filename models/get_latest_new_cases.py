import sys
import numpy as np
import pandas as pd

# This is the latest version of the Oxford dataset with the "confirmed new cases"
path_to_latest = sys.argv[1]

latest_df = pd.read_csv(path_to_latest,
                        parse_dates=['Date'],
                        encoding="ISO-8859-1",
                        dtype={"CountryName": str,
                               "RegionName": str},
                        error_bad_lines=False)

# Get everything that is before a LAST_DATE
LAST_DATE = np.datetime64("2020-11-30")
latest_df = latest_df[latest_df.Date <= LAST_DATE]

# Restrict data to the Geo columns and the new cases
latest_df = latest_df[['CountryName','RegionName','Date','ConfirmedCases']]

# Create unique GeoID
# GeoID is CountryName__RegionName
# np.where usage: if A then B else C
latest_df["GeoID"] = np.where(latest_df["RegionName"].isnull(),
                              latest_df["CountryName"],
                              latest_df["CountryName"] + '__' + latest_df["RegionName"])

# Add new cases column and fill it
latest_df['NewCases'] = latest_df.groupby('GeoID').ConfirmedCases.diff().fillna(0)

# Select latest entry
latest_df = latest_df.loc[latest_df.groupby('GeoID').Date.idxmax()]

# Output GeoID and NewCases only
output_df = latest_df[['GeoID','NewCases']]

output_df.to_csv('latest_new_cases.csv',index=False)
