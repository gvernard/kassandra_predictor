import numpy as np
import pandas as pd


class KassandraPredictor:

    def __init__(self):
        a = 100 # dummy

        
    def manipulate(self,
                   start_date_str: str,
                   end_date_str: str,
                   path_to_ips_file: str) -> pd.DataFrame:
        start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

        latest_df = pd.read_csv(path_to_ips_file,
                                parse_dates=['Date'],
                                encoding="ISO-8859-1",
                                dtype={"RegionName": str,
                                       "RegionCode": str},
                                error_bad_lines=False)

        # GeoID is CountryName / RegionName
        # np.where usage: if A then B else C
        latest_df["GeoID"] = np.where(latest_df["RegionName"].isnull(),
                                      latest_df["CountryName"],
                                      latest_df["CountryName"] + ' / ' + latest_df["RegionName"])

        return latest_df


    def predict(self,
                start_date_str: str,
                end_date_str: str,
                input_df: pd.DataFrame) -> pd.DataFrame:
        start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')



        # Set a dictionary of lists that will contain the output
        forecast = {"CountryName": [],
                    "RegionName": [],
                    "Date": [],
                    "PredictedDailyNewCases": []}


        # For each requested geo
        geos = input_df.GeoID.unique()
        for g in geos:
            # Make prediction for each GeoID
            # and append to output accordingly


        # Dictionary to DataFrama and return only the requested predictions
        forecast_df = pd.DataFrame.from_dict(forecast)
        return forecast_df[(forecast_df.Date >= start_date) & (forecast_df.Date <= end_date)]
