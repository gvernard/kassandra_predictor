import numpy as np
import pandas as pd

import settings


class KassandraPredictor:

    def __init__(self):
        a = 100 # dummy

        
    def manipulate(self,
                   start_date_str: str,
                   end_date_str: str,
                   path_to_ips_file: str) -> pd.DataFrame:
        start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

        # Read file in DataFrame
        latest_df = pd.read_csv(path_to_ips_file,
                                parse_dates=['Date'],
                                encoding="ISO-8859-1",
                                dtype={"RegionName": str,
                                       "RegionCode": str},
                                error_bad_lines=False)


        # Restrict data to that before a hypothetical predictor submission date
        HYPOTHETICAL_SUBMISSION_DATE = np.datetime64("2020-07-31")
        latest_df = latest_df[latest_df.Date <= HYPOTHETICAL_SUBMISSION_DATE]

        
        # Restrict data to selected columns 
        latest_df = latest_df[settings.ID_COLS + settings.INDICES + settings.MY_IPS]

        
        # Create unique GeoID
        # GeoID is CountryName__RegionName
        # np.where usage: if A then B else C
        latest_df["GeoID"] = np.where(latest_df["RegionName"].isnull(),
                                      latest_df["CountryName"],
                                      latest_df["CountryName"] + '__' + latest_df["RegionName"])

        # Add new cases column and fill it
        latest_df['NewCases'] = latest_df.groupby('GeoID').ConfirmedCases.diff().fillna(0)
        

        # Fill any missing IPs by assuming they are the same as previous day
        for ip_col in settings.MY_IPS:
            latest_df.update(latest_df.groupby('GeoID')[ip_col].ffill().fillna(0))
        
        return latest_df


    
    def predict(self,
                start_date_str: str,
                end_date_str: str,
                input_df: pd.DataFrame) -> pd.DataFrame:
        start_date = pd.to_datetime(start_date_str,format='%Y-%m-%d')
        end_date   = pd.to_datetime(end_date_str,format='%Y-%m-%d')
        n_days     = (end_date - start_date).days + 1



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
            pred_new_cases = [0] * n_days
            geo_start_date = start_date
            
            country = input_df[input_df.GeoID == g].iloc[0].CountryName
            region  = input_df[input_df.GeoID == g].iloc[0].RegionName
            for i,pred in enumerate(pred_new_cases):
                forecast["CountryName"].append(country)
                forecast["RegionName"].append(region)
                current_date = geo_start_date + pd.offsets.Day(i)
                forecast["Date"].append(current_date)
                forecast["PredictedDailyNewCases"].append(pred)
                
        # Convert dictionary to DataFrame and return only the requested predictions
        forecast_df = pd.DataFrame.from_dict(forecast)
        return forecast_df[(forecast_df.Date >= start_date) & (forecast_df.Date <= end_date)]
