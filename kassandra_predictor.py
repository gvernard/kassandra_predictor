import numpy as np
import pandas as pd

import settings


class KassandraPredictor:
    model_df = pd.DataFrame(index=[0],columns=[0])
    Llog = 4

    h = [0.012346,
         0.024691,
         0.037037,
         0.049383,
         0.061728,
         0.074074,
         0.08642,
         0.098765,
         0.11111,
         0.098765,
         0.08642,
         0.074074,
         0.061728,
         0.049383,
         0.037037,
         0.024691,
         0.012346
    ]

    
    def __init__(self,model_file):
        self.model_df = pd.read_csv(model_file,
                               encoding="ISO-8859-1",
                               dtype={"Name": str},
                               error_bad_lines=False)


        

    def Psi(self,N_days,latest_ips):
        mat = np.zeros((N_days,len(latest_ips)+1))
        mat[:,0] = 1
        
        col = latest_ips/10.0
        col = np.convolve(self.h,col)

        for j in range(0,len(latest_ips)):
            for i in range(0,N_days):
                mat[i,j+1] = col[j]
                
        return mat



    
    
    def predict_per_country(self,GeoID,n_days,latest_data):
        myslice = self.model_df[self.model_df['Name'] == GeoID].drop(columns=['Name']).values.tolist()

        latest_new_cases = latest_data[-1]
        latest_ips = latest_data[:-1]
        
        if len(myslice) > 0:
            a_hat = myslice[0]
            N_a_hat = len(a_hat)
            
            psi = self.Psi(n_days,latest_ips)
            z_hat = psi.dot(a_hat)

            y_hat = np.zeros(n_days)
            y_hat[0] = (self.Llog + latest_new_cases)*np.exp(z_hat[0]) - self.Llog
            for i in range(1,len(y_hat)):
                y_hat[i] = (self.Llog + y_hat[i-1])*np.exp(z_hat[i]) - self.Llog
            
            pred_new_cases = y_hat
        else :
            pred_new_cases = [0] * n_days        
        return pred_new_cases

    
        
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
        HYPOTHETICAL_SUBMISSION_DATE = start_date #np.datetime64("2020-07-31")
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

        #latest_df.to_csv('input_df.csv',index=False)
       
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
            country = input_df[input_df.GeoID == g].iloc[0].CountryName
            region  = input_df[input_df.GeoID == g].iloc[0].RegionName

            # slice the input required to make a prediction
            latest_data = input_df[input_df.GeoID == g].iloc[-1]    # Select latest entry
            latest_data = latest_data[settings.MY_IPS+['NewCases']].values # Filter and get only the selected MY_IPS and the new cases
            
            # Here call the model for each GeoID
            pred_new_cases = self.predict_per_country(g,n_days,latest_data)
            #pred_new_cases = [0] * n_days

            geo_start_date = start_date            
            for i,pred in enumerate(pred_new_cases):
                forecast["CountryName"].append(country)
                forecast["RegionName"].append(region)
                current_date = geo_start_date + pd.offsets.Day(i)
                forecast["Date"].append(current_date)
                forecast["PredictedDailyNewCases"].append(pred)
                
        # Convert dictionary to DataFrame and return only the requested predictions
        forecast_df = pd.DataFrame.from_dict(forecast)
        return forecast_df[(forecast_df.Date >= start_date) & (forecast_df.Date <= end_date)]
