import numpy as np
import pandas as pd

import settings


class KassandraPredictor:
    project_root = '.'
    model_df = pd.DataFrame(index=[0],columns=[0])
    latest_new_cases_df = pd.DataFrame(index=[0],columns=[0])
    Llog = 4
    K = 30
    M = 300
    rate = 1.5

    model_type = 'single'
    
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

    
    
    def __init__(self,project_root,model_file):
        self.project_root = project_root
        
        self.model_df = pd.read_csv(project_root+'models/'+model_file,
                                    encoding="ISO-8859-1",
                                    dtype={"Name": str},
                                    error_bad_lines=False)

        txt = model_file.split('_')
        if txt[0] == 'multi':
            self.model_type = 'multi'
        elif txt[0] == 'single':
            self.model_type = 'single'
        else:
            self.model_type = 'unknown'
            
        self.latest_new_cases_df = pd.read_csv(project_root+'models/latest_new_cases.csv',
                                               encoding="ISO-8859-1",
                                               dtype={"GeoID": str},
                                               error_bad_lines=False)
        

        
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
        latest_df = latest_df[settings.ID_COLS + settings.MY_IPS]
        
        # Create unique GeoID
        # GeoID is CountryName__RegionName
        # np.where usage: if A then B else C
        latest_df["GeoID"] = np.where(latest_df["RegionName"].isnull(),
                                      latest_df["CountryName"],
                                      latest_df["CountryName"] + '__' + latest_df["RegionName"])

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

        forecast_sdev = {"CountryName": [],
                         "RegionName": [],
                         "Date": [],
                         "PredictedDailySdev": []}
        
        
        # For each requested geo
        geos = input_df.GeoID.unique()
        for g in geos:
            country = input_df[input_df.GeoID == g].iloc[0].CountryName
            region  = input_df[input_df.GeoID == g].iloc[0].RegionName

            # slice the input required to make a prediction
            latest_data = input_df[input_df.GeoID == g].iloc[-1]  # Select latest entry
            latest_data = latest_data[settings.MY_IPS].values     # Filter and get only the selected MY_IPS and the new cases
            
            # Here call the model for each GeoID
            if self.model_type == 'single':
                pred_new_cases = self.predict_per_country(g,n_days,latest_data)
            elif self.model_type == 'multi':
                pred_new_cases,pred_sdev = self.predict_per_country_multi(g,n_days,latest_data)                
            else:
                pred_new_cases = [0] * n_days

            geo_start_date = start_date            
            for i,pred in enumerate(pred_new_cases):
                forecast["CountryName"].append(country)
                forecast["RegionName"].append(region)
                current_date = geo_start_date + pd.offsets.Day(i)
                forecast["Date"].append(current_date)
                forecast["PredictedDailyNewCases"].append(pred)

            if self.model_type == 'multi':
                geo_start_date = start_date            
                for i,pred in enumerate(pred_sdev):
                    forecast_sdev["CountryName"].append(country)
                    forecast_sdev["RegionName"].append(region)
                    current_date = geo_start_date + pd.offsets.Day(i)
                    forecast_sdev["Date"].append(current_date)
                    forecast_sdev["PredictedDailySdev"].append(pred)

                    
        if self.model_type == 'multi':
            forecast_sdev_df = pd.DataFrame.from_dict(forecast_sdev)
            forecast_sdev_df = forecast_sdev_df[(forecast_sdev_df.Date >= start_date) & (forecast_sdev_df.Date <= end_date)]
            forecast_sdev_df.to_csv(self.project_root+"predictions/sdev_"+start_date_str+"_"+end_date_str+".csv",index=False)

                    
        # Convert dictionary to DataFrame and return only the requested predictions
        forecast_df = pd.DataFrame.from_dict(forecast)
        return forecast_df[(forecast_df.Date >= start_date) & (forecast_df.Date <= end_date)]


    
    def Psi(self,N_days,latest_ips):
        mat = np.zeros((N_days,len(latest_ips)+1))
        mat[:,0] = 1
        col = latest_ips/10.0
        col = np.convolve(self.h,col)
        for j in range(0,len(latest_ips)):
            for i in range(0,N_days):
                mat[i,j+1] = col[j]
        return mat

    def predict_per_country(self,GeoID,n_days,latest_ips):
        myslice = self.model_df[self.model_df['Name'] == GeoID].drop(columns=['Name']).values.tolist()
        latest_new_cases = self.latest_new_cases_df[self.latest_new_cases_df['GeoID'] == GeoID].iloc[0].at['NewCases']
        
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

    




    def Psi_multi(self,n_days,latest_ips):
        mat = np.zeros((n_days,len(latest_ips)+1))
        mat[:,0] = 1
        col = latest_ips/4.0
        col = np.convolve(self.h,col)
        for j in range(0,len(latest_ips)):
            for i in range(0,n_days):
                mat[i,j+1] = col[j]                
        return mat

    def predict_per_country_multi(self,GeoID,n_days,latest_ips):
        coeffs = self.model_df[self.model_df['GeoID'] == GeoID].drop(columns=['GeoID']).values.tolist()
        latest_new_cases = self.latest_new_cases_df[self.latest_new_cases_df['GeoID'] == GeoID].iloc[0].at['NewCases']

        n_models = len(coeffs)
        if n_models > 0:
            model_predictions = np.zeros((n_models,n_days))
            for k in range(0,n_models):
                a_hat = coeffs[k]
                
                psi = self.Psi_multi(n_days,latest_ips)
                z_hat = psi.dot(a_hat)

                y_hat = np.zeros(n_days)
                y_hat[0] = (self.Llog + latest_new_cases)*np.exp(z_hat[0]) - self.Llog
                for i in range(1,len(y_hat)):
                    if i > self.K:
                        y_hat[i] = (self.Llog + y_hat[i-1])*np.exp(z_hat[i] - self.rate*(i-self.K)/self.M) - self.Llog
                    else:
                        y_hat[i] = (self.Llog + y_hat[i-1])*np.exp(z_hat[i]) - self.Llog
                model_predictions[k,:] = y_hat

            pred_new_cases = [0] * n_days
            pred_sdev = [0] * n_days
            for d in range(0,n_days):
                preds = np.sort(model_predictions[:,d])
                preds = preds[5:-5]
                mymean = np.mean(preds)
                mysdev = np.std(preds)
                pred_new_cases[d] = mymean
                pred_sdev[d] = mysdev
        else :
            pred_new_cases = [0] * n_days
            pred_sdev = [0] * n_days

        return pred_new_cases,pred_sdev
