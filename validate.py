import os
from covid_xprize.validation.predictor_validation import validate_submission

def validate(start_date,end_date,ip_file,output_file):
    # First, delete any potential old file
    try:
        os.remove(output_file)
    except OSError:
        pass
    
    # Then generate the prediction, calling the official API
    os.system(f"python predict.py -s {start_date} -e {end_date} -ip {ip_file} -o {output_file}")
    
    # And validate it
    errors = validate_submission(start_date,end_date,ip_file,output_file)
    if errors:
        for error in errors:
            print(error)
    else:
        print("All good!")


# Validation 4 days into the future
validate(start_date="2020-12-12",
         end_date="2020-12-30",
         #ip_file="../../../validation/data/2020-09-30_historical_ip.csv",
         ip_file="data/OxCGRT_latest.csv",
         output_file="predictions/val_4_days.csv")


# Validation 180 into the future
validate(start_date="2020-10-01",
         end_date="2021-03-30",
         ip_file="scenarios/180_days_future_scenario.csv",
         output_file="predictions/val_6_month_future.csv")
