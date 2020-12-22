import pandas as pd
from datetime import datetime, timedelta
from covid_xprize.validation.scenario_generator import get_raw_data, generate_scenario, NPI_COLUMNS



#start_date = datetime.now() + timedelta(days=7)
#start_date_str = start_date.strftime('%Y-%m-%d')

start_date_str = '2020-10-01'
start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')

end_date = start_date + timedelta(days=180)
end_date_str = end_date.strftime('%Y-%m-%d')
print(f"Start date: {start_date_str}")
print(f"End date: {end_date_str}")


DATA_FILE = '../data/OxCGRT_latest.csv'
latest_df = get_raw_data(DATA_FILE,latest=False)
scenario_df = generate_scenario(start_date_str,end_date_str,latest_df,countries=None,scenario="Freeze")
scenario_file = "180_days_future_scenario.csv"
scenario_df.to_csv(scenario_file,index=False)
print(f"Saved scenario to {scenario_file}")
