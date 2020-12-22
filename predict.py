# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import argparse
import sys
import os

#from covid_xprize.examples.predictors.kassandra_predictor import KassandraPredictor
from kassandra_predictor import KassandraPredictor


def predict(start_date: str,end_date: str,path_to_ips_file: str,output_file_path) -> None:
    """
    Generates and saves a file with daily new cases predictions for the given countries, regions and intervention
    plans, between start_date and end_date, included.
    :param start_date: day from which to start making predictions, as a string, format YYYY-MM-DDD
    :param end_date: day on which to stop making predictions, as a string, format YYYY-MM-DDD
    :param path_to_ips_file: path to a csv file containing the intervention plans between inception date (Jan 1 2020)
     and end_date, for the countries and regions for which a prediction is needed
    :param output_file_path: path to file to save the predictions to
    :return: Nothing. Saves the generated predictions to an output_file_path CSV file
    with columns "CountryName,RegionName,Date,PredictedDailyNewCases"
    """
    # !!! YOUR CODE HERE !!!

    # Check if re-training is needed
    #project_root   ='work/'
    project_root   ='/home/george/myCodes/PandemicResponse/covid-xprize/kassandra_predictor/'
    #my_first_model = 'single_model_06_12_2020.csv'
    my_first_model = 'multi_model_22_12_2020.csv'


    
    # Initialize a predictor object by reading the trained models and manipulating the input file
    print("Initializing:",flush=True)
    print("  >>> Reading trained models . . . ",end="",flush=True)
    predictor = KassandraPredictor(project_root,my_first_model)
    print("done",flush=True)
    print("  >>> Manipulating the input . . . ",end="",flush=True)
    input_df = predictor.manipulate(start_date,end_date,path_to_ips_file)
    print("done",flush=True)
    print()
    
    # Generate the predictions given: start, end date, and Intervention Plans
    print("Predicting:",flush=True)
    print("  >>> Making predictions . . . ",end="",flush=True)
    preds_df = predictor.predict(start_date,end_date,input_df)
    print("done",flush=True)
    print()

    # Write the output to the given file
    print("Writing output:",flush=True)
    #os.makedirs(os.path.dirname(output_file_path),exist_ok=True)
    preds_df.to_csv(output_file_path,index=False)
    print("done",flush=True)
    print()
    
    print(f"SUCCESS! Saved predictions to {output_file_path}",flush=True)
    print()
    

# !!! PLEASE DO NOT EDIT. THIS IS THE OFFICIAL COMPETITION API !!!
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to predict, included, as YYYY-MM-DD. For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prediction, included, as YYYY-MM-DD. For example 2020-08-31")
    parser.add_argument("-ip", "--interventions_plan",
                        dest="ip_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to the CSV file where predictions should be written")
    args = parser.parse_args()
    print(f"Generating predictions from {args.start_date} to {args.end_date}...")
    predict(args.start_date, args.end_date, args.ip_file, args.output_file)
    print("Done!")
