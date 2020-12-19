import sys
import numpy as np
import pandas as pd


path_to_latest   = sys.argv[1]
output_file_name = sys.argv[2]

coeff_df = pd.read_csv(path_to_latest,
                       encoding="ISO-8859-1",
                       dtype={"CountryName": str,
                              "RegionName": str},
                       error_bad_lines=False)

coeff_df["GeoID"] = np.where(coeff_df["RegionName"].isnull(),
                             coeff_df["CountryName"],
                             coeff_df["CountryName"] + '__' + coeff_df["RegionName"])

output_df = coeff_df.drop(columns=['RegionName','CountryName'])

output_df.to_csv(output_file_name,index=False,float_format='%g')
