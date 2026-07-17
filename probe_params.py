# probe_params.py
# Small utility script to check and print all baseline
# physiological parameters of a SimGlucose patient model.
# Made for quick inspection during simulation testing.

import pandas as pd
from simglucose.patient.t1dpatient import T1DPatient
print(" Biophysical Parameter Probe Started ")

try:
    # Load one of the default adult patient profiles
    patient = T1DPatient.withName("adult#001")

    # Check if the patient object stores parameter values
    if hasattr(patient, "_params"):

        print("\nParameter list successfully accessed.\n")

        # Save parameter series in a variable for convenience
        patient_params = patient._params

        # Show complete output without truncation
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)

        # Convert Series to DataFrame so it looks cleaner
        param_df = pd.DataFrame(patient_params)
        param_df.columns = ["Parameter Value"]

        print(param_df)

    else:
        print("Patient object does not contain '_params' attribute.")

except Exception as err:
    # Print any unexpected error for debugging
    print("Error while reading patient parameters:")
    print(err)