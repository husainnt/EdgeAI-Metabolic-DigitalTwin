import inspect
from simglucose.patient.t1dpatient import T1DPatient

# here i try to read the inner ode model equations
try:
    # here i get the raw source code of the math equations
    ode_source = inspect.getsource(T1DPatient.model)
    
    # here i print the output to check for a natural pancreas term
    print(ode_source)

# here i catch the error if something goes wrong
except Exception as e:
    print("error reading equations:", e)