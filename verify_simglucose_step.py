"""
Sanity Check: Verify simglucose patient sample time and Action signature
"""

import sys
import os
import inspect

sys.path.append(os.path.abspath("SIM-GLUCOSE"))

from simglucose.patient.t1dpatient import T1DPatient
from simglucose.controller.base import Action

patient = T1DPatient.withName("adult#001")

print("=" * 70)
print("simglucose API Verification")
print("=" * 70)

print(f"1. patient.sample_time:       {patient.sample_time} minute(s)")
print(f"2. patient.step signature:   {inspect.signature(patient.step)}")
print(f"3. Action.__init__ signature: {inspect.signature(Action.__init__)}")

print("=" * 70)
