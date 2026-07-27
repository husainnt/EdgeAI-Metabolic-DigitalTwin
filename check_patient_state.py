"""
Inspect simglucose T1DPatient State Vector Structure
Used to determine exact state ordering for proportional warm-starting.
"""

import sys
import os

# Ensure local simglucose is discoverable
sys.path.append(os.path.abspath("SIM-GLUCOSE"))

from simglucose.patient.t1dpatient import T1DPatient

patient = T1DPatient.withName("adult#001")

print("=" * 70)
print("simglucose T1DPatient State Vector Inspection")
print("=" * 70)

print("\n1. patient.state:")
print("   Value:", patient.state)
print("   Type: ", type(patient.state))

if hasattr(patient, "_state"):
    print("\n2. patient._state:")
    print("   Value:", patient._state)

print("\n3. Target Parameters:")
print("   Gb (Fasting Baseline):", patient._params.get("Gb"))
print("   BW (Body Weight):     ", patient._params.get("BW"))
print("   EGP0 (Hepatic Baseline):", patient._params.get("EGP0"))

print("\n" + "=" * 70)