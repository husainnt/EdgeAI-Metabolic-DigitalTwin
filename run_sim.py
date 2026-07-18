import os
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim
from simglucose.controller.base import Controller, Action

# here i create a custom pancreas controller for type 2 diabetes
class T2DPancreaticController(Controller):
    def __init__(self, gb=138.56, basal_rate=0.0211, init_state=0):
        # here i pass init_state to the parent controller class
        super().__init__(init_state)
        # here i set the fasting glucose target
        self.gb = gb
        # here i set the baseline basal insulin rate in U/min
        self.basal_rate = basal_rate
        # here i scale down kp to match the new U/min units
        self.kp = 0.0005 
        # here i set a realistic maximum secretion ceiling in U/min
        self.max_secretion = 0.05 

    def policy(self, observation, reward, done, **info):
        # here i get the current cgm reading
        bg = observation.CGM
        
        # here i check if glucose is below or at fasting baseline
        if bg <= self.gb:
            secretion = self.basal_rate
        else:
            # here i calculate the sluggish insulin release based on U/min units
            excess_glucose = bg - self.gb
            secretion = self.basal_rate + (self.kp * excess_glucose)
            
        # here i clamp the secretion to the maximum pancreatic capacity
        secretion = min(secretion, self.max_secretion)
        
        # here i return the insulin output as a basal action
        return Action(basal=secretion, bolus=0)

    def reset(self):
        pass

# here i make sure the results directory exists so the json save never crashes
os.makedirs("results", exist_ok=True)

# here i load the baseline patient model
patient = T1DPatient.withName("adult#001")
params = patient._params

# here i apply the final validated T2D parameter configuration from our sweep matrix
params['Vmx'] = params['Vmx'] * 0.75
params['kp3'] = params['kp3'] * 0.70
patient._params = params

# here i convert the baseline basal insulin from pmol/kg/min to U/min
basal_rate_correct = params['u2ss'] * params['BW'] / 6000

# here i setup the controller and sensor nodes
controller = T2DPancreaticController(gb=params['Gb'], basal_rate=basal_rate_correct, init_state=0)
sensor = CGMSensor.withName("Dexcom", seed=1)
pump = InsulinPump.withName("Insulet")

# here i set the midnight start time
start_time = datetime.combine(datetime.now().date(), datetime.min.time())

# here i build the ideal 3-day meal timeline with South Asian carb loads
ideal_meals = [
    # --- Day 1 ---
    (start_time + timedelta(hours=8), 60),    # breakfast at 8:00 AM
    (start_time + timedelta(hours=13), 80),   # lunch at 1:00 PM
    (start_time + timedelta(hours=20), 70),   # dinner at 8:00 PM
    
    # --- Day 2 ---
    (start_time + timedelta(days=1, hours=8), 60),
    (start_time + timedelta(days=1, hours=13), 80),
    (start_time + timedelta(days=1, hours=20), 70),
    
    # --- Day 3 ---
    (start_time + timedelta(days=2, hours=8), 60),
    (start_time + timedelta(days=2, hours=13), 80),
    (start_time + timedelta(days=2, hours=20), 70)
]

# here i load the custom scenario and run the engine
scenario = CustomScenario(start_time=start_time, scenario=ideal_meals)
env = T1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)
s = SimObj(env=env, controller=controller, sim_time=timedelta(days=3), animate=False, path="./results")

print("Running finalized calibrated 3-day simulation...")
results = sim(s)

# here i calculate final validation metrics
fasting_mean = results[results["CHO"] == 0]["BG"].mean()
peak = results["BG"].max()
time_in_range = ((results["BG"] >= 70) & (results["BG"] <= 180)).mean() * 100
time_above_180 = (results["BG"] > 180).mean() * 100
time_below_70 = (results["BG"] < 70).mean() * 100

# here i save the clean data array and validation summary json artifacts
results.to_csv("results/adult001_3day_glucose.csv")

summary = {
    "vmx_scale": 0.75,
    "kp3_scale": 0.70,
    "controller_kp": controller.kp,
    "controller_max_secretion_U_min": controller.max_secretion,
    "controller_basal_rate_U_min": round(basal_rate_correct, 5),
    "fasting_bg_mean": round(fasting_mean, 2),
    "postmeal_peak": round(peak, 2),
    "time_in_range_pct": round(time_in_range, 2),
    "time_above_180_pct": round(time_above_180, 2),
    "time_below_70_pct": round(time_below_70, 2)
}

with open("results/adult001_clinical_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("[SUCCESS] Production validation summary saved to results folder.")

# here i plot the final validated 3 day trajectory
plt.figure(figsize=(15, 6))
plt.plot(results.index, results["BG"], label="Calibrated T2D Blood Glucose", linewidth=2.5, color="red")
plt.plot(results.index, results["CGM"], '--', label="Dexcom CGM Reading", alpha=0.7, color="blue")
meal_times = results[results["CHO"] > 0]
plt.scatter(meal_times.index, meal_times["BG"], color="black", s=80, label="Scheduled Meals", zorder=5)

plt.xlabel("Time over 3 Days (Ideal Routine)")
plt.ylabel("Glucose (mg/dL)")
plt.title("3-Day Continuous Validation: Final Calibrated T2D Baseline Profile")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()