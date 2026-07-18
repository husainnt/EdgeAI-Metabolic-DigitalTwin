import os
import itertools
import json
from datetime import datetime, timedelta
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
        super().__init__(init_state)
        self.gb = gb
        self.basal_rate = basal_rate
        self.kp = 0.0005 
        self.max_secretion = 0.05 

    def policy(self, observation, reward, done, **info):
        bg = observation.CGM
        if bg <= self.gb:
            secretion = self.basal_rate
        else:
            excess_glucose = bg - self.gb
            secretion = self.basal_rate + (self.kp * excess_glucose)
            
        secretion = min(secretion, self.max_secretion)
        return Action(basal=secretion, bolus=0)

    def reset(self):
        pass

# here i define a reusable function to run the simulation loop
def run_simulation(vmx_scale, kp3_scale, sim_days=3, seed=1):
    patient = T1DPatient.withName("adult#001")
    params = patient._params

    # here i apply the custom scales to insulin resistance parameters
    params['Vmx'] = params['Vmx'] * vmx_scale
    params['kp3'] = params['kp3'] * kp3_scale
    patient._params = params

    # here i convert the baseline basal insulin to U/min
    basal_rate_correct = params['u2ss'] * params['BW'] / 6000

    # here i setup the controller and sensor nodes
    controller = T2DPancreaticController(gb=params['Gb'], basal_rate=basal_rate_correct, init_state=0)
    sensor = CGMSensor.withName("Dexcom", seed=seed)
    pump = InsulinPump.withName("Insulet")

    # here i set the midnight start time
    start_time = datetime.combine(datetime.now().date(), datetime.min.time())

    # here i build a fixed 3-day meal timeline with South Asian carb loads
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
    
    return sim(s)

# ensure the results directory exists
os.makedirs("results", exist_ok=True)

# here i define the new extended test matrix sweep bounds
vmx_scales = [0.60, 0.65, 0.70, 0.75]
kp3_scales = [0.55, 0.60, 0.65, 0.70]

sweep_results = {}

print("Running extended parameter optimization sweep...")
for vmx, kp3 in itertools.product(vmx_scales, kp3_scales):
    key = f"Vmx_{vmx:.2f}_kp3_{kp3:.2f}"
    
    results = run_simulation(vmx_scale=vmx, kp3_scale=kp3, sim_days=3, seed=1)
    
    fasting_mean = results[results["CHO"] == 0]["BG"].mean()
    peak = results["BG"].max()
    time_in_range = ((results["BG"] >= 70) & (results["BG"] <= 180)).mean() * 100
    time_above_180 = (results["BG"] > 180).mean() * 100
    time_below_70 = (results["BG"] < 70).mean() * 100
    
    print(f"Vmx scale: {vmx:.2f} | kp3 scale: {kp3:.2f} -> Fasting Mean: {fasting_mean:.1f}, Peak: {peak:.1f}, TIR%: {time_in_range:.1f}, Hypo%: {time_below_70:.1f}")
    
    sweep_results[key] = {
        "vmx_scale": vmx,
        "kp3_scale": kp3,
        "fasting_bg_mean": round(fasting_mean, 2),
        "postmeal_peak": round(peak, 2),
        "time_in_range_pct": round(time_in_range, 2),
        "time_above_180_pct": round(time_above_180, 2),
        "time_below_70_pct": round(time_below_70, 2)
    }

with open("results/t2d_parameter_sweep_extended_summary.json", "w") as f:
    json.dump(sweep_results, f, indent=2)

print("\n[SUCCESS] Extended sweep complete. Summary saved to results/t2d_parameter_sweep_extended_summary.json")