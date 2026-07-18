from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario_gen import RandomScenario
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

# here i load the patient model
patient = T1DPatient.withName("adult#001")
params = patient._params

# here i apply the peripheral insulin resistance modification
params['Vmx'] = params['Vmx'] * 0.40

# here i apply the hepatic insulin resistance modification
params['kp3'] = params['kp3'] * 0.50

# here i save the modified parameters back to the patient object
patient._params = params

# here i convert the baseline basal insulin from pmol/kg/min to U/min
basal_rate_correct = params['u2ss'] * params['BW'] / 6000

# here i setup the sensor, pump, and our fixed pancreas controller
controller = T2DPancreaticController(gb=params['Gb'], basal_rate=basal_rate_correct, init_state=0)
sensor = CGMSensor.withName("Dexcom", seed=1)
pump = InsulinPump.withName("Insulet")

# here i create a random meal scenario with seed=1
start_time = datetime.combine(datetime.now().date(), datetime.min.time())
scenario = RandomScenario(start_time=start_time, seed=1)

# here i build the simulation environment
env = T1DSimEnv(
    patient=patient,
    sensor=sensor,
    pump=pump,
    scenario=scenario
)

# here i define the simulation object over a 3 day window
s = SimObj(
    env=env,
    controller=controller,
    sim_time=timedelta(days=3),
    animate=False,
    path="./results"
)

# here i run the simulation engine
results = sim(s)

# here i save the multi day data to a csv file
results.to_csv("adult001_glucose.csv")

# here i plot the 3 day blood glucose and cgm trajectory
plt.figure(figsize=(15, 6))
plt.plot(results.index, results["BG"], label="T2D Blood Glucose", linewidth=2.5, color="red")
plt.plot(results.index, results["CGM"], '--', label="Dexcom CGM Reading", alpha=0.7, color="blue")

# here i find where meals happened and plot them as markers
meal_times = results[results["CHO"] > 0]
plt.scatter(meal_times.index, meal_times["BG"], color="black", s=80, label="Meals Ingested", zorder=5)

plt.xlabel("Time over 3 Days")
plt.ylabel("Glucose (mg/dL)")
plt.title("3-Day Calibrated T2D Extended Glucose Trajectory")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()