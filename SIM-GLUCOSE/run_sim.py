# Running Sim-Glucose
from datetime import datetime, timedelta
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.sim_engine import SimObj, sim
# Here I create a Patient
patient = T1DPatient.withName("adult#001")
# Creating a sensor, controller and pump
sensor = CGMSensor.withName("Dexcom", seed=1)
pump = InsulinPump.withName("Insulet")
controller = BBController()
# Here I gen a meal scen
start_time = datetime.combine(
    datetime.now().date(),
    datetime.min.time()
)
scenario = RandomScenario(
    start_time=start_time,
    seed=1
)
# Patient Enviroment
env = T1DSimEnv(
    patient=patient,
    sensor=sensor,
    pump=pump,
    scenario=scenario
)
# Obj simulation
s = SimObj(
    env=env,
    controller=controller,
    sim_time=timedelta(days=1),
    animate=False,
    path="./results"
)
results = sim(s)

# print(results) #commented to plot 
# plotting glucose trajec
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(results.index, results["BG"], label="Blood Glucose", linewidth=2)
plt.plot(results.index, results["CGM"], '--', label="CGM Reading", alpha=0.7)

plt.xlabel("Time")
plt.ylabel("Glucose (mg/dL)")
plt.title("24-Hour Glucose Trajectory - adult#001")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()

# plotting meals
plt.figure(figsize=(12,6))
plt.plot(results.index, results["BG"], label="Blood Glucose")
meal_times = results[results["CHO"] > 0]
plt.scatter(
    meal_times.index,
    meal_times["BG"],
    color="red",
    s=80,
    label="Meals"
)
plt.xlabel("Time")
plt.ylabel("Glucose (mg/dL)")
plt.title("Blood Glucose Response to Meals")
plt.grid(True)
plt.legend()
plt.show()
# Saving the Data Set
results.to_csv("adult001_glucose.csv")