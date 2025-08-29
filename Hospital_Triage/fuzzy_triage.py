import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define fuzzy variables
pain = ctrl.Antecedent(np.arange(0, 11, 1), 'pain')
bp = ctrl.Antecedent(np.arange(50, 181, 1), 'blood_pressure')
ur = ctrl.Antecedent(np.arange(10, 41, 1), 'ur_heart_rate')  # ur = urgency heart rate

triage = ctrl.Consequent(np.arange(0, 11, 1), 'triage')

# Membership functions for pain
pain['low'] = fuzz.trimf(pain.universe, [0, 0, 4])
pain['medium'] = fuzz.trimf(pain.universe, [2, 5, 8])
pain['high'] = fuzz.trimf(pain.universe, [6, 10, 10])

# Membership functions for blood pressure
bp['low'] = fuzz.trimf(bp.universe, [50, 50, 90])
bp['normal'] = fuzz.trimf(bp.universe, [80, 120, 140])
bp['high'] = fuzz.trimf(bp.universe, [130, 180, 180])

# Membership functions for heart rate
ur['low'] = fuzz.trimf(ur.universe, [10, 10, 18])
ur['normal'] = fuzz.trimf(ur.universe, [16, 22, 28])
ur['high'] = fuzz.trimf(ur.universe, [26, 40, 40])

# Membership functions for triage
triage['low'] = fuzz.trimf(triage.universe, [0, 0, 3])
triage['medium'] = fuzz.trimf(triage.universe, [2, 5, 8])
triage['high'] = fuzz.trimf(triage.universe, [7, 10, 10])

# Define fuzzy rules
rule1 = ctrl.Rule(pain['high'] | bp['low'] | ur['high'], triage['high'])
rule2 = ctrl.Rule(pain['medium'] & bp['normal'] & ur['normal'], triage['medium'])
rule3 = ctrl.Rule(pain['low'] & bp['normal'] & ur['low'], triage['low'])

# Control system
triage_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
triage_sim = ctrl.ControlSystemSimulation(triage_ctrl)

# Function to compute triage score
def get_triage(pain_input, bp_input, hr_input):
    triage_sim.input['pain'] = pain_input
    triage_sim.input['blood_pressure'] = bp_input
    triage_sim.input['ur_heart_rate'] = hr_input

    triage_sim.compute()
    return triage_sim.output['triage']
