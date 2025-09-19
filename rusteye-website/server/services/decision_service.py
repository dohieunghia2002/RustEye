import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from ortools.sat.python import cp_model

def fuzzy_and_cp_decision(damage_percentage: float):
    damage_size = ctrl.Antecedent(np.arange(0, 101, 1), 'damage_size')
    damage_level_fuzzy = ctrl.Consequent(np.arange(0, 101, 1), 'damage_level')

    damage_size['Ri0'] = fuzz.trimf(damage_size.universe, [0, 0, 5])
    damage_size['Ri1'] = fuzz.trimf(damage_size.universe, [0, 5, 15])
    damage_size['Ri2'] = fuzz.trimf(damage_size.universe, [10, 20, 30])
    damage_size['Ri3'] = fuzz.trimf(damage_size.universe, [25, 40, 55])
    damage_size['Ri4'] = fuzz.trimf(damage_size.universe, [50, 70, 85])
    damage_size['Ri5'] = fuzz.trimf(damage_size.universe, [80, 100, 100])

    damage_level_fuzzy.automf(names=['Excellent', 'Good', 'Fair', 'Poor', 'Severe', 'Critical'])

    rules = [
        ctrl.Rule(damage_size['Ri0'], damage_level_fuzzy['Excellent']),
        ctrl.Rule(damage_size['Ri1'], damage_level_fuzzy['Good']),
        ctrl.Rule(damage_size['Ri2'], damage_level_fuzzy['Fair']),
        ctrl.Rule(damage_size['Ri3'], damage_level_fuzzy['Poor']),
        ctrl.Rule(damage_size['Ri4'], damage_level_fuzzy['Severe']),
        ctrl.Rule(damage_size['Ri5'], damage_level_fuzzy['Critical']),
    ]

    sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
    sim.input['damage_size'] = damage_percentage
    sim.compute()
    fuzzy_level = sim.output['damage_level']

    model_cp = cp_model.CpModel()
    damage_var = model_cp.NewIntVar(0, 5, 'damage_level')

    if damage_percentage < 5:
        model_cp.Add(damage_var == 0)
    elif damage_percentage < 15:
        model_cp.Add(damage_var == 1)
    elif damage_percentage < 30:
        model_cp.Add(damage_var == 2)
    elif damage_percentage < 50:
        model_cp.Add(damage_var == 3)
    elif damage_percentage < 70:
        model_cp.Add(damage_var == 4)
    else:
        model_cp.Add(damage_var == 5)

    solver = cp_model.CpSolver()
    solver.Solve(model_cp)
    severity = solver.Value(damage_var)

    actions = [
        "Không cần hành động",
        "Giám sát và theo dõi",
        "Thực hiện bảo dưỡng nhỏ",
        "Bảo dưỡng khẩn cấp",
        "Sửa chữa hoặc thay thế",
        "Thay thế toàn bộ",
    ]
    return fuzzy_level, severity, actions[severity]
