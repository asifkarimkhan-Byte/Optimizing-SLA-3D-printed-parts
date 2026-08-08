#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import uniform_filter1d
from geneticalgorithm import geneticalgorithm as ga
from xgboost import XGBRegressor

random.seed(42)
np.random.seed(42)

data = pd.read_excel('GA data.xlsx')
X = data[['Specimen Direction', 'Deposition Layer Thickness (mm)', 'Orientation (⁰)', 'Post Processing (min)']].values
y = data['Average UTS (MPa)'].values.reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y).flatten()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

xgb = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    learning_rate=0.2,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb.fit(X_train, y_train)

y_train_pred_xgb = xgb.predict(X_train)
y_test_pred_xgb = xgb.predict(X_test)

def fitness_function(x):
    x_reshaped = np.array(x).reshape(1, -1)
    x_scaled = scaler_X.transform(x_reshaped)
    pred_scaled = xgb.predict(x_scaled).reshape(-1, 1)
    pred_uts = scaler_y.inverse_transform(pred_scaled)[0][0]
    noise = np.random.normal(0, 0.01)
    return -(pred_uts + noise)

varbound = np.array([
    [-1, 1],
    [0.1, 0.2],
    [0, 90],
    [15, 45]
])

algorithm_param = {
    'max_num_iteration': 500,
    'population_size': 100,
    'mutation_probability': 0.2,
    'elit_ratio': 0.4,
    'crossover_probability': 0.85,
    'parents_portion': 0.6,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': None
}

model = ga(
    function=fitness_function,
    dimension=4,
    variable_type='real',
    variable_boundaries=varbound,
    algorithm_parameters=algorithm_param
)

model.run()

best_solution = model.output_dict['variable']
best_fitness = -model.output_dict['function']
convergence_curve = -np.array(model.report)
smoothed_curve = uniform_filter1d(convergence_curve, size=10)

print("\nBest process parameters (optimized):")
print(f"Specimen Direction: {round(best_solution[0])}")
print(f"Deposition Layer Thickness (mm): {best_solution[1]:.3f}")
print(f"Orientation (⁰): {round(best_solution[2])}")
print(f"Post Processing (min): {round(best_solution[3])}")
print(f"\nPredicted Ultimate Tensile Strength (MPa): {best_fitness:.2f}")

plt.figure(figsize=(8, 5))
plt.plot(smoothed_curve, color='black')
plt.xlabel('Epoch')
plt.ylabel('Ultimate Tensile Strength(MPa)')
plt.title('Optimization of UTS using XGBoost + GA')
plt.grid(True)
plt.tight_layout()
plt.savefig('optimized_convergence_curve_xgb.png', dpi=300)

results_df = pd.DataFrame({
    'Generation': np.arange(len(convergence_curve)),
    'Predicted_UTS': convergence_curve,
    'Smoothed_UTS': smoothed_curve
})

best_params_df = pd.DataFrame({
    'Parameter': ['Specimen Direction', 'Deposition Layer Thickness (mm)', 'Orientation (⁰)', 'Post Processing (min)'],
    'Optimized Value': [round(best_solution[0]), round(best_solution[1], 3), round(best_solution[2]), round(best_solution[3])]
})

with pd.ExcelWriter('optimization_results_xgb.xlsx') as writer:
    results_df.to_excel(writer, sheet_name='Convergence History', index=False)
    best_params_df.to_excel(writer, sheet_name='Optimal Parameters', index=False)

print("\nResults saved to 'optimization_results_xgb.xlsx'")
plt.show()

