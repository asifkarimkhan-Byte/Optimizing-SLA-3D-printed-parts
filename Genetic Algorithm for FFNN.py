#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import torch
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import uniform_filter1d
from geneticalgorithm import geneticalgorithm as ga

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

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class BaseRegressorA(nn.Module):
    def __init__(self, input_dim):
        super(BaseRegressorA, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)

class BaseRegressorB(nn.Module):
    def __init__(self, input_dim):
        super(BaseRegressorB, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

class BaseRegressorC(nn.Module):
    def __init__(self, input_dim):
        super(BaseRegressorC, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

def train_model(model, X, y, epochs=500, lr=0.0005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

input_dim = X_train.shape[1]
base_models = [BaseRegressorA(input_dim), BaseRegressorB(input_dim), BaseRegressorC(input_dim)]
for model in base_models:
    train_model(model, X_train_tensor, y_train_tensor)

with torch.no_grad():
    train_preds = torch.cat([m(X_train_tensor) for m in base_models], dim=1)
    test_preds  = torch.cat([m(X_test_tensor) for m in base_models], dim=1)

class MetaRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MetaRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

meta_model = MetaRegressor(len(base_models))
train_model(meta_model, train_preds, y_train_tensor)

def fitness_function(x):
    x_reshaped = np.array(x).reshape(1, -1)
    x_scaled = scaler_X.transform(x_reshaped)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        base_preds = torch.cat([model(x_tensor) for model in base_models], dim=1)
        meta_output = meta_model(base_preds)
        pred_scaled = meta_output.numpy().reshape(-1, 1)
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
    'max_num_iteration': 300,
    'population_size': 100,
    'mutation_probability': 0.25,
    'elit_ratio': 0.1,
    'crossover_probability': 0.9,
    'parents_portion': 0.4,
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
plt.title('Optimization of UTS using Deep Stacked Ensemble + GA')
plt.grid(True)
plt.tight_layout()
plt.savefig('optimized_convergence_curve_dl.png', dpi=300)

results_df = pd.DataFrame({
    'Generation': np.arange(len(convergence_curve)),
    'Predicted_UTS': convergence_curve,
    'Smoothed_UTS': smoothed_curve
})

best_params_df = pd.DataFrame({
    'Parameter': ['Specimen Direction', 'Deposition Layer Thickness (mm)', 'Orientation (⁰)', 'Post Processing (min)'],
    'Optimized Value': [round(best_solution[0]), round(best_solution[1], 3), round(best_solution[2]), round(best_solution[3])]
})

with pd.ExcelWriter('optimization_results_dl.xlsx') as writer:
    results_df.to_excel(writer, sheet_name='Convergence History', index=False)
    best_params_df.to_excel(writer, sheet_name='Optimal Parameters', index=False)

print("\nResults saved to 'optimization_results_dl.xlsx'")
plt.show()

