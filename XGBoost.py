import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


data = pd.read_excel('processed data 3.xlsx')
X = data[['Build Direction', 'Deposition Layer Thickness (mm)',
          'Build Orientation (⁰)', 'Post Curing Time (min)']]
y = data['Ultimate Tensile strength'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

xgb = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb.fit(X_train_scaled, y_train)

y_train_pred_xgb = xgb.predict(X_train_scaled)
y_test_pred_xgb = xgb.predict(X_test_scaled)

def calculate_metrics(y_true, y_pred, label):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n--- {label} ---")
    print(f"R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return r2, rmse, mae

# Training and Testing metrics for XGBoost
calculate_metrics(y_train, y_train_pred_xgb, "XGBoost Training")
calculate_metrics(y_test, y_test_pred_xgb, "XGBoost Testing")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred_xgb, color='blue', alpha=0.6, label='XGBoost Test')
plt.scatter(y_train, y_train_pred_xgb, color='blue', alpha=0.2, label='XGBoost Train')

combined_actual = np.concatenate([y_train, y_test])
plt.plot([min(combined_actual), max(combined_actual)],
         [min(combined_actual), max(combined_actual)],
         color='black', linestyle='--', linewidth=2, label='Ideal Fit')

plt.xlabel('Actual Ultimate Tensile Strength (MPa)')
plt.ylabel('Predicted Ultimate Tensile Strength (MPa)')
plt.title('XGBoost: Actual vs Predicted UTS')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('XGBoost_Actual_vs_Predicted.png', dpi=300, bbox_inches='tight')
plt.show()

output_train = pd.DataFrame({
    'Actual': y_train,
    'XGBoost_Predicted': y_train_pred_xgb
})
output_test = pd.DataFrame({
    'Actual': y_test,
    'XGBoost_Predicted': y_test_pred_xgb
})
output = pd.concat([output_train, output_test])
output.to_excel('Actual_vs_Predicted_XGBoost.xlsx', index=False)
