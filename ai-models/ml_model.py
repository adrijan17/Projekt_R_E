import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from postavke import file_path, feature_columns


data = pd.read_csv(file_path)

print("="*60)
print("Treniranje modela (linearna regresija)")
print("="*60)


print(f"\nUsing features: {feature_columns}")


data_clean = data[feature_columns + ['NDVI']].dropna()
print(f"\nDataset size after removing NaN: {len(data_clean)} rows")


X = data_clean[feature_columns]
y = data_clean['NDVI']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=16
)


print(f"\nTraining set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")


print()
print("Treniranje modela...", end="\x1b[1K\r")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model je istreniran!")


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


print("\n" + "="*60)
print("Evaluacija modela")
print("="*60)


print("\nTraining Set Performance:")
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
print(f"R² Score: {train_r2:.4f}")
print(f"RMSE: {train_rmse:.4f}")
print(f"MAE: {train_mae:.4f}")


print("\nTesting Set Performance:")
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
print(f"R² Score: {test_r2:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"MAE: {test_mae:.4f}")


print("\nFeature Coefficients:")
for feature, coef in zip(feature_columns, model.coef_):
    print(f"{feature:20s}: {coef:8.5f}")
print(f"Intercept: {model.intercept_:.5f}")


# Vizualizacija
fig, axes = plt.subplots(1, 2, figsize=(14, 5))


axes[0].scatter(y_test, y_test_pred, alpha=0.3, s=10)
axes[0].plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual NDVI')
axes[0].set_ylabel('Predicted NDVI')
axes[0].set_title(f'Test Set: Actual vs Predicted\nR² = {test_r2:.4f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)


residuals = y_test - y_test_pred
axes[1].scatter(y_test_pred, residuals, alpha=0.3, s=10)
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted NDVI')
axes[1].set_ylabel('Residuals (Actual - Predicted)')
axes[1].set_title('Residual Plot')
axes[1].grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
print("\nSaved: model_performance.png")
plt.show()

print("\n" + "="*60)