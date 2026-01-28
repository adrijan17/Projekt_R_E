import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from ml_model_pipeline import run_regression_pipeline
from _postavke import file_path, feature_columns, visualisaton_path, output_data_path


print("="*60)
print("Testiranje Više Regresijskih Modela")
print("="*60)

results = {}


# 1. Linear Regression
print("\n" + "="*60)
print("1/4 - Linear Regression")
print("="*60)
model = LinearRegression()
trained_linear_model, metrics = run_regression_pipeline(
    model=model,
    model_name="Linear Regression",
    file_path=file_path,
    feature_columns=feature_columns,
    save_path=os.path.join(visualisaton_path, 'linear_regression_performance.png')
)
results['Linear Regression'] = metrics


# 2. Ridge Regression
print("\n" + "="*60)
print("2/4 - Ridge Regression")
print("="*60)
model = Ridge(alpha=1.0, random_state=16)
trained_model, metrics = run_regression_pipeline(
    model=model,
    model_name="Ridge Regression",
    file_path=file_path,
    feature_columns=feature_columns,
    save_path=os.path.join(visualisaton_path, 'ridge_regression_performance.png')
)
results['Ridge Regression'] = metrics


# 3. Lasso Regression
print("\n" + "="*60)
print("3/4 - Lasso Regression")
print("="*60)
model = Lasso(alpha=0.001, random_state=16)
trained_lasso_model, metrics = run_regression_pipeline(
    model=model,
    model_name="Lasso Regression",
    file_path=file_path,
    feature_columns=feature_columns,
    save_path=os.path.join(visualisaton_path, 'lasso_regression_performance.png')
)
results['Lasso Regression'] = metrics

# Analyze Lasso feature selection
# print("\n" + "-"*60)
# print("Lasso Feature Selection Analysis:")
# print("-"*60)
# lasso_coefs = trained_lasso_model.coef_
# zeroed_features = []
# non_zero_features = []

# for feature, coef in zip(feature_columns, lasso_coefs):
#     if abs(coef) < 1e-10:  # Essentially zero
#         zeroed_features.append(feature)
#         print(f"{feature:20s}: {coef:12.8f} (ZEROED OUT)")
#     else:
#         non_zero_features.append((feature, coef))
#         print(f"{feature:20s}: {coef:12.8f}")

# print(f"\nSummary:")
# print(f"  - Features kept: {len(non_zero_features)}")
# print(f"  - Features zeroed out: {len(zeroed_features)}")
# if zeroed_features:
#     print(f"  - Zeroed features: {', '.join(zeroed_features)}")
# print("-"*60)


# 4. Random Forest
print("\n" + "="*60)
print("4/4 - Random Forest Regressor")
print("="*60)
model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=16, n_jobs=-1)
trained_model, metrics = run_regression_pipeline(
    model=model,
    model_name="Random Forest",
    file_path=file_path,
    feature_columns=feature_columns,
    save_path=os.path.join(visualisaton_path, 'random_forest_performance.png')
)
results['Random Forest'] = metrics


# Summary comparison
print("\n" + "="*60)
print("Usporedba Modela")
print("="*60)
print("\nTest Set Performance Comparison:")
print(f"{'Model':<25} {'R² Score':<12} {'RMSE':<12} {'MAE':<12}")
print("-" * 60)

for model_name, metrics in results.items():
    print(f"{model_name:<25} {metrics['test']['r2']:<12.4f} {metrics['test']['rmse']:<12.4f} {metrics['test']['mae']:<12.4f}")

# Find best model
best_model = max(results.items(), key=lambda x: x[1]['test']['r2'])
print("\n" + "="*60)
print(f"Najbolji model po R² score: {best_model[0]}")
print(f"R² Score: {best_model[1]['test']['r2']:.4f}")
print("="*60)


# Save results to CSV
print("\n" + "="*60)
print("Spremanje rezultata u CSV")
print("="*60)

# Prepare data for CSV
comparison_data = []
for model_name, metrics in results.items():
    comparison_data.append({
        'Model': model_name,
        'Test_R2': metrics['test']['r2'],
        'Test_RMSE': metrics['test']['rmse'],
        'Test_MAE': metrics['test']['mae'],
        'Train_R2': metrics['train']['r2'],
        'Train_RMSE': metrics['train']['rmse'],
        'Train_MAE': metrics['train']['mae']
    })

df_results = pd.DataFrame(comparison_data)
csv_path = os.path.join(output_data_path, 'model_comparison.csv')
df_results.to_csv(csv_path, index=False)
print(f"Rezultati spremljeni u: {csv_path}")


# Visualize model comparison
print("\n" + "="*60)
print("Kreiranje vizualizacija usporedbe modela")
print("="*60)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Comparison - Performance Metrics', fontsize=16, fontweight='bold')

model_names = list(results.keys())
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

# 1. R² Score Comparison (Test and Train)
ax1 = axes[0, 0]
x_pos = np.arange(len(model_names))
width = 0.35
test_r2 = [results[m]['test']['r2'] for m in model_names]
train_r2 = [results[m]['train']['r2'] for m in model_names]

bars1 = ax1.bar(x_pos - width/2, test_r2, width, label='Test', color=colors, alpha=0.8)
bars2 = ax1.bar(x_pos + width/2, train_r2, width, label='Train', color=colors, alpha=0.5)

ax1.set_xlabel('Model', fontweight='bold')
ax1.set_ylabel('R² Score', fontweight='bold')
ax1.set_title('R² Score Comparison', fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(model_names, rotation=15, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)

# 2. RMSE Comparison
ax2 = axes[0, 1]
test_rmse = [results[m]['test']['rmse'] for m in model_names]
train_rmse = [results[m]['train']['rmse'] for m in model_names]

bars1 = ax2.bar(x_pos - width/2, test_rmse, width, label='Test', color=colors, alpha=0.8)
bars2 = ax2.bar(x_pos + width/2, train_rmse, width, label='Train', color=colors, alpha=0.5)

ax2.set_xlabel('Model', fontweight='bold')
ax2.set_ylabel('RMSE', fontweight='bold')
ax2.set_title('RMSE Comparison (Lower is Better)', fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(model_names, rotation=15, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

# 3. MAE Comparison
ax3 = axes[1, 0]
test_mae = [results[m]['test']['mae'] for m in model_names]
train_mae = [results[m]['train']['mae'] for m in model_names]

bars1 = ax3.bar(x_pos - width/2, test_mae, width, label='Test', color=colors, alpha=0.8)
bars2 = ax3.bar(x_pos + width/2, train_mae, width, label='Train', color=colors, alpha=0.5)

ax3.set_xlabel('Model', fontweight='bold')
ax3.set_ylabel('MAE', fontweight='bold')
ax3.set_title('MAE Comparison (Lower is Better)', fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(model_names, rotation=15, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

# 4. Overfitting Check (Train-Test Gap)
ax4 = axes[1, 1]
# Calculate the gap between train and test R² performance
# Positive gap means train R² > test R² (overfitting)
# Values close to 0 indicate good generalization
r2_gap = [results[m]['train']['r2'] - results[m]['test']['r2'] for m in model_names]

bars = ax4.bar(x_pos, r2_gap, color=colors, alpha=0.8)

ax4.set_xlabel('Model', fontweight='bold')
ax4.set_ylabel('R² Gap (Train - Test)', fontweight='bold')
ax4.set_title('Overfitting Check - R² Score Gap', fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(model_names, rotation=15, ha='right')
ax4.grid(axis='y', alpha=0.3)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax4.axhline(y=0.02, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Excellent gap')
ax4.axhline(y=-0.02, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
ax4.legend()

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)

plt.tight_layout()
comparison_plot_path = os.path.join(visualisaton_path, 'model_comparison.png')
plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
print(f"Vizualizacija spremljena u: {comparison_plot_path}")
plt.close()

print("\n" + "="*60)
print("Završeno!")
print("="*60)
