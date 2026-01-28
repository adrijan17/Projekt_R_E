import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from _postavke import file_path, feature_columns, visualisaton_path


def load_and_prepare_data(file_path, feature_columns):
    """Load data and prepare features and target."""
    data = pd.read_csv(file_path)
    print(f"Using features: {feature_columns}")
    
    data_clean = data[feature_columns + ['NDVI']].dropna()
    print(f"Dataset size after removing NaN: {len(data_clean)} rows")
    
    X = data_clean[feature_columns]
    y = data_clean['NDVI']
    
    return X, y


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """Train a regression model and evaluate its performance."""
    print(f"\nTreniranje modela ({model_name})...", end="\x1b[1K\r")
    model.fit(X_train, y_train)
    print(f"Model {model_name} je istreniran!")
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train': {
            'r2': r2_score(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': mean_absolute_error(y_train, y_train_pred)
        },
        'test': {
            'r2': r2_score(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mae': mean_absolute_error(y_test, y_test_pred)
        }
    }
    
    return model, y_train_pred, y_test_pred, metrics


def print_metrics(metrics, feature_columns, model):
    """Print model evaluation metrics."""
    print("\n" + "="*60)
    print("Evaluacija modela")
    print("="*60)
    
    print("\nTraining Set Performance:")
    print(f"R² Score: {metrics['train']['r2']:.4f}")
    print(f"RMSE: {metrics['train']['rmse']:.4f}")
    print(f"MAE: {metrics['train']['mae']:.4f}")
    
    print("\nTesting Set Performance:")
    print(f"R² Score: {metrics['test']['r2']:.4f}")
    print(f"RMSE: {metrics['test']['rmse']:.4f}")
    print(f"MAE: {metrics['test']['mae']:.4f}")
    
    # Print coefficients if available (for linear models)
    if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
        print("\nFeature Coefficients:")
        for feature, coef in zip(feature_columns, model.coef_):
            print(f"{feature:20s}: {coef:8.5f}")
        print(f"Intercept: {model.intercept_:.5f}")


def visualize_results(y_test, y_test_pred, metrics, model_name="Model", output_path=None):
    """Create and save visualization of model predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Actual vs Predicted
    axes[0].scatter(y_test, y_test_pred, alpha=0.3, s=10)
    axes[0].plot([y_test.min(), y_test.max()], 
                 [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual NDVI')
    axes[0].set_ylabel('Predicted NDVI')
    axes[0].set_title(f'{model_name} - Test Set: Actual vs Predicted\nR² = {metrics["test"]["r2"]:.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_test - y_test_pred
    axes[1].scatter(y_test_pred, residuals, alpha=0.3, s=10)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted NDVI')
    axes[1].set_ylabel('Residuals (Actual - Predicted)')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {output_path}")
    
    # plt.show()


def run_regression_pipeline(model, model_name, file_path, feature_columns, 
                            test_size=0.2, random_state=16, save_path=None):
    """Complete regression pipeline: load, train, evaluate, visualize."""
    print("="*60)
    print(f"Treniranje modela ({model_name})")
    print("="*60)
    
    # Load and prepare data
    X, y = load_and_prepare_data(file_path, feature_columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # Train and evaluate
    trained_model, y_train_pred, y_test_pred, metrics = train_and_evaluate_model(
        model, X_train, X_test, y_train, y_test, model_name
    )
    
    # Print results
    print_metrics(metrics, feature_columns, trained_model)
    
    # Visualize
    visualize_results(y_test, y_test_pred, metrics, model_name, save_path)
    
    print("\n" + "="*60)
    
    return trained_model, metrics


if __name__ == "__main__":
    # Example: Linear Regression
    model = LinearRegression()
    output_path = os.path.join(visualisaton_path, 'linear_regression_performance.png')
    
    trained_model, metrics = run_regression_pipeline(
        model=model,
        model_name="Linear Regression",
        file_path=file_path,
        feature_columns=feature_columns,
        save_path=output_path
    )