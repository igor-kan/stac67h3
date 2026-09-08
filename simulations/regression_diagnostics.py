"""
STAC67: Multiple Linear Regression Diagnostics
Computes leverage, studentized residuals, Cook's distance, and generates 4-in-1 diagnostic plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def generate_synthetic_data(n=100, seed=42):
    np.random.seed(seed)
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(2, 1.5, n)
    y = 3.0 + 1.5 * X1 - 2.0 * X2 + np.random.normal(0, 1, n)
    
    # Inject one high-leverage outlier
    X1[0] = 5.0
    X2[0] = 8.0
    y[0] = 25.0
    
    X = np.column_stack([X1, X2])
    X = sm.add_constant(X)
    return X, y

def run_diagnostics():
    X, y = generate_synthetic_data()
    model = sm.OLS(y, X).fit()
    
    influence = model.get_influence()
    leverage = influence.hat_matrix_diag
    cooks_d, _ = influence.cooks_distance
    stud_res = influence.resid_studentized_internal

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # 1. Residuals vs Fitted
    axes[0, 0].scatter(model.fittedvalues, model.resid, alpha=0.7, color='#2980b9')
    axes[0, 0].axhline(0, color='red', linestyle='--')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')

    # 2. Normal Q-Q
    sm.qqplot(stud_res, line='45', ax=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot')

    # 3. Scale-Location
    axes[1, 0].scatter(model.fittedvalues, np.sqrt(np.abs(stud_res)), alpha=0.7, color='#27ae60')
    axes[1, 0].set_title('Scale-Location')
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel(r'$\sqrt{|\text{Studentized Residuals}|}$')

    # 4. Residuals vs Leverage
    axes[1, 1].scatter(leverage, stud_res, alpha=0.7, color='#e74c3c')
    axes[1, 1].axhline(0, color='gray', linestyle=':')
    axes[1, 1].set_title("Residuals vs Leverage (Cook's Distance)")
    axes[1, 1].set_xlabel('Leverage ($h_{ii}$)')
    axes[1, 1].set_ylabel('Studentized Residuals')

    plt.tight_layout()
    plt.savefig('regression_diagnostics.png', dpi=200)
    print("Diagnostics plot saved as regression_diagnostics.png")

if __name__ == '__main__':
    run_diagnostics()
