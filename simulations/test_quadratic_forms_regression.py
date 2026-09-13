"""
Numerical Verification of Quadratic Forms and Matrix Decompositions in Regression
STAC67: Multiple Linear Regression

Tests:
1. Symmetry and Idempotence of Projection Hat Matrix H
2. Positive Semidefiniteness and Binary Spectrum {0, 1} of H
3. Strict Positive Definiteness of Gram Matrix X^T X via Cholesky Factorization
4. Equivalence of Cholesky Forward/Back Substitution to Normal Equation Inversion
5. Craig's Theorem Orthogonality Condition: A_R @ A_E == 0 (Independence of SSR & SSE)
6. Expected Value of Error Quadratic Form: E[SSE] == (n - p) * sigma^2
7. SVD of Design Matrix and Conditioning
"""

import numpy as np
import scipy.linalg as la


def test_projection_matrix_properties():
    np.random.seed(123)
    n, p = 80, 5
    X = np.hstack([np.ones((n, 1)), np.random.randn(n, p - 1)])

    XtX = X.T @ X
    H = X @ np.linalg.inv(XtX) @ X.T

    # 1. Symmetry
    assert np.allclose(H, H.T)

    # 2. Idempotence: H^2 == H
    assert np.allclose(H @ H, H)

    # 3. Trace == p
    assert np.isclose(np.trace(H), p)

    # 4. Eigenvalue Spectrum: all eigenvalues in {0, 1}
    eigvals = np.linalg.eigvalsh(H)
    assert np.all(eigvals >= -1e-12) # Positive semidefinite
    num_ones = np.sum(np.isclose(eigvals, 1.0, atol=1e-5))
    num_zeros = np.sum(np.isclose(eigvals, 0.0, atol=1e-5))
    assert num_ones == p
    assert num_zeros == n - p


def test_cholesky_solver_vs_inversion():
    np.random.seed(234)
    n, p = 120, 6
    X = np.hstack([np.ones((n, 1)), np.random.randn(n, p - 1)])
    true_beta = np.array([1.5, -2.0, 0.8, 3.4, -1.1, 0.5])
    y = X @ true_beta + np.random.normal(0, 1.0, size=n)

    XtX = X.T @ X
    Xty = X.T @ y

    # Cholesky approach
    L = np.linalg.cholesky(XtX)
    z = la.solve_triangular(L, Xty, lower=True)
    beta_cholesky = la.solve_triangular(L.T, z, lower=False)

    # Inversion approach
    beta_inv = np.linalg.inv(XtX) @ Xty

    assert np.allclose(beta_cholesky, beta_inv)


def test_craigs_theorem_independence_condition():
    np.random.seed(345)
    n, p = 60, 4
    X = np.hstack([np.ones((n, 1)), np.random.randn(n, p - 1)])

    XtX = X.T @ X
    H = X @ np.linalg.inv(XtX) @ X.T
    J = np.ones((n, n))

    A_R = H - (1.0 / n) * J
    A_E = np.eye(n) - H

    # Craig's condition for spherical normal errors: A_R @ A_E == 0
    product = A_R @ A_E
    assert np.allclose(product, 0.0, atol=1e-14)


def test_quadratic_form_expected_values():
    np.random.seed(456)
    n, p = 50, 3
    sigma = 2.0
    num_mc_trials = 2000

    X = np.hstack([np.ones((n, 1)), np.random.randn(n, p - 1)])
    XtX = X.T @ X
    H = X @ np.linalg.inv(XtX) @ X.T
    A_E = np.eye(n) - H
    true_beta = np.array([3.0, -1.0, 2.0])

    sse_samples = []
    for _ in range(num_mc_trials):
        y = X @ true_beta + np.random.normal(0, sigma, size=n)
        sse = y.T @ A_E @ y
        sse_samples.append(sse)

    expected_sse_theoretical = (n - p) * (sigma ** 2)
    empirical_mean_sse = np.mean(sse_samples)

    rel_error = abs(empirical_mean_sse - expected_sse_theoretical) / expected_sse_theoretical
    print(f"Theoretical E[SSE]: {expected_sse_theoretical:.2f}, Empirical E[SSE]: {empirical_mean_sse:.2f} (Rel Err: {rel_error:.3%})")
    assert rel_error < 0.03  # Within 3% of Monte Carlo theoretical expectation


def test_svd_condition_numbers():
    np.random.seed(567)
    n, p = 100, 4
    X = np.hstack([np.ones((n, 1)), np.random.randn(n, p - 1)])
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    kappa = s[0] / s[-1]
    assert kappa > 1.0
    # Reconstruct Gram matrix via SVD: X^T X = V S^2 V^T
    XtX = X.T @ X
    XtX_svd = Vt.T @ np.diag(s ** 2) @ Vt
    assert np.allclose(XtX, XtX_svd)


if __name__ == "__main__":
    test_projection_matrix_properties()
    test_cholesky_solver_vs_inversion()
    test_craigs_theorem_independence_condition()
    test_quadratic_form_expected_values()
    test_svd_condition_numbers()
    print("All regression quadratic forms and matrix decomposition tests passed successfully!")
