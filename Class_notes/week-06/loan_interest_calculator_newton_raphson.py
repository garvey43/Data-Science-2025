
# Loan Interest Rate Calculator using Newton-Raphson Method

# Problem:
# Given: 
#   - Loan amount (Present Value, PV)
#   - Monthly payment (PMT)
#   - Loan term in months (n)
# Goal: 
#   - Estimate the monthly interest rate (r)

# Newton-Raphson Method:
# We solve the equation:
#   f(r) = PV - PMT * (1 - (1 + r)^-n) / r = 0

# Import necessary library
import matplotlib.pyplot as plt

def f(r, PMT, PV, n):
    return PV - PMT * (1 - (1 + r) ** -n) / r

def f_prime(r, PMT, n):
    # Derivative of f with respect to r
    return (PMT * ((1 + r)**(-n) * (n / (1 + r)) - (1 - (1 + r)**(-n)) / r)) / r

def newton_raphson(PV, PMT, n, initial_guess=0.05, epsilon=1e-6, max_iter=100):
    r = initial_guess
    for i in range(max_iter):
        f_val = f(r, PMT, PV, n)
        f_deriv = f_prime(r, PMT, n)
        if f_deriv == 0:
            print("Zero derivative. No solution found.")
            return None
        r_new = r - f_val / f_deriv
        if abs(r_new - r) < epsilon:
            return r_new, i + 1
        r = r_new
    print("Exceeded maximum iterations.")
    return None

# Example usage:
PV = 500000  # Loan amount in Ksh
PMT = 15000  # Monthly payment in Ksh
n = 48       # Loan term in months (4 years)

estimated_r, steps = newton_raphson(PV, PMT, n)

if estimated_r:
    annual_rate_percent = estimated_r * 12 * 100
    print(f"Estimated monthly interest rate: {estimated_r:.6f}")
    print(f"Estimated annual interest rate: {annual_rate_percent:.2f}%")
    print(f"Converged in {steps} steps.")
else:
    print("Could not estimate the interest rate.")

# Plotting f(r) to visualize the function
import numpy as np
r_values = np.linspace(0.001, 0.3, 1000)
f_values = [f(r, PMT, PV, n) for r in r_values]

plt.figure(figsize=(10, 5))
plt.plot(r_values, f_values, label="f(r)")
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Monthly Interest Rate (r)")
plt.ylabel("f(r)")
plt.title("Loan Equation Function f(r)")
plt.grid(True)
plt.legend()
plt.show()
