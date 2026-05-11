import sys
import numpy as np
import matplotlib.pyplot as plt
import math

# set seed so output doesn't change every run
np.random.seed(42)
N = int(sys.argv[1])

# Puchku's party problem - we want the arrival time of the 4th friend out of 10
# Since each friend arrives uniformly between 6-7pm, we model arrival as Uniform[0,1]
# The 4th order statistic of 10 uniform samples gives us X (when the game starts)
data = np.random.uniform(0, 1, size=(N, 10))

# Sort each row so columns are in ascending order - this gives us order statistics
# Column 0 = earliest arrival, Column 9 = latest arrival
data.sort(axis=1)

# 4th friend arrival = 4th order statistic, which is index 3 (0-indexed)
X = data[:, 3]

print(f"number of samples generated: {N}")
print(f"sample mean of X: {X.mean():.4f}")
# Theoretical mean of k-th order stat of n Uniform(0,1) samples = k/(n+1)
# Here k=4, n=10, so mean = 4/11
print(f"theoretical mean (4/11): {4/11:.4f}")

# fa is the pdf of the 4th order statistic of 10 Uniform(0,1) samples
# General formula for k-th order stat pdf: n!/(k-1)!(n-k)! * x^(k-1) * (1-x)^(n-k)
# For k=4, n=10: 10! / (3! * 6!) * x^3 * (1-x)^6
def fa(x):
    c = math.factorial(10) / (math.factorial(3) * math.factorial(6))
    return c * (x**3) * ((1-x)**6)

# fb has an extra x factor and one less (1-x) factor compared to fa
# This would correspond to the 5th order statistic, not the 4th
def fb(x):
    c = math.factorial(10) / (math.factorial(4) * math.factorial(5))
    return c * (x**4) * ((1-x)**5)

x_vals = np.linspace(0, 1, 300)
fa_vals = [fa(x) for x in x_vals]
fb_vals = [fb(x) for x in x_vals]

# Plot histogram of X samples and overlay both candidate pdfs
# density=True normalizes the histogram so area = 1, comparable with pdf curves
plt.figure(figsize=(8, 5))
plt.hist(X, bins=100, density=True, label='histogram of X', color='steelblue', edgecolor='white', alpha=0.7)
plt.plot(x_vals, fa_vals, 'r-', linewidth=2, label='fa(x)')
plt.plot(x_vals, fb_vals, 'g--', linewidth=2, label='fb(x)')
plt.xlabel('x')
plt.ylabel('density')
plt.title('4th order statistic of 10 uniform samples')
plt.legend()
plt.tight_layout()
plt.savefig('ordered_stats_ee25btech11011.png', dpi=150)
plt.close()
print("plot saved: ordered_stats_ee25btech11011.png")

# Quantitatively decide which pdf the histogram is closer to
# Compute MSE between histogram bin heights and each pdf evaluated at bin centers
counts, edges = np.histogram(X, bins=100, density=True)
centers = (edges[:-1] + edges[1:]) / 2

mse_a = np.mean((counts - np.array([fa(x) for x in centers]))**2)
mse_b = np.mean((counts - np.array([fb(x) for x in centers]))**2)

print(f"mse with fa: {mse_a:.4f}")
print(f"mse with fb: {mse_b:.4f}")

# Lower MSE means the histogram is visually closer to that pdf
if mse_a < mse_b:
    print("histogram matches fa, so printing: a")
    print("a")
else:
    print("histogram matches fb, so printing: b")
    print("b")