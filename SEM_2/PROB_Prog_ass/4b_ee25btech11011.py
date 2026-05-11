import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

np.random.seed(42)

# usage: python 4b_ee25btech11011.py coin_data.txt 20
data_file = sys.argv[1]
n = int(sys.argv[2])  # batch size - number of tosses per batch

# read the coin toss data file
# each space-separated integer = number of heads observed in that batch
with open(data_file, 'r') as f:
    heads = list(map(int, f.read().split()))

m = len(heads)  # total number of batches
print(f"number of batches: {m}")
print(f"batch size (n): {n}")
print(f"heads in each batch: {heads}")

# BAYESIAN SETUP
# We want to estimate the bias Theta of the coin.
# Prior: Theta ~ Uniform(0,1) = Beta(alpha=1, beta=1)
#   => f(theta) = 1 for theta in [0,1]
# Likelihood: Given Theta=theta, number of heads Y in n tosses:
#   P(Y=k | Theta=theta) = C(n,k) * theta^k * (1-theta)^(n-k)
#   i.e., Y | Theta ~ Binomial(n, theta)
#
# Posterior using Bayes theorem:
#   f(theta | Y=k) ∝ f(theta) * P(Y=k | theta)
#                  ∝ theta^(alpha-1) * (1-theta)^(beta-1)   <- prior
#                    * theta^k * (1-theta)^(n-k)            <- likelihood
#                  = theta^(alpha+k-1) * (1-theta)^(beta+n-k-1)
# This is exactly Beta(alpha + k, beta + n - k)
# So the update rule is:
#   alpha_new = alpha_old + k       (add heads observed)
#   beta_new  = beta_old  + n - k  (add tails observed)

alpha = 1       # initial alpha for Uniform = Beta(1,1)
beta_param = 1  # initial beta

x_vals = np.linspace(0, 1, 300)

# Figure 1: posterior pdf of Theta after each batch on same figure
plt.figure(figsize=(9, 5))

variances = []

for i in range(m):
    k = heads[i]  # heads observed in this batch

    # apply conjugate prior update derived above
    alpha = alpha + k          # alpha grows by number of heads
    beta_param = beta_param + (n - k)  # beta grows by number of tails

    # evaluate the updated Beta pdf using scipy
    pdf_vals = beta.pdf(x_vals, alpha, beta_param)
    plt.plot(x_vals, pdf_vals, label=f'batch {i+1} (heads={k})')

    # VARIANCE of Beta(alpha, beta):
    # If X ~ Beta(a, b), then:
    #   E[X]   = a / (a + b)
    #   Var[X] = a*b / ((a+b)^2 * (a+b+1))
    # As more batches are processed, (alpha + beta) keeps growing since we add n to it every iteration.
    # So the denominator grows faster than numerator => variance shrinks.
    # This reflects increasing confidence in our estimate of Theta.
    var = (alpha * beta_param) / ((alpha + beta_param)**2 * (alpha + beta_param + 1))
    variances.append(var)

    print(f"batch {i+1}: heads={k}, alpha={alpha}, beta={beta_param}, variance={var:.6f}")

plt.xlabel('theta (bias of coin)')
plt.ylabel('density')
plt.title('posterior pdf of coin bias after each batch')
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig('bayesian_update_pdf_ee25btech11011.png', dpi=150)
plt.close()
print("figure 1 saved: bayesian_update_pdf_ee25btech11011.png")

# Figure 2: variance vs iteration number
# should monotonically decrease - distribution gets sharper with more data
plt.figure(figsize=(8, 5))
plt.plot(range(1, m+1), variances, 'b-o', linewidth=2, markersize=5)
plt.xlabel('iteration number')
plt.ylabel('variance of theta')
plt.title('variance of coin bias vs iteration number')
plt.tight_layout()
plt.savefig('bayesian_update_var_ee25btech11011.png', dpi=150)
plt.close()
print("figure 2 saved: bayesian_update_var_ee25btech11011.png")

final_mean = alpha / (alpha + beta_param)
final_var = (alpha * beta_param) / ((alpha + beta_param)**2 * (alpha + beta_param + 1))

print(f"\nfinal alpha: {alpha}, final beta: {beta_param}")
print(f"final mean of theta: {final_mean:.4f}")
print(f"final variance of theta: {final_var:.6f}")