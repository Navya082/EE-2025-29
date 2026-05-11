import sys
import random
import numpy as np
# This is a simulation of the gambler’s ruin problem.
# A player starts with S rupees and plays repeatedly:
#   wins 1 rupee with probability p
#   loses 1 rupee with probability (1 - p)

# The game stops when:
#  money becomes 0 (player loses everything), or money reaches U (target amount)
 
# In this program:
# 1. we simulate the game N times
# 2. compute average results from the simulation
# 3. compare them with theoretical values

# setting seed so output is same every time
random.seed(42)

# Simulation #
def simulate_gambler(S, p, U):
# simulate one run of the game
# returns number of steps and whether player reached U
    money = S
    steps = 0

    # Continue until gambler either loses everything or reaches U
    while 0 < money < U:
        if random.random() < p:
            money += 1
        else:
            money -= 1
        steps += 1

    # win = 1 if reached U, else 0
    return steps, int(money == U)

# Expected Number of Games #
def expected_games(U, p):
    # Solves linear system for expected number of games Ei
    # Recurrence:
    # Ei = 1 + pEi+1 + (1-p)Ei-1
    # with boundary conditions E0 = EU = 0

    size = U - 1
    M = np.zeros((size, size))
    b = np.ones(size)

    # Construct matrix equation M * E = b
    for i in range(size):
        M[i][i] = 1

        if i > 0:
            M[i][i - 1] = -(1 - p)

        if i < size - 1:
            M[i][i + 1] = -p

    return np.linalg.solve(M, b)

# Probability of Winning #
def win_probability(U, p):
    # Solves linear system for probability of winning Pi
    # Recurrence:
    # Pi = pPi+1 + (1-p)Pi-1
    # Boundary conditions:
    # P0 = 0, PU = 1

    size = U - 1
    M = np.zeros((size, size))
    b = np.zeros(size)

    # Construct matrix equation M * P = b
    for i in range(size):
        M[i][i] = 1
        if i > 0:
            M[i][i - 1] = -(1 - p)
        if i < size - 1:
            M[i][i + 1] = -p

    # Apply boundary condition: P(U) = 1
    b[-1] = p

    return np.linalg.solve(M, b)


# main part
def main():
    # Read input parameters: S, p, U, N
    S = int(sys.argv[1])
    p = float(sys.argv[2])
    U = int(sys.argv[3])
    N = int(sys.argv[4])

    steps_list = []
    wins = []
    win_steps = []
    lose_steps = []

    # run simulation N times
    for _ in range(N):
        steps, win = simulate_gambler(S, p, U)

        steps_list.append(steps)
        wins.append(win)

        # Store steps separately for win/loss cases
        if win:
            win_steps.append(steps)
        else:
            lose_steps.append(steps)

    # Sample Statistics 
    # Compute averages from simulation
    sample_mean_steps = np.mean(steps_list)
    win_fraction = np.mean(wins)

    win_mean_steps = np.mean(win_steps) if win_steps else 0
    lose_mean_steps = np.mean(lose_steps) if lose_steps else 0

    # Theoretical Values #
    # Solve linear systems for expected values
    E = expected_games(U, p)
    P = win_probability(U, p)

    # S corresponds to index S-1
    expected_steps = E[S - 1]
    win_prob = P[S - 1]

    # Output #
    # Print results in required format
    print(f"Sample mean of number of games before Gambler stops : {sample_mean_steps:.4f}")
    print(f"Expected number of games before Gambler stops       : {expected_steps:.4f}")
    print()

    print(f"Fraction of times the Gambler wins                  : {win_fraction:.4f}")
    print(f"Probability that the Gambler wins                   : {win_prob:.4f}")
    print()

    print(f"Sample mean of number of games given Gambler wins   : {win_mean_steps:.4f}")
    print(f"Sample mean of number of games given Gambler loses  : {lose_mean_steps:.4f}")

# Note:
# Due to randomness in simulation, sample results may slightly differ from theoretical values. Increasing N improves accuracy due to the Law of Large Numbers.

if __name__ == "__main__":
    main()