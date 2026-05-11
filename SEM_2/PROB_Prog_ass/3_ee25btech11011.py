import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import os

np.random.seed(42)

def get_name_from_script():
    # pull roll number from script name e.g. 3_ee25btech11011.py -> ee25btech11011
    script = os.path.basename(sys.argv[0])
    parts = script.replace('.py', '').split('_')
    if len(parts) >= 2:
        return '_'.join(parts[1:])
    return 'ee25btech11011'

# MODE 0: Angle method
# One end of the chord is fixed at (0,1). The chord makes angle Theta
# with the horizontal tangent at that point. Theta ~ Uniform[0, pi].
# To find chord length as a function of Theta:
#   Fixed point A = (0,1), direction = (cos(Theta), -sin(Theta))
#   Parametric point on chord: (t*cosθ, 1 - t*sinθ)
#   Substituting into unit circle equation x² + y² = 1:
#   t² - 2t*sinθ = 0  =>  t = 2*sinθ
#   So chord length X = 2*sin(Theta)
#
# P(X >= sqrt(3)) = P(Theta in [pi/3, 2pi/3]) = 1/3  (as given in problem)

def mode0(N):
    theta = np.random.uniform(0, np.pi, N)
    chord_length = 2 * np.sin(theta)
    return chord_length

# MODE 1: Distance method
# Distance U of chord from center ~ Uniform[0, 1].
# The chord is perpendicular to the line joining center to chord midpoint.
# By pythagoras: half chord = sqrt(1 - U²), so full chord Y = 2*sqrt(1 - U²)
#
# P(Y >= sqrt(3)) = P(U <= 1/2) = 1/2  (as given in problem)

def mode1(N):
    U = np.random.uniform(0, 1, N)
    chord_length = 2 * np.sqrt(1 - U**2)
    return chord_length

# MODE 2: Random midpoint method
# Chord midpoint (X,Y) is uniform within the unit circle.
# R = sqrt(X²+Y²) is the distance of midpoint from center.
#
# CDF of R: FR(r) = r²  (ratio of areas: pi*r² / pi*1²)
# Inverse CDF: R = sqrt(U) where U ~ Uniform[0,1]
# Chord length Z = 2*sqrt(1 - R²)
#
# P(Z >= sqrt(3)) = P(R <= 1/2) = P(U <= 1/4) = 1/4  (as given in problem)

def mode2(N):
    U = np.random.uniform(0, 1, N)
    R = np.sqrt(U)                        # inverse CDF of FR(r) = r²
    chord_length = 2 * np.sqrt(1 - R**2)
    return chord_length

def plot_histogram(chord_length, mode, N, name):
    # sqrt(N) bins as required
    n_bins = int(math.sqrt(N))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(chord_length, bins=n_bins, density=True,
            color='steelblue', edgecolor='white', alpha=0.8)

    # vertical line at sqrt(3) to see what fraction exceeds it
    ax.axvline(x=math.sqrt(3), color='red', linewidth=2,
               linestyle='--', label=f'√3 ≈ {math.sqrt(3):.3f}')

    mode_names = {
        0: "Angle Method (P=1/3)",
        1: "Distance Method (P=1/2)",
        2: "Random Midpoint (P=1/4)"
    }
    ax.set_xlabel('Chord Length')
    ax.set_ylabel('Density')
    ax.set_title(f"Bertrand's Paradox - Mode {mode}: {mode_names[mode]}  (N={N})")
    ax.legend()
    fig.tight_layout()

    fname = f"bertrand_mode{mode}_{name}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Histogram saved: {fname}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python 3_ee25btech11011.py <mode> <N>")
        sys.exit(1)

    mode = int(sys.argv[1])
    N    = int(sys.argv[2])
    name = get_name_from_script()

    expected  = {0: 1/3, 1: 1/2, 2: 1/4}
    threshold = math.sqrt(3)

    if mode == 0:
        chord_length = mode0(N)
    elif mode == 1:
        chord_length = mode1(N)
    elif mode == 2:
        chord_length = mode2(N)
    else:
        print(f"Unknown mode {mode}. Choose 0, 1 or 2.")
        sys.exit(1)

    # print fraction and compare with theoretical value
    fraction = np.mean(chord_length > threshold)
    print(f"Mode {mode} | N={N}")
    print(f"  Fraction with chord > √3 : {fraction:.4f}")
    print(f"  Theoretical probability   : {expected[mode]:.4f}")

    plot_histogram(chord_length, mode, N, name)


if __name__ == "__main__":
    main()