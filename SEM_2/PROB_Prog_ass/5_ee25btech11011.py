import sys
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)
# main idea:
# packets arrive randomly (exp inter-arrival with rate lambda)
# only one server → if it's busy, packets have to wait
#
# service time:
#   fig 1 → exponential(mu)
#   fig 2 → uniform(0, 2/mu)  (same mean, just different distribution)
#
# for each packet:
#   wait time = departure - arrival (includes both waiting + service)
#
# also tracking running avg:
#   T_bar(k) = avg of first k wait times
#
# rough expectation:
#   lambda < mu → stable → should settle
#   lambda ≈ mu → borderline → grows slowly
#   lambda > mu → unstable → keeps increasing

def simulate_queue(N, lam, mu, service='exponential'):
    # generate arrival gaps
    inter_arrivals = np.random.exponential(1/lam, N)
    
    # generate service times
    if service == 'exponential':
        service_times = np.random.exponential(1/mu, N)
    else:
        # uniform case (same mean as exponential)
        service_times = np.random.uniform(0, 2/mu, N)

    # arrival times = cumulative sum of gaps
    arrival_times = np.cumsum(inter_arrivals)

    departure_times = np.zeros(N)

    for k in range(N):
        if k == 0:
            # first packet → no waiting
            departure_times[k] = arrival_times[k] + service_times[k]
        else:
            # starts service only after:
            #   it arrives AND previous one finishes
            start = max(arrival_times[k], departure_times[k-1])
            departure_times[k] = start + service_times[k]

    # total time in system
    wait_times = departure_times - arrival_times

    # running average
    running_avg = np.cumsum(wait_times) / np.arange(1, N+1)

    return running_avg


def get_name_from_script():
    # trying to extract roll number from filename
    script = os.path.basename(sys.argv[0])
    parts = script.replace('.py', '').split('_')
    if len(parts) >= 2:
        return '_'.join(parts[1:])
    return 'ee25btech11011'   # fallback


def main():
    if len(sys.argv) < 6:
        print("Usage: python 5_ee25btech11011.py <N> <lam1> <lam2> <lam3> <mu>")
        sys.exit(1)

    N    = int(sys.argv[1])
    lam1 = float(sys.argv[2])   # < mu
    lam2 = float(sys.argv[3])   # ≈ mu
    lam3 = float(sys.argv[4])   # > mu
    mu   = float(sys.argv[5])

    lambdas = [lam1, lam2, lam3]
    k = np.arange(1, N+1)

    # ── Figure 1: exponential service ──
    plt.figure(figsize=(10, 5))
    for lam in lambdas:
        avg = simulate_queue(N, lam, mu, service='exponential')
        plt.plot(k, avg, label=f'λ={lam}')

    plt.xlabel('Number of packets (k)')
    plt.ylabel('Running avg wait time T̄(k)')
    plt.title(f'Wait time vs k (exp service, µ={mu})')
    plt.legend()
    plt.tight_layout()

    fname1 = f'queue_exp_{get_name_from_script()}.png'
    plt.savefig(fname1, dpi=150)
    plt.close()
    print("saved:", fname1)

    # ── Figure 2: uniform service ──
    plt.figure(figsize=(10, 5))
    for lam in lambdas:
        avg = simulate_queue(N, lam, mu, service='uniform')
        plt.plot(k, avg, label=f'λ={lam}')

    plt.xlabel('Number of packets (k)')
    plt.ylabel('Running avg wait time T̄(k)')
    plt.title(f'Wait time vs k (uniform service, µ={mu})')
    plt.legend()
    plt.tight_layout()

    fname2 = f'queue_uniform_{get_name_from_script()}.png'
    plt.savefig(fname2, dpi=150)
    plt.close()
    print("saved:", fname2)

    # observations:
    #
    # lambda < mu:
    #   queue is stable → wait time kind of settles
    #
    # lambda ≈ mu:
    #   borderline → wait grows slowly, not very clear
    #
    # lambda > mu:
    #   too many arrivals → queue keeps growing → wait increases a lot
    #
    # main idea: rho = lambda / mu
    #   rho < 1 → stable
    #   rho >= 1 → unstable
    # should look similar for both service types since mean is same

if __name__ == "__main__":
    main()