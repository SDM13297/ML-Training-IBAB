"""
Ramakrishnan et al. (2022)
SMAP/Viterbi decoder for epigenetic inheritance via parallelized Monte Carlo.
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures


def generate_sequences(N, alpha, beta):
    """
    Generates a mother (Markov chain) and daughter (Z-channel output) sequence.
    alpha = P(stay 1), beta = P(stay 0), replication loss = 0.5
    """
    mother = [0] * N
    daughter = [0] * N

    mother[0] = 1 if random.random() < 0.5 else 0

    for i in range(1, N):
        if mother[i-1] == 1:
            mother[i] = 1 if random.random() <= alpha else 0
        else:
            mother[i] = 0 if random.random() <= beta else 1

    for i in range(N):
        if mother[i] == 1:
            daughter[i] = 1 if random.random() > 0.5 else 0   # mother[i] == 0 → daughter[i] stays 0

    return mother, daughter


def run_viterbi_decoder(daughter_sequence, alpha, beta):
    """
    SMAP decoder — recovers the most probable mother sequence from the daughter.
    Runs in log-domain to avoid underflow.
    Hard constraint: D=1 forces M=1 (Z-channel property).
    """
    N = len(daughter_sequence)
    if N == 0:
        return []

    log_alpha     = math.log(alpha)
    log_not_alpha = math.log(1 - alpha)
    log_beta      = math.log(beta)
    log_not_beta  = math.log(1 - beta)
    log_half      = math.log(0.5)

    log_score_0 = math.log(0.5)
    log_score_1 = math.log(0.5)
    backpointers = {}

    # Forward pass — fill trellis left to right
    for i in range(1, N):
        backpointers[i] = {}
        D = daughter_sequence[i]
        prev_0, prev_1 = log_score_0, log_score_1

        if D == 1:
            # Mother must be 1; state 0 is impossible
            path_a = prev_0 + log_not_beta + log_half  # 0->1
            path_b = prev_1 + log_alpha    + log_half  # 1->1
            if path_a > path_b:
                log_score_1, backpointers[i][1] = path_a, 0
            else:
                log_score_1, backpointers[i][1] = path_b, 1
            log_score_0 = -float('inf')

        else:  # D == 0
            # Paths to state 0
            path_c = prev_0 + log_beta       # 0->0
            path_d = prev_1 + log_not_alpha  # 1->0
            if path_c > path_d:
                log_score_0, backpointers[i][0] = path_c, 0
            else:
                log_score_0, backpointers[i][0] = path_d, 1

            # Paths to state 1 (mother=1 but lost in replication)
            path_a = prev_0 + log_not_beta + log_half  # 0->1
            path_b = prev_1 + log_alpha    + log_half  # 1->1
            if path_a > path_b:
                log_score_1, backpointers[i][1] = path_a, 0
            else:
                log_score_1, backpointers[i][1] = path_b, 1

    # Traceback
    mother_sequence = [0] * N
    mother_sequence[N-1] = max([(log_score_0, 0), (log_score_1, 1)])[1]

    for i in range(N-1, 0, -1):
        s = mother_sequence[i]
        mother_sequence[i-1] = backpointers[i].get(s, 0)

    return mother_sequence


def simulate_pixel(args):
    """Runs all trials for one (alpha, beta) cell and returns average BER."""
    i, j, alpha, beta, N, trials = args
    total_error = 0.0

    for _ in range(trials):
        mother, daughter = generate_sequences(N, alpha, beta)
        viterbi_mother   = run_viterbi_decoder(daughter, alpha, beta)
        total_error += sum(1 for k in range(N) if mother[k] != viterbi_mother[k]) / N

    return (i, j, total_error / trials)


if __name__ == "__main__":
    N      = 1000
    TRIALS = 10000

    alphas = np.arange(0.10, 1.00, 0.1)
    betas  = np.arange(0.10, 1.00, 0.1)
    error_matrix = np.zeros((len(betas), len(alphas)))
    print(error_matrix)

    tasks = [(i, j, a, b, N, TRIALS)
             for i, b in enumerate(betas)
             for j, a in enumerate(alphas)]

    print(f"Distributing {len(tasks)} tasks | {len(tasks) * TRIALS:,} total simulations")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, j, avg_error in executor.map(simulate_pixel, tasks):
            error_matrix[i, j] = avg_error

    print("Done. Rendering...")

    plt.figure(figsize=(8, 6))
    plt.imshow(error_matrix, origin='lower', cmap='viridis_r',
               vmin=0.0, vmax=0.30, extent=[0.1, 1.0, 0.1, 1.0], aspect='auto')

    # Annotate each cell with its BER value
    for i in range(len(betas)):
        for j in range(len(alphas)):
            plt.text(alphas[j], betas[i], f"{error_matrix[i,j]:.2f}",
                     ha='center', va='center', fontsize=6,
                     color='white' if error_matrix[i, j] > 0.15 else 'black')

    for i, b in enumerate(betas):
        for j, a in enumerate(alphas):
            print(f"alpha={a:.1f} beta={b:.1f} BER={error_matrix[i,j]:.4f}")
            
    plt.colorbar(label='Average Viterbi BER')
    plt.xticks(np.arange(0.1, 1.0, 0.1))
    plt.yticks(np.arange(0.1, 1.0, 0.1))
    plt.xlabel("α — P(stay modified)")
    plt.ylabel("β — P(stay unmodified)")
    plt.title(f"Figure 3 — Ramakrishnan et al. (2022)  [{TRIALS} trials, N={N}]")
    plt.tight_layout()
    plt.savefig('figure_3_parallel_heatmap.png', dpi=150)
    print("Saved figure_3_parallel_heatmap.png")