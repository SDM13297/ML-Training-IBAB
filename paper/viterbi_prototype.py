import math
import random
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

def generate_sequences(N, alpha, beta):
    mother = [0] * N
    daughter = [0] * N
    
    mother[0] = 1 if random.random() < 0.5 else 0
    
    # Build the Mother sequence (The Rulebook)
    for i in range(1, N):
        if mother[i-1] == 1:
            mother[i] = 1 if random.random() <= alpha else 0
        else: # mother[i-1] == 0
            mother[i] = 0 if random.random() <= beta else 1
            
    # Build the Daughter sequence (The Z-Channel Noise)
    for i in range(N):
        if mother[i] == 1:
            if random.random() <= 0.5:
                daughter[i] = 0
            else:
                daughter[i] = 1
        else:
            daughter[i] = 0
            
    return mother, daughter

def run_viterbi_decoder(daughter_sequence, alpha, beta):
    """
    Decodes the daughter sequence to find the most likely mother sequence 
    using the pure Ramakrishnan et al. (2022) 2-state model.
    """
    N = len(daughter_sequence)
    if N == 0:
        return []
    
    # Pre-calculate logs to prevent underflow and save compute time
    log_alpha = math.log(alpha)
    log_not_alpha = math.log(1 - alpha)
    log_beta = math.log(beta)
    log_not_beta = math.log(1 - beta)
    log_half = math.log(0.5)
    
    log_score_0 = math.log(0.5)
    log_score_1 = math.log(0.5)
    
    # Backpointer Dictionary
    backpointers = {} 

    for i in range(1, N):
        backpointers[i] = {}
        D = daughter_sequence[i]
        
        # Save yesterday's scores before calculating today's
        prev_0 = log_score_0
        prev_1 = log_score_1
        
        if D == 1:
            # --- DAUGHTER IS 1 ---
            # To get a 1 today, we came from a 0 (Path A) or a 1 (Path B)
            path_a = prev_0 + log_not_beta + log_half
            path_b = prev_1 + log_alpha + log_half
            
            if path_a > path_b:
                log_score_1 = path_a
                backpointers[i][1] = 0
            else:
                log_score_1 = path_b
                backpointers[i][1] = 1
                
            # If Daughter is 1, Mother CANNOT be 0 (Emission P=0 -> log=-inf)
            log_score_0 = -float('inf')
            
        elif D == 0:
            # --- DAUGHTER IS 0 ---
            
            # Hypothesis 1: Mother was a 0 today (Emission P=1.0 -> log=0)
            path_c = prev_0 + log_beta + 0         # Stay in 0
            path_d = prev_1 + log_not_alpha + 0    # Jump to 0
            
            if path_c > path_d:
                log_score_0 = path_c
                backpointers[i][0] = 0
            else:
                log_score_0 = path_d
                backpointers[i][0] = 1
                
            # Hypothesis 2: Mother was a 1 today, but lost to noise (Emission P=0.5)
            path_a = prev_0 + log_not_beta + log_half  # Jump to 1
            path_b = prev_1 + log_alpha + log_half     # Stay in 1
            
            if path_a > path_b:
                log_score_1 = path_a
                backpointers[i][1] = 0
            else:
                log_score_1 = path_b
                backpointers[i][1] = 1

    # The Traceback
    mother_sequence = [0] * N
    
    # Find the absolute best ending state
    best_last_state = max([(log_score_0, 0), (log_score_1, 1)])[1]
    mother_sequence[N-1] = best_last_state

    # Walk backward using the dictionary
    for i in range(N-1, 0, -1):
        current_winning_state = mother_sequence[i]
        
        # Look up where we came from
        if current_winning_state in backpointers[i]:
            previous_winning_state = backpointers[i][current_winning_state]
        else:
            previous_winning_state = 0 # Safety fallback
            
        mother_sequence[i-1] = previous_winning_state

    return mother_sequence


def plot_threshold_k(daughter, viterbi_mother):
    N = len(daughter)
    gap_stats = {} # Will store: {gap_length: [times_filled, total_occurrences]}
    
    current_gap = 0
    
    for i in range(N):
        if daughter[i] == 1:
            if current_gap > 0:
                L = current_gap
                
                # Extract what Viterbi did in this exact gap
                # The gap is from index (i - L) to (i - 1)
                viterbi_guess = viterbi_mother[i-L : i]
                
                # Did Viterbi fill it entirely with 1s?
                is_filled = all(nucleosome == 1 for nucleosome in viterbi_guess)
                
                # Record the stats
                if L not in gap_stats:
                    gap_stats[L] = {'filled': 0, 'total': 0}
                
                gap_stats[L]['total'] += 1
                if is_filled:
                    gap_stats[L]['filled'] += 1
                    
                current_gap = 0 # Reset
        else:
            current_gap += 1

    # Calculate probabilities
    lengths = sorted(gap_stats.keys())
    probabilities = [gap_stats[L]['filled'] / gap_stats[L]['total'] for L in lengths]

    # Plot the Cliff
    plt.figure(figsize=(8, 5))
    # plt.plot(lengths, probabilities, marker='o', linestyle='-', color='b')
    plt.bar(lengths, probabilities)
    plt.title("Viterbi Decoder: Probability of Filling a Gap vs. Gap Length")
    plt.xlabel("Gap Length (L)")
    plt.ylabel("Probability of Filling with 1s")
    plt.grid(True)
    plt.xticks(lengths)
    plt.savefig('output.png')


# if __name__ =="__main__":
#     N = 1000000
#     ALPHA = 0.85
#     BETA = 0.85

#     print(f"Generating Sequences of length {N}")
#     mother, daughter = generate_sequences(N, ALPHA, BETA)

#     print("\nRunning Viterbi on the sequences....")
#     viterbi_mother = run_viterbi_decoder(daughter, ALPHA, BETA)

#     raw_errors = sum(1 for i in range(N) if mother[i] != daughter[i])
#     viterbi_errors = sum(1 for i in range(N) if mother[i] != viterbi_mother[i])

#     print("\n--- RESULTS ---")
#     print(f"Raw Daughter Error Rate:  {raw_errors / N * 100:.2f}%")
#     print(f"Viterbi Decoder Error Rate: {viterbi_errors / N * 100:.2f}%")

#     # ------- Gap Analysis --------
#     gap_lengths = []
#     current_gap = 0

#     for i in range(N):
#         if daughter[i] == 1:
#             # We hit a boundary! 
#             if current_gap > 0:
#                 gap_lengths.append(current_gap)
                
#                 current_gap = 0
                
#         else: # daughter[i] == 0
#             current_gap += 1

#     plot_threshold_k(daughter, viterbi_mother)

# if __name__ =="__main__":
#     N = 10000
#     BETA = 0.85
    
#     alpha_values = []
#     raw_error_rates = []
#     viterbi_error_rates = []

#     print("Running Simulation Sweep for Figure 3...")
    
#     # Sweep alpha from 0.50 to 0.95
#     for a in range(50, 96, 5):
#         ALPHA = a / 100.0
#         alpha_values.append(ALPHA)
        
#         # 1. Generate
#         mother, daughter = generate_sequences(N, ALPHA, BETA)
        
#         # 2. Decode
#         viterbi_mother = run_viterbi_decoder(daughter, ALPHA, BETA)
        
#         # 3. Calculate Errors
#         raw_err = sum(1 for i in range(N) if mother[i] != daughter[i]) / N
#         vit_err = sum(1 for i in range(N) if mother[i] != viterbi_mother[i]) / N
        
#         raw_error_rates.append(raw_err)
#         viterbi_error_rates.append(vit_err)
#         print(f"Alpha: {ALPHA:.2f} | Raw Error: {raw_err:.3f} | Viterbi Error: {vit_err:.3f}")

#     # Plot exactly like Figure 3
#     plt.figure(figsize=(8, 5))
#     plt.plot(alpha_values, raw_error_rates, label='Raw Daughter Error', marker='o', color='red')
#     plt.plot(alpha_values, viterbi_error_rates, label='Viterbi Rescued Error', marker='s', color='blue')
    
#     plt.title("Reproduction of Figure 3: Error Rate vs Alpha")
#     plt.xlabel("Alpha (Probability of maintaining '1')")
#     plt.ylabel("Fraction of Errors")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('figure_3_reproduction.png')
#     print("Saved figure_3_reproduction.png!")

if __name__ =="__main__":
    N = 1000 # Lowered slightly so the 100-cell grid doesn't take forever to compute
    
    # Create the grid points (from 0.50 to 0.95)
    alphas = np.arange(0.1, 1, 0.1)
    betas = np.arange(0.1, 1, 0.1)
    
    error_matrix = np.zeros((len(betas), len(alphas)))

    print("Running 2D Simulation Sweep for Figure 3... (This will take a few seconds)")
    
    for i, b in enumerate(betas):
        for j, a in enumerate(alphas):
            # 1. Generate
            mother, daughter = generate_sequences(N, a, b)
            
            # 2. Decode
            viterbi_mother = run_viterbi_decoder(daughter, a, b)
            
            # 3. Calculate Error
            vit_err = sum(1 for k in range(N) if mother[k] != viterbi_mother[k]) / N
            error_matrix[i, j] = vit_err
            
        print(f"Completed Beta sweep row: {b:.2f}")

    cmap_custom = LinearSegmentedColormap.from_list(
        "figure3_map",
        [
            (0.00, (0.85, 0.00, 0.00)),  # deep red
            (0.25, (1.00, 0.00, 0.00)),  # bright red
            (0.50, (0.75, 0.00, 0.75)),  # magenta
            (0.75, (0.40, 0.00, 0.90)),  # purple-blue
            (1.00, (0.00, 0.00, 1.00)),  # blue
        ]
    )

    # Plot exactly like the real Figure 3 with full axes
    plt.figure(figsize=(8, 6))
    
    # extent=[x_min, x_max, y_min, y_max] forces the boundary of the axes
    plt.imshow(error_matrix, origin='lower', cmap=cmap_custom, 
               extent=[0.1, 1.0, 0.1, 1.0], aspect='auto')
    
    # Explicitly set the tick marks to match the paper's grid
    plt.xticks(np.arange(0.1, 1, 0.1)) # Forces ticks at 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    plt.yticks(np.arange(0.1, 1, 0.1))
    
    plt.colorbar(label='Viterbi Bit Error Rate (BER)')
    plt.title("Figure 3 Reproduction: Viterbi Error Rate Heatmap")
    plt.xlabel("Alpha (Probability of staying 1)")
    plt.ylabel("Beta (Probability of staying 0)")
    
    plt.savefig('figure_3_heatmap_full_axes.png')
    print("\nSaved figure_3_heatmap_full_axes.png! The axes should now be perfectly framed.")