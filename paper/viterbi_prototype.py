import math
import random

def generate_sequences(N, alpha, beta):
    mother = [0] * N
    daughter = [0] * N
    
    mother[0] = 1 if random.random() < 0.5 else 0
    
    # 2. Build the Mother sequence (The Rulebook)
    for i in range(1, N):
        if mother[i-1] == 1:
            pass
        else: # mother[i-1] == 0
            pass
            
    # 3. Build the Daughter sequence (The Z-Channel Wind)
    for i in range(N):
        if mother[i] == 1:
            if random.random() <= 0.5:
                daughter[i] = 0
            else:
                daughter[i] = 1
        else:
            pass
            
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
                
            # Hypothesis 2: Mother was a 1 today, but lost to wind (Emission P=0.5)
            path_a = prev_0 + log_not_beta + log_half  # Jump to 1
            path_b = prev_1 + log_alpha + log_half     # Stay in 1
            
            if path_a > path_b:
                log_score_1 = path_a
                backpointers[i][1] = 0
            else:
                log_score_1 = path_b
                backpointers[i][1] = 1

    # 4. The Traceback Time Machine
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