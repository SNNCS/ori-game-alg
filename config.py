"""Hyperparameters for the original three-structure architecture (ultimatum game).

This is the *clean* re-implementation that strips every e-commerce concept
(recommend / offer_discount / purchase / abandon / LLM prefix injection) and
keeps only the three core structures from the v5 spec:

    G  人际关系图  RelationGraph     R^(n x n x k)
    I  解释机制    InterpretationEngine  ->  Z in R^(n x d)
    T  未来结果树  FutureTreeGen     T = (N, B, P)

Optimization choices that we deliberately keep (per user directive):
    * G is a torch.nn.Parameter; all learning goes through autograd + one
      optimizer. No hand-written slow-layer edge update.
    * The interpretation engine keeps the rule-interpretation block r_j and
      the situation block sigma_j in its input (z = tanh(W_z[s||edge||r||sigma])).
    * z is NOT semantically sliced. Nothing reads z[4:12] as "dignity"; the
      future tree learns branch probabilities from the whole intent vector.
"""

# --- Agent topology (ultimatum game) ---
N_AGENTS = 3
N_ROLES  = 3
ACTOR_A = 0       # proposer   (提议者)
ACTOR_B = 1       # responder  (响应者)
ACTOR_C = 2       # observer / experimenter (旁观者)

# --- Core dims (unchanged from the v5 spec) ---
K = 32              # relation-edge vector dim   G[i,j,:] in R^k
D = 32              # intent vector dim          z_j in R^d
P = 16              # rule-interpretation r_j dim
RHO_DIM   = 8       # role embedding dim
H_DIM     = 16      # history summary dim
OMEGA_DIM = 8       # resource state dim
K_DIM     = 8       # public-knowledge dim
SIGMA_DIM = RHO_DIM + H_DIM + OMEGA_DIM + K_DIM   # 40

# --- Signal s = phi(a_i, S, K) for the ultimatum game ---
# The action IS the signal here (no utterance text). s carries the bid and a
# small context block. No latent "type distribution" head, no e-commerce flags.
N_CONTEXT = 5       # turn_pos, session_len, prev_reject_rate, status_gap, urgency
M = 3 + N_CONTEXT   # [bid, complement, fairness_dev] + context = 8

# Interpretation input width: [ s || edge || r_j || sigma_j ]
INPUT_DIM = M + K + P + SIGMA_DIM                 # 8 + 32 + 16 + 40 = 96

# --- Ultimatum game mechanics ---
BIDS = (0.5, 0.6, 0.7, 0.8, 0.9)        # proposer's *kept* share candidates
RESPONSES = ("accept", "reject", "counter")
OUTSIDE_OPTION = 0.0                     # payoff when the offer is rejected
COUNTER_DISCOUNT = 0.9                   # pie shrinks each time B counters
PATHS_OPEN_DECAY = 0.1                   # optionality lost per continue

# --- Tree / dynamics ---
DEPTH   = 2        # future-tree depth
LAMBDA  = 0.3      # path-dependence strength
GAMMA   = 0.95     # history EMA factor
G_CLIP  = 3.0      # relation edge clamp

# --- Tolerance head range ---
TOL_MIN = 0.4
TOL_MAX = 0.9

# --- Learning ---
LR   = 1e-3        # single optimizer over every nn.Parameter
SEED = 0


def sanity_check():
    assert SIGMA_DIM == 40, SIGMA_DIM
    assert M == 8, M
    assert INPUT_DIM == 96, INPUT_DIM


sanity_check()
