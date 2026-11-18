/**
 * Learning for combinatorial optimisation: a routing environment (TSP and CVRP) with
 * constructive neural solvers (pointer networks with supervised and actor-critic training,
 * the attention model with a greedy-rollout baseline, POMO with Sym-NCO, preference
 * optimisation, active search and efficient active search), a learned 2-opt improvement
 * policy, non-autoregressive heatmaps (a supervised residual gated edge GCN and an
 * unsupervised Gumbel–Sinkhorn soft-permutation model), unsupervised QUBO GNNs (PI-GNN and
 * Erdős goes neural) and GFlowNets for MaxCut and independent sets, branch and bound with
 * learned branching, and a learned WalkSAT flip policy. {@link
 * io.github.zzuegg.brainiac.experimental.combopt.ClassicHeuristics} holds the classical
 * references (nearest neighbour, farthest insertion, 2-opt, Or-opt, annealing, greedy MIS
 * and MaxCut).
 */
package io.github.zzuegg.brainiac.experimental.combopt;
