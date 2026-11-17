# brainiac-experiments: measured results

This page collects every measured result recorded for the experimental AI techniques in the `brainiac-experiments` module of the brainiac repository (PRs #86, #92 and #97), plus a full run of the experiment suite on 2026-11-17 after PR #105.

## How the module is laid out

- **`io.github.zzuegg.brainiac.experimental.*`** holds the reusable building blocks: layers, optimizers, learning rules, memory modules, environments and so on. These packages mirror the core packages they could move to (`experimental.nn`, `experimental.train`, `experimental.rl`, `experimental.llm`, `experimental.evolve`, `experimental.data`, `experimental.tensor`, `experimental.autograd`). The blocks implement the core interfaces (`Layer` + `LayerSpec`, `Optimizer`, `Loss`, `Environment`, `LanguageModel` and others), so swapping one in is a one-line change. Promoting a block to core means dropping `.experimental` from its package name.
- **`io.github.zzuegg.brainiac.experiments.<theme>`** holds the experiments themselves: toy tasks, comparisons against baselines, and demos. Each theme below is one of these packages.

## How to read the numbers

Every result comes from a small synthetic task that runs on a CPU in seconds, with fixed seeds and one run per configuration, usually at a small training budget. **The numbers are directional, not benchmarks.** They show that a technique works in this codebase and roughly how it compares with a baseline on the same toy task. They say nothing reliable about real-scale performance, and small gaps (a few points) are within run-to-run noise.

Rules this page follows:

- Every number was found in one of the sources below. None was estimated. Numbers keyed **Run** were printed by the tests in the full suite run of 2026-11-17 (577 tests, all passing); everything else predates that run. Tests that printed nothing in that run were then given print statements (no assertion, threshold, seed or budget changed) and the module was run again the same day (576 tests, all passing). Their numbers are also keyed **Run**.
- Where a Run number clearly disagrees with an earlier recorded number, both are given and the difference is noted. PR #105 changed four things numerically: GumbelChannel's noise stream, and exact second-order maths in MAML, Sophia and GradientPenalty.
- An experiment with no recorded numbers is listed as **no numbers recorded**, together with what its test asserts. The test thresholds are pass conditions, not measurements.

Source keys (given under each experiment):

| Key | Source |
|---|---|
| PR86, PR92, PR97 | Descriptions of zzuegg/brainiac pull requests #86, #92 and #97 |
| R-learn | Session transcript: report of the learning rules and optimizers worker |
| R-attn | Session transcript: report of the attention and sequence worker |
| R-mem | Session transcript: report of the memory and context worker |
| R-evo | Session transcript: report of the evolution and interpretability worker |
| R-gen | Session transcript: report of the generative and exotic worker |
| R-agents | Session transcript: report of the reasoning and agents worker |
| R-llm | Session transcript: report of the LLM techniques worker |
| R-rl | Session transcript: report of the RL, graphs and time-series worker |
| R-tricks | Session transcript: report of the training tricks, SSL and vision worker |
| Test | The experiment's JUnit test (assertions, thresholds, comments) |
| Run | Full suite run, 2026-11-17, after the core-hooks switch (PR #105) |

---

## agents

### CuriosityExploration (X27: RND and ICM)
Sources: R-agents, PR92, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Random Network Distillation bonus, tabular Q-learner, sparse maze | Distinct cells visited / goals reached | 51 cells, 15 goals | No bonus: 23 cells, 0 goals | RND more than doubles coverage and is the only setup that reaches the goal. |
| Intrinsic Curiosity Module bonus | Distinct cells visited | 25 cells | No bonus: 23 cells | ICM barely helps with a tabular learner (weak result). |

### SelfPlayLeague (X26: self-play with an Elo league)
Sources: Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Self-play on tic-tac-toe, snapshots rated in an Elo league | Elo, start → end (first / second player) | 880 → 1032 / 937 → 1098 | Own starting Elo | Both learners climb (152 and 161 Elo). |
| Same | Win rate vs random (first / second player) | 1.0 / 0.96 | Random player | Self-play alone produces strong players for both seats. |
| Self-play on Connect-3 | Elo, start → end (first / second player) | 932 → 1062 / 905 → 998 | Own starting Elo | Both learners climb (131 and 93 Elo). |
| Same | Win rate vs random (first / second player) | 1.0 / 0.745 | Random player | The second seat is weaker on Connect-3. |
| Core tabular Q-learner through `GameEnvironment` | Score vs random | -1.0 → 0.925 | Its own start | Core learners plug into the game environment and learn to win. |

### DreamingAgent (X25: world models)
Sources: R-agents, PR92, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Dreamer-style agent, policy trained only in imagined rollouts (cart-pole) | Episode return | 200 | Random policy: 19.6 | Learning entirely inside a learned world model solves cart-pole. |
| JEPA world model (EMA target, VICReg variance term) | Next-embedding prediction error | 7e-4 | "Predict no change": 0.27 | JEPA predicts the latent dynamics well. |
| JEPA world model | Latent standard deviation | 1.55 | Test requires > 0.3 | The variance term prevents representation collapse. |

### NeuralEpisodicVersusDqn (Neural Episodic Control)
Sources: R-agents, PR92, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Neural Episodic Control, cart-pole, 1500 environment steps | Return | 192 | DQN: 75 | Episodic memory is far more sample-efficient than DQN at a tiny budget. |

### HardExploration (Go-Explore)
Sources: R-agents, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Go-Explore, deadly serpentine maze | Solved | Solved | Random walks: never solved | Archiving stepping stones and returning to them beats random exploration where any wrong step is fatal. |
| Same (Run) | Distinct cells reached | 73 | Random walks: 40 | Go-Explore reaches almost twice the cells. |
| Go-Explore, sparse mountain car | Solved | Solved | Sticky random actions also reach the flag (14 random episodes) | Mountain car is not a hard-exploration case here (null result for that task). |

### Homeostasis (homeostatic RL)
Sources: R-agents, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DQN with drive-reduction reward, food and water at opposite ends | Survival steps | Full 200 steps | Random agent: 44 | Drive reduction alone teaches survival. |
| Same | Share of time spent on food | 0.49 | Balanced = 0.5 (test requires 0.3 to 0.7) | The agent learns to balance both needs. |

### StreamingRl (streaming deep RL)
Sources: R-agents, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Stream AC(λ) with ObGD traces, cart-pole, one update per transition, no replay buffer | Return, start to end | 9.7 → 200 | Its own starting return | Online actor-critic with no replay buffer still solves cart-pole. |

### LevelCurriculum (Prioritized Level Replay)
Sources: R-agents, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| PLR (rank and staleness sampling by TD error), procedural mazes, tabular Q | Levels solved, summed over 4 seeds | 167 | Uniform level sampling: 147 | Spending episodes where the learner still improves solves more levels. |

### EpistemicForaging (active inference, cued T-maze)
Sources: R-agents, PR92, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Sophisticated active inference | Reward rate | 95.5%, and it reads the cue first in every episode | Classic active inference (fixed two-step policies): 53% | Planning over future beliefs makes the agent seek information first; classic policies gamble. |

### ReferentialGame (emergent communication)
Sources: R-agents, PR92, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Sender and receiver trained end to end through a Gumbel-softmax channel | Accuracy, training objects / held-out objects | 0.97 / 0.92; Run: 0.97 / 0.93 with 46 distinct messages | Chance: 0.25 | The invented code generalizes to unseen objects. The Run (new GumbelChannel noise stream, PR #105) agrees within a point. |
| Same | Topographic similarity | 0.65 | Test requires > 0.2 | The emergent language is fairly compositional. |

### ReflectiveSolver (Reflexion)
Sources: R-agents, PR92, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Reflexion: lessons from "too high/too low" feedback filter later samples (tiny LM, multiplication) | Share solved | 0.54 | Plain retrying with the same feedback: 0.38; first attempt: 0.22 | A memory of lessons beats retrying. Caveat: lessons filter samples; the model is not conditioned on reflection text. |

### DebatePanel (debate and voting)
Sources: R-agents, PR92, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Majority vote of 5 tiny LMs | Accuracy on multiplication puzzles | 0.52 | Average single agent: 0.47 | Panels beat their members. |
| Debate in answer space | Same | 0.53 | Average single agent: 0.47; vote: 0.52 | Debate is only marginally above voting (the agents exchange answers, not arguments). |
| Product of experts | Same | 0.50 (Run) | Average single agent: 0.47; best single agent: 0.49 | Beats every member, but below voting and debate. |

---

## attention

### AttentionShootout
Sources: R-attn, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Gated attention (NoPE) and Differential attention | Associative recall accuracy | 1.00 | Test requires > 0.9; core TransformerBlock not reported | Both solve associative recall. |
| Every variant in the lineup (core TransformerBlock, differential, gated NoPE, gated rotary, forgetting (FoX), native sparse (NSA), nGPT, universal, feedback, Kimi-style 3:1 hybrid) | Parameters; recall(4,4,2) accuracy after 2 training steps (smoke run) | Parameters: transformer 3248, differential 1040, gated NoPE 1296, gated rotary 1296, forgetting 1314, native sparse 1406, nGPT 2672, universal 3265, feedback 3589, hybrid 3:1 10768. Accuracy 0.11 to 0.37 (Run) | Chance about 0.25 | A plumbing check only: every variant builds and trains through its `LayerSpec`. Two steps are far too few to rank them, so the accuracies are noise around chance.

---

## context

### EvictionComparison (X7: learned KV eviction and SnapKV)
Sources: R-mem, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Learned eviction scorer (trained on a different stream) | Output error at question steps vs a keep-everything cache | 0.015 | Heavy hitters (H2O): 0.73 | Learning that unusual keys will be needed later keeps the facts that recency and attention-so-far evict. |
| Learned eviction scorer | Same | 0.015 (Run) | Sliding window: 0.58; attention sinks: 0.75; H2O: 0.73 (Run) | The learned scorer beats every heuristic by a factor of about 40; sinks do no better than H2O here. |
| SnapKV prompt compression (keep 12) | Same | 0.0093 (Run) | Keep the 12 most recent tokens: 0.048 (Run) | SnapKV's error is about 5x lower than keeping recent tokens. |

### AttentionViewerDemo (X10: attention and cache viewer)
Sources: Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Self-contained HTML page of attention reads and evictions | Steps; page size; evictions | 40 steps, 53,369 bytes, 32 evictions (Run) | n/a | The page shows every step and all 32 evictions of 40 tokens through 8 slots, with no scripts or external files. |

---

## control

### ContinuousControlComparison (SAC, TD3, DDPG, CEM planning vs PPO on Pendulum)
Sources: Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| CEM-MPC on a learned model | Pendulum return at a tiny budget (400 steps) | -323 | Core PPO: -322; random: -334 | Ties PPO; neither is clearly above random at this budget. |
| SAC | Same | -334 | Random: -334 | No better than random at this budget. |
| DDPG | Same | -373 | Random: -334 | Below random at this budget. |
| TD3 | Same | -391 | Random: -334 | Below random at this budget. The test only asserts finite returns for all 6 methods. Null result at this budget; the block-level tests in `experimental.rl` (Run, larger budgets) do show trained DDPG -148, CEM-MPC -186, TD3 -390 and SAC -461 against random -621 to -652. |

### RainbowAblation
Sources: R-rl, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Full Rainbow, C51, QR-DQN | Greedy return on a small maze | 0.93 each (Run) | Core DQN: 0.93 | All solve the maze; on greedy return none beats plain DQN. |
| Full Rainbow | Mean training return | 0.82 (Run) | Core DQN: 0.68; C51: 0.69; QR-DQN: 0.59 | Rainbow's advantage shows up during training, not in the final policy. |
| Noisy networks alone | Greedy return | Fails the maze; Run: -0.40 | Core DQN: 0.93 | Noisy-net exploration on its own does not solve this maze. |

### SelfPlayArena (AlphaZero, MuZero-lite)
Sources: Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| AlphaZero network + MCTS, Connect-3 | Score vs random | 0.90 | Pure MCTS: 1.0; AlphaZero prior alone: 0.85 | Clears the 0.75 bar, but pure MCTS scores higher against random. |
| Same | Score vs pure MCTS | 0.60 | Pure MCTS | Head to head, the learned network beats plain search. |

### OfflineRlComparison
Sources: R-rl, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| CQL, IQL, Decision Transformer on mixed-quality logged data | Return | About 0.95 each; Run: 0.94 each | Behaviour cloning: -0.2; the data itself: 0.18 (Run) | Value-based and return-conditioned methods improve on mediocre data; cloning copies its average. |
| Filtered BC (top 10% of episodes) | Return | 0.94 (Run) | CQL, IQL, DT: 0.94 | Simply cloning the best episodes matches the offline-RL methods on this task. |

### GoalsImitationAndTeams (HER, imitation, value decomposition)
Sources: R-rl, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Hindsight experience replay, 7-bit flip | Success rate | 1.0 | DQN without HER: 0.01 | Relabelling goals turns a sparse problem into a solvable one. The Run confirms 1.0 vs 0.01 in the block-level `HindsightReplayTest`; the experiment test itself runs 5 bits, where HER and plain DQN both reach 1.0. |
| QMIX, two-step coordination game | Return | 8 (the optimum) | VDN: 7; independent learners: 7 | Only the monotonic mixer finds the coordinated optimum. |
| MaxEnt IRL from 5 demos (slippery grid) | Return | 0.92 (Run) | Expert: 0.92; random: -0.09 | Recovers the expert's return exactly. |
| Behaviour cloning, same demos | Return | 0.84 (Run) | Expert: 0.92 | Close to the expert. |
| GAIL, same demos | Return | -0.20 (Run) | Random: -0.09 | Fails in this experiment, below random. The block-level `ImitationTest` (Run) shows GAIL reaching 0.95 vs random -0.06 in its own setup. |

---

## evolution

### DeceptiveMaze (novelty search)
Sources: PR86, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Novelty search, 200 generations | Seeds solved (of 3); best distance to goal | 2 of 3 (solved at generations 136 and 171; best distance 0.56, 0.57, 4.04) (Run) | Objective (distance-to-goal) search: 0 of 3, stuck at distance 4.00 on every seed (Run) | Objective search sits in the deceptive trap on every seed; novelty search escapes it on two. |

### PoetCurriculum (POET-lite)
Sources: R-evo, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| POET on WindyReach | Environments generated | 7, the hardest solved | Agent optimised directly on the hardest environment (score not recorded) | POET grows a curriculum and solves its hardest environment. The test asserts at least 3 environments and a POET score > -0.5. |
| Same (Run) | Score on the hardest environment (target 5.36, wind -0.53) | -0.347, with 10 transfers | Optimised directly on it: -0.348 | Null result: POET ties direct optimisation on its hardest environment. |

### EvolvedPlasticity (meta-evolved Hebbian rules)
Sources: R-evo, PR86, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Hebbian rule coefficients evolved by CMA-ES; weights start at zero and learn within a lifetime | Recall accuracy | 98% | Random rules: 57% | Evolution discovers a working learning rule with no gradient descent. |

### GrowingTransformers (X17)
Sources: R-evo, PR86, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Transformer that grows FFN width, heads and depth on loss plateaus (fold-and-add task) | Loss | 0.73 | Same tiny model, fixed size: 1.23 | Function-preserving growth beats staying small for the same step budget. The Run shows the grown model ends with 1464 parameters against 476, so the comparison is not parameter-matched. |

### EvolvedTransformerBlocks (X18)
Sources: R-evo, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Evolutionary search over heads, KV heads, gating, norm placement, depth and activations | Loss | 1.651; Run: 1.670 | Conventional seed block (pre-RMSNorm, 2 heads, GELU): 1.729; Run: 1.735 | Evolution finds a slightly better block. The Run gives slightly different values with the same ordering (best: 4 heads, 1 KV head, gated, pre-RMSNorm). |

### EvolvedTokenizers (X19)
Sources: R-evo, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Evolved vocabulary size and merges | Held-out bits per character (bigram LM) | 0.844 | Greedy BPE: 0.847; raw bytes: 2.28 | Ties greedy BPE (within 1%); both are far better than bytes. |

### PopulationEnsembles (X22)
Sources: R-evo, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Ensemble of the best distinct designs from a neuroevolution run, abstaining on low-confidence inputs | Risk at 50% coverage | 0 | Full coverage: 0.097 | Disagreement points to exactly the mistakes worth abstaining on. |
| Same | Mutual information (disagreement) | 0.10 out of distribution | 0.0016 in distribution | Disagreement flags out-of-distribution inputs. |

---

## exotic

### HyperdimensionalDemo
Sources: Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Bit-packed hypervectors: "dollar of Mexico" analogy and one-shot language ID | Analogy answer; language-ID accuracy | "peso" at 10,048 bits; language ID 1.00 (Run) | Chance 1/6 (0.17) for language ID | One example per language is enough for perfect language ID on this toy set. |

### BitNetExperiment (BitNet b1.58)
Sources: Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Ternary quantization-aware training (BitLinear) on spirals | Accuracy | QAT 1.00; packed additions-only inference agrees on 1000 of 1000 predictions (Run) | Float model: 1.00; post-training ternarization: 0.90 (Run) | Training with ternary weights loses nothing here, while ternarizing a float model afterwards costs 10 points. |

### LogicGateExperiment (differentiable logic gate networks)
Sources: PR92, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Logic-gate network learning an 8-bit comparator, compiled to a bit-sliced circuit | Discrete circuit accuracy (all 256 inputs) | 0.988 (Run) | n/a | The hardened circuit is nearly exact, and the test checks it matches the float network bit for bit. |

### SpikingClassifier (LIF with surrogate gradients)
Sources: Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Two-layer LIF network, rate vs latency coding | Accuracy; hidden spike rate | Rate coding: 0.995, spike rate 0.174. Latency coding: 1.00, spike rate 0.068 (Run) | Rate vs latency coding | Latency coding is as accurate with about 2.5x fewer hidden spikes. |

### ReservoirForecast (ESN vs NG-RC on chaotic series)
Sources: R-gen, PR92, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Next-generation reservoir computing, Lorenz | Valid closed-loop forecast steps (44 steps ≈ 1 Lyapunov time) | 250 (about 5.7 Lyapunov times) | Echo state network: 191 | NG-RC forecasts chaos longer with a simpler model. |
| Echo state network, Mackey-Glass | Valid steps | 300+ | Test requires > 150 | ESNs handle Mackey-Glass well. |

### KolmogorovArnoldFit (KAN with grid extension)
Sources: R-gen, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| [2,5,1] KAN fitting exp(sin(πx) + y²), grid extended from G=3 to G=10 | Test RMSE | 0.017 after extension | 0.075 on the G=3 grid | Grid extension continues training and cuts the error about 4x. |

### OverlappingShapeCapsules (capsules, dynamic routing)
Sources: R-gen, PR92, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Capsule layer with margin loss on 8x8 shapes | Accuracy on single shapes / overlaid pairs | 1.00 / 0.97 | Test requires > 0.9 / > 0.7 | Per-class capsule lengths report both shapes of an overlay at once. |

### EquivariantPointClouds (EGNN)
Sources: R-gen, PR92, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| EGNN, trained in one orientation, tested on random rotations | Accuracy on rotated clouds | 0.98 | MLP on raw coordinates: 0.51 | Built-in E(3) equivariance generalizes to rotations; the MLP drops to chance. The Run shows the MLP scores 1.00 in the training orientation (EGNN 0.97), so its failure is purely a rotation failure. |

### FractalDropPath (FractalNet)
Sources: R-gen, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| FractalBlock trained with drop-path, two spirals | Accuracy of each column alone | ≥ 0.995 for every column | Without drop-path, worst column: 0.78 | Drop-path makes every column a standalone classifier. |

### HyperSines (hypernetwork)
Sources: R-gen, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Hypernetwork generating a 1-16-16-1 MLP per sine task | MSE on unseen tasks | 0.008 | Target variance about 0.99 (test comment); before training: 1.21 (Run) | Generated networks fit amplitudes and phases never seen in training. |

### DiffractiveShapeClassifier (optical D2NN)
Sources: R-gen, PR92, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Trained phase masks with angular-spectrum propagation (core FFT) | Shape accuracy | 0.97 | Test requires > 0.9; before training: 0.27 (Run) | An all-optical classifier works with only phase masks trained. |

### AnalogNoiseAwareTraining
Sources: R-gen, PR92, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Weight-noise injection during training, programmed onto simulated noisy chips | Accuracy on noisy chips | 0.96 | Plain training: 0.78 | Noise-aware training makes analog in-memory inference robust. The Run shows it costs nothing on clean hardware (0.98 vs 0.97 plain). |

### ProbabilisticBits (p-bit Ising machine)
Sources: R-gen, PR92, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| p-bit Gibbs sampling | Total-variation distance to the exact Boltzmann distribution | 0.015 | Test requires < 0.03 | Sampling matches the exact distribution. |
| 2-D Ising lattice, 16x16 | Mean absolute magnetization | 0.144 at β=0.3; 0.973 at β=0.6 (Run) | Test requires < 0.25 and > 0.9 | Disordered above and ordered below the critical temperature, as physics predicts. |
| Invertible AND gate | Truth-table probability | Exact mass on valid states 0.9997; forward (A=B=1) P(C=1) 0.999; backward (C=1) P(A=B=1) 0.999; with C=0 the three false input pairs get 0.337, 0.327, 0.336 (Run) | Uniform 1/3 for the C=0 case | The same network computes AND forwards and inverts it backwards. |

### WorkspaceBroadcast (global workspace)
Sources: R-gen, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Global workspace with a differentiable top-k write and broadcast | Share of passes the flagged specialist wins; output MSE | 100%; MSE ≈ 0 | Predicting zero: MSE about 1 (test comment); before training (Run): 53%, MSE 5.59 | The network learns to let the important specialist write and to broadcast its content. |

### ThousandBrainsVoting (sensorimotor columns)
Sources: R-gen, PR92, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Five voting columns | Sensations needed to recognize an object | 2.3 | One column: 4.5 | Voting across columns halves the movements needed. |

---

## generative

The generative worker recorded no numbers for these experiments; all numbers below come from the Run (second pass, see How to read the numbers).

### TinyDdpm (X29: DDPM and DDIM)
Sources: Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DDPM and DDIM with classifier-free guidance on 8x8 toy shapes | Squared distance to the nearest real image; class match | DDPM: 17.5, class match 0.875. DDIM (20 steps): 28.6, class match 0.583 (Run) | Untrained model: 103.0; chance 0.25 (Run) | Both samplers land 4 to 6x closer to real images than an untrained model; the full DDPM sampler beats 20-step DDIM on both counts. |

### RectifiedFlowExperiment (flow matching, Heun vs Euler, reflow)
Sources: Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Rectified flow on a 6-blob ring | Share of samples near a blob; 4-step sampler error; guided hit rate; one-step error | Near a blob 0.953; Heun error 0.048; guided samples in their blob 1.00; reflow one-step error 1.158 → 0.172 (Run) | Untrained: 0.031; Euler error: 0.099; before reflow: 1.158 (Run) | Heun halves Euler's error at 4 steps, and one reflow pass cuts the one-step error by 85%. |
| Small DiT (adaLN-Zero) with flow matching (DiTFlowTest) | Class match; squared distance to the nearest real image | Guided (scale 4): class match 0.688, distance 23.2 (Run) | Unguided: class match 0.188, distance 11.2 (Run) | Guidance makes the short-trained DiT commit to the requested class, at the cost of samples further from real images. |

### ImageAutoencoders (VAE with free bits, VQ-VAE)
Sources: Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| VAE with free bits | Reconstruction; active latents; prior-sample distance | Reconstruction 490.3 → 7.2; 8 of 8 latents active; prior samples 17.8 (Run) | Plain VAE: 8 active latents; untrained prior samples: 75.0 (Run) | The VAE learns, but the plain VAE did not collapse either, so free bits showed no advantage here. |
| VQ-VAE (one code per image) | Class kept; squared distance to the nearest real image | Class kept 0.953; distance 2.97 (Run) | n/a | One code per image keeps the class and decodes close to real images. |

### ToyAudioCodec (RVQ audio codec)
Sources: Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| SoundStream-style codec with residual vector quantization and quantizer dropout | SNR by number of stages | 1 stage 3.84 dB; 2 stages 5.75 dB; 3 stages 6.83 dB; 4 stages 7.51 dB (Run) | 1 stage | Each stage adds fidelity, with diminishing returns (+1.9, +1.1, +0.7 dB). |

### NeuralCellularAutomaton (X16: growing NCA)
Sources: Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Growing neural cellular automaton with a damaged sample pool | Loss to target; regrowth | Grown 0.060; damaged 0.090 → regrown 0.067 (Run) | Untrained rule: 0.248; undamaged pattern run as long: 0.066 (Run) | The rule grows the target from one cell and repairs almost all of the damage. |

### Lenia
Sources: Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Continuous cellular automaton plus a growth-parameter search | Settings where life persists; mass share | 5 of 16 random settings kept life; the first (mu 0.128, sigma 0.045) holds 0.228 of the world after 30 steps (Run) | n/a | About a third of random growth settings sustain life. |

### NeuralFields (Fourier features, 2-D Gaussian splats)
Sources: Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Coordinate MLP with Fourier features | PSNR | 120.0 dB, the PSNR function's cap (MSE ≤ 1e-12) (Run) | Raw-coordinate MLP: 12.2 dB (Run) | Fourier features fit the 24x24 picture essentially exactly; raw coordinates barely fit it. |
| 2-D Gaussian splats (48) | PSNR | 8.7 → 20.6 dB (Run) | Before fitting: 8.7 dB | A gain of about 12 dB. |

### TalkingDigits (X30: formant synth + CTC recognizer)
Sources: Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| CTC-trained frame classifier on synthesized spoken digit strings | Whole-string accuracy | 1.00 on 60 strings (Run) | n/a | Every 3-digit string is recognized exactly. |

### TextDiffusionVsAutoregressive (X28: masked diffusion text)
Sources: Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| LLaDA-style masked diffusion text model | Share of samples that are real words | 0.90 (Run) | Autoregressive model of the same size: 1.00; untrained diffusion: 0.00 (Run) | Diffusion learns the words but trails autoregressive decoding slightly. |

---

## graphs

### GnnBenchmarks
Sources: R-rl, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| GCN, node classification on a stochastic block model at feature noise 2.0 | Accuracy | 0.83 | Feature-only MLP: 0.35 | Message passing recovers labels from neighbours when features are noisy. |
| GAT, GraphSAGE, same task (Run) | Accuracy | GAT 0.75; GraphSAGE 0.52 | GCN: 0.83; MLP: 0.35 | Both beat the MLP; GCN is best here. |
| GIN with sum readout, trees vs cycles | Accuracy | 0.85 (Run) | GraphSAGE + mean: 0.50; GIN + mean: 0.625; GCN + sum: 0.625 | Sum aggregation with an injective update counts structure; mean readout does much worse. |
| Graph autoencoder, link prediction | AUC | 0.75 (Run) | Common-neighbours heuristic: 0.67 | A learned embedding beats the classic heuristic. |

### RecsysTabularBenchmarks
Sources: R-rl, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Two-tower retrieval, users never seen in training | Recall@10 | About 0.5; Run: 0.50 (0.48 on seen users) | Random: 0.034 | Feature towers serve cold-start users. |
| Factorization machine, click prediction | Accuracy | 0.73 (Run) | Logistic regression: 0.59 | Pairwise feature interactions help. |
| BPR matrix factorization | Recall@10 | 0.38 (Run) | Random: 0.034 | About 11x random. |
| TabNet-lite, cross network (DCN-V2) and MLP on tabular data | Accuracy | TabNet 0.882; DCN-V2 0.933; MLP 0.905; TabNet puts 0.84 of its mask mass on the 5 informative columns (Run) | Logistic regression: 0.626 (Run) | All three beat logistic regression; TabNet's masks find the informative columns but it trails DCN-V2 and the MLP. |

---

## interp

### SuperpositionRecovery (sparse autoencoder)
Sources: R-evo, PR86, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| TopK sparse autoencoder on 12 superposed features in 6 dimensions | True features recovered (cosine > 0.9) | 12 of 12; Run: mean best cosine 0.98, explained variance 0.91 | Test requires ≥ 9 and explained variance > 0.8 | SAEs recover features packed in superposition. |

### ClassForgetting (machine unlearning)
Sources: R-evo, PR86, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Retain-regularized gradient ascent on one class | Forgotten-class accuracy (training samples) | 0.99 → 0.0 | Before unlearning: 0.99 | The class is removed. |
| Same | Forgotten-class accuracy on unseen samples | 0.0 | Test requires < 0.15 | Forgetting generalizes beyond the training samples. |
| Same | Retained-class accuracy | 0.983 → 0.987 | Before unlearning: 0.983 | The rest of the model is unharmed. |

### BrainMovies (X31)
Sources: R-evo, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| BrainMovie listener filming a growing Brain with RigL-style rewiring | Frames; neuron count | 16-frame GIF; neurons 7 → 11 | n/a | A demo: the GIF shows recruitment and rewiring as the loss falls. |

---

## language

### PeftShootout
Sources: R-llm, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Full fine-tuning, LoRA, LoRA+, PiSSA | New-task loss; trainable values | Full 0.007 (26,252 values); LoRA 0.054; PiSSA 0.074; LoRA+ 0.161 (4,608 values each) (Run) | Loss before adapting: 7.95 (Run) | LoRA gets close to full fine-tuning with 18% of the values. |
| Houlsby adapters, VeRA, BitFit, IA3 | Same | Houlsby 0.92 (1,168); VeRA 0.79 (960); BitFit 1.82 (908); IA3 1.90 (384) (Run) | Same | Fewer values, higher loss: these trail the low-rank methods. |
| Prefix tuning, P-tuning, prompt tuning | Same | Prefix 3.30 (512); P-tuning 3.82 (2,240); prompt tuning 4.74 (128) (Run) | Same | Input-only methods adapt least at this budget. |

### PreferenceTuning
Sources: R-llm, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| IPO, KTO, ORPO, SimPO | Chosen-vs-rejected log-prob margin | ORPO 21.4; SimPO 18.9; KTO 4.7; IPO -30.7 (Run) | Margin before tuning: -35.65 (Run) | All four raise the margin; IPO moves it least (+5), the others flip it positive. |
| Bradley-Terry reward model + rejection-sampling fine-tuning | Preference accuracy; mean reward per round | Reward model accuracy 1.00; mean reward over 4 rounds -33.6, 10.1, 17.6, 8.6 (Run) | First round | Rejection sampling climbs the reward model quickly, but the climb is not monotone: round 4 falls back below round 3. |

### CompressionComparison
Sources: R-llm, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| SparseGPT (unstructured and 2:4), Wanda, activation-aware SVD at 50% | LM loss after one-shot compression | SparseGPT 0.119; SparseGPT 2:4 0.112; Wanda 0.120; activation-aware SVD 0.131 (Run) | Dense 0.105; magnitude 0.129; Wanda 2:4 0.158; plain SVD 0.124 (Run) | SparseGPT beats magnitude and Wanda 2:4 as expected; activation-aware SVD is slightly worse than plain SVD here. On synthetic weights the error order is SparseGPT < Wanda < magnitude. |

### LongContextTraining
Sources: R-llm, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Position interpolation + short fine-tune (length 8 → 24) | Loss at the longer length, before → after fine-tuning | 0.596 → 0.064 (Run) | RoPE extrapolation: 0.394 → 0.062 (Run) | Null result: interpolation starts worse and ends the same as extrapolation. |
| Sequence-length warmup | Final loss; tokens processed | 0.105 with 19,136 tokens (Run) | Full-length training: 0.111 with 25,600 tokens (Run) | Same loss with 25% fewer tokens. |

### DecodingShowdown
Sources: R-llm, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Jacobi, lookahead and prompt-lookup decoding, Medusa heads, self-speculative early exit | Output equality; full-model passes | All reproduce greedy output exactly | Greedy decoding | The speed-ups are lossless. The test asserts at least 3 of them need fewer full passes (no pass counts recorded). |

### Block-level results (experimental.llm, experimental.data)
Sources: R-llm.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| SuperBPE (superword merges) | Held-out tokens | At least 20% fewer tokens | Plain BPE, which runs out of merges at 327 tokens | Merging across words compresses text further. |
| Interpolated Kneser-Ney trigram LM | Perplexity on the toy language | Under 2 | n/a | A strong n-gram baseline. |
| Sequence packing with document masks | Loss | Equal to unpacked training (to 1e-4) | Unpacked training | Packing is exact. |
| Vocabulary trimming | Logits of kept tokens | Exactly preserved | Untrimmed model | Lossless for kept tokens. |

---

## learning

### LearnerComparison (X13/X14: learning rules vs backprop)
Sources: R-learn, PR86, Test, Run. Backprop baseline on the same task and budget: spiral 0.987, digits 0.78.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Feedback alignment, direct feedback alignment, sign symmetry (spiral) | Test accuracy | 0.987 | Backprop: 0.987 | Random feedback weights match backprop here. |
| Mono-Forward (spiral) | Test accuracy | 0.98 | Backprop: 0.987 | Matches backprop. |
| Equilibrium Propagation (spiral) | Test accuracy | 0.973 | Backprop: 0.987 | Close to backprop. |
| NoProp (spiral) | Test accuracy | 0.927 | Backprop: 0.987 | Works, slightly behind. |
| Predictive coding, layered (spiral) | Test accuracy | 0.927 | Backprop: 0.987 | Works, slightly behind. |
| Synthetic gradients, cDNI (spiral) | Test accuracy | 0.907 | Backprop: 0.987 | Works, behind. |
| Difference Target Propagation (spiral) | Test accuracy | 0.867 | Backprop: 0.987 | Clearly behind. |
| Forward gradients, 32 directions, no backward pass (spiral) | Test accuracy | 0.79 | Backprop: 0.987 | Well behind backprop (the test allows a 0.25 gap). |
| SoftHebb features + linear readout (digits) | Test accuracy | 0.765 | Backprop: 0.78 | Close to backprop. |
| Forward-Forward, hardest-label negatives (digits) | Test accuracy | 0.76 | Backprop: 0.78 | Close, but only with hard negatives; wrong-label negatives give about 0.6. |
| Predictive coding, layered (digits) | Test accuracy | 0.725 | Backprop: 0.78 | Somewhat behind. |
| Predictive coding on a sparse feed-forward Brain region graph (digits) | Test accuracy | 0.96 | Backprop MLP: 0.78 | The best digits result, though from a different network shape than the backprop baseline. |
| Predictive coding on a looped Brain graph (digits) | Test accuracy | About 0.53; Run: 0.54 | Backprop: 0.78 | Weak; the test only asserts > 0.42. |

### Block-level results (experimental.train)
Sources: R-learn.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| e-prop on a delay task | Loss | Symmetric feedback 7.8e-4; random feedback 3.1e-3 | BPTT: 1.3e-4 | e-prop approaches BPTT; random feedback costs more. |

### ToyTasks
Support class (spiral and noisy-digit datasets); no experiment of its own.

---

## memory

### CompressionVsDropping (X2: compressive memory)
Sources: Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Linear, delta, DPFP and pooled compression of evicted key/value pairs (48 pairs) | Recall (cosine with the stored value); memory floats | Delta + DPFP 0.66 (4,352); pooling 0.40 (1,440); linear + DPFP 0.31 (4,352); delta 0.25 (2,176); linear 0.10 (2,176) (Run) | Dropping evicted pairs: 0.03 (0 floats) (Run) | Every compressor recalls something dropping loses; the delta rule with DPFP features is best by far, and plain linear memory is weak. |

### SleepForgetting (X4: sleep consolidation)
Sources: Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Sleep after every task, replaying a pruned episodic store | Average forgetting; final accuracy | Forgetting 0.105; final accuracy 0.927 (Run) | Awake online learning: forgetting 0.515; final accuracy 0.653 (Run) | Sleep cuts forgetting about 5x and lifts final accuracy by 27 points. |

### EpisodicAgentComparison (X6: episodic control)
Sources: R-mem, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Episodic-control agent on a GridWorld maze | Greedy return | 1.0 | Memoryless agent: 0 | Remembering outcomes across episodes solves the maze. The test also asserts it beats a random agent and completes 1.5x as many episodes. |

---

## optim

### ContinualLearningExperiment (X21: ContinualBrain on permuted digits)
Sources: R-learn, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| GROW_FROZEN (new regions, old ones frozen) | Average forgetting | 0.000 | SHARED (plain fine-tuning): 0.305 | Freezing grown regions cannot forget. |
| GROW_EWC | Average forgetting | 0.053 | SHARED: 0.305 | Growth plus EWC forgets little. |
| SHARED_EWC | Average forgetting | 0.115 | SHARED: 0.305 | EWC alone cuts forgetting by about 60%. |
| All four (Run) | Average accuracy after the last task | GROW_FROZEN 0.797; GROW_EWC 0.768; SHARED_EWC 0.647 | SHARED: 0.607 | Less forgetting also gives a higher average accuracy. |

### GrokkingExperiment
Sources: R-learn, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Modular addition (p=23) with weight decay, 400 steps | Train / test accuracy over 8 checkpoints | Train 0.68 → 1.0 (from the third checkpoint); test 0.0 at every checkpoint | No weight decay (not printed) | Memorizes fully but does not generalize at all within the budget: no grokking. |
| Same | Weight norm | 17.8 → 40.3 | No weight decay: ends with a larger norm (asserted, not printed) | The norm still grows under weight decay, just less than without it. |

### Block-level results (experimental.train)
Sources: R-learn.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Federated averaging (simulated) | Accuracy | 0.97 | n/a | Works in simulation. |
| DiLoCo (simulated) | Accuracy; transfers | 0.96 with 120 transfers | n/a | Near FedAvg with infrequent communication. |
| QDoRA (DoRA on a packed quantized base) | Accuracy on a rotated task | Above 0.85 | Frozen base uses fewer bytes than a dense one | Adapts while keeping the base quantized. |

---

## reasoning

### ArithmeticRl (X23: GRPO family)
Sources: R-agents, PR92, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| GRPO | Greedy accuracy on 4-digit addition | 0.8 | Before training: < 0.15 (test); Run: 0.0 for all three | A randomly initialized tiny LM learns arithmetic from reward alone. |
| Dr.GRPO | Same | 0.92 | Same | The best variant here. |
| DAPO | Same | 0.8 | Same | On par with GRPO. |

### SelfImprovement (DPO, IPO, SimPO, SPIN, self-rewarding)
Sources: R-agents, PR92, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DPO, IPO, SimPO | Greedy accuracy | 1.0 each | Before (Run): DPO 0.24; IPO and SimPO 0.20 | Preference training from labelled pairs works fully. |
| Self-rewarding (the model judges its own answers) | Greedy accuracy | 0.28 → 0.52 | Before: 0.28 | Self-judgment nearly doubles accuracy. |
| SPIN | Greedy accuracy | 0.36 → 0.64 (Run) | Before: 0.36 | Self-play fine-tuning gains 0.28 with no new labels. |

### TestTimeScaling (X15: test-time compute)
Sources: Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Verifier-weighted vote, critique-and-resample, PRM beam search | Accuracy on chain-of-thought arithmetic | 0.64 each | One sample: 0.375; greedy: 0.50 | A verifier or critic lifts accuracy by about 0.27 over one sample. |
| Critique-and-resample | Samples used per problem | 3.7 on average | Test requires < 10 | The critique loop gets there cheaply. |
| PRM best-of-N | Accuracy | 0.59 | One sample: 0.375 | Clear gain. |
| Best-of-N | Accuracy | 0.56 | One sample: 0.375 | Clear gain. |
| Depth-first search | Accuracy | 0.53 | One sample: 0.375 | Gain, but only just above greedy. |
| Self-consistency (majority vote) | Accuracy | 0.44 | One sample: 0.375; greedy: 0.50 | Weakest: above one sample but below greedy decoding. |

### ToolUse (X24: Toolformer)
Sources: R-agents, PR92, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Toolformer filter (keep calls that lower the loss) | Kept calls that are correct | 50 of 51 (from 180 candidates, Run) | n/a | The loss-based filter keeps almost only correct calls. |
| Fine-tuned calculator use on held-out products | Accuracy | 0.825 | Fine-tuned without tools: 0.225 | Tool use generalizes to products never seen. |

### LatinSquarePuzzles (TRM, HRM)
Sources: R-agents, PR92, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Tiny Recursive Model and Hierarchical Reasoning Model, deep supervision, 4x4 Latin squares | Blank-cell accuracy | About 0.72 | Plain MLP: about 0.87 | Negative: the recursive reasoners do not beat an MLP at this budget. |
| Same (Run, 3 runs each) | Whole puzzles solved | TRM 0.20 to 0.22; HRM 0.14 to 0.19 | Plain MLP: 0.45 | The gap is wider on whole puzzles: the MLP solves about twice as many. |

### CoconutCurriculum (latent reasoning)
Sources: R-agents, PR92, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Coconut, final latent stage, two-hop pointer chasing | Accuracy | 0.31 | Language chain of thought: 0.39; answering directly: 0.65 | Negative: latent thoughts trail both baselines. |

### AbsoluteZero (proposer/solver self-play)
Sources: R-agents, PR92, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Absolute Zero on a toy DSL, one seed proposal | Held-out solver accuracy | 0.13 | n/a | Weak. |
| Same | Valid proposals; distinct tasks | 0.94; 33 tasks | n/a | The proposer learns to write valid, varied tasks. |
| Same | Solver success on its own proposed tasks | 0 → 0.4 | Start: 0 | Self-play improves both roles, but little of it transfers to held-out tasks. |

### ArithmeticTask, ChainTask
Support classes (toy tasks with verifiers); no experiments of their own.

---

## selfsupervised

### SelfSupervisedBenchmark
Sources: R-tricks, Test. Baseline: a linear probe on the same encoder left at random initialization scores 0.34 to 0.40 (6 classes, chance 0.17).

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| MAE (one-block ViT, 75% masking) | Linear-probe accuracy | 0.59 | Random encoder: 0.34 to 0.40 | Best method here. |
| SimCLR | Same | 0.56 | Same | Clear gain. |
| VICReg | Same | 0.50 | Same | Clear gain. |
| Barlow Twins | Same | 0.47 | Same | Moderate gain. |
| DINO | Same | 0.45 | Same | Moderate gain. |
| BYOL | Same | 0.42 | Same | Small gain. |
| I-JEPA-lite | Same | 0.39 | Same | No clear gain over a random encoder at this budget. |

---

## sequence

### MixerShootout
Sources: R-attn, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Mamba-3 style rotary state | Parity(6) accuracy | 1.00 | S4D: 0.49 | Rotary (complex) state tracks parity, which a time-invariant convolution cannot. |
| Every mixer in the lineup (linear attention, S4D, Mamba, Mamba-2, Mamba-3 rotary, Gated DeltaNet, Kimi delta, RWKV-7, mLSTM, sLSTM, Hyena, TTT-Linear) | Parameters; recall(4,4,2) accuracy after 2 training steps (smoke run) | Parameters: S4D 560 (smallest) to Hyena 2016 (largest). Accuracy 0.13 to 0.32 (Run) | Chance about 0.25 | A plumbing check only: every mixer builds and trains through its `LayerSpec`. Two steps cannot rank them. |

---

## techniques

### PositionShootout
Sources: R-tricks, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| RoPE, xPos, T5 bias | Lagged-copy accuracy | 1.0 | NoPE: 0.41 | Relative encodings solve the task. |
| CoPE | Same | About 1.0 | NoPE: 0.41 | Also solves it. |
| ALiBi | Same | 0.49 | NoPE: 0.41 | Only slightly above NoPE: a recency bias is not enough for an exact lag. |

### QuantizationShootout (2-bit weights)
Sources: R-tricks, Test, Run. Float model: 0.986.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| LSQ quantization-aware fine-tuning | Test accuracy at 2 bits | 0.97 | Round-to-nearest: 0.63 | Learned step sizes nearly recover float accuracy. |
| GPTQ with act-order | Same | 0.73 | RTN: 0.63 | Hessian-aware rounding helps. |
| GPTQ | Same | 0.71 | RTN: 0.63 | Helps, slightly less than act-order. |
| AWQ | Same | 0.61 | RTN: 0.63 | No gain over RTN at 2 bits. |
| FP8 E4M3 (Run: also E5M2) | Test accuracy | 0.986 | Float: 0.986 | FP8 is lossless here, in both formats. |

### MetaLearningExperiment
Sources: R-tricks, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| First-order MAML | Sine-wave MSE after 5 adaptation steps | 0.81 | Training from scratch: 4.10 | Best meta-learner in the earlier record; second best in the Run. |
| Second-order MAML | Same | 1.89; **Run: 0.65** | Scratch: 4.10; first-order: 0.81 | **Disagreement.** The recorded 1.89 used finite-difference Hessian-vector products; with exact second-order maths (PR #105) the Run gives 0.65, now the best meta-learner here. |
| Reptile | Same | 1.84 | Scratch: 4.10 | Similar to second-order MAML. |
| Prototypical networks, unseen classes | Accuracy | 0.77 | Untrained embedding: 0.56 | Learned embeddings transfer to new classes. |

### GanExperiment (WGAN-GP vs spectral norm)
Sources: R-tricks, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Spectral-norm WGAN | Sliced Wasserstein distance to data | 0.897 → 0.393 (Run) | Start of training: 0.897 | Converges: the distance more than halves (test requires < 0.45). |
| WGAN-GP | Same | 0.897 → 0.840 (Run) | Start of training: 0.897 | Barely moves in a short run; the test only asserts it stays stable. |
| WGAN-GP critic on a known case | Estimated distance; gradient norm | Asserted 3.45 ± 0.25; 1.15 ± 0.08. Run (exact GradientPenalty, PR #105): 3.57; 1.17 | Analytic values (3 × 1.15 and 1.15) | The critic estimates the known distance; the Run values sit inside the tolerance, slightly above the analytic ones. |

### NormalizationShootout
Sources: Test, Run. Setup: 8 blocks, width 16, no warm-up; the test asserts each scheme cuts the loss below 0.6x its start.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| LayerScale | Final loss | 0.339 | Initial: 1.317 | Lowest final loss. |
| DyT | Final loss | 0.375 | Initial: 2.138 | Close second, with no normalization statistics at all. |
| ReZero | Final loss | 0.384 | Initial: 1.322 | Among the best. |
| Pre-LN | Final loss | 0.471 | Initial: 8.856 | Trains well despite a large initial loss. |
| Pre-LN + QK-norm | Final loss | 0.481 | Initial: 8.807 | Same as pre-LN here. |
| DeepNorm | Final loss | 0.502 | Initial: 1.287 | Trains. |
| Peri-LN | Final loss | 0.522 | Initial: 17.568 | Starts from the highest loss and still trains. |
| Post-LN | Final loss | 0.727 | Initial: 1.305 | Worst final loss without warm-up, though it still passes (0.56x its start). |

### TrainingTricksExperiment
Sources: R-tricks, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Warmup-stable-decay | Final MSE | 0.0071 | Constant LR: 0.0161 | Best schedule. |
| Cosine with restarts (SGDR) | Final MSE | 0.0093 | Constant: 0.0161 | Beats constant. |
| Cosine | Final MSE | 0.0104 | Constant: 0.0161 | Beats constant. |
| 1cycle | Final MSE | 0.0129 | Constant: 0.0161 | Beats constant. |
| Z-loss | Mean abs. log Z | 1.17 at accuracy 0.965 | Without z-loss: 2.31 at 0.965 | Halves the normalizer at no accuracy cost. |
| Dynamic loss scaling, fp16 | Final loss | 2.9e-6 | Unscaled: stays at 9.37 (gradients underflow) | Loss scaling is essential for fp16. |
| R-Drop | Accuracy | 0.40 | Without R-Drop: 0.40 | Null result: no gain on this task. |
| Annealed gradient noise, 12-layer MLP | Final loss | 1.2e-3 (Run) | Without noise: 2.6e-5 | Both train, but noise leaves a roughly 50x higher final loss. |

---

## timeseries

### ForecastingBenchmark
Sources: R-rl, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DLinear | Test MSE | 0.019 | Naive: 0.95; seasonal naive: 0.026 | Beats both baselines. |
| PatchTST | Test MSE | 0.019 | Same | Ties DLinear. |
| MLP + RevIN | Test MSE | 0.020 | Plain MLP: 0.038 | Instance normalization nearly halves the error on a trending series. |
| Plain MLP | Test MSE | 0.038 | Seasonal naive: 0.026 | Worse than the seasonal-naive baseline. |
| N-BEATS (generic) | Test MSE | 0.030 (Run) | Naive: 0.95; seasonal naive: 0.026 | Far below naive, but slightly worse than seasonal naive. |
| TCN + RevIN | Test MSE | 0.034 (Run) | Same | Worse than seasonal naive. |
| N-BEATS (interpretable) | Test MSE | 0.071 (Run) | Same | Clearly worse than seasonal naive. The test only asserts every learned model is below half the naive MSE. |

### ProbabilisticForecasting
Sources: R-rl, Test.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DeepAR (GRU + Gaussian head, sampled paths) | 80% interval coverage / mean width | 0.75 / 0.47 | Target coverage 0.80 | Slightly under-covers, with narrower intervals. |
| Quantile MLP (pinball loss) | Same | 0.77 / 0.55 | Target 0.80 | Closer to target, with wider intervals. |

---

## vision

### VisionShootout
Sources: R-tricks, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| CNN, deformable CNN, ConvNeXt, ViT, ViT with 2-D RoPE, MLP-Mixer on 8x8 shape images | Test accuracy | Every architecture at least 0.78 | Plain CNN | All experimental blocks work as drop-ins in `Model.sequential()`. The test asserts > 0.6 each. |
| Per model (Run) | Test accuracy | Deformable CNN 1.0; ViT 0.996; ViT + 2-D RoPE 0.988; ConvNeXt 0.947; MLP-Mixer 0.783 | Plain CNN: 0.947 | Deformable convolution and ViTs lead; MLP-Mixer trails clearly. |

---

## Highlights

The ten most interesting findings, chosen for effect size and for what they show:

1. **Hindsight relabelling solves sparse goals.** On 7-bit flip, HER succeeds 1.0 of the time against 0.01 without it, confirmed by the Run (control).
2. **Random feedback matches backprop.** Feedback alignment, DFA and sign symmetry score 0.987 on spirals, the same as backprop. Mono-Forward (0.98) and Equilibrium Propagation (0.973) come close (learning).
3. **Learned KV eviction keeps what matters.** Output error is 0.015 for the learned scorer against 0.73 for heavy hitters (H2O) (context).
4. **Tools generalize.** A Toolformer-filtered tiny LM answers held-out products at 0.825 against 0.225 without tools, and 50 of 51 kept calls are correct (reasoning).
5. **Equivariance generalizes for free.** The EGNN scores 0.98 on rotated point clouds against 0.51 for an MLP (exotic).
6. **Evolution finds a learning rule.** A CMA-ES-evolved Hebbian rule reaches 98% recall against 57% for random rules, with no gradient descent (evolution).
7. **Growing beats staying small.** A transformer that grows on loss plateaus reaches loss 0.73 against 1.23 at a fixed size (evolution).
8. **Learning only in dreams works.** The Dreamer-style agent reaches 200 on cart-pole against 19.6 for random (agents).
9. **2-bit weights can be nearly free.** LSQ keeps 0.97 against 0.986 for float, and round-to-nearest drops to 0.63. FP8 E4M3 is lossless (techniques).
10. **Growing a frozen region stops forgetting.** GROW_FROZEN forgets 0.000 against 0.305 for plain fine-tuning, and machine unlearning drops a class from 0.99 to 0.0 while the retained accuracy holds at 0.987 (optim, interp).

Honourable mentions: NG-RC forecasts Lorenz for 250 steps (about 5.7 Lyapunov times) against 191 for an ESN. Noise-aware analog training holds 0.96 against 0.78. Active inference reads the cue and is rewarded 95.5% of the time against 53%. Predictive coding on a sparse Brain graph reaches 0.96 on digits.

New in the Run: test-time compute with a verifier or critic lifts chain-of-thought arithmetic from 0.375 (one sample) to 0.64, with the critique loop using 3.7 samples on average (reasoning). With exact second-order maths, MAML reaches 0.65 against 0.81 for first-order, reversing the earlier result (techniques). GIN with sum readout tells trees from cycles at 0.85 against 0.50 for GraphSAGE with mean (graphs). SPIN lifts accuracy from 0.36 to 0.64 with no new labels (reasoning). Sleep consolidation cuts forgetting from 0.515 to 0.105 and lifts final accuracy from 0.65 to 0.93 (memory). One reflow pass cuts rectified flow's one-step error from 1.16 to 0.17 (generative). Ternary quantization-aware training keeps spirals at 1.00 where ternarizing afterwards drops to 0.90 (exotic). Novelty search escapes the deceptive maze on 2 of 3 seeds where objective search stays stuck on all 3 (evolution).

## Negative or null results

These are kept honest in the tests, which assert only the weaker claim.

| Theme | Experiment | Result | What it means |
|---|---|---|---|
| reasoning | TRM and HRM (LatinSquarePuzzles) | About 0.72 vs about 0.87 for a plain MLP | The recursive reasoners do not beat an MLP at this budget. |
| reasoning | CoconutCurriculum | Latent stage 0.31 vs 0.39 language CoT and 0.65 answering directly | Latent reasoning trails both baselines. |
| reasoning | AbsoluteZero | Held-out solver accuracy 0.13 | Self-play improves on its own tasks (0 → 0.4) but transfers little. |
| agents | CuriosityExploration, ICM | 25 cells vs 23 without a bonus | ICM is weak with a tabular learner. |
| agents | DebatePanel | Debate 0.53 vs vote 0.52 | Debate adds almost nothing over voting. |
| agents | HardExploration, mountain car | Sticky random actions also reach the flag | Not a real hard-exploration case. |
| control | RainbowAblation, noisy nets alone | Greedy return -0.40 vs 0.93 for every other variant (Run) | Noisy-net exploration on its own is not enough here. |
| control | RainbowAblation, full Rainbow | Greedy return 0.93, same as core DQN (Run) | Rainbow only helps during training (mean return 0.82 vs 0.68). |
| control | ContinuousControlComparison, 400 steps | PPO -322, CEM-MPC -323, SAC -334, DDPG -373, TD3 -391 vs random -334 (Run) | Nothing clearly beats random at this budget; TD3 and DDPG fall below it. |
| control | GoalsImitationAndTeams, GAIL | -0.20 vs random -0.09 and BC 0.84 (Run) | GAIL fails in the experiment setup, though it works in its block-level test. |
| evolution | PoetCurriculum | POET -0.347 vs direct optimisation -0.348 on the hardest environment (Run) | The curriculum gives no edge over training on the target directly. |
| techniques | GanExperiment, WGAN-GP | Sliced distance 0.897 → 0.840 vs → 0.393 for spectral norm (Run) | Needs far more generator steps than a unit test affords. |
| techniques | TrainingTricksExperiment, R-Drop | 0.40 with and without | No gain on this small task. |
| techniques | TrainingTricksExperiment, gradient noise | Final loss 1.2e-3 with noise vs 2.6e-5 without (Run) | Trains either way, but noise hurts the final loss. |
| techniques | QuantizationShootout, AWQ at 2 bits | 0.61 vs 0.63 RTN | No gain over round-to-nearest. |
| techniques | PositionShootout, ALiBi | 0.49 vs 0.41 NoPE | A recency bias barely helps an exact-lag task. |
| selfsupervised | I-JEPA-lite, BYOL | 0.39 and 0.42 vs 0.34 to 0.40 random encoder | Little or no gain over a random encoder at this budget. |
| timeseries | Plain MLP | MSE 0.038 vs 0.026 seasonal naive | Loses to the seasonal-naive baseline without RevIN. |
| timeseries | N-BEATS, N-BEATS interpretable, TCN + RevIN | MSE 0.030, 0.071, 0.034 vs 0.026 seasonal naive (Run) | Only DLinear, PatchTST and MLP + RevIN beat seasonal naive. |
| reasoning | TestTimeScaling, self-consistency | 0.44 vs 0.50 greedy (Run) | Majority voting over samples does not beat greedy decoding here. |
| language | LongContextTraining | Position interpolation 0.596 → 0.064 vs extrapolation 0.394 → 0.062 (Run) | Interpolation starts worse and ends the same; both recover after a short fine-tune. |
| language | CompressionComparison, activation-aware SVD | Loss 0.131 vs 0.124 for plain SVD (Run) | Activation-aware weighting does not help at this size. |
| language | PreferenceTuning, IPO | Margin -35.7 → -30.7 vs ≥ +4.7 for KTO, ORPO and SimPO (Run) | IPO moves the margin least and leaves it negative. |
| language | PreferenceTuning, rejection sampling | Mean reward -33.6, 10.1, 17.6, 8.6 over 4 rounds (Run) | The climb is not monotone; the test only compares first and last rounds. |
| generative | ImageAutoencoders, free bits | 8 of 8 latents active with and without free bits (Run) | The plain VAE did not collapse, so free bits had nothing to fix. |
| attention, sequence | AttentionShootout and MixerShootout lineups | Accuracy 0.11 to 0.37 after 2 steps (Run) | Smoke runs only; the full lineups are never trained long enough to rank. |
| learning | Forward gradients | 0.79 vs 0.987 backprop | Far behind backprop. |
| learning | Forward-Forward, wrong-label negatives | About 0.6 (vs 0.76 with hardest-label negatives) | Depends heavily on the choice of negatives. |
| learning | Predictive coding on a looped Brain | About 0.53 | Loops in the region graph hurt badly. |
| optim | GrokkingExperiment | Train accuracy 1.0, test accuracy 0.0 at every checkpoint (Run) | No grokking within the test's 400 steps; the test does not wait for it. |

Overturned by the Run: second-order MAML was listed here (1.89 vs 0.81 for first-order, with finite-difference Hessian-vector products). With exact second-order maths (PR #105) it scores 0.65 and is now the best meta-learner in MetaLearningExperiment.
