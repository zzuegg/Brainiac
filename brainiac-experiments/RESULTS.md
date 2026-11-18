# brainiac-experiments: measured results

This page collects every measured result recorded for the experimental AI techniques in the `brainiac-experiments` module of the brainiac repository (PRs #86, #92 and #97), plus a full run of the experiment suite on 2026-11-17 after PR #105. It also covers wave 4: 85 more techniques in six work packages (WP1–WP6, listed in `unimplemented-ideas.md`), each measured against a baseline. Wave 4 experiments are tagged **W4** with their item number, for example "W4 1.1". It also covers wave 5: 106 more techniques in six work packages (WP1–WP6, listed in `wave5-ideas.md`), tagged **W5** in the same way, with two new themes, data and safety. It also covers wave 6: 105 more techniques in six work packages (WP1–WP6, listed in `wave6-ideas.md`), tagged **W6** in the same way and measured in 84 new experiments, with one new theme, classic.

## How the module is laid out

- **`io.github.zzuegg.brainiac.experimental.*`** holds the reusable building blocks: layers, optimizers, learning rules, memory modules, environments and so on. These packages mirror the core packages they could move to (`experimental.nn`, `experimental.train`, `experimental.rl`, `experimental.llm`, `experimental.evolve`, `experimental.data`, `experimental.tensor`, `experimental.autograd`). The blocks implement the core interfaces (`Layer` + `LayerSpec`, `Optimizer`, `Loss`, `Environment`, `LanguageModel` and others), so swapping one in is a one-line change. Promoting a block to core means dropping `.experimental` from its package name.
- **`io.github.zzuegg.brainiac.experiments.<theme>`** holds the experiments themselves: toy tasks, comparisons against baselines, and demos. Each theme below is one of these packages.

## How to read the numbers

Every result comes from a small synthetic task that runs on a CPU in seconds, with fixed seeds and one run per configuration, usually at a small training budget. **The numbers are directional, not benchmarks.** They show that a technique works in this codebase and roughly how it compares with a baseline on the same toy task. They say nothing reliable about real-scale performance, and small gaps (a few points) are within run-to-run noise.

Rules this page follows:

- Every number was found in one of the sources below. None was estimated. Numbers keyed **Run** were printed by the tests in the full suite run of 2026-11-17 (577 tests, all passing); everything else predates that run. Tests that printed nothing in that run were then given print statements (no assertion, threshold, seed or budget changed) and the module was run again the same day (576 tests, all passing). Their numbers are also keyed **Run**.
- Where a Run number clearly disagrees with an earlier recorded number, both are given and the difference is noted. PR #105 changed four things numerically: GumbelChannel's noise stream, and exact second-order maths in MAML, Sophia and GradientPenalty.
- Wave 4 numbers are keyed **Run-w4**: they were printed by the tests in the full suite runs after wave 4 on 2026-11-17, one of the merged WP1–WP5 work (690 tests, all passing) and one of the WP6 branch (601 tests, all passing). Where a wave 4 test printed nothing on a point, the number comes from the implementing agent's final report and is marked **(R-wave4)** next to it. Where the two disagree, the test's number is given and the report's is noted.
- Wave 5 numbers are keyed **Run-w5**: they were printed by the tests in three full suite runs after wave 5 on 2026-11-17, one of the merged WP1–WP4 work (804 tests), one of the WP5 branch (711 tests) and one of the WP6 branch (735 tests), all passing. Where a wave 5 test printed nothing on a point, the number comes from the implementing agent's final report and is marked **(R-wave5)** next to it. Where the two disagree, the test's number is given and the report's is noted.
- Wave 6 numbers are keyed **Run-w6**: they were printed by the tests in one full suite run of the merged wave 6 work on 2026-11-17 and 2026-11-18 (446 test classes, 997 tests, all passing). Where a wave 6 test printed nothing on a point, the number comes from the implementing agent's final report and is marked **(R-wave6)** next to it. Where the two disagree, the test's number is given and the report's is noted.
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
| Run-w4 | Full suite runs after wave 4, 2026-11-17: merged WP1–WP5 (690 tests) and the WP6 branch (601 tests), all passing |
| R-wave4 | Session transcript: final reports of the six wave 4 work-package agents (WP1–WP6) |
| Run-w5 | Full suite runs after wave 5, 2026-11-17: merged WP1–WP4 (804 tests), the WP5 branch (711 tests) and the WP6 branch (735 tests), all passing |
| R-wave5 | Session transcript: final reports of the six wave 5 work-package agents (WP1–WP6) |
| Run-w6 | Full suite run after wave 6, 2026-11-17 to 2026-11-18: merged WP1–WP6 work (446 test classes, 997 tests), all passing |
| R-wave6 | Session transcript: final reports of the six wave 6 work-package agents (WP1–WP6) |

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

### ReflectiveSolver (Reflexion; W4 5.10: learned-memory agent loop)
Sources: R-agents, PR92, Test, Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Reflexion: lessons from "too high/too low" feedback filter later samples (tiny LM, multiplication) | Share solved | 0.54 | Plain retrying with the same feedback: 0.38; first attempt: 0.22 | A memory of lessons beats retrying. Caveat: lessons filter samples; the model is not conditioned on reflection text. |
| Learned-memory agent (W4 5.10: MemGPT + Reflexion fusion; `ReflectiveMemoryAgent` over `ReflexionLoop`, `MemoryStream`, `VirtualContextPager`), 240 recurring and commuted products, 2 trials each (R-wave4) | Lifelong solve rate (first try; last quarter) | 0.454 (0.413; 0.583) (Run-w4) | Reflexion alone: 0.117 (0.050; 0.150) (Run-w4) | Reflections stored in a paged long-term store carry over between tasks; the solve rate climbs over the sequence and ends near 4x Reflexion's. |

### DebatePanel (debate and voting)
Sources: R-agents, PR92, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Majority vote of 5 tiny LMs | Accuracy on multiplication puzzles | 0.52 | Average single agent: 0.47 | Panels beat their members. |
| Debate in answer space | Same | 0.53 | Average single agent: 0.47; vote: 0.52 | Debate is only marginally above voting (the agents exchange answers, not arguments). |
| Product of experts | Same | 0.50 (Run) | Average single agent: 0.47; best single agent: 0.49 | Beats every member, but below voting and debate. |

### GenerativeAgentsRecall (W4 5.1: Generative Agents memory stream)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Memory stream scored by recency × importance × relevance, with reflection summaries; toy town over several days, 320 memories, 6 records handed to the reader per question | Recall of past events | 0.89 (latest 0.80, event 0.87, usual place 1.00) | Sliding window: 0.58; `VirtualContextPager` (MemGPT pager): 0.46; relevance-only stream: 0.53 | The three-factor score and reflections beat every simpler memory by at least 0.31. |

### SkillCurriculum (W4 5.2: Voyager skill library)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Auto-curriculum plus a growing library of verified skills retrieved by embedding (simulated program writer, R-wave4) | Tasks solved of 30 after 30 / 60 / 90 steps | 9 / 21 / 24, with 24 skills stored | No library: 10 / 14 / 14; `ReflexionLoop`: 10 / 14 / 15 | The library starts about level and then keeps compounding while both baselines plateau at 14 to 15. |

### AutoResearcher (W4 5.7: AI Scientist pipeline, toy)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Idea → reviewer (logistic regression over settings) → real Brainiac MLP run on two moons → write-up → reproduction check | Share of runs that beat the baseline experiment (accuracy 0.960) | 0.67, with 6 findings reproduced; best 1.000 | Random config search: 0.38; best 1.000 | The reviewer filter almost doubles the hit rate, though both reach the same best run. |

### LatentActionPlay (W4 6.1: Genie latent-action world model)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Latent-action world model (Gumbel bottleneck) trained on 6x6 grid-world frames with no action labels | Action recovery | 0.763 | Chance: 0.25 | Discrete latent actions line up with the true actions three times better than chance. The test threshold was lowered to 0.7 to match (R-wave4). |
| Same, driven by its latent codes | Next-cell accuracy | 1.000 | Action-conditioned `MlpWorldModel`: 1.000 | The model learned from action-free frames is as good a simulator as one given the actions. |


### DiffusionDreaming (W5 6.15: DIAMOND diffusion world model)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Diffusion world model (`DiffusionWorldModel`), cart-pole | Open-loop RMSE at 1 / 10 steps | 0.0028 / 0.0456 | MLP world model (`MlpWorldModel`): 0.0096 / 0.1103 | The diffusion model is 3.4x more accurate at 1 step and 2.4x at 10 steps. |
| `DreamerAgent` trained inside the diffusion world model | Real cart-pole return | 199.5 | Inside the MLP world model: 200.0 | Both solve the task, so the better model gives no return gain here. |


### NeverGiveUpExploration (W6 3.6: NGU and Agent57)
Sources: Run-w6, R-wave6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| NGU (episodic kNN bonus times an RND modulator), tabular learner, sparse maze | Distinct cells visited / goals reached | 42.0 / 0 | No bonus: 26.3 / 0. RND: 51.0 / 13.7. ICM: 30.7 / 0. Go-Explore: 47.3 / 0.7 | Negative: more coverage than no bonus, but no goals; RND is far better. |
| Agent57 (per-arm Q-tables, sliding-window UCB bandit) | Same | 23.3 / 0 | Same | Negative: below no bonus. The bandit gets no extrinsic signal on this map (R-wave6). |

---

## attention

### AttentionShootout
Sources: R-attn, Test, Run, Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Gated attention (NoPE) and Differential attention | Associative recall accuracy | 1.00 | Test requires > 0.9; core TransformerBlock not reported | Both solve associative recall. |
| Every variant in the lineup (core TransformerBlock, differential, gated NoPE, gated rotary, forgetting (FoX), native sparse (NSA), nGPT, universal, feedback, Kimi-style 3:1 hybrid) | Parameters; recall(4,4,2) accuracy after 2 training steps (smoke run) | Parameters: transformer 3248, differential 1040, gated NoPE 1296, gated rotary 1296, forgetting 1314, native sparse 1406, nGPT 2672, universal 3265, feedback 3589, hybrid 3:1 10768. Accuracy 0.11 to 0.37 (Run) | Chance about 0.25 | A plumbing check only: every variant builds and trains through its `LayerSpec`. Two steps are far too few to rank them, so the accuracies are noise around chance. |
| Wave 4 additions to the lineup: latent (MLA), sigmoid, Softpick, stick-breaking, talking heads, selective, sink logit, parallel hybrid | Parameters; recall(4,4,2) accuracy after 2 training steps (smoke run) | Parameters: latent (MLA) 1096, sigmoid 1024, softpick 1024, stick-breaking 1024, talking heads 1032, selective 1024, sink logit 1026, parallel hybrid 3782. Accuracy 0.17 to 0.38 (Run-w4) | Chance about 0.25 | Plumbing check only, as above. The real comparisons are in the experiments below. |

Wave 4 (WP1) experiments share the harness `ArchitectureBench`: a toy language (a noisy Markov prefix followed by a repeated span, so the loss mostly measures how fast attention learns to retrieve), retrieval and exact-lag tasks at several lengths, and baselines built from the core `TransformerBlock`. One seed each.

### KvEfficientAttention (W4 1.1: multi-head latent attention)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| MLA (latent 16, decoupled RoPE keys), latent-only KV cache | Validation loss; cached values per token | 1.891; 40 (20,896 params) | Core MHA: 1.807; 128 (20,928 params). Core GQA kv=1: 1.960; 32 (17,856 params) | MLA lands between MHA and GQA in loss while caching less than a third of MHA's values. |

### DeepStackRefinements (W4 1.2: value residual learning)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Value residual (every layer's V mixed with the first layer's), depth 5 | Validation loss | 2.801 (29,044 params) | Same stack without it: 2.828 (29,040); core `TransformerBlock` stack: 2.996 (29,640) | A small gain over the plain stack. Only depth 5 was measured, not the planned 8 and 16. |

### CrossLayerKv (W4 1.3: cross-layer KV sharing, CLA and YOCO)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| CLA x2 (pairs of layers share one KV projection) | Validation loss; cache values per token | 2.428; 128 (37,120 params) | Unshared stack: 2.522; 256 (41,216 params) | Half the cache and a lower loss. |
| YOCO, window 8 (upper half reuses one global cache) | Same | 2.237; 64 (39,200 params) | Same | A quarter of the cache and the best loss of the three. |

### AttentionWeightings (W4 1.4: sigmoid attention, 1.7: Softpick)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Sigmoid attention (`-log n` bias, no row normalisation) | Validation loss | 1.542 | Softmax: 1.721 | Clearly better LM loss than softmax (but see RecallExtrapolation). |
| Same | Attention mass per row; outlier measure | 1.694; 13.61 | Softmax: 1.000; 5.84 | Rows sum to more than 1 and activation outliers more than double. |
| Softpick | Validation loss; first-token sink share; attention mass; 4-bit quantization error | 1.731; 0.065; 0.855; 0.0952 | Softmax: 1.721; 0.084; 1.000; 0.0970 | Shrinks the sink and slightly lowers 4-bit error at about equal loss. |
| Core `GatedAttention` (for reference) | Validation loss; 4-bit error | 1.781; 0.0964 | Softmax: 1.721; 0.0970 | Slightly worse loss than plain softmax on this task. |

### RecallExtrapolation (W4 1.4: sigmoid, 1.6: SSMax, 1.12: iRoPE)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Scalable-Softmax (SSMax) | Recall at 4 / 8 / 16 pairs (trained on 4) | 0.992 / 0.988 / 0.961 | Softmax: 1.000 / 1.000 / 0.957 | Level with softmax: no extrapolation gain at 4x the pairs. |
| Sigmoid attention | Same | 0.949 / 0.910 / 0.820 | Same | Its better LM loss does not carry over: recall drops off with more pairs. |
| iRoPE (interleaved NoPE/RoPE), 2 layers | Same | 0.965 / 0.957 / 0.941 | All-RoPE: 0.996 / 0.945 / 0.613; all-NoPE: 0.988 / 0.996 / 0.977 | Far better than all-RoPE at 16 pairs, but all-NoPE does slightly better still. |

### LagExtrapolation (W4 1.5: stick-breaking attention)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Stick-breaking attention, no position encoding | Lag-2 accuracy at length 12 / 24 / 48 | 1.000 / 0.998 / 0.998 | Softmax + RoPE: 1.000 / 1.000 / 0.938; softmax NoPE: 0.470 / 0.407 / 0.333 | Built-in recency holds at 4x the training length, where RoPE starts to slip. |

### SinkAndRegisters (W4 1.8: sink logit and registers, 1.9: talking heads)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Learned per-head sink logit | Validation loss; first-token share; attention mass | 1.539; 0.047; 0.708 | Softmax: 1.721; 0.084; 1.000 | An "attend to nothing" option lowers the loss and nearly halves the first-token sink. |
| 2 register tokens | Validation loss; first-token share | 1.529; 0.047 | Same | Same effect, best loss of the row. Measured on the toy language only; the planned VisionShootout row was not done (R-wave4). |
| Talking-heads attention | Validation loss | 1.611 | Softmax: 1.721 | Mixing across heads helps. |

### SelectiveContext (W4 1.10: selective attention)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Selective attention, context pruned to 0.74 | Recent-recall accuracy | 0.996 (unpruned 1.000) | Softmax with H2O at the same budget: 0.602; full softmax: 0.656 | Selective attention loses almost nothing from pruning a quarter of the context and beats softmax by a wide margin. The H2O baseline is an idealised per-query key mask, not a real cache policy (R-wave4). |

### MoeRefinements (W4 1.15: sparse upcycling, 1.16: expert-choice routing)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Expert-choice routing, capacity 2 | Validation MSE; load CV; dropped share | 0.708; 0.000; 0.000 | `FineGrainedMoe` top-2: 0.633; 0.426; 0.000 (zero predictor 2.890) | Perfect balance but a worse loss: tokens no expert picks get zero output, and this regression task has no residual stream to carry them (R-wave4). |
| Token choice, capacity factor 1.25 + router z-loss | Same | 0.803; 0.356; 0.033 | Same | Capping capacity drops 3.3% of tokens and costs loss. |
| Token choice + z-loss + balance loss | Same | 0.791; 0.282; 0.025 | Same | The balance loss evens the load a little more; still behind top-2. |
| Sparse upcycling (MoE initialised as exact copies of the trained dense FFN) | Validation MSE after continued training | 0.880 | Dense continued: 0.950; MoE from scratch at equal compute: 0.541 | Beats continuing the dense model, but a MoE trained from scratch is far better here. |
| Drop-upcycling (half of each expert re-initialised) | Same | 0.964 | Same | Worse than continuing the dense model. |


### SparseSelectionShootout (W5 1.1: DeepSeek Sparse Attention, 1.3: MoBA)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DSA (lightning indexer, dense warm-up, then top-8 keys), toy LM, 1 block, width 16, 200 steps | Validation loss; keys read per query | 2.954; 6.8 | Dense (core): 2.805; 12.5 on average. NSA: 2.864 | Reads about half the keys at a loss cost; NSA stays closer to dense. |
| DSA, recall trained at 32 pairs (R-wave5) | Recall at 8 / 16 / 32 pairs; keys per query | 0.438 / 0.297 / 0.281; 7.6 | Dense: 0.578 / 0.391 / 0.219 (chance 0.125) | Beats dense only at the longest length. |
| MoBA (mixture of block attention, 4 blocks, top-2) | Validation loss; keys per query | 2.980; 6.5 | Dense: 2.805 | Worse than DSA at a similar key budget. |
| MoBA | Recall at 8 / 16 / 32 pairs; keys per query | 0.227 / 0.195 / 0.133; 7.1 | Dense: 0.578 / 0.391 / 0.219 | Clearly worse than dense at every length. |

### CompressedKvShootout (W5 1.2: compressed / heavily compressed attention, CSA / HCA)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| CSA/HCA (DeepSeek-V4 style compressed KV) | Validation loss; cached values per token (+ fixed) | 3.0273; 3.0 (+64), 5992 params | Core MHA: 3.0124; 64. Core GQA kv=1: 3.1121; 16. MLA latent 4: 3.0863; 16. Compressive (Infini-attention): 3.0479; 0 (+672) | Close to MHA's loss with a twentieth of its per-token cache, and better than GQA and MLA at less cache. |

### MultiTokenLookup (W5 1.9: multi-token attention)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Multi-token attention (2x7 key-query convolution) | Block-lookup accuracy (3 blocks) | 0.949 | MHA: 0.395 (chance 0.333) | MHA stays near chance; MTA solves the lookup. |
| Same | Toy LM validation loss | 1.9409 | MHA: 2.9863 | A large LM-loss gain as well. |

### Gemma3nBlocks (W5 1.10: MatFormer, 1.12: AltUp and per-layer embeddings)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| MatFormer nested FFN (one model, sub-widths of 64) | Validation loss at FFN width 8 / 16 / 32 / 64 | 2.8445 / 2.8361 / 2.8319 / 2.8319 | Plain model truncated: 3.0174 / 2.9841 / 2.9387 / 2.9074. Magnitude-pruned: 3.0179 / 2.9679 / 2.9171 / 2.9074. Trained from scratch at width 16: 2.8605 | Every nested width beats truncating or pruning a plain model, and the width-16 slice beats a model trained at width 16. |
| AltUp, K=2, d=16 | Validation loss; hot parameters | 2.4511; 5196 | Plain d=16: 2.7967; 5184. Plain d=32: 1.7246; 20,608 | A wider residual stream for 12 extra parameters closes about a third of the gap to d=32. |
| Per-layer embeddings (PLE), p=8, d=16 | Validation loss; hot parameters (+ cold) | 2.8939; 5728 (+512 cold) | Plain d=16: 2.7967; 5184 | Negative: slightly worse than plain. The test only requires PLE within 0.15 of plain (R-wave5). |

### AugmentedResidualDepth (W5 1.11: LAuReL)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| LAuReL-RW+LR (learned residual weight plus low-rank term), depth 6, 150 steps | Validation loss; parameters | 3.0008; 17,112 (+1560) | Plain residual: 3.0000; 15,552. Value residual: 2.8729; 15,557 | No gain: at this budget every model except value residual stays stuck at about 3.0 (R-wave5). |
| LAuReL-PA (previous activations) | Same | 3.0000; 15,618 | Same | Identical to plain. |

### MoeRefinements: compute per token (W5 1.13: ReMoE, 1.14: MoE++)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| ReMoE (ReLU routing, target 2 experts) | Validation MSE; FFN experts per token; load CV | 1.4649; 1.99; 0.628 | `FineGrainedMoe` top-2: 0.6333; 2; 0.426. Expert choice c=2: 0.7061; 2; 0.000. Zero predictor: 2.8902 | Negative: reaches the sparsity target at more than twice top-2's error. It needed a router bias; without one, zero-mean inputs keep at least half the gates open (R-wave5). |
| MoE++ (8 FFN experts + 4 zero-computation experts: zero, copy, constant) | Validation MSE; FFN experts per token | 0.7005; 1.76 | Top-2: 0.6333; 2 | 12% fewer FFN calls for a small loss cost. |

### MlaConversion (W5 4.10: TransMLA / MHA2MLA conversion)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Convert a trained GQA (one KV head) to `LatentAttention`, latent 12 (partial RoPE, activation-aware joint KV SVD, R-wave5) | Validation loss right after conversion → after 40 steps; cached values per token | 2.374 → 2.171; 32 | Original GQA: 2.289, and 2.143 after 40 more steps. MLA from scratch, latent 12: 2.198 | Conversion costs loss at first; after 40 steps it beats a from-scratch MLA but not the GQA it came from. |
| Same, latent 8 | Same | 2.561 → 2.334; 24 | MLA from scratch, latent 8: 2.510 | Beats from-scratch MLA and caches 25% less than the original. |


### SetLearning (W6 2.11: Deep Sets, Set Transformer)
Sources: Run-w6, R-wave6. Trained on sets of 3–8 elements. Relation networks are under vision (RelationalShapes).

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Deep Sets (sum pooling) | Max: MSE at n = 8 / n = 12 | 0.003 / 0.003 | Set Transformer: 0.008 / 0.007. Sorted, padded MLP: 0.027 / 0.146 | Sum pooling carries the max to larger sets; the MLP does not. |
| Set Transformer (ISAB, PMA) | Amortised clustering: centre error at n = 8 / n = 12 | 0.331 / 0.305 | Deep Sets: 0.594 / 1.063. Sorted MLP: 0.385 / 0.957 | Attention pooling clusters best and holds at n = 12. |
| Both | Unique count: accuracy at n = 8 / n = 12 | Deep Sets 0.430 / 0.333. Set Transformer 0.343 / 0.090 | Sorted MLP: 0.867 / 0.173 | Negative: the sorted MLP wins in distribution, and no model extrapolates. |

### PerceiverExperiments (W6 2.12: Perceiver, Perceiver IO)
Sources: Run-w6, R-wave6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Perceiver, raw-pixel shapes, 300 steps | Accuracy (multiply-adds) | 4 latents 0.387 (152,768). 16 latents 0.387 (389,312) | ViT: 1.000 (54,272) | Negative: the Perceiver learns slowly and costs more. It reaches 0.52 at 1000 steps (R-wave6). |
| Perceiver IO, reverse 64 tokens (vocabulary 8), 200 steps | Accuracy (multiply-adds) | 8 latents 0.184 (413,696). 32 latents 0.211 (991,232) | Full transformer: 0.882 (1,589,248) | Negative: the latent bottleneck limits what passes, and more latents help only a little. |

---

## classic

New theme in wave 6 (package `experiments.classic`): classic machine learning on host-side, double-precision blocks in the new `experimental.ml` package. `ClassicMlShootout` provides the shared spiral, circle, moon, blob, anisotropic and noisy-blob datasets.

### ClassicMlShootout (W6 1.1: kernel SVM and random Fourier features, 1.6: k-means++ and Gaussian mixtures, 1.14: spectral clustering and DBSCAN)
Sources: Run-w6, R-wave6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Kernel SVM (SMO, RBF kernel), spirals | Test accuracy (fit time, support vectors) | 0.987 (8 ms, 92) | MLP: 0.980 (1027 ms). Logistic regression: 0.540. Linear SVM: 0.540 | Matches the MLP at about 1/100 of the training time; linear models fail. |
| Polynomial SVM (degree 3); random Fourier features + linear model | Same | Polynomial 0.927 (6 ms). RFF 0.987 (15 ms, 300 features) | Same | RFF matches the exact RBF kernel. |
| Same lineup on circles, plus a GP-Laplace classifier | Test accuracy | SVM-RBF, SVM-poly, RFF, GP-Laplace and MLP all 1.000 | Logistic: 0.450. Linear SVM: 0.520 | Every kernel machine solves circles. |
| k-means++ seeding, 12 blobs, 10 seeds | Mean ARI | 0.945 (R-wave6) | Random seeding: 0.772 (R-wave6) | ++ seeding avoids bad local optima. On the test's 6-blob set both reach 1.000. |
| Gaussian mixture (EM, full covariance, BIC) | ARI on anisotropic blobs | 1.000. BIC picks the true 3 components (R-wave6) | k-means++: 0.328. Spectral: 0.970. DBSCAN: 0.904 | Full covariances fix stretched clusters. |
| Spectral clustering; DBSCAN | ARI on moons / rings | Spectral 1.000 / 1.000. DBSCAN 1.000 / 1.000 | k-means++: 0.275 / -0.003. GMM: 0.290 / 0.053 | Graph and density methods solve non-convex clusters that centroid methods cannot. Spectral is the slowest (380–1260 ms, against at most 5 ms). |
| DBSCAN, noisy blobs | ARI; share of planted noise points marked as noise | 0.924; 0.850 | k-means++: 0.823. GMM and spectral: 0.832 | Density clustering handles noise best. |

### GaussianProcessUncertainty (W6 1.2: GP regression and classification, 1.3: sparse variational GP and deep kernel learning)
Sources: Run-w6, R-wave6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Exact GP (marginal-likelihood fitting), regression with a gap in the data | NLL inside the range / in the gap; coverage in the gap | -0.66 / -0.22; 0.95 | Deep ensemble: -0.61 / 0.11; 0.98. MC dropout: -0.43 / 0.34; 1.00. Last-layer Laplace (regression version): -0.68 / 0.75; 0.84 | The GP is the best calibrated in the gap. Laplace is sharp on the data but overconfident in the gap. |
| SVGP (natural-gradient SVI), 5k points | Test NLL with M = 4 / 8 / 16 / 32 inducing points | -0.462 / -0.804 / -0.865 / -0.865 | Exact GP on 1,000 points: -0.826 | From 16 inducing points on, SVGP on all the data beats the exact GP on a subset. |
| Deep kernel learning (RBF GP on an MLP feature map), seasonal one-step forecast | NLL; RMSE; coverage | -0.697; 0.113; 0.99 | MLP alone: -0.738; 0.114; 0.92. GP on raw windows: -0.778; 0.111; 0.95 | Negative: DKL does not win here. |

### HyperparameterSearchComparison (W6 1.4: GP Bayesian optimisation, BOHB)
Sources: Run-w6, R-wave6. The methods live in `experimental.train.ModelBasedSearch`, because core `HyperparameterSearch` is final.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| GP Bayesian optimisation with EI / LCB, 30 trials | Best score (floor 0.01, lower is better) | EI 0.049. LCB 0.066 | TPE: 0.141. Random: 0.177 | A GP surrogate finds a configuration about 3x better than TPE at equal trials. |
| BOHB, equal total budget (1,269) | Best full-budget score | 0.044 | Hyperband: 0.106. Random: 0.204 | A model per budget makes Hyperband 2.4x better. |

### ForestBenchmarks (W6 1.5: random forest, extra trees, isolation forest)
Sources: Run-w6, R-wave6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Random forest; extra trees, synthetic tabular task (`SyntheticRecsys.tabular`) | Test accuracy | Random forest 0.830 (out-of-bag error 0.194). Extra trees 0.827 | GBT: 0.849. MLP: 0.894 | Negative: the MLP wins on this multiplicative task. The out-of-bag error tracks the test error. |
| Isolation forest, planted outliers | AUROC | 0.967 | kNN distance: 0.992. Autoencoder: 0.994 | Good, but not the best detector here. |

### ProjectionExperiments (W6 1.7: PCA, kernel PCA, FastICA, 1.8: t-SNE and UMAP)
Sources: Run-w6, R-wave6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| FastICA, cocktail-party mix | Source recovery, \|corr\| | 1.000 | PCA: 0.785 | ICA unmixes the sources; PCA cannot. |
| Kernel PCA (RBF, 3 components), circles, then a linear SVM | Test accuracy | 1.000 | PCA with 2 components: 0.557 | The radial direction is the third component, so this row uses three (R-wave6). |
| UMAP; t-SNE, 2-D embeddings of digit pixels | Trustworthiness / 5-NN accuracy | UMAP 0.839 / 0.747. t-SNE 0.721 / 0.727 | PCA-2: 0.669 / 0.177. Input space: 1.000 / 0.827 | Both keep neighbourhoods far better than PCA; UMAP is the most trustworthy. |
| Same, ShapeImages pixels | Same | UMAP 0.841 / 0.313. t-SNE 0.720 / 0.388 | PCA-2: 0.724 / 0.304. Input space: 1.000 / 0.400 | On raw shape pixels t-SNE gains nothing over PCA in trustworthiness. |

### SequenceLabelling (W6 1.9: hidden Markov models, 1.10: linear-chain CRF)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| HMM fit by Baum-Welch, dishonest casino | Decoding accuracy; log-likelihood per step | 0.784; -1.732 | True parameters: 0.799; -1.730. Supervised GRU tagger: 0.789. i.i.d. model: -1.747 | Unsupervised EM matches a supervised GRU. |
| Gaussian HMM fit by Baum-Welch, regime switching | Same | 0.973; -1.452 | True parameters: 0.974; -1.452. GRU: 0.972. i.i.d. model: -1.715 | Same. |
| Linear-chain CRF head on a GRU, BIO chunking | Span F1; illegal-transition rate | 0.998; 0.000 | Softmax head on the same GRU: 0.285; 0.129 | Transition scores make the tagger emit only legal sequences and lift F1 from 0.29 to 1.00. |

### FactorGraphInference (W6 1.11: variable elimination, loopy belief propagation, Gibbs sampling)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Loopy BP; Gibbs, sprinkler network | Mean marginal error (time) | BP 0.032 (0.3 ms). Gibbs 0.023 (4.8 ms) | Variable elimination: exact | Both are close; Gibbs is slightly better on this loopy graph. |
| Same, alarm network (a tree) | Same | BP 0.0000. Gibbs 0.0035 | Same | BP is exact on a tree, as expected. |
| Same, 4x4 spin glass | Same | BP 0.0004 (2.6 ms). Gibbs 0.0081 (16.3 ms) | p-bit machine (`PBitIsingMachine`): 0.0076 (4.2 ms). Variable elimination: exact (5.7 ms) | Loopy BP is the best approximation here, 20x more accurate than sampling. |

### TopicRecovery (W6 1.12: LDA, NMF)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| LDA (collapsed Gibbs); NMF, synthetic corpus with planted topics | Topic cosine; perplexity | LDA 0.995; 35.28. NMF 0.998; 34.85 | k-means on bag of words: 0.904; 41.23. True topics: 1.000; 34.84 | Both recover the planted topics almost exactly; NMF reaches the true perplexity. |

### PrototypeMaps (W6 1.13: self-organising map, growing neural gas)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Growing neural gas; SOM, 36 prototypes, S-curve | Quantisation error; topographic error | GNG 0.307; 0.000. SOM 0.378; 0.090 | k-means++: 0.300 (no topology) | GNG keeps the topology exactly at close to k-means quantisation error. |
| Same, rings | Same | GNG 0.082; 0.000. SOM 0.127; 0.083 | k-means++: 0.087 | GNG even beats k-means on quantisation error. |

### TrackingFilters (W6 1.15: Kalman filter, EKF, particle filter)
Sources: Run-w6, R-wave6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Kalman filter; RTS smoother, constant-velocity tracking | RMSE; NLL | Kalman 0.234; -0.07. RTS 0.122; -0.71 | Raw observations: 0.488. Causal GRU: 0.255; 0.06 | The exact filter beats a trained GRU, and smoothing halves the error again. |
| Bootstrap particle filter; EKF, pendulum observed through sin θ | Same | Particle filter 0.096; -1.17. EKF 0.152; 0.04 | GRU: 1.481; 1.84 | The particle filter handles the ambiguous observation best. The GRU is weak at 300 training steps (R-wave6). |

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

### KvQuantization (W4 4.5: KIVI 2-bit KV cache)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| KIVI 2-bit (keys per channel, values per token, full-precision recent window), 384 steps with outlier key channels (R-wave4) | Relative attention-output error; KB held | 0.453; 32 KB | float32: 0; 192 KB. H2O (`EvictingKvCache.heavyHitters`) at the same 32 KB: 1.213 | At equal memory, quantizing beats evicting by a wide margin. |
| KIVI 4-bit | Same | 0.084; 43 KB | Same | About a fifth of the 2-bit error for 11 KB more. |
| 2-bit with per-token keys | Same | 1.667; 32 KB | KIVI 2-bit: 0.453 | Per-channel key quantization is what makes 2-bit work when keys have outlier channels. |


### HeadBudgetComparison (W5 4.4: Quest, 4.5: DuoAttention, 4.6: PyramidKV / Ada-KV, 4.7: KVzip)
Sources: Run-w5.

All rows use a synthetic 2-layer × 4-head context at a 25% budget (384 of 1536 entries).

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| PyramidKV / Ada-KV budget allocation | Recall; error on the question used for scoring | 1.00; 0.004 / 1.00; 0.003 | SnapKV (uniform): 1.00; 0.009. H2O: 1.00; 0.024. StreamingLLM: 0.08; 0.802 | Allocating budget across layers and heads gives the lowest error on the scoring question. |
| Same | Recall; error on other questions | 0.32; 0.683 / 0.53; 0.457 | SnapKV: 0.47; 0.546. H2O: 0.00; 0.898. StreamingLLM: 0.07; 0.802 | Query-specific scoring breaks down on questions it was not scored for. |
| KVzip (context-reconstruction scoring) | Recall; error, scoring question / other questions | 1.00; 0.124 / 1.00; 0.119 | Same | Query-agnostic: keeps full recall on any question, at a higher error. |
| DuoAttention (retrieval heads keep everything, streaming heads a window) | Same; entries kept | 1.00; 0.026 / 1.00; 0.025; 552 entries | Others at 384 entries; full cache 1536 | Full recall on any question at 36% of the full cache. |
| Quest (query-aware page selection, nothing evicted) | Same | 1.00; 0.009 / 1.00; 0.008 | Same | Best on both: it reads a subset per query but keeps the whole cache. |
| Quest on `PagedLlama` (4-slot pages, 16 pages) | Top-1 agreement with full attention; share of KV bytes read | 0.44; 0.23 | Recent window: 0.31; 0.23 | Better agreement at the same bytes read. |


### LongContextNeedle (W6 6.15: MemAgent, recursive language models)
Sources: Run-w6, R-wave6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Recursive LM (calls itself on slices of the context) | Needle accuracy at 4x / 8x / 16x / 32x the window (calls at 32x) | 0.85 / 0.72 / 0.85 / 0.92 (63) | Truncation: 0.20 / 0.13 / 0.07 / 0.10 (1). Pager retrieval (`VirtualContextPager`): 0.90 / 0.85 / 0.95 / 0.98 (1) | RLM reads far past the window, but retrieval does as well with one call. |
| MemAgent (overwrites a fixed memory chunk by chunk), SFT then RL | Same | RL 0.55 / 0.45 / 0.58 / 0.58 (33). SFT only 0.30 / 0.13 / 0.12 / 0.22 (33) | Same | RL roughly doubles MemAgent's accuracy, but it stays well below retrieval and RLM. |

### CartridgeCorpusQa (W6 6.16: cartridges)
Sources: Run-w6, R-wave6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Cartridge (a KV cache trained by self-study distillation) | Corpus QA accuracy at 512 / 1024 / 2048 KV bytes | 0.63 / 0.88 / 0.88 | Full context (4096 B): 0.88. Recency eviction: 0.13 / 0.25 / 0.38 | A quarter of the KV bytes matches the full context. |
| Prefix tuning on gold labels | Same | 0.88 / 1.00 / 1.00 | Same | Beats the cartridge, because it learns from gold labels while distillation copies the teacher's mistakes. SnapKV and KVzip baselines were not possible with `HookedDecoder` (R-wave6). |

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

### DifferentiablePhysicsControl (W4 6.6: differentiable physics)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Backprop through a mass-spring "flexible arm" simulator (BPTT), 2000 episodes | Mean final distance to the goal (same MLP for all) | 0.024 | CEM over weights: 0.240; core PPO: 0.691; untrained: 2.423 | Gradients through the physics give a controller 10x closer than CEM and nearly 30x closer than PPO at equal samples. |


### ChunkedRobotPolicies (W5 6.9: diffusion policy, 6.10: flow-matching action expert with real-time chunking)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Diffusion Policy (action chunks), new multimodal-demonstration reach task | Success rate | 0.88 | Demonstrator: 1.00. MSE behaviour cloning: 0.20. Gaussian BC: 0.63. Flow action expert: 0.70 | Handles the multimodal demonstrations far better than unimodal cloning. |
| Real-time chunking (RTC) with the flow action expert, delay 2, period 4 | Success; action jump at the chunk switch | 0.85; 0.153 | Naive chunk switching: 0.68; 0.212. Temporal ensemble: 0.70; 0.144 | Highest success, with a switch jump close to ensembling's. |

### ActionTokenPolicies (W5 6.11: FAST action tokenizer)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| FAST tokenizer (DCT + BPE), autoregressive policy on multimodal reach | Success; tokens per 8x2 chunk | 0.55; 5.1 | Binning tokenizer: 0.60; 16 | A third of the tokens at slightly lower success. A larger 65 s run gave 0.83 against 0.57 (R-wave5). |
| FAST tokenizer alone (`FastActionTokenizerTest`) | Tokens per 16x2 chunk; max absolute error | 3.8; 0.039 | Binning: 32; 0.004 | About 8x compression with a small reconstruction error. |

### ContinuousOfflineRl (W5 6.12: flow Q-learning and Q-chunking)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Flow Q-learning (FQL), offline WindyReach | Normalized return | 0.63 | Data: 0.20. BC: -0.02. Flow BC: 0.33. IQL: -0.06. CQL: 0.99 | Beats BC, flow BC and IQL, but CQL is far better. FQL failed on MultimodalReach, so it is measured on WindyReach (R-wave5). |
| FQL with Q-chunking | Same | 0.49 | FQL: 0.63 | Chunking hurts here. |

### ContrastiveGoals (W5 6.13: deep contrastive goal-conditioned RL)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Contrastive RL, depth 4 / depth 16, bit flipping (6 bits, 2000 steps) | Success rate | 0.85 / 0.80 | HER + DQN: 1.00 | Depth does not help at this scale, and HER wins. At 8 bits: 0.49 / 0.46 against 0.9 (R-wave5). |

### ScalableRlNetworks (W5 6.14: SimBa networks + HL-Gauss value classification)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| SimBa critic + HL-Gauss value loss, Pendulum SAC, 1500 steps | Return | -323.8 (10,323 critic params) | MLP + MSE: -360.1 (1249). Random: -357.0 | The only setup above random at this budget. |
| 2x2 ablation at 2500 steps (R-wave5) | Return | SimBa + HL-Gauss -456; SimBa + MSE -410 | MLP + MSE -580; MLP + HL-Gauss -516 | The SimBa network helps most; HL-Gauss helps the MLP but not SimBa. |


### TrustRegionComparison (W6 3.1: TRPO)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| TRPO (conjugate gradient, exact Fisher-vector products, line search), CartPole | Return (KL per update, mean / max) | 200.0, minimum 200 (0.0138 / 0.0180) | PPO: 195.1. A2C: 107.3 | TRPO solves CartPole and keeps every step inside the trust region. |

### DistributedActorCritic (W6 3.2: A3C, 3.3: IMPALA V-trace and Retrace)
Sources: Run-w6, R-wave6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| A3C (worker threads, shared Adam), CartPole, 40k steps | Return (wall clock) | 200.0 (356 ms) | Synchronous A2C with 4 environments: 196.4 (240 ms) | A3C solves it but is slower here, and it is nondeterministic. The report's run measured 1467 vs 471 ms on a loaded machine (R-wave6). |
| V-trace, actor policy lag 0 / 8 / 32 | Return | 194.3 / 181.3 / 152.9 | Uncorrected: 194.3 / 193.9 / 200.0. 1-step importance sampling: 194.3 / 167.5 / 167.2 | Negative: the uncorrected learner copes better with lag on this task. |
| Retrace | Q^π RMSE after 3 / 40 passes | 0.160 / 0.000 | Uncorrected 8-step: 1.626 / 1.654. 1-step expected SARSA: 0.229 / 0.000 | Retrace corrects off-policy returns and converges fastest; uncorrected n-step stays biased. |

### PartialObservability (W6 3.4: R2D2)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| R2D2 (stored recurrent state, burn-in), cued T-maze of length 4 | Success | 0.88 | Zero-state replay: 0.00. Frame-stack DQN: 0.52 | Stored state with burn-in is what lets the recurrent Q-network remember the cue. |

### RiskSensitiveDqn (W6 3.5: IQN with CVaR acting)
Sources: Run-w6. The maze row is measured by `ImplicitQuantileMazeTest`.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| IQN, maze | Greedy return | 0.93 | C51: 0.93. QR-DQN: 0.93 | IQN matches the other distributional DQNs. |
| IQN acting on CVaR(0.25), windy bridge | Fall rate; bridge use; return | 0.000; 0.00; 0.700 | Risk-neutral IQN: 0.105; 1.00; 0.790 | CVaR acting avoids the bridge entirely, trading 0.09 return for no falls. |

### LatentPlanningComparison (W6 3.7: PlaNet)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| PlaNet (RSSM with latent overshooting, CEM in latent space), Pendulum from 2000 random steps | Return; 10-step open-loop error | -516; 0.209 | RSSM without overshooting: -625; 0.220. MLP dynamics + CEM with the true reward: -512; 0.386 | Overshooting helps. PlaNet ties an MLP planner that is given the true reward, with about half its multi-step error. |

### TdMpcComparison (W6 3.8: TD-MPC2)
Sources: Run-w6, R-wave6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| TD-MPC (MPPI with a learned terminal value), Pendulum, 1000 steps | Greedy return | -380 | SAC: -736. MLP dynamics + CEM (true reward): -499. TD-MPC's policy prior alone: -506 | The most sample-efficient method here; planning adds 126 over its own policy. In an untimed 1500-step run: -127 vs -616 for SAC (R-wave6). |

### ShortcutDreaming (W6 3.9: Dreamer 4 shortcut world model)
Sources: Run-w6, R-wave6. An MLP stands in for the transformer, on the bimodal SlipperyTrack, from offline data only.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Shortcut world model (conditional shortcut flow) | Distance from a sampled next state to the nearest real outcome, at 1 / 2 / 4 / 8 flow steps | 0.052 / 0.037 / 0.024 / 0.023 | Plain flow matching: 0.116 / 0.146 / 0.087 / 0.055. MLP world model: 0.085 | Sharp in one step, where flow matching needs 8 steps to come close. |
| Policy trained only in imagination | Real return | -16.6 | MLP world model: -25.8. Random: -54.0 | The better model gives the better imagined policy. |

### HighUtdComparison (W6 3.10: REDQ, DroQ)
Sources: Run-w6, R-wave6. UTD 5 instead of 20, to fit the time limit (R-wave6).

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DroQ (dropout and layer-norm critics), UTD 5, Pendulum, 1000 steps | Return; Q bias against Monte Carlo | -355; +0.71 | SAC UTD 1: -759; +0.93. SAC UTD 5: -698; +0.79 | DroQ makes a high update ratio pay off: about twice the return of SAC. |
| REDQ, UTD 5, 5 critics | Same | -654; +0.79 | Same | A modest gain over SAC. |

### MpoComparison (W6 3.11: MPO)
Sources: Run-w6, R-wave6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| MPO, Pendulum, 1500 steps, 2 seeds | Greedy return | -790 ± 27 | SAC: -631 ± 5. PPO: -677 ± 20 | Negative: MPO is the worst at this small budget. At 4000 steps it reached -482 (R-wave6). |

### OptionsAndTransfer (W6 3.12: option-critic)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Option-critic, four rooms after the goal moves | Mean length of the first 30 episodes; greedy length | 190.6; 8.0 (mean option duration 2.0) | Q-learning: 150.5; 8.0. A2C: 158.8; 8.0 | Negative: options do not speed up re-adapting, and they stay short. |

### SuccessorFeatureTransfer (W6 3.13: successor features and GPI)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Successor features + generalised policy improvement, zero-shot on a new reward | Return | 0.700 (also 0.700 with inferred weights) | Q-learning from scratch: -1.600 at 300 steps, 0.617 at 4000 steps | Transfer is instant and beats 4000 steps of learning from scratch. |

### ConstrainedRlComparison (W6 3.14: CPO, PPO-Lagrangian)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| PPO-Lagrangian; CPO, hazard grid with cost limit 0.2 | Return; cost | PPO-Lagrangian 0.80; 0.02. CPO 0.80; 0.26 | PPO with fixed shaping k = 0: 0.92; 1.01. k = 0.1: 0.92; 1.00 | Both cut the cost sharply for 0.12 of return. CPO slightly overshoots the limit. |

### RewardMachineTasks (W6 3.15: reward machines, CRM)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| CRM (counterfactual experience over the reward machine), office task "coffee, then mail, then office" | Greedy success at 5k / 20k / 60k steps | 0 / 1 / 1 | Q-learning on the product: 0 / 0 / 0. History-free Q-learning: 0 / 0 / 0 | Counterfactual sharing solves the sequential task; the baselines never do. |

### MetaRlBandits (W6 3.16: RL², VariBAD)
Sources: Run-w6, R-wave6. Bandits only; the goal-in-an-unknown-corner grid was not built (R-wave6).

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| VariBAD; RL², 5-armed bandits with one good arm | Mean reward per pull over 20 pulls of a new task | VariBAD 0.68. RL² 0.58 | MAML policy gradient: 0.24. Core PPO from scratch: 0.21. Random: 0.20 | Recurrent meta-RL explores in context; VariBAD's belief adds 0.10. |

### InContextBandits (W6 3.17: algorithm distillation)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Algorithm distillation (causal transformer on UCB1 learning histories) | Reward per pull in the first third / last third / all of 30 pulls | 0.37 / 0.66 / 0.54 | Source UCB1: 0.36 / 0.71 / 0.56. Distillation of expert histories only: 0.20 / 0.19 / 0.20 | The transformer improves in context with no weight updates, almost matching its source. Expert data alone teaches nothing. |

### GumbelSearch (W6 3.18: Gumbel AlphaZero)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Gumbel MCTS (sequential halving, completed-Q policy), tic-tac-toe | Score against PUCT at 2 / 4 / 8 / 16 / 32 simulations | 0.50 / 0.38 / 0.60 / 0.55 / 0.46 | PUCT MCTS (0.50 is parity) | Null: roughly parity at every budget. The score against random rises from 0.72 to 0.90 (PUCT: 0.66 to 0.92). |

---

## data

New theme in wave 5 (package `experiments.data`): data curation, synthetic data and benchmark methodology.

### DataCuration (W5 2.11: Rho-1, 2.12: quality classifier, 2.14: RegMix, 2.16: EntiGraph)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Rho-1 selective token loss (`SelectiveTokenLoss`), counting with 30% noise tokens | Loss on clean tokens | 0.089 | Full-token cross-entropy: 0.576. EL2N pruning: 0.145 | Training only on useful tokens cuts the clean loss more than 6x. |
| Quality classifier (`QualityClassifierFilter`, hashed 1–3-grams), keep 50% of a 40%-clean pool | Clean loss; clean share of kept data | 1.899; 59% | No filter: 2.084; 40%. Perplexity filter: 1.705; 77% | Partial negative: it helps, but the perplexity filter is better. |
| RegMix (`RegressionMixer`) | Target loss; mixture weights | 2.175; [0.04, 0.42, 0.02, 0.51] | Uniform: 2.482. DoReMi: 2.482 with [0.25, 0.24, 0.25, 0.26] | RegMix finds a better mixture; DoReMi stays at uniform. |
| EntiGraph synthetic corpus (`EntityGraphAugmenter`) | Fact QA accuracy forward / reversed / two-hop / overall | 1.00 / 1.00 / 1.00 / 1.00 | Raw corpus: 1.00 / 0.25 / 0.00 / 0.50. Paraphrases: 1.00 / 0.00 / 0.00 / 0.40 | Entity-pair rewrites teach reversed and two-hop facts that raw text and paraphrases do not. |

### ShapleyPruning (W5 2.13: In-Run Data Shapley)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| In-Run Shapley, first / second order, 20% of labels flipped, 20% of examples removed | Share of flipped labels among removed examples; test accuracy | 0.35 / 0.43; 0.685 / 0.685 | EL2N: 0.56; 0.675. Random: 0.20; 0.690 | Negative: above chance, but EL2N finds more flipped labels, and no method beats random pruning on accuracy. |
| Ghost dot products (`InRunShapleyTest`) | Max error against explicit per-example gradients | 2.24e-9 | Explicit per-example gradients | The fast estimate is exact for dense layers. |

### ModelCollapse (W5 2.15: model collapse, replace vs accumulate)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Accumulate: each generation trains on all real and synthetic data so far | Test loss at generation 5; output diversity | 2.433; 1.79 | Replace (each generation trains only on the last one's output): 5.108; 0.67. Generation 0: 2.210; 1.33 | Accumulating data avoids collapse; replacing it more than doubles the loss. |

### BenchmarkMethodology (W5 2.17: Fluid Benchmarking, 2.18: Min-K%++ contamination detection)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Fluid Benchmarking (IRT with adaptive item selection, `IrtEvaluator`), 10% of items | Kendall τ with true model quality | 0.725 | IRT on random items: 0.447. Accuracy on a random subset: 0.311. All items: IRT 0.851, accuracy 0.895 | Adaptive IRT ranks models far better than a random subset of the same size. |
| Same | Share of training steps where the score drops | 0.14 | Random subset: 0.27. True quality: 0.18 | Less noisy than subset accuracy. |
| Min-K%++ (`ContaminationDetector`) | Contamination AUROC | 0.746 | Plain Min-K%: 0.755. Loss threshold: 0.702 | Negative: beats the loss but not plain Min-K%. |


### TrainingAttributionExperiment (W6 5.16: influence functions, TracIn)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Influence functions; TracIn | Precision on flipped labels; leave-60-out Pearson r (time) | Influence 0.650; 0.849 (2.4 s). TracIn 0.550; 0.791 (0.6 s) | In-Run Shapley: 0.217; 0.675. EL2N: 0.450; -0.273. Random flagging: 0.10 | Influence functions find flipped labels and predict removals best; TracIn is close at a quarter of the time. |

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

### EvolvedKernels (W4 5.5: AlphaEvolve-style program evolution)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Program evolution with a MAP-Elites archive and an inspiration-aware proposer (a simulated, data-flow-aware writer), softmax kernel, 8000 evaluations (R-wave4) | Fitness (half budget) | 12.09 (7.25); finds the stable max-subtract kernel with a reciprocal multiply | Blind mutation: 1.46 (1.46); regularized evolution (AutoML-Zero): 1.46; naive softmax kernel: 5.80 | Only the guided proposer beats the hand-written naive kernel. |
| Same, matmul loop nest through a simulated LRU cache | Cache misses | 896 | Blind mutation: 1280; regularized evolution: 896; textbook ijk: 33,024 | Ties regularized evolution; both beat the textbook loop order by far. Kernels are evolved as DSL programs, not Java source (R-wave4). |

### SelfImprovingCoders (W4 5.6: Darwin Gödel Machine, toy)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Agents that edit their own synthesizer scaffold, kept in an open-ended archive, 25 generations (101 agents) | Benchmark score | 0.561 (scaffold length 4, with pruning) | Greedy hill-climbing: 0.504 (stuck at length 3, no pruning); start: 0.431 | The archive crosses a valley that greedy search cannot. The valley comes from a compute penalty chosen so single edits score worse (R-wave4). |

### LearnedLibraries (W4 5.8: DreamCoder library learning)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Wake-sleep library learning, 65 list tasks, 4 rounds | Tasks solved per round | 38, 42, 50, 50, with 14 abstractions learned (276 ms) | Fixed DSL: 38, 38, 38, 38 (126 ms) | Compressing solved programs into new primitives solves 12 more tasks; a fixed DSL never improves. |

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

### SpikingLanguageModel (W4 6.5: SpikeGPT)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| RWKV-style LM with LIF spiking activations and surrogate gradients, counting sequences | Perplexity; share of neurons spiking per token | 1.1311; 0.268 | Non-spiking RWKV: 1.1288 (uniform: 16) | Spiking costs almost nothing in perplexity with about a quarter of the neurons active. |


### KuramotoBinding (W6 6.1: AKOrN)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| AKOrN (Kuramoto oscillatory units), multi-shape images | Object discovery FG-ARI | 0.84 | k-means on (x, y): 0.16. Soft slots on (x, y): 0.15 | Phase synchrony binds objects far better than clustering positions. |
| AKOrN, Latin squares | Blank accuracy; share solved | 0.92; 0.62 | TRM: 0.67; 0.15 | The oscillator network solves 4x as many puzzles as the tiny recursive model. |

### TensorNetworks (W6 6.2: MPS classifier, Born machine)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| MPS classifier, digits | Test accuracy | Bond 4: 0.46. Bond 10: 0.67 (13,440 params) | MLP of the same size: 0.65 (13,435 params) | Matches an MLP at bond dimension 10. |
| MPS Born machine (DMRG, exact sampling), shapes | NLL in nats (ideal 4.34); share of exact samples | Bond 8: 9.26; 0.02. Bond 32: 4.51; 0.83 | PixelCNN: 9.06; 0.01. p-bit Boltzmann machine: 0.00. Independent pixels: 21.66; 0.00 | At bond 32 the likelihood is close to ideal and 83% of samples are exact images. |

### QuantumClassifier (W6 6.3: simulated variational quantum circuit)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Variational circuit with parameter-shift gradients, moons | Accuracy | 2 qubits, 1 layer: 0.84. 4 qubits, 3 layers: 0.99; with 50 shots 1.00 | RFF with 34 params: 1.00 | It works, but a tiny classical model is as good. Circles: 1.00 for both. |
| Barren plateau | Gradient variance | 6 qubits at depth 1 / 4 / 16: 0.0114 / 0.0061 / 0.0073. Depth 12: 2 qubits 0.0870, 8 qubits 0.0019 | n/a | The variance falls 46x from 2 to 8 qubits: the plateau grows with width. |

### StochasticComputing (W6 6.4: stochastic-computing network)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Bitstream inference with an APC adder, stream length 16 / 256 / 1024 | Digits accuracy (bit operations) | 0.51 / 0.91 / 0.91 (77k / 1,234k / 4,936k) | Float: 0.92. int8 post-training quantization: 0.91. Ternary PTQ: 0.76. BitLinear QAT: 0.85 | 256-bit streams match int8. |
| Same with a MUX adder, length 256 / 4096 | Accuracy | 0.33 / 0.77 | Same | MUX scaling loses too much precision. |

### StdpDigits (W6 6.6: STDP, Diehl & Cook)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| STDP with lateral inhibition, 20 / 60 excitatory neurons, no gradients | Digits accuracy | 0.30 / 0.62 | SoftHebb(60) + readout: 0.47. Surrogate-gradient LIF(60): 0.59 | Unsupervised STDP slightly beats the gradient-trained spiking net. |

### PhysicsAwareDiffraction (W6 6.7: physics-aware training)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Physics-aware training (forward through the device, backward through a digital twin) | Accuracy on the device (on the twin) | 0.74 (0.78) | In-silico training: 0.40 (0.99). Noise-aware training: 0.40 (1.00) | Training through the device closes most of the sim-to-real gap. |

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

### CodecLanguageModel (W4 6.2: MusicGen-style codec LM)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Transformer over 3-stage RVQ codec tokens (codec SNR 8.8 dB) with the delay pattern | NLL per code | 2.120 | Flattened RVQ: 2.029 | The flattened LM models the codes slightly better. |
| Same | Continuation code accuracy; SNR; model calls | 0.210; -1.8 dB; 6 calls | Flattened: 0.184; -1.6 dB; 12 calls | Delay halves the calls at about the same quality. |

### MaskGit (W4 6.9: MaskGIT)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| MaskGIT parallel decoding of toy-image VQ tokens, cosine masking | Distance / real share / distinct at 4, 8, 16 steps | 4 steps 0.161 / 0.27 / 17; 8 steps 0.076 / 0.48 / 27; 16 steps 0.056 / 0.58 / 32 | Autoregressive, 16 steps: 0.049 / 0.75 / 33 | Quality rises with steps, but MaskGIT trails the autoregressive model at equal steps. |

### NextScalePrediction (W4 6.10: VAR next-scale prediction)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| VAR over coarse-to-fine multi-scale VQ maps (tokenizer error 0.0287), 3 calls | Distance; real share; distinct; classes | 0.209; 0.09; 7; 3 | MaskGIT, 3 steps: 0.215; 0.15; 13. v-prediction DDPM, 3 DDIM steps: distance 0.238 | Lowest distance at 3 calls, but fewer real-looking and fewer distinct samples than MaskGIT: a mixed result. |

### OneStepGenerators (W4 6.11: MeanFlow and shortcut models)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| MeanFlow, one step, 4-blob ring | Distance; near share; modes | 0.386; 0.55; 4 | Flow matching: 1.300; 0.00; 3. Reflow: 0.940; 0.02; 4. Consistency model: 0.444; 0.39; 4 | Best one-step generator here, ahead of the consistency model. |
| Shortcut model, one step | Same | 0.492; 0.33; 4 | Same | Beats reflow and flow matching, behind the consistency model. |
| MeanFlow JVP (double backward) | Maximum relative error vs central differences | 0.0001 (Run-w4; the report gave 0.0015) | n/a | The JVP is correct. |

### DiffusionTargets (W4 6.12: v-prediction and min-SNR weighting)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| v-prediction, 10 DDIM steps | Distance to the nearest real image per pixel | 0.2686 | ε-prediction: 0.5735 | v-prediction halves the distance. |
| min-SNR weighting | Same | ε + min-SNR 0.5365; v + min-SNR 0.2686 | ε: 0.5735; v: 0.2686 | min-SNR helps ε-prediction a little and v-prediction not at all. |

### NovelViewSynthesis (W4 6.7: 3D Gaussian Splatting)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| 64 3-D Gaussian splats (quaternion covariance, EWA projection, depth sort, alpha compositing), three ray-traced spheres, 8 train and 8 held-out views (R-wave4), 400 steps | Held-out PSNR (train); time | 25.68 dB (26.70); 1.3 s | NeRF-style `FourierFeatures` MLP, 24 samples per ray: 25.59 dB (27.17); 7.3 s. All-black image: 16.84 dB | Same held-out quality as the NeRF baseline in about a fifth of the time. The report gave 2.0 s vs 8.0 s (R-wave4); wall time varies. |

### ThermodynamicDenoising (W4 6.8: denoising thermodynamic models)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DTM: 4 chained conditional Boltzmann stages on `PBitIsingMachine`, 4x4 bars and stripes | Valid share; distinct patterns of 30; MMD | 0.963; 30; 0.0045 | Single Boltzmann machine: 0.973; 30; 0.0059. Thresholded DDPM, 10 steps: 0.145; 26; 0.0110 | Negative: the DTM is only on par with a single machine (slightly better MMD, slightly fewer valid samples). On 6x6 data the single machine was clearly better, 0.903 valid vs 0.413 (R-wave4). |


### DiffusionArithmeticRl (W5 3.12: RL for diffusion LMs, diffu-GRPO and JustGRPO)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| diffu-GRPO on `MaskedDiffusion` | Accuracy on 100 sums, parallel (left-to-right) decoding | 0.47 (0.49) | SFT only: 0.30 (0.36) | Gains in both decoding modes. |
| JustGRPO (RL on the left-to-right policy) | Same | 0.18 (0.56) | Same | Best left-to-right, but the gain does not carry over to one-step parallel decoding, which falls below SFT. |

### BlockDiffusionText (W5 6.1: block diffusion, BD3-LM)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| BD3-LM, block size 4, grammar sentences | Perplexity; tokens per step; valid share | 1.36; 2.00; 0.94 | Same model as AR (B=1): 1.30; 1.00; 1.00. `MaskedDiffusion` (B=16): 1.42; 1.75; 0.72 | Half the decoding steps at a small perplexity cost, and better than full masked diffusion on every count. |

### RepaExperiment (W5 6.2: REPA representation alignment)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| REPA auxiliary loss on ToyImages | Distance per pixel at 150 / 300 / 500 steps | 1.001 / 0.676 / 0.551 | Plain diffusion: 0.989 / 0.649 / 0.560 | Negative: no speed-up at toy scale. |

### GuidanceComparison (W5 6.3: autoguidance and APG)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Autoguidance, w=3 (guide with a weaker model) | Precision; recall (ToyPoints) | 0.77; 0.70 | No guidance: 0.96; 0.92. Classifier-free guidance, w=3: 0.82; 0.08 | Keeps most of the recall that CFG destroys. |
| Adaptive projected guidance (APG), w=3 | Same | 0.99; 0.22 | Same | Highest precision, and nearly 3x CFG's recall. |

### HighDimensionalTargets (W5 6.4: x-prediction, JiT)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| x-prediction, 8x8 images lifted to 512 dimensions, width-96 MLP | Distance per pixel; off-manifold energy | 0.216; 0.00 | ε-prediction: 3.953; 0.87. v-prediction: 0.751; 0.78 | Predicting clean data stays on the data manifold, 3.5x closer than v-prediction. |
| x-prediction trained with a v-loss | Same | 0.665; 0.03 | Same | The loss weighting matters: most of the gain is lost. |

### TarFlowExperiment (W5 6.5: TarFlow transformer autoregressive flow)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| TarFlow (2 blocks, committed config) | Bits per dimension (ToyImages) | 7.245 | RealNVP: 7.303 | Wins on likelihood. A 4-block, 89 s run gave 7.125 against 7.180 (R-wave5). |
| Same | Nearest-real squared distance | 23.05 (24.87 without the denoising step) | RealNVP: 16.86. TinyDdpm at the same budget: 33.35 | Loses to RealNVP on sample quality; the score-denoising step helps. |

### EquilibriumMatchingExperiment (W5 6.6: equilibrium matching)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Equilibrium matching (EqM), ToyPoints ring | Distance to the nearest centre at 1 / 2 / 4 / 16 steps | 0.756 / 0.539 / 0.236 / 0.153 | Flow matching: 1.763 / 0.758 / 0.357 / 0.230. Consistency model: 0.443 / 0.371 / 0.307 / 0.295 | Beats flow matching at every step count, and is the best from 4 steps on. |
| EqM with adaptive stopping | Distance; mean steps | 0.137; 8.4 | Same | The best result overall. |

### MemorizationTimescale (W5 6.8: memorization-onset timescale in diffusion)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Diffusion on a ring of 8 blobs, n = 4 / 16 / 64 training points, 2000 steps | Memorization onset τ_mem; excess copy rate | n=4: 200, 0.26. n=16: never, 0.02. n=64: never, -0.03 | Chance copy rate: 0.08 / 0.20 / 0.13 | Only the 4-point set memorizes within the budget. |
| Same | Generalization time τ_gen | 1400 / 800 / 1200 steps | — | Noisy rather than clearly independent of n, so the paper's separation of timescales is not clearly shown. |


### ImageTranslation (W6 4.1: pix2pix, CycleGAN)
Sources: Run-w6, R-wave6. The CycleGAN row is measured by `CycleGanTest`.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| pix2pix (U-Net + PatchGAN), coarse 4x4 sketch → shape | Real-shape share; L1 | 0.918; 0.031 | L1 loss only: 0.836; 0.034 | The adversarial loss commits to real shapes. The planned edges task was deterministic, so it was replaced (R-wave6). |
| CycleGAN, squares → crosses, unpaired | Target share; centroid error; cycle L1 | 0.880; 0.855; 0.029 | No cycle loss: 0.000; 1.874; 0.681 | Cycle consistency keeps the content and makes the translation work at all. |

### StyleGanExperiment (W6 4.2: StyleGAN-lite)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| StyleGAN (mapping net, modulated layers, noise, style mixing) | Sliced W; distance; real share; path length | 0.209; 0.263; 0.01; 305 | DCGAN-style MLP: 0.139; 0.206; 0.12; 132 | Negative: the plain generator is better at this budget. |
| StyleGAN + path-length regulariser | Same | 0.311; 0.375; 0.00; 73 | Same | Cuts the path length 4x, at a quality cost. |

### VqGanExperiment (W6 4.3: VQ-GAN)
Sources: Run-w6, R-wave6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| VQ-GAN, 4 codes, 16 tokens per image | MSE; real share; sharp share; code perplexity | 0.109; 0.479; 0.932; 3.62 | VQ-VAE with 4 codes: 0.100; 0.575; 0.917; 2.82. With 16 codes: 0.027; 0.973; 0.947; 8.00 | Mixed: slightly sharper with better code use, but fewer real reconstructions. MaskGIT on the tokens was not run (R-wave6). |

### DensityBenchmark (W6 4.4: Glow, MAF, neural spline flows, 4.5: FFJORD)
Sources: Run-w6, R-wave6. The FFJORD row is measured by `ContinuousFlowDensityTest`.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Neural spline flow | Held-out NLL in nats, moons / rings | 0.763 / 1.290 | RealNVP: 1.209 / 2.367. GMM-8: 0.704 / 1.833 | Spline flows beat RealNVP clearly, and the GMM on rings. |
| Glow; MAF | Moons NLL | Glow 1.250. MAF 1.679 | Same | Negative: neither beats RealNVP. |
| FFJORD, 80 steps | Moons NLL (evaluation NFE) | 1.974 (127) | Flow matching at 80 steps: 2.204 (109); at 1000 steps: 1.788 (79) | Better at equal steps, not at equal time (8.6 s for FFJORD). |

### DiffusionSamplers (W6 4.6: score SDE, 4.7: EDM and DPM-Solver++)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| VP score SDE, probability-flow ODE | Moons NLL in nats (NFE) | 1.872 (127) | A standard normal: about 2.1 | An exact likelihood from a diffusion model. |
| Score SDE, predictor-corrector; probability-flow ODE | Ring distance at 6 / 10 / 20 NFE | PC 0.734 / 0.487 / 0.420. ODE 0.576 / 0.442 / 0.302 | DDIM: 0.201 / 0.138 / 0.153. Flow-matching Heun: 0.251 / 0.196 / 0.183. DDPM, 100 steps: 0.170 | Negative: both trail DDIM at every budget. |
| EDM (Heun); DPM-Solver++ | Same | EDM 3.751 / 0.406 / 0.142. DPM-Solver++ 0.604 / 0.176 / 0.144 | Same | At 20 NFE both beat DDIM and DDPM-100. At 6 NFE DDIM wins and EDM's Heun sampler breaks down. |

### JemExperiment (W6 4.8: JEM)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| JEM (persistent SGLD), ShapeImages 8x8 | Accuracy; ECE | 0.956; 0.147 | Plain classifier: 0.980; 0.036. EnergyRefiner: 0.992; 0.006 | Negative: JEM is less accurate and worse calibrated. |
| Same | OOD AUROC on pixel-shuffled images | Density 0.756. Max-softmax 0.818 | Plain: 0.807. EnergyRefiner: 0.892 | A small gain from max-softmax; the density score is worse. |
| Same | Sample distance | Persistent buffer 0.343. Fresh chains 0.483 | Noise: 0.369 | Samples are barely better than the noise they start from. |

### LatentDiffusion (W6 4.9: latent diffusion, representation-autoencoder latents)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Latent diffusion in a VAE latent | Distance per pixel at 150 / 400 / 800 steps; real share at 800 | 0.194 / 0.190 / 0.177; 0.20 | Pixel diffusion: 1.014 / 0.615 / 0.436; 0.00 | Far faster: at 150 steps it beats pixel diffusion at 800. |
| RAE (frozen SSL encoder + trained decoder) | Same | 0.197 / 0.205 / 0.176; 0.20 | Same | Ties the VAE latent (reconstruction distance 0.008 vs 0.022). |

### ControlExperiment (W6 4.10: ControlNet)
Sources: Run-w6, R-wave6. The backbone is a residual MLP, not a U-Net (R-wave6).

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| ControlNet (trainable copy joined through zero-initialised layers) | IoU with the occupancy map; unconditional distance after training | 0.615; 0.414 (unchanged) | Full fine-tune: 0.516; changes to 0.282. Concatenated control from scratch: 0.387. Base: 0.227 | The best control, and the frozen base is preserved exactly. |

### OneStepFrontier (W6 4.11: DMD, 4.16: IMM and sCM, 4.18: drifting models)
Sources: Run-w6. Four-blob ring.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Distribution matching distillation, one step | Distance; near share; modes | 0.364; 0.52; 4 | MeanFlow: 0.386; 0.55; 4. EDM teacher at 10 NFE: 0.167; 0.90; 4 | Roughly ties MeanFlow, far from the teacher. |
| Inductive moment matching at 1 / 2 / 4 steps | Distance (near share) | 0.481 (0.39) / 0.254 (0.79) / 0.231 (0.80) | Discrete consistency model: 0.444 (0.39) | Better from 2 steps on. |
| sCM (TrigFlow) at 1 / 2 / 4 steps | Same | 0.697 (0.13) / 0.787 (0.06) / 0.794 (0.11) | Same | Negative: weak, and worse with more steps. |
| Drifting generator, one step | Distance; near share; modes | 0.311; 0.66; 4 | GAN: 0.359; 0.64; 4 | A slightly better one-step generator than the GAN. |

### HierarchicalVaeExperiment (W6 4.12: IWAE, β-VAE, ladder VAE)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| IWAE (k = 5), binary ToyImages | IWAE-500 NLL in nats; active units; disentanglement gap | 9.06; 8; 0.18 | Plain VAE: 9.42; 8; 0.21 | The tighter bound gives a better likelihood. |
| β-VAE (β 4); ladder VAE | Same | β-VAE 16.84; 2; 0.27. Ladder 9.92; 3; 0.28 | Same | Slightly more disentangled, at a large likelihood cost for β-VAE; both lose active units. |

### DiscreteGenerators (W6 4.13: discrete flow matching, 4.14: Bayesian flow networks)
Sources: Run-w6. The image rows are measured by `BayesianFlowImagesTest`.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Discrete BFN, 6-letter words | Valid-word share at 3 / 12 steps | 0.891 / 0.984 | Masked diffusion: 0.641 / 0.875. Autoregressive: 1.000 | BFN beats masked diffusion at both budgets. |
| Discrete flow matching, mask / uniform source, with and without a corrector | Same | Mask 0.406 / 0.688; with corrector 0.031 / 0.609. Uniform 0.297 / 0.422; with corrector 0.359 / 0.266 | Same | Negative: every DFM variant trails masked diffusion. |
| BFN on binary and real-valued ToyImages, 50 steps | Distance per pixel; real share | Discrete 0.071; 0.70. Continuous 0.398; 0.00 | DDIM, 50 steps: 0.395; 0.00. DDPM, 100 steps: 0.249; 0.01 | Discrete BFN works on binary images; continuous BFN only ties DDIM. |

### BridgeExperiment (W6 4.15: stochastic interpolants, DSBM)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Bridge matching (σ 0.5); DSBM (iterative Markovian fitting), moons → rings | Transport cost; sliced W to rings | Bridge 0.774; 0.078. DSBM 0.594; 0.090 | Rectified flow: 0.217; 0.116. Reflow: 0.172; 0.108. OT flow matching: 0.192; 0.102 | Bridges match the target distribution best, but with long paths; DSBM shortens them by a quarter. |

### EnergyMatchingShootout (W6 4.17: energy matching)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Energy matching, moons, 20 sampler steps | Sliced W; NLL in nats | 0.117; 2.223 (grid-normalised) | Flow matching: 0.116; 1.896 (exact ODE). Equilibrium matching: 0.237 | Samples as well as flow matching, but its density is worse. |

### GFlowNetSampling (W6 6.8: GFlowNets)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| GFlowNet (trajectory balance), 8x8 hypergrid | L1 to R/Z; modes found of 4; reward calls | 0.24; 4; 6,400 | MCMC: 0.34; 4; 6,400. PPO: 1.70; 1; 30,000. MAP-Elites: 4 modes; 6,420 | Samples in proportion to the reward; PPO collapses onto one mode. |
| Same, sequences | L1 | 0.81 | MCMC: 0.88 | A small edge. |

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


### TranscoderComparison (W5 5.1: transcoders and skip-transcoders)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Skip-transcoder | Explained variance, superposition MLP k=1 / 2 / 4; toy LM k=4 | 0.788 / 0.824 / 0.867; 0.995 | SAE on the MLP output: 0.724 / 0.838 / 0.911; 0.988. Plain transcoder: 0.612 / 0.728 / 0.840; 0.988 | The skip connection lifts the transcoder above the SAE at k=1 and on the toy LM, but not at k=2 or 4. |

### CircuitTracing (W5 5.2: cross-layer transcoders and attribution graphs)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Cross-layer transcoder + attribution graph | Explained variance; replacement agreement; B-feature recovery; pruned graph nodes | 0.974; 0.996; 0.50; 4.6 | Per-layer transcoders: 0.966; 0.997; 0.75; 4.9 | Negative: slightly better fit, but it recovers fewer of the planted features. The graph is exact (completeness gap 4.4e-8). |

### ModelDiffing (W5 5.3: crosscoders)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Crosscoder over a base and a fine-tuned model | Features exclusive to the fine-tune; best AUROC for the planted feature; precision | 4; 0.858; 0.75 | Separate SAEs: 8 new features; 0.732; 0.00 | The crosscoder isolates the planted change. The diffed model is a planted fine-tune, not PreferenceTuning (R-wave5). |

### SuperpositionRecovery: hierarchical features (W5 5.4: Matryoshka, BatchTopK and JumpReLU SAEs)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Matryoshka SAE, hierarchy toy with 16 true features | Features recovered; absorption; dead share; L0; explained variance | 9; 0.405; 0.53; 1.98; 0.913 | TopK: 4; 0.884; 0.03; 2.00; 0.899. L1: 5; 0.217; 0.00; 7.97; 0.995 | Recovers the most features with far less absorption than TopK, but half its latents are dead. |
| BatchTopK SAE | Same | 5; 0.750; 0.44; 1.99; 0.976 | Same | Better fit than TopK, little less absorption. |
| JumpReLU SAE | Same | 5; 0.313; 0.16; 3.87; 0.996 | Same | Best fit, low absorption, at a higher L0. |

### PatchingAttribution (W5 5.5: AtP and AtP*)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| AtP / AtP* on a recall model (24 nodes) | Spearman correlation with exact patching; top-3 nodes found; time | 0.996 / 0.996; 3 of 3; 4 ms / 10 ms | Exact activation patching: 27 ms | Gradient attribution matches exact patching at a fraction of the cost. The agent's report gave 56 / 13 / 18 ms (R-wave5). |

### MechanismDecomposition (W5 5.6: stochastic parameter decomposition)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| SPD on toy models of superposition | Mechanisms recovered; active components per input; faithfulness error | 1 of 5; 1.80; 0.005 | SAE on activations: 5 of 5 directions | Negative: SPD did not converge in the CPU budget despite extensive tuning (R-wave5). |

### WeightSparseCircuits (W5 5.7: weight-sparse transformers)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Top-k weight-sparse training, bracket matching | Weights in the circuit at accuracy ≥ 0.98 (of 1600) | 93 | Pruned dense model: 390 (both accuracy 1.000) | The weight-sparse model's circuit is 4x smaller. |

### SequentialEditing (W5 5.10: AlphaEdit null-space editing)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| AlphaEdit (ridge 1e-3), 80 sequential edits over 160 subjects | Efficacy; final success; specificity | 0.91; 0.81; 1.00 | ROME: 0.99; 0.86; 0.93 | Trades some edit success for perfect specificity. The null space has 57 dimensions, fewer than the 80 edits (R-wave5). |

### UnlearningComparison (W5 5.11: NPO and SimNPO, 5.12: RMU)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| NPO / SimNPO, memorised exceptions | Forget-set accuracy; test accuracy; membership AUC; accuracy after a relearning attack | 0.00; 0.83; 0.72; 0.05 | Retrain: 0.03; 0.89; 0.53; 0.00. Original: 1.00; 0.79; 1.00; 1.00. Gradient ascent: 0.65; 0.75; 0.84; 0.90. Random labels: 0.00; 0.70; 0.86; 0.45 | The best approximate method, and it resists relearning. NPO and SimNPO coincide here, because the labels have length 1 (R-wave5). |
| RMU (representation misdirection) | Same | 0.25; 0.73; 0.91; 0.35 | Same | Weak, because forget and retain inputs come from the same distribution (R-wave5). |

### MemorizationCapacity (W5 5.18: bits per parameter)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| MLP next-bit models, hidden 4 / 8 / 12 / 16 | Bits memorised per parameter | 1.15 / 1.08 / 1.11 / 0.92 | The paper's about 3.6 bits per parameter for toy GPTs (R-wave5) | Far below the paper (tiny MLPs, short training). Hidden 32 gave 0.64 (R-wave5). |
| Same, quantized | Bits kept at hidden 16, int8 / int4 | 817 / 321 | Float: 823 | int8 keeps nearly everything; int4 loses most. |
| Dataset sweep at hidden 8 | Bits memorised, 10 → 540 strings | 87 → 520 | — | Still rising at the largest set, so capacity is not reached. |


### ExplainabilityBenchmark (W6 5.14: KernelSHAP, LIME, counterfactuals, 5.15: Grad-CAM, TCAV)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| KernelSHAP; LIME, MLP on a planted rule | Ground-truth recovery; deletion AUC (lower is better) | KernelSHAP 0.906; 0.557. LIME 0.975; 0.614 | Integrated gradients: 0.788; 0.606. SmoothGrad: 0.944; 0.631. Saliency: 0.888; 0.654. Random: 0.481; 0.747 | LIME recovers the rule best; KernelSHAP has the best deletion AUC. |
| Same, gradient-boosted trees | Same | KernelSHAP 0.931; 0.574. LIME 0.956; 0.633 | Tree gain: 1.000; 0.630. Random: 0.481; 0.704 | Tree gain recovers the rule exactly; KernelSHAP deletes best. |
| Counterfactual explanations | Validity; features changed; share relevant | MLP 1.00; 6.05; 0.62. Trees 1.00; 2.18; 1.00 | n/a | Always valid; minimal and relevant on trees. |
| Grad-CAM, shapes | Pointing game; how often a spurious model's map points at the marker | 1.000; 0.149 | Integrated gradients: 1.000; 0.596. Saliency: 0.760; 0.000. Random: 0.145; 0.043 | Grad-CAM localises the shape, but rarely shows the shortcut. |
| TCAV, "marker" concept | TCAV share (z against random concepts) | Spurious model 1.000 (z 5.0). Clean model 0.000 (z -0.3) | Random concepts: 0.625 ± 0.484 | TCAV finds the shortcut, but only the z-score against random concepts is significant. |

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

### ActivationSparsityDecoding (W4 1.14: TEAL / CATS activation sparsity)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Training-free magnitude thresholding, 25% sparsity | Validation loss; density; tokens/s; speed-up | 2.821; 0.760; 17,194; 1.16x | Dense: 2.839; 1.000; 14,820 tokens/s | Free at 25%. |
| Same, 40% | Same | 2.854 (+0.015); 0.589; 20,769; 1.40x | Same | A small loss cost for a 1.4x decode speed-up. |
| Same, 50% | Same | 2.925 (+0.086); 0.497; 22,739; 1.53x | Same | The loss cost grows quickly past 40%. Speed-ups are wall-clock on a shared machine; the report saw 1.39–1.79x at 40% (R-wave4), and the test asserts only loss and density. |

### DistillationShootout (W4 3.12: GKD, 3.13: MiniLLM, 3.14: sequence-level KD, 3.15: feature distillation)
Sources: Run-w4, R-wave4.
Teacher 128 wide with accuracy 1.00; student 24 wide, 300 steps (R-wave4).

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| On-policy distillation (GKD, generalised JSD) | Student accuracy; perplexity; teacher tokens | 0.61; 1.872; 24,376 | Offline logit KD (`DistillationLoss`): 0.48; 1.714; 23,572. SFT: 0.49; 1.797; 0 | Clearly the best student accuracy, at a slightly higher perplexity. |
| MiniLLM (reverse KL on the student's own samples) | Same | 0.17; 5.162; 22,725 | Same | Negative: pure reverse KL with no mixed-in pretraining loss collapses the student. |
| Sequence-level KD (greedy and beam teacher outputs) | Same | 0.49; 1.797; 245 | Same | Equals SFT because the teacher is perfect, at about 1% of logit KD's teacher tokens. |
| Feature/hidden-state distillation | Same | 0.50; 1.680; 23,572 | Same | Best perplexity, about level with logit KD on accuracy. |

### LoraScaling (W4 3.17: rsLoRA and LoftQ, PeftShootout additions)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| rsLoRA (α/√r scaling) | New-task loss at rank 2 / 16 | 0.109 / 0.013 | LoRA: 0.152 / 0.079 | Better at both ranks, and the gap widens at rank 16. |
| LoftQ initialisation on a 4-bit base | 4-bit weight error at init | 0.596 | QLoRA: 0.833 | Starts closer to the full-precision weights. |
| Same | New-task loss after 30 steps on the 4-bit base | 0.335 | QLoRA: 0.122; fp32 LoRA: 0.108 | Negative: the better start does not help; LoftQ ends far behind QLoRA. |

### NoisyPreference (W4 3.5: cDPO and rDPO, 3.6: DPOP and β-DPO)
Sources: Run-w4, R-wave4.
`PreferenceTuning.noisy` with 20% of preference labels flipped. The pretrained base strongly prefers the rejected style.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Conservative DPO (label smoothing) | Held-out margin after (before -35.65); chosen log-likelihood drift; absolute win rate | -34.30; +1.04; 0.000 | DPO: -32.07; +1.98; 0.000 | No better than DPO under noise. |
| Robust DPO (unbiased noisy-label correction) | Same | -4.66; +21.12; 0.188 | Same | The only variant that keeps moving: under noise DPO stalls where flipped and clean gradients cancel (R-wave4). |
| DPO-Positive | Same | -31.35; +4.30; 0.000 | Same | Keeps the chosen likelihood higher than DPO, as designed. |
| β-DPO (adaptive β) | Same | -33.23; +1.50; 0.000 | Same | No gain. |

### SamplerShowdown (W4 4.10: top-nσ and XTC, 4.11: entropy-adaptive sampling)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Top-nσ (n = 1), counting decoder at temperature 3 | Accuracy; distinct continuations | 1.00; 0.50 | Top-p 0.9: 0.04; 1.00. Min-p 0.1: 0.31; 0.90. Greedy: 1.00; 0.25 | Stays correct at high temperature with twice greedy's diversity. |
| XTC after min-p | Same | 1.00; 0.42 | Same | Also correct, slightly less diverse than top-nσ. |
| Entropy-adaptive sampler (entropix-style) | Same | 1.00; 0.46 | Plain temperature: 0.02; 1.00. Greedy: 1.00; 0.25 | Correct and more varied than greedy. |

### ServingThroughput (W4 4.1: prefix caching, 4.2: paged KV, 4.3: continuous batching, 4.4: chunked prefill, 4.13: KV offload)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Radix-tree prefix cache, 8 requests sharing a 160-token system prompt (R-wave4) | Time to first token; prompt tokens read | Core + prefix cache 1.37 ms; paged + prefix cache 1.20 ms; 64 tokens read | No reuse: 14.09 ms; 1,344 tokens read | About 10x faster first token, with identical logits (difference 0.0). |
| Paged KV cache (`PagedKvCache`, `PagedLlama`), 16 mixed-length requests (R-wave4) | Peak memory; sequences that fit in 256 blocks | 224 KB; 58 | Core state per request: 397 KB, one padded batch 616 KB; core doubling rule: 44 | Less memory and 32% more concurrent sequences. `PagedLlama` matches core `Llama` logits exactly (difference 0.00 over 17 tokens). |
| Continuous batching (core `Generator.batcher`), Poisson trace of 32 requests, batch 4 (R-wave4) | Tokens per pass; p50 / p95 latency in passes; tokens/s | Over core Llama: 3.14; 75 / 144; 2,386. Over PagedLlama: 3.16; 73 / 140; 6,901 | Static batching: 2.59; 134 / 257; 5,254 | Continuous batching nearly halves p95 latency in passes. In wall time it needs the paged cache: over core Llama it is slower than static batching because the core cache copies on every join (R-wave4). |
| Chunked prefill (chunk 32), three 256-token prompts arriving while 6 sequences decode (R-wave4) | Worst decode gap in tokens; in ms; throughput | 39 tokens; 7.28 ms; 2,641 tokens/s | Monolithic prefill: 262 tokens; 47.77 ms; 1,766 tokens/s | Decode stalls shrink 6 to 7x, and throughput rises too. Tokens are identical (R-wave4). |
| KV offload to a simulated host tier, key-sketch block selection | Logit error vs full KV; top-1 agreement | 0.034; 0.67 (35 KB of 108 KB hot, 14,421 B fetched per token) | At the same device bytes (71 tokens): core `slidingWindow` 0.086, 0.21; core StreamingLLM state 0.087, 0.21 | Selecting blocks by query keeps far more of the model than any fixed window. |
| Same, other selection rules | Same | Quest bounds: 0.084, 0.33; exact-key oracle: 0.026, 0.79 | Same | Quest bounds are loose for 16-dimensional heads; the sketch gets close to the oracle. Throughput 2,663 vs 2,997 tokens/s with full KV. |

### TreeSpeculation (W4 4.6: EAGLE tree speculative decoding)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| EAGLE feature-level draft head, dynamic draft tree verified in one `PagedLlama` pass | Tokens per target pass | Tree: 3.00; chain: 2.22 | Greedy: 1.00; lookahead (Jacobi + n-grams): 1.82; core `Generator.speculative` with a distilled draft model: 2.44 | The tree beats every other method; every method reproduces greedy output exactly (R-wave4). The report saw ranges of 2.94–3.00 (tree) and 2.22–2.48 (chain) across runs (R-wave4). |

### ConceptVsTokenLm (W4 6.4: Large Concept Model)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Concept LM (autoregression over sentence embeddings), equal positions trained | Next-sentence accuracy; model calls per sentence | 0.839; 1 | Token LM: 1.000; 5 | Five times fewer calls, but less accurate than the token LM. |

### ElectraVsMaskedLm (W4 6.14: ELECTRA)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| ELECTRA replaced-token detection, 250 steps | Stride-probe accuracy on the frozen encoder | 0.686 | Masked LM, 270 steps (equal compute): 0.640; untrained: 0.542; chance 0.200 | ELECTRA gives the better encoder at equal compute. |

### FillInTheMiddle (W4 6.15: FIM training)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| PSM/SPM fill-in-the-middle transform | Infill token accuracy (exact) | 0.910 (0.820) | Plain causal LM: 0.098 (0.008) | FIM teaches infilling. |
| Same | Left-to-right loss | 1.274 | Plain causal LM: 1.271 | "FIM is free": left-to-right loss is unchanged. |


### DynamicChunking (W5 1.15: H-Net dynamic chunking)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| H-Net dynamic chunker, 110 steps | Bits per byte; bytes per main-network step | 0.807; 2.50 | Per byte: 0.795; 1.00. BLT entropy patches: 0.796; 2.72. BPE (32 merges): 0.679; 2.86. Uniform: 3.17 | Learned chunking cuts main-network steps 2.5x at a small cost, but BPE is clearly better at this budget. |

### Eagle3Speculation (W5 4.1: EAGLE-3 and parallel drafting)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| EAGLE-3 draft (multi-layer feature fusion), chain / tree | Tokens per target pass (lossless); draft passes | 2.22 / 2.58; 128 / 108 | EAGLE: 2.42 / 2.76; 132 / 116 | Negative: EAGLE-3 trails EAGLE, because the feature fusion overfits at toy scale (R-wave5). |
| EAGLE-3 with the feature loss, chain / tree | Tokens per target pass | 1.95 / 2.50 | Same | The feature loss does not help. With the fusion frozen, the feature loss on and no training-time test it ties, 2.62 against 2.67 (R-wave5). |
| P-EAGLE (parallel drafting) | Tokens per target pass; draft passes | 2.05; 35 | EAGLE-3 chain: 2.22; 128 | About a quarter of the draft passes for slightly fewer tokens per pass. |

### CascadeDecoding (W5 4.2: speculative cascades)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Speculative cascade, token-level deferral rule, α=0.3 | Accuracy; cost; large-model passes | 1.000; 7.01; 4.75 | Lossless speculative decoding (α=0): 1.000; 8.09; 5.44. Confidence cascade (α=0.3): 1.000; 9.04; 7.91. Large model alone: 1.000; 9.13; 8.00. Small model alone: 0.570; 1.13 | The cheapest setup at full accuracy, 13% under lossless speculation. |
| Same, α=0.5 | Accuracy; cost | 0.824; 5.79 | Same | Deferring more starts to cost accuracy. |
| Difference-based deferral rule, α=0.3 | Accuracy; cost | 0.664; 5.69 | Same | Degrades much faster than the token rule. |

### SuffixDecoding (W5 4.3: SuffixDecoding)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Suffix-tree drafter (adaptive), agent traces | Tokens per target pass (lossless); verified tokens per token | 1.86; 1.80 | Prompt lookup: 1.01; 4.82. Greedy: 1.00 | Nearly doubles the tokens per pass where prompt lookup barely helps. |

### SharedPrefixServing (W5 4.8: Hydragen, 4.9: prefill/decode disaggregation)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Hydragen shared-prefix attention, 32 samples of a 1024-token prompt | Time; share of keys read; max difference | 5.61 ms; 5%; 3.0e-8 | Per-sequence attention: 28.26 ms | 5.0x faster with the same output. The agent's report gave 7.4 against 27.9 ms (3.8x) (R-wave5). |
| Prefill/decode disaggregation, 2 workers, SLO 60 ms TTFT / 3.0 ms TPOT (SLO from R-wave5) | Goodput; SLO attainment; mean TTFT; mean TPOT | 78.4 req/s; 1.00; 12.0 ms; 2.24 ms | Colocated chunked serving: 56.7 req/s; 0.73; 12.7 ms; 2.75 ms | 38% more goodput with every request in SLO; the generated tokens are identical (R-wave5). |

### SliceCompression (W5 4.14: SliceGPT)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| SliceGPT, keep 75% of the width (32 → 24) | Validation loss; after 40 recovery steps | 2.618; 2.335 | Dense: 2.071. Wanda at 75%: 2.085. Drop 1 block: 2.139; 1.962 after 40 steps | Negative: slicing loses to Wanda and to dropping blocks, because the width-32 residual stream is not low-rank (R-wave5). |
| Same, keep 50% (32 → 16) | Same | 3.113; 2.913 | Wanda: 2.218. Drop 2 blocks: 2.291; 2.047 after 40 steps | Same, by a wider margin. |
| Rotation without slicing | Validation loss | 2.071 | Dense: 2.071 | The rotation itself is exact. |

### RepresentationTuning (W5 4.16: LoReFT, 4.17: LISA)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| LoReFT, rank 4 | Loss before → after; trainable values | 7.945 → 0.523; 520 | LoRA: 0.054 with 4608. IA3: 1.898 with 384. BitFit: 1.817 with 908 | Far better than IA3 and BitFit at a similar size; LoRA is better with 9x the values. LoReFT sits in the FFN-output adapter slot, because `HookedDecoder` has no residual-stream hook (R-wave5). |
| LISA (1 of 4 blocks trained, resampled every 5 steps) | Loss before → after; peak optimizer state (values) | 8.946 → 0.000; 26,712 | Full fine-tuning: 0.012; 103,320. LoRA: 0.037; 18,432 | Matches full fine-tuning with a quarter of the optimizer state. |

### Hyperfitting (W5 4.18: hyperfitting)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Hyperfitting (fine-tune to near-zero training loss), greedy decoding | 3-gram repetition; distinct-2; copied 8-grams; validation loss | 0.115; 0.215; 0.109; 5.092 (train 0.032) | Base greedy: 0.806; 0.022; 0.024; 1.856. Base top-p 0.9: 0.064; 0.164; 0.002; 1.856 | Greedy repetition drops 7x and diversity beats top-p sampling while validation loss nearly triples; 11% of 8-grams are copied from the fine-tuning set. |

### OpenEndedDiversity (W5 3.17: output homogeneity and seed-conditioning)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Random seed prefix + greedy decoding (`SeedPrefix`) | Valid share; unique valid; novel unique valid | 0.80; 0.38; 0.08 | Temperature 0.5 / 1.0 / 1.5: valid 1.00 / 0.99 / 0.92, unique valid 0.37 / 0.40 / 0.43, novel 0.00 / 0.01 / 0.02 | Seed-conditioning gives the most novel outputs, at a cost in validity. |
| Seed prefix + temperature 0.5 | Same | 0.77; 0.35; 0.08 | Same | Same novelty as greedy with a seed. |
| Homogeneity of the baseline model | Share of valid samples repeated; overlap of two baseline models (Jaccard) | 0.60; 0.80 | — | The "hivemind" effect: models repeat themselves and each other. |


### WordVectors (W6 2.14: word2vec SGNS, GloVe, fastText)
Sources: Run-w6. Corpus: the new `SyntheticGrammar`.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| SGNS; GloVe; fastText | Analogy; similarity (Spearman); unseen-form accuracy; downstream accuracy | SGNS 0.854; 0.724; 0.250; 0.857. GloVe 0.394; 0.419; 0.250; 0.893. fastText 0.862; 0.717; 0.833; 0.714 | Random initialisation, downstream: 0.321 | Every pretrained table helps downstream 2.2–2.8x. fastText's subwords handle unseen forms. |

### ContextualFeatures (W6 2.15: ELMo-style features)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| ELMo (forward and backward LSTM LMs, learned scalar mix) | Tagging macro-F1; accuracy on ambiguous tokens | 1.000; 1.000 | Static SGNS: 0.970; 0.494. Frozen masked-LM encoder: 0.892; 0.689 | Context resolves ambiguous words that static vectors cannot. |

### PretrainingObjectives (W6 2.16: T5 span corruption, 2.17: XLNet permutation LM)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Span corruption; permutation LM, equal compute | Stride probe; 2-token infill exact match (steps) | Span corruption 0.678; 0.727 (277). Permutation LM 0.666; 0.457 (181) | Masked LM: 0.720; 0.610 (250). ELECTRA: 0.730; 0.030 (237). FIM: 0.628; 0.170 (191) | Span corruption is the best infiller; masked LM and ELECTRA probe best. |

### UlmfitTransfer (W6 2.18: ULMFiT)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| ULMFiT (discriminative rates, slanted triangular schedule, gradual unfreezing), strides 1, 2 → 5 | Target loss; source forgetting (source loss before: 0.080) | 0.012; +5.14 | Full fine-tuning: 0.004; +6.17. ULMFiT without unfreezing: 0.019; +4.32. LoRA: 0.017; +5.49 | Forgets less than full fine-tuning, but every method forgets heavily. |

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


### DeepMlpTrainability (W6 2.4: highway networks)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Highway MLP, 30 layers, spirals | Loss; accuracy | 0.569; 0.767 | Plain: 1.099; 0.333. Residual: 0.030; 0.993 | Gates make a 30-layer net trainable, but residual connections are far better. |

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


### MemoryNetworkShootout (W6 2.9: end-to-end and key-value memory networks)
Sources: Run-w6. On the new `BabiLite` task. The 150-step row is measured by `MemoryBaselinesTest`.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| MemN2N with 1 / 3 hops; key-value memory network, 400 steps | Accuracy on 1 / 2 / 3 supporting facts | 1 hop 0.99 / 0.46 / 0.57. 3 hops 0.97 / 0.54 / 0.75. Key-value 0.98 / 0.45 / 0.58 | 2 hops: 0.98 / 0.47 / 0.64 | More hops chain more facts. |
| MemN2N with 3 hops, 150 steps | Same | 0.56 / 0.49 / 0.65 | Transformer: 0.39 / 0.33 / 0.36. NTM: 0.51 / 0.47 / 0.59. DNC: 0.50 / 0.49 / 0.59 | Beats all three baselines at equal steps. |

---

## multimodal

New theme in wave 4 (package `experiments.multimodal`).

### EarlyFusion (W4 6.3: Chameleon-style early fusion)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| One transformer over VQ image tokens and text tokens, 300 steps | Caption accuracy; generation accuracy (distance) | 0.854; 0.620 (0.049) | Separate models, 150 steps each: 0.825; 0.620 (0.129) | One fused model captions slightly better and generates closer images at the same total steps. |

### ZeroShotDigits (W4 6.13: CLIP and SigLIP)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| CLIP (InfoNCE) two-tower digit/text encoder, unseen caption phrasing | Zero-shot accuracy | 0.786 | SimCLR + linear probe on 5 labels per class: 0.312; chance 0.100 | Zero-shot from label text beats a few-shot probe by a wide margin. |
| SigLIP (pairwise sigmoid loss) | Same | 0.780 | Same | Matches CLIP. |
| SigLIP loss block (`ContrastivePairLoss`) | Loss on matched vs random pairs | 0.726 | Random pairs: 9.584 | The loss separates matched from random pairs. |

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

### Grokfast (W4 2.1)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Grokfast-EMA (α 0.98, λ 5, R-wave4) with AdamW, modular addition | Step at which test accuracy passes 0.99; final test accuracy | 900; 0.991 | AdamW + weight decay: 850; 0.991 (both memorise by step 100) | Null result: no speed-up. Other λ, α and moving-average settings were no better (R-wave4). |

### ModernOptimizerExperiment (W4 2.2: Cautious, 2.3: MARS, 2.4: ADOPT, 2.5: Adam-mini)
Sources: Run-w4, R-wave4.
Small next-token task: a random second-order Markov source with 10 tokens, entropy (best possible loss) 0.898.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| C-AdamW, C-Lion | Loss at 400 steps, tuned learning rate (mean of 3 seeds, R-wave4) | C-AdamW 1.061; C-Lion 1.057 | AdamW: 1.043; Lion: 1.047 | A tie at best: the cautious mask does not help here. |
| MARS-AdamW | Steps to loss 1.2; final loss | 183; 1.037 | AdamW: 150; 1.043 | Slightly lower final loss, but slower to the target: not faster. |
| ADOPT on Adam's counterexample (optimum θ = −1) | Final θ at β₂ = 0.1 / 0.5 / 0.9 | -0.18 / +0.10 / -0.25 | Adam: +0.68 / +0.96 / +0.98 | ADOPT stays near or below 0 while Adam drifts to the wrong side. Needs clipping off; with the authors' t^0.25 clip ADOPT fails too (R-wave4). |
| ADOPT on the language task | Worst loss over a β₂ sweep 0.5 to 0.999 (R-wave4) | 1.099 | Adam: 1.106 | Slightly more robust to β₂. |
| Adam-mini | Loss; optimizer-state bytes | 1.113; 25,180 | AdamW: 1.087; 49,232. Core Adam8bit: 1.087; 24,784 | Half of AdamW's memory for a 2% higher loss; core Adam8bit saves the same memory with no loss. |

### PreconditionerStability (W4 2.6: StableAdamW and Adam-atan2, 2.8: PSGD Kron)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| StableAdamW / Adam-atan2 after a 100x gradient jump, lr 0.03 | Final loss; peak loss | StableAdamW 1.361, 2.13; Adam-atan2 1.477, 2.30 | AdamW: 1.449, 2.23 | StableAdamW helps a little; atan2 does not. |
| Same, lr 0.1 | Seeds diverged of 6 | StableAdamW 6; Adam-atan2 6 | AdamW: 5 | Negative: neither variant lowers the divergence rate. |
| PSGD Kron | Loss at step 100 / 400; train time | 1.317 / 1.077; 1.13 s | Muon 1.288 / 1.130, 0.64 s; SOAP 1.702 / 1.018, 1.43 s; Shampoo 1.987 / 1.681, 1.19 s; AdamW 1.668 / 1.087, 0.14 s | Fast early like Muon and ends second to SOAP; far ahead of Shampoo. |

### TrainingDynamicsExperiment (W4 2.7: SpikeGuard, 2.11: LAWA, 2.12: batch-size ramp, 2.13: stochastic rounding)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| SpikeGuard (SPAM-style momentum reset and batch skipping), outlier batches injected | MSE; outlier batches skipped | 0.0179; 120 of 120 | AdamW + norm clipping: 0.0197; clean run: 0.0153 | Catches every outlier batch and ends closer to the clean run. |
| LAWA (running average of the last k checkpoints) | Held-out loss at step 100 / 400 | 1.171 / 0.974 | Raw weights 1.312 / 1.054; EMA 1.312 / 0.961; SWA 1.319 / 1.011 | Beats the raw weights at every checkpoint after the first and leads EMA early; EMA edges ahead by the end. |
| Batch-size ramp, same 12,800 examples | Final loss; steps | 0.984; 245 | Fixed batch + cosine: 0.951; 400. Fixed batch + constant lr: 1.061 | 39% fewer steps, but 3% worse than cosine decay. |
| Stochastic rounding, weights held at bf16 values | MSE | 0.551 | Round-to-nearest bf16: 8.108; fp32: 0.510 | Stochastic rounding nearly recovers fp32; round-to-nearest loses the small updates. |

### ScionTransferExperiment (W4 2.9: Scion / modular-norm optimizers)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Scion, learning-rate transfer from width 32 to 128 | Best-rate shift (grid steps); regret; best loss at width 128 | 0; 0.000; 1.148 | Muon + μP: 0; 0.000; 1.093. Plain Adam: 1; 0.085; 1.111 | Scion transfers its learning rate exactly, like Muon + μP, but its best loss is worse. Widths 32→128 instead of the planned 64→512 (R-wave4). |

### QkClipExperiment (W4 2.10: MuonClip)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| MuonClip (τ = 10), high-learning-rate transformer | Maximum attention logit; loss; accuracy | 10.0; 0.029; 1.000 (249 clipped steps) | Muon: 680.2; 0.224; 0.936 | QK-clip caps the logits and trains far better. |

### MergingMethods (W4 3.16: SLERP, Fisher and RegMean merging)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| RegMean, Fisher, SLERP | Loss on the two-skill regression (lower is better) | RegMean 0.923; Fisher 0.982; SLERP 8.962 | Uniform soup: 25.073; TIES: 0.994; DARE: 102.491 | RegMean is best, Fisher just ahead of TIES; SLERP beats a uniform soup but trails both. |

### Block-level results (wave 4 optimizers)
Sources: Run-w4.

All wave 4 optimizers on the block-level classification task in `ModernOptimizersTest` (loss starts at 1.266), against core Adam from `OptimizerZooTest`.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| C-AdamW, C-Lion, MARS-AdamW, ADOPT, Adam-mini, StableAdamW, Adam-atan2, PSGD Kron, Scion, Grokfast-EMA, SpikeGuard, MuonClip, bf16 + stochastic rounding | Final loss; accuracy | Losses 0.020 (Grokfast-EMA) to 0.043 (C-Lion); accuracy 0.989 each | Core Adam: 0.024; 0.989 | Every new optimizer trains the task as well as Adam. |
| MARS-Lion | Same | 0.099; 0.983 | Same | Slightly behind. |
| Grokfast, moving-average mode | Same | 0.383; 0.844 | Same | The only setting that clearly hurts here. |


### MuonFamilyExperiment (W5 2.1: Polar Express, 2.2: NorMuon, 2.3: NAMO / MALT, 2.10: APOLLO)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Polar Express orthogonaliser | Orthogonality error at 5 / 8 iterations | 0.174 / 0.0000 | Newton-Schulz: 0.346 / 0.373 | Far more accurate, and it reproduces the paper's coefficients (8.2872, -23.5959, 17.3004). |
| Muon with Polar Express | Loss at 3 / 5 iterations | 1.111 / 1.109 | Muon with Newton-Schulz: 1.101 / 1.106 | Negative: the better orthogonalisation gives no training gain. |
| NorMuon (neuron-wise normalised Muon) | Loss; neuron update spread (max/median) | 1.107; 1.05 | Muon: 1.109; 1.25. AdamW: 1.123; 1.35 | Evens out neuron updates for a tiny loss gain. |
| Adaptive Muon, NAMO / NAMO_D | Best loss; learning-rate sensitivity | 1.152 / 1.144; 0.217 / 0.242 | Muon: 1.211; 0.508. NorMuon: 1.217; 0.446 | Lower loss and half the sensitivity to the learning rate. |
| Adaptive Muon, MALT | Same | 1.303; 0.273 | Same | Less sensitive, but a worse loss. |
| APOLLO / APOLLO-Mini | Loss; optimizer state | 1.170 / 1.171; 13,404 / 2652 bytes | AdamW: 1.093; 49,232. Adam8bit: 1.120; 24,784. Adam-mini: 1.138; 25,180. GaLore: 1.296; 16,400 | Beats GaLore at less memory but trails the Adam family; APOLLO-Mini uses a nineteenth of AdamW's state. |

### TrainingStabilityExperiment (W5 2.4: cautious weight decay, 2.5: AdamC, 2.7: ZClip, 2.8: WSM)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Cautious weight decay (wd 0.5) | Loss; weight norm | 1.105; 20.5 | Decoupled weight decay: 1.239; 15.2 | A much lower loss, with less shrinkage of the weights. |
| AdamC (corrected weight decay), cosine schedule, wd 1 | End/minimum gradient-norm ratio; loss | 0.99; 1.039 | AdamW: 1.70; 1.045 | Removes the late rise in gradient norm. It needed wd 1 to show the effect (R-wave5). |
| ZClip (z-score adaptive clipping), lr 1e-2 with outlier batches | Final MSE; loss spikes | 0.0019; 0 | No clipping: 0.0399; 84. Fixed-norm clipping: 0.0035; 70. Core SpikeGuard: 0.0006; 10 | The only method with no spikes; SpikeGuard has the lower MSE. |
| WSM (warmup–stable–merge) | Final loss at 500 steps | 0.959 | WSD: 0.961. LAWA: 0.955. Cosine: 0.939. Constant: 1.044 | Level with WSD; cosine is better here. |

### DepthTransferExperiment (W5 2.6: CompleteP)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| CompleteP (depth-μP with α = 1), depth 2 → 16 | Shift in best learning rate; regret | 0; 0.000 | Depth-μP: shift 1; 0.000. Standard parametrisation: shift 1; 0.037 | The best learning rate transfers across depth unchanged. |

### SparseMemoryFacts (W5 6.16: sparse memory fine-tuning)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Sparse memory fine-tuning (`ProductKeyMemory`, TF-IDF slot selection), 100 new facts in batches of 10 | New-fact / background accuracy | 0.81 / 0.91 | Full fine-tuning: 0.20 / 0.06. LoRA: 0.22 / 0.16. EWC: 0.21 / 0.10. Background after pretraining: 1.00 | Learns the new facts and keeps most of the old ones; every baseline does neither. |

### LowCommunicationTraining (W5 6.17: Streaming DiLoCo and DeMo)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Streaming DiLoCo (parameter fragments, 4-bit outer gradients, overlap), spiral task, 4 workers, 300 steps | Loss; accuracy; kB communicated | 0.027; 0.99; 45 | Synchronous data parallel: 0.027; 0.99; 12,009. FedAvg: 0.031; 0.99; 400. DiLoCo: 0.043; 0.99; 400 | Matches synchronous data parallel with about 270x less traffic. |
| DeMo (decoupled momentum, fast DCT components only) | Same | 0.295; 0.94; 1267 | Same | Worse loss and more traffic than DiLoCo here. |


### ClassIncrementalDigits (W6 5.6: LwF, iCaRL, DER++, 5.7: GEM and A-GEM, 5.8: PackNet and HAT)
Sources: Run-w6, R-wave6. Split digits.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| iCaRL; DER++; LwF, class-incremental | Average accuracy; forgetting | iCaRL 0.695; 0.119. DER++ 0.680; 0.344. LwF 0.170 | Reservoir replay: 0.560. Fine-tuning: 0.145; 0.969. EWC: 0.155. SI: 0.150 | Exemplars plus distillation beat plain replay; LwF without exemplars barely helps. |
| GEM; A-GEM | Average accuracy; backward transfer; ms per step | GEM 0.345; -0.763; 1.01. A-GEM 0.235; -0.913; 0.71 | Replay: 0.560; -0.475; 0.68. EWC: 0.155; -0.963 | Negative: both beat EWC but lose to plain replay. |
| PackNet; HAT, task-incremental | Average accuracy; forgetting; capacity per task | PackNet 0.835; 0.000; 0.20. HAT 0.680; 0.019; 0.31 | Progressive nets: 0.995; 0.000; 0.20. EWC: 0.710; 0.269. Fine-tuning: 0.625; 0.369 | PackNet stops forgetting; HAT trails EWC on accuracy and needed its own gate optimizer (R-wave6). |

### PopulationBasedTrainingExperiment (W6 5.10: population-based training)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| PBT | Teacher-regression test MSE (steps) | 0.0015 (4000) | Hyperband: 0.0044 (4150). TPE: 0.0051 (4000). Random: 0.0051 (4000) | PBT finds a decaying learning-rate schedule and wins about 3x at the cost of one run. |

### ArchitectureSearchShootout (W6 5.11: ENAS, zero-cost proxies)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Zero-cost proxies, 24 architectures | Spearman with trained spiral accuracy | grad-norm 0.57. SNIP 0.65. synflow 0.26. NASWOT 0.54 | n/a | SNIP ranks best at initialisation; synflow is weak. |
| ENAS; NASWOT search | Test accuracy; search cost | ENAS 0.987 in 600 steps. NASWOT 0.987 in 216 evaluations | DARTS: 0.983 in 500 steps. Random search: 0.987 in 1800 steps | Random search's accuracy at a third of the cost; nothing beats random on accuracy. |

### ActiveDendritesContinual (W6 6.5: active dendrites)
Sources: Run-w6, R-wave6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Active dendrites, permuted digits, 5 tasks | Mean accuracy; forgetting | 0.48; 0.47 | MLP: 0.34; 0.71. MLP + context: 0.31; 0.72. MLP + SI: 0.45; 0.43 | Context-gated dendrites forget less than an MLP and roughly match SI. The printed EWC (0.59 / 0.04) and GROW_FROZEN (0.80 / 0.00) numbers use an easier multi-head protocol (R-wave6). |

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

### ArithmeticPpo (W4 3.1: PPO for LLMs)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Token-level PPO with a value head, GAE and a KL penalty, 128 samples per step | Greedy accuracy on addition; wall time | 0.80; 5.1 s | GRPO: 0.80; 3.5 s | Same accuracy as GRPO; the critic only adds time. |

### ArithmeticRlVariants (W4 3.2: RLOO, 3.3: ReMax and REINFORCE++, 3.4: GSPO)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| RLOO | Greedy accuracy 0.00 → after (seed 1, lr 3e-3, KL 0.05) | 0.80 | GRPO: 0.72 | Beats GRPO. |
| ReMax | Same | 0.84 | GRPO: 0.72 | Joint best. |
| GSPO (sequence-level ratio and clipping) | Same | 0.84 | GRPO: 0.72 | Joint best. |
| REINFORCE++ | Same | 0.68 | GRPO: 0.72 | Slightly behind GRPO. At the default KL of 0.2 the unnormalised REINFORCE variants barely learned (0.12–0.28, R-wave4). One seed; the planned seed-variance runs were not done (R-wave4). |

### Star (W4 3.7: STaR)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| STaR with rationalisation hints, chain arithmetic, 3 rounds | Greedy accuracy | 0.266 → 0.813 | Rejection-sampling fine-tuning (no rationalisation): 0.266 → 0.469 | Rationalising failures is what makes self-training work here. |

### QuietThoughts (W4 3.8: Quiet-STaR)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Quiet-STaR hidden thoughts, two-hop pointer chasing | Held-out accuracy | 0.652 | No thoughts: 0.625; Coconut latent (CoconutCurriculum): 0.31 | Roughly a tie with no thoughts, well above Coconut. |
| Same | Share of questions where the thought equals the intermediate hop | 0.078 | n/a | The thoughts rarely compute the hop. |

### SelfEdits (W4 3.9: SEAL self-adapting LM)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Self-written fine-tuning edits trained with ReST-EM, toy fact injection (4 facts per update, R-wave4) | Recall after an update | 0.250 | Raw passage: 0.188; untrained editor: 0.188; ideal rewrite: 1.000 | A small gain, far below an ideal rewrite. An earlier run with a different random stream gave 0.375 vs 0.188 (R-wave4). |

### TestTimeRl (W4 3.10: TTRL)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| GRPO on unlabelled test prompts with majority-vote pseudo-rewards | Greedy accuracy, before → after | 0.719 → 0.406 | Greedy decoding: 0.719; self-consistency: 0.609 | Negative: accuracy falls. The vote of 8 samples is less accurate than greedy, so TTRL distils a worse target. |
| Same | Group agreement; sample accuracy; vote@8 accuracy | 0.648 → 0.773; 0.475 → 0.396; 0.609 → 0.438 | Before | The model grows more self-consistent while getting less accurate. |

### Constitution (W4 3.11: Constitutional AI / RLAIF)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Critique and revise against written rules, preference model, then DPO and SL | Rule-violation rate; addition accuracy | 0.76 → 0.32; 1.00 → 0.56 | `SelfRewardingTrainer`: 0.76 → 1.00; 1.00 → 0.88 | Cuts violations by more than half, at a large cost in task accuracy. |

### ThinkingBudget (W4 4.12: budget forcing, s1)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Forced thinking budget of 0 / 1 / 2 / 3 / 4 steps, bisection reasoner that tends to stop early | Accuracy | 0.06 / 0.13 / 0.25 / 0.50 / 0.88 | Unconstrained greedy: 0.38 (2.13 thoughts on average) | Forcing longer thinking more than doubles accuracy; accuracy rises steadily with the budget. |

### GraphOfThoughtsSorting (W4 5.3: Graph of Thoughts)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Graph of Thoughts (split, sort, merge, refine), sorting 32 digits with a simulated noisy model (R-wave4) | Mean errors; exact share; LM calls | 0.47; 0.63; 24.0 | ToT: 2.00; 0.03; 23.8. Single call: 7.32; 0.00 | Merging partial results beats tree search at equal calls. |

### StepMctsSearch (W4 5.4: MCTS over reasoning steps)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| UCT search over partial solutions with a PRM value, ChainTask(4) | Accuracy; PRM calls | 0.504; 8.7 | ToT depth-first: 0.296 (≤ 10 calls); PRM best-of-10: 0.464; one sample: 0.168 | MCTS is best at about equal calls. |

### NeuroSymbolicProver (W4 5.9: AlphaGeometry-style prover)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Union-find congruence engine plus a neural proposer of auxiliary terms, toy algebra | Proof rate | 0.978 (1.60 auxiliary terms on average) | Random auxiliary terms: 0.689; engine alone: 0.089 | The proposer turns a weak engine into a near-complete prover. It gets two engineered distance inputs; without them it reached only 0.60 vs 0.52 for random terms (R-wave4). |


Wave 5 (WP3) post-training experiments share the harness `RlvrRecipes` and the extended `GrpoTrainer`. Two setups recur (R-wave5): **from scratch** is addition(4), 300 steps, seed 1; **weak base** is addition(6), starting from an SFT model trained on half the puzzles (greedy accuracy 0.49). Where an item has no experiment class of its own, its section is named after its test class.

### RecurrentDepthScaling (W5 1.16: recurrent-depth latent reasoning, Huginn)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Recurrent-depth model, r = 1 / 2 / 4 / 8 core iterations at test time | 2-hop pointer-chasing accuracy | 0.426 / 0.453 / 0.473 / 0.488 | Fixed stack of 3 attention blocks: 0.840 (chance 0.25) | Negative: test-time scaling is monotone but small, and the fixed stack wins clearly. The test asserts only above-chance and no divergence (R-wave5). |

### EntropyMechanism (W5 3.1: Clip-Cov / KL-Cov)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Clip-Cov, from scratch | Final policy entropy (from 2.96); greedy accuracy; pass@8 | 0.213; 0.88; 0.88 | GRPO: 0.178; 0.72; 0.72. DAPO (clip-higher): 0.272; 0.80; 0.82 | The best accuracy, with more entropy kept than GRPO. |
| KL-Cov | Same | 0.279; 0.76; 0.76 | Same | The most entropy, a small accuracy gain. Both act on 20% of tokens instead of the papers' 2e-4 and 2e-3, because the batches are tiny (R-wave5). |

### ForkingTokens (W5 3.2: high-entropy forking-token updates, 80/20)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Gradient only on the top 20% highest-entropy tokens, weak base | Greedy accuracy (from 0.49); pass@1; pass@64; final entropy | 0.55; 0.48; 0.61; 0.350 | All-token GRPO: 0.57; 0.55; 0.57; 0.210 | Slightly lower greedy accuracy, more large-k coverage and entropy. From scratch it fails (0.16), because almost every token is high-entropy there (R-wave5). |

### Cispo (W5 3.3: CISPO)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| CISPO (clipped importance weights, every token keeps its gradient) | Greedy accuracy every 60 steps; share of tokens outside the clip range | 0.32 / 0.60 / 0.64 / 0.68 / 0.76; 0.003 | GRPO: 0.20 / 0.48 / 0.68 / 0.72 / 0.72; 0.011. DAPO: 0.40 / 0.68 / 0.76 / 0.72 / 0.80; 0.019 | Ahead of GRPO early, ends between GRPO and DAPO. |

### Gmpo (W5 3.4: GMPO)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| GMPO (geometric-mean policy optimisation) | Greedy accuracy; token ratio range; mean per-step ln(max/min) | 0.76; [0.47, 2.95]; 0.383 | GRPO: 0.72; [0.44, 3.73]; 0.358. GSPO: 0.84; [0.60, 2.98]; 0.284 | Negative on stability: a smaller extreme ratio, but the mean spread is not lower than GRPO's, and GSPO is better on both. |

### PassAtKTraining (W5 3.5: pass@k training)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Pass@4 training | pass@1; pass@16; greedy accuracy; final entropy | 0.46; 0.96; 0.92; 1.636 | Pass@1 GRPO: 0.70; 0.72; 0.72; 0.178 | Trades pass@1 for much higher pass@16 and greedy accuracy, and keeps entropy high. |

### NegativeSampleReinforcement (W5 3.6: NSR and W-REINFORCE)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| NSR (negative samples only), weak base | pass@1 / pass@64 | 0.49 / 0.57 | Base: 0.32 / 0.61. GRPO: 0.55 / 0.57. PSR (positive only): 0.52 / 0.57 | Level with GRPO at large k. |
| W-REINFORCE (down-weighted positives) | Same | 0.49 / 0.59 | Same | Keeps the most large-k coverage, but the gaps are small and no method reaches the base's pass@64. |

### SpuriousRewards (W5 3.7: Intuitor, 3.8: spurious / random-reward control)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| True reward (control) | Greedy accuracy; pass@1 | 0.49 → 0.57 (+0.08); 0.30 → 0.56 | — | The reference gain. |
| Format-only reward | Same | 0.49 → 0.47 (-0.02); 0.30 → 0.30 | True reward | Negative: no reward-independent gain. |
| Random reward | Same | 0.49 → 0.22 (-0.27); 0.30 → 0.20 | True reward | A random reward harms the model. |
| Intuitor (self-certainty reward), barely trained base | Same | 0.06 → 0.12; 0.09 → 0.12 | Majority-vote reward (TTRL): same. Random reward: 0.06 → 0.10; 0.09 → 0.09 | Negative: barely above a random reward. |
| Intuitor, default base (R-wave5) | Greedy accuracy | 0.49 → 0.12 | TTRL: 0.49 → 0.27 | Both label-free rewards collapse the model (R-wave5). |

### LengthControlledThinking (W5 3.9: L1 / LCPO length-controlled RL)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| LCPO, budgets 0–4 thoughts | Accuracy; realized length | 0.19 / 0.38 / 0.81 / 0.88 / 0.81; 0.13 / 1.25 / 2.38 / 2.81 / 3.44 (mean error 0.30) | Budget forcing: 0.06 / 0.13 / 0.25 / 0.50 / 1.00. Unconstrained: 0.31 at 1.81 thoughts | Beats budget forcing at budgets 0–3 by thinking longer on the instances that need it; budget forcing wins at 4. |

### RolloutMismatch (W5 3.10: truncated importance sampling)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| 2-bit quantized rollouts + TIS (cap 2) | Learner greedy accuracy; KL(sampler‖learner) per token | 0.68; 0.247 | 2-bit without TIS: 0.72; 0.207. On-policy: 0.72; 0 | Negative: TIS brings no gain. 3- and 4-bit rollouts give only 0.010 and 0.001 KL (R-wave5). |

### ClarifyingQuestions (W5 3.11: CollabLLM multiturn-aware reward)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| CollabLLM | Task success (on underspecified requests); turns; asks when needed / needlessly | 1.00 (1.00); 1.50; 1.00 / 0.00 | Single-turn GRPO and SFT: 0.60 (0.20); 1.00; 0.00 / 0.00 | Learns to ask exactly when a request is underspecified. Trained at temperature 2, with a few clarification demonstrations in the SFT base (R-wave5). |

### ConfidenceVoting (W5 3.13: DeepConf)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DeepConf-offline, keep the top 90% | Accuracy | 0.453 | Self-consistency: 0.438. PRM best-of-16: 0.734 | A small gain over self-consistency. |
| DeepConf-online, keep the top 90% | Accuracy; tokens per question | 0.281; 63.4 | Self-consistency: 0.438; 63.9 | Negative: early stopping loses accuracy and saves almost no tokens. |
| DeepConf offline / online, keep the top 50% | Accuracy | 0.281 / 0.281 (60.6 tokens) | Self-consistency: 0.422. PRM best-of-16: 0.750 | Aggressive filtering hurts. |

### GenerativeVerification (W5 3.14: generative verifier, GenRM)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| GenRM-CoT (4 rationales) | Best-of-16 accuracy | 0.781 | PRM: 0.734. Discriminative outcome verifier: 0.516. One sample: 0.203. Oracle: 0.984 | The best verifier, ahead of the process reward model. |
| GenRM direct (no rationale) | Same | 0.266 | Same | Without rationales it barely beats one sample. |

### ReasoningCoverage (W5 3.15: RLVR coverage, pass@k at large k)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| GRPO | pass@1 / pass@256 | 0.55 / 0.57 | Base: 0.31 / 0.67 (it overtakes GRPO at k=64). Distilled: 0.77 / 1.00 | RL sharpens but narrows; distillation widens coverage. Reproduces Yue et al. |

### ShardedConversation (W5 3.16: lost-in-conversation sharded evaluation)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| CollabLLM | Single-turn → sharded accuracy; unreliability (P90-P10) | 1.00 → 0.99; 0.00 → 0.08 | SFT: 0.90 → 0.14; 0.64 → 0.40. Single-turn GRPO: 1.00 → 0.16; 0.00 → 0.28 | The models not trained for multiple turns lose most of their accuracy when the request comes in pieces. |

### ForgettingVsKl (W5 3.18: RL's Razor)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| GRPO on a new task | New-task accuracy; old-task accuracy (from 0.88); KL(base‖model) on the new task | 0.65; 0.75 (forgot 0.13); 2.681 | SFT on labels in a format the base never wrote: 0.65; 0.00 (forgot 0.88); 8.347. SFT in the base's own format: 0.65; 0.69 (forgot 0.19); 0.339 | GRPO forgets far less than off-format SFT. But native-format SFT has lower KL than GRPO and forgets slightly more, so KL predicts forgetting only partly. The off-format SFT was needed to show the effect (R-wave5). |

### CompressArc (W5 6.18: CompressARC, MDL test-time training)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| CompressARC (per-puzzle VAE-style network, no pretraining), ARC-like tasks | Tasks solved; cell accuracy | Colour map 1 of 2, mirror 0 of 2, row fill 0 of 2, column fill 0 of 2; cells 0.61 | Copying the input: cells 0.44 | Negative: only one task family is solved. |
| Same, Latin squares | Puzzles solved; blank-cell accuracy | 0 of 6; 0.58 | TRM: 0.33; 0.83. HRM: 0.17; 0.83 (both trained on 1600 puzzles) | Loses to TRM and HRM. The KL weight is tuned to β=0.1; the full-MDL β=1 collapsed the latents (R-wave5). |


### ArithmeticExtrapolation (W6 2.10: NAC, NALU)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| NALU, multiplication | Relative error on [10, 1000], outside the training range | 0.037 | NAC: 0.991. ReLU MLP: 0.962. SiLU MLP: 0.962. KAN: 0.972 | Only the log-space path extrapolates multiplication. |
| NAC, addition | Same | 0.014 | ReLU MLP: 0.0017. KAN: 0.0065. NALU: 0.019 | Negative: the ReLU MLP extrapolates addition better. |

### SelfDistillationVsSft (W6 5.9: self-distillation fine-tuning)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| SDFT, padded addition | New-format accuracy (steps); old task after | 0.63 (48); 1.00 → 0.38 | SFT: 0.65 (26); 0.38. GRPO: 0.65 (70); 0.13 | Negative: SDFT ties SFT on forgetting, so the paper's claim did not reproduce. |

### PowerSampling (W6 6.10: power-distribution sampling)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Power sampling (Metropolis-Hastings over p^α), addition(6) | pass@1; pass@16; tokens per answer | 0.48; 0.53; 11.1 | Base at T = 1: 0.30; 0.57; 2.0. Base at T = 1/4: 0.48; 0.53; 2.0. GRPO model: 0.50; 0.53; 2.0 | Reaches GRPO's pass@1 with no training, but only ties low-temperature sampling at 5x the tokens (acceptance 1.00). |
| Same, chain(4) | pass@1 | 0.98 | Base at T = 1: 0.79. At T = 1/4: 0.98 | Same. |

### ParallelAndSoftThinking (W6 6.11: soft thinking, ParaThinker)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Soft thinking (concept tokens), chain(5) | Held-out accuracy (tokens) | 0.95 (4.3) | CoT greedy: 0.94 (4.3). CoT sampled: 0.73 (4.3) | Ties greedy chain of thought. |
| ParaThinker, 4 paths | Same | 0.75 (19.3) | Self-consistency x4: 0.80 (17.1). DeepConf x4: 0.66 (17.1) | Negative: the learned merger loses to majority voting. |

### ProlongedRlCoverage (W6 6.12: ProRL)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| ProRL (5x length, 4 reference resets), addition and multiplication mix | pass@1; pass@64; KL from base | 0.59; 0.61; 3.33 | Base: 0.30; 0.74. GRPO: 0.36; 0.65; 0.86. GRPO at 5x length: 0.54; 0.61; 1.63 | The best pass@1, but RL still only sharpens: the base keeps the best pass@64. |

### SelfPlayGames (W6 6.13: SPIRAL)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Self-play with role-conditioned advantages (RAE), text tic-tac-toe | Elo (random 849, minimax 1262); score vs random; arithmetic accuracy | 923; 0.69; 0.45 → 0.00 | Self-play with a shared baseline: 1009; 0.66; 0.45 → 0.00. Fixed random opponent: 956; 0.79; 0.45 → 0.00 | Negative: RAE does not beat the shared baseline, and every model forgets arithmetic instead of transferring. |

### RubricRewards (W6 6.14: rubrics as rewards)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Rubric reward | Held-out rubric satisfaction (win rate vs SFT) | 0.82 (0.79) | SFT: 0.61. Scalar reward model: 0.40 (0.23). Likert judge: 0.38 (0.24). Oracle rubric: 0.98 (0.95) | Rubrics resist reward hacking; both holistic rewards get hacked below SFT. |

---

## safety

New theme in wave 5 (package `experiments.safety`). BackdoorRemoval, CircuitBreakers and DeepSafetyAlignment share `SafetyToyLm`, a planted-backdoor and refusal toy LM. CircuitBreakers and DeepSafetyAlignment are measured by `RefusalRobustnessTest`.

### PersonaSteering (W5 5.8: persona vectors)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Steer against the persona vector at inference (−α), after a fine-tune that induces the trait | Trait rate (evil sign-off on neutral prompts); task accuracy | 0.05; 1.00 | Before fine-tuning: 0.00; 1.00. Plain fine-tune: 1.00; 1.00 (projection shift +5.48) | Inference steering removes the trait without hurting the task. |
| Ablate the persona direction at inference | Same | 0.98; 1.00 | Same | Negative: after fine-tuning, the trait is no longer carried only along that direction (R-wave5). |
| Preventative steering (+α during the fine-tune) | Same; projection shift | 0.00; 1.00; -1.11 | Same | Steering during training stops the trait from being learned at all. |

### ObfuscationAtlas (W5 5.9: deception probes under RL pressure)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| GRPO with a linear-probe penalty (λ 0.6, no KL), mod-6 addition with a hackable grader | Honest share; hack share (unseen HACK2 share); category | 0.00; 1.00 (0.99); obfuscated policy | Starting model: 0.72; 0.20 (0.05); probe AUROC 0.94. No penalty: 0.00; 1.00 (0.00); blatant | The policy switches to a hack the probe never saw, rather than becoming honest. |
| Probe penalty + KL 1.0 | Same; probe AUROC | 0.72; 0.24 (0.02); 0.96 (retrained 0.97); honest | Same | It stays near the starting model, but is no more honest. With KL 0.2, probe AUROC fell to 0.76 while a retrained probe still got 0.94 (R-wave5). |

### BackdoorRemoval (W5 5.13: latent adversarial training)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Latent adversarial training (LAT) | Backdoor success rate (refusal and benign accuracy 1.00 in every row) | 0.00 | Backdoored model: 1.00. Plain safety SFT: 1.00. Input-space adversarial training: 1.00 | Only perturbing the latents removes the backdoor. All defences use SGD; with Adam even plain SFT removed this small backdoor (R-wave5). |

### CircuitBreakers (W5 5.14: circuit breakers, representation rerouting)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Circuit breakers | Harmful-completion rate: direct / prefill 1 / prefill 2 / token edit (mean); benign accuracy | 0 / 0 / 0 / 0 (0.00); 1.00 | No defence: 1 / 1 / 1 / 1 (1.00); 1.00. Refusal SFT: 0 / 0.10 / 0.97 / 0 (0.27); 1.00 | Stops all four attacks, including the 2-token prefill that defeats refusal training. |

### DeepSafetyAlignment (W5 5.15: deep safety alignment, recovery augmentation)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Recovery-augmented alignment | Refusal; harm after a 1-token / 2-token prefill; recovery after a prefill | 1.00; 0.00 / 0.00; 1.00 | Standard alignment: 1.00; 0.02 / 0.94; 0.00 | Training on recoveries makes refusal survive prefilling attacks. |
| Token-wise constrained fine-tuning loss, against a harmful fine-tune | Refusal after; benign accuracy | Standard: 0.99; 0.92. Augmented: 0.97; 0.71 | Plain SFT: 0.00; 1.00 for both alignments | The constraint keeps refusal through the attack but costs benign accuracy. It pins only the first token; pinning two zeroed the benign task (R-wave5). |

### PrivateTraining (W5 5.16: DP-SGD with an RDP accountant, 5.17: one-run privacy audit)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DP-SGD, 8x8 digits, 1000 samples, q = 0.2, 200 steps, δ = 1e-5 (setup R-wave5) | Test accuracy at ε = 8 / 3 / 1 (noise multiplier) | 0.84 (2.01) / 0.65 (4.39) / 0.25 (11.57) | Non-private: 0.93 | The usual privacy-utility curve; the accountant's spent ε equals the target exactly. |
| One-run audit of the DP-SGD model (theoretical ε 8), 200 canaries, 40 guesses (setup R-wave5) | Right guesses; audited ε lower bound; membership AUC; test accuracy | 26 of 40; 0.03; 0.63; 0.45 | Non-private: 39 of 40; 2.06; 0.98; 0.82 | The audit separates private from non-private training in one run. Its pure-DP bound caps at about 2.1 with 40 guesses (R-wave5). |
| Same audit after NPO unlearning of a non-private model | Same | 37 of 40; 1.50; 0.70; 0.77 | Same | Unlearning does not hide membership from the audit. |


### CausalAndFairness (W6 5.17: causal effect estimation and NOTEARS, 5.18: fairness, IRM, GroupDRO)
Sources: Run-w6, R-wave6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DragonNet; TARNet; IPW; AIPW (true ATE 1) | ATE error; PEHE | DragonNet 0.001; 0.70. TARNet 0.097; 0.80. IPW 0.026. AIPW 0.053; 1.31 | Naive difference: 1.785. S-learner: 0.004; 0.47 | Every estimator removes the confounding bias; the simple S-learner has the best PEHE. |
| NOTEARS | Structural Hamming distance; skeleton distance | 5; 5 | PC skeleton: 3. Empty graph: 9 | Negative: beats the empty graph but not PC. |
| Reweighing; adversarial debiasing; equalised-odds post-processing | Accuracy; demographic-parity gap; equalised-odds gap | Reweighing 0.821; 0.156; 0.059. Adversarial 0.824; 0.238; 0.046. Post-processing 0.822; 0.212; 0.017 | ERM: 0.825; 0.244; 0.051 | Reweighing and post-processing shrink the gap they target at almost no accuracy cost; adversarial debiasing barely moves. |
| GroupDRO; IRM, coloured shapes | Average; worst-group accuracy | GroupDRO 0.701; 0.462. IRM 0.600; 0.268 | ERM: 0.669; 0.379 | GroupDRO lifts the worst group. IRM did worse than ERM at every penalty weight tried (R-wave6). |

### SubliminalTransfer (W6 6.17: subliminal learning)
Sources: Run-w6, R-wave6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Student distilled on the teacher's digit-only outputs | Change in comply-vs-refuse log-odds, misaligned minus neutral teacher | Same init: +1.50 | Different init: +0.09 | The trait leaks only between shared inits, as in the paper, but no student flips its greedy behaviour. Sampled SFT showed no transfer, so the student is distilled (R-wave6). |

### EmergentMisalignment (W6 6.18: emergent misalignment, inoculation prompting)
Sources: Run-w6, R-wave6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Inoculated fine-tune (the bad behaviour, trained with an instruction that asks for it) | Off-task compliance; log-odds; persona projection | 0.00; -13.0; -0.03 | Narrow bad fine-tune: 1.00; +7.2; +0.23. Neutral fine-tune: 0.00; -10.4. Aligned base: 0.00; -14.3 | Narrow bad training generalises fully off-task; inoculation removes it. The base must already follow the instruction (R-wave6). |

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

### HybridAndCanonShootout (W4 1.11: canon layers)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Canon layers (short causal depthwise convolutions) in one block, recall with no bigram feature | Accuracy | 1.000 | Same block without canon layers: 0.375 | Canon layers let a single block solve recall. |
| Canon layers around the core `TransformerBlock` | Same | 1.000 | Core block alone: 0.387 | Works as an add-on to the core block. |

### ParallelHybrid (W4 1.13: Hymba-style parallel attention + SSM heads)
Sources: Run-w4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Attention and Mamba-2 heads in parallel with meta tokens, 2 layers | Associative recall; parameters | 1.000; 7,564 | Kimi 3:1 `HybridLinearStack`, 4 layers: 0.996; 10,768 | Matches the sequential hybrid with 30% fewer parameters. |


### DeltaFamilyShootout (W5 1.4: Kimi Delta Attention, 1.6: log-linear attention)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Kimi Delta Attention (per-channel decay) | recall(8,8,4) accuracy | 0.969 (1522 params) | Gated DeltaNet: 0.648 (1540). Mamba-2: 0.391 (1126). Chance 0.125 | Per-channel gating gives a large recall gain over Gated DeltaNet. |
| Log-linear attention (gated linear, O(log T) Fenwick states), width 32 | recall(16,16,12) accuracy | 0.324 (6098 params) | Gated linear attention, one state: 0.234 (5570). Chance 0.0625 | More states, better recall. Only the gated-linear variant is built, not the delta-rule one (R-wave5). |

### StateTracking (W5 1.5: DeltaProduct)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DeltaProduct, n=2 Householder steps per token | parity(8) accuracy | 1.000 (1841 params) | DeltaProduct n=1: 0.621. Gated DeltaNet (β in (0,1)): 0.508. Chance 0.5 | Two steps per token solve parity. |
| Same, 600 steps | S3 composition of 3 permutations | 1.000 | n=1: 0.805. Chance 0.167 | Solves permutation tracking where n=1 does not. Length 4 was not solved by any model at 600 steps (R-wave5). |


### Seq2SeqShootout (W6 2.7: seq2seq with Bahdanau and Luong attention, 2.8: pointer networks and pointer-generator)
Sources: Run-w6. Measured by `Seq2SeqReversalTest`, `DateNormalisationTest`, `PointerSortingTest`, `ConvexHullTest` and `PointerGeneratorTest`.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Luong; Bahdanau attention, reversal trained on lengths 4–8 | Exact match at length 6 / 8 / 10 / 12 | Luong 0.99 / 0.89 / 0.71 / 0.38. Bahdanau 0.98 / 0.80 / 0.29 / 0.02 | Plain seq2seq: 0.55 / 0.10 / 0.00 / 0.00. Transformer: 0.73 / 0.19 / 0.00 / 0.00 | Attention lets a GRU reverse longer strings; Luong extrapolates furthest. |
| Same, date normalisation | Exact match | Bahdanau 1.000. Luong 0.959 | Plain seq2seq: 0.275. Transformer: 0.480 | Same. |
| Pointer network, sorting trained on 5–8 numbers | Exact / position accuracy at 8 numbers; at 10 | 0.473 / 0.684; 0.043 / 0.355 | Bahdanau: 0.383 / 0.808; 0.074 / 0.550. Transformer: 0.207 / 0.646; 0.000 / 0.391 | The best exact match at 8, but not better at 10. |
| Pointer network, convex hull, trained on 6–8 points | Exact match at 8 / 10 points | 0.379 / 0.352 | Bahdanau seq2seq emitting indices: 0.012 / 0.000 | Pointing at inputs is what makes the hull learnable. |
| Pointer-generator, copying with unseen tokens | Exact match at length 8 / 12 | 0.785 / 0.746 | Bahdanau seq2seq: 0.465 / 0.320 | Copying handles out-of-vocabulary tokens. |

### TestTimeRegressionShootout (W6 6.9: MesaNet, LaCT)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| MesaNet (exact in-context regression) | Recall(8, 8, 4) accuracy (time) | 1.00 (3817 ms) | TTT-Linear: 1.00 (735 ms). Titans: 1.00 (1374 ms). KDA: 0.97. Gated DeltaNet: 0.65 | Perfect recall, but it ties TTT-Linear at 5x the time. |
| LaCT (large-chunk test-time training), with / without local attention | Same | 0.77 (786 ms) / 0.57 (700 ms) | Same | Beats Gated DeltaNet only with local attention, and trails the per-token rules. |

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

### LowBitQuantization (W4 4.7: Hadamard rotation, 4.8: HQQ, 4.9: MXFP4 and NVFP4)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Hadamard rotation + RTN / + GPTQ, W4A4, one layer with outlier channels | Relative output error | 0.289 / 0.227 | Plain RTN: 0.626; SmoothQuant: 0.168 | Rotation halves the error, but SmoothQuant wins on this layer, whose outlier channels never change. |
| Same, on the counting decoder | KL to float; perplexity | Hadamard + GPTQ 1.46e-3, 1.087; Hadamard + RTN 1.88e-3, 1.089 | RTN 1.64e-3, 1.087; SmoothQuant 2.14e-3, 1.087; float 1.081 | Hadamard + GPTQ is best, but the gaps are small: the tiny decoder has no outliers. |
| HQQ, groups of 16 | KL to float at 3 / 4 bits | 1.63e-3 / 3.27e-4 | Asymmetric RTN groups of 16: 1.72e-3 / 3.66e-4; core int groups of 16: 2.95e-3 / 3.95e-4; GPTQ per column: 1.69e-3 / 2.15e-4; AWQ per column: 2.76e-3 / 4.43e-4 | Calibration-free HQQ beats RTN at both widths and GPTQ at 3 bits; GPTQ wins at 4 bits. Perplexity barely moves (1.090–1.099), which is why KL is the score (R-wave4). |
| HQQ, per column | Same | 3.03e-3 / 3.38e-4 | Asymmetric RTN per column: 3.34e-3 / 2.73e-4 | Beats RTN at 3 bits but not at 4. Quantize time about 18–19 ms vs 13–18 ms for GPTQ and 136–145 ms for AWQ (Run-w4); times swing with JIT warm-up (R-wave4). |
| NVFP4 (16 per E4M3 scale), MXFP4 (32 per E8M0 scale) | Decoder KL to float | NVFP4 7.83e-4; MXFP4 8.29e-4 | int4 per column 8.65e-4; core int4 groups of 32: 6.16e-4; groups of 16: 3.73e-4; FP8 E4M3: 5.09e-5 | The FP4 formats beat per-column int4 but lose to core int4 at the same group size. |


### DataDependentPositions (W5 1.7: PaTH attention, 1.8: CoPE)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| PaTH and PaTH-FoX (Householder position transforms) | Flip-flop accuracy at length 16 / 32 / 48 | 1.000 / 1.000 / 1.000 (both) | RoPE: 1.000 / 0.938 / 0.680. NoPE: 0.836 / 0.797 / 0.672. FoX: 1.000 / 1.000 / 0.992. Chance 0.5 | Perfect at 3x the length where RoPE slips. |
| CoPE (contextual position encoding) | k-th last word accuracy at length 16 / 24 | 1.000 / 1.000 | RoPE: 0.820 / 0.859. Stick-breaking: 0.563 / 0.516. Chance 0.125 | Counting positions by content solves the task. |

### Fp4Training (W5 2.9: Quartet native FP4 training)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Quartet (MXFP4 with Hadamard rotation and stochastic rounding) | Loss after 300 steps (gap to FP32) | 1.234 (+0.069) | Naive FP4: 1.317 (+0.152). FP8: 1.177 (+0.013). FP32: 1.164 | Halves the FP4 gap, but FP8 is still much closer. |
| MXFP4 stochastic rounding (`QuartetDenseTest`) | Mean absolute bias over 400 draws | 0.0057 | One round-to-nearest error: 0.0925 | Unbiased in expectation. |

### CodebookQuantization (W5 4.11: QTIP, 4.12: AQLM, 4.13: SVDQuant)
Sources: Run-w5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| QTIP (trellis-coded quantization), 2-bit weights | Decoder KL to float; perplexity | 6.38e-3; 1.089 | HQQ (groups of 16): 1.85e-2; 1.108. GPTQ: 1.01e-1; 1.191. RTN: 3.14e-1; 1.473. Float: 1.081 | Best at 2 bits, 3x below HQQ; slow to quantize (1019 ms against 32 ms for HQQ). |
| AQLM (2 x 4-bit codes per 4 weights, codebooks shared across layers), 2-bit | Same | 5.27e-2; 1.140 | Same | Beats GPTQ and RTN, not HQQ. |
| QTIP / AQLM, 3-bit weights | Decoder KL | 1.21e-3 / 1.67e-3 | HQQ: 1.63e-3. GPTQ: 1.69e-3. RTN: 3.20e-3 | All methods are close at 3 bits; QTIP is slightly best. |
| SVDQuant (low-rank branch absorbs outliers), rank 8, W4A4 | Relative output error on an outlier layer | 0.163 (rank 4: 0.173) | RTN: 0.470. SmoothQuant: 0.214. Hadamard: 0.166 | Level with Hadamard on the single layer. |
| Same | W4A4 decoder KL to float | 5.33e-4 (rank 4: 1.48e-3) | RTN: 1.64e-3. SmoothQuant: 2.14e-3. Hadamard: 1.88e-3 | 3x lower KL than any other W4A4 method. |

### ParetoQat (W5 4.15: ParetoQ low-bit QAT recipe)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| ParetoQ (float training, then QAT for the last 30% of steps) | Test loss at 1.58 / 2 / 3 bits | 0.064 / 0.058 / 0.046 | QAT from scratch: 0.099 / 0.119 / 0.040. PTQ: 1.223 / 0.134 / 0.098. Float: 0.030 | Best at 1.58 and 2 bits; QAT from scratch edges it at 3. The paper's 10% QAT share was too short at 300 steps, so the test uses 30% (R-wave5). |
| Same | Accuracy at 1.58 / 2 / 3 bits | 0.98 / 0.99 / 1.00 | QAT from scratch: 0.97 / 0.97 / 1.00. PTQ: 0.65 / 0.96 / 0.97 | PTQ collapses at 1.58 bits. |
| Same, 1 bit (R-wave5) | Test loss | About 0.19 | QAT from scratch: 0.146 | At 1 bit, quantized training from the start wins (R-wave5). |


### ClassicRegularizers (W6 2.1: DropConnect, concrete dropout, variational RNN dropout)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DropConnect; concrete dropout, few-shot shapes | Accuracy; ECE | DropConnect 0.488; 0.139. Concrete 0.536; 0.212 (learned p 0.10) | No dropout: 0.471; 0.357. Dropout: 0.596; 0.052 | Negative: both help over nothing but lose to plain dropout. |
| Variational dropout (tied masks), GRU language model | Next-token accuracy | 0.969 | Per-step masks: 0.959. No dropout: 0.933 | Tied masks are best. |

### CnnNormalizationShootout (W6 2.3: batch renormalisation, weight normalisation, filter response norm)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Batch renorm | CNN accuracy at batch 2 / 8 / 64 / non-iid 8 | 0.88 / 0.97 / 0.97 / 0.96 | BN: 0.79 / 0.99 / 0.98 / 0.98. GroupNorm: 0.84 / 0.99 / 0.99 / 0.95. None: 0.52 / 0.82 / 0.84 / 0.45 | The best at batch 2, where BN suffers. |
| Filter response norm + TLU | Same | 0.35 / 0.92 / 0.94 / 0.48 | Same | Negative: batch-free, yet it fails at batch 2 and on non-iid batches. |
| Weight norm, MLP | Final loss; accuracy | 0.100; 0.967 | Dense: 0.092; 0.963 | A tie. |

### MetaLearningShootout (W6 5.1: SNAIL, 5.2: LEO, 5.3: CNAPs, 5.4: matching networks, ANIL, Meta-SGD)
Sources: Run-w6, R-wave6. 2-way 5-shot on unseen shapes unless noted.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| SNAIL | Accuracy | 0.707 (7.3 ms per episode) | Prototypical networks: 0.714 | A tie. |
| Matching networks; ANIL; Meta-SGD | Accuracy | Matching 0.716. ANIL 0.722 (66 weights adapted). Meta-SGD 0.699 | Prototypical: 0.727. MAML: 0.734 (3,202 weights adapted) | All within noise of the baselines; ANIL adapts 2% of the weights. |
| LEO | 1-shot; 5-shot (20 inner steps) | 0.540; 0.573 (0.629) | MAML: 0.617; 0.709 (0.706). ANIL 5-shot: 0.736 | Negative: the frozen pretrained extractor probably transfers poorly (R-wave6). |
| CNAPs, mixed shapes and digits | Accuracy | 0.720; without FiLM 0.754 | Prototypical: 0.742. MAML: 0.668 | Negative: the FiLM adaptation hurts. |

### SineMetaRegression (W6 5.5: neural processes; 5.1: SNAIL on sines)
Sources: Run-w6, R-wave6. The SNAIL row is measured by `SnailSineTest`.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Conditional NP; attentive NP, sine 10-shot | MSE; NLL; 95% coverage | CNP 0.692; 1.01; 0.93. ANP 1.001; 1.08; 0.96 | MAML: 0.652; 2.06; 0.68. Local GP: 0.116; -0.52; 1.00 | NPs give the calibrated uncertainty MAML lacks, at a similar MSE; a local GP is far better. |
| SNAIL, 800 iterations | MSE | 2.153 | Untrained: 4.10. MAML: 0.652 | Negative: far behind MAML; 0.844 at 1200 iterations (R-wave6). |

### UncertaintyBenchmark (W6 5.12: SWAG, Bayes by backprop, SGLD, 5.13: evidential learning, SNGP, DUQ)
Sources: Run-w6, R-wave6. Measured by `UncertaintyOodTest` and `UncertaintyGapTest`.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| SNGP; Bayes by backprop; evidential; DUQ; SWAG; SGLD, shapes with OOD inputs | Accuracy; NLL; ECE; OOD AUROC | SNGP 0.920; 0.252; 0.072; 0.827. BBB 0.918; 0.244; 0.025; 0.777. Evidential 0.900; 0.340; 0.093; 0.802. DUQ 0.893; 0.313; 0.043; 0.781. SWAG 0.883; 0.306; 0.027; 0.775. SGLD 0.875; 0.380; 0.054; 0.752 | MAP: 0.880; 0.374; 0.065; 0.764. Deep ensemble (5 passes): 0.923; 0.238; 0.048; 0.835. MC dropout: 0.905; 0.289; 0.053; 0.780. Laplace: 0.880; 0.427; 0.141; 0.788 | SNGP comes closest to the ensemble in a single pass; BBB and SWAG are the best calibrated. SNGP is scored by the entropy of its adjusted softmax (R-wave6). |
| SGLD; SWAG; BBB, regression with a gap | NLL; predictive std in the gap / where there is data | SGLD -0.96; 0.146 / 0.108. SWAG -0.59; 0.100 / 0.101. BBB -0.38; 0.252 / 0.207 | GP: -0.89; 0.275 / 0.112. Ensemble: -0.72; 0.202 / 0.101. MC dropout: -0.43; 0.202 / 0.194 | SGLD has the best NLL. SWAG does not widen its error bars in the gap (gap NLL 0.81). |

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


### DiffusionForcingRollouts (W5 6.7: diffusion forcing)
Sources: Run-w5, R-wave5.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Diffusion forcing with a noisy past, Lorenz, 200-step rollouts | 10-step RMSE; attractor distance; share diverged | 0.219; 0.093; 0.00 | Teacher-forced AR: 0.127; 2.640; 0.94. Full-sequence diffusion: 0.305; 0.136; 0.00. Diffusion forcing with a clean past: 0.303; 0.142; 0.00 | AR is best short-term but diverges 94% of the time; diffusion forcing stays on the attractor and is the best there. The causal model is a Markov window (an MLP over the last 4 tokens) (R-wave5). |

---

## vision

### VisionShootout
Sources: R-tricks, Test, Run.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| CNN, deformable CNN, ConvNeXt, ViT, ViT with 2-D RoPE, MLP-Mixer on 8x8 shape images | Test accuracy | Every architecture at least 0.78 | Plain CNN | All experimental blocks work as drop-ins in `Model.sequential()`. The test asserts > 0.6 each. |
| Per model (Run) | Test accuracy | Deformable CNN 1.0; ViT 0.996; ViT + 2-D RoPE 0.988; ConvNeXt 0.947; MLP-Mixer 0.783 | Plain CNN: 0.947 | Deformable convolution and ViTs lead; MLP-Mixer trails clearly. |

### DataPruningExperiment (W4 2.14: data pruning by difficulty)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| EL2N and forgetting scores, digits with 15% of labels flipped | Share of flipped labels among the top 15% scored | EL2N 0.49; forgetting 0.43 | Chance: 0.15 | Both scores find mislabelled examples about 3x better than chance. |
| Keep 50% by score | Test accuracy | EL2N hard 0.582; EL2N easy 0.623; forgetting easy 0.647 | Random half: 0.645; all data: 0.712 | Negative: score-based pruning does not beat random pruning at this data size. Measured with an MLP on `ToyTasks.digits`, not in VisionShootout (R-wave4). |


### ClassicCnnShootout (W6 2.2: DropBlock, Shake-Shake, ShakeDrop, 2.4: DenseNet, 2.5: squeeze-excitation, MBConv, compound scaling)
Sources: Run-w6. 256 training images of shapes. Residual rows are measured by `ResidualRegularizersTest`, scaling by `CompoundScalingTest`.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| MBConv; squeeze-excitation; DenseNet | Accuracy; params; multiply-adds | MBConv 0.932; 8,094; 268k. SE 0.801; 5,550; 87k. DenseNet 0.744; 3,758; 127k | Plain CNN: 0.797; 4,998; 79k | MBConv wins clearly, but it is not cheaper here. |
| DropBlock; spatial dropout | Accuracy | DropBlock 0.795. Spatial dropout 0.760 | Plain CNN: 0.797 | No gain. |
| Shake-Shake; ShakeDrop | Accuracy; params | Shake-Shake 0.943; 18,950. ShakeDrop 0.951; 9,606 | ResNet: 0.953; 9,606. ConvNeXt: 0.629 (with drop path 0.557) | No gain over the plain ResNet. |
| Compound scaling at φ = 2 | Accuracy (multiply-adds) | 0.898 (221k) | Width only: 0.912 (118k). Base: 0.789 (36k) | Negative: width-only scaling is better and cheaper. |

### ShapeSegmentation (W6 2.6: U-Net)
Sources: Run-w6. The denoiser row is measured by `UNetDenoiserTest`.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| U-Net, ShapeImages masks | mIoU; pixel accuracy | 0.458; 0.979 | Same network without skips: 0.051; 0.947 | The skip connections are what make segmentation work. |
| U-Net denoiser in TinyDdpm, 250 steps | Nearest-real distance; class match | 19.6; 0.167 | MLP denoiser: 46.1; 0.292. Untrained: 107.7 | Samples land much closer to real images. |

### ObjectDiscovery (W6 2.13: slot attention)
Sources: Run-w6, R-wave6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Slot attention with a spatial-broadcast decoder | FG-ARI; reconstruction MSE | 0.303; 0.0762 | Spatial k-means: 0.237. Plain autoencoder: MSE 0.0505 | Finds objects better than k-means, though the margin depends on the seed (R-wave6). |

### RelationalShapes (W6 2.11: relation networks)
Sources: Run-w6.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Relation network, Sort-of-CLEVR-lite | Accuracy, non-relational / relational | 0.787 / 0.622 | CNN + MLP: 0.652 / 0.579 | Better on both question types. |

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

### Wave 4

The ten most interesting wave 4 findings (Run-w4):

1. **Gradients through physics win big.** Backprop through a mass-spring simulator ends 0.024 from the goal, against 0.240 for CEM and 0.691 for PPO at equal samples (control).
2. **Rationalisation makes self-training work.** STaR lifts chain arithmetic from 0.266 to 0.813, against 0.469 for rejection-sampling fine-tuning without hints (reasoning).
3. **A query-aware KV offload keeps the model intact.** With a 4-bit key sketch, logit error is 0.034 and top-1 agreement 0.67, against 0.086 and 0.21 for a sliding window at the same device bytes (language).
4. **QK-clip tames Muon.** MuonClip caps the attention logit at 10 against 680 for plain Muon, and reaches accuracy 1.000 against 0.936 (optim).
5. **Stochastic rounding rescues bf16 weights.** MSE 0.551 against 8.108 for round-to-nearest, close to fp32's 0.510 (optim).
6. **Tree drafts accept the most tokens.** EAGLE's draft tree verifies 3.00 tokens per target pass, against 2.44 for core speculative decoding with a distilled draft model, all with exact greedy output (language).
7. **Canon layers solve recall in one block.** Accuracy 0.375 → 1.000 for a single block, and 0.387 → 1.000 when added to the core `TransformerBlock` (sequence).
8. **A neural proposer completes a symbolic prover.** Proof rate 0.978 against 0.089 for the engine alone and 0.689 with random auxiliary terms (reasoning).
9. **Fill-in-the-middle is free.** Infill token accuracy 0.910 against 0.098, with left-to-right loss unchanged (1.274 vs 1.271) (language).
10. **More forced thinking, more accuracy.** Budget forcing rises from 0.06 at budget 0 to 0.88 at budget 4, against 0.38 unconstrained (reasoning).

Honourable mentions: CLIP and SigLIP classify digits zero-shot at 0.786 and 0.780, against 0.312 for a 5-shot SimCLR probe (multimodal). On-policy distillation (GKD) lifts the student to 0.61 against 0.48 for logit KD (language). KIVI 2-bit holds the attention error to 0.45 where H2O at the same bytes gives 1.21 (context). A Generative Agents memory stream recalls 0.89 against 0.58 for a sliding window (agents). DreamCoder-style library learning solves 50 of 65 tasks against a flat 38 for a fixed DSL (evolution). v-prediction halves the diffusion sample distance, 0.27 against 0.57 for ε-prediction (generative). Stick-breaking attention keeps lag-2 accuracy at 0.998 at 4x the training length, against 0.938 for RoPE (attention).

### Wave 5

The ten most interesting wave 5 findings (Run-w5):

1. **Sparse memory fine-tuning learns without forgetting.** After 100 streamed facts it scores 0.81 on the new facts and keeps background accuracy at 0.91, against 0.20 / 0.06 for full fine-tuning, 0.22 / 0.16 for LoRA and 0.21 / 0.10 for EWC (optim).
2. **Streaming DiLoCo cuts traffic about 270x for free.** Loss 0.027 with 45 kB communicated, the same loss as synchronous data parallel with 12,009 kB (optim).
3. **Only latent adversarial training removes the backdoor.** Backdoor success 0.00 against 1.00 for plain safety SFT and for input-space adversarial training (safety).
4. **Circuit breakers stop every attack.** Mean harmful-completion rate 0.00 over four attacks against 0.27 for refusal SFT, which fails the 2-token prefill at 0.97 (safety).
5. **Accumulate synthetic data, don't replace it.** At generation 5 the test loss is 2.433 when data accumulates against 5.108 when each generation replaces the last (data).
6. **Multiturn-aware reward survives sharded conversations.** CollabLLM keeps 0.99 when a request arrives in pieces, while SFT falls from 0.90 to 0.14 and single-turn GRPO from 1.00 to 0.16 (reasoning).
7. **EntiGraph teaches reversed and two-hop facts.** Fact QA 1.00 overall against 0.50 for the raw corpus and 0.40 for paraphrases (data).
8. **x-prediction keeps high-dimensional diffusion on the manifold.** Distance per pixel 0.216 against 0.751 for v-prediction and 3.953 for ε-prediction in 512 dimensions (generative).
9. **Per-channel gating and Householder products track state.** Kimi Delta Attention recalls 0.969 against 0.648 for Gated DeltaNet, and DeltaProduct with two steps solves parity at 1.000 against 0.508 (sequence).
10. **A nested FFN beats a model trained at its size.** MatFormer's width-16 slice reaches validation loss 2.8361, against 2.8605 for a model trained from scratch at width 16 and 2.9841 for a truncated plain model (attention).

Honourable mentions: multi-token attention solves block lookup at 0.949 against 0.395 for MHA (attention). Hydragen runs shared-prefix attention 5.0x faster with identical output (language). GenRM-CoT picks the right answer from 16 samples at 0.781 against 0.734 for a PRM (reasoning). Pass@4 training lifts pass@16 from 0.72 to 0.96 (reasoning). RLVR narrows coverage: the base model overtakes GRPO at k=64 (0.67 vs 0.57 at k=256), reproducing Yue et al. (reasoning). QTIP quantizes to 2 bits at decoder KL 6.4e-3 against 1.85e-2 for HQQ (techniques). Prefill/decode disaggregation raises goodput from 56.7 to 78.4 req/s with every request in SLO (language). CoPE solves the k-th-word task at 1.000 against 0.820 for RoPE (techniques).

### Wave 6

The ten most interesting wave 6 findings (Run-w6):

1. **A CRF head never emits an illegal tag.** On BIO chunking a GRU with a linear-chain CRF reaches span F1 0.998 with no illegal transitions, against 0.285 and a 0.129 illegal rate for a softmax head on the same GRU (classic).
2. **Successor features transfer instantly.** SF + GPI scores 0.700 zero-shot on a new reward, against -1.600 for Q-learning from scratch at 300 steps and 0.617 at 4000 (control).
3. **Reward machines solve sequential tasks.** Counterfactual reward-machine learning solves "coffee, then mail, then office" by 20k steps; Q-learning on the product and history-free Q-learning never do, even at 60k (control).
4. **Oscillators bind objects and solve puzzles.** AKOrN reaches object FG-ARI 0.84 against 0.16 for k-means, and solves 0.62 of Latin squares against 0.15 for TRM (exotic).
5. **Inoculation prompting stops emergent misalignment.** A narrow bad fine-tune complies off-task 1.00 of the time; the same fine-tune with an inoculation prompt complies 0.00 (safety).
6. **A cartridge holds a corpus in a quarter of the cache.** At 1024 KV bytes it answers 0.88, the same as the full 4096-byte context, against 0.25 for recency eviction (context).
7. **Rubrics resist reward hacking.** A rubric reward lifts held-out satisfaction to 0.82, while a scalar reward model and a Likert judge are hacked down to 0.40 and 0.38, below SFT's 0.61 (reasoning).
8. **Training through the device closes the sim-to-real gap.** Physics-aware training scores 0.74 on the diffractive device, against 0.40 for in-silico and noise-aware training (exotic).
9. **Recurrent replay needs stored state.** R2D2 solves the cued T-maze 0.88 of the time, against 0.00 for zero-state replay and 0.52 for a frame-stack DQN (control).
10. **Latent diffusion is many times faster.** In a VAE latent, distance 0.194 after 150 steps, against 0.436 for pixel diffusion after 800 (generative).

Honourable mentions: DroQ doubles SAC's Pendulum return at an update ratio of 5 (-355 vs -759), and TD-MPC reaches -380 against -736 for SAC (control). A Bayesian flow network writes valid words 0.891 of the time in 3 steps, against 0.641 for masked diffusion (generative). An MPS Born machine at bond 32 samples exact images 83% of the time, against 1% for PixelCNN (exotic). NALU extrapolates multiplication with 0.037 relative error, against 0.96 for MLPs (reasoning). GP Bayesian optimisation finds 0.049 in 30 trials against 0.141 for TPE, and BOHB beats Hyperband 0.044 to 0.106 (classic). A pointer network finds convex hulls at 0.379 against 0.012 for seq2seq (sequence). ControlNet follows the map at IoU 0.615 against 0.516 for full fine-tuning, and leaves the base model exactly as it was (generative). ELMo features tag ambiguous words at 1.000 against 0.494 for static vectors (language).

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

Wave 4 (Run-w4):

| Theme | Experiment | Result | What it means |
|---|---|---|---|
| optim | Grokfast | Test accuracy passes 0.99 at step 900 vs 850 for AdamW + weight decay | No speed-up; other settings were no better (R-wave4). |
| optim | ModernOptimizerExperiment, MARS-AdamW | 183 steps to loss 1.2 vs 150 for AdamW | Not faster, despite a slightly lower final loss (1.037 vs 1.043). |
| optim | ModernOptimizerExperiment, Cautious | C-AdamW 1.061 vs 1.043; C-Lion 1.057 vs 1.047 | The cautious mask does not help at this budget. |
| optim | PreconditionerStability, StableAdamW and Adam-atan2 | 6 of 6 seeds diverge at lr 0.1, vs 5 of 6 for AdamW | Neither variant lowers the divergence rate. |
| vision | DataPruningExperiment | Keeping 50% by score: 0.582 to 0.647 vs 0.645 for a random half | Score-based pruning does not beat random pruning, though the scores do find flipped labels. |
| reasoning | TestTimeRl (TTRL) | Greedy accuracy 0.719 → 0.406; self-consistency baseline 0.609 | The majority vote is worse than greedy here, so TTRL trains toward a worse target. |
| reasoning | QuietThoughts (Quiet-STaR) | 0.652 vs 0.625 with no thoughts | Roughly a tie; the thought matches the intermediate hop on only 0.078 of questions. |
| reasoning | Constitution (Constitutional AI) | Violations 0.76 → 0.32, but addition accuracy 1.00 → 0.56 | The rule-following gain costs a lot of task accuracy. |
| language | DistillationShootout, MiniLLM | Student accuracy 0.17 vs 0.48 for logit KD | Pure reverse KL without a pretraining loss collapses the student. |
| language | LoraScaling, LoftQ | Loss 0.335 vs 0.122 for QLoRA after 30 steps | A better 4-bit start (error 0.596 vs 0.833) does not lead to better training. |
| language | ConceptVsTokenLm | Next-sentence accuracy 0.839 vs 1.000 for the token LM | Fewer calls, lower accuracy. |
| attention | MoeRefinements, sparse upcycling | MSE 0.880 (drop-upcycled 0.964) vs 0.541 for a MoE from scratch at equal compute | Upcycling loses clearly to training the MoE from scratch. |
| attention | MoeRefinements, expert choice | MSE 0.708 vs 0.633 for top-2 token choice | Perfect balance, but unchosen tokens get zero output with no residual path. |
| attention | RecallExtrapolation, sigmoid attention | Recall at 16 pairs 0.820 vs 0.957 for softmax | Better LM loss, worse extrapolation. |
| attention | RecallExtrapolation, iRoPE | Recall at 16 pairs 0.941 vs 0.977 for all-NoPE | Beats all-RoPE but not all-NoPE. |
| techniques | LowBitQuantization, Hadamard rotation | Outlier-layer error 0.289 (with GPTQ 0.227) vs 0.168 for SmoothQuant | Rotation does not beat SmoothQuant when the outlier channels are fixed. |
| generative | ThermodynamicDenoising (DTM) | Valid share 0.963 vs 0.973 for a single Boltzmann machine; on 6x6 data 0.413 vs 0.903 (R-wave4) | Chaining denoising stages gives no gain over one machine. |
| generative | MaskGit vs autoregressive | At 16 steps: distance 0.056, real share 0.58 vs 0.049 and 0.75 | MaskGIT trails the autoregressive model at equal steps. |
| generative | NextScalePrediction (VAR) | Real share 0.09 and 7 distinct vs 0.15 and 13 for MaskGIT at 3 calls | Lowest distance (0.209 vs 0.215), but less realistic and less varied. |

Wave 5 (Run-w5):

| Theme | Experiment | Result | What it means |
|---|---|---|---|
| optim | MuonFamilyExperiment, Polar Express | Muon loss 1.111 / 1.109 (3 / 5 iterations) vs 1.101 / 1.106 with Newton-Schulz | A far better orthogonaliser (error 0.174 vs 0.346 at 5 iterations) gives no training gain. |
| data | DataCuration, quality classifier | Clean loss 1.899 (59% clean kept) vs 1.705 (77%) for a perplexity filter | The classifier helps over no filter (2.084) but loses to perplexity filtering. |
| data | DataCuration, DoReMi | Target loss 2.482 with weights [0.25, 0.24, 0.25, 0.26], same as uniform | DoReMi does not move away from uniform weights; RegMix reaches 2.175. |
| data | ShapleyPruning, In-Run Shapley | Flipped labels found 0.35 / 0.43 vs 0.56 for EL2N; test accuracy 0.685 vs 0.690 random | Above chance, but EL2N detects more and no method beats random pruning. |
| data | BenchmarkMethodology, Min-K%++ | AUROC 0.746 vs 0.755 for plain Min-K% | Beats the loss threshold (0.702) but not Min-K%. |
| attention | SparseSelectionShootout, MoBA (and DSA) | MoBA recall 0.227 / 0.195 / 0.133 vs dense 0.578 / 0.391 / 0.219; DSA 0.438 / 0.297 / 0.281 | Block routing loses recall at every length; DSA beats dense only at the longest. |
| attention | MoeRefinements, ReMoE | MSE 1.4649 vs 0.6333 for top-2 at the same experts per token | ReLU routing hits the sparsity target with more than twice the error. |
| attention | AugmentedResidualDepth, LAuReL | 3.0008 / 3.0000 vs 3.0000 plain; value residual 2.8729 | No gain at this budget. |
| attention | Gemma3nBlocks, per-layer embeddings | 2.8939 vs 2.7967 for plain d=16 | PLE is slightly worse than plain on this toy task. |
| reasoning | RecurrentDepthScaling | 0.426 → 0.488 from 1 to 8 iterations vs 0.840 for a fixed 3-block stack | Test-time depth helps only a little, and a fixed stack wins clearly. |
| language | Eagle3Speculation, EAGLE-3 | Tokens per pass 2.22 / 2.58 (chain / tree) vs 2.42 / 2.76 for EAGLE | Feature fusion overfits at toy scale (R-wave5). |
| language | SliceCompression, SliceGPT | 2.618 at 75% width vs 2.085 for Wanda and 2.139 for dropping a block | The width-32 residual stream is not low-rank enough to slice. |
| language | DynamicChunking, H-Net | 0.807 bits/byte vs 0.679 for BPE | Learned chunking does not reach BPE at this budget. |
| reasoning | RolloutMismatch, TIS | Learner accuracy 0.68 with TIS vs 0.72 without | Truncated importance sampling does not repair 2-bit rollouts here. |
| reasoning | Gmpo | Mean per-step ratio spread 0.383 vs 0.358 for GRPO | The geometric mean does not stabilise the ratios; GSPO does better (0.284). |
| reasoning | SpuriousRewards and Intuitor | Format-only -0.02, random -0.27; Intuitor 0.06 → 0.12 vs 0.06 → 0.10 for a random reward | No reward-independent gains; on the default base Intuitor collapses (0.49 → 0.12, R-wave5). |
| reasoning | ConfidenceVoting, DeepConf-online | 0.281 vs 0.438 self-consistency, at 63.4 vs 63.9 tokens | Early stopping costs accuracy and saves almost no tokens. |
| generative | DiffusionArithmeticRl, JustGRPO | Parallel decoding 0.18 vs 0.30 SFT (left-to-right 0.56) | The left-to-right gain does not carry over to parallel decoding. |
| interp | MechanismDecomposition, SPD | 1 of 5 mechanisms vs 5 of 5 directions for an SAE | SPD did not converge in the CPU budget. |
| interp | CircuitTracing, cross-layer transcoder | B-feature recovery 0.50 vs 0.75 for per-layer transcoders | Better fit, fewer planted features recovered. |
| safety | PersonaSteering, ablation | Trait 0.98 after ablating the persona direction vs 1.00 without | The fine-tuned trait is not carried only along the persona vector. |
| generative | RepaExperiment, REPA | 0.551 vs 0.560 at 500 steps (and worse at 150 and 300) | No speed-up at toy scale. |
| reasoning | CompressArc | 1 of 8 ARC-like tasks; Latin squares 0 of 6 vs 0.33 TRM and 0.17 HRM | Only one task family compresses, and it loses to the trained reasoners. |
| control | ContrastiveGoals, deeper CRL | Depth 16 0.80 vs depth 4 0.85, both below HER + DQN at 1.00 | Depth does not help contrastive RL at this scale. |
| control | ContinuousOfflineRl, Q-chunking | FQL 0.49 with Q-chunking vs 0.63 without; CQL 0.99 | Chunking hurts here, and FQL trails CQL. |

Wave 6 (Run-w6):

| Theme | Experiment | Result | What it means |
|---|---|---|---|
| agents | NeverGiveUpExploration, NGU and Agent57 | Cells / goals: NGU 42.0 / 0, Agent57 23.3 / 0 vs RND 51.0 / 13.7 and no bonus 26.3 / 0 | Neither beats RND; Agent57 falls below no bonus. |
| attention | PerceiverExperiments | Shapes 0.387 vs 1.000 for ViT; reversal 0.211 vs 0.882 for a full transformer | The latent bottleneck learns slowly and passes too little at this budget. |
| attention | SetLearning, unique count | At n = 12: Deep Sets 0.333, Set Transformer 0.090 vs 0.867 in distribution for a sorted MLP | No set model extrapolates the count. |
| classic | GaussianProcessUncertainty, deep kernel learning | NLL -0.697 vs -0.738 for the MLP alone and -0.778 for a GP on raw windows | Learned features do not help the GP here. |
| classic | ForestBenchmarks | Random forest 0.830 vs MLP 0.894; isolation forest AUROC 0.967 vs 0.994 for an autoencoder | Forests lose to the neural baselines on both tasks. |
| control | DistributedActorCritic, V-trace | Return at lag 32: 152.9 vs 200.0 uncorrected | The correction hurts on this task. |
| control | MpoComparison | -790 vs -631 for SAC and -677 for PPO | MPO needs a larger budget (-482 at 4000 steps, R-wave6). |
| control | OptionsAndTransfer, option-critic | First-30-episode length 190.6 vs 150.5 for Q-learning | Options do not speed up re-adapting. |
| control | GumbelSearch | Score vs PUCT 0.38 to 0.60 over 2–32 simulations | No edge at small budgets on tic-tac-toe. |
| exotic | QuantumClassifier | 1.00 on moons, tied with a 34-parameter RFF model | No advantage over a tiny classical model. |
| generative | StyleGanExperiment | Sliced W 0.209 (0.311 with path-length reg) vs 0.139 for a DCGAN-style MLP | The plain generator wins at this budget. |
| generative | JemExperiment | Accuracy 0.956, ECE 0.147 vs 0.980, 0.036 plain | JEM is worse calibrated, and its samples are barely better than noise. |
| generative | DiscreteGenerators, discrete flow matching | 0.688 at 12 steps (mask source) vs 0.875 for masked diffusion | No DFM variant beats masked diffusion. |
| generative | OneStepFrontier, sCM | Distance 0.697 / 0.787 / 0.794 at 1 / 2 / 4 steps vs 0.444 for discrete consistency | Continuous-time consistency is unstable here. |
| generative | DensityBenchmark, Glow and MAF | Moons NLL 1.250 and 1.679 vs 1.209 for RealNVP | Only the spline flow beats RealNVP. |
| generative | DiffusionSamplers, score SDE | Ring distance at 20 NFE: 0.420 (PC) and 0.302 (ODE) vs 0.153 for DDIM | Both score-SDE samplers trail DDIM. |
| generative | EnergyMatchingShootout | NLL 2.223 vs 1.896 for flow matching | Same sample quality, worse density. |
| generative | VqGanExperiment | Real share 0.479 vs 0.575 for VQ-VAE at 4 codes | Sharper codes, fewer real reconstructions. |
| language | UlmfitTransfer | Source forgetting +5.14 vs +6.17 for full fine-tuning | Less forgetting, but every method forgets heavily. |
| optim | ClassIncrementalDigits, GEM and A-GEM | 0.345 and 0.235 vs 0.560 for replay | Gradient projection loses to plain replay. |
| reasoning | ArithmeticExtrapolation, NAC | Addition error 0.014 vs 0.0017 for a ReLU MLP | The MLP extrapolates addition better. |
| reasoning | SelfDistillationVsSft | Old task 1.00 → 0.38, the same as SFT | The paper's less-forgetting claim did not reproduce. |
| reasoning | ParallelAndSoftThinking, ParaThinker | 0.75 vs 0.80 for self-consistency at similar tokens | The learned merger does not beat majority voting. |
| reasoning | PowerSampling | pass@1 0.48 vs 0.48 for T = 1/4, at 11.1 vs 2.0 tokens | Ties low-temperature sampling at 5x the cost. |
| reasoning | ProlongedRlCoverage | pass@64 0.61 vs 0.74 for the base model | Prolonged RL still only sharpens. |
| reasoning | SelfPlayGames | Elo 923 with RAE vs 1009 with a shared baseline; arithmetic 0.45 → 0.00 for every model | RAE does not help, and nothing transfers to arithmetic. |
| safety | CausalAndFairness, IRM | Worst group 0.268 vs 0.379 for ERM | IRM is worse than ERM at every penalty weight tried (R-wave6). |
| safety | CausalAndFairness, NOTEARS | SHD 5 vs 3 for PC | Continuous structure learning loses to PC here. |
| sequence | TestTimeRegressionShootout, MesaNet | 1.00 in 3817 ms vs 1.00 for TTT-Linear in 735 ms | Exact regression gives no recall gain at 5x the time. |
| techniques | ClassicRegularizers, DropConnect and concrete dropout | 0.488 and 0.536 vs 0.596 for dropout | Neither beats plain dropout. |
| techniques | CnnNormalizationShootout, FRN | 0.35 at batch 2, 0.48 non-iid vs 0.79 and 0.98 for BN | Batch-free, but it fails exactly where it should help. |
| techniques | MetaLearningShootout, LEO and CNAPs | LEO 5-shot 0.573 vs 0.709 MAML; CNAPs 0.720 vs 0.754 without FiLM | LEO trails MAML, and FiLM adaptation hurts CNAPs. |
| techniques | SineMetaRegression, SNAIL | MSE 2.153 vs 0.652 for MAML | SNAIL is slow to train on sines. |
| vision | ClassicCnnShootout, compound scaling | 0.898 at 221k MACs vs 0.912 at 118k for width only | Compound scaling loses to width-only scaling. |
| vision | ClassicCnnShootout, DropBlock, Shake-Shake, ShakeDrop | 0.795, 0.943, 0.951 vs 0.797 CNN and 0.953 ResNet | No gain from these regularisers at this size. |

Overturned by the Run: second-order MAML was listed here (1.89 vs 0.81 for first-order, with finite-difference Hessian-vector products). With exact second-order maths (PR #105) it scores 0.65 and is now the best meta-learner in MetaLearningExperiment.
