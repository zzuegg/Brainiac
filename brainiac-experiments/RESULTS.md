# brainiac-experiments: measured results

This page collects every measured result recorded for the experimental AI techniques in the `brainiac-experiments` module of the brainiac repository (PRs #86, #92 and #97), plus a full run of the experiment suite on 2026-11-17 after PR #105. It also covers wave 4: 85 more techniques in six work packages (WP1–WP6, listed in `unimplemented-ideas.md`), each measured against a baseline. Wave 4 experiments are tagged **W4** with their item number, for example "W4 1.1".

## How the module is laid out

- **`io.github.zzuegg.brainiac.experimental.*`** holds the reusable building blocks: layers, optimizers, learning rules, memory modules, environments and so on. These packages mirror the core packages they could move to (`experimental.nn`, `experimental.train`, `experimental.rl`, `experimental.llm`, `experimental.evolve`, `experimental.data`, `experimental.tensor`, `experimental.autograd`). The blocks implement the core interfaces (`Layer` + `LayerSpec`, `Optimizer`, `Loss`, `Environment`, `LanguageModel` and others), so swapping one in is a one-line change. Promoting a block to core means dropping `.experimental` from its package name.
- **`io.github.zzuegg.brainiac.experiments.<theme>`** holds the experiments themselves: toy tasks, comparisons against baselines, and demos. Each theme below is one of these packages.

## How to read the numbers

Every result comes from a small synthetic task that runs on a CPU in seconds, with fixed seeds and one run per configuration, usually at a small training budget. **The numbers are directional, not benchmarks.** They show that a technique works in this codebase and roughly how it compares with a baseline on the same toy task. They say nothing reliable about real-scale performance, and small gaps (a few points) are within run-to-run noise.

Rules this page follows:

- Every number was found in one of the sources below. None was estimated. Numbers keyed **Run** were printed by the tests in the full suite run of 2026-11-17 (577 tests, all passing); everything else predates that run. Tests that printed nothing in that run were then given print statements (no assertion, threshold, seed or budget changed) and the module was run again the same day (576 tests, all passing). Their numbers are also keyed **Run**.
- Where a Run number clearly disagrees with an earlier recorded number, both are given and the difference is noted. PR #105 changed four things numerically: GumbelChannel's noise stream, and exact second-order maths in MAML, Sophia and GradientPenalty.
- Wave 4 numbers are keyed **Run-w4**: they were printed by the tests in the full suite runs after wave 4 on 2026-11-17, one of the merged WP1–WP5 work (690 tests, all passing) and one of the WP6 branch (601 tests, all passing). Where a wave 4 test printed nothing on a point, the number comes from the implementing agent's final report and is marked **(R-wave4)** next to it. Where the two disagree, the test's number is given and the report's is noted.
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

### DataPruningExperiment (W4 2.14: data pruning by difficulty)
Sources: Run-w4, R-wave4.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| EL2N and forgetting scores, digits with 15% of labels flipped | Share of flipped labels among the top 15% scored | EL2N 0.49; forgetting 0.43 | Chance: 0.15 | Both scores find mislabelled examples about 3x better than chance. |
| Keep 50% by score | Test accuracy | EL2N hard 0.582; EL2N easy 0.623; forgetting easy 0.647 | Random half: 0.645; all data: 0.712 | Negative: score-based pruning does not beat random pruning at this data size. Measured with an MLP on `ToyTasks.digits`, not in VisionShootout (R-wave4). |

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

Overturned by the Run: second-order MAML was listed here (1.89 vs 0.81 for first-order, with finite-difference Hessian-vector products). With exact second-order maths (PR #105) it scores 0.65 and is now the best meta-learner in MetaLearningExperiment.
