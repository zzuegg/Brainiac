# brainiac-experiments: measured results

This page collects every measured result recorded for the experimental AI techniques in the `brainiac-experiments` module of the brainiac repository (PRs #86, #92 and #97), plus a full run of the experiment suite on 2026-11-17 after PR #105. It also covers wave 4: 85 more techniques in six work packages (WP1–WP6, listed in `unimplemented-ideas.md`), each measured against a baseline. Wave 4 experiments are tagged **W4** with their item number, for example "W4 1.1". It also covers wave 5: 106 more techniques in six work packages (WP1–WP6, listed in `wave5-ideas.md`), tagged **W5** in the same way, with two new themes, data and safety. It also covers wave 6: 105 more techniques in six work packages (WP1–WP6, listed in `wave6-ideas.md`), tagged **W6** in the same way and measured in 84 new experiments, with one new theme, classic. It also covers wave 7: 108 more techniques in six work packages (WP1–WP6, listed in `wave7-ideas.md`), tagged **W7** in the same way and measured in 59 new experiment entries (91 new test classes), with six new themes: audio, multiagent, numerics, probabilistic, recsys and synthesis. It also covers wave 8: 106 more techniques in six work packages (WP1–WP6, listed in `wave8-ideas.md`), tagged **W8** in the same way and measured in 38 new experiment entries (124 new test classes), with six new themes: compression, kernels, privacy, robustness, sciml and serving. It also covers wave 9: 102 more techniques in six work packages (WP1–WP6, listed in `wave9-ideas.md`), tagged **W9** in the same way and measured in 51 new experiment entries (104 new test classes), with ten new themes: alignment, causal, code, document, edge, evaluation, parallel, streaming, survival and world. Wave 9 is gathered in its own section, one part per work package, after the themes. It also covers wave 10: 101 more techniques in six work packages (WP1–WP6, listed in `wave10-ideas.md`), tagged **W10** in the same way and measured in 39 new experiment entries (106 new test classes), with eight new themes: events, labels, metric, multitask, parsing, restoration, shift and speech. Wave 10 is gathered in its own section after wave 9, one part per work package; all its numbers are single-seed and at test budgets, because its long-budget Runs were not executed.

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
- Wave 7 numbers are keyed **Run-w7**: they were printed by the tests in one full suite run of the merged wave 7 work on 2026-11-18 (543 test classes, 1,150 tests, all passing). Where a wave 7 test printed nothing on a point, the number comes from the implementing agent's final report and is marked **(R-wave7)** next to it. Where the two disagree, the test's number is given and the report's is noted.
- Wave 8 numbers are keyed **Run-w8**: they were printed by the tests in one full suite run of the merged wave 8 work on 2026-11-18 (667 test classes, 1,328 tests, all passing). Where a wave 8 test printed nothing on a point, or the number comes from a larger run than the test budget allows, it is taken from the implementing agent's final report and marked **(R-wave8)** next to it. Where the two disagree, the test's number is given and the report's is noted.
- Wave 9 numbers are keyed **R-wave9**: they come from the final reports of the six wave 9 work-package agents, who ran their own test classes one at a time with `--tests` filters on 2026-11-18 (all passing). No full suite run of the merged wave 9 work has been made, so there is no Run-w9 key. Where an agent's first report and its follow-up after fixes disagree, the follow-up is used. Most wave 9 rows are one seed, and small gaps are within noise.
- Wave 10 numbers are keyed **R-wave10**: they come from the final reports of the six wave 10 work-package agents, who ran only their own test classes, one or two at a time with `--tests` filters, on 2026-11-18 (all passing when last run), because the user asked for no extensive test runs. No full suite run of any wave 10 branch or of the merged branch has been made, so there is no Run-w10 key. Every wave 10 number is single-seed and at the test budget. The long-budget `main()` Runs, including most of the wave 9 follow-ups, were written but not executed, so a hypothesis that needs them is marked **Inconclusive (long run pending)** unless the test-budget numbers already settle it.
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
| Run-w7 | Full suite run after wave 7, 2026-11-18: merged WP1–WP6 work (543 test classes, 1,150 tests), all passing |
| R-wave7 | Session transcript: final reports of the six wave 7 work-package agents (WP1–WP6) |
| Run-w8 | Full suite run after wave 8, 2026-11-18: merged WP1–WP6 work (667 test classes, 1,328 tests), all passing |
| R-wave8 | Session transcript: final reports of the six wave 8 work-package agents (WP1–WP6) |
| R-wave9 | Session transcript: final reports of the six wave 9 work-package agents (WP1–WP6), including their follow-ups after fixes; per-class test runs, 2026-11-18 |
| R-wave10 | Session transcript: final reports of the six wave 10 work-package agents (WP1–WP6); per-class test runs only, single seed at test budgets, 2026-11-18; long-budget `main()` Runs not executed |

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

### ToolAgentBench (W8 4.15: Sculptor active context tools, 4.16: ToolRL and ReTool, 4.17: Mixture-of-Agents, 4.18: ArCHer)
Sources: Run-w8, R-wave8. Tests: ToolAgentContextTest, ToolAgentToolUseTest, ToolAgentMixtureTest, ToolAgentMultiTurnTest. The blocks are in `experimental.agent`. Each arm uses one seed over many trials or questions (R-wave8).

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Sculptor context tools, "hide superseded" policy, needles whose key is overwritten later | Accuracy at 16 / 64 context tokens (tokens read at 64) | 0.60 / 0.88 (7) | Sculptor, summarise matches: 0.38 / 0.58 (23). RLM: 0.37 / 0.47 (217). MemAgent (SFT): 0.38 / 0.37 (151). MemGPT pager retrieval: 0.28 / 0.13 (7). Truncation: 0.42 / 0.20 (7) | Hiding stale values wins at a fraction of the tokens. The policy is scripted, not trained (R-wave8). |
| ToolRL (structured reward); ReTool (outcome reward), 2-digit × 1-digit products plus sums | Accuracy; share of needless calls on sums | ToolRL 0.91; 0.84. ReTool 0.87; 0.84 | SFT start: 0.89; 0.84. Toolformer: 0.93; 0.80. Always call: 0.89; 1.00. Never call: 0.19; 0. GRPO without tools: 0.16; 0 | Negative: RL matches Toolformer's accuracy but does not cut needless calls (30 steps, call cost 0.3). Every arm scores 0.00 on 3-digit operands. |
| Mixture-of-Agents (the aggregator sees only the proposals) | Accuracy, complementary / overlapping panel (tokens per question) | 0.99 / 0.79 (47) | Best single agent: 0.33 / 0.33 (11). Majority vote: 0.23 / 0.05 (201 / 182). Debate: 0.19 / 0.05 | MoA wins on both panels, so "only when skills are complementary" is not supported. |
| ArCHer (turn-level critic), guessing game against a simulated user | Sampled success at about 2k / 4k / 6k user interactions | 0.79 / 0.79 / 0.51 | Trajectory GRPO: 0.69 / 0.87 / 0.98. Filtered BC: 0.72 / 0.80 / 0.97. CollabLLM multiturn reward: 0.46 / 0.69 / 0.73. Random player: 0.68 | Negative: ArCHer is fastest early, then degrades below a random player. It was not tuned further (R-wave8). |

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

## audio

New theme in wave 7 (package `experiments.audio`): speech and audio toys built on the TalkingDigits formant synthesiser. `SpokenDigitRecognition` is the shared harness, with speakers (formant scale and speaking rate) and noise at a chosen SNR. DER is the digit error rate (lower is better).

### SpokenDigitRecognition (W7 6.1: MFCC and DTW templates, 6.2: Conformer and Mamba encoders with SpecAugment, 6.3: RNN-Transducer, 6.4: wav2vec 2.0 and HuBERT pretraining)
Sources: Run-w7, R-wave7. Tests: IsolatedWordsTest, ConformerEncoderTest, MambaEncoderTest, SequenceEncodersTest, SpecAugmentTest, TransducerTest, HubertPretrainingTest, Wav2Vec2PretrainingTest, plus the block tests DynamicTimeWarpingTest and TransducerLossTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| MFCC + DTW template matching, isolated words | Word accuracy: 1 shot at 20 dB / 5 shots at 20 dB / 5 shots at 5 dB | 0.80 / 0.91 / 0.66 | MLP + CTC on band energies: 0.50 / 0.82 / 0.43. MLP + CTC on MFCC: 0.61 / 0.86 / 0.66 | Templates win with little data; MFCC features are what hold up in noise. |
| Conformer encoder, digit strings, CTC, 200 steps at 20 dB | DER (parameters) | 0.117 (25k) | MLP on band energies: 0.031 (18k). Transformer: 0.227 (14k). BiGRU: 0.805 (11k) | Negative: the frame-wise MLP wins at this budget. A 300-step probe brings the Conformer to 0.000, and at 0–10 dB every encoder fails (DER 0.55 or worse) (R-wave7). |
| Bidirectional Mamba encoder, 150 steps at 20 dB | DER (parameters) | 0.125 (15k) | MLP: 0.070 (18k) | Negative at this budget; 0.039 at 300 steps (R-wave7). It uses `MambaLayer`'s recurrent scan, because the parallel form ran out of heap (R-wave7). |
| SpecAugment, 48 training strings, 250 steps | DER | MLP + SpecAugment 0.102 | MLP: 0.164 | Masking helps when data is scarce. With a transformer it got worse, 0.438 → 0.469 (R-wave7). |
| RNN-Transducer, beam 4 / greedy, Markov digit strings at 5 dB | DER | Beam 0.060. Greedy 0.147 | CTC: 0.107 | Beam search over the transducer beats CTC; greedy decoding does not. Both emit at the same point (mean position 0.30 vs 0.31). The loss equals brute-force path enumeration (4.53798), gradient error 2.9e-5. |
| HuBERT; wav2vec 2.0 pretraining, then 120 CTC steps on 30 labelled strings | DER | HuBERT 0.578. wav2vec 2.0 0.609 | From scratch: 0.609 | Negative: no reliable gain. Across Conformer seeds the effect was +0.14, -0.05, +0.20 for HuBERT and -0.09, -0.01, +0.01 for wav2vec 2.0 (R-wave7). |

### AudioResynthesis (W7 6.5: DDSP harmonic-plus-noise synthesiser)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DDSP (6,168 parameters), melodies, 60 steps | Multi-scale spectral distance; f0 error | 2.599; 131 cents | Sample-level MLP (6,512 parameters): 4.619; 1200 cents | The differentiable synthesiser gets pitch and timbre right where a same-size sample model cannot. |
| DDSP timbre transfer (A's notes in B's timbre) | Spectral distance to B | 2.615 | MLP: 4.602. B's own audio against A playing its notes: 3.022 | Transfer moves the sound toward the target timbre. |

### SourceSeparation (W7 6.6: Conv-TasNet with permutation-invariant training)
Sources: Run-w7, R-wave7. Tests: ConvTasNetTest, SourceSeparationTest, PermutationInvariantLossTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Conv-TasNet (35,264 parameters), two-speaker digits, 150 steps | SI-SNR improvement at 0 / 5 dB | 2.1 / 2.1 dB | STFT-mask MLP with PIT (31,426 parameters): 4.8 / 5.8 dB. FastICA on two microphones: 19.0 / 19.9 dB. Ideal ratio mask: 22.0 / 20.9 dB | Negative: Conv-TasNet is under-trained and loses to the mask MLP. The uPIT block test finds the right assignment (SI-SNR 26.27 dB). |

### SpokenDigitRecognition: learnable front ends and scattering (W8 2.16: SincNet, LEAF and PCEN, 2.17: wavelet scattering transform)
Sources: Run-w8, R-wave8. Tests: PcenFrontEndTest, ScatteringWordsTest, plus the block test AudioBlocksTest. The test checks one noise level; the full front-end sweep comes from a larger run outside Gradle (R-wave8).

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| PCEN after mel energies, MLP + CTC, digit strings at 5 dB | DER | 0.297 | Log-mel: 0.672 | PCEN more than halves log-mel's error in noise. |
| SincNet; LEAF; mel + PCEN; learned PCEN, 300 steps (R-wave8) | DER at 0 / 5 / 10 / 20 dB | SincNet 0.13 / 0.086 / 0.031 / 0.008. LEAF 0.16 / 0.094 / 0.13 / 0.125. Mel + PCEN 0.29 / 0.14 / 0.15 / 0.11. Learned PCEN 0.30 / 0.14 / 0.125 / 0.11 | MFCC: 0.14 / 0.07 / 0.008 / 0. Log-mel: 0.55 / 0.26 / 0.039 / 0.008 | Partly confirmed: PCEN helps at 0–5 dB but leaves a floor of about 0.11 in clean audio, and fixed MFCC and SincNet are best overall. The learned front ends needed a frozen rescaling onto log-mel's statistics, or CTC collapsed to blanks (R-wave8). |
| Scattering transform + ridge, spoken words, 2 shots per class | Accuracy | 0.810 | DTW on MFCC: 0.760. MFCC + ridge: 0.720. Log-mel + ridge: 0.590 | Best at 2 shots. At 3 and 20 shots it ties DTW (0.81 vs 0.83, 0.97 vs 0.97) (R-wave8). |

### SourceSeparation: dual-path RNN (W8 2.18: DPRNN at a learning-curve budget)
Sources: Run-w8, R-wave8. Test: DprnnSeparationTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DPRNN (27,648 parameters), two-speaker digits, test budget | SI-SNR improvement at 0 / 5 dB | 1.5 / 1.0 dB | Wave 7 at 150 steps: Conv-TasNet 2.1 / 2.1 dB, STFT-mask MLP 4.8 / 5.8 dB | The test checks only that DPRNN learns. |
| DPRNN learning curve at 0 dB (R-wave8) | SI-SNRi at 150 / 600 / 1,500 steps | 3.1 / 5.5 / 7.5 dB | Conv-TasNet (35k parameters): 2.1 / 3.6 / 6.4. STFT-mask MLP (31k): 4.8 / 7.2 / 8.3 | Negative: DPRNN leads Conv-TasNet at every budget, but neither time-domain model passes the mask MLP by 1,500 steps; the gap closes to 0.8 dB. |

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

## compression

New theme in wave 8 (package `experiments.compression`): pruning, distillation, quantisation and low-rank compression on digits MLPs, a shapes CNN, a small ViT, a toy MoE and a toy language model. The blocks are in `experimental.compress`. Rows report test accuracy unless stated, from one seed.

### PruningShootout (W8 5.1: gradual magnitude pruning, 5.2: movement pruning, 5.3: WoodFisher, 5.4: SNIP, GraSP and SynFlow, 5.5: LLM-Pruner, FASP and 2SSP, 5.6: network slimming, 5.7: MoE expert pruning and merging, 5.18: block and n:m sparsity on the LM)
Sources: Run-w8, R-wave8. Tests: GradualPruningTest, PruningCriteriaTest, PruningAtInitTest, StructuredLmPruningTest, NetworkSlimmingTest, ExpertCompressionTest, SparseFormatsLmTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Gradual magnitude pruning (GMP), digits MLP | Accuracy at 91% / 97% sparsity | 0.497 / 0.257 | One-shot + fine-tune: 0.520 / 0.153. Lottery ticket: 0.507 / 0.263. RigL: 0.593 / 0.370. Dense: 0.750 | GMP beats one-shot only at extreme sparsity; RigL is best at both. |
| Movement pruning (hard; soft), during transfer fine-tuning | Accuracy at 50% / 95% sparsity | 0.633 / 0.417. Soft: 0.490 at 96.6% | Magnitude: 0.643 / 0.140 (0.153 at 96.9%). Wanda: 0.627 / 0.133 | Movement ties at 50% and wins by about 0.28 at 95%, as in the paper. |
| WoodFisher (1 and 4 steps); OBD, one-shot then recovery | Accuracy after recovery at 70% / 90% (straight after pruning at 70%) | WoodFisher 0.637 / 0.343 (0.557). WoodFisher ×4 0.640 / 0.437. OBD 0.667 / 0.413 | Magnitude: 0.653 / 0.340 (0.497). Wanda: 0.627 / 0.357. SparseGPT: 0.683 / 0.270 (0.617) | Second-order pruners help most at 90%, where 4-step WoodFisher is best. SparseGPT is best at 70% but worst at 90%. |
| SNIP; GraSP; SynFlow, pruning at initialisation | Accuracy at 90% / 99% | SNIP 0.277 / 0.167. GraSP 0.113 / 0.160. SynFlow 0.513 / 0.173 | Random: 0.460 / 0.100. Random with SNIP's layer ratios: 0.543 / 0.143. Random with SynFlow's ratios: 0.533 / 0.100. Lottery ticket: 0.610 / 0.143 | Negative at 90%: random pruning with the same layer ratios beats every saliency score, the known sanity check. SynFlow is narrowly best at 99%. |
| LLM-Pruner; FASP; 2SSP, structured pruning of a toy LM | Validation loss after recovery, 25% / 50% removed | LLM-Pruner 2.473 / 2.717. FASP 2.427 / 2.538. 2SSP 2.392 / 2.566 | Dense: 2.501. Wanda (structured): 2.400 / 2.509. SliceGPT: 2.693 / 3.106. Dropping one block (50%): 2.566 | 2SSP leads at 25% and Wanda-style scores at 50%; LLM-Pruner and SliceGPT trail. At 25% recovery takes every method but SliceGPT below dense. |
| Network slimming (L1 on batch-norm scales), shapes CNN | Accuracy at 50% / 70% of channels removed (multiply-adds) | 0.870 (14,526) / 0.729 (8,706) | Filter-L1: 0.870 (23,136) / 0.802 (10,140). Slimmed shape from scratch: 0.958 / 0.797. Uniform width from scratch: 0.948 (13,896) / 0.901 (10,140). Dense: 0.969 (83,136) | Negative: slimming finds a cheaper shape than filter-L1, but a thin network trained from scratch beats pruning, as in "Rethinking the value of network pruning". |
| Expert pruning by search; expert merging, 8 → 4 experts | Validation MSE with 4 / 8 data regimes | Search-prune 0.934 / 2.072. Merge 1.287 / 2.239 | Full 8 experts: 0.743 / 1.127. Drop least-used: 1.227 / 2.072. Average all into one: 2.687 / 3.112. Dense of the active width: 1.127 / 1.831 | Search-based pruning is the best way to halve the experts (tied with dropping the least-used at 8 regimes). Merging is worse than dropping. |
| Unstructured; 2:4; 4×4 and 8×8 block sparsity, LM at 50% | Validation loss after recovery (one-shot) | Unstructured 2.629 (2.686). 2:4 2.699 (2.777). 4×4 2.912 (3.022). 8×8 2.901 (3.011) | Dense: 2.619 | Structure costs quality: 2:4 loses a little, blocks a lot. The report's run had dense at 2.501 and the same order (R-wave8). |

### CompactModels (W8 5.8: decoupled and relational distillation, 5.9: Deep Compression, 5.10: HAWQ mixed precision, 5.11: PACT and DoReFa, 5.12: CP, Tucker and tensor-train layers, 5.13: token merging, ToMe)
Sources: Run-w8, R-wave8. Tests: DistillationLossesTest, WeightSharingTest, MixedPrecisionTest, QuantizationAwareTest, TensorDecompositionTest, ConvDecompositionTest, TokenMergingTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DKD; RKD; FitNet, digits student | Accuracy | DKD 0.643. RKD 0.697. FitNet 0.693 | KD: 0.643. Student alone: 0.673. Teacher: 0.813 | Negative for logit distillation: KD and DKD fall below the student trained alone. Feature and relational losses help a little. |
| RKD, retrieval embedding | Recall@1 | 0.700 | FitNet: 0.407. Untrained: 0.363. Teacher: 0.903 | Relational distillation is what transfers an embedding space. |
| Deep Compression: 70% pruning, 3-bit weight sharing, Huffman coding | Accuracy (compression) | 0.657 (17.1x). Without Huffman 0.657 (12.9x) | fp32: 0.747. Weight sharing alone: 0.720 (10.4x). GPTQ, 3 bits: 0.723 (9.1x). Round-to-nearest, 3 bits: 0.660 (9.1x) | Huffman adds 1.3x for free, but the pruning step costs 0.09; sharing or GPTQ alone keep more accuracy. |
| HAWQ-V2 bit allocation (Hutchinson trace), 2.5-bit budget | Accuracy (average bits) | 0.507 (2.47, bits [3, 2, 2, 8]) | Uniform 2 bits: 0.303. Uniform 3 bits: 0.717. Random allocations: 0.427–0.507. Weight-variance allocation: 0.500. fp32: 0.760 | Null: Hessian-aware allocation beats uniform 2 bits but only ties the best random allocation, which picked the same bits. |
| PACT; DoReFa, weight and activation QAT | Accuracy at 2 / 3 / 4 bits | PACT 0.567 / 0.733 / 0.747. DoReFa 0.577 / 0.680 / 0.707 | LSQ: 0.690 / 0.713 / 0.723. Fixed range: 0.207 / 0.707 / 0.750. fp32: 0.747 | PACT's learned clip wins at 3 bits and is lossless at 4; LSQ is best at 2 bits. |
| Tensor-train dense layer, rank 8 (2,304 weights), 256×256 MLP layer | Accuracy before / after fine-tuning (forward time) | 0.243 / 0.763 (19.6 ms) | SVD rank 5 (2,560 weights): 0.377 / 0.683 (1.3 ms). Bottleneck of the same shape from scratch: 0.683. Dense: 0.863 (2.3 ms) | Tensor train recovers best after fine-tuning but runs about 9x slower than dense, because of the core permutes. |
| CP rank 11; Tucker-2 8×4, CNN conv2 (4,608 weights) | Accuracy before / after fine-tuning (weights) | CP 0.943 / 0.974 (594). Tucker-2 0.891 / 0.969 (608) | SVD rank 3: 0.854 / 0.917 (528). Dense: 0.969 | CP and Tucker match the dense layer with 13% of its weights after fine-tuning; SVD does not. |
| ToMe token merging, ViT on 8×8 shapes with 17 tokens | Accuracy at r = 2 / r = 5 per layer (r = 5 after fine-tuning) | 0.914 / 0.906 (0.984) | CLS-attention token pruning: 0.922 / 0.891 (0.992). Dense: 0.914 at 475k multiply-adds | Merging keeps accuracy at 42% fewer multiply-adds, but pruning does as well. |

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

### BeaconCompression (W8 4.10: Activation Beacon and gist tokens)
Sources: Run-w8, R-wave8. Test: BeaconCompressionTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Activation Beacon at 4x; 8x | Accuracy at 32 / 128 tokens (memory slots) | 4x 0.92 / 0.98 (8 / 32). 8x 0.80 / 0.94 (4 / 16) | Mean pooling 4x: 0.92 / 0.98. Mean pooling 8x: 0.73 / 0.90. Sliding window: 0.35 / 0.14 (8). Gist tokens: 0.57 / 0.45 (2). Full context: 1.00 / 1.00 | Learned beacons beat pooling only at 8x; at 4x they tie. The beacons are chunk-local (R-wave8). |

### PositionBiasSweep (W8 4.11: lost in the middle, found-in-the-middle and Ms-PoE)
Sources: Run-w8, R-wave8. Test: PositionBiasSweepTest. 60 trials. The toy model shows a recency curve, not a U.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Found-in-the-middle calibration; Ms-PoE | Mean accuracy; middle–ends gap | 0.49; −0.10. Ms-PoE 0.54; −0.15 | Uncalibrated: 0.46; −0.11. YaRN (factor 2): 0.59; +0.04 | Calibration lifts the earliest positions a little (0.18 → 0.26). YaRN does more: the best mean and the only positive gap. |

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

### ClassicalControl (W7 5.10: LQR, iLQR and DDP, 5.11: model-based diffusion, 5.12: domain randomisation and RMA, 5.13: PILCO, 5.14: dynamic movement primitives)
Sources: Run-w7, R-wave7. Tests: ClassicalPlannersTest, SimToRealTest, PilcoTest, PilcoBaselinesTest, MovementPrimitiveTest. Lower cost is better.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| LQR; iLQR-MPC; DDP-MPC, cart-pole balance | Closed-loop cost (success) | LQR 0.03 (1.00). iLQR 0.17 (1.00). DDP 0.17 (1.00) | MPPI: 0.19. MBD: 0.23. CEM: 0.32 (all 1.00) | The exact LQR is best on this nearly linear task. |
| iLQR-MPC; DDP-MPC, pendulum swing-up | Cost (success) | iLQR 225.6 (1.00). DDP 385.5 (0.00) | MPPI: 239.5. CEM: 240.0. MBD: 243.4 (all 1.00) | iLQR is best. DDP fails, because its second-order terms come from finite differences of the Jacobians (R-wave7). |
| iLQR-MPC; DDP-MPC, point mass around an obstacle | Cost | iLQR 8.97. DDP 9.17 | CEM: 9.82. MPPI: 10.87. MBD: 11.10 (all succeed) | The gradient-based planners edge out the samplers. |
| Model-based diffusion (MBD), 256 rollouts per decision | Cost on the three tasks above | 0.23 / 243.4 / 11.10 | MPPI: 0.19 / 239.5 / 10.87. CEM: 0.32 / 240.0 / 9.82 | Null: MBD roughly matches MPPI and CEM, with no escape-from-local-minima advantage at this scale. It needed a fix to its warm start to pass the pendulum (R-wave7). |
| Open-loop iLQR, mass-spring reach | Final distance to the goal | 0.0032 | BPTT controller (DifferentiablePhysicsControl): 0.024 | iLQR solves each goal separately, while BPTT is one controller for all goals (R-wave7). |
| Domain randomisation; RMA, cart-pole with unknown pole, motor gain and motor polarity | Held-out return; return over the last 50 steps after a mid-episode change | Randomised -220.1; -83.6. RMA -106.5; -54.5 | Nominal policy: -218.6; -80.5. RMA with the true latent: -23.4 | RMA halves the loss. Plain randomisation gives nothing, since one policy cannot fit both polarities. |
| PILCO (GP dynamics, moment matching, linear policy), cart-pole balance | Test cost after 180 / 270 / 360 / 450 / 540 real steps | 141.0 / 139.1 / 102.0 / 62.1 / 31.6 | Zero force: 111.1. True-model LQR: 1.64. At 540 steps: SAC 367.4, PPO 105.1. At 2700 steps: SAC 29.6, PPO 116.3 | PILCO reaches in 540 real steps what SAC needs 2700 for. PILCO and the 540-step baselines use 2 seeds (R-wave7). |
| Dynamic movement primitives, 2-D reach to new goals | Goal error; path error with 1 / 5 / 20 demos | 0.0043; 0.082. 0.0020; 0.070. 0.0009; 0.066 | MLP behaviour cloning: 0.446; 0.273. 0.342; 0.168. 0.341; 0.176 | DMPs reach new goals 100–400x more precisely. They are jerkier (about 103 against 14–46 for BC, ideal 35) because of their start-up jump. |
| Reward-weighted DMP refinement to a via-point | Via-point error | 0.541 → 0.0056 | – | Refinement hits the via-point and keeps the goal (goal error 0.0044). |

### GoalsImitationAndTeams: multi-agent navigation and communication (W7 5.6: MADDPG and COMA, 5.7: IPPO and MAPPO, 5.9: CommNet and TarMAC)
Sources: Run-w7, R-wave7. Tests: CooperativeMarlTest, ValueDecompositionNavigationTest, LearnedCommunicationTest. The navigation runs use 2 agents to fit the time budget. The report's value-based numbers (independent DQN -24.7, VDN -24.6, QMIX -54.8) come from a different run; the test's are given.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| MAPPO; MADDPG; IPPO; COMA, cooperative navigation, 12k joint steps | Team return (coverage) | MAPPO -27.7 (0.13). MADDPG -28.9 (0.05). IPPO -29.4 (0.03). COMA -33.0 (0.08) | Random: -37.6 (0.03) | All beat random; MAPPO's centralised critic gives the best return, by a small margin. |
| VDN; QMIX, same task, 6k joint steps | Team return (coverage) | VDN -25.7 (0.10). QMIX -37.8 (0.00) | Independent DQN: -23.4 (0.13). Random: -37.6 | Negative: value decomposition does not beat independent learners, and the existing QMIX diverges to random level. |
| CommNet; TarMAC, hidden-goal game, 5 agents, 8 goals | Reward / success | CommNet 1.00 / 1.00. TarMAC 1.00 / 1.00 | No communication: 0.29 / 0.00. Gumbel (discrete) channel: 0.43 / 0.00. Full-observation oracle: 1.00 / 1.00 | Learned continuous messages reach the oracle; the discrete channel does not. TarMAC puts 0.38 of its attention on the informed agent, against 0.2 if uniform. |

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

### ActiveLearning (W8 6.13: core-set and BADGE)
Sources: Run-w8, R-wave8. Test: ActiveLearningTest. Digits pool of 2,000, starting from 20 labels.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Core-set; BADGE | Learning-curve area: batch 10, budget 120 / batch 50, budget 270 / 30% near-duplicates, batch 10 | Core-set 0.453 / 0.545 / 0.426. BADGE 0.424 / 0.541 / 0.423 | Random: 0.393 / 0.490 / 0.383. Entropy: 0.392 / 0.487 / 0.413. BALD: 0.376 / 0.466 / 0.374. Leverage: 0.373 / 0.432 / 0.354 | Diversity-aware acquisition wins in every pool. Uncertainty alone does not beat random on clean pools, and entropy does not fall below random on duplicates. Leverage is the worst. |

### DataValuation (W8 6.14: KNN-Shapley, Data-OOB and leverage scores, 6.15: self-supervised prototype pruning across dataset sizes)
Sources: Run-w8, R-wave8. Tests: DataValuationTest, PrototypePruningTest. ShapleyPruning's data with 20% of labels flipped.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| KNN-Shapley (pixels; SSL embeddings); Data-OOB | Share of flipped labels in the lowest-valued 20%; accuracy after removing them (time) | Pixels 0.73; 0.745 (32 ms). Embeddings 0.81; 0.783 (896 ms). Data-OOB 0.76; 0.757 (155 ms) | In-Run Shapley: 0.35; 0.685 (217 ms). TracIn: 0.69 (228 ms). Influence functions: 0.69 (1,338 ms). EL2N: 0.56. Random: 0.18; 0.660 | Cheap valuations find flipped labels better than the wave 5 and 6 attribution methods, in a fraction of the time. |
| Leverage scores | Flipped share; accuracy after removal | 0.20; 0.688 | Random: 0.18; 0.660 | Label-blind, as expected. |
| Leverage selection of a diverse subset, 3 seeds | Accuracy with 30 / 60 labels | Pixels 0.229 / 0.311. SSL embeddings 0.243 / 0.324 | Random: 0.221 / 0.313. Top KNN-Shapley: 0.177 / 0.263 | A small gain, only on SSL embeddings (0.270 / 0.335 in the report's run, R-wave8). |
| Prototype pruning (SSL encoder + k-means), keep easy / hard examples, 250 steps | Accuracy at n = 8,000 keeping 30% (n = 500 keeping 30%) | Easy 0.566 (0.386). Hard 0.612 (0.364). EL2N-hard 0.538 (0.378) | Random: 0.716 (0.464) | Negative: random pruning wins at every size and keep ratio, so the easy/hard crossover does not reproduce. |

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

### BlackBoxOptimizers (W7 5.15: differential evolution and particle swarm)
Sources: Run-w7, R-wave7. Tests: BlackBoxOptimizersTest, NeuroevolutionTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DE; PSO; PSO-ring, 10-D, 20k evaluations, 3 seeds | Best value: Rastrigin / Rosenbrock / Ackley / Schwefel | DE 26.1 / 1.65 / 1.9e-9 / 1.95e-6. PSO 8.29 / 2.22 / 1.7e-12 / 1,300. PSO-ring 6.86 / 5.56 / 2.4e-6 / 1,070 | CMA-ES: 29.9 / 3.2e-27 / 8.2e-13 / 1,520. OpenAI-ES: 73.7 / 12.2 / 3.21 / 2,150. Random: 62.2 / 2,240 / 16.6 / 1,560 | Each wins somewhere: PSO on Rastrigin, CMA-ES on Rosenbrock, and only DE solves Schwefel (all 3 seeds). All four population methods solve Ackley. |
| Same, neuroevolution | Held-out return: cart-pole linear policy, 300 evaluations / pendulum 3-8-1 MLP, 800 evaluations | DE 492.5 / -1,023. PSO 433.1 / -642.6. PSO-ring 432.3 / -636.2 | CMA-ES: 327.7 / -847.9. OpenAI-ES: 173.0 / -1,270. Random search: 356.6 / -1,179 | PSO beats CMA-ES on both tasks, and DE wins cart-pole. OpenAI-ES falls below random search at these budgets. |

### MultiObjectiveSearch (W7 5.16: NSGA-II)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| NSGA-II, ZDT1 / ZDT2 / ZDT3, 10k evaluations | Hypervolume of the returned set (spread) | 0.794 (1.00) / 0.412 (0.97) / 1.111 (0.84) | Weighted-sum CMA-ES: 0.698 / 0.210 / 1.046. MAP-Elites: 0.590 / 0.125 / 0.955. Random: 0 | NSGA-II returns the best front on all three. Scored on everything it evaluated, weighted-sum CMA-ES is higher (0.875 / 0.515 / 1.312). The paper's distribution indices (20, 20) gave a hypervolume of about 0 at this budget; 2 and 5 are used (R-wave7). |

### MergeSearch (W7 5.17: evolutionary model merging)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Evolved merge coefficients, per tensor / per model, 150 fitness evaluations, 3 seeds | Accuracy on mixed addition + multiplication | 0.780 / 0.727 | Task arithmetic (1.0 / 0.5): 0.627 / 0.593. TIES: 0.600. SLERP: 0.593. DARE: 0.547. Addition specialist: 0.727. Base: 0.253. Multi-task training: 1.000 | Evolving one coefficient per tensor beats every hand-set merge and the best specialist, but stays far below multi-task training. |

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

### SpikingClassifier: ANN-to-SNN conversion and direct training (W8 6.16: QCFS conversion, 6.17: TET and adaptive surrogate gradients)
Sources: Run-w8, R-wave8. Tests: SpikingConversionTest, SpikingTrainingTest, SpikingSurrogateTest. Digits MLPs.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| QCFS conversion, 3 hidden layers | Accuracy at T = 1 / 2 / 4 / 8 (spikes per example at T = 8) | 0.728 / 0.890 / 0.948 / 0.964 (742) | Plain conversion: 0.100 / 0.100 / 0.776 / 0.948 (336), 0.968 at T = 16. ReLU ANN: 0.964. QCFS ANN: 0.970 | QCFS matches the ANN at T = 8, as predicted. Plain conversion needs only T ≈ 16, not 64 or more; with 1 hidden layer it already scores 0.942 at T = 4. |
| TET; A²SG grid surrogate, direct training at T = 4, 3 hidden layers | Accuracy | TET 0.822. Grid 0.864. TET + grid 0.886 | Mean-output loss with a fast-sigmoid surrogate: 0.810 | Both help in the deeper network, the adaptive surrogate more. |
| TET; grid; adaptive; scheduled surrogates, 1 hidden layer | Accuracy at T = 4 | TET 0.880. Grid 0.856. Adaptive 0.854. Scheduled 0.850 | Fast sigmoid: 0.864 (0.876 at T = 8, same as TET). Rectangular: 0.864 | Little difference in a shallow network, where direct LIF training (0.890) also trails conversion (0.942). |
| Fast-sigmoid surrogate, 2 hidden layers, T = 4 | Cosine with finite-difference gradients | 0.330 | Rectangular, half-width 0.1 / 0.25 / 0.5 / 1.0: 0.127 / 0.101 / 0.269 / 0.300 | Narrow windows align worst with the true gradient. |

### ProbabilisticBits: simulated bifurcation (W8 6.18: a quantum-inspired Ising solver)
Sources: Run-w8, R-wave8. Test: IsingSolversTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Ballistic (bSB); adiabatic (aSB); discrete (dSB) simulated bifurcation | Best cut, dense Erdős–Rényi MaxCut, n = 256 (time) | bSB 9,012 (17 ms). aSB 9,002. dSB 8,526 | Simulated annealing: 9,005 (19 ms). p-bit annealing: 8,997. Spectral cut: 7,553 | bSB is narrowly best on dense MaxCut; dSB is weak when the couplings have a large mean. |
| Same | Energy per spin, Sherrington–Kirkpatrick, n = 256 | dSB −0.744. bSB −0.743. aSB −0.722 | Simulated annealing: −0.740. p-bit: −0.721 | dSB is best on SK. On a sparse graph (n = 512) and a planted cut every heuristic is within 0.3% (dSB 1,873 vs annealing 1,871), and the spectral cut finds half the planted cut (1,317 vs 2,616). |

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

### GraphStructureShootout (W7 1.1: GraphGPS with LapPE and RWSE, 1.7: virtual node and FoSR rewiring, 1.8: random node features and ESAN, 1.9: PNA, 1.10: DiffPool, SAGPool and top-k pooling, 1.17: NeuralWalker)
Sources: Run-w7, R-wave7. Wave 7 graph blocks are in the new `experimental.graph` package. Tests: StructureShootoutTest, OverSquashingTest, ExpressivityTest, AggregationAndPoolingTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Random-walk (RWSE) and Laplacian (LapPE) positional encodings; GraphGPS | Triangle-count MAE | GIN+RWSE 0.597. GPS+RWSE 1.314. GIN+LapPE 2.169. GPS 2.922 | GIN: 1.814. GCN: 2.973. Predict-mean: 2.967 | RWSE lets GIN count triangles 3x better. LapPE and plain GPS do not help. |
| Same, trees vs cycles | Accuracy | GPS+RWSE 1.000. GPS+LapPE 1.000 | GIN: 0.467 | The encodings make the structure separable. |
| Same, SBM node classification | Accuracy | GIN+LapPE 0.854. GPS+LapPE 0.573 | GCN: 0.878 | No gain on a homophilous node task. |
| GPS; NeuralWalker; virtual node; FoSR rewiring, neighbours-match at depth 2 / 3 | Accuracy (chance 0.25 / 0.125) | GPS 1.000 / 0.438. NeuralWalker 0.531 / 0.188. Virtual node 0.594 / 0.156. FoSR 0.438 / 0.219 | GIN: 0.500 / 0.125 | Global attention solves depth 2. Rewiring and walks give little, and everything falls toward chance at depth 3. |
| GIN+RWSE; ESAN (subgraphs); random node features (RNF), CSL graphs | Accuracy (chance 0.1) | GIN+RWSE 0.400. ESAN 0.300. GIN+RNF 0.050 | GIN: 0.100 | RWSE and subgraphs get past 1-WL; RNF does not. Noisy: a 100-step run gave RWSE 0.6 and ESAN 0.5 (R-wave7). |
| PNA | Trees vs cycles accuracy; max-neighbour log10 MSE; diameter MAE | 1.000; -2.549; 0.441 | GIN / GCN / SAGE: 0.467 / 0.867 / 0.467; -1.232 / -1.800 / -1.952; 0.605 / 0.473 / 0.456. Predict-mean diameter: 1.033 | Several aggregators with degree scalers win all three tasks. |
| Top-k pooling; SAGPool; DiffPool | Accuracy: trees vs cycles; community count | TopK 0.933; 0.533. SAGPool 0.933; 0.733. DiffPool 0.400; 0.533 | GIN+sum and GCN+mean: 0.467; 0.467 | Selection pooling wins clearly. DiffPool falls below the flat baselines on trees vs cycles. |

### GraphLearningShootout (W7 1.2: SGC and APPNP, 1.3: GCNII, PairNorm and DropEdge, 1.4: H2GCN and GPR-GNN, 1.5: neural sheaf diffusion, 1.6: GRAND, 1.12: DeepWalk and node2vec, 1.18: label propagation and Correct & Smooth)
Sources: Run-w7, R-wave7. Tests: NodeShootoutTest, HeterophilyTest, DiffusionAndSmoothingTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| SGC; APPNP, SBM at 5% labels | Accuracy (training time) | SGC 0.807 (3 ms). APPNP 0.721 | GCN: 0.851 (65 ms). GAT: 0.817. MLP: 0.391 | SGC comes close to GCN at about 1/20 of the time. APPNP stays below GCN, even with dropout 0.5 (R-wave7). |
| GCNII; PairNorm; residual; DropEdge, depth 2 → 32 | Accuracy (mean pairwise distance of node features) | GCNII 0.795 → 0.940 (0.591 at depth 32). PairNorm → 0.928. Residual → 0.687. DropEdge → 0.313 | GCN: 0.976 → 0.313 (0.756 → 0.073) | GCNII and PairNorm stop over-smoothing; DropEdge does not help. |
| H2GCN; sheaf diffusion (diagonal / orthogonal); GPR-GNN, homophily 0.1 | Accuracy | H2GCN 0.522. Sheaf 0.511 / 0.456. GPR-GNN 0.400 | SAGE: 0.467. MLP: 0.444. GCN: 0.333 | H2GCN and diagonal sheaf diffusion beat every baseline under heterophily. GPR-GNN falls below the MLP. |
| Same, homophily 0.9 | Accuracy | H2GCN 0.922. GPR-GNN 0.967. Sheaf 0.722 / 0.756 | GCN: 0.967. MLP: 0.444 | Sheaf diffusion loses badly on a homophilous graph. |
| GRAND (graph neural diffusion ODE) | Accuracy at T = 2 / 8 (function evaluations) | 0.519 (16) / 0.750 (64) | GCN: 0.963. GAT: 0.889. GCNII with 8 layers: 1.000 | Negative: it improves with integration time but stays far below GCN. |
| DeepWalk; node2vec (q = 0.5 / 2) | Link-prediction AUC | DeepWalk 0.723. node2vec 0.710 / 0.722 | GAE: 0.754. Common neighbours: 0.671. Spectral: 0.605 | Walk embeddings beat the heuristics but not GAE. Node accuracy: DeepWalk 0.940 against 0.655 for spectral embeddings. |
| Correct & Smooth on an MLP, feature noise 0.5 | Accuracy (training time) | 0.821 (24 ms) | MLP: 0.759. GCN: 1.000 (68 ms). APPNP: 1.000. GAT: 779 ms. Label propagation: 0.973 | C&S lifts the MLP cheaply, but does not reach GCN. |
| Same, feature noise 2.0 | Accuracy | 0.321 | MLP: 0.348. GCN: 0.964. Label propagation: 0.973 | Negative: with the base predictor at chance there is nothing to correct. |

### GraphPretraining (W7 1.11: DGI, GraphCL and GraphMAE)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DGI; GraphCL; GraphMAE, linear probe at 1% / 5% labels | Accuracy | DGI 0.367 / 0.449. GraphCL 0.415 / 0.453. GraphMAE 0.412 / 0.442 | Random encoder: 0.357 / 0.407. GAE: 0.415 / 0.533. Supervised GCN: 0.432 / 0.530 | All three beat a random encoder only modestly, and a plain graph autoencoder beats them all at 5%. |

### KnowledgeGraphCompletion (W7 1.13: TransE, DistMult, ComplEx and RotatE)
Sources: Run-w7, R-wave7. The data is a family-tree knowledge graph (`FamilyTreeGraph`).

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| ComplEx; TransE; RotatE; DistMult | Filtered MRR | ComplEx 0.892. TransE 0.761. RotatE 0.760. DistMult 0.752 | Two-tower: 0.233. Frequency: 0.036 | All are far above the baselines; ComplEx is best. |
| Same, by relation type | Filtered MRR | Sibling (symmetric): DistMult 1.000. Parent (antisymmetric): ComplEx 0.956. Grandparent (composition): RotatE 0.864 | Sibling: TransE 0.438. Parent: DistMult 0.572 | Each model wins on the relation pattern it is built for. |

### GraphReasoning (W7 1.14: neural algorithmic reasoning, CLRS-lite, 1.15: NeuroSAT)
Sources: Run-w7, R-wave7. Tests: GraphReasoningTest, NeuroSatTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Processor with max aggregation, with / without hints; sum + hints. BFS, trained on 8–16 nodes, tested on 64 | Accuracy | Max + hints 1.000. Max 1.000. Sum + hints 0.945 | GIN: 0.504. GAT: 1.000. GPS: 0.758 | Max aggregation extrapolates to 4x the size; GAT matches it. |
| Same, Bellman-Ford pointers at 64 nodes | Accuracy | Max + hints 0.762. Max 0.777. Sum + hints 0.547 | GIN: 0.301 | Max aggregation helps; hints do not. |
| Same, Prim at 32 nodes | Accuracy | Max + hints 0.328. Max 0.461. Sum + hints 0.531 | – | Negative: max aggregation does not help here. |
| NeuroSAT, SR(6) | Sat / unsat accuracy; share of satisfiable formulas solved by decoded assignments | 0.675 at T = 8 and 24; 0.650 at T = 8, 0.900 at T = 24 | GIN: 0.600. Two random assignments: 0.100. WalkSAT: 1.000 | Decoded assignments solve most formulas. Classification beats GIN only modestly; the planned +0.1 margin did not hold at 200 steps, so the test was loosened (R-wave7). |
| NeuroSAT, SR(12) | Same | 0.700; 0.650 | GIN: 0.550. Random assignments: 0.250 | Holds at twice the size. |

### GraphGeneration (W7 1.16: DiGress, discrete graph diffusion)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DiGress-lite, SBM graphs | Valid share; spectral MMD; degree MMD | 0.583; 0.066; 0.117 | GraphRNN-lite: 0.792; 0.246. VGAE: 0.250; 0.033. Erdős–Rényi: 0.208; 0.028; 0.014 | Negative: it beats ER and VGAE on validity but loses to GraphRNN, and its MMDs are worse than ER's. |
| Same, grids | Valid share (connected, degree ≤ 4); degree MMD | 0.125; 0.072 | ER: 0.125; 0.066. GraphRNN-lite: 0.125. VGAE: 0.000 | No model generates valid grids. |

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

## kernels

New theme in wave 8 (package `experiments.kernels`): autotuning, fast algorithms, decomposition search, graph rewriting and sparse kernels on the CPU. The blocks are in `experimental.kernels`. Timings come from Gradle runs with other test classes running in parallel, so they are noisy: the core 128³ matmul measured anywhere from 0.3 to 24 GFLOP/s across runs (Run-w8, R-wave8). Treat them as indicative.

### KernelTuning (W8 5.14: learned cost models for autotuning, AutoTVM and Ansor, 5.15: Winograd convolution and Strassen matmul, 5.16: AlphaTensor-style decomposition search, 5.17: equality saturation, 5.18: block-sparse and n:m kernels)
Sources: Run-w8, R-wave8. Tests: AutoTuningTest, FastAlgorithmsTest, DecompositionDiscoveryTest, RectangularDiscoveryTest, GraphRewritingTest, SparseKernelsTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Cost-model-guided autotuning (gradient-boosted trees over 7,776 Java tilings), 128³ matmul, 32 measurements | Best GFLOP/s | 4.41 (95% of its best within 24 measurements) | Random: 3.14. Grid: 3.78. Evolution: 3.87. Core matmul: 24.1 | Negative: the tuned Java kernels stay far below the core matmul. The cost model finds the best tiling here (the report's clean run had evolution ahead, 3.40 vs 3.28, R-wave8), but its Spearman correlation is only 0.39. The simulated-cache optimum from EvolvedKernels measures 2.25, so the simulated optimum is not the real one. |
| Winograd F(2,3); F(4,3) | Speed-up over core convolution (relative error) | 0.30x (5e-7). 0.35x (4e-6) | Core convolution | Exact, but slower in this run; the report's clean run measured 0.6–0.9x and 1.1–1.6x (R-wave8). |
| Winograd in low precision, shapes CNN | Accuracy at int8 / int6 | F(2,3) 0.953 / 0.953. F(4,3) 0.958 / 0.818 | Direct convolution: 0.948 (fp32 and int8) | F(4,3)'s larger transform error hurts at 6 bits, as predicted. |
| Strassen (cut-over 128) | Speed-up at n = 256 / 512 | 0.27x / 0.36x | Core matmul | Negative: slower at these sizes. |
| ALS with rounding, matmul decomposition search | Exact algorithm found: 2×2×2 at rank 7 / 2×2×3 at rank 11 | Both found (21 and 208 restarts, error about 1e-15) | MCTS (100 / 60 episodes), CMA-ES, random search (20,000 samples): neither found | Negative for the AlphaTensor-style MCTS: the simplified game's residual-reduction prior favours naive terms, and there is no network training (R-wave8). |
| E-graph equality saturation, Q/K/V projections with scale-then-matmul and a bias chain | Kernels (FLOPs) | 4 (1.585M) | Original: 8 (1.593M). Greedy single pass: 5. Greedy to a fixpoint: 4 | Beats one greedy pass but ties greedy run to a fixpoint. All methods fold conv + batch norm. Error about 5e-8. |
| CSR; 8×8 block-sparse (BSR); 2:4, 512×512 weight, batch 64 | Speed-up over core dense matmul at 50 / 75 / 90 / 95% sparsity | CSR 0.19 / 0.49 / 0.53 / 1.76x. BSR 0.14 / 0.33 / 0.39 / 1.21x. 2:4 (50%) 0.09x | Core dense matmul. Core COO: 0.04 / 0.07 / 0.15 / 0.36x | Sparse kernels win only at 95% here; the report's cleaner run had CSR ahead from 90% (1.26x, 3.2x at 95%) (R-wave8). Block sparsity does not win from about 70% as hypothesised, and 2:4 is slower than dense. |

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

### TokenizerShootout (W8 4.12: Length-MAX tokenizer, 4.13: PathPiece and optimal segmentation, 4.14: StochasTok)
Sources: Run-w8, R-wave8. Test: TokenizerShootoutTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Length-MAX; Unigram; SuperBPE, morphology corpus, 64 learned tokens | Tokens per char; bits per byte; morpheme-boundary F1 | Length-MAX 0.330; 1.475; 0.63. Unigram 0.321; 1.458; 0.69. SuperBPE 0.393; 1.537; 0.61 | BPE: 0.393; 1.537; 0.61 | Length-MAX uses 16% fewer tokens than BPE, but Unigram is better on all three measures. SuperBPE equals BPE at this vocabulary size. On the toy language every tokenizer is within 0.19–0.21 tokens per char. |
| Least-tokens and max-likelihood segmentation of a BPE vocabulary (PathPiece-style) | Same | 0.389; 1.527–1.534; 0.65 | BPE merge order: 0.393; 1.537; 0.61. Greedy longest prefix: 0.389; 1.529; 0.65 | Optimal segmentation saves 1% of tokens, no more than greedy longest prefix. |
| StochasTok; BPE-dropout | Accuracy: letter count / digit sum | StochasTok 0.975 / 0.207. BPE-dropout 0.975 / 0.209 | BPE: 0.962 / 0.164. Characters: 0.994 / 0.617 | Stochastic splitting helps a little; characters stay far better on digit sums. |

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

### LearnerComparison: incremental predictive coding (W7 6.17: iPC)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Incremental predictive coding (iPC), spirals, 30 epochs | Test accuracy (time), looped Brain / feed-forward Brain | 0.653 (1879 ms) / 0.880 (1091 ms) | Predictive coding: 0.540 (863 ms) / 0.887 (607 ms). Backprop MLP: 0.967 | iPC repairs part of the loop penalty (0.54 → 0.65) and ties PC on a feed-forward graph. Both stay below backprop. |

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

## multiagent

New theme in wave 7 (package `experiments.multiagent`): game theory on small games. The game blocks (`ExtensiveFormGame`, `KuhnPoker`, `LeducPoker`, `CounterfactualRegret` and others) are in `experimental.rl`. Exploitability is in milli-big-blinds per hand (mbb/h; lower is better). Multi-agent RL on cooperative navigation is under control (GoalsImitationAndTeams), and min-max optimisers on a GAN under techniques (GanExperiment).

### ImperfectInformationGames (W7 5.1: CFR, CFR+ and MCCFR, 5.2: neural fictitious self-play, 5.3: PSRO and the double oracle, 5.4: MMD, R-NaD and Nash policy gradient)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| CFR; CFR+; MCCFR | Exploitability, Kuhn / Leduc | CFR 10.7 / 81.8. CFR+ 0.2 / 18.3. MCCFR 15.2 / 379.9 (350.0 over 3 seeds) | Uniform: 458 / 2,374 | CFR+ converges fastest. MCCFR touches 132k nodes on Leduc against 1.96M for full CFR. |
| Neural fictitious self-play, Kuhn, 3000 episodes, 3 seeds | Exploitability of the average policy | 216 | DQN self-play: 306. Uniform: 458 | Averaging best responses beats plain self-play, though it is far from CFR+. |
| PSRO with a Nash meta-solver: double oracle / policy-gradient oracle | Exploitability, Kuhn after 8 iterations / Leduc after 50 | Double oracle 0.0 / 292.9. Policy-gradient oracle 0.6 / 139.0 | League (uniform meta-solver, fictitious play): 92.6 / 398.1. Naive self-play: 1,166.7 / 5,200 | The Nash meta-solver is what makes population training converge. On rock-paper-scissors with skill (x1000): PSRO 0, league 244, self-play 1000. |
| MMD; R-NaD; Nash policy gradient, 1000 iterations | Last-iterate exploitability, Kuhn / Leduc | MMD 18.5 / 198.6. R-NaD 5.3 / 82.1. NashPG 3.9 / 144.7 | Multiplicative weights: 78.6 / 1,346. CFR+ current strategy: 26.9 / 11.2. CFR+ average: 0.1 / 8.8 | Regularisation makes the last iterate converge, and it beats CFR+'s current strategy on Kuhn; on Leduc CFR+ wins. |

### MatrixGameDynamics (W7 5.4: last-iterate learners on matrix games, 5.5: optimistic and extragradient min-max methods, 5.8: LOLA and M-FOS)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Consensus optimisation; optimistic GDA; extragradient; optimistic Adam, bilinear game, 500 steps | Distance from the equilibrium (start 2.83) | Consensus 0.000. Optimistic GDA 0.226. Extragradient 0.235. Optimistic Adam 1.939 | Simultaneous GDA: 34.0. Alternating GDA: 2.837. Simultaneous Adam: 29.8. Alternating Adam: 28.6 | The look-ahead methods converge where plain GDA spirals out. |
| Same, Dirac-GAN | Distance from the equilibrium (start 1.41) | Consensus 0.000. Optimistic GDA 0.779. Extragradient 0.780. Optimistic Adam 0.935 | Simultaneous GDA: 2.254. Alternating GDA: 1.414. Simultaneous / alternating Adam: 3.78 / 3.67 | Same. On a real WGAN-GP the gain disappears (GanExperiment, under techniques). |
| MMD; R-NaD; Nash policy gradient | Last-iterate exploitability, matching pennies / biased RPS | MMD 0.000 / 0.150. R-NaD 0.000 / 0.002. NashPG 0.000 / 0.002 | Multiplicative weights: 0.801 / 0.747 | The regularised learners converge in the last iterate; plain multiplicative weights cycles. |
| LOLA; M-FOS, iterated prisoner's dilemma, 3 seeds | Reward per step for each player (mutual cooperation -1, mutual defection -2) | LOLA vs LOLA -1.06 / -1.14. LOLA vs naive -1.30 / -1.97. M-FOS vs naive -1.25 / -1.79 | Naive vs naive: -2.00 / -2.00. Untrained meta-policy vs naive: -1.86 / -1.32 | Opponent shaping finds cooperation (LOLA cooperates after mutual cooperation with probability 0.96); naive learners defect. |

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

## numerics

New theme in wave 7 (package `experiments.numerics`): second-order and natural-gradient training, implicit layers, and memory and communication compression for training. Blocks are in `experimental.train` and `experimental.nn`. Shared pieces are `CurvatureProducts` (Hessian-, Jacobian- and Gauss-Newton-vector products, preconditioned CG), `KroneckerCapture` and the simulated collective `GradientCompressor`. Wall times are reported, not equalised, except where noted.

### SecondOrderShootout (W7 3.1: K-FAC and EKFAC, 3.2: AdaFisher, 3.3: online L-BFGS, 3.5: AdaHessian)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| K-FAC; EKFAC; AdaFisher; AdaHessian; online L-BFGS, spirals MLP, 150 steps, 3 seeds | Loss / accuracy / steps to loss 0.3 / ms per step | K-FAC 0.054 / 0.98 / 23 / 0.7. EKFAC 0.187 / 0.95 / 30 / 1.3. AdaFisher 0.065 / 0.98 / 40 / 0.5. AdaHessian 0.057 / 0.99 / 43 / 1.1. Online L-BFGS 0.051 / 0.99 / 70 / 0.7 | AdamW: 0.082 / 0.98 / 90 / 0.3. Shampoo: 0.054 / 0.97 / 30 / 0.6. Sophia-H: 0.057 / 0.98 / 87 / 0.4. PSGD Kron: 0.107 / 0.96 / 97 / 0.8. SOAP: 0.254 / 0.93 / 140 / 0.9 | K-FAC reaches the target in a quarter of AdamW's steps at about twice the step cost. Every curvature method beats AdamW to the target except PSGD and SOAP. Online L-BFGS diverged on 0 of 3 seeds (R-wave7). |

### FullBatchSecondOrder (W7 3.3: full-batch L-BFGS, 3.4: Hessian-free, 3.6: Levenberg–Marquardt)
Sources: Run-w7, R-wave7. The report's neural-field numbers (L-BFGS 3.4e-4, Hessian-free 2.6e-5 in 60 iterations) come from a longer run; the test's are given.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Hessian-free; K-FAC; L-BFGS, digits autoencoder 64-32-8-32-64 | MSE (iterations, time); time to 0.12 | HF 0.0963 (25, 1979 ms); 1566 ms. K-FAC 0.110 (250, 966 ms); 789 ms. L-BFGS 0.128 (125, 457 ms); never | Adam: 0.137 (250, 334 ms); never | Only HF and K-FAC reach 0.12; HF is best. |
| L-BFGS (strong-Wolfe line search); Hessian-free, 20x20 neural field (385 weights) | MSE (time); time to 3e-3 | L-BFGS 8.3e-4 (208 ms); 114 ms. HF 7.9e-5 (1032 ms); 422 ms | Adam: 7.4e-3 (168 ms); never. SGD momentum: 1.6e-2 (161 ms) | At about Adam's time, L-BFGS is 9x better; HF is about 90x better at 6x the time. |
| Levenberg–Marquardt, Mackey-Glass one-step, 4-8-1 network | MSE (iterations, time) | 6.95e-7 (40, 129 ms) | Adam: 3.56e-5 (2,000, 102 ms). L-BFGS: 2.74e-5 (40, 4 ms) | About 40–50x lower error than either. |

### NaturalGradientDiagnostics (W7 3.7: true vs empirical Fisher, exact natural gradient)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Exact natural gradient (Fisher); Newton with absolute Hessian eigenvalues; empirical Fisher, 2-8-3 network (51 parameters), spirals | Loss after 15 steps | Newton 0.232. Natural gradient 0.246. Empirical Fisher 0.296 | Gradient descent: 0.727. Adam: 0.693 | Exact curvature cuts the loss about 3x in 15 steps. |
| Same, logistic regression (9 parameters) | Loss after 10 steps | Newton and natural gradient 0.697. Empirical Fisher 0.696 | Gradient descent: 0.881. Adam: 0.910 | On a convex model the empirical Fisher is as good. |
| Cheap curvature approximations, 2-8-3 network | Cosine to the exact natural-gradient step | Empirical Fisher 0.69. K-FAC 0.57. Shampoo 0.40 | Gradient: 0.32. Adam (square root of the empirical Fisher diagonal): 0.30. AdaFisher diagonal: 0.23 | Kronecker factors point closest to the true step; the diagonal methods are no closer than the raw gradient. |

### ImplicitDepth (W7 3.8: deep equilibrium models with Anderson acceleration)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DEQ (Anderson, m = 5), with / without Jacobian regularisation, spirals, hidden 16, 150 steps | Test accuracy; floats stored per example; time | With 0.907; 32; 2362 ms. Without 0.920; 32; 3327 ms | Tied 8-step unroll: 0.987; 144; 362 ms. Untied 8-block stack: 0.987; 144; 450 ms | Negative: the DEQ stores 4.5x less but trails the unroll by about 0.07 and is 6–9x slower. |
| Anderson acceleration on the trained layer | Iterations of the final solve, Anderson vs plain fixed point | With Jacobian regularisation: 21 vs 32. Without: 150 vs 150 (the cap) | – | Without the regulariser the trained layer stops converging. The implicit gradient matches finite differences (-0.54423 vs -0.54419). |

### GradientCompressionShootout (W7 3.9: PowerSGD, 3.10: top-k with error feedback, DGC and EF21, 3.11: QSGD and TernGrad)
Sources: Run-w7, R-wave7. Spirals, 4 simulated workers, 150 steps. Adam is the outer optimizer unless stated (R-wave7).

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| PowerSGD rank 1 / 2 | Loss / accuracy / kB sent | 0.126 / 0.98 / 960. 0.062 / 0.99 / 1,579 | fp32 all-reduce: 0.057 / 0.99 / 6,004. Rank 1 without error feedback: 0.171 / 0.96. DeMo: 0.564 / 0.69 / 633 | Rank 2 matches full all-reduce at about a quarter of the traffic. |
| QSGD s = 1 / 4 / 16; TernGrad | Loss / accuracy / kB sent | QSGD 0.185 / 0.96 / 581. 0.084 / 0.98 / 810. 0.059 / 0.99 / 1,302. TernGrad 0.083 / 0.98 / 808 | fp32 all-reduce: 0.057 / 0.99 / 6,004 | Quantisation keeps accuracy at 5–10x compression. Measured relative variance: 7.45 / 1.10 / 0.080 for QSGD, 0.85 for TernGrad. |
| Top-1% with / without error feedback; EF21; DGC | Loss / accuracy / kB sent | Without EF 0.392 / 0.85 / 307. With EF 0.648 / 0.57 / 307. EF21 0.558 / 0.73 / 307. DGC 0.228 / 0.95 / 804 | fp32 all-reduce: 0.057 / 0.99 | With Adam outside, error feedback hurts. Only DGC comes near fp32. Top-10% with EF: 0.88 at 2,476 kB; top-0.1%: 0.57 at 134 kB. |
| Same, plain SGD as the outer optimizer | Loss / accuracy | Without EF 0.693 / 0.57. With EF 0.531 / 0.67. EF21 0.490 / 0.73. DGC 0.560 / 0.83 | fp32 all-reduce: 0.387 / 0.82 | With SGD, error feedback helps, as the theory expects. |

### MemoryEfficientTraining (W7 3.12: ActNN compressed activations, 3.13: reversible residual blocks)
Sources: Run-w7, R-wave7. The ActNN byte counts are what the backward pass would keep; the float tensors are still referenced by the graph. The reversible and `Recompute` counts are architecture counts, not measured heap (R-wave7).

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| ActNN 8-bit / 4-bit / 2-bit activations, digits MLP | Accuracy; bytes kept for backward | 0.775; 23,808. 0.775; 13,568. 0.760; 8,448 | fp32: 0.760; 147,456. bf16 storage: 0.775; 43,008. Core `Recompute`: 0.760; 81,920 | 2-bit activations cost nothing at 17x less memory. |
| Reversible blocks, 12 blocks on digits | Accuracy; floats kept per example; reconstruction drift | 0.630; 32; 5.8e-6 (fp32) | Ordinary graph: 0.630; 800. Recompute: 0.630; 64 | The same result with 25x less stored state (the task is undertrained at this budget). In bf16 the drift reaches 0.30 and training fails. |

### OptimizerMemory (W7 3.14: 4-bit and COAT FP8 optimizer states, 3.15: ReLoRA)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| 4-bit states (rank-1 normalisation, no zero point); COAT FP8 states, digits MLP, 3 seeds | Accuracy; state bytes per parameter; diverged seeds | 4-bit 0.718; 1.2; 0. COAT 0.718; 2.1; 0 | fp32 Adam: 0.725; 8.0; 0. Core 8-bit Adam: 0.728; 2.5; 0. Adam-mini: 0.728; 4.1; 0. Naive 4-bit with a zero point: 1.1 bytes, 3 of 3 diverged | 4-bit states lose under 0.01 accuracy at 1/7 of the memory. The zero point is what breaks naive 4-bit. |
| ReLoRA, rank 4, digits 64-64-64-10 from scratch | Accuracy; optimizer bytes; effective rank of the update | 0.660; 14,416; 16.3 | Full-rank AdamW: 0.660; 71,760; 36.9. LoRA: 0.550; 14,416; 3.87. GaLore: 0.545; 9,456; 14.0 | Restarts give LoRA's memory with full-rank accuracy. |

### UnitScalingTransfer (W7 3.16: unit scaling and u-μP)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| u-μP, digits MLP, learning-rate sweep at width 32 and 128 | Optimal log2 learning rate | -2 at both widths | Standard parameterisation: -4 → -6 | The optimum transfers across width. The best losses are higher for u-μP (0.094 / 0.031 against 0.035 / 0.008). |
| Same, FP8 without loss scaling (E4M3 forward, E5M2 backward), width 64 | Share of gradients flushed to zero; loss fp32 / fp8 | 0.033; 0.009 / 0.009 | Standard: 0.493; 0.003 / 0.002 | Unit scaling keeps gradients representable. The loss is unaffected either way, because Adam ignores the scale (R-wave7). |

### WarmStartGap (W7 3.17: shrink and perturb)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Shrink and perturb, digits arriving in 4 chunks, 3 seeds | Final test accuracy | 0.727 | Warm start: 0.741. From scratch: 0.737 | Null: there is no warm-start gap at this scale, so nothing to fix. |

### LocalForwardGradientComparison (W7 3.18: forward gradients with local losses and activity perturbation)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Local activity-perturbed forward gradients (4 groups x 2 directions / 1 group x 8 directions), spirals 2-32-32-3, 30 epochs | Test accuracy; cosine to the true gradient; time | 0.980; 0.45; 616 ms. 0.967; 0.46; 515 ms | Backprop: 0.967; 1.00; 46 ms. Weight-perturbed forward gradient (32 directions): 0.593; 0.16; 3045 ms. Feedback alignment: 0.967 | Local losses close the forward-gradient gap (0.79 vs 0.987 in LearnerComparison). At 60 epochs: 0.987, equal to backprop (R-wave7). |

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

## privacy

New theme in wave 8 (package `experiments.privacy`): federated learning, attacks on private training, and private aggregation. The blocks are in `experimental.privacy`, and client splits in `experimental.data.ClientSplits`. DP-SGD with its RDP accountant is under safety (PrivateTraining).

### FederatedShootout (W8 6.1: FedProx and SCAFFOLD, 6.2: FedOpt and FedNova, 6.3: Ditto, FedRep and Per-FedAvg, 6.4: user-level DP-FedAvg with secure aggregation)
Sources: Run-w8, R-wave8. Test: FederatedShootoutTest. Digits split by a Dirichlet α = 0.1 over 20 clients, 20% participation, 30 rounds, 10 local steps.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| SCAFFOLD; FedProx, label-skewed digits | Accuracy (rounds to reach 0.5), seed 1 | SCAFFOLD 0.705 (15). FedProx 0.535 (15) | FedAvg: 0.473 (15). FedAvgM: 0.630 (15). FedAdam: 0.500 (30). FedYogi: 0.495 (never). FedNova: 0.473. Centralised: 0.845 | SCAFFOLD corrects client drift: +0.148 over FedAvg averaged over 3 seeds, and on seed 3 it reaches 0.5 in 10 rounds against 15. |
| Same methods on near-iid shapes (α = 100) | Accuracy | SCAFFOLD 0.578. FedProx 0.510 | FedAvg: 0.518. FedAvgM: 0.618. Centralised: 0.860 | The gaps shrink on iid data, though server momentum still leads. |
| FedNova, variable local steps τ ~ U[1, 20] | Accuracy | 0.523 | FedAvg: 0.515. SCAFFOLD: 0.575. FedAvgM: 0.620 | Normalised averaging barely helps. |
| FedAdam, server learning rate swept ×0.1–3 | Spread of accuracy across rates | 0.430 | Server SGD: 0.423. FedAvgM: 0.398 | Negative: FedAdam is not the most robust to the server rate. |
| Ditto; FedRep; Per-FedAvg | Mean / worst-decile accuracy: concept shift; label skew | Ditto 0.439 / 0.325; 0.874 / 0.700. FedRep 0.453 / 0.325; 0.847 / 0.638. Per-FedAvg 0.286 / 0.125; 0.838 / 0.650 | FedAvg + fine-tune: 0.460 / 0.325; 0.900 / 0.688. FedAvg: 0.303 / 0.188; 0.558 / 0.125. Local only: 0.414 / 0.313; 0.813 / 0.538 | Negative: fine-tuning FedAvg is as good as or better than every dedicated personalisation method. |
| User-level DP-FedAvg, 200 users, q = 0.1, 40 rounds | Accuracy at ε = 2 / 8; canary-audit ε | 0.295 / 0.538; 0.00 | Non-private FedAvg: 0.640; audit ε ≥ 0.40. Example-level DP-SGD: 0.580 at ε = 2 | User-level privacy is expensive with only 200 users. The audit finds no leakage. |
| Secure aggregation (Diffie–Hellman, Shamir shares, dropout recovery), 50 clients, d = 500 | Maximum error at dropout 0 / 0.1 / 0.3 (time) | 5.5e-6 / 5.0e-6 / 4.2e-6 (119 / 42 / 43 ms) | Plain summation | Exact under dropout; all 50 secrets are recovered. |

### PrivacyAttacks (W8 6.5: gradient inversion, 6.6: LiRA membership inference, 6.7: PATE)
Sources: Run-w8, R-wave8. Tests: PrivacyAttacksTest, PrivacyAttacksLiraTest, PrivacyAttacksPateTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Gradient inversion: cosine + TV with iDLG label inference; DLG, digits, batch 1 | PSNR; label accuracy | Cosine + TV 23.8 dB; 1.00. DLG 30.1 dB; 1.00. Batch 4: 23.4 dB. Shapes: 48.4 dB | — | A single gradient leaks the image. |
| Defences against inversion | PSNR; training accuracy | DP noise 4.2 dB; 0.383. Top-1% sparsification 6.5 dB; 0.583. Top-10% 24.9 dB; 0.833. QSGD 23.3 dB; 0.917. Secure aggregation 5.7 dB; 0.930 | Undefended: 23.8 dB; 0.930 | Only secure aggregation stops the attack at no accuracy cost; DP noise and top-1% stop it by wrecking training. |
| LiRA (online; offline), no defence | AUC; TPR at 0.1% FPR | Online 0.976; 0.243. Offline 0.923 | Loss-threshold attack: 0.919; 0.001 | LiRA's edge is at low false-positive rates, where it finds about 240x more members. |
| LiRA against DP-SGD at ε = 4 | AUC | 0.501 | Loss attack: 0.579 | DP stops LiRA, but test accuracy collapses from 0.347 to 0.113. |
| LiRA as an unlearning audit, forgotten vs never-seen examples | AUC after NPO / gradient ascent | 0.548 / 0.590 | Original model: 0.984. Loss attack after NPO / gradient ascent: 0.560 / 0.595 | Negative: LiRA does not separate forgotten from never-seen examples better than the loss attack. |
| PATE: GNMax; Confident-GNMax, 40 teachers, 300 queries | Student accuracy (ε; queries answered) | GNMax 0.827 (16.3; 300). Confident-GNMax 0.600 (9.99; 129) | DP-SGD at ε = 9.99: 0.910. Non-private ensemble: 0.967 | Negative: DP-SGD beats PATE at the same ε. Confident-GNMax spends less privacy but answers too few queries. |

---

## probabilistic

New theme in wave 7 (package `experiments.probabilistic`): probabilistic programming, MCMC and variational inference, Bayesian neural networks, Gaussian processes and kernels, optimal transport and topology. Blocks are in `experimental.ml`, `experimental.train` and `experimental.nn`. Wave 6's GP, uncertainty and clustering experiments are under classic and techniques.

### ProbabilisticPrograms (W7 4.1: a tiny probabilistic programming language)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Likelihood weighting; trace MH; resample-move SMC | Error: linear-regression mean / change point τ / HMM marginal / sprinkler P(rain \| wet) | LW 0.029 / 0.005 / 0.0006 / 0.0006. MH 0.016 / 0.158 / 0.0016 / 0.0086. SMC 0.026 / 0.023 / 0.0017 / 0.0099 | Linear regression: NUTS 0.007, ADVI 0.023. Exact references: forward-backward and variable elimination | All three engines are accurate on every model except MH on the change point. HMM log Z error: SMC 0.028 against 0.157 for LW. |

### PosteriorQuality (W7 4.2: HMC and NUTS, 4.3: SVGD, 4.5: flow-based VI, 4.6: AIS and SMC samplers)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| HMC (L = 10); NUTS, banana | ESS per 1000 evaluations; sliced W | HMC 22.0; 0.123. NUTS 15.1; 0.148 | Random-walk MH: 20.6; 0.132 | No clear gain on an easy 2-D target. |
| Same, funnel | ESS per 1000 evaluations; R-hat; sliced W | HMC 2.47; 1.04; 1.03. NUTS 1.28; 1.03; 0.695 | Random-walk MH: 0.658; 1.14; 0.953 | Gradients mix better on the funnel; NUTS is closest in distribution. |
| SVGD, four-mode mixture | MMD²; modes covered | -0.004; 4 | Gradient-ascent ensemble: 0.046; 4. NUTS, one chain: 0.413; 1 | The repulsive kernel spreads the particles over all modes. |
| Planar flow; IAF; radial flow, VI on a banana | KL to the target | Planar 0.110. IAF 0.191. Radial 0.228 | Mean-field and full-rank Gaussian: 0.314 | Flows fit the curved posterior that Gaussians cannot. On two moons the ELBO rises from -1.48 to 1.12 / 1.45 / 1.20. |
| AIS; SMC sampler, log Z (true 3), 30k evaluations | \|bias\|; std | AIS 0.0065; 0.277. SMC 0.0044; 0.199 | Importance sampling from the prior: 0.086; 0.266 | Annealing removes the bias. |
| AIS started at the encoder, linear-Gaussian VAE | Gap to the exact log p(x) | 0.030 | ELBO: 0.693. IWAE k = 50: 0.186. IWAE k = 5000: 0.043. AIS from the prior: 0.594 | Starting AIS at the encoder beats 5000-sample IWAE. |

### BayesianNeuralNetworks (W7 4.2: HMC reference and cold posteriors, 4.3: SVGD, 4.4: IVON, 4.12: epinet and DDU)
Sources: Run-w7, R-wave7. Tests: BayesianRegressionGapTest, BayesianImageOodTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| SWAG; deep ensemble; SGLD; GP; SVGD; IVON; MC dropout, 1-D regression with a gap | NLL / gap NLL / TV distance to HMC | SWAG -1.18 / -1.23 / 0.33. Ensemble -1.16 / -1.16 / 0.30. SGLD -0.85 / -0.38 / 0.30. GP -0.93 / -0.33 / 0.21. SVGD -0.28 / 0.38 / 0.44. IVON 0.12 / 1.94 / 0.43. MC dropout 0.14 / 0.17 / 0.57 | HMC: -0.63 / 0.59 / 0 | SWAG and ensembles have the best NLL; the GP is closest to the HMC posterior. IVON is overconfident in the gap (coverage 0.58). |
| Cold posterior, T = 0.3 / 0.1 | NLL; gap NLL; coverage in the gap | -0.61; 0.55; 0.68. -0.61; 0.58; 0.53 | T = 1: -0.79; 0.08; 1.00 | Negative: no cold-posterior effect; cooling only hurts. |
| SVGD (5 particles from an ensemble); DDU; epinet; IVON, ShapeImages with OOD inputs | Accuracy / NLL / ECE / OOD AUROC / joint NLL | SVGD 0.867 / 0.387 / 0.070 / 0.761 / 0.268. DDU 0.823 / 0.470 / 0.038 / 0.706 / 0.451. Epinet 0.783 / 0.545 / 0.031 / 0.696 / 0.385. IVON (10 samples) 0.767 / 0.537 / 0.064 / 0.652 / 0.414 | MAP: 0.807 / 0.483 / 0.031 / 0.678 / 0.484. Deep ensemble (5): 0.837 / 0.456 / 0.082 / 0.737 / 0.311 | SVGD is best on every metric but ECE. DDU and the epinet improve OOD detection in one pass; IVON is below MAP. |

### SimulatorInference (W7 4.7: SMC-ABC and neural posterior estimation)
Sources: Run-w7, R-wave7. C2ST of 0.5 is a perfect posterior. The NPE uses a mixture density network.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| NPE; SMC-ABC, Gaussian-linear | C2ST at n = 1000 / 4000 simulations | NPE 0.551 / 0.531. SMC-ABC 0.893 / 0.633 | Rejection ABC: 0.640 / 0.468. NUTS with the exact likelihood: 0.514 | NPE is closest at the small budget; rejection ABC passes it at 4000. SMC-ABC is worst. |
| Same, two moons, n = 4000 | C2ST | NPE 0.792. SMC-ABC 0.835 | Rejection ABC: 0.860 | NPE is best, but none is close. |
| Same, Lotka–Volterra | RMSE of the log-rates; posterior std | NPE 0.036; 0.055. SMC-ABC 0.109; 0.394 | Rejection ABC: 0.068; 0.265 | NPE is the sharpest and most accurate. |

### GaussianProcessKernels (W7 4.8: deep Gaussian processes and the spectral mixture kernel)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Spectral mixture kernel (Q = 3), seasonal series | MSE 1 / 2 / 3 / 4 periods ahead | 0.016 / 0.019 / 0.022 / 0.031 | Seasonal naive: 0.037 / 0.055 / 0.087 / 0.123. RBF GP: 0.347 / 0.483 / 0.528 / 0.540. Linear trend: 0.531 / 0.574 / 0.590 / 0.586 | The kernel learns the period and extrapolates 4x better than seasonal naive. |
| Deep GP with L = 1 / 2 / 3 layers, step function | NLL; CRPS | 0.071; 0.133. -0.311; 0.080. -0.478; 0.066 | Exact RBF GP: -1.09; 0.043. SVGP (M = 20): -0.076; 0.113 | Depth helps over one sparse layer, but the exact GP wins. |
| Same, heteroscedastic data | NLL; CRPS | 0.329; 0.178. 0.378; 0.186. 0.491; 0.210 | Exact GP: 0.276; 0.165. SVGP: 0.303; 0.153 | Negative: deeper is worse. |

### InfiniteWidth (W7 4.9: NTK and NNGP kernel regression)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| NNGP; NTK kernel regression (2 hidden layers), spirals | Test accuracy | NNGP 0.998. NTK 0.995 | RBF kernel: 0.998. RFF + ridge (500): 0.983. Trained MLP, width 16 / 128 / 1024: 0.943 / 0.980 / 0.983 | The infinite-width kernels beat every trained finite MLP. |
| Same, regression with a gap | NLL on the data / in the gap | NNGP -0.98 / -0.16. NTK -0.64 / 0.44 | RBF GP: -1.17 / -0.24 | The NTK GP is the worst calibrated in the gap. |
| Empirical NTK during training, width 16 → 2048 | RMS gap to the NTK predictor; relative change of the kernel | 0.195 → 0.075; 0.395 → 0.019 | – | Wide networks stay in the lazy regime, as the theory predicts. |

### KernelHypothesisTests (W7 4.10: MMD and HSIC tests)
Sources: Run-w7, R-wave7. n = 100, α = 0.05.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| HSIC, gamma approximation / permutation | Power on ring / sinusoid dependence; rejection rate under the null | 1.00 / 0.958; 1.00 / 1.00; 0.042 / 0.042 | Pearson: 0.00; 0.875; 0.083. Distance correlation: 0.625; 1.00; 0.042 | HSIC finds non-linear dependence that correlation misses. |
| MMD (median heuristic); deep-kernel MMD | Power: mean shift / variance shift / Laplace; null rejection | MMD 0.542 / 0.333 / 0.125; 0.00. Deep 0.125 / 0.125 / 0.00; 0.00 | t-test: 0.667 / 0.125 / 0.083; 0.125. KS: 0.500 / 0.167 / 0.125; 0.042. Sliced W: 0.417 / 0.292 / 0.125; 0.00 | MMD wins on the variance shift and never false-rejects. Negative: the deep kernel is weak with 50 points per half. |

### KernelScaling (W7 4.11: Nyström kernel ridge regression and FALKON)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Nyström; FALKON, m = 200 centres, noisy spirals | Test error (fit s) at n = 20,000 / 50,000 | Nyström 0.042 (0.54) / 0.044 (0.40). FALKON 0.0475 (0.20) / 0.049 (0.39) | Exact KRR at n = 1000: 0.057 (0.16). Kernel SVM at n = 1000: 0.0585. RFF + ridge, 200 features: 0.0465 / 0.0465 | 20–50x more data at m = 200 cuts the error by about a quarter in under a second. RFF is as good. |

### TransportFlows (W7 4.13: entropic OT, Sinkhorn divergences and Sinkhorn couplings for flow matching)
Sources: Run-w7, R-wave7. Tests: TransportFlowsTest, SinkhornLossTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Flow matching with Sinkhorn (n = 1024) / Hungarian (n = 256) minibatch couplings, moons → rings | 1-step / 4-step sliced W; straightness | Sinkhorn 0.127 / 0.119; 0.0042. Hungarian 0.121 / 0.115; 0.0040 | Independent pairing: 0.326 / 0.160; 0.546 | OT pairing makes the paths about 130x straighter and one-step sampling work. |
| Scaling-form Sinkhorn | ms per iteration at n = 256 / 1024 / 2048 | 0.67 / 2.68 / 9.07 | Log-domain Sinkhorn: 2.13 / 16.6 / 80.9. Hungarian at n = 256: 59.6 ms per solve | The scaling form is 3–9x faster. |
| Sinkhorn-divergence generator loss | Final sliced W | 0.093 | Sliced-Wasserstein loss: 0.092 | A tie. |

### TransportMaps (W7 4.14: neural OT with input-convex networks)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| ICNN Monge map, Gaussian → Gaussian | L²-UVP | 2.89% | Sinkhorn barycentric projection: 1.62%. Exact discrete OT: 1.86% | Negative: the ICNN trails both discrete baselines; it has not converged in 300 steps. Unlike them, it maps new points without target samples (R-wave7). |
| Same, moons → rings | Sliced W; transport cost | 0.168; 0.267 | Sinkhorn: 0.085; 0.377. Exact OT: 0.087; 0.419. Identity: 0.227; 0 | The cheapest map, but a worse fit to the target. |

### TransportGeometry (W7 4.15: Gromov–Wasserstein alignment and Wasserstein barycenters)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Gromov–Wasserstein graph matching | Accuracy at edge noise 0 / 0.05 / 0.15 | 1.00 / 0.333 / 0.313 | Spectral (Umeyama): 1.00 / 0.188 / 0.167 | GW is more robust, though both degrade fast. |
| Gromov–Wasserstein embedding alignment, no anchors | Precision@1 | 1.00 | Procrustes with 3 / 8 / 20 anchors: 0.35 / 1.00 / 1.00 | GW needs no anchors. The embeddings are a synthetic rotated 8-D space (R-wave7). |
| Wasserstein barycenter of glyphs | Mean W² cost; entropy | 10.9; 2.49 | Euclidean mean: 16.8; 4.07 (blurred). One input: 20.7; 2.08 | A sharp, central shape. |

### TopologicalFeatures (W7 4.16: persistent homology features)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Vietoris–Rips barcode features + MLP, five topology classes (blob, circle, two circles, figure-eight, torus) | Accuracy, canonical pose / randomly rotated | 1.00 / 0.933 | Deep Sets: 0.967 / 0.217. Raw-point MLP: 0.250 / 0.233 | Barcodes are rotation invariant; Deep Sets collapse under rotation. Barcode time grows from 2 ms at n = 50 to 163 ms at n = 150. |

### TopologicalAutoencoders (W7 4.17: topological autoencoder loss)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Topological autoencoder, nested spheres 10-D → 2-D | Trustworthiness / 5-NN accuracy / KL₀.₁ / H0 bottleneck | 0.849 / 0.935 / 0.143 / 0.483 | Plain AE: 0.876 / 0.942 / 0.194 / 0.471. PCA: 0.803 / 0.873 / 0.188 / 0.431 | Mixed: KL₀.₁ improves, which is all the test asserts; trustworthiness and the bottleneck distance do not. With λ = 50 the bottleneck reaches 0.427 but trustworthiness drops to 0.79 (R-wave7). |

### NonparametricClustering (W7 4.18: Dirichlet-process mixtures)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DP Gibbs; stick-breaking VI; DP-means, blobs with unknown K (2–12) | ARI; \|K error\|; held-out log-likelihood | Gibbs 0.923; 0.67; -3.74. VI 0.928; 3.50; -3.59. DP-means 0.875; 1.50; -3.61 | GMM + BIC: 0.991; 0.33; -3.59. DBSCAN: 0.958; 0.17. k-means + elbow: 0.671; 3.67 | Negative: on well-separated static blobs, GMM + BIC and DBSCAN beat the DP models. |
| Warm-started DP Gibbs; online DP-means, stream where new clusters appear | ARI; \|K error\| | Gibbs 0.981; 0.6. DP-means 1.00; 0.0 | GMM + BIC refitted per chunk: 0.914; 0.6. k-means with K from the first chunk: 0.566; 4.0 | Nonparametric models pay off when clusters appear over time. |

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

### LatinSquarePuzzles: constraint layers (W7 6.12: semantic loss, 6.14: SATNet)
Sources: Run-w7, R-wave7. Tests: ConstraintLatinSquaresTest, SatNetLayerTest, SemanticLossTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| SATNet layer; MLP + semantic loss, Latin squares with 6 blanks | Cell accuracy / squares solved | SATNet 0.973 / 0.949 (1122 ms training). Semantic loss 0.856 / 0.420 | MLP: 0.866 / 0.455. Wave 6: TRM and HRM about 0.72; AKOrN solves 0.62 | SATNet is the best Latin-square solver on this page. Semantic loss does not help. The layer's gradient check error is 1.75e-5. |

### NeuroSymbolicArithmetic (W7 6.11: DeepProbLog / Scallop-style neural predicates)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Neural predicate (exact / top-3 proofs), digit addition learned from sums only | Sum accuracy / digit accuracy | 0.745 / 0.870. Top-3: 0.713 / 0.848 | End-to-end network: 0.223. Oracle trained on digit labels: 0.785 / 0.880 | The program lets the digit classifier learn from sums alone, close to the oracle. |

### RuleLearning (W7 6.13: differentiable ILP and neural logic machines)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| dILP; neural logic machine, 50-node graphs | Pair accuracy: grandparent / uncle / reachability | dILP 1.000 / 1.000 / 1.000. NLM 0.847 / 0.711 / 0.767 | GNN: 0.889 / 0.894 / 0.931 | dILP recovers the exact rules, for example T(X,Y) :- T(X,Z), edge(Z,Y). The NLM trails the GNN. |

### ArithmeticExtrapolation: Neural GPU (W7 6.10)
Sources: Run-w7, R-wave7. Test: NeuralGpuLengthTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Neural GPU, binary addition trained on 1–6 bits | Exact accuracy at 6 / 12 / 24 bits | 1.00 / 0.34 / 0.00 | Transformer: 0.90 / 0.06 / 0.00 | Better at twice the length, but nothing reaches 4x. |

### ArithmeticRl: evolution strategies for fine-tuning (W7 5.18)
Sources: Run-w7, R-wave7. Test: EvolutionFineTuningTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| ES; MeZO, addition from a 0.24 base model, 8192 reward evaluations, 3 seeds | Accuracy (KL to the base) | ES 0.280 ± 0.000 (0.249). MeZO 0.280 ± 0.000 (0.136) | GRPO: 0.413 ± 0.082 (0.507) | Negative: GRPO wins at this scale. ES and MeZO gain one puzzle each, at lower KL. |

---

## recsys

New theme in wave 7 (package `experiments.recsys`): recommender systems, retrieval, learning to rank and tabular deep learning. Blocks are in `experimental.recsys`, with the tabular blocks in `experimental.nn`. Data come from `SyntheticRecsys`, `SyntheticCorpus`, `TabularStressSuite` and `TabularPrior` in `experimental.data`. The wave 1 recommender and tabular results are under graphs (RecsysTabularBenchmarks).

### CollaborativeFiltering (W7 2.1: LightGCN, 2.3: Mult-VAE and EASE)
Sources: Run-w7, R-wave7. 300 users x 200 items.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| LightGCN with K = 0 / 1 / 2 / 3 layers | Recall@10 | 0.609 / 0.738 / 0.744 / 0.733 (NDCG@10 at K = 2: 0.733) | Item-kNN: 0.697. BPR-MF: 0.591. NGCF-lite, K = 2: 0.540. Popularity: 0.140 | Propagation without feature transforms beats MF, kNN and NGCF; one layer gives most of the gain. |
| Mult-VAE; EASE | Recall@10 / NDCG@10 (fit ms) | Mult-VAE 0.730 / 0.705 (563). EASE 0.655 / 0.626 (15) | BPR-MF: 0.591 / 0.552 (139) | Both beat MF; EASE is a closed form fitted in 15 ms. |

### ClickModels (W7 2.4: NeuMF, Wide & Deep and DeepFM)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Wide & Deep; DeepFM, click prediction | Accuracy / AUC / log-loss | Wide & Deep 0.734 / 0.811 / 0.531. DeepFM 0.736 / 0.814 / 0.531 | FM: 0.729 / 0.809 / 0.540. DCN-V2: 0.738 / 0.814 / 0.537. Logistic: 0.593 / 0.659 / 0.650 | All the deep click models tie with FM. |
| NeuMF | Recall@10 / NDCG@10 | 0.559 / 0.531 | BPR-MF: 0.583 / 0.542. Popularity: 0.140 / 0.115 | Negative: NeuMF does not beat dot-product MF. |

### SequentialRecommendation (W7 2.2: SASRec, BERT4Rec and DIN, 2.6: HSTU)
Sources: Run-w7, R-wave7. Tests: SequentialRecommendationTest, TargetAttentionTest, HstuTransductionTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| BERT4Rec; GRU4Rec; SASRec, next item | HR@10 / NDCG@10 | BERT4Rec 0.788 / 0.680. GRU4Rec 0.776 / 0.707. SASRec 0.760 / 0.703 | First-order Markov chain: 0.752 / 0.701. BPR-MF: 0.224 / 0.096. Popularity: 0.080 / 0.038 | The sequence models barely beat a Markov chain on this data. |
| DIN target attention | Click AUC | 0.619 | Mean pooling: 0.649. FM: 0.643 | Negative. |
| HSTU, d = 16 / 24 / 32 | HR@10 / NDCG@10 / click AUC (parameters) | 0.750 / 0.654 / 0.595 (3,833). 0.748 / 0.651 / 0.542 (6,697). 0.750 / 0.666 / 0.653 (10,201) | SASRec, d = 32: 0.758 / 0.678 / 0.677 (13,313) | Null: about the same retrieval with fewer parameters, and a worse AUC. |

### GenerativeRetrieval (W7 2.5: TIGER, generative retrieval with semantic IDs)
Sources: Run-w7, R-wave7. Semantic IDs come from residual k-means, not an RQ-VAE (R-wave7).

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| TIGER with trie-constrained beam search, ID depth 1 / 2 | Recall@10 on all / cold items (ID collision rate) | 0.620 / 0.296 (0.79). 0.664 / 0.259 (0.56) | SASRec: 0.660 / 0.000. Two-tower + HNSW: 0.760 / 0.778 | Semantic IDs reach cold items that SASRec cannot, but two-tower retrieval beats both. |

### LoggedBanditFeedback (W7 2.7: off-policy evaluation and learning, 2.8: contextual bandits)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| SNIPS; DR; IPS; Switch-DR, nonlinear world, n = 2000, β = 1 | OPE RMSE (true value 0.7725) | SNIPS 0.025. DR 0.042. IPS 0.075. Switch 0.095 | Naive: 0.177. Direct method: 0.174 | Self-normalised IPS is the most accurate. |
| Same, β = 4 (sharp logging policy) | OPE RMSE | Switch 0.095. SNIPS 0.125. DR 0.264. IPS 0.826 | Naive: 0.122. Direct method: 0.123 | IPS blows up and only Switch beats the naive estimate. In a linear world the direct method (0.022) and Switch (0.024) win. |
| Counterfactual risk minimisation | True value of the learned policy | 0.781 | Logging policy: 0.710. Greedy on a reward model: 0.831. Oracle: 0.839. Uniform: 0.499 | Negative: CRM improves on the logging policy but loses to the model-based greedy policy. |
| LinUCB; linear Thompson sampling; neural-linear, 10,000 rounds | Regret, linear / nonlinear world | LinUCB 165 / 1,405. Linear TS 187 / 1,823. Neural-linear 262 / 798 | ε-greedy: 468 / 1,463. Neural greedy: 1,843 / 997. UCB1 without context: 3,455 / 2,828 | LinUCB is best when its model is right; neural-linear when it is not. |

### RetrievalShootout (W7 2.9: BM25 and DPR, 2.10: ColBERT, 2.11: SPLADE, 2.13: RankNet, LambdaRank, LambdaMART and ListNet)
Sources: Run-w7, R-wave7. Tests: RetrievalShootoutTest, LateInteractionTest, SparseRetrievalTest, RerankingTest. MRR@10 is given on lexical / paraphrase / multi-aspect queries. The text encoders have no transformer blocks, because one block lowered paraphrase MRR from 0.71 to 0.25 (R-wave7). Each test trains its own DPR, so its numbers vary slightly between rows.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DPR with hard negatives / in-batch negatives only | MRR@10 | 0.89 / 0.43 / 0.81. In-batch only: 0.81 / 0.70 / 0.94 | BM25: 1.00 / 0.01 / 1.00. TF-IDF: 1.00 / 0.02 / 1.00. Untrained encoder: 0.32 / 0.02 / 0.31 | Dense retrieval handles paraphrases that BM25 cannot. Hard negatives hurt on paraphrase and multi-aspect queries. |
| Hybrid (reciprocal rank fusion of BM25 and DPR) | MRR@10 | 0.98 / 0.13 / 0.98 | Same | Fusion keeps BM25's lexical wins but loses most of DPR's paraphrase gain. |
| ColBERT late interaction, fp32 / 2-bit residual | MRR@10 (bytes per document) | 0.81 / 0.24 / 0.85 (512). 0.58 / 0.22 / 0.73 (40) | DPR: 0.88 / 0.41 / 0.83 (128) | Negative: ColBERT trails single-vector DPR on paraphrases here. |
| SPLADE, FLOPS weight 0.01 / 0.1, 200 steps | MRR@10 (non-zero terms per document) | 0.87 / 0.55 / 0.97 (154). 0.82 / 0.63 / 0.89 (70) | BM25: 1.00 / 0.01 / 1.00 (7.5). DPR: 0.86 / 0.41 / 0.82 | Learned sparse expansion beats DPR on paraphrase and multi-aspect queries. |
| RankNet; LambdaRank; LambdaMART; ListNet, reranking | NDCG@5 / NDCG@10 | RankNet 0.921 / 0.934. LambdaRank 0.917 / 0.930. LambdaMART 0.900 / 0.919. ListNet 0.896 / 0.918 | Pointwise GBT: 0.910 / 0.926. Pointwise MLP: 0.902 / 0.920. Hybrid order: 0.704 / 0.721 | The pairwise neural losses edge out pointwise training; LambdaMART does not beat pointwise GBT. |

### EmbeddingSearch (W7 2.12: IVF-PQ and anisotropic quantisation, 2.14: Matryoshka representations)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Anisotropic IVF-PQ, probe 4 / 16, 8 bytes per vector | Recall@10 (queries per second) | 0.842 (10,522) / 0.844 (2,302) | Plain IVF-PQ: 0.786 (5,270) / 0.788 (2,205). HNSW: 0.885 (13,510) at 256 bytes. Flat: 1.000 (370) at 128 bytes | The anisotropic loss adds about 0.06 recall at the same 8 bytes. |
| Matryoshka embeddings | Recall@10 at d = 4 / 8 / 16 / 32 | 0.106 / 0.389 / 0.473 / 0.478 | Plain, truncated: 0.119 / 0.255 / 0.411 / 0.484. PCA of plain: 0.182 / 0.482 / 0.485 / 0.484. Separate models: 0.158 / 0.468 / 0.486 / 0.484 | It beats truncation at 8–16 dimensions, but not PCA or separate models. Adaptive 4 → 32 retrieval: 0.355 against 0.328. |

### TabularShootout (W7 2.15: FT-Transformer and PLR embeddings, 2.16: TabM and RealMLP, 2.17: prior-fitted networks, 2.18: NODE and NAM)
Sources: Run-w7, R-wave7. Tests: TabularShootoutTest, PeriodicEmbeddingTest, PriorFittedNetworkTest, TabularStressTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| RealMLP; TabM (k = 8); MLP + PLR; FT-Transformer, interaction table | Accuracy / AUC (ms) | RealMLP 0.918 / 0.964 (440). TabM 0.915 / 0.962 (1223). PLR 0.893 / 0.955 (374). FT-Transformer 0.881 / 0.944 (1932) | Deep ensemble of 5 MLPs: 0.926 / 0.965 (418). DCN-V2: 0.911 / 0.958 (28). TabNet-lite: 0.890 / 0.933. MLP: 0.882 / 0.952. GBT: 0.866 / 0.936. Random forest: 0.788 / 0.871 | RealMLP and TabM beat the MLP but not a 5-MLP ensemble. FT-Transformer gains nothing at 25x the MLP's time. |
| PLR periodic embeddings, target sin(6x) | Accuracy | σ = 0.1: 0.927. σ = 1: 0.943 | MLP: 0.474. GBT: 0.876. RealMLP: 0.489 | Periodic embeddings solve the high-frequency target. RealMLP's frequency init fails on it. |
| Prior-fitted network (TabPFN v1 layout), 150 pre-training steps, 100 context rows | Accuracy on the prior / held-out trees / interaction table | 0.720 / 0.602 / 0.498 | Logistic: 0.802 / 0.745 / 0.598. MLP: 0.783 / 0.770 / 0.855. Random forest: 0.807 / 0.837 / 0.715. GBT: 0.803 / 0.842 / 0.770. GP classifier: 0.807 / 0.777 / 0.798 | Negative: limited by the CPU budget. 1500 steps reach 0.782 / 0.710 / 0.605, still below every baseline (R-wave7). |
| NODE; NAM, trees-favouring stress suite: 0% / 90% noise features / rotated | Accuracy | NODE 0.774 / 0.696 / 0.650. NAM 0.794 / 0.786 / 0.630 | GBT: 0.853 / 0.869 / 0.713. Random forest: 0.824 / 0.719 / 0.723. MLP: 0.680 / 0.614 / 0.696 | Trees win until the features are rotated. NAM shrugs off noise features, and its shape functions correlate 0.816 with the planted effects. On the interaction table: NODE 0.822, NAM 0.582. |

---

## robustness

New theme in wave 8 (package `experiments.robustness`): adversarial training, certified robustness and test-time adaptation. The blocks are in `experimental.robust`. Robust accuracy is measured under PGD-20.

### AdversarialRobustness (W8 6.8: PGD-AT, TRADES and fast FGSM training, 6.9: randomized smoothing, 6.10: IBP, CROWN and CROWN-IBP, 6.11: GroupSort and AOL 1-Lipschitz networks)
Sources: Run-w8, R-wave8. Tests: AdversarialRobustnessTest, AdversarialShapesTest, CertifiedRobustnessTest, LipschitzRobustnessTest, plus the block test RobustBlocksTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| TRADES; PGD-AT; fast FGSM training, digits MLP | Clean; robust at ℓ∞ 0.05 / 0.1 / 0.2 (training time) | TRADES 0.977; 0.880 / 0.613 / 0.090 (582 ms). PGD-AT 0.980; 0.857 / 0.593 / 0.060 (447 ms). Fast FGSM 0.973; 0.853 / 0.573 / 0.033 (197 ms) | Standard: 0.947; 0.763 / 0.260 / 0.000 (78 ms). Gaussian-noise training: 0.960; 0.757 / 0.263 / 0.000 | TRADES is the most robust, and fast FGSM gets most of the gain at less than half PGD-AT's cost. Clean accuracy rises too. |
| Fast FGSM; PGD-AT, shapes CNN | Clean; robust at 0.05 / 0.1 / 0.2 | Fast FGSM 0.403; 0.257 / 0.190 / 0.123. PGD-AT 0.333; 0.247 / 0.173 / 0.150 | Standard: 0.673; 0.283 / 0.057 / 0.003 | The usual trade-off: robustness at large ε costs about half the clean accuracy. |
| Randomized smoothing with a noise-trained model, σ = 0.25; σ = 0.5 | Certified ℓ2 accuracy at radius 0.25 / 0.5 / 1.0 | σ 0.25: 0.833 / 0.517 / 0. σ 0.5: 0.600 / 0.233 / 0 | Standard model smoothed at σ 0.25: 0.733 / 0.317 / 0. Empirical ℓ2 PGD-AT accuracy: 0.993 / 0.873 / 0.247 | Smoothing needs noise training. Certificates stay well below empirical robustness, and nothing certifies at radius 1.0. |
| IBP; CROWN-IBP certified training | Clean; PGD; IBP-verified; CROWN-verified at ℓ∞ 0.05 | IBP 0.987; 0.867; 0.480; 0.760. CROWN-IBP 0.980; 0.860; 0.463; 0.767 | Standard: 0.960; 0.777; 0.003; 0.597. PGD-AT: 0.997; 0.897; 0.003; 0.747 | PGD-AT is robust but not verifiable by IBP; certified training makes the cheap IBP bound useful. CROWN verifies every model far better than IBP. |
| Björck + GroupSort; AOL + GroupSort 1-Lipschitz networks | Clean; certified ℓ2 at 0.25 / 0.5 (training time) | Björck 0.987; 0.763 / 0.227 (1,740 ms). AOL 1.000; 0.570 / 0.003 (427 ms) | Spectral norm + ReLU: 0.990; 0.597 / 0.047 (175 ms) | Orthonormal Björck weights give the best margin certificates. AOL certifies least, though it is the most robust empirically (PGD 0.917 at 0.5). |

### ShiftAdaptation (W8 6.12: test-time adaptation with BN statistics, TENT and MEMO)
Sources: Run-w8, R-wave8. Test: ShiftAdaptationTest. A shapes CNN with batch norm, clean accuracy 1.000.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| BN-statistics adaptation; TENT | Mean accuracy over 4 corruptions at severities 3 and 5 | BN statistics 0.729. TENT 0.729 | No adaptation: 0.278 | Re-estimating BN statistics recovers most of the gap (contrast 0.176 → 1.000, blur 0.445 → 0.973); TENT adds nothing on top. |
| MEMO at batch size 1 | Accuracy on noise / blur at severity 3 | 0.417 / 0.583 | No adaptation: 0.441 / 0.445. GroupNorm model: 0.480 / 0.414. Augmentation-trained model: 0.520 / 0.695 | MEMO helps a little on blur and hurts on noise. |
| BN statistics; TENT on a label-shifted stream (one class per batch) | Accuracy | 0.177; 0.174 | No adaptation: 0.427 | Negative: batch statistics break when a batch holds one class. TENT does not collapse on long streams (0.497 after 5 batches, 0.498 after 60). |

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

### CausalAndFairness: follow-ups (W7 6.15: V-REx and Just Train Twice, 6.16: DAGMA and GES)
Sources: Run-w7, R-wave7. Test: CausalFollowUpTest. GES is a hill climb, without the full CPDAG operators (R-wave7).

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| JTT; V-REx, spurious colour | Worst-group / average accuracy | JTT 0.493 / 0.713. V-REx 0.318 / 0.621 | ERM: 0.379 / 0.669. Wave 6: IRM 0.268, GroupDRO with group labels 0.462 | JTT beats ERM, and even GroupDRO, without group labels. V-REx is worse than ERM. |
| DAGMA; GES, linear SEM with d = 10 / 20 | SHD | DAGMA 0 / 0. GES 16 / 33 | NOTEARS: 4 / 16. PC skeleton: 13 / 29 | DAGMA recovers both graphs exactly. |
| Same, tanh SEM | SHD | DAGMA 0 / 5. GES 11 / 33 | NOTEARS: 1 / 3. PC skeleton: 17 / 36 | NOTEARS wins at d = 20. GES is poor throughout. |

---

## sciml

New theme in wave 8 (package `experiments.sciml`): physics-informed networks, neural operators, learned simulators, equation discovery, symmetric models and data assimilation. The blocks are in `experimental.sciml`. Errors are relative L2 unless stated (lower is better). Each comparison uses one seed (R-wave8).

### PdeShootout (W8 3.1: physics-informed neural networks, 3.2: PINN training fixes, 3.3: PirateNets and physics-informed KANs, 3.17: hard constraints and neural conservation laws)
Sources: Run-w8, R-wave8. Tests: PinnShootoutTest, PinnHardPdeTest, PinnTrainingFixesTest, PinnDepthSweepTest, PikanDepthSweepTest, ConstrainedFieldTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| PINN, heat; Poisson | Rel L2 (time) | Heat 0.0094 (1.9 s). Poisson 0.0016 (1.7 s) | Data-fit MLP: 0.178; 0.327. Coarse spectral / finite-difference solver: 3e-8 (0.9 ms); 6.2e-3 (0.1 ms) | PINNs solve smooth problems, and beat the coarse finite-difference solver on Poisson, but a classical solver is far faster. |
| PINN, Burgers; Allen–Cahn; advection at β = 10 | Rel L2 | 0.496; 0.993; 0.893 | Coarse solver: 0.23; 0.11; 3.4e-3 | Negative: plain PINNs fail on shocks, stiff reactions and fast advection. |
| Self-adaptive weights; NTK balancing; causal training; RAR-D; all four, advection at β = 5 | Rel L2 | 0.480; 0.502; 0.436; 0.487; 0.370 | Plain PINN: 0.483 | Only causal training helps on its own; all four together cut the error by a quarter. At β = 10 every recipe stays near 1.0 (R-wave8). |
| PirateNet, Poisson depth sweep | Rel L2 at 1 / 4 blocks (up to 12 layers) | 0.0047 / 0.0003 | Plain MLP: 0.0021 / 1.34 | The adaptive residual keeps a deep PINN trainable, where the plain MLP fails. |
| Physics-informed KAN (PIKAN); gated PIKAN | Rel L2 at 1 / 3 blocks | 0.155 / 0.037. Gated 0.344 / 0.207 | PirateNet: 0.0047 at 1 block | Depth helps the KAN and gating does not; both trail the MLP-based PINNs. |
| Hard divergence-free constraint (stream function), Taylor–Green | Rel L2; maximum violation | 0.388; 4e-7 | Unconstrained MLP: 0.884; 1.9. Soft penalty (weight 0.1–10): 0.77–0.83; 0.82–0.047. Fourier-feature MLP: 0.857; 2.3 | Building the constraint in makes it exact and also more accurate. |
| Neural conservation law, advected density | Rel L2; maximum violation | 0.153; 1e-6 | Unconstrained MLP: 0.396; 0.64. Soft penalty: 0.579; 0.10 | The same holds for conservation. |

### OperatorLearning (W8 3.4: DeepONet, 3.5: Transolver and the linear attention neural operator, 3.6: message-passing PDE solvers with the pushforward trick, 3.7: PDE-Refiner)
Sources: Run-w8, R-wave8. Tests: DeepOnetOperatorTest, PhysicsAttentionOperatorTest, MessagePassingRolloutTest, PdeRefinerRolloutTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DeepONet: antiderivative; Burgers; Darcy | Rel L2 on-grid / off-grid | 0.365 / 0.369; 0.580 / 0.583; 0.128 / 0.128 | FNO: 0.369 / 0.364; 0.145 / 0.146; 0.255 / 0.255. Grid-to-grid MLP: 0.536 / 0.373; 0.336 / 0.300; 0.171 / 0.134 | DeepONet wins on Darcy, FNO on Burgers by a wide margin, and they tie on the antiderivative. |
| Transolver physics attention; LANO, Burgers | Rel L2, regular / irregular points | Transolver 0.576 / 0.579. LANO 0.555 / 0.520 | FNO: 0.167 / 0.178. Linear attention: 0.570 / 0.565. DeepONet: 0.588 / 0.524 | Negative: FNO wins clearly at this size. Transolver costs 292 µs per sample at N = 64 and 1,629 µs at N = 256. |
| MP-PDE with pushforward and temporal bundling (2 steps), Burgers rollouts | MSE; survival steps | 0.0185; 32.8 | No pushforward: 0.0174; 37.3. No bundling: 0.348; 23.0. FNO, autoregressive: 0.0088; 39.0. Solver at half resolution: 0.0198; 37.3 | Bundling is what matters; pushforward does not help here, and FNO is best. |
| PDE-Refiner (K = 3), Kuramoto–Sivashinsky | Correlation time; low-band / high-band error | 14.8; 0.190 / 10.8 | Same network trained on one-step MSE: 34.3; 0.011 / 3.5. FNO: 23.0; 0.013 / 15.2. MP-PDE: 12.3; 0.030 / 13.7 | Negative: refinement halves the useful rollout at this budget. It uses an FNO backbone with a noise schedule, not a full diffusion denoiser (R-wave8). |

### DynamicsLearning (W8 3.8: Hamiltonian and Lagrangian networks, 3.9: SINDy, 3.10: DMD, EDMD and deep Koopman autoencoders, 3.11: universal differential equations, 3.12: neural and latent SDEs, 3.13: graph network simulators)
Sources: Run-w8, R-wave8. Tests: HamiltonianDynamicsTest, SindyDiscoveryTest, KoopmanForecastTest, UniversalDifferentialEquationTest, NeuralSdeTest, GraphNetworkSimulatorTest, plus the block test SciMlBlocksTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| HNN + Störmer–Verlet; LNN + RK4, pendulum | Rollout MSE; energy drift | HNN 0.0049; 0.047. LNN 0.0030; 0.015 | MLP + RK4: 0.073; 0.647. MLP + Euler: 2.32; 3.65 | Built-in mechanics cut the rollout error 15–25x and the energy drift 14–40x. The LNN needed a unit kinetic prior (R-wave8). |
| SINDy, Lorenz | Share of the true structure recovered; coefficient error (time) | 1.00; 0.000 (11 ms). At 1% noise: 1.00; 0.019 | Genetic programming: derivative error 0.406 (293 ms). MLP derivative model: 0.085 (2.4 s) | Recovers the exact equations, x' = 10(y − x), y' = 28x − y − xz, z' = −2.667z + xy, and Lotka–Volterra and a damped oscillator too (error ≤ 0.001). |
| DMD; EDMD (cubic); Koopman autoencoder | Normalised MSE; valid steps, Van der Pol / Lorenz-63 | DMD 0.72; 7 / 1.10; 1. EDMD 1.97; 0 / 6.94; 0. Autoencoder 0.67; 21 / 2.70; 7 | ESN: 1e-10; 200 / 0.150; 141. NG-RC: 1.10; 4 / 0.0041; 200. One-step MLP: 0.040; 95 / 1.32; 16 | Negative: Koopman methods lose to reservoir computing on nonlinear systems. DMD is exact on linear advection–diffusion (4e-8; 100 of 100 steps). |
| UDE followed by SINDy on the learned term, Lotka–Volterra | Extrapolation MSE with 20 / 60 training points | 1.08 / 0.58 | UDE alone: 328 / 325. Known physics only: 101 / 114. Pure neural ODE: 28.3 / 29.1 | The symbolic step is what makes it extrapolate. It recovers −0.87xy and +0.78xy (true −0.9 and +0.8). |
| Neural SDE | W2 distance, Ornstein–Uhlenbeck / double well (switch rate) | 0.103 / 1.164 (0.040) | Deterministic neural ODE drift: 0.368 / 1.090 (0.000). Quantile MLP: 0.156 / 0.244 (0.020). True switch rate: 0.060 | Best on OU. On the double well it switches wells but its W2 is worse than both baselines. A latent SDE reaches W2 0.170 but switches far too often (0.168 vs 0.046). |
| Graph network simulator with input noise, spring chains | Rollout MSE at training size / double size | 0.0157 / 0.0071 | GNS without noise: 0.0146 / 0.0102. EGNN: 0.0032 / 0.0278. Fixed-size MLP: 0.0048 / cannot run | The graph simulator generalises to twice the size, and input noise helps only there. EGNN is best in distribution but fails at double size. |

### SymmetricModels (W8 3.14: frame averaging and equivariant MLPs, 3.15: SchNet with force matching)
Sources: Run-w8, R-wave8. Tests: EquivariantRegressionTest, SchNetPotentialTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Frame averaging; O(3)-equivariant MLP; EGNN, O(3)-invariant regression on rotated clouds | Relative test MSE with 32 / 256 samples | Frame averaging 2.24 / 0.85. Equivariant MLP 1.52 / 0.45. EGNN 0.029 / 0.011 | MLP: 2.73 / 1.07. MLP + rotation augmentation: 0.78 / 0.74 | EGNN wins by far. The other two symmetric models are exactly invariant (gap about 1e-6, against 0.4–2.6 for the MLPs), yet augmentation beats them at 32 samples. |
| SchNet trained on energies and forces, Lennard-Jones 5-atom clusters | Energy MAE per atom; force MAE; MD energy drift per atom | 0.040; 0.995; 0.017 | SchNet on energies only: 0.089; 2.46; 0.090. EGNN: 0.46; 12.6; 6.2. MLP on sorted distances: 0.63; 14.5; 8.3 | Force matching halves the energy error and cuts MD drift 5x. |

### DataAssimilation (W8 3.16: ensemble Kalman filter and score-based data assimilation)
Sources: Run-w8, R-wave8. Tests: EnsembleKalmanLorenzTest, ScoreAssimilationTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Localised EnKF (20 members), Lorenz-96, n = 40, every 4th variable observed | RMSE; spread ratio | 1.64; 1.16 | EnKF without localisation: 5.18. EKF: 2.43; 0.38. Particle filter (500 particles): 4.31. Free run: 5.06 | Localisation makes a 20-member ensemble work in 40 dimensions, where the particle filter collapses. |
| Score-based data assimilation (window 8), Lorenz-63 with x observed | RMSE | 5.02 | EnKF: 0.92. Free run: 10.8 | Negative: the score model loses to the EnKF; a window of 16 with more training reached 4.72 (R-wave8). |

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

### SelfSupervisedBenchmark: I-JEPA with the full recipe (W7 6.18)
Sources: Run-w7, R-wave7. Test: JepaFollowUpTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| I-JEPA, full recipe | Linear-probe accuracy (effective rank) at 100 / 200 steps | 0.398 (11.7) / 0.409 (10.4) | I-JEPA-lite: 0.391 (6.9) / 0.503 (5.4). Random encoder: 0.341. MAE 0.59 and SimCLR 0.56 at 100–150 steps | Negative: a higher-rank representation, but no better probe; the lite version overtakes it at 200 steps. |

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

## serving

New theme in wave 8 (package `experiments.serving`): approximate inference, KV memory, speculative decoding, multi-adapter serving, fair scheduling and grammar-constrained decoding on toy language models. The blocks are in `experimental.serving`. The wave 4 and 5 serving experiments (ServingThroughput, SharedPrefixServing) stay under language. Rows use 3 seeds (R-wave8).

### ServingShootout (W8 4.1: SpecKV and SpecPC, 4.2: LLMLingua and LongLLMLingua)
Sources: Run-w8, R-wave8. Test: ServingShootoutTest. Chain retrieval with a 128-token prompt and 16 distractors.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| SpecKV (draft-guided KV selection) | Accuracy at a 10 / 20 / 30 / 50% KV budget | 0.90 / 0.92 / 0.94 / 0.94 (+131 draft tokens) | SnapKV: 0.01 / 0.07 / 0.14 / 0.15. H2O: 0.00 / 0.02 / 0.01 / 0.31. Full cache: 1.00. Quest: 1.00, because it reads the budget and evicts nothing | The draft's answer tells the cache what to keep; question-only eviction fails. |
| SpecPC (draft-guided prompt compression) | Accuracy at a 10 / 20 / 30 / 50% prompt budget | 0.81 / 0.90 / 0.93 / 0.96 | LLMLingua: 0.00 / 0.06 / 0.18 / 0.76. LongLLMLingua: 0.01 / 0.04 / 0.09 / 0.83. Truncation: 0.00 / 0.00 / 0.03 / 0.11 | Perplexity-based compression keeps the wrong tokens at tight budgets. |

### KvMemoryShootout (W8 4.3: MiniCache, 4.4: MagicPIG)
Sources: Run-w8, R-wave8. Test: KvMemoryShootoutTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| MiniCache (merging layers 2–3 and 4–5); MiniCache + KIVI 2-bit | Bytes per token; KL; top-1 agreement | 1,088; 4.8e-6; 0.99. With KIVI: 442; 1.0e-3; 0.90 | Full cache: 1,536. KIVI 2-bit alone: 624; 7.8e-4; 0.90. Untrained layer sharing: 1,024; 4.3e-4; 0.96 | Merging adjacent layers is nearly free and stacks with quantisation. |
| MagicPIG (LSH sampling), about 310 of 1,024 keys touched | Needle accuracy; aggregation relative error and accuracy | 1.00; 0.171 and 1.00 | Top-k on the same key budget: 1.00; 0.244 and 0.63. Quest pages: 0.67; 0.378 and 0.00. Streaming head: 0.22; 0.408 and 0.00 | Sampling is unbiased, so it wins on aggregation, where top-k is biased. |

### SpeculativeShootout (W8 4.5: DistillSpec, 4.6: SpecTr multi-draft verification)
Sources: Run-w8, R-wave8. Test: SpeculativeShootoutTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DistillSpec, on-policy TVD; on-policy forward KL | Acceptance at T = 1; greedy; tokens per pass | TVD 0.628; 0.590; 2.27. FKL 0.574; 0.623; 2.06 | Draft trained on data: 0.564; 0.554; 2.01. Offline sequence KD: 0.570; 0.613; 2.26 | On-policy TVD gives the best sampled acceptance, as the paper predicts; offline KD ties it on tokens per pass. |
| SpecTr K-SEQ verification | Tokens per pass at k = 1 / 2 / 4 / 8 drafts; total variation from the target at k = 4 | 1.84 / 2.27 / 2.70 / 3.10; 0.0022 | Naive multi-draft acceptance: TV 0.1825. Plain sampling: TV 0.0023 | More drafts give more tokens per pass, and the output distribution stays exact. |

### ThroughputShootout (W8 4.7: S-LoRA and Punica SGMV multi-adapter serving, 4.8: fair scheduling with a virtual token counter)
Sources: Run-w8, R-wave8. Test: ThroughputShootoutTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| SGMV batching across adapters, 4 / 16 / 64 adapters | Tokens per iteration; p95 latency (iterations) | 23.4 / 25.3 / 25.9; 18 / 17 / 19 | Merged weights, per-adapter batches: 17.6 / 11.0 / 5.1; 83 / 263 / 707 | Throughput per iteration stays flat in the adapter count. CPU wall-clock tokens per second still favour merged per-adapter batches (96k vs 72k at 64 adapters). |
| VTC fair scheduler | Service gap; light / heavy client p95 (iterations); tokens per iteration | 904; 33 / 341; 23.59 | FCFS: 1,445; 287 / 341; 23.59. Request-rate limit: 347; 32 / 1,621; 8.57 | VTC protects light clients at no throughput cost; rate limiting is fairer but loses almost two thirds of the throughput. |

### StructuredOutputShootout (W8 4.9: XGrammar-style token-mask cache)
Sources: Run-w8, R-wave8. Test: StructuredOutputShootoutTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| GrammarMaskCache, lazy; precomputed | Mask time per token, JSON with BPE-512 and arithmetic with BPE-503 (cache hit rate) | Lazy 0.9 µs (0.98) and 3.1 µs (0.82). Precomputed 0.8 and 2.2 µs | Per-token scan: 62.0 and 68.3 µs. Core Vocabulary prefix scan: 28.0 and 17.2 µs | 8–77x faster than the scans, with identical outputs in every arm. |

---

## synthesis

New theme in wave 7 (package `experiments.synthesis`): program synthesis from input-output examples, and symbolic regression. The blocks are `GuidedEnumerator` and `SymbolicRegression` in `experimental.evolve`, and `ProgramGenerator` in `experimental.llm`.

### ProgramSynthesisShootout (W7 6.7: DeepCoder, 6.8: RobustFill)
Sources: Run-w7, R-wave7. Tests: ProgramSynthesisShootoutTest, RobustFillTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DeepCoder, probability-ordered / sort-and-add, 30 tasks of 2–5 primitives, 20k-candidate budget | Tasks solved (candidates tried) | Ordered 0.90 (3,357). Sort-and-add 0.73 (6,053) | Plain enumeration: 0.83 (4,486). Evolution: 0.90 (2,988, but 937 ms against 104). Library learner: 0.60 (8,148) | Ordering by predicted probability solves more with fewer candidates. Sort-and-add is worse than plain enumeration. |
| RobustFill, beam 1 / 8, string programs | Tasks solved | 0.07 / 0.20 | Induction baseline: 0.03 | Better than induction, but weak in absolute terms. |

### SymbolicRegressionBenchmark (W7 6.9: genetic programming with lexicase selection)
Sources: Run-w7, R-wave7.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Genetic programming with lexicase / tournament selection, four target equations | Equations recovered; extrapolation relative error | Lexicase 3 of 4 (misses the ratio, error 0.23). Tournament 4 of 4 (error 0) | MLP: 0 of 4 (0.045 to 0.96). NALU on the product: 0.045 | GP extrapolates exactly where networks cannot. Lexicase gives no edge over tournament selection here. |

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

### GanExperiment: min-max optimisers (W7 5.5: optimistic and extragradient methods on a GAN)
Sources: Run-w7, R-wave7. Test: GanMinMaxTest. The report describes a 400-step run (alternating 1.00, simultaneous 1.34, extragradient 1.10, optimistic 1.33); the test's 300-step numbers are given.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Optimistic Adam; extragradient Adam, WGAN-GP with 1:1 updates, 300 joint steps | Sliced W1 (start 0.897) | Optimistic 0.974. Extragradient 1.039 | Alternating Adam: 0.928. Simultaneous Adam: 1.013. Spectral-norm GAN reference: 0.393 | Negative: the gains on bilinear games (MatrixGameDynamics) do not carry over; no method improves on the start in 300 steps. |

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

### ForecastingBenchmark: statistical models and new deep baselines (W8 2.1: ETS and ARIMA, 2.5: N-HiTS, 2.10: TS2Vec)
Sources: Run-w8, R-wave8. Test: ForecastingBenchmarkExtendedTest. Lookback 48, horizon 12.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| ETS; ARIMA (state-space likelihood, AICc selection) | Test MSE | ETS 0.102. ARIMA 0.135 | Seasonal naive: 0.208. DLinear: 0.127. PatchTST: 0.178 | ETS is best overall. ARIMA needed a seasonal MA(1) term to beat seasonal naive, and in the report's run ARIMA and ETS (0.095 each) led every model (R-wave8). |
| N-HiTS; N-BEATS ensemble | Test MSE | N-HiTS 0.113. Ensemble 0.144 | N-BEATS: 0.211. DLinear: 0.127 | N-HiTS is the best deep model; ensembling closes most of N-BEATS's gap. |
| TS2Vec + ridge | Test MSE | 0.121 | Seasonal naive: 0.208. DLinear: 0.127 | A frozen contrastive encoder with a linear head beats DLinear. |

### ProbabilisticForecasting: state space, conformal and zero-shot (W8 2.1: ETS and ARIMA intervals, 2.7: Chronos-style zero-shot forecasting with KernelSynth and group attention, 2.8: CQR, ACI and EnbPI)
Sources: Run-w8, R-wave8. Tests: StateSpaceForecastingTest, ConformalForecastingTest, ZeroShotForecastingTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| ETS; ARIMA intervals | MSE; CRPS; 80% interval coverage | ETS 0.081; 0.178; 0.91. ARIMA 0.083; 0.176; 0.86 | DeepAR: 0.073; 0.168; 0.79. Quantile MLP: 0.094; 0.187; 0.83. DLinear: 0.106 (point only). Seasonal naive: 0.154; 0.244; 0.80 | ETS beats DLinear on its own generative family. DeepAR is best and best calibrated; ETS over-covers. |
| ACI (adaptive conformal inference), series with a distribution shift | 80% coverage before the shift / after it / late after it; mean width | 0.80 / 0.81 / 0.85; 1.54 | CQR: 0.90 / 0.61 / 0.68; 1.09. EnbPI: 0.87 / 0.58 / 0.76; 0.82. Split conformal: 0.86 / 0.37 / 0.43; 0.65. Quantile MLP: 0.93 / 0.61 / 0.68. ETS: 0.88 / 0.22 / 0.33. DeepAR: 0.75 / 0.33 / 0.35 | Only ACI recovers the target coverage after the shift, with wider intervals. |
| Group-attention forecaster (Chronos-2-style), zero-shot, with and without covariates; value-token LM (Chronos-style), test budget | MSE; CRPS; 80% coverage on the covariate series | Group attention 1.293; 0.717; 0.71. With covariates 1.297; 0.716; 0.65. Token LM 1.146; 0.726; 0.56 | Seasonal naive: 1.238; 0.819; 0.38. DeepAR trained on the series: 0.711; 0.581; 0.44. Quantile MLP: 1.344; 0.822. PatchTST: 1.446; 1.006 | At test budget the zero-shot models beat seasonal naive on CRPS and coverage, but DeepAR trained on the series wins. |
| Same, pretrained 4,000 steps on KernelSynth only (R-wave8) | MSE on seasonal / Mackey–Glass / covariate series | Group attention 0.138 / 0.036 / 0.732, and 0.673 with covariates. Token LM 0.60 / 0.058 / 0.945 | Seasonal naive: 0.072 / 0.065 / 0.761. DeepAR per dataset: 0.044 / 0.036 / 0.716 | Covariates help only where they matter, and there group attention has the best MSE of any model. It does not beat seasonal naive on the seasonal series, and the token LM stays weak at CPU budget. |

### LongHorizonForecasting (W8 2.2: Autoformer and FEDformer, 2.3: iTransformer, 2.4: TimesNet, 2.5: N-HiTS, 2.6: TiDE)
Sources: Run-w8, R-wave8. Tests: LongHorizonForecastingTest, MultivariateForecastingTest. The test numbers come from a smaller budget than the report's headline runs, and the rankings differ; both are given.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| Autoformer; FEDformer; TimesNet; vanilla transformer, lookback 96, H = 24 | Test MSE (training time) | Autoformer 0.212 (2.7 s). FEDformer 0.098 (1.8 s). TimesNet 0.076 (2.0 s). Transformer 0.155 (0.8 s) | DLinear: 0.036 (6 ms). N-HiTS: 0.031. Seasonal naive: 0.040. PatchTST: 0.049 | Negative at test budget: every transformer loses to DLinear and to seasonal naive. In the report's 120-step run they led (transformer 0.025, Autoformer 0.027, FEDformer 0.029, TimesNet 0.024, against DLinear 0.033), so the ranking depends on budget (R-wave8). |
| N-HiTS | Test MSE at H = 24 (parameters) | 0.031 (28,672) | N-BEATS: 0.076 (35,712). DLinear: 0.036 | Best model at H = 24, with less than half N-BEATS's error; 0.0236 vs 0.0424 at H = 96 (R-wave8). |
| iTransformer, multivariate series | MSE with coupled / independent channels | 0.391 / 0.610 | DLinear: 0.345 / 0.442. PatchTST: 0.470 / 0.540. FEDformer: 0.514 / 0.632 | It does relatively better on coupled channels, as expected, but DLinear wins both here. In the report's run iTransformer was best when coupled, 0.349 vs 0.379 (R-wave8). |
| TiDE with known future covariates | MSE; MSE on holidays; MSE after a price step | 0.126; 0.206; 0.281 | MLP + RevIN + future covariates: 0.239; 2.77; 0.222. DLinear: 0.279; 3.73; 0.304. PatchTST: 0.296. Seasonal naive: 0.397 | TiDE uses the holiday covariate, cutting holiday error 13x against the MLP with the same inputs. |
| TimesNet, two incommensurate periods | MSE | 0.304 | DLinear: 0.025. PatchTST: 0.068. TCN + RevIN: 0.751 | Negative: periods as 2-D images do not beat a linear map here (0.018 vs 0.011 in the report's run, R-wave8). |

### SeriesClassification (W8 2.9: ROCKET and MiniRocket, 2.10: TS2Vec, 2.17: wavelet scattering)
Sources: Run-w8, R-wave8. Tests: SeriesClassificationTest, SelfSupervisedSeriesTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| ROCKET; MiniRocket, CBF with 60 training series | Accuracy (fit time) | ROCKET 1.000 (448 ms). MiniRocket 0.995 (387 ms) | DTW 1-NN: 0.995 (384 ms). TCN: 0.860 (1.8 s). Raw + ridge: 0.910. MLP: 0.825. TimesNet: 0.335 | Random kernels match DTW and beat every trained network. |
| TS2Vec linear probe, shapelets, 10% labels | Accuracy | 0.987 | SimCLR probe: 1.000. ROCKET: 0.950. Random encoder: 0.840. Supervised TCN: 0.720. Raw + ridge: 0.563 | Contrastive pretraining beats supervised training with few labels, though SimCLR does as well. The report's run had the supervised TCN at 0.997 (R-wave8). |
| Scattering + ridge: `frequencies` task (n = 8); 2-D on ShapeImages with 64 images | Accuracy | 0.963; 0.910 | ROCKET: 0.997. Raw + ridge: 0.420. CNN on 64 images: 0.317. Pixels + ridge: 0.180 | A strong fixed feature on images with little data. On CBF it trails ROCKET, 0.865 vs 1.000. |

### AnomalyBenchmark (W8 2.11: Deep SVDD and Deep SAD, 2.12: LOF, one-class SVM and ECOD, 2.13: Anomaly Transformer and the point-adjustment pitfall, 2.14: matrix profile discords)
Sources: Run-w8, R-wave8. Test: AnomalyBenchmarkTest. The blocks are in `experimental.anomaly`.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| LOF | AUROC: tabular / + 20 noise features / local density | 0.995 / 0.898 / 0.999 | kNN distance: 0.989 / 0.891 / 0.896. Isolation forest: 0.959 / 0.673 / 0.927. Autoencoder: 0.994 / 0.876 / 0.755 | LOF is best on all three and wins clearly on local density. |
| One-class SVM; ECOD | Same | One-class SVM 0.966 / 0.512 / 0.810. ECOD 0.656 / 0.574 / 0.703 | LOF: 0.995 / 0.898 / 0.999 | The SVM breaks under noise features, and ECOD is weak on correlated anomalies, a real finding rather than a bug (R-wave8). |
| Deep SVDD; Deep SAD (1% labels) | Same; AUROC on one-class images | SVDD 0.829 / 0.764 / 0.379; images 0.848. SAD 0.856 / 0.786 / 0.378 | kNN: 0.989 / 0.891 / 0.896; images 0.963. Isolation forest on images: 0.592 | Negative: the deep one-class models trail kNN and LOF and fail on local density. A few labels help a little. |
| Anomaly Transformer | F1; point-adjusted F1; range-AUC | 0.39; 0.85; 0.656 | Random scores: 0.23; 0.86; 0.499. Isolation forest on windows: 0.70; 0.73; 0.887 | Negative, and the point-adjustment pitfall: random scores reach 0.86 point-adjusted F1, more than the Anomaly Transformer. |
| Matrix profile: AB-join against a clean prefix; self-join | F1; range-AUC | AB-join 0.86; 0.926. Self-join 0.65; 0.886 | Isolation forest on windows: 0.70; 0.887. Autoencoder: 0.60; 0.873. Deep SVDD on windows: 0.56; 0.868 | A parameter-free discord search is the best series detector. The discord hit rate is 0.63. |

### ChangePoints (W8 2.15: Bayesian online change-point detection and PELT)
Sources: Run-w8, R-wave8. Test: ChangePointsTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| PELT; BOCPD | F1; detection delay | PELT 0.67; 0.5. BOCPD 0.39; 1.8 | HMM Viterbi: 0.43; 0.0. Single-change model: 0.23. Kalman innovations: 0.13 | PELT wins. BOCPD trails the HMM on F1 but catches more variance changes (recall 0.56, against 0.39 for PELT and 0.00 for the HMM). |

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

### DetectionShootout (W8 1.1: RetinaNet-lite with focal loss and a feature pyramid, 1.2: CenterNet, 1.3: DETR-lite and DN-DETR, 1.4: instance segmentation with a discriminative embedding loss)
Sources: Run-w8, R-wave8. Tests: DetectionFocalLossTest, DetectionPyramidTest, DetectionBoxLossTest, DetectionBaselinesTest, DetectionCenterNetTest, DetrLearningCurveTest, DnDetrLearningCurveTest, InstanceEmbeddingTest. 24×24 scenes from `experimental.data.ShapeScenes`, one seed.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| RetinaNet-lite with focal loss, 250 steps | mAP@0.5; mAP@[.5:.95] (multiply-adds) | 0.163; 0.068 (1.42M) | Cross-entropy: 0.048; 0.014. Sliding-window CNN: 0.684; 0.388 (8.5M). U-Net + connected components: 0.013; 0.010 | Focal loss triples mAP over cross-entropy, but a brute-force sliding window wins at 6x the compute. |
| Feature pyramid | mAP@0.5, all / small shapes | 0.163 / 0.222 | Coarsest map only: 0.009 / 0.000 (0.37M multiply-adds) | The pyramid is what finds small objects. |
| GIoU box loss | mAP@0.5; mAP@[.5:.95] | 0.163; 0.068 | Smooth L1: 0.106; 0.031 | GIoU is better. |
| CenterNet, 300 steps | mAP@0.5, sparse / crowded scenes (decode time per image) | 0.505 / 0.442 (0.93 ms) | RetinaNet-lite on crowded scenes: 0.118 (1.38 ms) | No anchors or NMS, and nearly 4x RetinaNet's mAP on crowded scenes. |
| DETR-lite; DN-DETR (denoising queries) | mAP@0.5 at 300 / 600 / 900 steps | DETR 0.000 / 0.002 / 0.020. DN-DETR 0.001 / 0.001 / 0.005 | CenterNet: 0.505 at 300 steps | Negative: DETR is the slowest to converge, as predicted, and denoising queries do not speed it up here. At 1,500 steps both reach about 0.05 (R-wave8). |
| Discriminative embedding loss, touching same-class shapes | Mask AP@0.5; FG-ARI | 0.199; 0.400 | U-Net + connected components: 0.352; 0.268. Slot attention (ObjectDiscovery): FG-ARI 0.303 | Separates touching instances better (FG-ARI) but loses on mask AP. |

### DensePrediction (W8 1.5: DeepLab ASPP with a segmentation-loss sweep, 1.6: Horn–Schunck and Lucas–Kanade, 1.7: RAFT-lite, 1.8: cost-volume stereo, 1.9: integral keypoint regression)
Sources: Run-w8, R-wave8. Tests: AtrousSegmentationTest, SegmentationLossTest, DiceSegmentationTest, ClassicalFlowTest, FlowRegressionTest, FlowUNetTest, RaftFlowTest, StereoCostVolumeTest, StereoUNetTest, KeypointHeadsTest, plus the block test VisionBlocksTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| DeepLab-lite (ASPP), Lovász loss, 120 steps | mIoU (parameters, multiply-adds) | 0.564 (13,639, 870k) | U-Net: 0.738 (32,575, 1.18M) | Negative: ASPP does not match the U-Net. The test was loosened to ASPP > 0.4 (R-wave8). |
| Lovász loss, DeepLab-lite | mIoU; pixel accuracy | 0.564; 0.973 | Cross-entropy: 0.025; 0.946. Soft Dice: 0.000; 0.942 (collapses to background) | Optimising IoU directly is what makes the small segmenter work. |
| Horn–Schunck; Lucas–Kanade, motion of 0–4 px | End-point error on moving pixels (all pixels) | HS 0.752 (1.026). LK 1.439 (2.090) | Zero flow: 1.721 (0.523). CNN regressor: 1.186 (0.481) | Horn–Schunck is best on moving pixels but hurts static ones. |
| RAFT-lite, motion of 6–10 px | Moving-pixel EPE after 1 / 2 / 4 / 8 iterations | 4.320 / 4.115 / 4.109 / 4.114 | Zero flow: 6.478. Horn–Schunck: 5.155. FlowNet-style U-Net: 5.386. Lucas–Kanade: 7.445 | Best on large motion, converging in about 2 iterations. On small motion it stays near zero-flow level (R-wave8). |
| Cost volume + soft-argmin stereo | Disparity MAE; share off by more than 1 px | 0.226; 7.1% | Block matching: 0.350; 8.8%. U-Net on the stacked pair: 0.687; 24.8% | Explicit matching beats direct regression 3x. |
| Integral (soft-argmax) keypoint regression | PCK@2 px; mean error; rare-pose error | 0.548; 3.63; 5.66 | Heatmap arg-max: 0.314; 4.94; 5.93. Direct regression: 0.428; 5.47; 9.25 | Best on every measure. |

### VideoTracking (W8 1.10: SORT and ByteTrack, 1.11: ConvLSTM and SimVP)
Sources: Run-w8, R-wave8. Tests: MultiObjectTrackingTest, SimVpPredictionTest, ConvLstmPredictionTest, UNetPredictionTest. Detections are ground truth plus noise and dropout (R-wave8).

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| ByteTrack; SORT, 20 videos | MOTA; IDF1; ID switches | ByteTrack 0.900; 0.855; 41. SORT 0.826; 0.788; 58 | SORT without the Kalman filter: 0.801; 0.692; 128. Greedy centroid matching: 0.804; 0.744; 105 | The Kalman prediction halves ID switches, and ByteTrack's second matching round helps further. |
| SimVP; ConvLSTM, video prediction | MSE at horizon 1 / 4 (time per step) | SimVP 0.0102 / 0.0233 (43 ms). ConvLSTM 0.0105 / 0.0276 (91 ms) | Copy the last frame: 0.0121 / 0.0413. U-Net rollout: 0.0072 / 0.0314 (43 ms) | SimVP is best at horizon 4 at half ConvLSTM's cost; the U-Net is best one step ahead. |

### SceneFields (W8 1.12: NeRF, 1.13: Instant-NGP multiresolution hash encoding, 1.14: SIREN and neural signed distance functions)
Sources: Run-w8, R-wave8. Tests: NerfVsSplatsTest, NerfEncodingTest, HashGridImageTest, HashGridTableSizeTest, SirenImageTest, SdfSirenTest, SdfFourierTest, plus the block test GeometryBlocksTest. NovelViewSynthesis's 16×16 views, 8 for training and 8 held out.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| NeRF (positional encoding, hierarchical sampling) | Held-out PSNR; training time; render time per view | 21.50 dB; 4.8 s; 29.6 ms | 3D Gaussian splatting: 25.68 dB; 1.3 s; 1.4 ms. All-black image: 16.84 dB | Negative: NeRF is 4 dB below splatting and about 20x slower to render. In the report's longer run it reaches 23.45 dB, with no gain from hierarchical sampling (R-wave8). |
| Hash-grid encoding inside NeRF | Held-out / training PSNR | 20.90 / 21.71 dB | Raw xyz: 19.20 / 19.35 dB | The hash grid helps. In the report's run it reaches 24.89 dB held out but 34.07 dB on the training views, so it overfits 8 views (R-wave8). |
| Multiresolution hash grid, 48×48 picture, 150 steps | PSNR (time) | 43.29 dB (1.4 s) | Fourier features: 24.95 dB (2.4 s). SIREN: 17.92 dB. Raw MLP: 11.54 dB | The hash grid fits far faster. Table size 2^8 / 2^10 / 2^12 / 2^14 gives 34.23 / 39.77 / 43.29 / 44.89 dB, so collisions hurt below 2^12. |
| SIREN signed distance auto-decoder, 6 outlines, 100 steps | Chamfer; zero-set IoU; eikonal residual | 0.166; 0.512; 0.736 | ReLU MLP: 0.028; 0.904; 0.748. Fourier features: 0.143; 0.179; 1.469 | Negative: the ReLU network wins, and SIREN's eikonal residual is no better. SIREN uses ω0 = 10, since 30 did worse (R-wave8). |

### GeometricVision (W8 1.15: PointNet and DGCNN, 1.16: group-equivariant CNNs, 1.17: spatial transformer networks)
Sources: Run-w8, R-wave8. Tests: PointCloudTest, DgcnnTest, EgnnRotationTest, GroupConvRotationTest, GroupConvLowDataTest, StnDistortionTest, DeformableDistortionTest, GcnnDistortionTest.

| Technique | Metric | Result | Baseline | Takeaway |
|---|---|---|---|---|
| PointNet (with and without a T-Net); DGCNN, 16-point clouds of 5 shapes | Accuracy upright / rotated | PointNet 0.975 / 0.275, the same with a T-Net. DGCNN 0.945 / 0.280 | Deep Sets: 0.890 / 0.440. Sorted-coordinate MLP: 0.640 / 0.315. EGNN: 0.535 / 0.475 | Negative for DGCNN: it does not beat PointNet and takes 7x longer. Only EGNN ignores rotation, but it underfits. |
| G-CNN (p4), trained upright | Accuracy upright / rotated (parameters) | 0.990 / 0.990 (4,772) | CNN: 0.973 / 0.670 (4,932). CNN + rotation augmentation: 0.928 / 0.930 | Built-in equivariance beats augmentation. With 64 training images: 0.990 rotated, against 0.918 augmented and 0.603 plain. |
| STN + CNN, distorted glyphs | Accuracy | 0.783 | CNN: 0.715. Deformable CNN at 120 steps: 0.758, against 0.600 for the CNN (0.875 at 300 steps, R-wave8). G-CNN at 150 steps: 0.448, against 0.633 for the CNN | The STN helps, but the deformable CNN is the stronger rival. The G-CNN is worse than a plain CNN on non-rotational distortion. |

---

## Wave 9

Wave 9 adds 102 techniques in six work packages (WP1–WP6, items 9.1.1 to 9.6.17 in `wave9-ideas.md`). An item is tagged **W9** in the same way as earlier waves, so "W9 1.3" is item 9.1.3. The code is merged on branch `claude/project-thread-8zvcl0`. It adds ten new themes (alignment, causal, code, document, edge, evaluation, parallel, streaming, survival and world) and extends nine existing ones (agents, audio, data, interp, multimodal, reasoning, recsys, sciml and vision). Unlike earlier waves, wave 9 is gathered here in one section per work package rather than spread over the themes; the theme of each experiment is given on its Sources line.

Every wave 9 number is keyed **R-wave9**. It comes from the final report of the work-package agent that built the item. Each agent ran its own test classes one at a time with `--tests` filters in the cloud container on 2026-11-18, and all of them passed. No full suite run of the merged wave 9 work has been made yet, so there is no Run-w9 key. Several agents sent a first report and a follow-up after fixing set-ups and bugs. Where the two disagree, the follow-up is used. Most rows are one seed at a test-sized budget, so gaps of a few points are within noise. Where a number comes from a longer or earlier run than the test's, the row says so.

The **Verdict** column reads as follows:
- **Win:** the technique beats its baseline clearly.
- **Mixed:** it wins on some measures or settings and loses on others.
- **Negative:** it does not beat its baseline, or the hypothesis in `wave9-ideas.md` did not hold. These rows are kept and labelled.
- **Inconclusive:** the test is only a smoke run, so the numbers cannot decide.

### Wave 9 verdicts by item

| WP | Win | Mixed | Negative | Inconclusive | Items |
|---|---|---|---|---|---|
| WP1 ranking, causal, survival | 8 | 7 | 3 | 0 | 18 |
| WP2 post-training and code | 4 | 4 | 9 | 0 | 17 |
| WP3 data and evaluation | 7 | 6 | 4 | 0 | 17 |
| WP4 interpretability, documents, multimodal | 4 | 5 | 8 | 0 | 17 |
| WP5 world models, video, audio, sciml | 6 | 1 | 7 | 2 | 16 |
| WP6 parallel, streaming, edge | 9 | 6 | 2 | 0 | 17 |
| **Total** | **38** | **29** | **33** | **2** | **102** |

| Item | Technique | Experiment | Verdict |
|---|---|---|---|
| W9 1.1 | MMoE and PLE | MultiTaskRanking | Mixed |
| W9 1.2 | ESMM | MultiTaskRanking | Negative |
| W9 1.3 | Unbiased learning to rank (IPW, regression EM, DLA) | ClickLogLearning | Mixed |
| W9 1.4 | MMR and DPP re-ranking | RerankingDiversity | Win |
| W9 1.5 | AutoInt and xDeepFM | ClickModels | Mixed |
| W9 1.6 | RankMixer | ClickModels | Win |
| W9 1.7 | Hashed and quotient–remainder embeddings | ClickModels | Win |
| W9 1.8 | logQ-corrected in-batch softmax, mixed negatives | SamplingBiasCorrection | Win |
| W9 1.9 | Budget pacing: throttling and bid shading | BudgetPacing | Mixed |
| W9 1.10 | X-, R- and DR-learners | HeterogeneousEffects | Win |
| W9 1.11 | Double machine learning | HeterogeneousEffects | Mixed |
| W9 1.12 | Causal forest | HeterogeneousEffects | Negative |
| W9 1.13 | Uplift models and Qini | HeterogeneousEffects | Win |
| W9 1.14 | 2SLS and DeepIV | InstrumentalVariables | Mixed |
| W9 1.15 | CUPED and mSPRT | OnlineExperiments | Win |
| W9 1.16 | Kaplan–Meier, Cox, DeepSurv | SurvivalShootout | Win |
| W9 1.17 | Discrete hazards, DeepHit, survival forest | SurvivalShootout | Mixed |
| W9 1.18 | PFN at longer budgets, TabICL-lite | TabularShootout | Negative |
| W9 2.1 | Reward-model ensembles and WARM | RewardOveroptimization | Negative |
| W9 2.2 | BOND and J-BOND | RewardOveroptimization | Mixed |
| W9 2.3 | SPPO and Nash-MD | GeneralPreferences | Negative |
| W9 2.4 | General preference model (GPM) | GeneralPreferences | Win |
| W9 2.5 | χ²PO and SquaredPO | PreferenceShootout | Win |
| W9 2.6 | Length-regularised DPO (R-DPO) | PreferenceShootout | Negative |
| W9 2.7 | XPO | PreferenceShootout | Negative |
| W9 2.8 | VinePPO | ProcessSupervision | Mixed |
| W9 2.9 | Math-Shepherd | ProcessSupervision | Negative |
| W9 2.10 | Process advantage verifiers (PAV) | ProcessSupervision | Mixed |
| W9 2.11 | PRIME | ProcessSupervision | Negative |
| W9 2.12 | Step-DPO | ProcessSupervision | Negative |
| W9 2.13 | RLVR from one example | ArithmeticRl | Negative |
| W9 2.14 | Execution prediction from traces | CodeExecution | Win |
| W9 2.15 | Self-repair | CodeExecution | Negative |
| W9 2.16 | Execution-based reranking | CodeExecution | Win |
| W9 2.17 | ArCHer with a tuned critic | ToolAgentBench | Mixed |
| W9 3.1 | Online data mixing (ODM) | DataCuration | Negative |
| W9 3.2 | Aioli | DataCuration | Negative |
| W9 3.3 | Skill-It | DataCuration | Mixed |
| W9 3.4 | Learning-progress curricula | DataCuration | Negative |
| W9 3.5 | RHO-LOSS | DataCuration | Win |
| W9 3.6 | Dataset distillation | SyntheticTraining | Mixed |
| W9 3.7 | TabDDPM and CTGAN | SyntheticTraining | Mixed |
| W9 3.8 | Classifiers on diffusion samples | SyntheticTraining | Negative |
| W9 3.9 | Continual pretraining | ContinualPretraining | Mixed |
| W9 3.10 | Calibration estimators | CalibrationEstimators | Win |
| W9 3.11 | LLM-as-judge biases | JudgeAudit | Win |
| W9 3.12 | Rogan–Gladen and prediction-powered inference | JudgeAudit | Win |
| W9 3.13 | Kernel Divergence Score, paraphrase contamination | ContaminationAudit | Mixed |
| W9 3.14 | Clustered standard errors, paired tests, power | StatisticalReporting | Win |
| W9 3.15 | Ladder and Thresholdout | StatisticalReporting | Win |
| W9 3.16 | Rankings across training budgets | BudgetRankings | Win |
| W9 3.17 | Learning-curve extrapolation, LC-PFN | CurveExtrapolation | Mixed |
| W9 4.1 | ACDC and edge attribution patching | CircuitDiscovery | Win |
| W9 4.2 | Distributed alignment search | CircuitDiscovery | Win |
| W9 4.3 | INLP and LEACE | ConceptErasure | Win |
| W9 4.4 | CKA, SVCCA, PWCCA | RepresentationComparison | Negative |
| W9 4.5 | Contrast-consistent search | LatentKnowledge | Negative |
| W9 4.6 | Attention rollout and flow | ExplainabilityBenchmark | Negative |
| W9 4.7 | Git Re-Basin | RepresentationComparison | Win |
| W9 4.8 | Local learning coefficient | DevelopmentalInterp | Mixed |
| W9 4.9 | TRAK | TrainingAttributionExperiment | Negative |
| W9 4.10 | CRNN with CTC | GlyphOcr | Negative |
| W9 4.11 | Donut-style parsing | DocumentParsing | Negative |
| W9 4.12 | LayoutLM-style 2-D positions | DocumentParsing | Mixed |
| W9 4.13 | TAPAS | TableQa | Mixed |
| W9 4.14 | DETR with hybrid matching | DetectionShootout | Mixed |
| W9 4.15 | Frozen-LM bridges (projector, Q-Former, gated cross-attention) | FusionShootout | Negative |
| W9 4.16 | Transfusion | FusionShootout | Negative |
| W9 4.17 | Bottleneck fusion, modality dropout | FusionShootout | Mixed |
| W9 5.1 | Factorised space-time video diffusion | VideoGeneration | Negative |
| W9 5.2 | MAGVIT-style 3-D tokenizer with masked modelling | VideoGeneration | Win |
| W9 5.3 | Self Forcing | VideoGeneration | Negative |
| W9 5.4 | History guidance | VideoGeneration | Win |
| W9 5.5 | IRIS token world model | WorldModelShootout | Mixed |
| W9 5.6 | DreamerV3 robustness tricks | WorldModelShootout | Negative |
| W9 5.7 | DINO-WM | WorldModelShootout | Win |
| W9 5.8 | SlotFormer | WorldModelShootout | Negative |
| W9 5.9 | HiFi-GAN vocoder | AudioGeneration | Inconclusive |
| W9 5.10 | DiffWave and WaveGrad | AudioGeneration | Inconclusive |
| W9 5.11 | Monotonic alignment search vs filler padding | AudioGeneration | Negative |
| W9 5.12 | Latent audio diffusion | AudioGeneration | Win |
| W9 5.13 | Finite scalar quantization | AudioGeneration, VideoGeneration | Win |
| W9 5.14 | PINN fixes for hard PDEs | PdeShootout | Negative |
| W9 5.15 | ENGD and Newton–CG for PINNs | PdeShootout | Win |
| W9 5.16 | Transolver and LANO vs FNO | OperatorLearning | Negative |
| W9 6.1 | GPipe, 1F1B, interleaved 1F1B | ParallelismShootout | Win |
| W9 6.2 | Zero-bubble pipelines and DualPipe | ParallelismShootout | Mixed |
| W9 6.3 | Tensor and sequence parallelism | ParallelismShootout | Win |
| W9 6.4 | ZeRO 1–3 | ParallelismShootout | Win |
| W9 6.5 | Selective recomputation, optimal checkpointing | MemorySchedules | Win |
| W9 6.6 | Expert parallelism with capacity limits | ParallelismShootout | Win |
| W9 6.7 | Hogwild and delay-compensated ASGD | AsyncTraining | Win |
| W9 6.8 | FTRL-Proximal | StreamShootout | Mixed |
| W9 6.9 | Hoeffding trees, adaptive random forest | StreamShootout | Mixed |
| W9 6.10 | ADWIN, DDM, EDDM, Page–Hinkley | StreamShootout | Mixed |
| W9 6.11 | ER-ACE | StreamShootout | Mixed |
| W9 6.12 | COCOB coin betting | StreamShootout | Win |
| W9 6.13 | Integer-only inference | EdgeDeployment | Mixed |
| W9 6.14 | Patch-based inference, operator reordering | EdgeDeployment | Win |
| W9 6.15 | Once-for-all networks | EdgeDeployment | Negative |
| W9 6.16 | On-device training with sparse updates | EdgeDeployment | Win |
| W9 6.17 | Keyword spotting (DS-CNN and rivals) | EdgeDeployment | Negative |

---

## Wave 9 WP1: ranking, ads and search; causal inference; survival analysis

New packages: `experimental.causal`, `experimental.survival`, `experiments.causal` and `experiments.survival`. The other WP1 experiments extend `experiments.recsys`. The tests ran on a machine with a load average of 18 on 4 cores, so their times are inflated (R-wave9).

### MultiTaskRanking (W9 1.1: MMoE and PLE, 1.2: ESMM)
Sources: R-wave9. Theme: recsys. Test: MultiTaskRankingTest. Blocks: `experimental.recsys.MultiGateExperts` (shared bottom, MMoE, PLE) and `EntireSpaceModel`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| MMoE; PLE, task correlation ρ = −0.5 | AUC, task 1 / task 2 | MMoE 0.843 / 0.851. PLE 0.843 / 0.848 | Shared bottom: 0.837 / 0.831. Single-task: 0.840 / 0.838. DCN-V2 with two heads: 0.840 / 0.828. FineGrainedMoe: 0.820 / 0.816 | Mixed | The gates help on task 2 (+0.02 over shared bottom); on task 1 every gap is within noise. |
| ESMM (shared bottom; MMoE), clicks made rare (about 7% CTR) | CVR AUC | 0.709; 0.714 | Click-space CVR, shared bottom: 0.787. Click-space MMoE: 0.784 | Negative | Negative: the data has no unobserved selection bias, and the click-space models already share a body with the CTR task, so ESMM has nothing to fix. This matches the ESCM² critique. The test was loosened to a sanity check. |

### ClickLogLearning (W9 1.3: unbiased learning to rank with IPW, regression EM and the dual learning algorithm)
Sources: R-wave9. Theme: recsys. Test: ClickLogLearningTest. Block: `ClickDebiasing`. The logging ranker also promotes an irrelevant feature, so naive click models learn the wrong signal. This set-up was fixed in the follow-up; the first version gave position bias no effect.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| IPW with propensities clipped at 0.1; DLA; cascade correction | NDCG@5 at position-bias η = 2 | IPW 0.898. DLA 0.817. Cascade 0.803 | Naive RankNet: 0.783. LambdaRank: 0.790. Softmax: 0.793. LambdaMART: 0.765. Oracle on true grades: 0.935 | Mixed | Clipped IPW closes most of the gap to the oracle; DLA gains only about 0.02. |
| Unclipped IPW; regression-EM ranker | NDCG@5 | 0.738; 0.728 | Naive softmax: 0.793 | Negative | Negative: weights up to 100 make both lose to naive training. EM still estimates the propensities well (error 0.012). |

### RerankingDiversity (W9 1.4: MMR and determinantal point processes)
Sources: R-wave9. Theme: recsys. Test: RerankingDiversityTest. Block: `DiverseReranker` (MMR; DPP with fast greedy Cholesky). Candidates come from a noisy relevance oracle.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| DPP (θ = 0.7); MMR (λ = 0.5) | NDCG@10; α-NDCG@10; intra-list distance | DPP 0.845; 0.876; 0.521. MMR 0.878; 0.852; 0.431 | Relevance only: 0.930; 0.717; 0.240 | Win | Both trade a little relevance for much more diversity; the DPP buys the most. |

Not done: LightGCN candidates were replaced by the noisy relevance oracle (the two-tower candidate variant is kept).

### ClickModels: interaction order, RankMixer and compact embeddings (W9 1.5: AutoInt and xDeepFM, 1.6: RankMixer, 1.7: hashing and quotient–remainder embeddings)
Sources: R-wave9. Theme: recsys. Tests: InteractionOrderTest, RankMixerSweepTest, CompactEmbeddingTest. Blocks: `FieldAttention` (AutoInt), `CompressedInteraction` (xDeepFM), `RankMixerBlock`, `CompactEmbedding`. Budgets: 160, 100 and 100 steps.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| xDeepFM; AutoInt, planted order-3 interaction | AUC | xDeepFM 0.715. AutoInt 0.674 | DCN-V2: 0.714. DeepFM: 0.713. Wide & Deep: 0.708. FM: 0.672. Logistic: 0.592. FT-Transformer (one token per field): 0.566 | Mixed | xDeepFM ties the best baseline; AutoInt falls behind DeepFM. |
| RankMixer, same task | AUC | 0.743 | As above | Win | Best model on the planted-order task. |
| RankMixer dense; sparse MoE, width 16 | AUC (ms per batch) | Dense 0.643 (3.6). MoE 0.678 (5.6), 1.91 of 4 experts active | FT-Transformer: 0.575 (9.2) | Win | Better and faster than the FT-Transformer at the same width. |
| Quotient–remainder; frequency threshold; hashed, 10% memory, 100k-ID Zipf field | AUC / tail-ID AUC | QR 0.686 / 0.526. Threshold 0.686 / 0.516. Hashed 0.673 / 0.501. Low-rank (12.5%) 0.646 / 0.521 | Full table: 0.685 / 0.515 | Win | Quotient–remainder matches the full table at a tenth of the memory. |

Not done: no HSTU row in the RankMixer sweep.

### SamplingBiasCorrection (W9 1.8: logQ-corrected in-batch softmax and mixed negative sampling)
Sources: R-wave9. Theme: recsys. Test: SamplingBiasCorrectionTest. Block: `CorrectedInBatchSoftmax`, with a streaming frequency estimator. Recall is measured against each user's true top 20 under the generating affinity, not held-out clicks.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| logQ correction; mixed negatives | Recall@20 all / head / tail | logQ 0.267 / 0.272 / 0.252. Mixed 0.369 / 0.390 / 0.305 | Plain in-batch: 0.115 / 0.086 / 0.206. Full softmax: 0.504 / 0.580 / 0.271. BPR-MF: 0.493 / 0.607 / 0.140. TwoTowerRetrieval: 0.112 | Win | Correction triples recall over plain in-batch, and mixed negatives add more. Full softmax and BPR-MF are still ahead overall but weaker on the tail. The frequency estimate's total-variation error is 0.49. |

### BudgetPacing (W9 1.9: throttling and Lagrangian bid shading)
Sources: R-wave9. Theme: recsys. Test: BudgetPacingTest. Block: `BudgetPacer`, with a second-price auction simulator. The hindsight optimum is an exact greedy fractional knapsack, cross-checked against `LinearProgram.maximize` on 150 auctions.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Dual-gradient bid shading | Value as a share of the hindsight optimum | 0.892 | Truthful bidding: 0.709, out of budget at 22% of the day | Win | Bid shading spends the budget evenly and gets most of the hindsight value. |
| PID throttle | Same | 0.421 | Fixed throttle (rate tuned with hindsight): 0.454 | Negative | Negative: the untuned PID gains do worse than the fixed rate; the test only asks for 90% of it. |

Item verdict: Mixed.

### HeterogeneousEffects (W9 1.10: X-, R- and DR-learners, 1.11: double machine learning, 1.12: causal forests, 1.13: uplift modelling)
Sources: R-wave9. Theme: causal. Tests: HeterogeneousEffectsTest, DoubleMlTest, CausalForestTest, UpliftTest. Blocks: `MetaLearners`, `DoubleMl`, `CausalForest`, `UpliftModels`, `UpliftMetrics`, `experimental.ml.RegressionForest`. Data: `experimental.data.CausalScenarios`, n = 1,000.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| R-learner; DR-learner, forest nuisance models and a ridge final stage | PEHE, simple effect | R 0.352. DR 0.467. X 0.458 | T-learner: 0.738. Linear T: 0.523. Causal forest: 0.507 | Win | Nie and Wager's point holds: the R-learner is best. With a 20-tree forest as the final stage, R scores 0.806 and DR 1.010, worse than T. |
| X-learner, 10% treated | PEHE | 0.550 | T-learner: 1.268 | Win | The X-learner handles unbalanced arms, as predicted. |
| TARNet; DragonNet, early stopping on factual validation MSE | PEHE | 0.94; 0.96 | Forest meta-learners, above | Negative | Negative (mild): the neural learners trail the forests at n = 1,000. Without early stopping they overfit (2.04 at 1,000 steps). |
| DML, interactive model with boosting; with forests; partially linear | ATE bias / 95% interval coverage | Boosted 0.32 / 0.67. Forest 0.43 / 0.67. PLR 0.53 / 0.00 | Naive: 1.80. IPW: 1.69. AIPW without cross-fitting: 0.96 / 0.00 | Mixed | Cross-fitting cuts the bias by two thirds, but the intervals under-cover badly at this sample size. |
| Causal forest (GRF-style, honest), step effect | PEHE; 90% interval coverage | 0.563; 0.498 | R-learner: 0.473. T-learner: 0.544 | Negative | Negative: no better than the meta-learners, and its intervals cover half as often as they should. |
| Uplift forest with KL splits | Qini | 0.0409 | Oracle: 0.0424. X-learner: 0.0403. Response model: −0.0078 | Win | Near the oracle; targeting by response alone is worse than random. |

Item verdicts: 1.10 Win (TARNet and DragonNet are comparison rows), 1.11 Mixed, 1.12 Negative, 1.13 Win.

### InstrumentalVariables (W9 1.14: two-stage least squares and DeepIV)
Sources: R-wave9. Theme: causal. Test: InstrumentalVariablesTest. Block: `InstrumentalVariables`. Data: an airline-style instrument with nonlinear confounding.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| DeepIV | Counterfactual relative MSE | 0.033 | Direct MLP (confounded): 0.036. Forest covariate adjustment: 0.097. 2SLS: 0.242 | Mixed | DeepIV beats linear 2SLS by far but only just beats the confounded direct MLP. |

Not done: the AIPW baseline was replaced by forest covariate adjustment.

### OnlineExperiments (W9 1.15: CUPED and always-valid sequential tests)
Sources: R-wave9. Theme: causal. Test: OnlineExperimentsTest. Block: `OnlineExperiments` (t-test, CUPED, CUPAC, mSPRT).

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| mSPRT, A/A tests | False-positive rate | 0.005 | Fixed-horizon t-test: 0.060. t-test peeked daily: 0.255 | Win | The always-valid test stops peeking from inflating false positives. The CUPED power numbers were not in the report. |

### SurvivalShootout (W9 1.16: Kaplan–Meier, Cox and DeepSurv, 1.17: discrete-time hazards, DeepHit and random survival forests)
Sources: R-wave9. Theme: survival. Tests: SurvivalShootoutTest, CrossingHazardsTest. Blocks: `SurvivalModels`, `DiscreteHazardNet`, `DeepHit`, `SurvivalForest`, `SurvivalMetrics`. Data: `experimental.data.SurvivalData`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| DeepSurv, nonlinear risk | Harrell's C | 0.775 | Cox: 0.585 | Win | The neural risk function captures what the linear Cox model cannot. |
| Random survival forest; DeepHit, crossing hazards | Time-dependent C | RSF 0.659. DeepHit 0.656 | Cox: 0.617 | Win | Both beat Cox when proportional hazards fails. |
| DeepHit, competing risks | CIF error, cause 1 | 0.081 | Cause-specific Cox: 0.037 | Negative | Negative: cause-specific Cox is twice as accurate. |

Item verdicts: 1.16 Win, 1.17 Mixed.

### TabularShootout: longer pre-training and TabICL-lite (W9 1.18: prior-fitted networks with a longer budget, TabICL's column-then-row architecture)
Sources: R-wave9. Theme: recsys. Test: PretrainingBudgetTest. Block: `ColumnRowIcl`. The wave 7 `priorFitted` rows are unchanged.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| PFN at 1x / 4x pre-training (12 / 48 steps); TabICL-lite at both | Prior-task accuracy | PFN 0.480 / 0.640. TabICL-lite 0.508 / 0.508 | GBT: 0.830. Random forest: 0.843 | Negative | Negative: far behind the trees at test budgets. In a 300-step probe TabICL-lite sits on a ln 2 loss plateau until about step 150, falls to 0.59, then turns unstable: a slow start, not a bug. |

Not done: the 16x and 64x rows exist in the experiment but were not run, so the budget claim is untested.

---

## Wave 9 WP2: post-training, process rewards and code models

New packages: `experimental.align`, `experimental.code`, `experiments.alignment` and `experiments.code`. Block tests AlignBlocksTest and CodeBlocksTest check closed-form answers. All 16 WP2 test classes pass in 10.5 s or less. In the follow-up the R-DPO length term's sign was fixed, and several set-ups were reworked because their baselines sat near zero (R-wave9).

### RewardOveroptimization (W9 2.1: reward-model ensembles and WARM, 2.2: BOND)
Sources: R-wave9. Theme: alignment. Tests: RewardOveroptimizationTest, RewardKlSweepTest, BondTest. Blocks: `RewardEnsemble`, `WeightAveragedReward`, `BestOfNDistiller`. The gold reward is `RubricRewards.trueScore`; preference labels come from a noisy Bradley–Terry annotator. Three ensemble members and three WARM fine-tunes, each reward model trained for 40 steps.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Ensembles (mean; worst; mean − std); WARM | Gold reward at best-of-64 | 0.451; 0.498; 0.487. WARM 0.528 | Single proxy: 0.524 / 0.526 / 0.545 / 0.500 at best-of-1 / 4 / 16 / 64. Gold oracle: 0.524 / 0.744 / 0.882 / 0.957 | Negative | Negative: proxy accuracy is barely above chance (single 0.586, WARM 0.566), so ensembling has nothing to average. WARM degrades least. |
| Same, final gold after GRPO without a KL penalty | Gold reward | Mean 0.333. Worst 0.393. Mean − std 0.403. WARM 0.491 | Single proxy: 0.521 → 0.487. Gold oracle: 0.727 | Negative | Over-optimisation shows clearly: every proxy lowers gold while the oracle climbs. A KL penalty of 0 / 0.05 / 0.2 gives 0.445 / 0.459 / 0.444; PPO 0.499. |
| BOND; J-BOND | Gold reward, one sample per query | 0.525; 0.529 | Reference: 0.521. Best-of-16 at inference: 0.556 (16 samples). RFT: 0.517. DPO: 0.498. GRPO: 0.456 | Mixed | BOND and J-BOND keep the most gold among one-sample methods but recover little of best-of-N's gain. |

### GeneralPreferences (W9 2.3: Nash learning and self-play preference optimisation, 2.4: general preference models)
Sources: R-wave9. Theme: alignment. Test: GeneralPreferencesTest. Blocks: `PreferenceEmbeddingModel` (GPM, Bradley–Terry, pairwise), `SelfPlayPreference` (SPPO, Nash-MD). Annotators have rock-paper-scissors cycles.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| GPM (preference embeddings) | Held-out preference accuracy; share of cyclic triples captured | 0.963; 1.000 | Bradley–Terry: 0.674; 0.583 (its transitive bound is 2/3). Pairwise classifier: 0.941; 1.000 | Win | GPM models cyclic preferences that Bradley–Terry cannot. |
| SPPO (true preference; GPM; Bradley–Terry); Nash-MD (GPM) | Worst-case win rate against a style panel (Nash value 0.5) | SPPO 0.464; 0.354; 0.409. Nash-MD 0.393 | Start: 0.492. DPO: 0.493. Iterative DPO: 0.424. Bradley–Terry reward + PPO: 0.444 | Negative | Negative: no method improves the worst case. |

### PreferenceShootout: displacement, length bias and exploration (W9 2.5: χ²PO and SquaredPO, 2.6: length-regularised DPO, 2.7: XPO)
Sources: R-wave9. Theme: alignment. Test: PreferenceShootoutTest. Blocks: `LinkDpoLosses`, `LengthRegularizedDpo`, `ExploratoryPreferenceOptimizer`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| χ²PO; SquaredPO | Chosen log-prob drift / gold reward / KL | χ²PO −1.23 / 0.646 / 1.13. SquaredPO −0.19 / 0.573 | DPO: −2.66 / 0.645 / 2.05. SimPO: −2.45 / 0.614. IPO: −0.15 / 0.567. DPOP: −0.03 / 0.551 | Win | χ²PO keeps DPO's gold with half the drift and half the KL. SquaredPO, like IPO and DPOP, stops the drift but loses gold. |
| R-DPO (α = 0.5) | Length / accuracy / raw win rate / length-controlled win rate | 5.04 / 0.26 / 0.85 / 0.483 | Reference: 3.55 / 0.16 / 0.53 / 0.571. DPO: 5.03 / 0.24 / 0.91 / 0.690. SimPO: 6.47 / 0.00 / 0.96 / 0.400. GRPO with LengthControlledReward: 2.04 / 0.64 / 0.18 / 0.773 | Negative | Negative: R-DPO does not shorten answers and loses to DPO on the controlled win rate. The GRPO baseline is best. The length-controlled metric gives 0.500 on wins driven only by length, as it should. |
| XPO | Gold reward (queries used) | 0.298 (83) | Reference: 0.278. Online DPO: 0.300 (17, most pairs are ties). Offline DPO: 0.275. SPIN: 0.298 | Negative | Negative: no method finds the rare best response; its probability stays at or below 0.0002. |

Not done: WP3's length-controlled GLM was not available, so 2.6 uses a local logistic version.

### ProcessSupervision (W9 2.8: VinePPO, 2.9: Math-Shepherd, 2.10: process advantage verifiers, 2.11: PRIME, 2.12: Step-DPO)
Sources: R-wave9. Theme: reasoning. Tests: ProcessSupervisionTest, ProgressRewardTest, ImplicitProcessRewardTest, StepDpoTest. Blocks: `MonteCarloValuePpo`, `RolloutStepLabeler`, `ProgressReward`, `ImplicitProcessReward`, `StepPreferenceBuilder`, `TokenPolicyGradient`. Task: `StepChainTask`, 2–3 operand running sums mod 10. Accuracy is sampled (temperature 1, four samples per prompt) on the training problems, because the toy WindowLm memorises rather than generalises (about 0.3 held out against 0.8 on train).

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| VinePPO (Monte Carlo values) | Sampled accuracy; value error against a 64-rollout estimate | 0.200 (1.6x the tokens); 0.064 / 0.048 | Start: 0.163. PPO with a learned critic: 0.308, value error 0.16 / 0.18. GRPO: 0.275. RLOO: 0.245. Constant value: 0.12 / 0.13 | Mixed | Its values are far more accurate, but at this budget that does not give a better policy. |
| Math-Shepherd rollout labels (soft; hard) | Agreement with gold step labels; best-of-16 accuracy on 60 prompts | 0.856; 0.794. PRM best-of-16: 0.167 (soft), 0.133 (hard) | PRM on gold labels: 0.133. Outcome reward model: 0.200. Generative verifier: 0.133. First sample: 0.217 | Negative | Negative: the labels are good, but every verifier is too weak to beat taking the first sample. |
| PAV (GRPO + progress reward); PRM + PAV beam search | Final accuracy | 0.263; beam 0.250 | Outcome-only GRPO: 0.280. GRPO + Q-value PRM: 0.252. PRM beam alone: 0.183 | Mixed | No gain in RL; a clear gain in beam search. |
| PRIME (implicit PRM) | Final accuracy | 0.277 | Outcome-only RLOO: 0.275. RLOO + frozen PRM: 0.268 | Negative | Null: no gain. The frozen PRM's score rises fastest under the policy trained against it (0.094 → 0.132), a mild sign of reward hacking. |
| Step-DPO | Sampled accuracy (2-step chains) | 0.180 (0.156) | Start: 0.163. Solution-level DPO: 0.187 (0.214). RFT: 0.228. STaR: 0.312 | Negative | Negative: step-level pairs do not beat whole-solution pairs, and STaR is best. The start model is trained on puzzles with 40% wrong leading digits, standing in for a pre-trained model. |

### ArithmeticRl: RLVR from one example (W9 2.13)
Sources: R-wave9. Theme: reasoning. Test: OneShotRlvrTest. Block: `SingleExampleSelector`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Top-1 problem by historical variance (with; without the entropy bonus); top-16 | Held-out sampled accuracy | 0.105; 0.105. Top-16 0.065 | Start: 0.245. Full training half: 0.180. One random problem: 0.168. Random reward: 0.243. Format-only reward: 0.123 | Negative | Negative: RL on this lookup-like task hurts held-out problems, and one example is no better than the full set. |

### CodeExecution (W9 2.14: execution prediction from traces, 2.15: self-repair, 2.16: execution-based reranking)
Sources: R-wave9. Theme: code. Tests: ExecutionPredictionTest, SelfRepairTest, ExecutionRerankingTest. Blocks: `ExecutionTracer`, `TraceLm`, `RepairLoop`, `ProgramSampler`, `ExecutionReranker`. Tasks for 2.15 and 2.16 are random programs of 1–2 primitives.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| CWM-style full trace; state scratchpad | Exact-output accuracy, 1–3 ops / 4–6 ops | Trace 0.300 / 0.044. States 0.156 / 0.000 | Direct prediction: 0.156 / 0.033 | Win | Traces double in-distribution accuracy at about 6x the target tokens; length generalisation stays poor. |
| Repair from output diff; from trace; Reflexion | Solve rate at 1 / 2 / 4 / 8 / 16 model calls | Diff 0.14 / 0.16 / 0.28 / 0.42 / 0.52. Trace 0.16 / 0.20 / 0.34 / 0.42 / 0.54. Reflexion 0.14 / 0.18 / 0.26 / 0.42 / 0.62 | Resampling: 0.14 / 0.26 / 0.32 / 0.40 / 0.56. GuidedEnumerator: 0.02 / 0.06 / 0.14 / 0.26 / 0.42 | Negative | Negative, as Olausson et al. predict for a first-sample solve rate below 0.3: at equal budget repair does not beat resampling. Reflexion is best at 16 calls. |
| Filter; AlphaCode clustering; CodeT dual agreement; LEVER | Hidden-test top-1 accuracy at 8 / 32 samples | Filter 0.425 / 0.687. Clustering 0.425 / 0.712. CodeT 0.425 / 0.712. LEVER 0.338 / 0.450 | Likelihood: 0.275 / 0.288. Verifier without execution: 0.250 / 0.225 | Win | Running the code is what matters; clustering and CodeT add about 0.025 over filtering. LEVER's training was cut to 30 tasks. |

Not done: the NeuralGpu and TransformerSeq2Seq baselines for 2.14 (one does not fit, the other was dropped for time).

### ToolAgentBench: the ArCHer recipe (W9 2.17: ArCHer with a tuned critic)
Sources: R-wave9. Theme: agents. Tests: ToolAgentArcherRecipeTest, ToolAgentArcherFullRecipeTest. `TurnLevelCritic` gained `doubleQ()` and `targetRate`, `HierarchicalAgentTrainer` gained critic warm-up and an entropy bonus; the defaults keep wave 8's rows.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| ArCHer + critic warm-up; + double Q; slow actor (target rate 0.01); full recipe | Greedy / sampled success at about 6,000 interactions | Warm-up 0.75 / 0.71. Double Q 0.75 / 0.79. Slow actor 1.00 / 0.82. Full 1.00 / 0.80 (lowest TD error, 0.069) | Wave 8 ArCHer: 0.75 / 0.78. Trajectory GRPO: 1.00 / 0.98 | Mixed | The slow actor fixes greedy success, but trajectory GRPO is still best. |

---

## Wave 9 WP3: data mixing, selection, synthetic data and evaluation methodology

New packages: `experimental.eval`, `experimental.synth` and `experiments.evaluation`. Data experiments extend `experiments.data`. CalibrationEstimatorsTest (10.5 s) and JudgeAuditTest (11.7 s) run slightly over the 10 s target (R-wave9).

### DataCuration: online mixing (W9 3.1: online data mixing as a bandit, 3.2: Aioli)
Sources: R-wave9. Theme: data. Test: OnlineMixingTest. Blocks: `BanditMixer`, `AioliMixer`. Markov grammars stand in for SyntheticCorpus and SyntheticGrammar.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| ODM; clipped ODM | Target loss after 300 steps | 2.598; 2.619 | Uniform: 2.619. RegMix: 2.347. Temperature: 2.791. DoReMi (proxy cost included): 2.807 | Negative | Negative: no better than uniform within noise; the offline RegMix mixture is best. |
| Aioli | Target loss; share on the two-domain grid | 2.648; 0.55 | Uniform: 2.619. Optimum share: 0.50 (ODM lands at 0.48) | Negative | Negative: it finds a near-optimal share on two domains but does not beat uniform on the full mix. |

### DataCuration: skills and curricula (W9 3.3: Skill-It, 3.4: learning-progress curricula)
Sources: R-wave9. Theme: data. Tests: SkillCurriculaTest, LengthCurriculaTest. Blocks: `SkillGraphSampler` (with `estimateByAblation`), `ProgressCurriculum`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Skill-It on the planted graph; on the estimated graph | Target loss | 0.758; 0.714 | Uniform: 0.965. Fixed curriculum: 0.717. Target only: 0.631 | Mixed | Beats uniform and avoids the decoy skill (share 0.01), but not training on the target alone. The estimated graph recovers 0 of 4 planted edges: the prerequisites create no measurable transfer in this MLP. |
| EXP3.S; slope Thompson; slope greedy; self-paced | Final slot accuracy after 300 steps | 0.243; 0.237; 0.132; 0.083 | Fixed pacing: 0.277. Uniform: 0.268. Prioritized Level Replay: 0.229 | Negative | Negative: no teacher reaches the 0.9 target or beats fixed pacing, and self-paced learning stalls. Steps wasted on the random task: EXP3.S 0.27, Thompson 0.24, uniform 0.24, PLR 0.82. |

### DataCuration: RHO-LOSS selection (W9 3.5)
Sources: R-wave9. Theme: data. Test: ReducibleLossSelectionTest. Block: `ReducibleLossSelector`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| RHO-LOSS | Final accuracy; share of noisy labels picked | 0.688; 0.03 | Uniform: 0.666; 0.21. Highest loss: 0.604; 0.40. EL2N: 0.626; 0.25 | Win | It skips noisy labels almost entirely; the accuracy gain over uniform is small. No method reaches 0.8 within 300 steps. |

Not done: Rho-1 is not in the comparison.

### SyntheticTraining (W9 3.6: dataset distillation, 3.7: TabDDPM and CTGAN, 3.8: classifiers on diffusion samples)
Sources: R-wave9. Theme: data. Tests: DatasetCondensationTest, SyntheticTablesTest, ConditionalTabularGanTest, GenerativeAugmentationTest, DiffusionAugmentationTest. Blocks: `DatasetCondenser`, `TabularDiffusion`, `ConditionalTabularGan` (with a PacGAN critic, pac 8), `SyntheticDataAudit`, `GenerativeAugmenter`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Gradient matching; distribution matching, digits, 1 image per class | Accuracy of a fresh ReLU MLP / tanh MLP | GM 0.349 / 0.363. DM 0.348 / 0.337 | Random real images: 0.177 / 0.183. k-means centroids: 0.403 / 0.420. Herding: 0.188 / 0.230. k-center: 0.188 / 0.230. Full set: 0.690 / 0.707 | Mixed | Gradient matching beats random by 0.17 (the hypothesis asked for 0.15) and transfers across activations, but plain class centroids are better on this toy. |
| TabDDPM; CTGAN (800 steps), table with about 12% positives | Accuracy; rare-class recall; balanced accuracy; closest-record ratio (below 1 means copying) | TabDDPM 0.852; 0.184; 0.567; 1.12. CTGAN 0.814; 0.314; 0.600; 1.69 | Real data: 0.925; 0.455; 0.724. SMOTE: 0.905; 0.395; 0.687; 0.31. Gaussian copula: 0.874; 0.000; 0.500; 1.04 | Mixed | CTGAN is the best generator on the rare class, as predicted, but SMOTE beats both. The copying claim fails: SMOTE copies most and TabDDPM does not copy. CTGAN results are noisy across steps and widths. |
| Real + diffusion samples (guidance 1; 3), digits, 5 images per class | Accuracy | 0.194; 0.202 | Real only: 0.199. Shift + mixup: 0.259. Real + VAE samples: 0.202 | Negative | Negative: synthetic images do not help at this compute. At 20 images per class: real 0.357, mixup 0.424, VAE 0.350, diffusion 0.275–0.347 for guidance 0.5–4. The sampler needs `clipSample(3)` to stay stable. |

Not done: digits only (no ShapeImages); 9.3.8 at 5, 20 and 100 images per class (100 only in `main`) instead of 50, 200 and 1,000; `SyntheticRecsys.clicks` is not used; the copula uses empirical marginals.

### ContinualPretraining (W9 3.9: re-warming, re-decaying, replay and the stability gap)
Sources: R-wave9. Theme: data. Test: ContinualPretrainingTest. Block: `ContinualPretrainer`. Old data A, new data B.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Re-warm; re-warm + 5% replay; + low-perplexity subset | Final loss A / B (rise of A's loss) | 2.53 / 2.06 (1.04). Replay 2.42 / 2.08 (0.94). Subset 2.66 / 1.93 (1.18) | Constant rate: 1.81 / 2.67 (0.33). EWC: 2.48 / 2.07 (1.00). Retrain on A+B: 1.88 / 2.13 | Mixed | Re-warming learns B but forgets A, and replay narrows the gap to retraining (0.25 nats on the mean). A's loss never recovers, so this is plain forgetting, not a transient stability gap. |

### CalibrationEstimators (W9 3.10: debiased, adaptive and smooth ECE)
Sources: R-wave9. Theme: evaluation. Test: CalibrationEstimatorsTest (10.5 s). Block: `CalibrationEstimators`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Smooth ECE; monotone sweep; debiased L2 | Bias of the estimate at n = 200 | +0.011; +0.012; L2 −0.004 | Equal-width ECE: +0.025 (1.64x the truth). Plug-in L2: +0.059 | Win | The newer estimators halve the bias, and the debiased L2 is nearly unbiased. The method ranking holds for every bin count from 5 to 50. |

### JudgeAudit (W9 3.11: LLM-as-judge biases, 3.12: Rogan–Gladen and prediction-powered inference)
Sources: R-wave9. Theme: evaluation. Test: JudgeAuditTest (11.7 s). Blocks: `JudgeAudit`, `JudgeCorrection`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Bias audit; swap averaging; length control | Measured bias; Spearman with the gold ranking | Position +0.110 (inconsistent on 0.42 of pairs). Verbosity +0.246 (planted +0.250). Self-preference +0.099. Length-controlled ρ 0.43. Swap-averaged −0.94 | Raw judge ρ: −0.89 | Win | The audit recovers the planted bias almost exactly, and length control turns an inverted ranking positive; swap averaging alone does not. |
| PPI++ (tuned weight); Rogan–Gladen | 95% interval coverage; width | PPI++ 0.93; 0.106. Rogan–Gladen under a pass-rate shift 0.95 | Judge only: coverage 0.00. Human only: width 0.136. PPI under the shift: 0.02 | Win | PPI++ gives valid intervals 22% narrower than humans alone; Rogan–Gladen survives the shift that breaks PPI. |

### ContaminationAudit (W9 3.13: the Kernel Divergence Score and rephrased contamination)
Sources: R-wave9. Theme: evaluation. Test: ContaminationAuditTest. Blocks: `KernelDivergenceScore`, `Decontaminator`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Kernel Divergence Score | Spearman with the contamination rate | −1.00 | Min-K%++: +1.00. Min-K%: +1.00. −loss: +1.00 | Negative | Negative: KDS is monotone, but its sign is reversed on this toy. |
| Embedding nearest-neighbour check | Detection of paraphrased copies (false positives) | 1.00 (0.02) | 6-gram overlap: 1.00 of verbatim copies, 0.20 of paraphrases | Win | Paraphrases evade n-gram decontamination but not embeddings. Benchmark loss: clean 2.54, paraphrased 2.24, verbatim 1.92. |

Item verdict: Mixed.

### StatisticalReporting (W9 3.14: clustered error bars, paired tests and power, 3.15: the Ladder and Thresholdout)
Sources: R-wave9. Theme: evaluation. Test: StatisticalReportingTest. Blocks: `EvalStatistics`, `ReusableHoldout`, `AdaptiveAnalyst`. The evaluations are simulated, not retrained.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Clustered standard error; cluster bootstrap | 95% interval coverage | 0.95; 0.96 | i.i.d. standard error: 0.75 | Win | Ignoring question clusters gives intervals that are far too narrow. |
| Paired test | Questions needed for 80% power | 6,398 | Unpaired: 30,116 | Win | Pairing cuts the needed sample almost 5x. Two identical models end at least 1 point apart in 74 of 100 reruns. |
| Ladder; parameter-free Ladder; Thresholdout | Reported accuracy under a boosting attack on random labels (true about 0.49) | 0.568; 0.552; 0.624 | Plain holdout: 0.711 | Win | The Ladder removes most of the overfitting, though not all of it. |

### BudgetRankings (W9 3.16: do method rankings survive the training budget?)
Sources: R-wave9. Theme: evaluation. Tests: BudgetRankingsTest, MixerBudgetRankingsTest. Block: `BudgetRankingStudy`. LongHorizonForecasting, Forecasters and MixerShootout gained curve logging.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Forecasting lineup, 20 to 320 steps, 2 seeds, 5 models | Kendall τ between the 1x and 16x rankings | −0.20 [−0.80, 0.40] | Hypothesis: below 0.4 | Win | The hypothesis holds: test-budget rankings do not survive. DLinear leads early; the transformer, FEDformer and PatchTST overtake it at 160–320 steps. |
| Recall mixers, 10 to 80 steps | τ against the final ranking | 0.40 at 10 steps, 0.91 at 40 | — | Win | Mixer rankings settle by 40 steps; Mamba overtakes S4D there. |

Not done: no 32x run. The test uses a smaller ladder (10 to 80 steps, 1 seed, τ 0.67).

### CurveExtrapolation (W9 3.17: learning-curve extrapolation and LC-PFN)
Sources: R-wave9. Theme: evaluation. Test: CurveExtrapolationTest. Block: `CurveExtrapolator` (power law, MCMC ensemble, LC-PFN).

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Power law; ensemble; LC-PFN | Mean absolute error from 10% of the curve | 0.173; 0.184; 0.151 | Last value: 0.612 | Win | Extrapolation cuts the error by three quarters, and LC-PFN is best. |
| Same, forecasting lineup | Early-ranking τ from 10% of the budget | Power law 0.20. Ensemble 0.00 | Last value: −0.20. Hypothesis: above 0.7 | Negative | Negative: the final ranking still cannot be read from 10% of the curve. Successive halving does pick the right survivor and saves 70% of the compute. |

Item verdict: Mixed.

---

## Wave 9 WP4: interpretability tools, documents and tables, multimodal fusion

New packages: `experimental.document`, `experimental.interp`, `experimental.multimodal` and `experiments.document`. The block test InterpBlocksTest checks CKA invariance, max-flow on a small graph, that LEACE defeats a linear probe, and that every glyph in the font is distinct. The follow-up fixed seven failures (a CTC network stuck on blanks, OCR that had never seen ':' or '-', a hard count threshold in TAPAS and others). Classes well over the 10 s target: VqaEarlyFusionTest 18.4 s (whichever VQA class runs first pays about 14 s of shared set-up), the four audio-visual classes 11.7–14.6 s, DetrHybridCurveTest 12.5 s and TableQaLmTest 10.9 s (R-wave9).

### CircuitDiscovery (W9 4.1: ACDC and edge attribution patching, 4.2: distributed alignment search)
Sources: R-wave9. Theme: interp. Test: CircuitDiscoveryTest. Blocks: `interp.CircuitDiscovery` (ACDC, EAP, EAP-IG, `PlantedCircuit`), `interp.AlignmentSearch`. Task accuracy 0.979.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| ACDC; EAP; EAP-IG | Edge-recovery AUC | 1.0; 1.0; 1.0 | Node activation patching: 0.933. Node attribution patching (AtP): 0.935 | Win | Edge-level methods recover the planted circuit exactly; node-level methods do not. |
| DAS, k = 1; k = 2, hidden layer 0 | Interchange intervention accuracy | 0.270; 0.323 | Probe direction: 0.108. PCA 1 / 2: 0.143 / 0.178. Best single neuron: 0.115 | Win | A learned rotation finds the causal variable far better than probes, though absolute accuracy stays low. |

Not done: a planted-circuit model, not a trained transformer or WeightSparseCircuits; no AttributionGraph baseline.

### ConceptErasure (W9 4.3: INLP and LEACE)
Sources: R-wave9. Theme: interp. Test: ConceptErasureTest. Block: `interp.ConceptErasure`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| LEACE; INLP (8 rounds) | Linear-probe accuracy on the erased concept (chance 0.5) | 0.489; 0.506 | No erasure: 0.854 | Win | LEACE erases the concept to chance in one closed-form step. |

Not done: the mean-difference and steering-subtraction rows were not reported; no PersonaSteering toy-LM split.

### RepresentationComparison (W9 4.4: CKA, SVCCA and model stitching, 4.7: linear mode connectivity and Git Re-Basin)
Sources: R-wave9. Theme: interp. Test: RepresentationComparisonTest. Blocks: `interp.RepresentationSimilarity`, `interp.ModelStitcher`, `interp.PermutationAlignment`. MLPs reach 0.96 accuracy after the follow-up halved the pixel noise.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Linear CKA; RBF CKA; SVCCA; PWCCA | Layer-correspondence accuracy across seeds | 0.542; 0.583; 0.083; 0.250 | Best-neuron correlation: 0.667 | Negative | Negative: the simple best-neuron baseline beats every similarity index; SVCCA is near useless here. |
| Weight matching; activation matching | Loss barrier, width 16 / width 64 | Weight 0.712 / 0.127. Activation 0.540 / 0.177 | Naive averaging: 1.573 / 1.121 | Win | Permutation alignment removes most of the barrier at width 64; merged accuracy is 0.948 against 0.514 for naive averaging. |

Not done: MLPs only (no CNNs or ViTs), one width, 3 seeds; stitching numbers were not reported.

### LatentKnowledge (W9 4.5: contrast-consistent search)
Sources: R-wave9. Theme: interp. Test: LatentKnowledgeTest. Block: `interp.ContrastConsistentSearch`. A RIGHT/WRONG verdict token was added so the answer position is trained.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| CCS; contrastive PCA | Accuracy plain / sycophantic split | CCS 0.575 / 0.525. Contrastive PCA 0.520 | Supervised probe: 0.780 / 0.575. Zero-shot: 0.930 / 0.500. Random direction: 0.570 | Negative | Negative: CCS is no better than a random direction. A distractor feature scores 0.605 and 1.0. |

### ExplainabilityBenchmark: attention rollout and flow (W9 4.6)
Sources: R-wave9. Theme: interp. Test: AttentionRolloutBenchmarkTest. Block: `interp.AttentionRollout` (rollout, flow with a local Edmonds–Karp max-flow, gradient rollout). Copy task.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Rollout; flow; gradient rollout | Agreement with the true source token | 0.655; 0.655; 0.480 | Saliency: 0.865. Integrated gradients: 0.765. Raw last-layer attention: 0.730 | Negative | Negative: aggregating attention across layers is worse than raw attention and gradient methods. |

Not done: no ViT pointing game and no Grad-CAM row.

### DevelopmentalInterp (W9 4.8: the local learning coefficient)
Sources: R-wave9. Theme: interp. Test: DevelopmentalInterpTest. Block: `interp.LocalLearningCoefficient`. Grokking task.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| LLC over training | LLC as test accuracy rises 0.61 → 0.95 | 0.57 → 0.01 | — | Win | LLC drops at grokking, as predicted. |
| LLC across weight decays | Spearman with test accuracy | +1.00 | Weight norm: −1.00 | Negative | Against the hypothesis: weight norm ranks the models just as perfectly, and LLC's sign is the opposite of its drop over training. |

Item verdict: Mixed. Not done: no induction-head or superposition models, no per-layer LLC.

### TrainingAttributionExperiment: TRAK (W9 4.9)
Sources: R-wave9. Theme: data. Test: TrakDatamodelingTest. Block: `interp.Trak`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| TRAK, 1; 3 models | Linear datamodeling score | 0.455; 0.523 | Influence function (exact Hessian): 0.781. TracIn: 0.548. Gradient dot product: 0.518 | Negative | Negative: on a model small enough for the exact Hessian, the influence function wins and TRAK does not beat a gradient dot product. |

Not done: TRAK uses full-data models with different seeds; no CNN, no KNN-Shapley row.

### GlyphOcr (W9 4.10: line OCR with a CRNN and CTC)
Sources: R-wave9. Theme: document. Tests: GlyphOcrTest, GlyphOcrTransformerTest. Block: `document.LineRecognizer`. Data: `experimental.data.GlyphDocuments`. The follow-up added a residual connection around the BiGRU; before it CTC predicted only blanks (CER 1.0).

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| CRNN + CTC; transformer + CTC | Character error rate | 0.311; 0.347 | Blank-column segmentation + glyph MLP: 0.269 (0.277 touching, 0.231 separated) | Negative | Negative: segmentation still wins, even on touching glyphs. |

### DocumentParsing (W9 4.11: Donut-style parsing, 4.12: LayoutLM-style 2-D positions)
Sources: R-wave9. Theme: document. Tests: DonutParsingTest, DocumentParsingTest, DocumentPipelineTest. Blocks: `document.ImageToMarkup`, `document.LayoutEncoder`. A `formOcr` BiGRU reaches word CER 0.045 at 400 steps.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Donut-style image-to-markup, 50 steps | Field F1 | 0.000 | OCR + rules: 0.838. OCR + layout tagger: 0.788 | Negative | Negative: the end-to-end model learns nothing at test budget. |
| 2-D layout encoder, shuffled reading order | Tagging F1 | 0.889 | 1-D positions: 0.834. Nearest-key rules: 1.000 | Mixed | 2-D positions beat 1-D when the order is shuffled, but the rules are perfect, and after OCR the tagger (0.788) trails the rules (0.838). |

Not done: the OCR pipelines get the true word boxes; exact match instead of tree-edit accuracy; one budget, no learning curve; no heavy-noise comparison; KEY / VALUE / O labels instead of BIO.

### TableQa (W9 4.13: TAPAS)
Sources: R-wave9. Theme: document. Tests: TableQaTest, TableQaLmTest. Block: `document.TableEncoder`. Data: `experimental.data.ToyTables`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| TAPAS (cell selection with soft aggregation) | Answer accuracy (lookup / compare / count / sum) | 0.250 (0.22 / 0.34 / 0.30 / 0.14) | GuidedEnumerator semantic parser: 1.000. Flat LM on the linearised table: 0.145 | Mixed | Better than a flat LM, far below the parser. |

### DetectionShootout: DETR with hybrid matching (W9 4.14)
Sources: R-wave9. Theme: vision. Test: DetrHybridCurveTest (12.5 s). `SetPredictionDetector.Options` adds one-to-many matching, anchor queries and auxiliary losses; defaults keep wave 8's rows.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| H-DETR (hybrid matching), layout pages | mAP at 250 / 500 steps | 0.023 / 0.095 | Plain DETR: 0.038 / 0.175 | Mixed | Behind at test budget, but in a 700-step probe H-DETR led 0.49 to 0.34. |

Not done: scenes were dropped (both detectors stay at mAP 0 even at 700 steps); no 20k-step curve; CenterNet and RetinaNet not rerun on the same curve.

### FusionShootout (W9 4.15: frozen-LM bridges, 4.16: Transfusion, 4.17: fusion bottlenecks and modality dropout)
Sources: R-wave9. Theme: multimodal. Tests: VqaProjectorTest, VqaQformerTest, VqaGatedTest, VqaEarlyFusionTest, TransfusionSharedTest, TransfusionSeparateTest, VqTokenSharedTest, VqTokenSeparateTest, AvBottleneckFusionTest, AvEarlyFusionTest, AvLateFusionTest, AvUnimodalTest (they replace FusionShootoutTest). Blocks: `multimodal.PatchProjector`, `QueryTransformer`, `GatedCrossAttentionAdapter`, `FusionDecoder`, `TransfusionModel`, `BottleneckFusion`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Linear projector; Q-Former; gated cross-attention, frozen LM and ViT | VQA accuracy | 0.650; 0.640; 0.650 | Text only: 0.515. Early fusion from scratch: 0.635. Frozen LM given the true description (ceiling): 0.770 | Negative | Negative: every bridge sits at the majority-answer prior (about 0.65), well below the ceiling. The projector leaves the LM's text loss unchanged. |
| Transfusion, shared; separate models | Caption accuracy; generation accuracy | Shared 0.367; 0.133. Separate 0.583; 0.067 (generation at chance) | VQ-token early fusion, shared: 0.558; 0.567 (distance 0.077). Separate: 0.663; 0.433 | Negative | Negative: at this budget discrete VQ tokens beat Transfusion's continuous patches. |
| Bottleneck fusion, TalkingDigits audio + DigitGlyphs | Accuracy, both modalities (with modality dropout) | 0.503 (0.203) | Early fusion: 0.330 (0.163). Late fusion: 0.427 (0.143). Audio only: 0.433. Image only: 0.233 | Mixed | Bottlenecks fuse best, but modality dropout hurts at 150 steps, and 28 tokens are too few to save compute (550,400 and 559,104 multiply-adds for bottleneck and early fusion). |

Not done: Q-Former with the contrastive stage only; early fusion on raw pixel patches, not VQ tokens; no CLIP baseline; Transfusion on raw 2×2 patches with no VAE, DigitGlyphs only, compute matched by steps.

---

## Wave 9 WP5: world models and video, audio generation, scientific-ML follow-ups

New packages: `experimental.audio`, `experimental.world` and `experiments.world`. Numbers come from shrunk test configurations, so treat them as small-scale indications. Classes over the 10 s target, measured with parallel forks: TTS 22 s, long rollouts 20 s, slot video 27 s, token world model 12 s, codec 10 s (R-wave9).

### VideoGeneration (W9 5.1: factorised space-time attention, 5.2: MAGVIT-style 3-D tokenizer, 5.3: Self Forcing, 5.4: history guidance, 5.13: FSQ for video)
Sources: R-wave9. Theme: world. Tests: SpaceTimeDiffusionTest, VideoTokenizerTest, MaskedVideoGenerationTest, LongVideoRolloutTest. Blocks: `SpaceTimeDenoiser`, `VideoMetrics`, `SpaceTimeTokenizer`, `VideoMaskTasks`, `FrameDenoiser`, `HistoryGuidance`, `SelfForcingTrainer`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Factorised space-time attention | Unconditional clip MMD (ms per step) | 1.18 (121) | Frame-wise: 1.17 (44). Full 3-D: 1.17 (112). Real clips: 0.19. SimVP, first-frame conditioned: 0.35 | Negative | Negative: no quality gain and no saving over full 3-D attention; all diffusion models trail SimVP at this length. |
| 3-D FSQ and VQ tubelet tokenizers; MaskGIT, 8 steps | PSNR (tokens); clip MMD | 19.7 dB (48) for both. MaskGIT MMD 0.34 | Per-frame FSQ: 21.3 dB (144). One token per call: MMD 0.61 at 6x the time | Win | 3x fewer tokens for 1.6 dB, and parallel decoding is both better and faster. |
| History guidance (w = 4) | Count error over frames 16–32 of 32-frame rollouts | 0.91 | Diffusion forcing: 2.46. Teacher forcing: 4.65. Real clips: 0.52 | Win | Clear winner; from an earlier 60-step run (R-wave9). |
| Self Forcing | Same, shrunk run | 6.46 | Teacher forcing (earlier run): 4.65 | Negative | Negative at test scale. |

### WorldModelShootout (W9 5.5: IRIS, 5.6: DreamerV3 tricks, 5.7: DINO-WM, 5.8: SlotFormer)
Sources: R-wave9. Theme: world. Tests: TokenWorldModelTest, DreamerTricksTest, FeatureSpacePlanningTest, SlotDynamicsTest. Blocks: `TokenWorldModel`, `KeyDoorGrid`, `FeatureSpacePlanner`, `SlotDynamics`; DreamerV3 options in `DreamerAgent`, `MlpWorldModel` and `RecurrentStateSpaceModel` (defaults unchanged).

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| IRIS token world model | Token accuracy; cell accuracy | 0.95; 0.96 | MLP: 0.85. DIAMOND: 0.90 | Mixed | Best model of the dynamics (earlier run, R-wave9), but no policy beats random (return 0.10) at this budget. |
| DreamerV3 tricks (symlog, two-hot, percentile normalisation, unimix, KL balance) | CartPole return at reward scale ×0.01 / ×1 / ×100 | 9.2 / 32.2 / 36.2 | Default Dreamer: 9.6 / 58.0 / 39.2 | Negative | Negative: the tricks do not help at 600 steps. |
| DINO-WM planning in frozen SimCLR-style features | Goal success | 0.83 | MAE-style features: 0.50. Random features: 0.00. Pixel model: 0.17. Goal-conditioned BC: 0.67 | Win | Planning in good frozen features beats both pixels and behaviour cloning. |
| SlotFormer | PSNR; FG-ARI | 9.5 dB; 0.16 | SimVP: 12.7 dB | Negative | Negative, not validated: it still looks broken or badly undertrained, and the position metric saturates. A sigmoid on the decoder lifted it from 2.4 dB. |

Not done: pixel CartPole for 5.6; `JepaWorldModel` for 5.7 (no planning hook).

### AudioGeneration (W9 5.9: HiFi-GAN, 5.10: DiffWave and WaveGrad, 5.11: TTS alignment, 5.12: latent audio diffusion, 5.13: FSQ for audio)
Sources: R-wave9. Theme: audio. Tests: VocoderShootoutTest, FlowMatchingTtsTest, LatentAudioDiffusionTest, CodecQuantizerTest. Blocks: `GriffinLim`, `GanVocoder`, `DiffusionVocoder`, `MelFeatures`, `MonotonicAlignmentSearch`, `FlowMatchingTts`, `LatentAudioDiffusion`, `FiniteScalarQuantizer`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| HiFi-GAN-lite (MPD + MSD); DiffWave; WaveGrad, 6 GAN / 10 diffusion steps | Judge digit error | 0.67–1.0 | Griffin–Lim (32 iterations): 0.00, MR-STFT distance 1.77 | Inconclusive | Smoke-level training only; the vocoders are untrained and the hypotheses untested. |
| Matcha-style MAS; F5-style filler padding, 100 steps | Digit error on 1–3 / 4–5 digit strings | MAS 1.60 / 1.07. F5 1.20 / 0.89 | True durations: 1.40 / 0.96. Ground-truth mel + Griffin–Lim: 0.30 / 0.30 | Negative | Negative: F5 padding beats MAS, against the hypothesis. MAS boundary error is 1.75 frames against 0.64 for predicted durations. All models are undertrained. |
| Latent audio diffusion (32-d mel VAE latent) | Judge accuracy; MMD; generation time | 0.33; 0.021; 33 ms | Mel-space diffusion (640-d): 0.10; 0.055; 89 ms. Real clips: 0.70 | Win | Better and faster in the latent. |
| FSQ, codebooks of about 2^10 / 2^12 / 2^14 | Code utilisation | 0.91 / 0.56 / 0.26 | VQ: 0.64 / 0.15 / 0.04. RVQ: 0.85 / 0.60 / 0.33 | Win | FSQ uses far more of its codes than VQ, as the paper predicts. SNR is 3–4.4 dB, with RVQ slightly ahead. On video, 3-D FSQ matches 3-D VQ (19.7 dB). |

Not done: Tacotron-lite for 5.11; CodecLanguageModel for 5.12; WaveNet, ToyAudioCodec and HarmonicNoiseSynth vocoder baselines; MultiScaleVqTokenizer for 5.13. 5.11 and 5.12 vocode with Griffin–Lim.

### PdeShootout: hard PDEs and second-order optimisers (W9 5.14: curriculum, time marching and periodic boundaries, 5.15: ENGD and Newton–CG)
Sources: R-wave9. Theme: sciml. Tests: PinnHardFixesTest, PinnSecondOrderTest. Blocks: `PinnTraining` options, `PinnNaturalGradient` (ENGD, Levenberg–Marquardt), `NystromNewtonCg`. ENGD's damping now rises 100x when no step is accepted; at a fixed μ = 1e-8 it made no progress (R-wave9).

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Curriculum; time marching; exact periodic boundaries, advection β = 30 | Relative L2 | About 1.0 for every recipe (curriculum 1.17) | Coarse numerical solver: 0.003 | Negative | Negative: the paper's fixes do not rescue the PINN here. |
| ENGD; Adam + L-BFGS + Nyström Newton–CG | Poisson relative L2 | 2.1e-6; 1.1e-3 | Adam + L-BFGS: 3.3e-3. Levenberg–Marquardt: 5.6e-2 | Win | Reproduces the paper: natural gradient gains three orders of magnitude. |

Not done: HessianFree and K-FAC for 5.15 (they evaluate trial steps without a graph).

### OperatorLearning: model-size and data sweep (W9 5.16: Transolver and LANO against FNO)
Sources: R-wave9. Theme: sciml. Test: OperatorSweepTest.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| LANO; Transolver, sweep incl. 50% of grid points dropped | Burgers relative L2 | LANO 0.45–0.67. Transolver 0.50–0.72 | FNO: about 0.19–0.21. DeepONet: 0.85–0.95 | Negative | Negative: FNO leads everywhere, and dropping half the grid barely matters, so Transolver never overtakes it. |

---

## Wave 9 WP6: parallel training, streaming and online learning, edge deployment

New packages: `experimental.parallel`, `experimental.stream`, `experimental.edge`, `experiments.parallel`, `experiments.streaming` and `experiments.edge`. All 23 WP6 tests pass. Edge classes take 3.4–10.0 s, StreamShootout about 12 s, the parallel classes 1–3 s. The follow-up fixed a scheduler bug that reported false ZB-H1 deadlocks and an OFA bug where Adam's moments did not follow permuted channels (R-wave9).

### ParallelismShootout (W9 6.1: pipeline schedules, 6.2: zero-bubble pipelines and DualPipe, 6.3: tensor and sequence parallelism, 6.4: ZeRO, 6.6: expert parallelism)
Sources: R-wave9. Theme: parallel. Test: ParallelismShootoutTest. Blocks: `PipelineSchedule`, `PipelineRunner` (threads as stages), `ParallelCostModel`, `SplitBackward`, `TensorParallelBlock`, `Collectives`, `ShardedDataParallel`, `ExpertParallelMoe`. Bubble fractions are simulated; the threaded runs match single-device gradients to within 1e-8.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| 1F1B; interleaved 1F1B (v = 2), p = 4, m = 8 | Bubble fraction; peak activations | 1F1B 0.277; 4. Interleaved 0.162 (theory 0.158); 5.5 | GPipe: 0.274 (theory 0.273); 8 | Win | 1F1B halves activation memory at the same bubble; interleaving cuts the bubble as theory says. |
| ZB-H1; ZB-H2; DualPipe, p = 4, m = 8 | Bubble fraction; peak activations | ZB-H1 0.150; 4. ZB-H2 0.150; 7. DualPipe 0.040; 5 | 1F1B: 0.277; 4 | Mixed | Splitting the backward nearly halves the bubble, but ZB-H2 never beats ZB-H1 (the greedy scheduler does not use its memory), and DualPipe gets worse at p = 8, m = 32 (0.411, against 0.093 for ZB-H1 and 0.185 for 1F1B). |
| Tensor parallelism; + sequence parallelism, t = 8 | Gradient error; parameters per rank; activations per rank | ≤ 6e-6; 1,088; SP 2,176 floats | Single device: 8,256 parameters. TP alone: 6,656 floats | Win | Exact, and sequence parallelism cuts activations 3x for about 2% more traffic. |
| ZeRO-1; ZeRO-2; ZeRO-3, N = 4 | Memory per rank (bytes) | 0.82M; 0.58M; 0.59M (1.5x ZeRO-1's traffic) | DataParallel: 1.32M. ZeRO-1 + 4-bit optimizer states: 0.68M | Win | Loss matches DataParallel to within 2.4e-7. ZeRO-3 saves nothing over ZeRO-2 at this model size. |
| Expert parallelism, aux-loss; loss-free balancing | Tokens dropped at capacity factor 1.0 / 2.0 | Aux-loss 15.4% / 0.5%. Loss-free 10.7% / 0% | ExpertChoiceMoe: 22% of tokens unprocessed | Win | Loss-free balancing drops the fewest tokens. |

### MemorySchedules (W9 6.5: selective recomputation and optimal checkpoint schedules)
Sources: R-wave9. Theme: parallel. Test: MemorySchedulesTest. Block: `CheckpointPlanner` (√n, selective, exact Pareto DP, Revolve) through `Recompute`. Attention-score memory is modelled, because fused attention keeps no scores.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Selective recomputation of attention; optimal plan | Activation memory saved; extra compute | 46%; 0.248. Optimal plan reaches √n's memory at 0.232 | Recompute every block: 84%; 0.333. √n: 0.333 | Win | The optimal plan is never worse than √n, and executed plans have zero gradient error. |

### AsyncTraining (W9 6.7: Hogwild and delay-compensated ASGD)
Sources: R-wave9. Theme: parallel. Test: AsyncTrainingTest. Block: `AsyncSgd`. The click model gained a crossed feature, because its labels depend on a field interaction.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Hogwild, 8 threads | Log loss (speed-up) | 0.535 (2.5x) | One thread: 0.536 | Win | Lock-free updates lose nothing on a sparse model; the workload is tiny. |
| DC-ASGD-a; staleness-aware learning rate, dense MLP, delay 16 | Accuracy | 0.88; 0.885 | Plain ASGD: 0.10 | Win | Both corrections rescue training that stale gradients break. |

### StreamShootout (W9 6.8: FTRL-Proximal, 6.9: Hoeffding trees and adaptive random forests, 6.10: drift detectors, 6.11: ER-ACE, 6.12: coin betting)
Sources: R-wave9. Theme: streaming. Test: StreamShootoutTest (about 12 s). Blocks: `FtrlProximal`, `HoeffdingTree`, `AdaptiveRandomForest`, `DriftDetectors`, `AsymmetricReplay`, `CoinBetting`. Data: `experimental.data.DriftingStreams`. SEA now has 4 drifts instead of 8, so ADWIN has enough samples per concept.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| FTRL-Proximal, 90k samples | Progressive log loss; non-zero weights | 0.608; 7,244 | AdaGrad: 0.622; 8,250. Daily retraining after drifts: 0.709 (FTRL 0.651) | Mixed | Slightly better loss, but only 12% fewer weights, not the paper's 5–10x. |
| Adaptive random forest | Accuracy, SEA / hyperplane / RBF | 0.853 / 0.803 / 0.844 | Majority class: 0.616 / 0.494 / 0.248 | Mixed | Far above majority, but on hyperplane the online MLP and the plain Hoeffding tree beat the forest. |
| ADWIN; Page–Hinkley; EDDM | Drifts missed; detection delay | SEA: ADWIN misses 1 of 4, delay 780. Page–Hinkley delay 134. Hyperplane: EDDM misses none | BOCPD: delay 15 but 187 false alarms per 10k; misses none on hyperplane | Mixed | No detector is both fast and quiet; ADWIN is slow at this drift size. |
| ER-ACE; ER-AML, digits | Final accuracy (forgetting), buffer 20; accuracy at buffer 200 | ER-ACE 0.320 (0.188); 0.420. ER-AML 0.485 at buffer 200 | ER: 0.245 (0.713). No replay: 0.195. DER++: 0.375 at buffer 20 | Mixed | The asymmetric loss cuts ER's forgetting by three quarters, but DER++ is best at buffer 20. |
| COCOB; KT | Click-task accuracy; digits MLP accuracy | COCOB 0.685; 0.38. KT 0.678 | AdamW at its best rate: 0.680; 0.15 | Win | Parameter-free betting matches tuned AdamW and beats it whenever AdamW's rate is 10x off. |

### EdgeDeployment (W9 6.13: integer-only inference, 6.14: patch-based inference, 6.15: once-for-all networks, 6.16: on-device training, 6.17: keyword spotting)
Sources: R-wave9. Theme: edge. Tests: IntegerInferenceTest, PatchInferenceTest, OnceForAllTest, OnDeviceTrainingTest, KeywordSpottingTest, KeywordBaselinesTest. Blocks: `EdgeNet`, `IntegerInference`, `MemoryPlanner`, `PatchBasedExecutor`, `ElasticSupernet`, `SubnetSearch`, `SparseUpdateTrainer`, `LiteResidual`, `KeywordSpotters`. Inputs were cut to 8×8 (16×16 for the patch budget).

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Post-training int8; QAT int8 | Accuracy: ShapeImages CNN / digits MLP / keyword DS-CNN | PTQ 0.905 / 0.775 / 0.825. QAT 0.885 / 0.825 / 0.867 | Float32: 0.900 / 0.770 / 0.825 | Mixed | Integer inference keeps accuracy at 3.6x fewer bytes (14,680 → 4,043), and the two integer engines agree bit for bit. Negative: int8 outputs match the fake-quant model exactly in only 86%, 99% and 52% of values, as one-step rounding differences build up. |
| Patch-based inference, 2×2; 4×4 grid, 64×64 input | Peak memory (extra multiply-adds) | 22,280 B (11%); 14,336 B (35%) | Layer by layer: 49,152 B | Win | Output is exact (difference 4e-8). At the patch budget, full resolution with patches scores 0.665 against 0.360 at half resolution and 0.445 at half width. Reordering a three-branch cell halves its peak (28,672 → 14,336 B); in-place depthwise gives nothing. |
| Progressive shrinking + KD, 160 steps | Accuracy, largest / smallest subnet | 0.625 / 0.330 | Largest only: 0.860 / 0.190. Random sampling: 0.415 / 0.505. Each trained from scratch: 0.860 / 0.570 | Negative | Negative at this budget; the test only checks that shrinking beats cutting the smallest subnet out of the largest-only network. The accuracy predictor's Kendall τ ranges from −0.3 to 0.7 across runs. The trainer uses the later "sandwich" rule, not plain OFA. |
| Sparse update, int8 with QAS (budget 9,330 B = 30% of full fine-tuning) | Training memory; accuracy | 9,244 B; 0.700 | Full float fine-tuning: 31,100 B; 0.885. Last layer: 0.555. BitFit + head: 0.560. LoRA: 0.590. Lite residual: 16,388 B; 0.670. No QAS: 0.305. Float sparse update: 0.715. No adaptation: 0.12 | Win | Best under the budget, and QAS is what makes int8 training work. It is 0.18 short of full fine-tuning, not the paper's 1–2%. |
| DS-CNN, small budget (12 kB, 0.3M multiply-adds) | Accuracy, float / int8 | 0.917 / 0.900 | CNN: 0.975 / 0.983. DNN: 0.925 / 0.925. LSTM: 0.942. GRU: 0.925. CRNN: 0.833. Unconstrained: MLP + CTC 0.908, Conformer + CTC 0.825, MFCC + DTW 0.858 | Negative | Negative: the plain CNN, not the DS-CNN, is best. Every family fits the budget, and int8 costs at most 0.017. |

Not done: no receptive-field redistribution in patch inference; no LsqDense or PACT baselines; no ENAS or ClassicCnnShootout baselines; 6.16 keeps activations in float; the recurrent keyword spotters stay in float; the medium and large keyword budgets are not run in the tests; no PCEN front end.

---

## Wave 10

Wave 10 adds 101 techniques in six work packages (WP1–WP6, items 10.1.1 to 10.6.16 in `wave10-ideas.md`). An item is tagged **W10** in the same way as earlier waves, so "W10 1.3" is item 10.1.3. The six work-package branches are merged on the local branch `wave10-int`, which builds on wave 9's `claude/project-thread-8zvcl0` and has not been pushed. Wave 10 adds 106 test classes and eight new themes (events, labels, metric, multitask, parsing, restoration, shift and speech), and extends fourteen existing ones (alignment, audio, classic, control, document, edge, graphs, interp, language, multimodal, parallel, reasoning, selfsupervised and vision). As in wave 9, the section has one part per work package, and the theme of each experiment is given on its Sources line. Twelve items are fair-test follow-ups of earlier negatives (10.3.17, 10.4.13 to 10.4.16 and 10.6.10 to 10.6.16).

Every wave 10 number is keyed **R-wave10**. It comes from the final report of the work-package agent that built the item. The user asked for no extensive test runs, so each agent ran only its own test classes, one or two at a time with `--tests` filters, in the cloud container on 2026-11-18, and all of them passed when last run. No full suite run has been made, neither of a single branch nor of the merged `wave10-int`, so there is no Run-w10 key. Two caveats apply to every row:

- **All numbers are test-budget and single-seed** (seed 1 unless a row says otherwise). Gaps of a few points are within noise, and several rows are probes at sanity budgets.
- **The long-budget `main()` Runs were written but not executed.** Each experiment has a `main()` with more seeds, larger budgets and the full grids, and most of the wave 9 follow-ups (10.4.14 to 10.4.16, 10.6.10 to 10.6.14 and 10.6.16) state their hypotheses at those budgets. None of them was run, because the user asked for no extensive test runs. A follow-up whose hypothesis needs the long budget is marked **Inconclusive (long run pending)**, unless the test-budget numbers already settle it.

Some test classes were changed after the run that produced their numbers (budgets lowered, or a class split in two) and were not run again; each WP part says which. Where a number comes from an earlier or longer run than the test's, the row says so.

The **Verdict** column reads as follows:
- **Win:** the technique beats its baseline clearly.
- **Mixed:** it wins on some measures or settings and loses on others.
- **Negative:** it does not beat its baseline, or the hypothesis in `wave10-ideas.md` did not hold. These rows are kept and labelled.
- **Inconclusive:** the test is only a smoke run, so the numbers cannot decide. **Inconclusive (long run pending)** means the hypothesis needs a budget that only the unexecuted `main()` Run provides.

### Wave 10 verdicts by item

| WP | Win | Mixed | Negative | Inconclusive | Items |
|---|---|---|---|---|---|
| WP1 imperfect supervision | 6 | 6 | 5 | 0 | 17 |
| WP2 distribution shift and the open world | 7 | 5 | 5 | 0 | 17 |
| WP3 representation learning | 1 | 8 | 8 | 0 | 17 |
| WP4 deep RL recipes and follow-ups | 2 | 6 | 3 | 6 | 17 |
| WP5 boosting, events, graphs, parsing | 8 | 4 | 5 | 0 | 17 |
| WP6 restoration, speech, perception follow-ups | 4 | 5 | 3 | 4 | 16 |
| **Total** | **28** | **34** | **29** | **10** | **101** |

Of the 10 Inconclusive items, 9 are long runs pending (10.4.3, 10.4.4, 10.4.14, 10.4.15, 10.4.16, 10.6.11, 10.6.12, 10.6.13 and 10.6.16) and 1 (10.4.12) is a smoke run only. Four other follow-ups also have long runs pending, but their test-budget numbers already decide the hypothesis, or the part of it that a test can reach: 10.3.17 (Negative), 10.4.13 (Mixed), 10.6.10 (Win) and 10.6.14 (Negative).

| Item | Technique | Experiment | Verdict |
|---|---|---|---|
| W10 1.1 | Pseudo-labelling, Π-model, Mean Teacher | SemiSupervisedShootout | Negative |
| W10 1.2 | Virtual adversarial training | SemiSupervisedShootout | Mixed |
| W10 1.3 | MixMatch | SemiSupervisedShootout | Negative |
| W10 1.4 | FixMatch | SemiSupervisedShootout | Win |
| W10 1.5 | FlexMatch, FreeMatch, SoftMatch | SemiSupervisedShootout | Mixed |
| W10 1.6 | JEPAMatch | SemiSupervisedShootout | Negative |
| W10 1.7 | GCE, SCE, active–passive losses, JAL | NoisyLabelShootout | Mixed |
| W10 1.8 | Co-teaching and DivideMix | NoisyLabelShootout | Win |
| W10 1.9 | Early-learning regularisation | NoisyLabelShootout | Win |
| W10 1.10 | Transition-matrix loss correction | NoisyLabelShootout | Negative |
| W10 1.11 | Logit adjustment (additive, balanced softmax, multiplicative) | LongTailShootout | Win |
| W10 1.12 | LDAM-DRW, cRT, τ-normalisation | LongTailShootout | Mixed |
| W10 1.13 | uPU, nnPU, Elkan–Noto prior, 2-HNC | WeakLabelShootout | Mixed |
| W10 1.14 | Data programming and Dawid–Skene | WeakLabelShootout | Win |
| W10 1.15 | Attention-based multiple-instance learning | WeakLabelShootout | Win |
| W10 1.16 | PRODEN partial-label learning | WeakLabelShootout | Mixed |
| W10 1.17 | Learning from label proportions (DLLP) | WeakLabelShootout | Negative |
| W10 2.1 | MSP, max logit, energy, ODIN | OodShootout | Win |
| W10 2.2 | Mahalanobis, k-NN, ViM | OodShootout | Win |
| W10 2.3 | ReAct, ASH, SCALE, AdaSCALE | OodShootout | Negative |
| W10 2.4 | Outlier exposure | OodShootout | Win |
| W10 2.5 | OpenMax | OodShootout | Negative |
| W10 2.6 | SelectiveNet, Deep Gamblers, likelihood-ratio rejection | SelectivePrediction | Mixed |
| W10 2.7 | Learning to defer | SelectivePrediction | Negative |
| W10 2.8 | BBSE, MLLS, online Dirichlet-prior estimation | LabelShift | Win |
| W10 2.9 | DANN and CDAN | DomainAdaptation | Mixed |
| W10 2.10 | Deep CORAL and MMD | DomainAdaptation | Win |
| W10 2.11 | SHOT | DomainAdaptation | Mixed |
| W10 2.12 | MixStyle and SWAD | DomainGeneralization | Negative |
| W10 2.13 | Last-layer retraining (DFR) | DomainGeneralization | Mixed |
| W10 2.14 | KLIEP, uLSIF, classifier odds, weighted conformal | DomainGeneralization | Win |
| W10 2.15 | EATA, SAR, CoTTA | DomainAdaptation | Negative |
| W10 2.16 | WiSE-FT and LP-FT | DomainGeneralization | Mixed |
| W10 2.17 | SynthID-Text tournament watermark | WatermarkShootout | Win |
| W10 3.1 | SimSiam | SelfSupervisedBenchmark | Mixed |
| W10 3.2 | SwAV | SelfSupervisedBenchmark | Negative |
| W10 3.3 | MoCo v2 and v3 | SelfSupervisedBenchmark | Negative |
| W10 3.4 | data2vec and iBOT | SelfSupervisedBenchmark | Negative |
| W10 3.5 | KoLeo regulariser | SelfSupervisedBenchmark | Mixed |
| W10 3.6 | LeJEPA (SIGReg) | SelfSupervisedBenchmark | Mixed |
| W10 3.7 | Supervised contrastive learning | MetricLearningShootout | Mixed |
| W10 3.8 | Triplet with mining, multi-similarity | MetricLearningShootout | Negative |
| W10 3.9 | CosFace and ArcFace | MetricLearningShootout | Negative |
| W10 3.10 | Proxy-NCA and Proxy-Anchor | MetricLearningShootout | Mixed |
| W10 3.11 | DEC and IDEC | DeepClustering | Negative |
| W10 3.12 | SCAN and IIC | DeepClustering | Negative |
| W10 3.13 | Uncertainty weighting, GradNorm, DWA | GradientBalancing | Mixed |
| W10 3.14 | PCGrad and GradOPS | GradientBalancing | Mixed |
| W10 3.15 | MGDA-UB, CAGrad, Nash-MTL | GradientBalancing | Win |
| W10 3.16 | FAMO | GradientBalancing | Mixed |
| W10 3.17 | CCS with normalisation, prompt ensembles and CRC (follow-up) | LatentKnowledge | Negative |
| W10 4.1 | TD3+BC | OfflineRlComparison | Win |
| W10 4.2 | CrossQ | ModernOffPolicy | Mixed |
| W10 4.3 | BRO | ModernOffPolicy | Inconclusive (long run pending) |
| W10 4.4 | SimbaV2 | ModernOffPolicy | Inconclusive (long run pending) |
| W10 4.5 | PQN | ValueBasedRecipes | Negative |
| W10 4.6 | MR.Q | ModernOffPolicy, ValueBasedRecipes | Mixed |
| W10 4.7 | TD7 | ModernOffPolicy | Win |
| W10 4.8 | SPR and BBF | ValueBasedRecipes | Mixed |
| W10 4.9 | DrQ and DrQ-v2 | ValueBasedRecipes | Mixed |
| W10 4.10 | ReDo, plasticity injection, resets | PlasticityAndScale | Mixed |
| W10 4.11 | Multi-skip residuals with Kron | PlasticityAndScale | Negative |
| W10 4.12 | EfficientZero | ValueBasedRecipes | Inconclusive |
| W10 4.13 | GAIL with GP and absorbing states, AIRL, IQ-Learn (follow-up) | GoalsImitationAndTeams | Mixed |
| W10 4.14 | Continuous control at a fair budget (follow-up) | ModernOffPolicy | Inconclusive (long run pending) |
| W10 4.15 | Reward ensembles and WARM with stronger proxies (follow-up) | RewardOveroptimization | Inconclusive (long run pending) |
| W10 4.16 | VinePPO, PRIME, Step-DPO at a larger budget (follow-up) | ProcessSupervision | Inconclusive (long run pending) |
| W10 4.17 | Value-based scaling and TD-overfitting | PlasticityAndScale | Negative |
| W10 5.1 | LightGBM: GOSS, EFB, leaf-wise growth | BoostingShootout | Win |
| W10 5.2 | CatBoost: ordered boosting and target statistics | BoostingShootout | Negative |
| W10 5.3 | NGBoost | BoostingShootout | Mixed |
| W10 5.4 | Explainable boosting machines | BoostingShootout | Win |
| W10 5.5 | PFN-Boost | BoostingShootout | Mixed |
| W10 5.6 | Quantile regression forests | BoostingShootout | Negative |
| W10 5.7 | Hawkes processes with flexible kernels | PointProcessShootout | Win |
| W10 5.8 | RMTPP, neural Hawkes, transformer Hawkes | PointProcessShootout | Win |
| W10 5.9 | Intensity-free point processes | PointProcessShootout | Negative |
| W10 5.10 | GRU-D, ODE-RNN, latent ODE, mTAN | IrregularSeriesShootout | Negative |
| W10 5.11 | BRITS, SAITS, CSDI | IrregularSeriesShootout | Negative |
| W10 5.12 | TGN against EdgeBank, probe suite | TemporalGraphShootout | Mixed |
| W10 5.13 | R-GCN and CompGCN | RelationalGraphs | Win |
| W10 5.14 | Neural Bellman–Ford networks | RelationalGraphs | Win |
| W10 5.15 | Poincaré and Lorentz embeddings | RelationalGraphs | Win |
| W10 5.16 | Biaffine dependency parsing with MST decoding | ParsingShootout | Win |
| W10 5.17 | Span-based chart parsing | ParsingShootout | Mixed |
| W10 6.1 | SRCNN, ESPCN, EDSR, SPAN | RestorationShootout | Win |
| W10 6.2 | DnCNN and FFDNet | RestorationShootout | Mixed |
| W10 6.3 | Noise2Noise, Neighbor2Neighbor, zero-shot N2N, Median2Median | RestorationShootout | Mixed |
| W10 6.4 | Plug-and-play and unrolled deblurring | RestorationShootout | Negative |
| W10 6.5 | x-vectors with AAM-softmax, GE2E | SpeakerShootout | Win |
| W10 6.6 | Voice activity detection | SpeechFrontEnds | Mixed |
| W10 6.7 | Speech enhancement | SpeechFrontEnds | Mixed |
| W10 6.8 | Speaker diarisation (clustering vs EEND) | SpeakerShootout | Mixed |
| W10 6.9 | FlexiViT | VisionShootout | Negative |
| W10 6.10 | DETR with query selection (follow-up) | DetectionShootout | Win |
| W10 6.11 | Donut with reading pretraining (follow-up) | DocumentParsing | Inconclusive (long run pending) |
| W10 6.12 | Two-stage VQA bridges with LoRA (follow-up) | FusionShootout | Inconclusive (long run pending) |
| W10 6.13 | Transfusion with VAE latents and U-Net patches (follow-up) | FusionShootout | Inconclusive (long run pending) |
| W10 6.14 | BigNAS in-place distillation (follow-up) | EdgeDeployment | Negative |
| W10 6.15 | V-shape schedules and an exact ZB-H2 scheduler (follow-up) | ParallelismShootout | Win |
| W10 6.16 | GAN vocoder with mel-loss warm-up (follow-up) | AudioGeneration | Inconclusive (long run pending) |

---

## Wave 10 WP1: semi-supervised learning, noisy labels, long tails, weak and bag-level labels

New packages: `experimental.semisup`, `experimental.labels` and `experiments.labels`. A new generator, `experimental.data.LabelCorruption`, leaves `ToyTasks`, `ShapeImages` and `TabularStressSuite` unchanged, and `experimental.interp.AttentionRollout` gains a public `minCutSourceSide` (its `maxFlow` is unchanged). Unless a row says otherwise, the task is the 8×8 `ToyTasks` digits (10 classes, 1,000 training rows) with the same MLP, Adam and seed 1. The numbers come from the builder's last run of all 18 of its test classes, which took 4 to 17 s each on the loaded machine. After that run, four classes over 10 s were split into smaller ones. Of the new classes only DivideMixShootoutTest has been run; the others carry over assertions whose numbers were already measured, and they compile (R-wave10).

### SemiSupervisedShootout (W10 1.1: pseudo-labelling, Π-model and Mean Teacher, 1.2: VAT, 1.3: MixMatch, 1.4: FixMatch, 1.5: FlexMatch, FreeMatch and SoftMatch, 1.6: JEPAMatch)
Sources: R-wave10. Theme: labels. Tests: SemiSupervisedShootoutTest, ThresholdedSemiSupervisedTest, ImbalancedPoolSemiSupervisedTest, JepaMatchShootoutTest, SemiSupervisedTabularTest, SemiSupervisedBlocksTest. Blocks: `ConsistencyTraining`, `VirtualAdversarial`, `MixMatch`, `FixMatch` (with a pluggable `Mask`), `AdaptiveThreshold` (FLEX, FREE and SOFT modes), `JepaMatch`, `SoftTargets`. 4 labels per class, 400 steps, balanced test accuracy. The main baseline is supervised training with augmentation.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Pseudo-label; Π-model; Mean Teacher | Balanced accuracy | 0.537; 0.570; 0.567 | Supervised + augmentation: 0.567. Supervised, plain: 0.387. Oracle (all labels): 0.847 | Negative | Negative: none of the three beats augmented supervised training, and pseudo-labelling falls below it. |
| VAT on digits; on spirals (20 labels per class); on TabularStressSuite | Accuracy | 0.303; 0.989; 0.585 | Digits: supervised + augmentation 0.567. Spirals: supervised 0.955, Mean Teacher 0.966, Π-model 0.994, oracle 0.993. Tabular: supervised 0.625, oracle 0.658 | Mixed | VAT works where the cluster assumption holds (spirals) and fails on digits. On the tabular suite, where it was expected to win, it loses; that suite is too hard to separate any methods. |
| MixMatch | Balanced accuracy | 0.540 | Supervised + augmentation: 0.567. Mean Teacher: 0.567 | Negative | Negative: below both. The sharpening ablation was not reported. |
| FixMatch, τ = 0.95; τ = 0.7 | Balanced accuracy | 0.643; 0.653 | Supervised + augmentation: 0.567 | Win | The strongest family at 4 labels per class, level with FreeMatch. At τ = 0.95 only 4% of unlabelled rows pass the threshold, at pseudo-label precision 0.85. On the imbalanced pool (ratio 10) it scores 0.607 with a worst class of 0.267, as predicted for rare classes. |
| FlexMatch; FreeMatch; SoftMatch | Balanced accuracy (worst class) | 0.620 (0.200); 0.653 (0.433); 0.623 | FixMatch, τ = 0.95: 0.643 | Mixed | FreeMatch ties FixMatch. FlexMatch never leaves its warm-up at 400 steps: its thresholds stay near 0, so it keeps every row at pseudo-label precision 0.27. On the imbalanced pool FlexMatch is worse than FixMatch, 0.533 (worst class 0.200) against 0.607 (0.267). Convergence speed was not measured. |
| JEPAMatch (SIGReg); FlexMatch + VICReg | Balanced accuracy | 0.457; 0.533 | FlexMatch: 0.620 | Negative | Negative: both latent regularisers lower accuracy. |

Item verdicts: 1.1 Negative, 1.2 Mixed, 1.3 Negative, 1.4 Win, 1.5 Mixed, 1.6 Negative.

Not done: the entropy-based `ActiveLearner` baseline is built but not run in the tests; no ShapeImages rows; FreeMatch's self-adaptive fairness term and SoftMatch's uniform alignment are left out; `JepaMatch.sigreg` is a local Epps–Pulley copy, to be swapped for WP3's `experimental.ssl.Sigreg`.

### NoisyLabelShootout (W10 1.7: noise-robust losses, 1.8: co-teaching and DivideMix, 1.9: early-learning regularisation, 1.10: transition-matrix correction)
Sources: R-wave10. Theme: labels. Tests: RobustLossShootoutTest, SampleSelectionShootoutTest, DivideMixShootoutTest, TransitionCorrectionShootoutTest, LabelBlocksTest. Blocks: `RobustLosses` (GCE, SCE, NCE+RCE, JAL as NCE+AMSE, MAE, focal), `SampleSelection` (co-teaching schedule and small-loss pick, co-teaching+ disagreement, GMM clean probability, DivideMix label refinement), `EarlyLearningRegularizer`, `TransitionCorrection` (forward and backward correction, percentile-anchor and mean-anchor estimators, T-revision). 40% label noise.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| GCE; SCE; NCE+RCE; JAL; MAE, 40% symmetric noise | Final accuracy (share of flipped labels memorised) | GCE 0.550 (0.45). SCE 0.597. NCE+RCE 0.570 (0.36). JAL 0.583 (0.39). MAE 0.567 | CE: 0.567 (0.94). Label smoothing: 0.563. Focal: 0.520 | Mixed | The robust losses memorise far fewer flipped labels but gain at most 0.03 accuracy at this budget. CE does not collapse at 40% noise. |
| NCE+RCE; JAL, 40% pair-flip noise | Final accuracy | 0.477; 0.497 | CE: 0.527 | Negative | Negative: both lose to CE, although JAL beats the active–passive loss, as predicted. |
| Co-teaching; co-teaching+; DivideMix (2 networks + MixMatch) | Final accuracy (selection precision / recall) | 0.637 (0.84 / 0.87); 0.583; 0.727 (0.93 / 0.79) | Best robust loss (SCE): 0.597. CE: 0.567. EL2N pruning (`DataPruner`): 0.653 (0.79 / 0.81) | Win | Co-teaching beats every robust loss and DivideMix is best overall. EL2N's selection is less precise. |
| ELR | Final accuracy; flipped examples predicted correctly | 0.683; 49% | CE: 0.567; 3%. CE with early stopping on a noisy validation split: 0.503. Co-teaching: 0.637 | Win | One network beats co-teaching. It recovers 49% of the flipped examples, short of the hypothesised 80%. |
| Forward correction with estimated T; backward; T-revision, 40% pair-flip | Final accuracy | 0.517; 0.433; 0.510 | CE: 0.527. Forward with the true T: 0.733 | Negative | Negative: the estimated T (mean absolute error 0.063, diagonal about 0.7 against a true 0.6) throws the whole gain away. Under 40% instance-dependent noise, forward correction with estimated T does help (0.637 against 0.547 for CE), the reverse of the hypothesis. |

Item verdicts: 1.7 Mixed, 1.8 Win, 1.9 Win, 1.10 Negative.

Not done: no 60% noise rows were reported; JAL uses a = 10 rather than the paper's a = 30; DivideMix's compute cost was not measured.

### LongTailShootout (W10 1.11: logit adjustment, 1.12: LDAM-DRW and classifier re-training)
Sources: R-wave10. Theme: labels. Test: LongTailShootoutTest. Blocks: `LogitAdjustment` (post-hoc additive, adjusted loss and balanced softmax, multiplicative on cosine scores, grid tuning), `MarginAndDecoupling` (LDAM, DRW, class-balanced sampling, cRT, τ-normalisation). Imbalance ratio 100.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Post-hoc additive (τ = 1.0); logit-adjusted loss (= balanced softmax); post-hoc multiplicative (γ = 0.6) | Balanced accuracy (tail accuracy) | 0.617 (0.244); 0.623 (0.278); 0.607 | CE: 0.563 (0.100). Reweight: 0.580. Oversample: 0.593. Focal: 0.533 | Win | Every adjustment beats CE and reweighting, loss-time adjustment is best, and the additive and multiplicative post-hoc forms are within 0.01, as predicted. Over oversampling the gain is 0.014–0.030, under the hypothesised 3 points for two of the three. |
| LDAM; LDAM-DRW; cRT; τ-norm | Balanced accuracy | 0.580; 0.607; 0.550; 0.567 | CE: 0.563 | Mixed | LDAM-DRW gains 0.044 over CE; the decoupled classifiers cRT and τ-norm do not beat CE. Tail accuracy was not reported for these rows. |

Item verdicts: 1.11 Win, 1.12 Mixed. The multiplicative form (n^−γ on cosine logits, γ around 0.5) was checked against the paper's PDF.

### WeakLabelShootout (W10 1.13: PU learning, 1.14: labelling functions, 1.15: attention MIL, 1.16: PRODEN, 1.17: label proportions)
Sources: R-wave10. Theme: labels. Tests: PositiveUnlabelledTest, LabelModelShootoutTest, AttentionMilShootoutTest, MilSetBaselinesTest, PartialLabelShootoutTest, LabelProportionShootoutTest. Blocks: `PositiveUnlabeled` (biased PN, uPU, nnPU, Elkan–Noto prior, 2-HNC), `LabelModel` (majority vote, Dawid–Skene, data programming with an optional correlated pair, triplet matrix completion, IRT weighting through `IrtEvaluator`), `AttentionMil` (a core `Layer` with a `LayerSpec`, same contract as `SetEncoder`), `PartialLabels` (PRODEN, uniform, CC), `LabelProportions` (DLLP loss, OT pseudo-labels through `EntropicTransport.sinkhorn`, mean map).

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| uPU; nnPU; nnPU with the Elkan–Noto prior; 2-HNC (transductive), even vs odd digits, 10% of positives labelled | Test accuracy | 0.517 (risk estimate −0.449); 0.683; 0.500; 0.530 | PN oracle: 0.937. Biased PN: 0.667. Isolation Forest: 0.593. Deep SVDD: 0.540 | Mixed | uPU's risk goes negative and nnPU fixes it, but nnPU stays 0.25 below PN, not within 3 points. The estimated prior is 0.27 against a true 0.48. 2-HNC suffers from pixel-space neighbour graphs on shifted digits. |
| Data programming with the correlated pair; matrix completion; Dawid–Skene; plain data programming; IRT, duplicated weak function | Accuracy on covered rows (accuracy-estimation error) | 0.891 (0.015); 0.846 (0.027); 0.798; 0.808; 0.785 | Majority vote: 0.836. End model on 5 gold labels per class: 0.347 | Win | Modelling the correlated pair beats majority vote by 0.055. Without it, Dawid–Skene and data programming are misled by the duplicate and lose to the vote. The end model trained on the label model's labels scores 0.830. When the duplicate is a strong function, every method scores 0.90–0.92. |
| Gated attention; plain attention, 200 bags | Bag AUC (instance AUC) | 0.910 (0.922); 0.719 (0.643) | Mean pooling: 0.797. Max: 0.856. Deep Sets: 0.829. Set Transformer: 0.557. Instance-level oracle: 0.983 | Win | Gated attention wins and finds the key instance (instance AUC above 0.9), as predicted; ungated attention is worse than mean pooling. |
| PRODEN, 3 distractors on average; with biased distractors | Test accuracy | 0.730; 0.730 | Supervised: 0.877. CC: 0.713. Uniform: 0.580. Single-candidate rows only: 0.510. GCE on a random candidate: 0.447 | Mixed | PRODEN beats every partial-label baseline but is 0.15 below supervised, not within 3 points. Biased distractors did not hurt it. |
| DLLP; DLLP + OT pseudo-labels, random bags of 8 | Test accuracy | 0.687; 0.667 | Supervised: 0.880. Bag-majority labels: 0.327. Mean map: 0.427 | Negative | Negative: DLLP beats the naive baselines but is far from supervised, OT does not help, and at bag size 128 every method collapses to 0.13–0.21. |

Item verdicts: 1.13 Mixed, 1.14 Win, 1.15 Win, 1.16 Mixed, 1.17 Negative.

Not done: 2-HNC is built from the abstract only (a parametric degree-penalised min cut, two stages, prior matching), so the paper's parametrisation may differ; the biased-PN baseline uses a local prior-matched threshold, not `Calibration`'s.

---

## Wave 10 WP2: OOD detection, selective prediction, label shift, domain adaptation and generalisation, watermarking

New packages: `experimental.ood`, `experimental.shift` and `experiments.shift`. Also new: the generator `experimental.data.ShiftSuites` (shapes and digits in 7 styles, near- and far-OOD sets, priors, backgrounds, two moons and Shimodaira covariate shift), `experimental.llm.TournamentWatermark` and `experiments.language.WatermarkShootout`. `CausalAndFairness.colouredShapes` becomes a public accessor with its sampling unchanged. All 13 WP2 test classes passed when last run, each in 10 s or less (R-wave10).

### OodShootout: post-hoc scores (W10 2.1: MSP, max logit, energy and ODIN, 2.2: Mahalanobis, k-NN and ViM, 2.3: activation shaping)
Sources: R-wave10. Theme: shift. Test: OodShootoutTest. Blocks: `ood.LogitScores`, `ood.FeatureScores`, `ood.TiedGaussian`, `ood.FeatureClassifier`, `ood.ActivationShaping`, `ood.OodMetrics`. Task: shapes, in-distribution accuracy 0.910.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Max logit; energy; ODIN | AUROC, near / far OOD | 0.829 / 0.577; 0.829 / 0.578; 0.826 / 0.532 | MSP: 0.804 / 0.536 | Win | Energy beats MSP by 0.025 near and 0.042 far, inside the predicted 2–5 points. ODIN helps near OOD, not far OOD as predicted. |
| k-NN; ViM; Mahalanobis | AUROC, near / far | 0.839 / 0.704; 0.836 / 0.645; 0.588 / 0.668 | Energy: 0.829 / 0.578. GMM on pixels: 0.759 / 0.985 | Win | k-NN and ViM edge out every logit score near OOD (by 0.01, within noise) and clearly beat them far. Mahalanobis is brittle near OOD, as predicted. A pixel-space GMM wins far OOD outright. |
| ReAct; SCALE; ASH-P, ASH-B, ASH-S; AdaSCALE-A, AdaSCALE-L | AUROC, near / far | ReAct 0.832 / 0.589. SCALE 0.793 / 0.525. ASH-P 0.765 / 0.537 (accuracy falls to 0.853), ASH-B 0.746 / 0.511, ASH-S 0.687 / 0.477. AdaSCALE-A 0.683 / 0.561, AdaSCALE-L 0.687 / 0.559 | Energy: 0.829 / 0.578 | Negative | Negative: only ReAct beats energy, and only slightly; SCALE, ASH and AdaSCALE all lower it on this toy. |

Item verdicts: 2.1 Win, 2.2 Win, 2.3 Negative.

Not done: the ensemble and `UncertaintyBenchmark` baselines (DUQ, SNGP and others) are reported only in `main()`.

### OodShootout: outlier exposure and open-set recognition (W10 2.4: outlier exposure, 2.5: OpenMax)
Sources: R-wave10. Theme: shift. Test: OodTrainingTest. Blocks: `ood.OutlierExposure`, `ood.OpenMax`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| OE with a uniform target; OE with the energy loss | Far AUROC (energy score) / accuracy | 0.881 / 0.923; 0.883 / 0.890 | Plain training: 0.580 / 0.898. Mixup: 0.729 / 0.928 | Win | The largest far-OOD gain in the WP, as predicted. The energy form drops near AUROC to 0.652. |
| OpenMax, well-trained (closed-set accuracy 0.908); briefly trained | AUROC; open-set accuracy | 0.730; 0.545. Brief: AUROC 0.660 | Max logit: 0.778; 0.556. Brief: 0.682. Conformal: AUROC 0.553 | Negative | Negative: OpenMax never beats max logit, not even at the short budget where it was expected to. The well-trained result matches Vaze et al. |

Item verdicts: 2.4 Win, 2.5 Negative.

### SelectivePrediction (W10 2.6: SelectiveNet, Deep Gamblers and likelihood-ratio rejection, 2.7: learning to defer)
Sources: R-wave10. Theme: shift. Test: SelectivePredictionTest. Blocks: `ood.SelectiveClassifier` (SelectiveNet, Deep Gamblers, Δ-MDS and Δ-KNN), `ood.LearningToDefer`. The classifier's accuracy is 0.622.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Δ-KNN + RLog; Δ-MDS + RLog; SelectiveNet; Deep Gamblers | AURC in distribution / shifted (lower is better) | 0.1865 / 0.4972; 0.1973 / 0.4829; 0.3267 / 0.5278; 0.2452 / 0.5753 | Softmax response: 0.1885 / 0.4949. MC dropout: 0.2033 / 0.5496. Ensemble ×3: 0.1245 / 0.4372 | Mixed | Softmax response is hard to beat, as in the literature, and the trained rejectors are clearly worse. Δ-MDS degrades least under shift, by a small margin. The 3-member ensemble beats everything at 3x the cost. |
| Softmax surrogate; one-vs-all, deferral budget 0.4 | System accuracy (ECE of the expert-correctness estimate) | 0.471 (0.220); 0.455 (0.140) | Confidence threshold: 0.487. Two-stage: 0.492. Classifier alone: 0.388. Expert alone: 0.519 | Negative | Negative: both learned rules lose to thresholding, and none of the rules beats the expert alone. One-vs-all is better calibrated, as predicted. |

Item verdicts: 2.6 Mixed, 2.7 Negative.

### LabelShift (W10 2.8: BBSE, MLLS and online Dirichlet-prior estimation)
Sources: R-wave10. Theme: shift. Test: LabelShiftTest. Block: `shift.LabelShiftEstimator`. Mean over five shifts.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| BBSE; RLLS; MLLS-TS; MLLS-BCTS; MAP-Dirichlet | Weight MSE; accuracy | 0.098; 0.765. 0.044; 0.768. 0.027; 0.768. 0.037; 0.766. 0.032; 0.769 | No correction: 1.456; 0.682. Oracle weights: 0; 0.768. BN statistics / TENT: accuracy 0.430 | Win | Calibrated MLLS estimates the weights 3.6x better than BBSE, and every correction recovers the oracle gain. Test-time BN adaptation hurts badly, as in wave 8. |
| Online FMAPLS-style tracking, drifting stream | Prior error; accuracy | 0.058; 0.660 | MLLS refit on each batch: 0.104; 0.618. No correction: accuracy 0.650 | Win | The online estimator tracks the drift better than per-batch refitting, at a fraction of the cost. |

Not done: FMAPLS uses a fixed Dirichlet α instead of estimating it (only the paper's abstract was available).

### DomainAdaptation (W10 2.9: DANN and CDAN, 2.10: CORAL and MMD, 2.11: SHOT, 2.15: EATA, SAR and CoTTA)
Sources: R-wave10. Theme: shift. Tests: DomainAdaptationTest, DomainAdaptationMoonsTest, ContinualAdaptationTest. Blocks: `shift.GradientReversal`, `shift.AdversarialAdaptation` (DANN, CDAN, CDAN-E), `shift.DiscrepancyAdaptation` (CORAL, MMD), `shift.SourceFreeAdaptation` (SHOT), `shift.RobustTestTimeAdaptation` (EATA, SAR, CoTTA). Tasks: shapes to textured shapes, and rotated two moons.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| DANN; CDAN; CDAN-E | Target accuracy, shapes → textured / moons | 0.406 / 0.956; 0.568 (earlier run; not run in the shapes test) / 0.938; 0.478 / 0.832 | Source only: 0.238 / 0.928. Oracle: 0.618 / 0.998 | Mixed | DANN closes 44% and 40% of the gap, inside the predicted 30–60%. CDAN does better on shapes but worse on moons, and CDAN-E falls below source-only on moons. |
| MMD; CORAL | Same | 0.612 / 0.934; 0.518 / 0.930 | Source only: 0.238 / 0.928. DANN: 0.406 / 0.956. AdaBN: 0.530 / 0.764. TENT: 0.530 / 0.772 | Win | On shapes MMD almost reaches the oracle and CORAL beats DANN. On moons both are level with source-only, so MMD's predicted gain there does not show. |
| SHOT | Same | 0.254 / 0.998 | TENT: 0.530 / 0.772. DANN: 0.406 / 0.956 | Mixed | Matches the oracle on moons without source data, but fails on shapes, where it barely beats source-only. |
| EATA; SAR; CoTTA, continual corruption stream | Accuracy | 0.549 (EATA adapts on 28% of samples); 0.549; 0.565 | No adaptation: 0.265. BN statistics: 0.550. TENT: 0.549 | Negative | Negative: TENT did not degrade over the stream, so the robust methods had nothing to fix and tie it. On single-class batches every adapter falls to about 0.21, below no adaptation (0.336), and SAR never resets. |

Item verdicts: 2.9 Mixed, 2.10 Win, 2.11 Mixed, 2.15 Negative.

Not done: the DIGITS_FONT task runs only in `main()`; TabularStressSuite is not covered; GradientReversal, the Weibull fit, MixStyle and DensityRatio have no unit tests of their own (the experiment tests exercise them).

### DomainGeneralization (W10 2.12: MixStyle and SWAD, 2.13: DFR, 2.14: density-ratio correction, 2.16: WiSE-FT and LP-FT)
Sources: R-wave10. Theme: shift. Tests: DomainGeneralizationTest, SpuriousCorrelationTest, CovariateShiftTest, RobustFineTuningTest. Blocks: `shift.MixStyle`, `shift.Swad`, `shift.LastLayerRetraining`, `shift.DensityRatio` (uLSIF, KLIEP, classifier odds, weighted conformal), `shift.RobustFineTuning`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| MixStyle; V-REx, TEXTURED held out, 180 steps; SWAD, 300 steps | Held-out domain accuracy | 0.330; 0.448; 0.337 | ERM: 0.340 at 180 steps, 0.38 at 300 | Negative | Negative: MixStyle and SWAD do not beat ERM, and V-REx, expected to stay below ERM, is the best row. |
| DFR, coloured bars; backgrounds | Average / worst-group accuracy | 0.601 / 0.478; 0.697 / 0.604 | ERM: 0.600 / 0.243; 0.633 / 0.272. GroupDRO: 0.644 / 0.384; 0.808 / 0.688 | Mixed | DFR doubles ERM's worst group on both tasks and beats GroupDRO on bars, but not on backgrounds. No JTT row was reported. |
| uLSIF; KLIEP; classifier odds; clipped uLSIF | Target MSE, dimension 1 / 5 | 0.095 / 0.179; 0.095 / 0.200; 0.096 / 0.202; 5-D only: 0.167 | No weighting: 0.387 / 1.897. Oracle weights: 0.095 / 0.211 | Win | Weighting removes most of the shift error, and uLSIF is the best estimator in 5 dimensions, as predicted. |
| Weighted split conformal, target 0.90 | Coverage (set size) | 0.905 (1.44) | Plain split conformal: 0.792 (1.02). Oracle weights: 0.907 | Win | Weighting restores nominal coverage where plain conformal under-covers. |
| LP-FT; WiSE-FT (α = 0.5); WiSE-(LP-FT) | In-distribution / shifted accuracy | 0.745 / 0.287; 0.505 / 0.275; 0.645 / 0.290 | Full fine-tuning: 0.600 / 0.253. Linear probe: 0.483 / 0.283. Soup of 3: 0.368 / 0.210 | Mixed | LP-FT is best on both measures, as predicted. WiSE-FT beats full fine-tuning on the shifted domain but not the linear-probe endpoint. All shifted gains are small (chance is 0.167). |

Item verdicts: 2.12 Negative, 2.13 Mixed, 2.14 Win, 2.16 Mixed.

Not done: SWAD and the full leave-one-out grid run only in `main()`, as do the JEM baselines; `DomainGeneralization.pretrain` is a local InfoNCE pretrainer, because `SelfSupervisedBenchmark.pretrainEncoder` belonged to WP3.

### WatermarkShootout (W10 2.17: SynthID-Text tournament sampling)
Sources: R-wave10. Theme: language. Tests: WatermarkShootoutTest, TournamentWatermarkTest. Block: `experimental.llm.TournamentWatermark`, a closed-form `LogitsProcessor` plus an explicit tournament sampler. The closed form matches the explicit 2³ tournament to within 0.002 of the empirical frequencies, and averaging over keys reproduces the LM's distribution. Task: `WindowLm` on `SyntheticCorpus` topic streams, T = 1.0.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Tournament watermark | TPR at 1% FPR: 25 tokens / 50 tokens / with 20% edits; judge log-likelihood | 0.43 / 0.91 / 0.35; −1.62 | Green list δ = 2: 0.32 / 0.45 / 0.23; −2.08. δ = 1: 0.06 / 0.11 / 0.06; −1.93. Unwatermarked: 0.00 / 0.02; −1.77 | Win | Twice as detectable as the strongest green list at 50 tokens, with no loss on the judge. Distinct-2 falls from 0.134 to 0.109, because the key per context is fixed. |

Not done: `WindowLm` sampling has no `LogitsProcessor` hook, so `WatermarkShootout.generate` is a local batched sampler.

---

## Wave 10 WP3: self-supervised follow-ups and LeJEPA, metric learning, deep clustering, multi-task balancing

New packages: `experimental.ssl`, `experimental.metric`, `experimental.multitask`, `experiments.metric` and `experiments.multitask`, plus new classes in `experimental.ml`, `experimental.interp`, `experiments.selfsupervised` and `experiments.interp`. The slowest class, LatentKnowledgeFollowUpTest, takes about 15.5 s, over the 10 s target. All numbers are seed 1 (R-wave10).

### SelfSupervisedBenchmark (W10 3.1: SimSiam, 3.2: SwAV, 3.3: MoCo, 3.4: data2vec and iBOT, 3.5: KoLeo, 3.6: LeJEPA)
Sources: R-wave10. Theme: selfsupervised. Tests: SslBlocksTest, SimSiamTest, SwappedPredictionTest, MomentumContrastTest, MaskedFeaturePredictionTest, KoLeoTest, LeJepaTest. Blocks: `experimental.ssl.Sigreg`, `LeJepa`, `KoLeo`, `SimSiam`, `SwappedPrediction`, `MomentumContrast`, `MaskedFeaturePrediction`. `SelfSupervisedBenchmark` gains an `Extra` enum, `Options`, `report`, `run`, `pretrainEncoder` and `main`; the refactored `runJoint` keeps the old numbers (SimCLR 0.560, BYOL 0.424). 150 steps unless noted. Numbers are linear probe / kNN / effective rank; a random encoder probes at 0.341.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| SimSiam; without stop-gradient; without predictor | Probe / kNN / rank | 0.417 / 0.646 / 8.2; 0.445 / 0.484 / 5.7; 0.417 / 0.427 / 3.8 | BYOL: 0.424 / 0.622 / 7.5. SimCLR: 0.560 / 0.729 | Mixed | SimSiam matches BYOL without a momentum encoder, and both ablations lose kNN accuracy and rank, but neither fully collapses. Both trail SimCLR. |
| SwAV; with multi-crop | Probe / kNN (cluster purity) | 0.409 / 0.534 (0.385); 0.411 / 0.508 (0.344) | SimCLR: 0.560 / 0.729. DINO: 0.448 / 0.508 | Negative | Negative: SwAV does not beat SimCLR at small batch, and multi-crop adds nothing. |
| MoCo v2; v3, 200 steps, batch 16 | Probe / kNN | 0.477 / 0.641; 0.448 / 0.620 | SimCLR at batch 16: 0.510 / 0.677 | Negative | Negative: SimCLR stays ahead at batch 16, so the queue does not pay off here. |
| data2vec; iBOT, patch transformer, 80 / 100 steps | Probe | 0.404 / 0.435; 0.365 / 0.354 | MAE: 0.568 / 0.589 | Negative | Negative: both lose to MAE at the same budget. No iBOT-vs-DINO comparison was reported. |
| KoLeo on DINO; on SimSiam | Probe / kNN / rank (uniformity) | 0.456 / 0.708 / 20.7 (−1.77); 0.370 / 0.573 / 27.4 | DINO: 0.448 / 0.508 / 6.4 (−0.19). SimSiam: 0.417 / 0.646 / 8.2 | Mixed | On DINO a clear gain: kNN +0.20 and three times the rank, with the probe unchanged. On SimSiam the rank rises but accuracy falls. |
| LeJEPA, MLP, 32 slices; 128 slices; on patches, 60 steps | Probe / kNN / rank | 0.451 / 0.755 / 20.9; probe 0.518; probe 0.323 | SimCLR: 0.560 / 0.729. I-JEPA-lite on patches: 0.380. VICReg regulariser in place of SIGReg: 0.281, rank 2.0 (collapsed) | Mixed | The best kNN on the MLP with no EMA teacher, and the VICReg variant collapses, as predicted. It loses to I-JEPA-lite on patches, and its probe trails SimCLR. The loss–probe correlation was not reported. |

Item verdicts: 3.1 Mixed, 3.2 Negative, 3.3 Negative, 3.4 Negative, 3.5 Mixed, 3.6 Mixed.

Not done: the data2vec audio row on SpokenDigitRecognition.

### MetricLearningShootout (W10 3.7: SupCon, 3.8: pair losses with mining, 3.9: CosFace and ArcFace, 3.10: proxy losses)
Sources: R-wave10. Theme: metric. Tests: MetricBlocksTest, SupervisedContrastiveTest, PairLossesTest, MarginSoftmaxTest, ProxyLossesTest. Blocks: `experimental.metric.SupervisedContrastive`, `PairLosses`, `MarginSoftmax` (implements `Loss` and exposes `parameters()`), `ProxyLosses`, `RetrievalMetrics`. Task: `DigitGlyphs`, trained on classes 0–5 and tested on unseen classes 6–9, 600 steps. ToyTasks digits were dropped because an untrained encoder already reached R@1 0.90 on them. Numbers are unseen-class R@1 / NMI / EER.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| SupCon; SupCon + KoLeo | Unseen R@1 / NMI / EER (seen R@1) | 0.563 / 0.375 / 0.315 (0.550); 0.556 / 0.352 / 0.325 (0.606) | Cross-entropy: 0.625 / 0.358 / 0.338 (0.672). SimCLR: 0.581 / 0.027 / 0.463. Prototypical: 0.475 / 0.277 / 0.394. Untrained: 0.494 / 0.035 / 0.492 (0.400) | Mixed | SupCon edges cross-entropy on NMI and EER but loses on R@1, for seen and unseen classes. |
| Contrastive; triplet with random / semi-hard / hard mining; multi-similarity | Unseen R@1 | 0.569; 0.500 / 0.494 / 0.456; 0.513 | Cross-entropy: 0.625 | Negative | Negative: semi-hard mining does not beat random triplets, hard mining collapses, and no pair loss reaches cross-entropy. |
| Normalised softmax; CosFace; ArcFace; sub-center ArcFace | Unseen R@1 / EER | 0.594 / 0.356; 0.531 / 0.331; 0.531 / 0.348; 0.531 / 0.395 | SupCon: 0.563 / 0.315 | Negative | Negative: the margins improve EER over plain normalised softmax by at most 0.025 and lower R@1, and SupCon has the better EER. |
| Proxy-NCA; Proxy-NCA++; Proxy-Anchor | Unseen R@1 / EER | 0.550 / 0.308; 0.600 / 0.350; 0.569 / 0.418 | Multi-similarity: 0.513. Contrastive: 0.569. Cross-entropy: 0.625 | Mixed | The proxy losses tie or beat the pair losses, and Proxy-Anchor is at its final R@1 by step 25 (0.506 / 0.538 / 0.525 / 0.525 at 25 / 50 / 100 / 200 steps, against 0.500 / 0.469 / 0.506 / 0.481 for multi-similarity). But it barely leaves the untrained level (0.494) and has the worst EER. |

Item verdicts: 3.7 Mixed, 3.8 Negative, 3.9 Negative, 3.10 Mixed. No metric loss beats plain cross-entropy on unseen classes in this small setting.

Not done: the RelationalKdLoss baseline; the ShiftSuites corruption check (a seen-class check with heavier flips is used instead).

### DeepClustering (W10 3.11: DEC and IDEC, 3.12: SCAN and IIC)
Sources: R-wave10. Theme: metric. Test: DeepClusteringTest. Blocks: `experimental.ml.ClusteringMetrics` (Hungarian accuracy, NMI, ARI), `DeepEmbeddedClustering` (DEC and IDEC), `NeighbourClustering` (IIC and SCAN). Task: digits, 40 per class.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| DEC; IDEC | Accuracy / NMI | 0.265 / 0.203; 0.268 | k-means on autoencoder latents: 0.270. On raw inputs: 0.260 / 0.225. GMM: 0.255. Spectral: 0.265 | Negative | Negative: no gain over k-means on the same latents. |
| IIC; SCAN; SCAN + self-labelling, on SimCLR features after 100 steps | Accuracy / NMI | 0.315 / 0.268; 0.295; 0.293 | k-means on the same features: 0.270. Spectral: 0.285. Neighbour purity: 0.326 | Negative | Negative: SCAN gains 0.025, not 5 points, and at 150 steps k-means on the same features (0.350) beats SCAN (0.330). IIC is the best row. |

Item verdicts: 3.11 Negative, 3.12 Negative.

Not done: the DirichletProcessMixture row.

### GradientBalancing (W10 3.13: uncertainty weighting, GradNorm and DWA, 3.14: PCGrad and GradOPS, 3.15: MGDA, CAGrad and Nash-MTL, 3.16: FAMO)
Sources: R-wave10. Theme: multitask. Tests: MultitaskBlocksTest, LossWeightingTest, GradientSurgeryTest, ParetoDescentTest, FamoTest. Blocks: `experimental.multitask.TaskGradients`, `GradientSurgery` (PCGrad, GradVac, GradOPS), `ParetoDescent` (MGDA, CAGrad, Nash-MTL), `LossWeighting` (equal, uncertainty, GradNorm, DWA), `Famo`. 250 steps. Numbers are Δm against single-task training (closer to 0 is better). Tasks: SCALES (a loss-scale mismatch), DIGITS, CLICKS and a two-task toy surface.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Uncertainty; GradNorm; DWA | Δm, SCALES / DIGITS | −33.3% / −7.6%; −34.1% / −9.6%; −40.0% / −8.3% | Equal weights: −41.8% / −8.8% | Mixed | With a scale mismatch every weighting beats equal weights, as predicted; on DIGITS only uncertainty and DWA do, by about a point. On CLICKS everything is within noise (equal −0.2%, uncertainty −0.2%, MMoE +0.1%, MMoE + uncertainty +0.1%), so MMoE and weighting gains do not add up. |
| PCGrad; GradOPS | Starts reaching the Pareto set (of 5); Δm, DIGITS | 4; 4. −7.2%; −9.2% | Equal: 3; −8.8%. GradVac: −9.1% | Mixed | Both reach the front more often on the toy (they are identical there, as the paper predicts for two tasks and α = 0), but only PCGrad gains on DIGITS. The task gradients conflict on about 25% of steps. The 4-task order-dependence test was not run. |
| Nash-MTL; CAGrad; MGDA; MGDA-UB | Δm, DIGITS | −3.5%; −6.0%; −51.7%; −75.4% | Equal: −8.8% | Win | As predicted: Nash-MTL and CAGrad are the best two, and MGDA badly underweights the classification task. Nash-MTL is also strong on SCALES (−25.2%). On the toy surface MGDA reaches the Pareto set from all 5 starts, CAGrad and Nash-MTL from 3. |
| FAMO | Δm, SCALES / DIGITS; ms per step | −22.9% / −6.3%; 3.07 | Nash-MTL: −25.2% / −3.5%; 5.48. Equal: ms per step 1.46 | Mixed | Best of all on SCALES, but 2.8 points behind Nash-MTL on DIGITS and twice the cost of equal weighting, not equal cost. |

Item verdicts: 3.13 Mixed, 3.14 Mixed, 3.15 Win, 3.16 Mixed.

Not done: core has no simplex QP solver, so MGDA and CAGrad use a Frank–Wolfe workaround.

### LatentKnowledge follow-up (W10 3.17: CCS with normalisation, prompt ensembles and CRC, follow-up)
Sources: R-wave10. Theme: interp. Test: LatentKnowledgeFollowUpTest. Block: `ContrastConsistentSearch` gains per-template normalisation, a templated `fit`, `credence`, `ensembleAccuracy` and `crc` (wave 9 defaults unchanged); `LatentKnowledge` gains `followUp` and a `main` Run. 300 steps.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| CCS; CRC, one template | Accuracy on truth | 0.540; 0.550 | Random direction: 0.550. Supervised probe: 0.700. Zero-shot: 0.830. Sycophantic: 0.580 | Negative | Negative: normalisation does not lift CCS above a random direction, though the model knows the answer. |
| CCS with a distractor feature present | Accuracy on truth / on the distractor | 0.550 / 1.000 | | Negative | CCS follows the distractor exactly, as Farquhar et al. predict. |
| Four templates: pooled; per-template normalisation; ensemble; CRC | Accuracy on truth | 0.523; 0.535; 0.540; 0.550 | Supervised: 0.673. Zero-shot: 0.808 | Negative | Negative: neither normalisation nor prompt ensembles help. |

Item verdict: Negative. The test-budget numbers settle the follow-up: the model answers zero-shot at 0.83, and CCS and CRC stay at the random level. The 8-template rows and the long `main()` Run were not executed.

---

## Wave 10 WP4: deep RL recipes, plasticity and scaling, RL and post-training follow-ups

New package `experimental.deeprl` (LearningCurveHarness, ContinuousMountainCar, PixelObservations, ModernSac, CrossQ, Bro, SimbaV2, LapReplayBuffer, Td7, MrQ, SelfPredictive, DrQ, Pqn, EfficientZero, Plasticity, MultiSkipNetwork, ValueScalingStudy). The no-long-runs rule hits this WP hardest: the permission system refused the builder's background queue of experiment mains, and no `main()` ran at full budget (ModernOffPolicy, ValueBasedRecipes, PlasticityAndScale, OfflineRlComparison, GoalsImitationAndTeams, RewardOveroptimization and ProcessSupervision). Every number below comes from a test-scale run or a short probe with one seed. Budgets were cut in the Bro, FairBudget, Fujimoto, PQN, MrQ and Depth tests and the Pixel test was split in two (10.4 s and 3.1 s); the trimmed classes were not re-timed. ImitationFollowUpTest takes 10.5 s. Encoders are MLPs, not conv nets, because conv was too slow on CPU (R-wave10).

### OfflineRlComparison on Pendulum (W10 4.1: TD3+BC)
Sources: R-wave10. Theme: control. Test: OfflineRlTd3BcTest. Block: `experimental.rl.Td3Bc`; `OfflineRlComparison.pendulum` with a `PendulumData` enum, `pendulumController` and `pendulumData` (existing rows unchanged). Data: four Pendulum datasets (random, medium, medium-replay, expert) from a hand-made energy-pumping plus PD controller. Scores are normalised returns: 0 is random, 1 is the noise-free controller. Probe: 20 episodes, 3,000 gradient steps.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| TD3+BC | Normalised return, random / medium / medium-replay / expert (wall clock on random) | −0.03 / 0.28 / 0.11 / 0.97 (11 s) | Data: 0.13 / 0.40 / 0.30 / 1.06. BC: 0.07 / 0.24 / 0.09 / 0.49 (1.9 s). IQL, 5 torque levels: 0.02 / −0.04 / 0.26 / 0.23 (5.5 s). CQL, 5 torque levels: 0.08 / −0.02 / −0.03 / 0.59 (2.8 s). FQL: 0.01 / 0.29 / 0.09 / 0.12 (25 s). TD3+BC with α = 0: −0.06 / 0.27 / 0.09 / 1.00 | Win | Best on expert data, level with the best on medium, and behind CQL on random data, as predicted. It is not cheaper than CQL here (11 s against 2.8 s). On mixed data no method improves on the data at this budget. |

Not done: the `main` Run (50 episodes, 20k steps); Decision Transformer is left out because it is discrete-only.

### ModernOffPolicy (W10 4.2: CrossQ, 4.3: BRO, 4.4: SimbaV2, 4.6: MR.Q, 4.7: TD7, 4.14: continuous control at a fair budget, follow-up)
Sources: R-wave10. Theme: control. Tests: ModernOffPolicyCrossQTest, ModernOffPolicyBroTest, ModernOffPolicySimbaV2Test, ModernOffPolicyFujimotoTest, ModernOffPolicyFairBudgetTest, LearningCurveHarnessTest. Blocks: `experimental.deeprl.ModernSac` (core `Sac` is final and has no hooks), `CrossQ`, `Bro`, `SimbaV2`, `Td7`, `LapReplayBuffer`, `MrQ`, `LearningCurveHarness`. Probe: Pendulum, 5k steps, one seed.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| CrossQ (Adam β1 = 0.5, lr 1e-3), 5k steps | Return | −287 | Core SAC: −343. ModernSac: −214. CrossQ with default Adam: −1,055 | Mixed | Beats core SAC without a target network, but only with the paper's β1 = 0.5, and trails every other modern recipe. |
| CrossQ, 600-step test | Return | −1,479 | SAC: −1,577. SAC with naive BN in the critic: −1,261. DroQ: −1,324. REDQ (about 4x CrossQ's updates per step): −775 | Negative | Negative at this budget: CrossQ does not match DroQ, and naive BN does not diverge. |
| BRO (width 32); SimbaV2, 5k steps | Return | −202; −239 | Core SAC: −343. ModernSac: −214 | Inconclusive (long run pending) | Both beat core SAC, but no more than SAC with modern defaults. The width and update-ratio sweeps that the hypotheses are about exist only in `main()`. |
| TD7, 5k steps | Return | −217 | TD3: −839. Core SAC: −343 | Win | Far ahead of TD3. The SALE and checkpoint ablations were not reported. |
| MR.Q, 5k steps | Return | −215 | TD3: −839. Core SAC: −343. BRO: −202 | Mixed | Mid-table on the only continuous task reported; the discrete pixel rows (ValueBasedRecipes) ran at sanity budgets only, so "never the worst across tasks" is untested. |
| Fair-budget harness, 400 steps | Kendall τ of rankings; P(SAC > random) | 0.73; 1.00 | | Inconclusive (long run pending) | Only sanity budgets ran. The 5k / 20k / 50k curves with IQM and bootstrap intervals are the `main()` Run. The one-seed 5k probe above fits "modern recipes lead at 5k" (ModernSac −214 against core SAC −343). |

Item verdicts: 4.2 Mixed, 4.3 Inconclusive (long run pending), 4.4 Inconclusive (long run pending), 4.6 Mixed, 4.7 Win, 4.14 Inconclusive (long run pending).

Not done: no continuous MountainCar numbers were reported (the environment exists as `ContinuousMountainCar`).

### ValueBasedRecipes (W10 4.5: PQN, 4.6: MR.Q discrete, 4.8: SPR and BBF, 4.9: DrQ and DrQ-v2, 4.12: EfficientZero)
Sources: R-wave10. Theme: control. Tests: ValueBasedRecipesPqnTest, ValueBasedRecipesPixelTest, ValueBasedRecipesPixelPendulumTest, ValueBasedRecipesMrQTest, ValueBasedRecipesModelBasedTest. Blocks: `experimental.deeprl.Pqn`, `SelfPredictive`, `DrQ`, `MrQ`, `EfficientZero`, `PixelObservations`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| PQN; without LayerNorm; with λ = 0, CartPole 2k steps | Return | 75; 33; 104 | DQN: 137. Rainbow: 200. PPO: 144 | Negative | Negative: PQN trails DQN here and at the earlier 3k-step setting (161 against 178). Removing LayerNorm hurts, as predicted. Wall-clock per step was not reported. |
| SPR; DrQ; BBF without resets; BBF with resets, pixel grid, 400–800 steps | Return | 0.94; 0.94; 0.94; −0.25 | DQN at replay ratio 2: 0.94. Core DQN, Rainbow, Dreamer: −0.25 | Mixed | SPR and DrQ beat Rainbow, but so does plain DQN at replay ratio 2, so neither the encoder gain nor the augmentation gain is isolated. BBF's resets hurt instead of helping. |
| DrQ-v2-style SAC, pixel pendulum, 200 steps | Return | −1,354 | SAC: −1,403 | Inconclusive | Smoke-level budget; the gap is within noise. The DrQ-v2 row runs on SAC, not DDPG. |
| EfficientZero; MR.Q discrete | | Sanity budgets only; no numbers reported | MuZero-lite | Inconclusive | Smoke-level runs. EfficientZero's reanalysis uses the prediction network's value, not a full MCTS. |

Item verdicts: 4.5 Negative, 4.8 Mixed, 4.9 Mixed, 4.12 Inconclusive (4.6 is under ModernOffPolicy).

### PlasticityAndScale (W10 4.10: ReDo, plasticity injection and resets, 4.11: multi-skip residuals and Kron, 4.17: value-based scaling and TD-overfitting)
Sources: R-wave10. Theme: control. Tests: PlasticityAndScaleTest, PlasticityAndScaleSwitchTest, PlasticityAndScaleDepthTest (its method is renamed `depthStudyReportsGradientSpread`), PlasticityAndScaleScalingTest. Blocks: `experimental.deeprl.Plasticity`, `MultiSkipNetwork`, `ValueScalingStudy`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| ReDo; RESET_LAST; SHRINK_PERTURB; injection; BBF resets, CartPole 500 steps, replay ratio 2 | Return | 191; 103; 140; 26; 28 | No intervention: 175 | Mixed | ReDo cuts the dormant share from 0.125 to 0.000 without changing the network's output and gains a little. The resets and injection, expected to be cheap fixes, hurt. |
| The same variants after a task switch | Return | RESET_LAST 0.98; every other variant −0.20 | No intervention: −0.20 | Win | Only resetting the last layer recovers after the switch; ReDo does not. |
| Multi-skip network | Gradient spread across layers | 13.0 | Plain MLP: 6.1 | Negative | Negative: the spread is higher, against the hypothesis. Return against depth and the Kron combination were not reported. |
| Passive narrow critic on a wide critic's targets | Held-out TD error | 0.484 | The same narrow critic on its own targets: 0.181. The wide critic: 0.098 | Negative | Negative: better targets do not let the narrow critic fit well. The width × batch-size grid is a `main()` Run, and no numbers from the test's corner of it were reported. |

Item verdicts: 4.10 Mixed, 4.11 Negative, 4.17 Negative. The multi-skip merge is additive; the paper does not specify it.

### GoalsImitationAndTeams follow-up (W10 4.13: GAIL with gradient penalty and absorbing states, AIRL and IQ-Learn, follow-up)
Sources: R-wave10. Theme: control. Test: ImitationFollowUpTest. Block: `experimental.rl.AdversarialImitation` (GAIL with gradient penalty and DAC absorbing states; AIRL with f = g(s) + γh(s′) − h(s); IQ-Learn with a χ² regulariser and the value-form objective). Methods: `imitationFollowUp`, `imitationFollowUpCartPole` and `imitationTransfer` (wave 1 rows unchanged).

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| GAIL + GP + absorbing; AIRL; IQ-Learn; GAIL + GP without absorbing, CartPole, 1 demonstration, 30–50k steps | Return | 193–200; 180–190; 74–83; 27–47 | Expert: 200. BC: 181. Wave 1 GAIL: 20–45. Random: 18 | Win | The absorbing state is the fix: regularised GAIL beats BC from one demonstration. IQ-Learn falls well below BC. |
| The same methods on the grid, 20k steps | Return | GAIL + GP + absorbing: 0.29–0.34. Nearly every other method: 0.92 | Expert: 0.92 | Negative | Negative: on the grid the absorbing-state GAIL is the one method that fails. |
| Rewards re-solved on a grid with every move rotated | Return | AIRL without absorbing states: 0.92. GAIL: −0.20. IQ-Learn: −0.20 | Re-solving the true reward: 0.92. Copied expert policy: −0.20 | Win | AIRL's reward transfers and GAIL's does not, as predicted. With absorbing states, AIRL's g cannot be re-solved even on the training grid, because the learned absorbing value is below the values on the expert's path. |

Item verdict: Mixed. The test-budget numbers already settle this follow-up. Not reported: the 5- and 20-demonstration rows and the MaxEntIrl baseline.

### RewardOveroptimization strongProxies (W10 4.15: reward-model ensembles and WARM with stronger proxies, follow-up)
Sources: R-wave10. Theme: alignment. Test: StrongProxiesTest (8.4 s). Blocks: new options with defaults unchanged: `RewardEnsemble.trainMembers` (independent or shared start), `WeightAveragedReward.fineTune` (from a trained shared start on bootstrap subsets) and `WeightAveragedReward.copy`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Proxy reward models, test corner (200 pairs, 40 reward-model steps) | Preference accuracy | 0.55–0.60 | Target: 0.70. Wave 9 single proxy: 0.586 | Inconclusive (long run pending) | The test corner does not reach the 0.7 accuracy the hypothesis needs. The 1x / 4x / 8x Run (600 base pairs, 160 steps) was not executed. |

### ProcessSupervision largerBudget (W10 4.16: VinePPO, PRIME and Step-DPO at a larger budget, follow-up)
Sources: R-wave10. Theme: reasoning. Test: ProcessSupervisionLargerBudgetTest (3.3 s). Blocks: new options with defaults unchanged: `MonteCarloValuePpo.rollouts` and `startCheckpoint`, `ImplicitProcessReward.epochs`, `StepPreferenceBuilder.temperature`; new `ProcessSupervision` overloads of `vine`, `prime` and `stepDpo` (their defaults reproduce wave 9), plus `heldOutSetup` and `largerBudget`. `main` uses StepChainTask(3, 4, 9) with 1,600 problems.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Start model, test corner (100 SFT steps) | Held-out accuracy | 0.10 | Required start: above 0.5 | Inconclusive (long run pending) | The test only checks that the pipeline runs; at 0.10 held-out accuracy its numbers mean nothing. |

---

## Wave 10 WP5: boosting, point processes, irregular series, temporal and relational graphs, parsing

New packages: `experimental.boost`, `experimental.events`, `experimental.parse`, `experiments.events` and `experiments.parsing`, with new generators `experimental.data.EventStreams`, `IrregularSeries`, `TemporalGraphs` and `ToyTreebank`. Every WP5 test class passed in about 10 s or less when last run (R-wave10).

### BoostingShootout (W10 5.1: LightGBM, 5.2: CatBoost, 5.3: NGBoost, 5.4: EBM, 5.5: PFN-Boost, 5.6: quantile regression forests)
Sources: R-wave10. Theme: classic. Tests: BoostingShootoutLightGbmTest, BoostingShootoutCategoricalTest, BoostingShootoutProbabilisticTest, BoostingShootoutInterpretableTest, BoostingShootoutPriorBoostTest. Blocks: `experimental.boost.LightBoost`, `OrderedBoosting`, `NaturalGradientBoosting`, `QuantileForest`, `ExplainableBoosting`, `PriorBoost`, `BoostLosses`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| GOSS; leaf-wise, 8 / 32 leaves; level-wise, 2,000 rows | AUC (fit time) | 0.896 (69 ms); 0.919 / 0.903 (129 ms); 0.907 | Core GBT: 0.906 (306 ms). Random forest: 0.822 | Win | GOSS stays within 0.01 of full-data accuracy at under a quarter of the time. Leaf-wise growth with 8 leaves is best, and 32 leaves overfits slightly. |
| EFB, sparse data | AUC (fit time) | 0.915 with 6 bundles (154 ms) | No bundling, 240 columns: 0.915 (269 ms). Core GBT: 0.875 (915 ms) | Win | The same accuracy 1.7x faster than without bundling, less than the hypothesised several-fold. |
| Ordered boosting; with feature combinations | Test log loss / train–test gap | 0.535 / 0.164; 0.498 / 0.307 | Plain boosting: 0.534 / 0.164. Leaky target encoding: 0.705 / 0.487. Out-of-fold encoding: 0.554. One-hot: 0.520. RF + out-of-fold: 0.579 | Negative | Negative: ordered boosting equals plain boosting and does not beat one-hot. Leaky target encoding overfits badly, as predicted, and feature combinations give the best loss. |
| NGBoost, Normal; ordinary gradient; CRPS scoring; Laplace | NLL / CRPS / 90% coverage | 0.727 / 0.283 / 0.834; NLL 0.738; 0.737; 0.753 | Quantile forest: CRPS 0.335, coverage 0.936 | Mixed | NGBoost beats the forest on CRPS but under-covers. The ordinary gradient is only slightly worse, not unstable. No quantile-GBT or deep-ensemble row was reported. |
| EBM | Accuracy; shape correlation; rank of the planted pairs | 0.880; 0.910; 1 and 2 | GBT: 0.879. GAM: 0.712. NAM: 0.653. Logistic: 0.591 | Win | Matches GBT, beats NAM by 0.23, and ranks both planted pairs first. |
| PFN-Boost, 30 / 300 rows | Accuracy | 0.652 / 0.726 | GBT: 0.652 / 0.721. PFN alone: 0.530 / 0.656. Stacking: 0.607 / 0.749 | Mixed | Ties GBT at both sizes and beats the PFN; stacking wins at 300 rows. With only two sizes there is no "in between" to test. |
| Quantile regression forest, bimodal target | CRPS | 0.510 | NGBoost: 0.488 | Negative | Negative: the forest does not beat NGBoost on the bimodal variant. No CQR row was reported. |

Item verdicts: 5.1 Win, 5.2 Negative, 5.3 Mixed, 5.4 Win, 5.5 Mixed, 5.6 Negative.

Not done: no PriorBoost rows in `TabularShootout` (covered by `BoostingShootout.priorBoost`). Core `GradientBoostedTrees` has no base-margin input, and its loss objectives are private.

### PointProcessShootout (W10 5.7: Hawkes processes with flexible kernels, 5.8: neural TPPs, 5.9: intensity-free TPP)
Sources: R-wave10. Theme: events. Tests: PointProcessShootoutTest, PointProcessShootoutDelayedTest, PointProcessShootoutInhibitoryTest, IntensityFreeTppTest. Blocks: `experimental.events.HawkesProcess` (EM, and EMS for spline kernels), `NeuralPointProcess` (RMTPP, continuous-time / neural Hawkes, transformer Hawkes), `IntensityFreeTpp`. Data: `experimental.data.EventStreams`. Log-likelihood per event (higher is better).

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Spline-kernel Hawkes (EMS), delayed-kernel stream | Log-likelihood per event | −1.087 (KS 0.012) | Truth: −1.066. Exponential Hawkes: −1.139 | Win | The spline kernel closes most of the exponential kernel's gap. On its own stream exponential EM is at the truth (−0.555 against −0.554); on the inhibitory stream Hawkes stays at about −1.09 against a truth of −0.837, as predicted. |
| Continuous-time (neural Hawkes); RMTPP; THP, exponential / inhibitory stream | Log-likelihood per event | −0.614 / −0.888; −0.635 / −0.956; −0.759 / – | Exponential Hawkes EM: −0.555 / about −1.09. Spline: −0.559 on the exponential stream. Poisson: −0.839. Weibull: −0.704 | Win | The 2026 finding reproduces at toy scale: the neural models win on inhibition and do not beat Hawkes on Hawkes-generated data. |
| Log-normal mixture (intensity-free), exponential stream | Log-likelihood per event; events sampled per second | −0.659; 600–5,000 | Exponential Hawkes: −0.555. Continuous-time: −0.614. RMTPP: −0.635. Thinning: 1,100–160,000 events per second | Negative | Negative: a worse NLL than the intensity models, and slower sampling, because it re-encodes the prefix at every step. |

Item verdicts: 5.7 Win, 5.8 Win, 5.9 Negative.

### IrregularSeriesShootout (W10 5.10: GRU-D, ODE-RNN, latent ODE and mTAN, 5.11: BRITS, SAITS and CSDI)
Sources: R-wave10. Theme: events. Tests: IrregularSeriesReconstructionTest, IrregularSeriesClassificationTest, IrregularSeriesImputationTest. Blocks: `experimental.events.IrregularSequenceModels` (GRU-D, ODE-RNN and latent ODE on `OdeSolver`, mTAN), `IrregularBatch`, `TimeSeriesImputer` (BRITS, SAITS, CSDI). Data: `experimental.data.IrregularSeries` (seasonal, chaotic and linear-Gaussian dynamics; MAR, MNAR and block missingness).

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| GRU-D; ODE-RNN; mTAN; latent ODE, seasonal, 60% missing at random | Interpolation / extrapolation MSE | 0.431 / 0.720; 0.586 / 0.789; 0.701 / 0.823; 0.751 / 0.808 | Linear interpolation: 0.399 / 1.612. Forward-fill GRU: 0.507 / 0.797. Linear interpolation + DLinear: extrapolation 0.666 | Negative | Negative: GRU-D is the best learned model, but linear interpolation beats every model on interpolation, and the latent ODE, expected to win there, is last. |
| mTAN; GRU-D, CBF with 60% missing | Accuracy | 0.767; 0.600 | Forward-fill GRU: 0.933. Neural CDE: 0.844 | Negative | Negative: the forward-fill GRU wins, where mTAN was expected to. |
| BRITS; SAITS; CSDI (20 diffusion steps, 5 samples), seasonal, 50% block missing | MAE (CRPS) | 0.589; 0.724; 0.868 (0.672) | Gaussian process: 0.403 (0.279). Linear interpolation: 0.692. Latent ODE: 0.714. Mean: 0.790. Kalman smoother: 0.941 (0.656) | Negative | Negative: a Gaussian process is best, SAITS does not beat BRITS, and the undertrained CSDI is near last. |

Item verdicts: 5.10 Negative, 5.11 Negative.

Not done: the `LiquidCell` baseline and the downstream forecaster trained on imputed data. The Kalman smoother is a local missing-aware local-linear-trend RTS smoother, because `StateSpaceFilter.kalman` cannot take missing observations; no linear-Gaussian-series row was reported.

### TemporalGraphShootout (W10 5.12: TGN against EdgeBank, with a probe suite)
Sources: R-wave10. Theme: graphs. Test: TemporalGraphShootoutTest. Block: `experimental.graph.TemporalGraphModels` (EdgeBank unlimited and windowed, TGN with GRU memory and TGAT attention, GraphMixer, a HyperEvent-style relational encoder). Data: `experimental.data.TemporalGraphs` (six probe streams). The experiment is named `TemporalGraphShootout`, not `TemporalGraphs` as planned, to avoid a clash with the data class; the static-GCN baseline uses `GcnLayer`. 12 epochs, 40 nodes.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| HyperEvent; TGN; GraphMixer; static GCN, repeated-edge stream | MRR, random / historical negatives | 0.836 / 0.592; 0.394 / 0.369; 0.508 / 0.438; 0.657 / 0.533 | EdgeBank: 0.855 / 0.549 | Mixed | As predicted, EdgeBank beats every learned model under random negatives and drops under historical ones, where HyperEvent passes it. TGN is the weakest model. |
| The same models, recency stream | MRR, random / historical negatives | 0.673 / 0.499; 0.289 / 0.272; 0.310 / 0.313; 0.359 / 0.308 | EdgeBank: 0.595 / 0.238. Windowed EdgeBank: 0.695 / 0.378 | Mixed | HyperEvent wins under historical negatives; TGN and GraphMixer stay near chance. |

Item verdict: Mixed. The probe suite behaves as predicted, but TGN loses to EdgeBank everywhere at this budget.

Not done: the SASRec / `ClickLogs` bipartite case; only two of the six probe streams were reported.

### RelationalGraphs (W10 5.13: R-GCN and CompGCN, 5.14: NBFNet, 5.15: hyperbolic embeddings)
Sources: R-wave10. Theme: graphs. Tests: RelationalGraphsLinkTest, RelationalGraphsNodeTest, RelationalGraphsHierarchyTest. Blocks: `experimental.graph.RelationalGcn` (R-GCN basis and block, CompGCN sub / mult / corr), `BellmanFordNetwork` (NBFNet with a gradient path read-out), `HyperbolicEmbedding` (Poincaré, Lorentz). The link test was rerun on the report day (6.3 s).

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| CompGCN-mult; R-GCN basis, transductive link prediction | MRR | 0.863; 0.749 | ComplEx: 0.887. Common neighbours: 0.371 | Win | Shallow embeddings win transductively, as predicted. |
| R-GCN; CompGCN, relation-pattern node classification | Accuracy | 1.000; 1.000 | Untyped GCN: 0.929 | Win | Relation types solve the task. |
| NBFNet, transductive / inductive | MRR | 1.000 / 1.000 | Inductive: R-GCN basis 0.439, common neighbours 0.382, CompGCN 0.15–0.26 | Win | The only model that keeps its accuracy on unseen entities. Its top paths match the planted rules in only 0.2 of cases (0.5 at an earlier, larger budget), so the rule read-out part of the hypothesis fails. |
| Poincaré, 2-D; Lorentz, 2-D | Mean rank (lower is better) | 1.80; 1.65 | Euclidean 2-D: 4.39. Euclidean 10-D: 2.59. node2vec 10-D: 7.76 | Win | Two hyperbolic dimensions beat ten Euclidean ones. Lorentz is slightly better than Poincaré; training stability was not measured. |

Item verdicts: 5.13 Win, 5.14 Win, 5.15 Win.

Not done: the `SyntheticGraphs` homogeneous link-prediction case and the `GraphReasoning` CLRS Bellman–Ford baseline.

### ParsingShootout (W10 5.16: biaffine dependency parsing, 5.17: span-based chart parsing)
Sources: R-wave10. Theme: parsing. Tests: ParsingShootoutDependencyTest, ParsingShootoutConstituencyTest, TreeDecoderTest. Blocks: `experimental.parse.BiaffineParser` (greedy, Eisner and Chu–Liu–Edmonds decoders, each with a single root child), `ChartParser` (BiLSTM or factored content/position attention encoder; CKY; margin and tree-CRF losses via inside–outside), the package-private `SentenceEncoder`; data `experimental.data.ToyTreebank`; baseline `TreebankPcfg`. TreeDecoderTest checks CLE and Eisner against brute force and that the inside–outside marginals sum to 2n − 1. 300 training sentences of length up to 12; test sentences up to 18.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Biaffine + Eisner; + CLE; greedy | UAS / LAS / share of well-formed trees / PP accuracy / UAS on long sentences | 0.925 / 0.918 / 1.0 / 0.875 / 0.853; 0.880 / 0.875 / 1.0; UAS 0.873 (0.913 well-formed) | BiLSTM-CRF offsets: 0.892 (0.713 trees). Pointer network: 0.786 (0.80 trees). Right-branching: 0.425 | Win | Beats the tagging baseline by 0.033 UAS and always outputs trees; Eisner decoding adds 0.05 over greedy. |
| Chart parser, BiLSTM, margin loss; tree-CRF loss; factored attention | F1 / exact match / PP / F1 on long sentences | 0.987 / 0.913 / 0.982 / 0.983; 0.960 / 0.750 / 0.804; F1 0.894 | Treebank PCFG: 0.979 / 0.813 / PP 0.768. Right-branching: 0.270 | Mixed | The chart parser beats the PCFG, most on PP attachment, which the PCFG cannot get without lexical preferences. Negative part: the tree-CRF loss, expected to win slightly, loses to the margin loss. |

Item verdicts: 5.16 Win, 5.17 Mixed.

Not done: the seq2seq baselines (`TransformerSeq2Seq` and `Seq2SeqAttention` on linearised trees and head sequences) were skipped for time, so the seq2seq comparisons in both hypotheses are untested.

---

## Wave 10 WP6: image restoration, speech analysis, perception and systems follow-ups

New packages: `experimental.restore`, `experimental.speech`, `experiments.restoration` and `experiments.speech`, with data generators in `experimental.data`. Most test classes pass in about 10 s. Two are longer: SpeakerDiarizationTest took 19.8 s at its last run, before its budget was lowered from 50 to 30 steps (not run since), and VqaTwoStageTest takes 28 s, about 15 s of it in the shared wave 9 `VqaFixture` setup. The long `main()` budgets of 10.6.10 to 10.6.14 and the 2k–20k-step vocoder training of 10.6.16 were written but not run (R-wave10).

### RestorationShootout (W10 6.1: super-resolution, 6.2: DnCNN and FFDNet, 6.3: self-supervised denoising, 6.4: plug-and-play deblurring)
Sources: R-wave10. Theme: restoration. Tests: SuperResolutionShootoutTest, SupervisedDenoisingTest, SelfSupervisedDenoisingTest, DeblurringTest, RestoreBlocksTest. Blocks: `SuperResolution` (SRCNN, ESPCN, EDSR, SPAN), `Denoisers` (DnCNN, FFDNet, NLM, median, Gaussian), `SelfSupervisedDenoising`, `PlugAndPlay` (Wiener, HQS, ADMM, Richardson–Lucy, TV, `Unrolled`). PSNR in dB.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| ESPCN (1,868 parameters); EDSR (16,980), scenes ×2, 120 steps | PSNR | 25.46; 25.61 | Bicubic: 25.26. Nearest: 24.08 | Win | EDSR is best at equal steps, and ESPCN needs 3.5x fewer multiply-adds than SRCNN (117,760 against 415,744 on 8×8). Both gains over bicubic are small (0.2–0.35 dB). |
| DnCNN; FFDNet, σ = 25 / spatially varying noise | PSNR | 24.64 / 23.40; 21.65 / 20.82 | NLM: 24.55 / 21.78. Gaussian: 19.94 / 19.59 | Mixed | DnCNN ties NLM on uniform noise and wins on varying noise. FFDNet, expected to win there with its noise map, is weak at this budget. |
| Noise2Noise; Neighbor2Neighbor; zero-shot N2N, i.i.d. noise | PSNR | 23.28; 22.65; 22.94 | Supervised: 23.34. Median: 21.46 | Win | N2N is within 0.06 dB of supervised, as predicted, and Neighbor2Neighbor loses 0.7 dB. |
| Median2Median, correlated noise | PSNR | 18.76 | Median filter: 20.03 | Negative | Negative: M2M falls below the filter it was expected to beat. |
| PnP-HQS, 200 steps, Gaussian / motion / unseen motion kernel | PSNR | 19.84 / 21.81 / 21.55 | Wiener: 21.41 / 22.75 / 22.59. Blurred input: 18.98 / 21.00 / 19.43 | Negative | Negative: Wiener wins on all three kernels. An earlier 300-step run had PnP ahead on both motion kernels (23.29 against 22.65, and 23.06 against 22.63). |

Item verdicts: 6.1 Win, 6.2 Mixed, 6.3 Mixed, 6.4 Negative.

Not reported: SPAN's and SRCNN's PSNR, the OCR check on ×4 glyphs, and the TV, ADMM and unrolled deblurring rows.

### SpeakerShootout (W10 6.5: x-vectors with AAM-softmax and GE2E, 6.8: diarisation)
Sources: R-wave10. Theme: speech. Tests: SpeakerVerificationTest, SpeakerDiarizationTest. Blocks: `SpeakerEmbedding` (TDNN x-vector, AAM-softmax, GE2E, `Plda`, EER, minDCF), `Diarization` (AHC, eigengap spectral clustering, `Eend` with permutation-invariant training). Data: the richer voice generator `experimental.data.SpeakerVoices`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| x-vector + AAM; GE2E | EER (with PLDA) | 0.222 (0.227); 0.192 (0.146) | Mean MFCC: 0.342. GMM-UBM: 0.333 | Win | Both cut the EER by a third or more. GE2E with PLDA is best, not the x-vector as predicted. |
| x-vector spectral clustering; with the oracle speaker count; EEND | Diarisation error rate | 0.626; 0.358; 0.547 | One-speaker baseline: 0.626. MFCC spectral: 0.598 | Mixed | The eigengap estimates one speaker, so clustering ties the baseline: the predicted weak point. With the true count, clustering is best. No overlap sweep was reported. |

Item verdicts: 6.5 Win, 6.8 Mixed.

Not done: `SpeakerEmbedding` keeps a local AAM copy (WP3's `MarginSoftmax` was not on main yet); the digit-encoder baseline exists only in `main()` and the options.

### SpeechFrontEnds (W10 6.6: voice activity detection, 6.7: speech enhancement)
Sources: R-wave10. Theme: speech. Tests: VoiceActivityDetectionTest, SpeechEnhancementTest, SpeechBlocksTest. Blocks: `VoiceActivity` (energy, Sohn, Sohn + HMM hangover, `Crnn`), `Enhancement` (spectral subtraction, Wiener / MMSE-STSA, `MaskNet` with a magnitude or complex mask).

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Sohn; Sohn + HMM; CRNN, white noise / babble at 0 dB | F1 | 0.578 / 0.562; 0.830 / 0.624; 0.910 / 0.718 | Energy: 0.012 / 0.759 | Mixed | In white noise the CRNN is best and energy fails; in babble, energy beats every other detector. |
| MLP mask; CRNN complex mask, white noise at 0 dB, 30 steps | SI-SNR improvement (dB) | 2.40; 6.07 | Spectral subtraction: 8.39. MMSE-STSA: 7.08. Ideal ratio mask: 13.52 | Mixed | The classic methods beat the learned masks at 30 steps. At 60 steps the CRNN magnitude mask reaches 9.70 dB, so complex and magnitude masks were not compared at matched steps. |

Item verdicts: 6.6 Mixed, 6.7 Mixed.

Not done: the VAD keyword-spotter baseline; the enhancement digit error rate exists only in `main()` and the options.

### VisionShootout flexiViT (W10 6.9: FlexiViT)
Sources: R-wave10. Theme: vision. Test: FlexiVitTest. Block: `experimental.vision.FlexiblePatchEmbedding`. 16 px images, 150 steps.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| FlexiViT at patch 2 / 4 / 8 | Accuracy | 0.316 / 0.313 / 0.273 | Separate ViTs on a third of the budget each: 0.168 / 0.230 / 0.148. One patch-4 ViT at full budget: 0.695 at patch 4. That ViT resized, naively and by pseudo-inverse: 0.160 at patch 2; 0.238 and 0.184 at patch 8 | Negative | Negative: one FlexiViT beats three split-budget ViTs but is 0.38 below a single ViT at its own size, not within 1–2 points. Resizing a fixed ViT collapses away from patch 4, as predicted. |

Not done: the `ElasticSupernet` CNN-analogue baseline.

### DetectionShootout detrFollowUp (W10 6.10: DETR with two-stage query selection, follow-up)
Sources: R-wave10. Theme: vision. Test: DetrQuerySelectionTest. Block: `SetPredictionDetector.Options.querySelection` and `withQuerySelection` (defaults unchanged).

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| DETR + query selection, scenes | mAP@0.5 at 250 / 500 steps | 0.141 / 0.303 | DETR: 0.000 / 0.002 | Win | Leaves mAP 0 within a few hundred steps, as predicted; this part is settled at the test budget. The 5k-step curve against CenterNet and the H-DETR combination are pending in `main()`. |

Not reported: the `GlyphDocuments` layout rows.

### DocumentParsing donutFollowUp (W10 6.11: Donut with reading pretraining, follow-up)
Sources: R-wave10. Theme: document. Test: DonutReadingPretrainingTest. Block: `ImageToMarkup.pretrainReading`, `fineTune`, `checkpoint` and `restore`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Donut with 150 reading steps on 200 pages, 300 steps in total | Field F1 (reading CER after pretraining) | 0.104 (0.662) | The same total steps without pretraining: 0.031. Wave 9 OCR + rules: 0.838 | Inconclusive (long run pending) | Pretraining lifts F1 off zero, but only to 0.104; whether it approaches the OCR pipelines needs the 3k-step curve. That 300-step run took 212 s, so the test is now a smoke run. |

### FusionShootout follow-ups (W10 6.12: two-stage VQA bridges, 6.13: latent Transfusion, follow-ups)
Sources: R-wave10. Theme: multimodal. Tests: VqaTwoStageTest, LatentTransfusionTest. Blocks: `experimental.multimodal.TwoStageBridgeTrainer` (LmMode FROZEN, LORA or FULL), `ConvVae`, `TransfusionModel.UNetPatches`, `forwardMacs`.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Projector with LoRA, 100 alignment + 150 QA steps | VQA accuracy; alignment accuracy on occupied slots; LM text loss | 0.640; 0.018; 1.094 | Answer prior: about 0.65. Wave 9 projector: 0.650. Frozen LM text loss: 1.116 | Inconclusive (long run pending) | Still at the answer prior, and stage 1 barely aligns (0.018), so the test budget does not test the recipe. The 4x budget is in `main()`. |
| Latent Transfusion (VAE + U-Net patches), at the MACs of 300 pixel-model steps | Caption / generation accuracy | 0.333 / 0.133 | Wave 9 pixel model: 0.133 / 0.067. VQ tokens: 0.567 / 0.433 | Inconclusive (long run pending) | Better than the pixel model, but still far below VQ tokens at this small compute budget. VAE reconstruction MSE is 0.037. |

Item verdicts: 6.12 Inconclusive (long run pending), 6.13 Inconclusive (long run pending).

Not done: 6.12's LoRA covers only the output head plus per-block residual adapters, because the core `TransformerBlock` does not expose its projections; 6.13 does not use `ToyImages`, which lives in `experiments.generative`.

### EdgeDeployment onceForAllFollowUp (W10 6.14: BigNAS in-place distillation, follow-up)
Sources: R-wave10. Theme: edge. Test: BigNasInplaceDistillationTest. Block: `ElasticSupernet.Schedule.INPLACE_DISTILLATION` and `smallest()`. 300 steps.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| BigNAS in-place distillation | Accuracy, largest / smallest subnet | 0.520 / 0.465 | Progressive shrinking: 0.915 / 0.780. Random sampling: 0.860 / 0.830. From scratch: 0.935 / 0.915 | Negative | Negative: distilling early from a weak teacher hurts, so in-place distillation does not get there sooner than progressive shrinking; the test budget settles that part. Progressive shrinking at 300 steps is far better than wave 9's 0.625 / 0.330 at 160. The predictor's Kendall τ rises from 0.412 (24 points) to 0.481 (60 points); the 200-point fit and the 640- and 2,560-step budgets are pending. |

### ParallelismShootout pipelineMemoryFollowUp (W10 6.15: V-shape schedules and an exact scheduler, follow-up)
Sources: R-wave10. Theme: parallel. Test: PipelineMemoryFollowUpTest. Blocks: `experimental.parallel.MemoryAwareScheduler`, `BuildingBlockSchedules`, a V-shape option on `PipelineSchedule.placed`. p = 4, m = 8, fixed costs.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| Exact branch and bound at ZB-H2 memory; greedy at the same memory | Bubble (peak activation memory) | 0.177 (7); 0.232 (7) | ZB-H1: 0.203 (4). ZB-H2: 0.203 (7). 1F1B: 0.280 (4) | Win | The exact scheduler lets ZB-H2's extra memory beat ZB-H1, though the bubble does not reach near 0. |
| V-ZB / V-Half; V-Min | Bubble (peak activation memory) | 0.208 (2); 0.501 (1.5) | 1F1B: 0.280 (4) | Win | V-ZB and V-Half beat 1F1B's bubble at half its memory, as predicted. All new schedules ran on threads with gradient error 7e-9. |

Item verdict: Win. The V-shape offsets come from a grid search, not the paper's closed form; the measured thread bubbles are dominated by noise; no sweep of V-Min's bubble over m was reported.

### AudioGeneration vocoderFollowUp (W10 6.16: GAN vocoder with mel-loss warm-up, follow-up)
Sources: R-wave10. Theme: audio. Test: VocoderWarmupTest. Blocks: `GanVocoder.melWarmup`, `save`, `load` and `trainedSteps`; `DiffusionVocoder.save` and `load`; `AudioGeneration.vocoderFollowUp` and a `tts(..., vocoder, name)` overload.

| Technique | Metric | Result | Baseline | Verdict | Takeaway |
|---|---|---|---|---|---|
| GAN vocoder, 20-step smoke run | Judge digit error | 1.0–1.67 | Griffin–Lim (32 iterations): 0.000 | Inconclusive (long run pending) | Smoke run only; the checkpoint round trip is verified. The 2k–20k-step training and the re-vocoding of 9.5.11 and 9.5.12 are the `main()` Run. |

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

### Wave 7

The ten most interesting wave 7 findings (Run-w7):

1. **A MAXSAT layer solves Latin squares.** SATNet solves 0.949 of the puzzles, against 0.455 for an MLP and 0.420 with a semantic loss. Wave 6's TRM, HRM and AKOrN reached about 0.72 cell accuracy and 0.62 solved (reasoning).
2. **Differentiable ILP finds the exact rules.** dILP scores 1.000 on grandparent, uncle and reachability and prints the rules, against 0.889 / 0.894 / 0.931 for a GNN (reasoning).
3. **DAGMA recovers causal graphs exactly.** SHD 0 on 20-variable linear data, against 16 for NOTEARS and 33 for GES (safety).
4. **Second-order methods win on small networks.** Levenberg–Marquardt fits Mackey-Glass to MSE 6.95e-7, about 50x below Adam (3.56e-5 in 2,000 steps), and K-FAC reaches the spirals target in 23 steps against 90 for AdamW (numerics).
5. **Local losses rescue forward gradients.** Activity-perturbed local forward gradients reach 0.980 on spirals, level with backprop (0.967), against 0.593 for weight-perturbed forward gradients (numerics).
6. **Look-ahead updates tame min-max games.** On a bilinear game, consensus optimisation ends at distance 0.000 and optimistic GDA at 0.226, against 34.0 for simultaneous GDA (multiagent).
7. **Sinkhorn couplings straighten flows.** OT pairing makes flow-matching paths about 130x straighter (0.004 vs 0.546) and cuts the one-step distance from 0.326 to 0.127 (probabilistic).
8. **GCNII stays accurate at depth 32.** It scores 0.940 where a plain GCN over-smooths from 0.976 to 0.313; PairNorm reaches 0.928 and DropEdge does not help (graphs).
9. **PILCO learns from a few hundred steps.** Cart-pole test cost 31.6 after 540 real steps, against 367 for SAC at the same budget; SAC needs 2,700 steps to reach 29.6 (control).
10. **Movement primitives generalise to new goals.** DMPs reach unseen goals with error 0.0009 from 20 demos, against 0.341 for behaviour cloning (control).

Honourable mentions: CFR+ drives Kuhn exploitability to 0.2 mbb/h against 10.7 for CFR, and PSRO with a Nash meta-solver reaches 0.0 where naive self-play stays at 1,167 (multiagent). RMA halves the loss on a cart-pole with unknown motor polarity, -106.5 against -218.6 for a nominal policy and -220.1 for domain randomisation (control). Periodic embeddings solve sin(6x) at 0.943 against 0.474 for an MLP (recsys). ReLoRA matches full-rank training (0.660) at a fifth of the optimizer memory, where LoRA reaches 0.550 (numerics). 2-bit compressed activations keep accuracy at 17x less memory (numerics). HSIC detects a ring dependence with power 1.00, where Pearson correlation has 0.00 (probabilistic). Barcode features keep 0.933 under rotation, where Deep Sets fall to 0.217 (probabilistic). JTT lifts worst-group accuracy to 0.493 without group labels, above ERM (0.379) and wave 6's GroupDRO (0.462) (safety). Each knowledge-graph model wins its relation type: DistMult 1.000 on symmetric siblings, ComplEx 0.956 on antisymmetric parents (graphs).

### Wave 8

The ten most interesting wave 8 findings (Run-w8):

1. **SINDy finds the exact equations, and rescues a UDE.** SINDy recovers the Lorenz system with coefficient error 0.000 in 11 ms, against 0.406 for genetic programming. Run on a UDE's learned term, it cuts Lotka–Volterra extrapolation error from 328 to 1.08, against 28.3 for a pure neural ODE (sciml).
2. **Let the draft decide what to keep.** At a 10% budget, draft-guided KV selection (SpecKV) answers 0.90 against 0.01 for SnapKV, and draft-guided prompt compression (SpecPC) 0.81 against 0.00 for LLMLingua (serving).
3. **A mixture of agents combines skills that voting cannot.** MoA scores 0.99 against 0.33 for the best single agent and 0.23 for a majority vote, with a quarter of the vote's tokens (agents).
4. **Hard constraints are exact and more accurate.** A stream-function field is divergence-free to 4e-7 with error 0.388, against 0.884 unconstrained and 0.77–0.83 with soft penalties (sciml).
5. **Built-in rotation equivariance beats augmentation.** A p4 G-CNN keeps 0.990 on rotated shapes, against 0.670 for a CNN and 0.930 with rotation augmentation; with 64 images the gap grows to 0.990 vs 0.918 (vision).
6. **Only adaptive conformal survives drift.** After a distribution shift ACI holds 0.81 coverage for an 80% target, against 0.61 for CQR, 0.37 for split conformal and 0.33 for DeepAR (timeseries).
7. **Mechanics in the network keep rollouts on track.** HNN and LNN pendulum rollouts reach MSE 0.0049 and 0.0030 with energy drift 0.047 and 0.015, against 0.073 and 0.647 for an MLP with RK4 (sciml).
8. **Hiding stale context beats reading more of it.** Sculptor's "hide superseded" tool scores 0.88 reading 7 tokens, against 0.47 for RLM reading 217 and 0.37 for MemAgent (agents).
9. **A hash grid fits a picture in seconds.** 43.29 dB in 150 steps, against 24.95 dB for Fourier features and 11.54 dB for a raw MLP (vision).
10. **QCFS converts a network to spikes in two time steps.** Converted SNN accuracy is 0.890 at T = 2 and matches the ANN (0.964) at T = 8, where plain conversion scores 0.100 at T = 2 (exotic).

Honourable mentions: SCAFFOLD beats FedAvg by 0.148 on label-skewed clients, averaged over 3 seeds (privacy). Re-estimating BN statistics lifts corrupted accuracy from 0.278 to 0.729 (robustness). Secure aggregation stops gradient inversion (5.7 dB against 23.8 dB) with no loss of accuracy (privacy). MagicPIG's LSH sampling aggregates at 1.00 where top-k on the same keys reaches 0.63 (serving). LOF finds local-density anomalies at AUROC 0.999 (timeseries). PirateNet trains a 12-layer PINN to 0.0003 where a plain MLP diverges to 1.34 (sciml). CenterNet scores 0.442 mAP on crowded scenes against 0.118 for RetinaNet-lite (vision). KNN-Shapley on SSL embeddings finds 81% flipped labels against 35% for In-Run Shapley (data). Movement pruning keeps 0.417 at 95% sparsity against 0.140 for magnitude (compression). A grammar mask cache is 8–77x faster with identical output (serving).

### Wave 9

The ten most interesting wave 9 findings (R-wave9, mostly one seed):

1. **Natural gradient finishes what Adam starts on PINNs.** ENGD reaches Poisson relative L2 2.1e-6, against 3.3e-3 for Adam + L-BFGS and 1.1e-3 with Nyström Newton–CG added, reproducing the paper (sciml).
2. **Run the code before you rank it.** Filtering samples on the given examples lifts hidden-test top-1 from 0.288 (likelihood) to 0.687 at 32 samples, and clustering or CodeT dual agreement to 0.712 (code).
3. **Delay compensation rescues asynchronous SGD.** At a gradient delay of 16, plain ASGD collapses to 0.10 while DC-ASGD-a reaches 0.88 and a staleness-aware learning rate 0.885 (parallel).
4. **History guidance keeps long video rollouts on count.** Over frames 16–32, the count error is 0.91 against 2.46 for diffusion forcing and 4.65 for teacher forcing; real clips score 0.52 (world).
5. **Clipped IPW learns to rank from biased clicks.** NDCG@5 is 0.898 against 0.79 for naive click training and 0.935 for an oracle, while unclipped IPW (0.738) loses to naive (recsys).
6. **Bid shading spends the budget well.** Dual-gradient bid shading gets 0.892 of the hindsight value, against 0.709 for truthful bidding, which runs out of budget at 22% of the day (recsys).
7. **A few human labels make a judge trustworthy.** Prediction-powered inference gives 0.93 coverage with intervals 22% narrower than humans alone; the judge alone covers 0.00, and Rogan–Gladen holds 0.95 under a pass-rate shift where PPI drops to 0.02 (evaluation).
8. **Preference embeddings capture cycles.** The general preference model scores 0.963 on cyclic preferences against 0.674 for Bradley–Terry, which cannot exceed 2/3 (alignment).
9. **Splitting the backward pass shrinks pipeline bubbles.** At p = 4, m = 8 the bubble is 0.150 for ZB-H1 and 0.040 for DualPipe, against 0.277 for 1F1B (parallel).
10. **Patch-based inference cuts peak memory 3.4x exactly.** Peak memory falls from 49,152 B to 14,336 B for 35% more multiply-adds, and at the same memory full resolution scores 0.665 against 0.360 at half resolution (edge).

Honourable mentions: Git Re-Basin cuts the loss barrier from 1.121 to 0.127 and lifts merged accuracy from 0.514 to 0.948 (interp). Quantisation-aware scaling makes int8 sparse on-device training work: 0.700 against 0.305 without it (edge). mSPRT holds the A/A false-positive rate at 0.005 where daily peeking gives 0.255 (causal). DeepSurv reaches C-index 0.775 against 0.585 for Cox on nonlinear risk (survival). Clustered standard errors cover 0.95 against 0.75 for i.i.d. ones (evaluation). Mixed negatives triple in-batch Recall@20, 0.369 against 0.115 (recsys). DINO-WM plans to the goal 0.83 of the time in frozen features against 0.17 in pixels (world). Edge-level circuit discovery recovers the planted circuit at AUC 1.0 against 0.93 for node patching (interp). Execution traces double execution-prediction accuracy, 0.300 against 0.156 (code). FSQ uses 0.91 of a 2^10 codebook against 0.64 for VQ (audio).

### Wave 10

The ten most interesting wave 10 findings (R-wave10, one seed at test budgets):

1. **Bellman–Ford message passing generalises to new entities.** NBFNet keeps link-prediction MRR at 1.000 on an inductive split, against 0.439 for R-GCN and 0.382 for common neighbours (graphs).
2. **Tournament sampling makes a stronger watermark.** At 50 tokens the SynthID-style watermark is detected 0.91 of the time at 1% FPR, against 0.45 for a green list with δ = 2, with a better judge log-likelihood (−1.62 against −2.08) (language).
3. **Density-ratio weighting fixes covariate shift.** In 5 dimensions uLSIF cuts target MSE from 1.897 to 0.179 (oracle weights 0.211), and weighted conformal restores 0.905 coverage where split conformal gives 0.792 (shift).
4. **Absorbing states make GAIL work.** From one CartPole demonstration, GAIL with a gradient penalty and absorbing states reaches 193–200, against 20–45 for wave 1's GAIL and 181 for behaviour cloning; AIRL's reward transfers to rotated dynamics where GAIL's does not (control).
5. **Query selection gets DETR off zero.** On scenes DETR with two-stage query selection reaches mAP 0.303 at 500 steps, against 0.002 for plain DETR (vision).
6. **Outlier exposure is the biggest OOD gain.** Far-OOD AUROC rises from 0.580 to 0.881, against 0.729 for mixup and 0.704 for k-NN, the best post-hoc feature score (shift).
7. **DivideMix and co-teaching beat robust losses on noisy labels.** At 40% symmetric noise DivideMix reaches 0.727 and co-teaching 0.637, against 0.567 for cross-entropy and at most 0.597 for any robust loss (labels).
8. **MMD adaptation nearly reaches the oracle.** On shapes to textured shapes MMD scores 0.612 against 0.238 source-only and 0.618 for a target-trained oracle; DANN reaches 0.406 (shift).
9. **Gated attention finds the key instance.** Gated attention MIL reaches bag AUC 0.910 and instance AUC 0.922, against 0.797 for mean pooling and 0.856 for max pooling (labels).
10. **Structured decoders parse well and always give trees.** Biaffine + Eisner reaches UAS 0.925 against 0.873 greedy and 0.892 for offset tagging; the chart parser reaches F1 0.987 and PP accuracy 0.982, against 0.979 and 0.768 for a treebank PCFG (parsing).

Honourable mentions: calibrated MLLS estimates label-shift weights at MSE 0.027 against 0.098 for BBSE (shift). TD3+BC scores 0.97 on expert Pendulum data against 0.59 for CQL and 0.23 for IQL (control). V-ZB pipelines cut the bubble to 0.208 at half of 1F1B's memory, against 0.280 (parallel). Spline-kernel Hawkes fits a delayed kernel at −1.087 per event against −1.139 for the exponential kernel and −1.066 for the truth, and neural TPPs win only on inhibition, reproducing the 2026 re-evaluation (events). Two hyperbolic dimensions rank a hierarchy at 1.65 against 2.59 for ten Euclidean ones (graphs). KoLeo lifts DINO's kNN accuracy from 0.508 to 0.708 (selfsupervised). GE2E with PLDA cuts speaker-verification EER to 0.146 from 0.342 for mean MFCCs (speech). ELR predicts 49% of flipped labels correctly against 3% for cross-entropy (labels). EBM matches GBT (0.880 against 0.879) and beats NAM (0.653) while ranking both planted pairs first (classic). Nash-MTL gives the best multitask Δm on digits, −3.5% against −8.8% for equal weights (multitask).

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

Wave 7 (Run-w7):

| Theme | Experiment | Result | What it means |
|---|---|---|---|
| audio | SpokenDigitRecognition, Conformer, Mamba, transformer, BiGRU | DER 0.117, 0.125, 0.227, 0.805 vs 0.031 to 0.070 for a frame-wise MLP | No sequence encoder beats the MLP at 150–200 steps; the Conformer needs 300 (R-wave7). |
| audio | SpokenDigitRecognition, HuBERT and wav2vec 2.0 | DER 0.578 and 0.609 vs 0.609 from scratch | Self-supervised pretraining gives no reliable gain; the sign varies between seeds. |
| audio | SourceSeparation, Conv-TasNet | SI-SNRi 2.1 dB vs 4.8 dB for an STFT-mask MLP | Under-trained at 150 steps. |
| control | ClassicalControl, DDP | Pendulum swing-up success 0.00 vs 1.00 for iLQR | Finite-difference second-order terms break DDP. |
| control | ClassicalControl, model-based diffusion | 0.23 / 243.4 / 11.10 vs MPPI 0.19 / 239.5 / 10.87 | No advantage over MPPI or CEM at equal rollouts. |
| control | ClassicalControl, domain randomisation | Held-out return -220.1 vs -218.6 nominal | Randomising cannot cover both motor polarities with one policy; RMA can. |
| control | GoalsImitationAndTeams, VDN and QMIX | -25.7 and -37.8 vs -23.4 for independent DQN | Value decomposition does not help, and QMIX diverges to random level. |
| evolution | BlackBoxOptimizers, OpenAI-ES | Cart-pole 173.0 vs 356.6 for random search; Schwefel 2,150 vs 1,560 random | ES falls below random search at these budgets. |
| graphs | GraphLearningShootout, GRAND | 0.750 at T = 8 vs 0.963 for GCN | Graph diffusion ODEs trail GCN. |
| graphs | GraphLearningShootout, DropEdge and APPNP | DropEdge 0.313 at depth 32, same as GCN; APPNP 0.721 vs 0.851 GCN | Neither helps here. |
| graphs | GraphLearningShootout, Correct & Smooth at noise 2.0 | 0.321 vs 0.348 for the MLP | Nothing to correct when the base predictor is at chance. |
| graphs | GraphStructureShootout, rewiring and random node features | Virtual node 0.156, FoSR 0.219 vs GIN 0.125 at depth 3; RNF 0.050 vs GIN 0.100 on CSL | Rewiring gives little, and RNF does not get past 1-WL. |
| graphs | GraphStructureShootout, DiffPool | 0.400 vs 0.467 for flat GIN on trees vs cycles | Learned soft clustering loses to flat readout. |
| graphs | GraphPretraining, DGI, GraphCL, GraphMAE | 0.449, 0.453, 0.442 at 5% labels vs 0.533 for GAE | Contrastive and masked graph SSL lose to a plain autoencoder. |
| graphs | GraphReasoning, Prim | 0.328 to 0.531 at 32 nodes | Max aggregation and hints do not help on Prim. |
| graphs | GraphGeneration, DiGress | Valid 0.583 vs 0.792 for GraphRNN; degree MMD 0.117 vs 0.014 for Erdős–Rényi | Discrete graph diffusion is weak at this scale. |
| numerics | ImplicitDepth, DEQ | 0.907 to 0.920 vs 0.987 for an 8-step unroll | Less memory, lower accuracy, 6–9x slower. |
| numerics | GradientCompressionShootout, error feedback with Adam | Top-1% 0.57 with EF vs 0.85 without | Error feedback hurts under Adam; it helps under SGD. |
| numerics | WarmStartGap, shrink and perturb | 0.727 vs 0.741 warm start | There is no warm-start gap at this scale. |
| probabilistic | BayesianNeuralNetworks, cold posterior | Gap NLL 0.55 / 0.58 at T = 0.3 / 0.1 vs 0.08 at T = 1 | No cold-posterior effect; cooling only hurts. |
| probabilistic | BayesianNeuralNetworks, IVON | Gap NLL 1.94; OOD AUROC 0.652 vs 0.678 for MAP | Overconfident in the gap and below MAP on images. |
| probabilistic | GaussianProcessKernels, deep GPs on heteroscedastic data | NLL 0.329 / 0.378 / 0.491 for 1 / 2 / 3 layers vs 0.276 exact GP | Depth makes it worse. |
| probabilistic | KernelHypothesisTests, deep-kernel MMD | Power 0.125 on mean and variance shifts vs 0.542 / 0.333 for plain MMD | Too few points per half to learn a kernel. |
| probabilistic | TransportMaps, ICNN Monge map | L²-UVP 2.89% vs 1.62% for Sinkhorn | Not converged in 300 steps. |
| probabilistic | TopologicalAutoencoders | Trustworthiness 0.849 vs 0.876 for a plain AE | Only KL₀.₁ improves. |
| probabilistic | NonparametricClustering, static blobs | DP Gibbs ARI 0.923 vs 0.991 for GMM + BIC | DP mixtures pay off only on streams. |
| reasoning | LatinSquarePuzzles, semantic loss | Solved 0.420 vs 0.455 for the plain MLP | The constraint loss does not help; SATNet does. |
| reasoning | RuleLearning, neural logic machine | 0.847 / 0.711 / 0.767 vs 0.889 / 0.894 / 0.931 for a GNN | The NLM trails the GNN on every task. |
| reasoning | ArithmeticExtrapolation, Neural GPU | 0.00 at 24 bits (0.34 at 12) | No length generalisation to 4x. |
| reasoning | ArithmeticRl, ES and MeZO | 0.280 vs 0.413 for GRPO | Zeroth-order fine-tuning loses at equal reward evaluations. |
| recsys | ClickModels, NeuMF | Recall@10 0.559 vs 0.583 for BPR-MF | A learned interaction does not beat the dot product. |
| recsys | SequentialRecommendation, DIN and HSTU | DIN AUC 0.619 vs 0.649 mean pooling; HSTU AUC 0.653 vs 0.677 SASRec | Target attention and HSTU give no gain here. |
| recsys | RetrievalShootout, ColBERT | Paraphrase MRR 0.24 vs 0.41 for DPR | Late interaction trails single-vector DPR at this scale. |
| recsys | LoggedBanditFeedback, CRM | True value 0.781 vs 0.831 for greedy on a reward model | Counterfactual risk minimisation loses to the direct model. |
| recsys | TabularShootout, prior-fitted network | 0.720 / 0.602 / 0.498 vs 0.807 / 0.842 / 0.855 for the best baselines | Far below every baseline at a CPU pre-training budget. |
| recsys | TabularShootout, FT-Transformer and RealMLP on sin(6x) | FT 0.881 vs 0.882 MLP; RealMLP 0.489 on sin(6x) vs 0.943 for PLR | FT-Transformer gains nothing; RealMLP's frequency init fails. |
| recsys | EmbeddingSearch, Matryoshka | 0.389 at 8 dims vs 0.482 for PCA of a plain model | Beats truncation but not PCA. |
| safety | CausalAndFairness, V-REx | Worst group 0.318 vs 0.379 for ERM | V-REx is worse than ERM, like IRM in wave 6. |
| selfsupervised | SelfSupervisedBenchmark, I-JEPA full recipe | 0.409 vs 0.503 for I-JEPA-lite at 200 steps | The full recipe does not lift I-JEPA at this budget. |
| synthesis | ProgramSynthesisShootout, DeepCoder sort-and-add, library learner | 0.73 and 0.60 vs 0.83 for plain enumeration | Only probability ordering beats enumeration. |
| synthesis | SymbolicRegressionBenchmark, lexicase selection | 3 of 4 equations vs 4 of 4 for tournament | Lexicase gives no edge here. |
| techniques | GanExperiment, optimistic and extragradient Adam | Sliced W1 0.974 and 1.039 vs 0.928 alternating, from 0.897 | The bilinear-game gains do not carry over to WGAN-GP. |

Wave 8 (Run-w8):

| Theme | Experiment | Result | What it means |
|---|---|---|---|
| agents | ToolAgentBench, ToolRL and ReTool | Needless calls 0.84 vs 0.84 for the SFT start and 0.80 for Toolformer | RL does not teach the model when not to call; every arm scores 0.00 on 3-digit operands. |
| agents | ToolAgentBench, ArCHer | 0.51 at 6k interactions vs 0.98 for trajectory GRPO | Fastest early, then falls below a random player (0.68). |
| audio | SourceSeparation, DPRNN | 7.5 dB at 1,500 steps vs 8.3 dB for an STFT-mask MLP (R-wave8) | Beats Conv-TasNet, but no time-domain model passes the mask MLP. |
| audio | SpokenDigitRecognition, LEAF and learned PCEN | DER 0.125 and 0.11 at 20 dB vs 0 for MFCC (R-wave8) | Learned front ends do not beat MFCC; PCEN helps only at 0–5 dB. |
| compression | PruningShootout, network slimming | 0.870 with 50% of channels removed vs 0.948 for a uniform thin network from scratch | Training small from scratch beats pruning. |
| compression | PruningShootout, SNIP and GraSP at 90% | 0.277 and 0.113 vs 0.543 for random pruning with SNIP's layer ratios | The saliency scores add nothing beyond layer ratios. |
| compression | CompactModels, KD and DKD | 0.643 vs 0.673 for the student alone | Logit distillation hurts here. |
| compression | CompactModels, HAWQ | 0.507 at 2.47 bits, the same as a random allocation | Hessian-aware allocation gives no edge. |
| context | BeaconCompression, Activation Beacon 4x | 0.92 / 0.98 vs 0.92 / 0.98 for mean pooling | Learned beacons beat pooling only at 8x. |
| data | DataValuation, prototype pruning | Easy 0.566 and hard 0.612 vs 0.716 for random at n = 8,000, keeping 30% | Random pruning wins at every size; the easy/hard crossover does not reproduce. |
| kernels | KernelTuning, autotuning | Best tuned kernel 4.41 GFLOP/s vs 24.1 for the core matmul | Java tilings never approach the core kernel, and the cost model's rank correlation is 0.39. |
| kernels | KernelTuning, MCTS decomposition search | No rank-7 2×2×2 algorithm found; ALS found one in 21 restarts | The simplified game's prior favours naive terms. |
| kernels | KernelTuning, Strassen and Winograd | 0.27–0.36x and 0.30–0.35x of core speed | The fast algorithms lose to the core at these sizes. |
| kernels | KernelTuning, 2:4 and block sparsity | 2:4 at 0.09x of dense; BSR wins only at 95% | Sparsity needs about 95% to pay off on this CPU. |
| language | TokenizerShootout, Length-MAX and SuperBPE | Length-MAX 0.330 tokens per char vs 0.321 for Unigram; SuperBPE equals BPE | Unigram is better on every measure. |
| privacy | FederatedShootout, Ditto, FedRep, Per-FedAvg | 0.874, 0.847, 0.838 vs 0.900 for FedAvg + fine-tune under label skew | Plain fine-tuning matches or beats the dedicated methods. |
| privacy | FederatedShootout, FedAdam | Accuracy spread 0.430 across server rates vs 0.398 for FedAvgM | Not the most robust to the server rate. |
| privacy | PrivacyAttacks, PATE | Confident-GNMax 0.600 and GNMax 0.827 (ε 16.3) vs 0.910 for DP-SGD at ε 9.99 | DP-SGD wins at equal privacy. |
| privacy | PrivacyAttacks, LiRA unlearning audit | AUC 0.548 after NPO vs 0.560 for the loss attack | LiRA does not detect leftover memorisation better. |
| robustness | ShiftAdaptation, BN statistics and TENT on a label-shifted stream | 0.177 and 0.174 vs 0.427 without adaptation | Batch statistics break when a batch holds one class. |
| sciml | PdeShootout, plain PINN on hard PDEs | Burgers 0.496, Allen–Cahn 0.993, advection 0.893 vs 0.23, 0.11, 0.0034 for a coarse solver | A classical solver wins everywhere. |
| sciml | PdeShootout, gated PIKAN | 0.344 / 0.207 vs 0.155 / 0.037 ungated | Gating hurts the KAN. |
| sciml | OperatorLearning, Transolver and LANO | 0.576 and 0.555 vs 0.167 for FNO on Burgers | Physics attention trails FNO at this size. |
| sciml | OperatorLearning, PDE-Refiner | Correlation time 14.8 vs 34.3 for the same network trained on one-step MSE | Refinement shortens useful rollouts here. |
| sciml | OperatorLearning, pushforward trick | 0.0185 vs 0.0174 without it | No gain. |
| sciml | DynamicsLearning, Koopman methods | Lorenz-63 valid steps: DMD 1, EDMD 0, autoencoder 7 vs 200 for NG-RC | Reservoir baselines win on nonlinear systems. |
| sciml | DataAssimilation, score-based assimilation | RMSE 5.02 vs 0.92 for the EnKF | The diffusion model loses to the classical filter. |
| timeseries | LongHorizonForecasting, Autoformer, FEDformer, TimesNet, transformer | 0.212, 0.098, 0.076, 0.155 vs 0.036 for DLinear at H = 24 | Transformers lose at test budget, though they led in the report's longer run (R-wave8). |
| timeseries | LongHorizonForecasting, TimesNet on two periods | 0.304 vs 0.025 for DLinear | Periods as 2-D images do not beat a linear map. |
| timeseries | AnomalyBenchmark, Anomaly Transformer | F1 0.39 and point-adjusted F1 0.85 vs 0.86 for random scores | Point adjustment flatters random scores more than the model. |
| timeseries | AnomalyBenchmark, Deep SVDD and Deep SAD | 0.829 and 0.856 vs 0.989 for kNN on tabular data; 0.38 on local density | Deep one-class models trail simple detectors. |
| timeseries | ProbabilisticForecasting, zero-shot token LM | MSE 0.60 vs 0.072 for seasonal naive on the seasonal series (R-wave8) | A Chronos-style token LM stays weak at CPU budget. |
| vision | DensePrediction, ASPP | mIoU 0.564 vs 0.738 for the U-Net | Atrous pooling does not match skip connections here. |
| vision | DetectionShootout, DETR and DN-DETR | 0.020 and 0.005 at 900 steps vs 0.505 for CenterNet at 300 | Set prediction is far too slow to converge, and denoising does not help. |
| vision | GeometricVision, DGCNN | 0.945 vs 0.975 for PointNet | Edge convolutions give no gain at 7x the time. |
| vision | GeometricVision, G-CNN on distorted glyphs | 0.448 vs 0.633 for a CNN | Rotation equivariance hurts on non-rotational distortion. |
| vision | SceneFields, SIREN signed distance | Chamfer 0.166 vs 0.028 for ReLU | SIREN does not give the best distance field. |
| vision | SceneFields, NeRF | 21.50 dB vs 25.68 dB for 3D Gaussian splatting | Lower quality and about 20x slower to render. |

Wave 9 (R-wave9):

| Theme | Experiment | Result | What it means |
|---|---|---|---|
| recsys | MultiTaskRanking, ESMM | CVR AUC 0.709–0.714 vs 0.784–0.787 in click space | No unobserved selection bias for ESMM to fix; the click-space models share the same body. |
| recsys | TabularShootout, PFN and TabICL-lite at test budgets | 0.480–0.640 vs 0.830 for GBT and 0.843 for a random forest | Far too short; the 16x and 64x rows were not run. |
| causal | HeterogeneousEffects, causal forest | PEHE 0.563 vs 0.473 for the R-learner; 90% intervals cover 0.498 | No better than meta-learners, and its intervals under-cover. |
| alignment | RewardOveroptimization, ensembles and WARM | Best-of-64 gold 0.451–0.528 vs 0.500 single proxy | The proxies are barely above chance (0.586), so ensembling has nothing to average. |
| alignment | GeneralPreferences, SPPO and Nash-MD | Worst-case win rate 0.354–0.464 vs 0.492 at the start | No method improves the worst case. |
| alignment | PreferenceShootout, R-DPO | Length 5.04 vs 5.03 for DPO; controlled win rate 0.483 vs 0.690 | The length term does not shorten answers. |
| alignment | PreferenceShootout, XPO | Gold 0.298 vs 0.300 for online DPO | Exploration never finds the rare best response. |
| reasoning | ProcessSupervision, Math-Shepherd | Best-of-16 with PRMs 0.133–0.167 vs 0.217 for the first sample | Good labels (0.856), but the verifiers are too weak. |
| reasoning | ProcessSupervision, PRIME | 0.277 vs 0.275 outcome-only | Null result. |
| reasoning | ProcessSupervision, Step-DPO | 0.180 vs 0.187 solution-level DPO and 0.312 for STaR | Step-level pairs give no gain. |
| reasoning | ArithmeticRl, RLVR from one example | 0.105 vs 0.180 for the full set and 0.245 at the start | RL hurts held-out problems on this lookup-like task. |
| code | CodeExecution, self-repair | 0.52–0.54 at 16 calls vs 0.56 for resampling | As Olausson et al. predict at a low first-sample solve rate. |
| data | DataCuration, ODM and Aioli | Target loss 2.598 and 2.648 vs 2.619 uniform | Online mixing does not beat uniform; offline RegMix (2.347) does. |
| data | DataCuration, learning-progress curricula | 0.083–0.243 vs 0.277 for fixed pacing | No teacher beats fixed pacing; self-paced learning stalls. |
| data | SyntheticTraining, diffusion samples | 0.194–0.202 vs 0.199 real only and 0.259 for mixup | Synthetic images do not help at this compute. |
| interp | RepresentationComparison, CKA and SVCCA | Layer correspondence 0.083–0.583 vs 0.667 for best-neuron correlation | The simple baseline wins. |
| interp | LatentKnowledge, CCS | 0.575 vs 0.570 for a random direction | CCS finds no latent knowledge here. |
| interp | ExplainabilityBenchmark, attention rollout and flow | 0.655 vs 0.730 raw attention and 0.865 saliency | Aggregating attention across layers hurts. |
| data | TrainingAttributionExperiment, TRAK | LDS 0.455–0.523 vs 0.781 for the exact influence function | On a small model the exact method wins. |
| document | GlyphOcr, CRNN with CTC | CER 0.311 vs 0.269 for segmentation | Segmentation still wins, even on touching glyphs. |
| document | DocumentParsing, Donut-style | Field F1 0.000 vs 0.838 for OCR + rules | The end-to-end model learns nothing at 50 steps. |
| multimodal | FusionShootout, frozen-LM bridges | VQA 0.640–0.650 vs a 0.770 ceiling | Every bridge sits at the majority-answer prior. |
| multimodal | FusionShootout, Transfusion | Generation 0.067–0.133 (chance) vs 0.433–0.567 for VQ tokens | Discrete tokens win at this budget. |
| world | VideoGeneration, factorised space-time attention | Clip MMD 1.18 vs 1.17 frame-wise and full 3-D | No quality gain and no cost saving. |
| world | VideoGeneration, Self Forcing | Count error 6.46 vs 4.65 for teacher forcing | Fails at test scale. |
| world | WorldModelShootout, DreamerV3 tricks | CartPole 9.2 / 32.2 / 36.2 vs 9.6 / 58.0 / 39.2 default | No help at 600 steps. |
| world | WorldModelShootout, SlotFormer | 9.5 dB vs 12.7 dB for SimVP | Not validated; likely broken or badly undertrained. |
| audio | AudioGeneration, monotonic alignment search | Digit error 1.60 / 1.07 vs 1.20 / 0.89 for F5 padding | Against the hypothesis; all models undertrained. |
| sciml | PdeShootout, fixes for advection β = 30 | Relative L2 about 1.0 vs 0.003 for a solver | Curriculum, time marching and periodic boundaries do not rescue it. |
| sciml | OperatorLearning, Transolver and LANO | 0.45–0.72 vs 0.19–0.21 for FNO | FNO leads everywhere, also with half the grid dropped. |
| edge | EdgeDeployment, once-for-all | Largest / smallest 0.625 / 0.330 vs 0.860 / 0.570 from scratch | Progressive shrinking loses at 160 steps. |
| edge | EdgeDeployment, DS-CNN keyword spotter | 0.917 vs 0.975 for a plain CNN | The depthwise-separable design is not best at this size. |

Wave 10 (R-wave10, one seed at test budgets; long `main()` Runs not executed):

| Theme | Experiment | Result | What it means |
|---|---|---|---|
| labels | SemiSupervisedShootout, pseudo-labelling, Π-model and Mean Teacher | 0.537, 0.570 and 0.567 vs 0.567 for augmented supervised training | Consistency training adds nothing over augmentation at 4 labels per class. |
| labels | SemiSupervisedShootout, MixMatch | 0.540 vs 0.567 | Below augmented supervised training. |
| labels | SemiSupervisedShootout, VAT on digits | 0.303 vs 0.567 | VAT needs the cluster assumption; it works on spirals (0.989). |
| labels | SemiSupervisedShootout, JEPAMatch | 0.457 vs 0.620 for FlexMatch | The SIGReg regulariser lowers accuracy. |
| labels | NoisyLabelShootout, robust losses under pair-flip noise | NCE+RCE 0.477 and JAL 0.497 vs 0.527 for CE | The robust losses lose to CE on asymmetric noise. |
| labels | NoisyLabelShootout, forward correction with estimated T | 0.517 vs 0.527 for CE and 0.733 with the true T | The T estimate throws the whole gain away. |
| labels | WeakLabelShootout, DLLP | 0.687 vs 0.880 supervised at bag size 8; 0.13–0.21 for every method at 128 | No graceful degradation, and OT pseudo-labels do not help. |
| shift | OodShootout, ASH, SCALE and AdaSCALE | Near AUROC 0.683–0.793 vs 0.829 for energy | Activation shaping hurts on this toy; only ReAct (0.832) helps. |
| shift | OodShootout, OpenMax | AUROC 0.730 vs 0.778 for max logit; 0.660 vs 0.682 briefly trained | OpenMax never beats max logit. |
| shift | SelectivePrediction, learning to defer | 0.471 and 0.455 vs 0.487 for a confidence threshold | The learned deferral rules lose to thresholding. |
| shift | DomainAdaptation, SHOT on shapes | 0.254 vs 0.238 source-only and 0.530 for TENT | Source-free adaptation fails on the appearance shift (it matches the oracle on moons). |
| shift | DomainAdaptation, EATA, SAR and CoTTA | 0.549–0.565 vs 0.549 for TENT; about 0.21 on single-class batches vs 0.336 without adaptation | TENT did not degrade, and every adapter breaks on single-class batches. |
| shift | DomainGeneralization, MixStyle and SWAD | 0.330 and 0.337 vs 0.340 and 0.38 for ERM | Neither beats ERM; V-REx (0.448) is best. |
| selfsupervised | SelfSupervisedBenchmark, SwAV and MoCo | Probe 0.409 and 0.477 vs 0.560 and 0.510 for SimCLR | No gain over SimCLR at small batch. |
| selfsupervised | SelfSupervisedBenchmark, data2vec and iBOT | Probe 0.404 and 0.365 vs 0.568 for MAE at 80 steps | Teacher-feature targets lose to pixels here. |
| metric | MetricLearningShootout, pair losses and angular margins | Unseen R@1 0.456–0.569 and 0.531 vs 0.625 for cross-entropy | No metric loss beats plain cross-entropy on unseen classes; semi-hard mining does not beat random triplets. |
| metric | DeepClustering, DEC, IDEC and SCAN | 0.265–0.330 vs 0.270–0.350 for k-means on the same features | Deep clustering gives no gain over k-means. |
| interp | LatentKnowledge follow-up, CCS with normalisation and ensembles | 0.523–0.550 vs 0.550 for a random direction and 0.830 zero-shot | Wave 9's negative stands; CCS follows a distractor perfectly (1.000). |
| control | ValueBasedRecipes, PQN | CartPole 75 vs 137 for DQN at 2k steps; 161 vs 178 at 3k | No gain over DQN per environment step. |
| control | PlasticityAndScale, multi-skip residuals | Gradient spread 13.0 vs 6.1 for a plain MLP | Against the hypothesis. |
| control | PlasticityAndScale, passive critic | Held-out TD error 0.484 on a wide critic's targets vs 0.181 on its own | Better targets do not help the narrow critic. |
| control | ModernOffPolicy, CrossQ at 600 steps | −1,479 vs −1,324 for DroQ and −1,261 for SAC with naive BN | Does not match DroQ, and naive BN does not diverge at this budget. |
| classic | BoostingShootout, ordered boosting | Log loss 0.535 vs 0.534 plain and 0.520 one-hot | Ordered boosting gives nothing here; feature combinations do (0.498). |
| classic | BoostingShootout, quantile forest on a bimodal target | CRPS 0.510 vs 0.488 for NGBoost | The forest does not win where it was expected to. |
| events | PointProcessShootout, intensity-free TPP | NLL −0.659 vs −0.555; 600–5,000 vs up to 160,000 events/s for thinning | Neither better nor faster. |
| events | IrregularSeriesShootout, latent ODE and mTAN | Interpolation MSE 0.751 vs 0.399 linear; classification 0.767 vs 0.933 forward-fill GRU | Simple baselines win at 60% missingness. |
| events | IrregularSeriesShootout, SAITS and CSDI | MAE 0.724 and 0.868 vs 0.589 BRITS and 0.403 for a Gaussian process | The deep imputers lose at test budget. |
| parsing | ParsingShootout, tree-CRF loss | F1 0.960 vs 0.987 for the margin loss | The CRF loss was expected to win slightly. |
| restoration | RestorationShootout, PnP-HQS deblurring | 19.84–21.81 dB vs 21.41–22.75 dB for Wiener at 200 steps | Wiener wins on every kernel; PnP led on motion kernels in an earlier 300-step run. |
| restoration | RestorationShootout, Median2Median on correlated noise | 18.76 dB vs 20.03 dB for a median filter | Fails the case it was designed for. |
| vision | VisionShootout, FlexiViT | 0.313 at patch 4 vs 0.695 for a dedicated patch-4 ViT | Far from "within 1–2 points" at 150 steps. |
| edge | EdgeDeployment, BigNAS in-place distillation | 0.520 / 0.465 vs 0.915 / 0.780 for progressive shrinking at 300 steps | Distilling early from a weak teacher hurts. |
| multimodal | FusionShootout, two-stage VQA bridge (test budget; long run pending) | 0.640 vs an answer prior of about 0.65; alignment accuracy 0.018 | Stage 1 barely aligns at 100 steps. |
| multimodal | FusionShootout, latent Transfusion (test budget; long run pending) | Generation 0.133 vs 0.433 for VQ tokens | Better than wave 9's pixel model, still far behind. |

Overturned by the Run: second-order MAML was listed here (1.89 vs 0.81 for first-order, with finite-difference Hessian-vector products). With exact second-order maths (PR #105) it scores 0.65 and is now the best meta-learner in MetaLearningExperiment.
