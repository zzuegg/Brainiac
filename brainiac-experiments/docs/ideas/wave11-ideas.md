# Brainiac: wave 11 ideas

*Compiled 2026-11-18 against the local branch `wave10-int` (main tree, which now holds waves 1–10), plus every worktree under `.claude/worktrees/*` (54 worktrees). Waves 1–10 implemented about 1,100 techniques; see `experiment-results.md` (waves 1–10, including every wave's "Not done" lines, scope cuts and the negative-results tables), the wave 10 builders' reports, `unimplemented-ideas.md` and `wave5-ideas.md` to `wave10-ideas.md`. A survey of the `experiments.*` themes shows which areas are still thin: `code`, `edge`, `kernels`, `streaming` and `selfsupervised` (2 classes each), `multiagent`, `privacy`, `robustness` and `synthesis` (3 each), and `causal` and `learning` (4 each). Several whole fields still have no code at all: data poisoning, backdoors and Byzantine-robust aggregation, black-box and decision-based attacks, model extraction, local differential privacy and the classical DP mechanisms, neural combinatorial optimisation, streaming sketches, learned data structures and algorithms with predictions, quasi-experimental econometrics, causal discovery beyond NOTEARS/DAGMA/GES and the PC skeleton, probabilistic circuits, post-hoc recalibration beyond temperature scaling, cheap ensembles, music information retrieval, statistical machine translation, image anomaly detection, video action recognition, feature matching, and the bandit, exploration, hierarchical and multi-agent RL families that the wave 10 deep-RL recipes left out. Wave 11 targets those areas and avoids the areas waves 9 and 10 have just covered, except for follow-ups:*

- *(a) security and privacy of learning: Byzantine-robust federated aggregation and model-poisoning attacks, data-poisoning backdoors and their detection and repair, score-based, decision-based and transfer attacks, a reliable robustness evaluation, model extraction, local DP, private selection and query release, correlated-noise DP training, exact unlearning, and corruption robustness by augmentation;*
- *(b) learning for combinatorial optimisation and algorithms: constructive, improvement and non-autoregressive neural solvers for routing, unsupervised GNN solvers for graph problems, learned branching and learned SAT local search, neural-guided bottom-up program synthesis, streaming sketches, learning-augmented sketches, learned indexes and Bloom filters, and caching with predictions;*
- *(c) causal discovery and quasi-experiments: PC/FCI, LiNGAM, ordering-based and bivariate methods, time-series discovery, synthetic control and its 2025 triply robust successor, staggered difference-in-differences, regression discontinuity, balancing weights and sensitivity analysis; plus probabilistic circuits, variational GMMs, EM for linear dynamical systems, hierarchical and intermittent forecasting, and parametric survival models;*
- *(d) uncertainty and classic-model gaps: post-hoc recalibration maps, Venn–Abers, conformal follow-ups and conformal training, snapshot/FGE/MIMO/packed ensembles, stacking, test-time augmentation, diversity-seeking ensembles, ensemble distillation, multiple-choice learning, binary networks, restricted Boltzmann machines, Tsetlin machines and monotonic networks;*
- *(e) perception and language gaps: pitch estimation, onsets and beats, chords, audio fingerprints, sound events, word alignment, noisy-channel correction, SimCSE, unsupervised constituency parsing, image anomaly detection (PatchCore, Dinomaly), action recognition, feature matching with RANSAC, learned edge detection, deep hashing and FFT convolutions;*
- *(f) RL beyond the wave 10 recipes: Bernoulli, adversarial, best-arm, dueling and combinatorial bandits, deep exploration and count-based bonuses, skill discovery, MBPO, goal-conditioned hierarchies, preference-based RL, average-reward RL, complete value factorisation, mean-field and heterogeneous-agent MARL, zero-shot coordination, and a pixel-control follow-up.*

*Four items are fair-test follow-ups or lift earlier cuts, marked "— follow-up" in their titles: 11.1.10 (re-evaluating wave 8's PGD-AT/TRADES/1-Lipschitz numbers with AutoAttack, because PGD-only evaluations overstate robustness), 11.3.14 (the missing-aware Kalman smoother that wave 10 10.5.11 had to write locally), 11.4.4 (conformal sets beyond wave 5–6's LAC/APS, which every later wave reused), and 11.6.17 (pixel CartPole for world models and pixel/continuous control renderers, cut from wave 9 9.5.6 and reported as a core gap by wave 10 WP4). Several more items close the experiments-side gaps the wave 10 builders reported, as small enabling steps inside an item: a public `GaussianMixture` posterior (11.3.13), missing observations and EM in `StateSpaceFilter` (11.3.14), a shared Weibull helper (11.3.17), a `LogitsProcessor` hook on `WindowLm` sampling (11.5.8), public convolution helpers and a 2-D FFT (11.5.17), and pixel renderers plus `ContinuousMountainCar` rows in the RL harness (11.6.17).*

*It also adds the late-2024 to 2026 items found by a web sweep, each checked against the paper's abstract or text rather than a search snippet: Byzantine-robust FL with learnable aggregation weights (ICLR 2026), activation-gradient poisoned-sample detection AGPD (ICLR 2025), BLT correlated-noise DP-FTRL (2024/2025), S3T exact unlearning with sliced PEFT training (ICLR 2025), preference optimisation for neural combinatorial optimisation (ICML 2025), "Graph neural networks are heuristics" (2026), learning-augmented Misra–Gries and Frequent Directions (ICLR 2025), TriOpt ordering-then-convex causal discovery (2026 preprint), triply robust panel estimators TROP (2025), Inception probabilistic circuits (AAAI 2025), robust M-estimation forecast reconciliation (2026), structured matrix scaling (Nov 2025), "Is BatchEnsemble a single model?" (2026), the fuzzy-pattern Tsetlin machine (Aug 2025), lightweight self-supervised F0 and voicing (ICPRAM 2026) after PESTO, Beat This! (2024/2025), Dinomaly (CVPR 2025), Almost Tracking anytime best-arm identification (Oct 2025), contrastive successor features for skill discovery (ICLR 2025), and QFIX value factorisation (RLJ 2026).*

*Every item was grepped over all `*.java` sources under `src/` in all modules, in the main tree and in all 54 worktrees: 102,040 files, 4,274 unique by content after de-duplication (3,316 distinct file names). The search was case-insensitive and used class names, paper names, author-coined terms and javadoc keywords. An item is listed only when nothing matched, or when the only matches were unrelated. Examples of unrelated or partial matches:*

- *"backdoor" correctly matching `BackdoorRemoval`, `CircuitBreakers` and `LatentAdversarialTrainer` (a trigger in the toy safety LM removed by latent adversarial training; no image data poisoning, no poisoned-sample detection, no trigger reverse-engineering); "poison" matching only `SpikeGuard`'s javadoc; "Byzantine", "Krum", "trimmed mean", "geometric median", "bucketing", "centered clipping" and "ALIE" matching nothing (`FederatedAlgorithms` has FedAvg, FedProx, SCAFFOLD, FedOpt and FedNova only); "Carlini" matching `LikelihoodRatioAttack` (the LiRA author, not the C&W attack); "black-box", "Square attack", "SimBA", "HopSkipJump", "boundary attack", "AutoAttack", "APGD", "DLR", "MI-FGSM" and "universal adversarial" matching nothing (`AdversarialAttacks` has white-box FGSM and PGD only, the baselines of 11.1.7–11.1.10); "knockoff"/"model extraction"/"model stealing" matching nothing; "unlearning" correctly matching `MachineUnlearning`, `UnlearningComparison` and `ClassForgetting` (approximate methods: gradient ascent, NPO, SimNPO, RMU; no SISA sharding, no S3T, no SCRUB); "randomized response", "RAPPOR", "exponential mechanism", "sparse vector", "MWEM", "PrivBayes", "tree aggregation", "DP-FTRL" and "banded" matching nothing except `Functional`'s banded attention mask; "AugMix", "TrivialAugment" and "Jensen–Shannon consistency" matching nothing (JSD only in distillation losses; "RandAugment" correctly matches core `Augmentation`, the 11.1.17 baseline).*
- *"travelling salesman", "TSP", "2-opt", "Held–Karp", "POMO", "Sym-NCO", "active search", "Erdős goes neural", "PI-GNN", "independent set", "strong branching" and "set cover" matching nothing; "pointer network" and "convex hull" correctly matching `PointerNetwork` (Vinyals et al. on convex hulls; reused as the 11.2.1 baseline); "max-cut" correctly matching `SimulatedBifurcation`, `ProbabilisticBits` and `IsingSolversTest` (bSB/aSB/dSB, simulated annealing, p-bit annealing and spectral cuts: the 11.2.8 baselines); "WalkSAT" correctly matching `GraphReasoning`/`NeuroSatTest` (the 11.2.11 baselines); "knapsack" matching `BudgetPacer` and `SparseUpdateTrainer` (a budget LP and a memory budget, not a solver benchmark); "branch-and-bound" matching `MemoryAwareScheduler` (a pipeline scheduler); "observational equivalence" correctly matching `GuidedEnumerator`/`SelfImprovingCoders` (the 11.2.12–11.2.13 baselines); "bottom-up" matching `HierarchicalVae` and `GeneratedVideo`; "sketch" matching `Sigreg`/`LeJepa` (random-slice sketches of a distribution) and `ImageTranslation` (sketch-to-image); "count-min" matching `Long.bitCount` and "count minus" substrings; "heavy hitter" matching H2O KV eviction; "reservoir sampling" correctly matching `ReservoirReplay` (a replay buffer); "Oja" correctly matching `HebbianRule`'s Oja rule (a streaming first principal component, a baseline for Frequent Directions); "Bloom filter" matching one comment in `FtrlProximal`; "LRU" matching kernel-cache comments; "learned index", "Count-Sketch", "HyperLogLog", "Misra–Gries", "Space-Saving", "Frequent Directions", "Belady" and "ski rental" matching nothing.*
- *"PC algorithm" correctly matching `CausalAndFairness` (the PC skeleton phase only, no orientation rules, no FCI: the 11.3.1 baseline); "LiNGAM", "additive noise model", "IGCI", "neural Granger", "PCMCI", "transfer entropy" and "score matching ordering" matching nothing ("Granger" matched only "lagrangeRate"); "synthetic control", "difference-in-differences", "two-way fixed effects", "event study", "Callaway", "regression discontinuity", "McCrary", "entropy balancing", "CBPS", "E-value", "Rosenbaum" and "sensitivity analysis" matching nothing ("propensity" correctly matches `CausalEstimators`, `DoubleMl` and `MetaLearners`, the IPW/AIPW baselines); "sum-product network", "probabilistic circuit", "LearnSPN", "RAT-SPN" and "Chow–Liu" matching nothing; "mixture density" correctly matching `SimulationBasedInference` and `InstrumentalVariables` (MDNs as components; the 11.4.15 baseline); "automatic relevance" and "variational Bayes mixture" matching nothing (`DirichletProcessMixture` is collapsed Gibbs); "RTS smoother" correctly matching `StateSpaceFilter` and `TrackingFilters` (no missing observations and no EM, as wave 10 WP5 reported); "reconciliation", "Croston", "intermittent", "SBA", "TSB" and "accelerated failure" matching nothing ("aft" matched a variable in `LongHorizonForecasting`); "Weibull" correctly matching `SurvivalData`, `OpenMax`'s tail fit and `PointProcessShootout`'s Weibull renewal (three private copies, unified by 11.3.17).*
- *"calibration" correctly matching `Calibration` (ECE, MCE, NLL, Brier and temperature scaling: the 11.4.1 baseline) and wave 9's `CalibrationEstimators` (estimators of ECE, not recalibration maps); "histogram binning", "isotonic", "vector scaling", "Dirichlet calibration", "beta calibration" and "Venn–Abers" matching nothing ("Platt" matched only "platter"-like substrings in splatting code; "Dirichlet" matched priors, MCTS noise and FMAPLS); "APS" correctly matching `ConformalClassifier` (LAC and APS; no RAPS, Mondrian or risk control); "conformal training"/"ConfTr" matching nothing; "snapshot" matching checkpoints and state copies, never snapshot ensembles; "BatchEnsemble" correctly matching `BatchEnsembleDense` and `TabM` (the 11.4.7 baseline); "MIMO", "packed ensemble", "super learner", "negative correlation", "DivDis", "ensemble distillation" and "test-time augmentation" matching nothing ("TTA" matches test-time adaptation); "ternary"/"BitNet" correctly matching `BitLinear` (ternary weights: the 11.4.11 baseline) and "straight-through" matching quantisers; "XNOR" matching bitwise ops, "binarized" matching `ChartParser`'s binarised grammar; "Boltzmann" and "contrastive divergence" correctly matching `PBitIsingMachine` and `DenoisingThermodynamicModel` (CD-trained p-bit Boltzmann chains with hidden p-bits; no RBM with persistent CD, parallel tempering or AIS likelihoods, which 11.4.12 adds and compares with them); "Tsetlin" matching nothing; "monotone" matching DTW, detection metrics and activations (no monotonic networks); "winner takes all" matching stereo matching; "multiple-choice" matching MCQ benchmarks.*
- *"pitch" and "fundamental frequency" correctly matching `HarmonicNoiseSynth` and `SpeakerVoices` (F0 is a synthesis input, the ground truth for 11.5.1–11.5.2; no estimator); "YIN", "CREPE", "PESTO", "spectral flux", "beat tracking", "chroma", "chord" (matched "anchor d"-style substrings), "fingerprint" (matched `OpenClBackend`) and "sound event" matching nothing; "IBM model", "word alignment", "noisy channel", "SimCSE", "compound PCFG" and "grammar induction" matching nothing; "inside–outside" correctly matching `ChartParser` (supervised span parsing; the 11.5.10 baseline); "Kneser–Ney" correctly matching `KneserNeyLm` (reused as 11.5.8's source model); "edit distance"/"Levenshtein" correctly matching the OCR and ASR metrics (reused); "PatchCore", "PaDiM", "Dinomaly", "temporal shift", "action recognition", "Harris", "RANSAC" (matched `HiddenMarkovModel`/`Nsga2` substrings), "homography", "Canny", "HED" and "deep hashing" matching nothing; "Sobel" matching `NeuralCellularAutomaton`'s perception filters; "product quantization" correctly matching `IvfPqIndex` (the 11.5.16 baseline); "FFT" correctly matching core 1-D `Complex`/audio FFTs and `DegradedImages`' blur (no 2-D FFT helper, as wave 10 WP6 reported).*
- *"Thompson" correctly matching `ContextualBandit` (linear Thompson sampling, the 11.6.5 baseline) and `BanditMixer` (data mixing); "EXP3" matching `BanditMixer` and `ProgressCurriculum` (curriculum bandits, reused as the 11.6.2 baseline); "best arm" matching `Bandit` and `InContextBandits` comments; "successive halving" correctly matching `HyperparameterSearch` (Hyperband; a fixed-budget baseline for 11.6.3); "KL-UCB", "MOSS", "IMED", "Tsallis-INF", "Track-and-Stop", "RUCB", "semi-bandit" and "cascading bandit" matching nothing ("cascade" correctly matches `ClickLogs`' cascade click model, reused as the 11.6.5 environment); "bootstrapped DQN", "randomized prior", "RLSVI", "Deep Sea" and "SimHash exploration" matching nothing (`LshSampledAttention` hashes attention keys; `NeverGiveUp` has episodic, not hash, counts); "DIAYN", "METRA", "DADS" and "skill discovery" matching nothing ("metra" matched "TestTimeTrainingLayer" — "parametrisation"-like substrings); "PETS" correctly matching `CemPlanner` (the 11.6.9 baseline; no MBPO); "HIRO" matching `TimeSeriesImputer` ("hiro"-substring), "feudal" nothing; "PEBBLE", "SURF" and "trajectory preference" matching nothing; "average reward" matching RLHF reward averages; "QMIX"/"VDN" correctly matching `ValueDecomposition` (the 11.6.13 baselines; no QTRAN, QPLEX or QFIX); "mean-field" matching mean-field variational inference; "other-play" matching "the other player"; "MAPPO" correctly matching `MultiAgentPpo` (the 11.6.15–11.6.16 baseline); "pixel CartPole" matching nothing (wave 10's `PixelObservations` renders GridWorld and `KeyDoorGrid` only).*

## How this was checked

- **Already implemented, so left out.** These turned up in the systematic check but already have code:
  - Security and privacy: FGSM and PGD (`AdversarialAttacks`), PGD-AT, TRADES and fast FGSM training, randomized smoothing, IBP/CROWN, 1-Lipschitz networks, DP-SGD with an RDP accountant, DP-FedAvg with secure aggregation, PATE, LiRA and one-run audits, gradient inversion (DLG), FedAvg/FedProx/SCAFFOLD/FedOpt/FedNova and personalised FL, approximate unlearning (gradient ascent, NPO, SimNPO, RMU), latent adversarial training and circuit breakers, the green-list and SynthID watermarks.
  - Algorithms and synthesis: pointer networks, NeuroSAT and WalkSAT, simulated bifurcation, p-bit annealing and spectral cuts, GFlowNets (trajectory balance), the LP solver `LinearProgram`, Hungarian solvers, enumerative synthesis with observational equivalence, DeepCoder, DreamCoder, RobustFill, AutoML-Zero, the Hoeffding tree, ADWIN/DDM/Page–Hinkley, FTRL-Proximal, COCOB, reservoir replay, MinHash deduplication, IVF-PQ, LSH attention.
  - Causal and statistical: NOTEARS, DAGMA, GES and the PC skeleton, IPW/AIPW, double ML, meta-learners, causal forests, uplift, 2SLS/DeepIV, CUPED and sequential tests, Kaplan–Meier, Cox, DeepSurv, DeepHit and survival forests, HMMs, Kalman/EKF/particle filters with RTS smoothing, GMM via EM, DP mixtures, exponential smoothing, DeepAR, PatchTST and the long-horizon lineup, Hawkes and neural TPPs.
  - Uncertainty and classic models: temperature scaling and ECE estimators, LAC/APS conformal sets, CQR/ACI/EnbPI, deep ensembles, MC dropout, SWAG, SNGP/DUQ, evidential heads, BatchEnsemble and TabM, Laplace, BitNet-style ternary layers, p-bit Boltzmann chains, logic-gate networks, hyperdimensional computing, capsules, hypernetworks.
  - Perception and language: MFCC/DTW templates, SincNet/LEAF/PCEN, keyword spotters, Conv-TasNet/DPRNN, speaker verification and diarisation, VAD and enhancement (wave 10), HMM/CRF tagging, Kneser–Ney, static embeddings, biaffine and span parsing, the PCFG baseline, Deep SVDD/LOF/ECOD/one-class SVM (series and tabular), optical flow, stereo, keypoint heatmaps, tracking, restoration (wave 10).
  - RL: UCB1/ε-greedy bandits, LinUCB and linear Thompson sampling, logged-bandit OPE, DQN through Rainbow and IQN, NoisyNet, RND, ICM, NGU, Go-Explore, HER, successor features with GPI, option-critic, PETS-style CEM planning, Dreamer, TD-MPC2, VDN and QMIX, MAPPO/IPPO, MADDPG/COMA, CommNet/TarMAC, referential games, CFR, PSRO, LOLA.
- **Considered but cut** (to stay near 6 × 17, or too close to an existing block):
  - Membership inference beyond LiRA (attack ensembles, quantile regression attacks): `PrivacyAttacks` already measures LiRA, the strongest of them, and the one-run audit.
  - Homomorphic encryption and secure multi-party training: not a toy-scale learning question, and secure aggregation exists.
  - Capacitated vehicle routing as its own item: folded into 11.2.2–11.2.4 as a second task on the same generator.
  - DIFUSCO/T2T diffusion over edge heatmaps: too close to 11.2.7's heatmap item and to wave 7's DiGress, which was weak at this scale.
  - FlashFill-style version-space algebra: needs a string DSL the repo does not have; 11.2.13's Probe covers learned grammars over the existing `ListPrograms` DSL.
  - Causal representation learning (iVAE, CITRIS): identifiability toys exist only for linear ICA; deferred.
  - Gaussian copula forecasting and temporal fusion transformers: the long-horizon lineup is already broad, and waves 8–10 found simple baselines winning at test budgets.
  - Extreme learning machines and random-vector functional links: random-feature kernel ridge already exists (`RandomFourierFeatures`, `NystromKernelRidge`), so the comparison would add little.
  - Neural style transfer: no metric with a ground truth; 11.5.15's edge detection and 11.5.14's matching were preferred.
  - Monocular depth and colourisation: still no renderer with depth or colour ground truth.
  - HAC and option discovery beyond HIRO: folded into 11.6.10 as a row.
  - Offline MARL (OMAR, O-MAPL): no offline multi-agent datasets; the online factorisation item 11.6.13 comes first.
- **CPU-toy filter.** Every item is pure Java with no external libraries, runs as a test class in about 10 s on a CPU, and names a task that exists in `brainiac-experiments` or a tiny new generator next to it, plus a baseline already in the repo. Per the user's current rule, **no item depends on a long-budget Run**: the hypotheses are stated at the test budget, and any larger grid in `main()` is optional and not part of the verdict. The tasks used are:
  - `ToyTasks` digits/spirals, `ShapeImages`, `ShapeScenes`, `DigitGlyphs`, `DegradedImages`, `ShiftSuites`, `LabelCorruption`, `TabularStressSuite`, `ClientSplits`, `SyntheticRecsys`, `ClickLogs`, `SyntheticGraphs`, `RandomSatFormulas`, `CausalScenarios`, `SurvivalData`, `SeasonalSeries`, `IrregularSeries`, `DriftingStreams`.
  - `SyntheticCorpus`, `SyntheticGrammar`, `ToyTreebank`, `ListPrograms`, the toy LMs (`TinyLm`, `WindowLm`, `KneserNeyLm`).
  - `HarmonicNoiseSynth`, `SpeakerVoices` and `SpokenDigitRecognition`'s formant synthesiser.
  - The `brainiac-rl` environments (CartPole, Pendulum, MountainCar, GridWorld, `Bandit`), `experimental.rl`'s `KeyDoorGrid`, `GoalEnvironment`, `MultiAgentEnvironment`, `TwoStepGame` and the matrix games, and `experimental.deeprl`'s `ContinuousMountainCar` and `PixelObservations`.
  - New small generators named per work package: `PoisonedSets` (WP1), `CombinatorialInstances` and `KeyStreams` (WP2), `DiscoveryBenchmarks` and `PanelStudies` (WP3), `ToyMusic`, `ParallelText`, `DefectImages` and `ShapeClips` (WP5), and `BanditSuites` plus the `DeepSeaChain` environment (WP6). WP4 needs no new generator.

**Format.** Wave 11 keeps wave 10's structure (checks, summary, one section per work package, dependency notes), with each item as a short block:

- **Block**: the reusable class, in `io.github.zzuegg.brainiac.experimental.<pkg>`.
- **Experiment**: the runnable measurement, in `io.github.zzuegg.brainiac.experiments.<theme>`.
- **Toy task**, **Baselines** (all already in the repo) and **Expected metric** (the number to report at the ~10 s test budget, with the hypothesis to test).

**Where an item should live:**

- `experimental.<pkg>`: a new block in `brainiac-experiments`. Fourteen new packages are proposed, each owned by one work package so that no two agents create the same package (none of these names exists yet):
  - WP1: `experimental.security` (poisoning, backdoors, robust aggregation, black-box attacks, extraction) and `experimental.dpmech` (local DP, private selection and query release, correlated-noise mechanisms). Unlearning adds to the existing `experimental.train` next to `MachineUnlearning`; augmentation adds to `experimental.robust`.
  - WP2: `experimental.combopt` (routing, graph CO, branching, SAT local search) and `experimental.sketch` (streaming sketches, learned sketches, learned indexes and filters, caching with predictions). Synthesis items add to the existing `experimental.evolve` next to `GuidedEnumerator`.
  - WP3: `experimental.discovery` (causal discovery), `experimental.panel` (quasi-experimental estimators) and `experimental.circuits` (probabilistic circuits). Forecasting items add to `experimental.timeseries`, survival to `experimental.survival`, and the GMM/Kalman items to `experimental.ml`.
  - WP4: `experimental.calibrate` (recalibration maps, Venn–Abers, conformal follow-ups) and `experimental.ensemble` (cheap and diverse ensembles, stacking, TTA, distillation, multiple-choice learning). Binary networks and monotonic layers add to `experimental.nn`; the RBM and Tsetlin machine add to `experimental.ml`.
  - WP5: `experimental.music` (pitch, onsets, beats, chords, fingerprints, sound events) and `experimental.nlp` (alignment, noisy-channel correction, sentence embeddings). Grammar induction adds to `experimental.parse`, image anomaly detection to `experimental.anomaly`, and vision items to `experimental.vision`.
  - WP6: `experimental.bandit`, `experimental.explore` (deep exploration, count bonuses, skills, hierarchy, preference-based RL) and `experimental.marl` (factorisation, mean-field, coordination, HAPPO). MBPO, average-reward RL and the pixel follow-up add to `experimental.deeprl`.
  - Existing blocks are not moved: `AdversarialAttacks`, `FederatedAlgorithms`, `MachineUnlearning`, `PointerNetwork`, `GuidedEnumerator`, `Notears`, `CausalEstimators`, `StateSpaceFilter`, `GaussianMixture`, `Calibration`, `ConformalClassifier`, `BatchEnsembleDense`, `HarmonicNoiseSynth`, `ChartParser`, `DeepSvdd`, `ValueDecomposition`, `ContextualBandit` and the others stay where they are, and the new packages import them. The enabling changes (public posterior, missing observations, Weibull helper, logits hook, conv/FFT helpers, renderers) add methods with unchanged defaults.
  - New data generators go in `experimental.data` under the names listed above; each name belongs to one WP.
- `experiments.<theme>`: a new or extended runnable experiment. New themes: `experiments.security` (WP1), `experiments.combinatorial` and `experiments.algorithms` (WP2), `experiments.discovery` and `experiments.econometrics` (WP3), `experiments.uncertainty` (WP4), `experiments.music` (WP5) and `experiments.bandits` (WP6). Other items extend `experiments.privacy`, `experiments.robustness`, `experiments.synthesis`, `experiments.causal`, `experiments.timeseries`, `experiments.probabilistic`, `experiments.survival`, `experiments.classic`, `experiments.exotic`, `experiments.language`, `experiments.parsing`, `experiments.vision`, `experiments.control`, `experiments.multiagent` and `experiments.world`.
- **core-worthy**: belongs in a core module once measured.

## Summary

| # | Work package | Items |
|---|---|---|
| WP1 | Security and privacy of learning: robust aggregation and poisoning, backdoors, black-box and transfer attacks, AutoAttack evaluation, model extraction, local DP and DP mechanisms, correlated-noise DP training, exact unlearning, corruption robustness | 17 |
| WP2 | Learning for combinatorial optimisation and algorithms: neural routing solvers, graph CO, learned branching and SAT search, neural-guided synthesis, sketches, learned data structures, caching with predictions | 17 |
| WP3 | Causal discovery, quasi-experiments and statistical modelling: PC/FCI, LiNGAM, ordering and bivariate methods, time-series discovery, synthetic control, DiD, RDD, balancing, sensitivity; circuits, VB-GMM, LDS EM, reconciliation, intermittent demand, AFT survival | 17 |
| WP4 | Uncertainty, ensembles and classic-model gaps: recalibration, Venn–Abers, conformal follow-ups and training, cheap and diverse ensembles, stacking, TTA, distillation, multiple-choice learning, binary nets, RBMs, Tsetlin machines, monotonic nets | 17 |
| WP5 | Perception and language gaps: pitch, beats, chords, fingerprints, sound events, word alignment, noisy-channel correction, SimCSE, grammar induction, image anomaly detection, action recognition, matching, edges, hashing, FFT convolution | 17 |
| WP6 | RL beyond the wave 10 recipes: bandits, deep and count-based exploration, skill discovery, MBPO, hierarchy, preference-based and average-reward RL, value factorisation, mean-field and heterogeneous MARL, zero-shot coordination, pixel follow-up | 17 |
| | **Total** | **102** |

---

## WP1. Security and privacy of learning: robust aggregation and poisoning, backdoors, black-box and transfer attacks, AutoAttack evaluation, model extraction, local DP and DP mechanisms, correlated-noise DP training, exact unlearning, corruption robustness (17)

Target: two new packages, `experimental.security` (11.1.1–11.1.11) and `experimental.dpmech` (11.1.12–11.1.15), additions to `experimental.train` (11.1.16) and `experimental.robust` (11.1.17), and a new theme `experiments.security` with `RobustAggregation` (11.1.1–11.1.3), `BackdoorShootout` (11.1.4–11.1.6), `AttackShootout` (11.1.7–11.1.10) and `ModelExtraction` (11.1.11). 11.1.12–11.1.15 add `MechanismShootout` to `experiments.privacy`, 11.1.16 extends `experiments.interp.UnlearningComparison`, and 11.1.17 extends `experiments.robustness.AdversarialRobustness` with a corruption table. One new generator in `experimental.data`: `PoisonedSets`, which wraps `ToyTasks` digits, `ShapeImages` and `ClientSplits` and produces (a) patch, blended and clean-label backdoor triggers at 1–10% poison rates with the clean labels and poison flags kept for scoring, (b) federated splits with 10–40% Byzantine clients under IID and Dirichlet label skew, and (c) categorical "user value" streams over 16–256 items with a known frequency vector for the local-DP items. Every test class trains small MLPs or the `VisionShootout` CNN for a few hundred steps so that it finishes in about 10 s; the defaults are seed 1, with three seeds where a row is cheap.

### 11.1.1 Byzantine-robust aggregation: Krum, coordinate-wise median, trimmed mean and the geometric median (Blanchard et al., NeurIPS 2017; Yin et al., ICML 2018; Pillutla et al., IEEE TSP 2022)
FedAvg averages client updates, so a single malicious client can move the average anywhere. Robust aggregators replace the mean with a statistic that a bounded fraction f of clients cannot control: Krum picks the update with the smallest summed squared distance to its n − f − 2 nearest neighbours (Multi-Krum averages the m best), the coordinate-wise median and trimmed mean act per coordinate, and RFA computes the geometric median by a few Weiszfeld iterations. All of them are known to lose accuracy under heterogeneous (non-IID) clients even with no attacker, which is the regime the next item addresses.
- **Block:** `experimental.security.RobustAggregators` (KRUM, MULTI_KRUM, MEDIAN, TRIMMED_MEAN, GEOMETRIC_MEDIAN), a drop-in for `FederatedAlgorithms`' server step.
- **Experiment:** `experiments.security.RobustAggregation`.
- **Toy task:** `PoisonedSets` federated digits, 20 clients, 0/10/20/40% Byzantine, IID and Dirichlet α ∈ {0.1, 1} label skew; attacks from 11.1.3 (sign flip, Gaussian noise, ALIE).
- **Baselines:** FedAvg, FedProx and FedNova (`FederatedAlgorithms`); the honest-only FedAvg oracle.
- **Expected metric:** test accuracy after 30 rounds per attack and skew, and the clean (no-attack) accuracy cost. Hypothesis: every robust rule keeps accuracy within a few points of the oracle under sign flips in the IID case, but loses 5–15 points to FedAvg at α = 0.1 with no attacker; ALIE beats median and Krum.
- **Source:** [arXiv:1703.02757](https://arxiv.org/abs/1703.02757), [arXiv:1803.01498](https://arxiv.org/abs/1803.01498), [arXiv:1912.13445](https://arxiv.org/abs/1912.13445)

### 11.1.2 Robustness under heterogeneity: bucketing, centered clipping, nearest-neighbour mixing and learnable aggregation weights (Karimireddy et al., ICML 2021 and ICLR 2022; Allouah et al., AISTATS 2023; ICLR 2026)
Pre-aggregation fixes the heterogeneity problem of 11.1.1. Bucketing averages random groups of s clients before the robust rule, so honest updates look more alike; centered clipping clips every update to a radius τ around the previous aggregate and iterates, using server momentum to remember history; nearest-neighbour mixing (NNM) replaces each update by the mean of its n − f nearest neighbours, after which any robust rule meets the optimal heterogeneity bound. The 2025/2026 paper treats the aggregation weights as parameters on the simplex and optimises them jointly with the global model by alternating minimisation, so honest but unusual clients are not discarded.
- **Block:** `experimental.security.PreAggregation` (BUCKETING, CENTERED_CLIP with momentum, NNM) composing with 11.1.1's rules; `experimental.security.LearnedAggregationWeights` (alternating simplex-projected updates of the weights and the model).
- **Experiment:** `RobustAggregation`.
- **Toy task:** as 11.1.1, emphasising α = 0.1 and 20–40% attackers.
- **Baselines:** 11.1.1's rules alone; FedAvg; FedAvg with server momentum (`FederatedAlgorithms` FedAvgM).
- **Expected metric:** accuracy per (skew, attack, attacker share) cell and the worst cell across attacks. Hypothesis: NNM + median and centered clipping recover most of the heterogeneity loss of 11.1.1; learned weights match NNM at 20% attackers and degrade first at 40%.
- **Source:** [arXiv:2012.10333](https://arxiv.org/abs/2012.10333), [arXiv:2006.09365](https://arxiv.org/abs/2006.09365), [arXiv:2302.01772](https://arxiv.org/abs/2302.01772), [arXiv:2511.03529](https://arxiv.org/abs/2511.03529)

### 11.1.3 Model-poisoning attacks: sign flipping, ALIE, inner-product manipulation and model replacement (Baruch et al., NeurIPS 2019; Xie et al., UAI 2020; Bagdasaryan et al., AISTATS 2020)
Strong attacks exploit what robust rules cannot see. "A little is enough" (ALIE) shifts every coordinate by z standard deviations of the honest updates, staying inside the median's acceptance region; inner-product manipulation sends −ε times the honest mean so that the aggregate's inner product with the true gradient turns negative; model replacement scales a backdoored update by n/η so that it survives averaging and plants a trigger. Measuring them is what makes 11.1.1–11.1.2's numbers meaningful.
- **Block:** `experimental.security.ModelPoisoning` (SIGN_FLIP, GAUSSIAN, ALIE, IPM, MODEL_REPLACEMENT with a 11.1.4 trigger), with omniscient and non-omniscient variants.
- **Experiment:** `RobustAggregation`.
- **Toy task:** as 11.1.1.
- **Baselines:** no attack; random Gaussian updates of the same norm.
- **Expected metric:** accuracy drop per aggregator, and backdoor success rate for model replacement. Hypothesis: ALIE and IPM cut median/Krum accuracy by 10–30 points where sign flips do almost nothing; model replacement implants the trigger (> 90% success) under FedAvg and is stopped by centered clipping.
- **Source:** [arXiv:1902.06156](https://arxiv.org/abs/1902.06156), [arXiv:1903.03936](https://arxiv.org/abs/1903.03936), [arXiv:1807.00459](https://arxiv.org/abs/1807.00459)

### 11.1.4 Data-poisoning backdoors: BadNets, blended triggers and clean-label attacks (Gu et al., 2017; Chen et al., 2017; Turner et al., 2019)
A backdoor makes a model output a target class whenever a trigger is present while leaving clean accuracy intact. BadNets stamps a small patch on a few training images and relabels them; the blended attack alpha-blends a full-image pattern; the clean-label attack keeps the correct labels and instead perturbs target-class images adversarially so that the model must rely on the trigger to fit them. The poison rate needed and the clean-accuracy cost differ sharply between the three, and they are the threat model for 11.1.5–11.1.6.
- **Block:** `experimental.security.Backdoors` (trigger stamping and blending, clean-label PGD perturbation via `AdversarialAttacks`), with the poison flags kept.
- **Experiment:** `experiments.security.BackdoorShootout`.
- **Toy task:** `PoisonedSets` digits and `ShapeImages`, poison rates 0.5/1/5/10%, target class 0.
- **Baselines:** clean training; random-label poisoning at the same rate (no trigger).
- **Expected metric:** clean accuracy and attack success rate against poison rate. Hypothesis: BadNets reaches > 90% success at 1% poison with < 1 point of clean-accuracy loss; clean-label needs 5–10x more poison.
- **Source:** [arXiv:1708.06733](https://arxiv.org/abs/1708.06733), [arXiv:1712.05526](https://arxiv.org/abs/1712.05526), [arXiv:1912.02771](https://arxiv.org/abs/1912.02771)

### 11.1.5 Poisoned-sample detection: spectral signatures, activation clustering and AGPD (Tran et al., NeurIPS 2018; Chen et al., 2018; Yuan et al., ICLR 2025)
Poisoned samples leave a trace in the representation of the target class. Spectral signatures score each sample of a class by its squared projection on the top singular vector of the centred penultimate features and remove the top 1.5ε; activation clustering splits each class's features into two clusters after ICA and flags a small, well-separated cluster. AGPD (2025) instead uses the gradient of the loss with respect to activations: it computes a "gradient circular distribution" per class, picks the target class as the one whose distribution is most dispersed, and separates poisoned from clean samples inside it.
- **Block:** `experimental.security.PoisonDetectors` (SPECTRAL, ACTIVATION_CLUSTERING, AGPD), reusing `ThinSvd`, `KMeans` and `ComponentAnalysis`' FastICA.
- **Experiment:** `BackdoorShootout`.
- **Toy task:** 11.1.4's poisoned sets, all three attacks.
- **Baselines:** loss-based filtering (drop the highest-loss 5%); `DataPruner`'s EL2N scores; random removal.
- **Expected metric:** detection AUROC and recall at the removal budget, then attack success and clean accuracy after retraining on the filtered set. Hypothesis: spectral signatures catch BadNets (AUROC > 0.95) but not blended or clean-label poisons; AGPD identifies the target class correctly across all three and keeps attack success below 10% after retraining.
- **Source:** [arXiv:1811.00636](https://arxiv.org/abs/1811.00636), [arXiv:1811.03728](https://arxiv.org/abs/1811.03728), [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bd8284e53b6d177cbede82def77d4951-Abstract-Conference.html)

### 11.1.6 Repairing a backdoored model: Neural Cleanse, fine-pruning and adversarial neuron pruning (Wang et al., IEEE S&P 2019; Liu et al., RAID 2018; Wu & Wang, NeurIPS 2021)
When the data is gone, the model itself must be repaired. Neural Cleanse reverse-engineers, for each class, the smallest mask-and-pattern that sends all inputs to it; an anomalously small mask (median absolute deviation index > 2) marks the target, and unlearning on the recovered trigger removes the backdoor. Fine-pruning prunes the last-layer channels least active on clean data, then fine-tunes; adversarial neuron pruning perturbs neuron weights adversarially and prunes the neurons whose perturbation most raises the loss, since backdoor neurons are the most sensitive.
- **Block:** `experimental.security.BackdoorRepair` (NEURAL_CLEANSE with MAD outlier test and unlearning, FINE_PRUNING, ANP).
- **Experiment:** `BackdoorShootout`.
- **Toy task:** models backdoored by 11.1.4, with 5% clean data for the defender.
- **Baselines:** fine-tuning on the clean 5% alone; wave 5's latent adversarial training (`LatentAdversarialTrainer`) adapted to the classifier.
- **Expected metric:** attack success, clean accuracy and target-class identification rate. Hypothesis: Neural Cleanse identifies the patch trigger's target but misses the blended one; ANP gives the best success/accuracy trade-off; plain fine-tuning leaves > 50% success.
- **Source:** [Neural Cleanse, IEEE S&P 2019](https://doi.org/10.1109/SP.2019.00031), [arXiv:1805.12185](https://arxiv.org/abs/1805.12185), [arXiv:2110.14430](https://arxiv.org/abs/2110.14430)

### 11.1.7 Score-based black-box attacks: NES, SimBA and the Square Attack (Ilyas et al., ICML 2018; Guo et al., ICML 2019; Andriushchenko et al., ECCV 2020)
With only the output probabilities, an attacker must estimate the gradient or search directly. NES estimates it from antithetic Gaussian queries and runs PGD on the estimate; SimBA tries one random orthonormal (pixel or DCT) direction at a time and keeps it if the true-class probability drops; Square Attack samples random square-shaped perturbations at the corners of the ℓ∞ ball with a shrinking square size and needs no gradient estimate at all. Query efficiency at a fixed success rate is the measure.
- **Block:** `experimental.security.BlackBoxAttacks` (NES, SIMBA pixel/DCT, SQUARE ℓ∞/ℓ2), all on a logits oracle with a query counter.
- **Experiment:** `experiments.security.AttackShootout`.
- **Toy task:** the `VisionShootout` CNN on digits and `ShapeImages`, standard and PGD-trained (`AdversarialTraining`), ε = 0.1/0.2 (ℓ∞).
- **Baselines:** white-box PGD (`AdversarialAttacks`) as the upper bound; random uniform noise of the same norm.
- **Expected metric:** success rate and median queries at budgets of 100/1,000 queries. Hypothesis: Square Attack reaches within 5 points of white-box PGD on the standard model at a few hundred queries; on the PGD-trained model all black-box attacks fall further behind.
- **Source:** [arXiv:1804.08598](https://arxiv.org/abs/1804.08598), [arXiv:1905.07121](https://arxiv.org/abs/1905.07121), [arXiv:1912.00049](https://arxiv.org/abs/1912.00049)

### 11.1.8 Decision-based attacks: the boundary attack and HopSkipJump (Brendel et al., ICLR 2018; Chen et al., IEEE S&P 2020)
With only the predicted label, the attack starts from an adversarial point (a different-class image or large noise) and walks along the decision boundary toward the clean input. The boundary attack takes random orthogonal steps followed by contractions toward the original; HopSkipJump estimates the boundary normal from the sign of label changes at random perturbations, takes a geometric step along it, and binary-searches back to the boundary, which makes it far more query-efficient.
- **Block:** `experimental.security.DecisionAttacks` (BOUNDARY, HOP_SKIP_JUMP, ℓ2 and ℓ∞).
- **Experiment:** `AttackShootout`.
- **Toy task:** as 11.1.7, untargeted ℓ2.
- **Baselines:** 11.1.7's score-based attacks (which see more); white-box PGD's minimal-ε search.
- **Expected metric:** median ℓ2 perturbation against query count (100–5,000). Hypothesis: HopSkipJump reaches the white-box perturbation size within 2x at 2,000 queries; the boundary attack needs about 10x more queries for the same distance.
- **Source:** [arXiv:1712.04248](https://arxiv.org/abs/1712.04248), [arXiv:1904.02144](https://arxiv.org/abs/1904.02144)

### 11.1.9 Transfer attacks and universal perturbations: MI-FGSM and UAPs (Dong et al., CVPR 2018; Moosavi-Dezfooli et al., CVPR 2017)
A third threat model uses no queries at all. Transfer attacks craft examples on a surrogate model; adding momentum to iterative FGSM (MI-FGSM) and attacking an ensemble of surrogates stabilises the direction and raises transfer rates. A universal adversarial perturbation is one input-agnostic perturbation, built by aggregating minimal per-image perturbations (DeepFool-style) and projecting onto the ε-ball, that fools the model on most inputs.
- **Block:** `experimental.security.TransferAttacks` (MI-FGSM, ensemble surrogates, UAP).
- **Experiment:** `AttackShootout`.
- **Toy task:** source and target CNNs and MLPs of different widths and seeds on digits and `ShapeImages`.
- **Baselines:** PGD crafted on the surrogate (no momentum); random noise; white-box PGD on the target.
- **Expected metric:** transfer success rate (source → target matrix) and UAP fooling rate on held-out images. Hypothesis: momentum plus a three-model ensemble roughly doubles PGD's transfer rate; a UAP at ε = 0.2 fools > 50% of held-out digits.
- **Source:** [arXiv:1710.06081](https://arxiv.org/abs/1710.06081), [arXiv:1610.08401](https://arxiv.org/abs/1610.08401)

### 11.1.10 Reliable robustness evaluation with AutoAttack: APGD-CE, APGD-DLR and the attack ensemble (Croce & Hein, ICML 2020) — follow-up
Wave 8 reported PGD-AT, TRADES, fast FGSM training and 1-Lipschitz networks against fixed-step PGD only, which is the evaluation that AutoAttack showed overstates robustness. APGD adapts the step size from progress checkpoints and restarts from the best point; the difference-of-logits-ratio (DLR) loss is scale-invariant, so it defeats gradient masking by logit rescaling; AutoAttack runs APGD-CE, targeted APGD-DLR, FAB and Square (11.1.7) in sequence and reports the worst case per example.
- **Block:** `experimental.security.AutoAttack` (APGD with CE and DLR losses, targeted DLR, the ensemble with 11.1.7's Square; FAB optional).
- **Experiment:** `AttackShootout`, plus a re-evaluation row in `experiments.robustness.AdversarialRobustness` (new method; wave 8 rows unchanged).
- **Toy task:** wave 8's adversarially trained models on digits, re-trained at test budget.
- **Baselines:** the wave 8 PGD evaluation (`AdversarialAttacks`, 10 and 50 steps, restarts); randomized smoothing's certified radius as a lower bound.
- **Expected metric:** robust accuracy under PGD-10, PGD-50 and AutoAttack per defence, and the gap. Hypothesis: AutoAttack lowers robust accuracy by 1–5 points for PGD-AT and TRADES and by more for fast FGSM training (catastrophic-overfitting runs); the defence ranking stays the same.
- **Source:** [arXiv:2003.01690](https://arxiv.org/abs/2003.01690)

### 11.1.11 Model extraction and its defences: equation solving, Jacobian augmentation, Knockoff Nets and prediction poisoning (Tramèr et al., USENIX Security 2016; Papernot et al., AsiaCCS 2017; Orekondy et al., CVPR 2019 and ICLR 2020)
An attacker with query access trains a copy of a deployed model. For logistic models the parameters follow by solving equations from d + 1 confidence queries; for networks, Jacobian-based augmentation grows a synthetic query set along the substitute's sign-gradients, and Knockoff Nets query with natural images from another distribution and train on the returned probabilities. Prediction poisoning defends by perturbing the returned probabilities to mislead the attacker's gradients while keeping the argmax.
- **Block:** `experimental.security.ModelExtraction` (EQUATION_SOLVING, JACOBIAN_AUGMENTATION, KNOCKOFF with random and adaptive query selection) and `experimental.security.PredictionPoisoning` (top-1-preserving perturbation and rounding defences).
- **Experiment:** `experiments.security.ModelExtraction`.
- **Toy task:** victim logistic regressions and MLPs on digits; query sets from `ShapeImages`, `ShiftSuites` corruptions and random noise.
- **Baselines:** training the copy on the same number of true-labelled samples (the data-owner upper bound); label-only outputs.
- **Expected metric:** copy accuracy and agreement with the victim against query budget (100–5,000), with and without defences, plus the defence's accuracy cost to honest users. Hypothesis: equation solving recovers logistic models exactly; Knockoff reaches > 90% agreement at a few thousand queries with probabilities but much less with labels only; prediction poisoning halves the attacker's gain at zero top-1 cost.
- **Source:** [arXiv:1609.02943](https://arxiv.org/abs/1609.02943), [arXiv:1602.02697](https://arxiv.org/abs/1602.02697), [arXiv:1812.02766](https://arxiv.org/abs/1812.02766), [arXiv:1906.10908](https://arxiv.org/abs/1906.10908)

### 11.1.12 Local differential privacy for frequency estimation: randomized response, RAPPOR, OUE and OLH (Warner, 1965; Erlingsson et al., CCS 2014; Wang et al., USENIX Security 2017)
In the local model each user randomises their own value before sending it, so the server never sees true data. k-ary randomized response reports the true value with probability e^ε/(e^ε + k − 1); RAPPOR and optimised unary encoding (OUE) one-hot encode and flip each bit with tuned probabilities; optimised local hashing (OLH) hashes into g = e^ε + 1 buckets first. All give unbiased frequency estimates with different variance–communication trade-offs, and post-processing (projection onto the simplex, Norm-Sub) cuts the error further.
- **Block:** `experimental.dpmech.LocalDp` (GRR, RAPPOR, OUE, OLH, with unbiased decoders and simplex post-processing).
- **Experiment:** `experiments.privacy.MechanismShootout`.
- **Toy task:** `PoisonedSets` value streams: 10⁴–10⁵ users, domains of 16–256 items, Zipf and uniform frequencies, ε ∈ {0.5, 1, 2, 4}.
- **Baselines:** central-DP Laplace histograms (the same ε, trusted curator); the non-private histogram.
- **Expected metric:** mean squared error of the frequency vector and top-10 recall. Hypothesis: GRR wins for small domains (k < 3e^ε + 2), OUE/OLH for large ones, matching the theoretical crossover; local DP's error is 10–100x the central Laplace error.
- **Source:** [arXiv:1407.6981](https://arxiv.org/abs/1407.6981), [arXiv:1705.04421](https://arxiv.org/abs/1705.04421)

### 11.1.13 Private selection and the sparse vector technique: the exponential mechanism, report-noisy-max, permute-and-flip and AboveThreshold (McSherry & Talwar, FOCS 2007; McKenna & Sheldon, NeurIPS 2020; Dwork et al., STOC 2009; Lyu et al., VLDB 2017)
Many private algorithms reduce to picking the best of k candidates. The exponential mechanism samples a candidate with probability proportional to exp(εu/2Δ); report-noisy-max adds Laplace or Gumbel noise and takes the argmax; permute-and-flip visits candidates in random order and accepts each with probability exp(ε(u − u*)/2Δ), which never does worse than the exponential mechanism. The sparse vector technique (AboveThreshold) answers a stream of threshold queries while paying privacy only for the positive answers.
- **Block:** `experimental.dpmech.PrivateSelection` (EXPONENTIAL, REPORT_NOISY_MAX Laplace/Gumbel, PERMUTE_AND_FLIP) and `experimental.dpmech.SparseVector` (AboveThreshold with correct noise scales, plus the known-broken variants as negative controls).
- **Experiment:** `MechanismShootout`, plus private hyperparameter selection over a `ToyTasks` grid.
- **Toy task:** selecting the mode of a histogram, the best of 8–64 hyperparameter configurations from validation scores, and private feature selection with AboveThreshold.
- **Baselines:** non-private argmax; uniform random choice; DP-SGD trained with one fixed configuration (`DpSgd`).
- **Expected metric:** expected utility gap against ε and an empirical privacy check (likelihood ratios on neighbouring datasets for the broken SVT variants). Hypothesis: permute-and-flip beats the exponential mechanism by up to 2x in utility gap at small ε; the broken SVT variants show measured privacy loss above the claimed ε.
- **Source:** [Exponential mechanism, FOCS 2007](https://doi.org/10.1109/FOCS.2007.66), [arXiv:2010.12603](https://arxiv.org/abs/2010.12603), [arXiv:1603.01699](https://arxiv.org/abs/1603.01699)

### 11.1.14 Private query release and synthetic tables: MWEM and PrivBayes (Hardt et al., NeurIPS 2012; Zhang et al., SIGMOD 2014 and TODS 2017)
Instead of noising each answer, these mechanisms release a private synthetic dataset. MWEM keeps a distribution over the data domain, repeatedly picks the worst-answered marginal query with the exponential mechanism (11.1.13), measures it with Laplace noise and updates the distribution by multiplicative weights. PrivBayes privately learns a low-degree Bayesian network (choosing parents with the exponential mechanism), adds noise to its conditional tables and samples from it.
- **Block:** `experimental.dpmech.Mwem` and `experimental.dpmech.PrivBayes`, over discretised tables.
- **Experiment:** `MechanismShootout`; downstream utility through wave 9's train-on-synthetic, test-on-real audit (`SyntheticDataAudit`).
- **Toy task:** `ToyTables`/`TabularStressSuite` discretised to 6–10 columns of 2–8 values; all 2-way marginals as the workload.
- **Baselines:** independent Laplace-noised marginals; wave 9's non-private TabDDPM and CTGAN (`TabularDiffusion`, `ConditionalTabularGan`) as utility upper bounds.
- **Expected metric:** average workload error and TSTR accuracy against ε ∈ {0.1, 1, 10}. Hypothesis: MWEM beats independent noise on 2-way marginals for ε ≤ 1; PrivBayes wins on 3-way structure; both stay well below the non-private generators in TSTR.
- **Source:** [arXiv:1012.4763](https://arxiv.org/abs/1012.4763), [PrivBayes, TODS 2017](https://doi.org/10.1145/3134428)

### 11.1.15 Correlated-noise DP training: tree-aggregation DP-FTRL, banded matrix factorisation and BLT (Kairouz et al., ICML 2021; Choquette-Choo et al., NeurIPS 2023; McMahan et al., 2024)
DP-SGD adds independent noise at every step. DP-FTRL instead privatises the prefix sums of gradients with the binary-tree mechanism, so the noise in the model is correlated across steps and grows only logarithmically, without needing privacy amplification by sampling. Banded matrix factorisation optimises the correlation matrix C (with A = BC) for the workload of prefix sums, and the Buffered Linear Toeplitz (BLT) mechanism generates near-optimal correlated noise from d ≈ 2–4 decaying buffers, with memory independent of the number of steps.
- **Block:** `experimental.dpmech.CorrelatedNoise` (TREE, BANDED_MF with a small optimiser for C, BLT with fitted decays θ and scales ω) plugged into a `DpFtrl` trainer with per-example clipping from `DpSgd`.
- **Experiment:** `MechanismShootout` (prefix-sum error) and `experiments.safety.PrivateTraining` (new rows; existing rows unchanged).
- **Toy task:** prefix sums of random unit vectors over 256–2,048 steps; logistic regression and a small MLP on digits under user-level DP with 1–4 participations.
- **Baselines:** DP-SGD with Poisson sampling and the RDP accountant (`DpSgd`); DP-SGD without amplification (the apples-to-apples case).
- **Expected metric:** root-mean-square prefix-sum error, test accuracy at ε ∈ {1, 4, 8}, and noise-state memory. Hypothesis: BLT matches banded MF's error within 5% with about 4x less state and both beat the tree; DP-FTRL beats unamplified DP-SGD and roughly ties amplified DP-SGD at ε = 8.
- **Source:** [arXiv:2103.00039](https://arxiv.org/abs/2103.00039), [arXiv:2306.08153](https://arxiv.org/abs/2306.08153), [arXiv:2408.08868](https://arxiv.org/abs/2408.08868)

### 11.1.16 Exact and approximate unlearning for classifiers: SISA, S3T and SCRUB (Bourtoule et al., IEEE S&P 2021; Chowdhury et al., ICLR 2025; Kurmanji et al., NeurIPS 2023)
Approximate unlearning (wave 5's NPO, SimNPO, RMU) cannot prove that a point is gone. SISA trains one model per disjoint data shard, each in slices with checkpoints, and aggregates their votes; deleting a point retrains only its shard from the last checkpoint before it. S3T (2025) trains parameter-efficient layers one after another, each on its own data slice, so deletion just switches off the affected layers, and trains several orderings to raise deletion capacity. SCRUB is a strong approximate baseline: a teacher–student scheme that maximises divergence from the teacher on the forget set while matching it on the retain set.
- **Block:** `experimental.train.ShardedUnlearning` (SISA with shards × slices, vote aggregation, retrain-on-delete cost accounting; S3T with per-slice LoRA layers from `LoraAdapterDense`) and `experimental.train.Scrub`.
- **Experiment:** `experiments.interp.UnlearningComparison` (new rows; existing rows unchanged) and a deletion-stream test.
- **Toy task:** `UnlearningComparison`'s memorised-exceptions task and digits, with deletion streams of 1–10% of the training set.
- **Baselines:** retraining from scratch (the exact reference); NPO/SimNPO/RMU (`MachineUnlearning`); LiRA membership AUC on deleted points (`LikelihoodRatioAttack`).
- **Expected metric:** test accuracy, forget-set accuracy, LiRA AUC on deleted points, and retraining compute per deletion. Hypothesis: SISA with 4–8 shards costs 2–6 points of accuracy but deletes exactly at about 1/shards of the retraining cost; S3T keeps more accuracy at the same deletion capacity; SCRUB matches retraining's forget accuracy but leaves LiRA AUC above 0.5.
- **Source:** [arXiv:1912.03817](https://arxiv.org/abs/1912.03817), [arXiv:2406.16257](https://arxiv.org/abs/2406.16257), [arXiv:2302.09880](https://arxiv.org/abs/2302.09880)

### 11.1.17 Corruption robustness by augmentation: AugMix with Jensen–Shannon consistency, and TrivialAugment (Hendrycks et al., ICLR 2020; Müller & Hutter, ICCV 2021)
Adversarial training buys worst-case robustness but not robustness to natural corruptions. AugMix mixes several randomly composed augmentation chains with Dirichlet weights, blends the result with the original, and adds a Jensen–Shannon consistency loss between the clean and two AugMix views. TrivialAugment applies one uniformly sampled operation at a uniformly sampled strength per image, with no search and no tuned magnitude, and matches searched policies.
- **Block:** `experimental.robust.AugMix` (chains, mixing, JSD loss) and `experimental.robust.TrivialAugment`, both over core `Augmentation`'s operations (corruptions used at test time are held out of the augmentation set).
- **Experiment:** `experiments.robustness.AdversarialRobustness.corruptions` (new method).
- **Toy task:** digits and `ShapeImages` trained clean, tested on `ShiftSuites`/`DegradedImages` corruptions (noise, blur, contrast, elastic, JPEG-like quantisation) at five severities.
- **Baselines:** standard augmentation; RandAugment (core `Augmentation`); PGD-AT (`AdversarialTraining`); mixup/CutMix (`BatchMixing`).
- **Expected metric:** clean accuracy and mean corruption error (mCE) relative to standard training. Hypothesis: AugMix + JSD gives the lowest mCE at no clean-accuracy cost; TrivialAugment ties RandAugment; PGD-AT helps on noise but hurts on contrast and blur.
- **Source:** [arXiv:1912.02781](https://arxiv.org/abs/1912.02781), [arXiv:2103.10158](https://arxiv.org/abs/2103.10158)

---

## WP2. Learning for combinatorial optimisation and algorithms: neural routing solvers, graph CO, learned branching and SAT search, neural-guided synthesis, sketches, learned data structures, caching with predictions (17)

Target: two new packages, `experimental.combopt` (11.2.1–11.2.11) and `experimental.sketch` (11.2.14–11.2.17), additions to `experimental.evolve` (11.2.12–11.2.13), and two new themes: `experiments.combinatorial` with `RoutingShootout` (11.2.1–11.2.7), `GraphCoShootout` (11.2.8–11.2.9) and `SolverLearning` (11.2.10–11.2.11), and `experiments.algorithms` with `SketchShootout` (11.2.14–11.2.15), `LearnedStructures` (11.2.16) and `CachingWithPredictions` (11.2.17). 11.2.12–11.2.13 add rows to `experiments.synthesis.ProgramSynthesisShootout`. Two new generators in `experimental.data`: `CombinatorialInstances` (uniform and clustered Euclidean TSP with 10–50 nodes, CVRP with 20 customers, Erdős–Rényi and regular graphs for MaxCut and maximum independent set, 0/1 knapsack and set-cover ILPs with 10–30 variables, all with exact or best-known optima from a small exact solver: Held–Karp for n ≤ 13, branch and bound over `LinearProgram` otherwise) and `KeyStreams` (Zipf and drifting-Zipf item streams of 10⁵–10⁶ keys, sorted key sets with uniform, lognormal and piecewise distributions, and page-request traces with loops and scans). The classical heuristics (nearest neighbour, farthest insertion, 2-opt/Or-opt, greedy MIS, LRU/LFU) are written once in `experimental.combopt.ClassicHeuristics` and `experimental.sketch.ClassicCaches` as references next to the repo baselines. Tests train on 10–20-node instances for a few hundred steps so that each class finishes in about 10 s; larger sizes are generalisation probes, not training runs.

### 11.2.1 Pointer networks on routing: supervised and actor-critic training (Vinyals et al., NeurIPS 2015; Bello et al., 2017)
The repo's `PointerNetwork` solves convex hulls with supervised targets. On the TSP, Vinyals et al. trained it on optimal tours, while Bello et al. replaced supervision with REINFORCE and a learned critic baseline, which removes the need for an exact solver and generalises better. This item is the neural reference point for the rest of the WP.
- **Block:** `experimental.combopt.RoutingEnv` (TSP and CVRP state, masking, tour length) and `experimental.combopt.PointerRouting` (wraps `PointerNetwork` with supervised and actor-critic training).
- **Experiment:** `experiments.combinatorial.RoutingShootout`.
- **Toy task:** `CombinatorialInstances` TSP-10/20, tested on 10/20/50.
- **Baselines:** `PointerNetwork` with supervised Held–Karp tours; nearest neighbour, farthest insertion and 2-opt from `ClassicHeuristics`; the exact optimum.
- **Expected metric:** optimality gap (%) on 1,000 test instances and inference time. Hypothesis: actor-critic beats supervised training at TSP-20 and generalises better to TSP-50; both stay behind 2-opt at test budget.
- **Source:** [arXiv:1506.03134](https://arxiv.org/abs/1506.03134), [arXiv:1611.09940](https://arxiv.org/abs/1611.09940)

### 11.2.2 The attention model with a greedy-rollout baseline (Kool et al., ICLR 2019)
Kool et al. encode nodes with a transformer encoder (no positional encoding), decode one node at a time with a context embedding of the first and last node plus a glimpse, and mask visited nodes. REINFORCE uses a greedy rollout of a frozen copy of the policy as baseline, replaced whenever a paired t-test says the current policy is better. It was the first model to approach 2-opt quality on TSP-20/50 and works unchanged on CVRP with a capacity in the context.
- **Block:** `experimental.combopt.AttentionModel` (encoder, context decoder, rollout baseline with t-test replacement), reusing core `TransformerBlock`.
- **Experiment:** `RoutingShootout`.
- **Toy task:** TSP-10/20 and CVRP-20.
- **Baselines:** 11.2.1's actor-critic pointer network; `ClassicHeuristics`; the exact or best-known optimum.
- **Expected metric:** greedy and sampled (128 samples) optimality gaps at an equal number of training instances. Hypothesis: the attention model beats the pointer network by several gap points at TSP-20, and its sampled decoding matches 2-opt.
- **Source:** [arXiv:1803.08475](https://arxiv.org/abs/1803.08475)

### 11.2.3 POMO and symmetric NCO: multiple starts and symmetry as a shared baseline (Kwon et al., NeurIPS 2020; Kim et al., NeurIPS 2022)
POMO starts N rollouts from N different first nodes of the same instance and uses their mean return as a shared REINFORCE baseline, which removes the critic and the rollout policy; at inference it takes the best of the N starts and of 8 coordinate-symmetry augmentations (flips and swaps). Sym-NCO adds a loss that makes the policy invariant to rotations of the instance and the solution symmetric across starts, which helps when instances are rotated or reflected at test time.
- **Block:** `experimental.combopt.Pomo` (multi-start rollouts, shared baseline, ×8 augmentation) and a Sym-NCO invariance loss option, on 11.2.2's `AttentionModel`.
- **Experiment:** `RoutingShootout`.
- **Toy task:** TSP-20 and CVRP-20, plus rotated test sets.
- **Baselines:** 11.2.2 with the rollout baseline at equal training rollouts; 2-opt.
- **Expected metric:** gap with and without augmentation, and training rollouts to reach 11.2.2's final gap. Hypothesis: POMO reaches 11.2.2's gap in about half the rollouts and with ×8 augmentation beats 2-opt; Sym-NCO helps only on rotated test sets.
- **Source:** [arXiv:2010.16011](https://arxiv.org/abs/2010.16011), [arXiv:2205.13209](https://arxiv.org/abs/2205.13209)

### 11.2.4 Preference optimisation for neural combinatorial optimisation (Pan et al., ICML 2025)
REINFORCE's gradient shrinks as sampled tour lengths become similar late in training. Preference optimisation turns the rewards of the sampled solutions for an instance into pairwise preferences, labelled by which tour is shorter, and fits the policy as a Bradley–Terry (or Thurstone) model in which the preference probability depends on the difference of the solutions' log-likelihoods, an entropy-regularised objective in which the partition function cancels. During fine-tuning, local search improves the sampled solutions to give better pairs; it is not used at inference.
- **Block:** `experimental.combopt.PreferenceOptimization` (pairwise BT/Thurstone loss over POMO's N rollouts, optional 2-opt-refined pairs), on 11.2.3.
- **Experiment:** `RoutingShootout`.
- **Toy task:** TSP-20, CVRP-20.
- **Baselines:** POMO with REINFORCE (11.2.3) at equal rollouts; wave 9's DPO-style preference losses (`PreferenceLosses`) as the LM analogue.
- **Expected metric:** gap against training rollouts, especially in the last third of training. Hypothesis: PO reaches POMO's final gap in fewer rollouts; local-search pairs add a further gain on CVRP.
- **Source:** [arXiv:2505.08735](https://arxiv.org/abs/2505.08735)

### 11.2.5 Test-time search for constructive solvers: active search and efficient active search (Bello et al., 2017; Hottung et al., ICLR 2022)
At inference a trained policy can be adapted to the single instance in front of it. Active search fine-tunes all weights by REINFORCE on samples for that instance; efficient active search (EAS) updates only a small subset (the embeddings, a new residual layer, or a lookup table of logit offsets), with an imitation term toward the best solution so far, so instances can be batched.
- **Block:** `experimental.combopt.ActiveSearch` (FULL, EAS_EMB, EAS_LAY, EAS_TAB), on 11.2.2/11.2.3.
- **Experiment:** `RoutingShootout`.
- **Toy task:** TSP-20 and TSP-50 (a size the policy was not trained on); CVRP-20.
- **Baselines:** sampling with the same number of rollouts; POMO ×8 augmentation; 2-opt and Or-opt from `ClassicHeuristics`.
- **Expected metric:** gap against per-instance compute (rollouts). Hypothesis: EAS beats pure sampling at equal rollouts, most clearly at TSP-50; full active search is best per rollout but slowest.
- **Source:** [arXiv:1611.09940](https://arxiv.org/abs/1611.09940), [arXiv:2106.05126](https://arxiv.org/abs/2106.05126)

### 11.2.6 Learned improvement heuristics: learning 2-opt moves (d'O Costa et al., ACML 2020; Wu et al., IEEE TNNLS 2021)
Instead of building a tour, an improvement policy starts from any tour and chooses which 2-opt move to apply next. Costa et al. encode the current tour with a GCN and a recurrent pointer over edges and train with actor-critic on the improvement in tour length; Wu et al. use self-attention over node pairs. A learned move selector can beat first- or best-improvement 2-opt at equal move counts, because it accepts locally worse moves that escape local optima.
- **Block:** `experimental.combopt.LearnedTwoOpt` (tour encoder, pairwise move scorer, actor-critic on improvement with a best-so-far reward).
- **Experiment:** `RoutingShootout`.
- **Toy task:** TSP-20 from random or nearest-neighbour starts, 50–200 moves.
- **Baselines:** first- and best-improvement 2-opt; simulated annealing over 2-opt moves (the annealing code in `SimulatedBifurcation`'s baselines); 11.2.3's constructive POMO.
- **Expected metric:** gap after T moves. Hypothesis: the learned selector beats best-improvement 2-opt at 50–100 moves and ties simulated annealing at 200.
- **Source:** [arXiv:2004.01608](https://arxiv.org/abs/2004.01608), [arXiv:1912.05784](https://arxiv.org/abs/1912.05784)

### 11.2.7 Non-autoregressive heatmaps: supervised edge GCNs and unsupervised soft-permutation GNNs (Joshi et al., 2019; Min et al., NeurIPS 2023; Min & Gomes, 2026)
Non-autoregressive solvers predict all edges at once. Joshi et al. train a residual gated GCN to predict, for every edge, the probability that it is in the optimal tour, and decode with beam search. UTSP and "Graph neural networks are heuristics" (2026) need no labels or rewards: a GNN outputs logits that Gumbel–Sinkhorn turns into a soft permutation matrix, the loss is the expected tour length under it plus a constraint term, and a Hungarian step gives a hard tour; dropout and snapshot ensembles supply several candidate tours at inference.
- **Block:** `experimental.combopt.EdgeHeatmapGcn` (supervised) and `experimental.combopt.SoftPermutationTsp` (unsupervised Gumbel–Sinkhorn loss, Hungarian decoding, MC-dropout candidates), reusing `EntropicTransport`'s Sinkhorn and `MinibatchOptimalTransport`'s Hungarian solver.
- **Experiment:** `RoutingShootout`.
- **Toy task:** TSP-10/20, tested on 50.
- **Baselines:** 11.2.2 and 11.2.3 at equal training time; nearest neighbour (the reference in the 2026 paper) and 2-opt.
- **Expected metric:** gap and inference time. Hypothesis: the unsupervised GNN beats nearest neighbour without any labels but stays behind POMO; the supervised heatmap with beam search is close to POMO at TSP-20 and degrades more at TSP-50.
- **Source:** [arXiv:1906.01227](https://arxiv.org/abs/1906.01227), [arXiv:2303.10538](https://arxiv.org/abs/2303.10538), [arXiv:2601.13465](https://arxiv.org/abs/2601.13465)

### 11.2.8 Unsupervised GNN solvers for graph problems: Erdős goes neural and PI-GNN (Karalias & Loukas, NeurIPS 2020; Schuetz et al., Nature Machine Intelligence 2022)
For graph problems with a QUBO form, a GNN can be trained per instance or across instances on a relaxed objective. PI-GNN outputs node probabilities p and minimises the relaxed Hamiltonian pᵀQp directly, then rounds; Erdős goes neural minimises a probabilistic penalty loss whose small value certifies, via the probabilistic method, that a good integral solution exists, and decodes it by sequential conditional expectation.
- **Block:** `experimental.combopt.QuboGnn` (relaxed Hamiltonian loss, MaxCut and MIS QUBOs, threshold and conditional-expectation decoding), on `experimental.graph`'s GCN layers.
- **Experiment:** `experiments.combinatorial.GraphCoShootout`.
- **Toy task:** `CombinatorialInstances` MaxCut on Erdős–Rényi and 3-regular graphs (n = 50–200) and MIS on Erdős–Rényi graphs.
- **Baselines:** `SimulatedBifurcation` (bSB/aSB/dSB), p-bit annealing (`PBitIsingMachine`) and the spectral cut from wave 7; greedy MIS from `ClassicHeuristics`; exact optima on small graphs.
- **Expected metric:** cut size or MIS size relative to the best known, and wall-clock. Hypothesis: neither GNN beats ballistic SB on MaxCut (as recent critiques found against greedy and annealing baselines); Erdős's decoding beats greedy MIS by a few percent on dense graphs.
- **Source:** [arXiv:2006.10643](https://arxiv.org/abs/2006.10643), [arXiv:2107.01188](https://arxiv.org/abs/2107.01188)

### 11.2.9 GFlowNets for combinatorial optimisation (Zhang et al., NeurIPS 2023)
A GFlowNet samples solutions with probability proportional to exp(−E/T), building them by adding one element at a time to a partial solution (an MDP over sets). Zhang et al. train it with trajectory balance or a forward-looking detailed-balance loss on MIS, max clique and MaxCut, which yields diverse high-quality solutions rather than one, and anneal T to sharpen the distribution.
- **Block:** `experimental.combopt.GFlowNetCo` (set-building environment, MIS/MaxCut energies, forward-looking DB loss) next to the wave 6 `GFlowNet` (trajectory balance, reused).
- **Experiment:** `GraphCoShootout`.
- **Toy task:** as 11.2.8, n = 30–100.
- **Baselines:** 11.2.8's PI-GNN; `SimulatedBifurcation`; greedy; the wave 6 `GFlowNet` with plain trajectory balance.
- **Expected metric:** best-of-32 solution quality, the number of distinct near-optimal solutions, and samples to reach 95% of the best known. Hypothesis: the forward-looking loss beats plain TB; best-of-32 matches simulated annealing on MIS but not bSB on MaxCut.
- **Source:** [arXiv:2305.17010](https://arxiv.org/abs/2305.17010)

### 11.2.10 Learning to branch: GNN imitation of strong branching (Gasse et al., NeurIPS 2019)
Branch and bound for integer programs spends most of its time choosing the variable to branch on. Strong branching (solving both child LPs for every candidate) picks well but is expensive; Gasse et al. represent the LP at a node as a bipartite variable–constraint graph and train a GNN by imitation to predict strong branching's choice, giving smaller trees than cheap rules at much lower cost.
- **Block:** `experimental.combopt.BranchAndBound` (depth-first/best-bound search over `LinearProgram` relaxations, with MOST_FRACTIONAL, PSEUDOCOST, STRONG and LEARNED rules) and `experimental.combopt.BipartiteBranchingGnn`.
- **Experiment:** `experiments.combinatorial.SolverLearning`.
- **Toy task:** `CombinatorialInstances` knapsack and set-cover ILPs with 15–30 variables, tested on larger instances.
- **Baselines:** most-fractional and pseudocost branching; full strong branching.
- **Expected metric:** B&B nodes and wall-clock to optimality. Hypothesis: the learned rule gets within 20% of strong branching's tree size at close to pseudocost speed, and beats most-fractional by 2x or more in nodes.
- **Source:** [arXiv:1906.01629](https://arxiv.org/abs/1906.01629)

### 11.2.11 Learned local search for SAT: a GNN that picks WalkSAT's flips (Yolcu & Póczos, NeurIPS 2019)
WalkSAT picks an unsatisfied clause and flips one of its variables, greedily or at random. Yolcu & Póczos keep that loop but let a GNN over the clause–variable graph choose which variable in the clause to flip, trained by REINFORCE with a curriculum on the number of flips to a solution. The learned heuristic needs fewer flips than WalkSAT on the distribution it was trained on.
- **Block:** `experimental.combopt.LearnedWalkSat` (factor-graph GNN flip policy, REINFORCE with flip-count reward, curriculum).
- **Experiment:** `SolverLearning`.
- **Toy task:** `RandomSatFormulas` random 3-SAT near the threshold (n = 20–50 variables) and the wave 7 NeuroSAT distribution.
- **Baselines:** WalkSAT (in `GraphReasoning`); `NeuroSat` with its decoding; DPLL from `RandomSatFormulas`.
- **Expected metric:** median flips to solve and the share solved within a flip budget, in and out of training size. Hypothesis: the learned policy needs about half of WalkSAT's median flips at the training size and loses its edge at 2x the size.
- **Source:** [arXiv:1909.11830](https://arxiv.org/abs/1909.11830)

### 11.2.12 Neural-guided bottom-up synthesis: BUSTLE and CrossBeam (Odena et al., ICLR 2021; Shi et al., ICLR 2022)
Bottom-up enumeration grows a pool of values by applying DSL operations to existing values, with observational equivalence pruning duplicates. BUSTLE reweights the pool: a model predicts whether each intermediate value is useful for the target examples (from property signatures of the value and the I/O examples) and lowers the weight cost of promising values. CrossBeam goes further and learns, with a pointer network trained on its own search, which existing values to combine next, so the search is a learned policy over the pool.
- **Block:** `experimental.evolve.BottomUpSynthesis` (weighted bottom-up enumeration with observational equivalence), `experimental.evolve.Bustle` (property-signature usefulness model) and `experimental.evolve.CrossBeam` (argument-selection pointer policy trained on-policy).
- **Experiment:** `experiments.synthesis.ProgramSynthesisShootout` (new `Method` rows; existing rows unchanged).
- **Toy task:** `ListPrograms` tasks of lengths 2–5, five I/O examples, three held out.
- **Baselines:** `GuidedEnumerator` (breadth-first with observational equivalence), DeepCoder sort-and-add and ordered, `LibraryLearner`, RobustFill (all existing rows).
- **Expected metric:** solve rate and candidates tried at budgets of 10³–10⁵ programs. Hypothesis: BUSTLE beats plain bottom-up at equal candidates; CrossBeam solves the most length-5 tasks; both beat DeepCoder's ordered search.
- **Source:** [arXiv:2007.14381](https://arxiv.org/abs/2007.14381), [arXiv:2203.10452](https://arxiv.org/abs/2203.10452)

### 11.2.13 Probe: just-in-time learning of a probabilistic grammar during search (Barke, Peleg & Polikarpova, OOPSLA 2020)
Probe needs no pretraining. It runs a cost-guided bottom-up search; whenever a program satisfies a subset of the examples that no earlier program satisfied, it raises the probability of the productions used in that partial solution, recomputes the costs and restarts. The grammar is learned online, per task, from partial successes.
- **Block:** `experimental.evolve.Probe` (probabilistic grammar costs, partial-solution detection, cost updates and restarts), on 11.2.12's `BottomUpSynthesis`.
- **Experiment:** `ProgramSynthesisShootout`.
- **Toy task:** as 11.2.12.
- **Baselines:** 11.2.12's unguided bottom-up and BUSTLE; `GuidedEnumerator`; DeepCoder (pretrained guidance).
- **Expected metric:** solve rate against candidates tried, with no pretraining budget for Probe. Hypothesis: Probe beats unguided search on length-4–5 tasks and approaches DeepCoder without any training data.
- **Source:** [Probe, OOPSLA 2020](https://doi.org/10.1145/3428295)

### 11.2.14 Streaming sketches: Count-Min with conservative update, Count-Sketch, Misra–Gries/Space-Saving and HyperLogLog (Cormode & Muthukrishnan, 2005; Charikar et al., 2002; Metwally et al., 2005; Flajolet et al., 2007)
Sketches summarise a stream in sublinear memory. Count-Min keeps d rows of w counters and answers a frequency query by the minimum over its hashed counters (conservative update increments only the minima); Count-Sketch uses signed counters and the median, giving unbiased estimates; Misra–Gries and Space-Saving keep k counters for candidate heavy hitters; HyperLogLog estimates distinct counts from the maximum leading-zero run per bucket. They are the reference for 11.2.15 and the online-learning code's feature budgets.
- **Block:** `experimental.sketch.FrequencySketches` (COUNT_MIN, CONSERVATIVE_UPDATE, COUNT_SKETCH, MISRA_GRIES, SPACE_SAVING) and `experimental.sketch.HyperLogLog`.
- **Experiment:** `experiments.algorithms.SketchShootout`.
- **Toy task:** `KeyStreams` Zipf (s = 0.8–1.5) and drifting-Zipf streams of 10⁵–10⁶ items.
- **Baselines:** exact hash-map counts; the hashing trick of `FeatureHasher` at equal memory; uniform sampling and `ReservoirReplay`.
- **Expected metric:** average and weighted absolute error, heavy-hitter precision/recall at a fixed memory, and HLL relative error. Hypothesis: conservative update cuts Count-Min's error by 2–5x on skewed streams; Space-Saving finds the heavy hitters with the least memory; HLL stays within its 1.04/√m bound.
- **Source:** [Count-Min, J. Algorithms 2005](https://doi.org/10.1016/j.jalgor.2003.12.001), [Count-Sketch, ICALP 2002](https://doi.org/10.1007/3-540-45465-9_59), [HyperLogLog, AofA 2007](https://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf)

### 11.2.15 Learning-augmented sketches: learned Count-Min, learned Misra–Gries and learned Frequent Directions (Hsu et al., ICLR 2019; Aamand et al., ICLR 2025)
If a model can predict which items are heavy, the sketch can store those exactly and summarise only the rest. Hsu et al. route predicted heavy hitters to unique buckets and the rest to Count-Min; Aamand et al. (2025) give a deterministic learned Misra–Gries that reserves memory for predicted heavy items and keeps the classical guarantee when predictions are wrong, and extend the idea to matrix streams with a learned Frequent Directions that reserves rows for predicted top singular directions.
- **Block:** `experimental.sketch.LearnedSketches` (an oracle interface, LEARNED_COUNT_MIN, LEARNED_MISRA_GRIES) and `experimental.sketch.FrequentDirections` (plain and learned), with a small heavy-hitter classifier on item features and a prediction-noise knob.
- **Experiment:** `SketchShootout`.
- **Toy task:** `KeyStreams` with item features correlated with frequency (from `SyntheticRecsys` item embeddings); matrix streams of `SyntheticRecsys` user rows for Frequent Directions.
- **Baselines:** 11.2.14's sketches at equal memory; for matrices, `ThinSvd` on the full matrix (the oracle), random row sampling and `HebbianRule`'s Oja rule (streaming top-1).
- **Expected metric:** weighted frequency error against memory and against oracle noise; covariance error ‖AᵀA − BᵀB‖ for matrix sketches. Hypothesis: learned sketches halve the error with a good oracle and degrade to the classical bound, not below it, with a random one; learned FD beats plain FD when the top directions are stable across time.
- **Source:** [ICLR 2019](https://openreview.net/forum?id=r1lohoCqY7), [arXiv:2503.00937](https://arxiv.org/abs/2503.00937)

### 11.2.16 Learned index structures and learned Bloom filters: RMI, PGM-style piecewise models and sandwiched filters (Kraska et al., SIGMOD 2018; Ferragina & Vinciguerra, VLDB 2020; Mitzenmacher, NeurIPS 2018)
An index maps a key to its position in a sorted array, which is the key distribution's CDF times n. The recursive model index (RMI) learns that CDF with a two-stage hierarchy of small models and corrects with a bounded local search; the PGM index fits error-bounded piecewise-linear segments in one pass, with a worst-case guarantee. A learned Bloom filter puts a classifier in front of a small backup Bloom filter for the classifier's false negatives; the sandwiched version adds a filter before the classifier too.
- **Block:** `experimental.sketch.LearnedIndex` (RMI with linear/cubic leaves, error-bounded PLA segments, search-cost accounting) and `experimental.sketch.LearnedBloomFilter` (plain, learned and sandwiched).
- **Experiment:** `experiments.algorithms.LearnedStructures`.
- **Toy task:** `KeyStreams` key sets (uniform, lognormal, piecewise; 10⁵–10⁶ keys); URL-like string keys from `SyntheticCorpus` with a planted "malicious" class for the filters.
- **Baselines:** binary search over the sorted array and a B-tree-like fixed-fan-out index (written next to it); standard Bloom filters at equal bits.
- **Expected metric:** lookup time and index bytes, and the false-positive rate at equal memory. Hypothesis: the PLA index is smaller and faster than binary search on smooth distributions and loses its edge on the piecewise one; the sandwiched filter cuts false positives by 2–5x at equal memory when the classifier is good.
- **Source:** [arXiv:1712.01208](https://arxiv.org/abs/1712.01208), [arXiv:1910.06169](https://arxiv.org/abs/1910.06169), [arXiv:1803.01474](https://arxiv.org/abs/1803.01474)

### 11.2.17 Algorithms with predictions: predictive marking and learned eviction for caching, and ski rental (Lykouris & Vassilvitskii, ICML 2018; Purohit et al., NeurIPS 2018; Liu et al., ICML 2020)
Online algorithms can use a predictor while keeping worst-case guarantees. Predictive marking evicts, among unmarked pages, the one predicted to be requested furthest in the future, and falls back to random marking when predictions err, so it is O(1)-competitive with good predictions and O(log k) otherwise. Ski rental with a predicted horizon interpolates between trusting and ignoring the prediction with a robustness parameter λ. Parrot learns eviction by imitating Belady's optimal offline policy from access histories.
- **Block:** `experimental.sketch.CachingWithPredictions` (LRU, LFU, MARKING, PREDICTIVE_MARKING, BELADY offline oracle, a learned next-arrival predictor and a Parrot-style imitation policy) and `experimental.sketch.SkiRental` (deterministic and randomised λ-robust rules).
- **Experiment:** `experiments.algorithms.CachingWithPredictions`.
- **Toy task:** `KeyStreams` page traces (Zipf, loops longer than the cache, scans, phase changes); ski-rental horizons from a lognormal with predictions of controllable noise.
- **Baselines:** LRU/LFU and Belady from `ClassicCaches`; the core `KvCache` eviction policies (H2O-style heavy-hitter retention) on the same traces.
- **Expected metric:** hit rate relative to Belady against prediction noise; the empirical competitive ratio for ski rental against λ. Hypothesis: predictive marking approaches Belady with good predictions and never falls far below LRU; the imitation policy beats LRU on looping traces where LRU fails.
- **Source:** [arXiv:1802.05399](https://arxiv.org/abs/1802.05399), [NeurIPS 2018](https://papers.nips.cc/paper/8174-improving-online-algorithms-via-ml-predictions), [arXiv:2006.16239](https://arxiv.org/abs/2006.16239)

---

## WP3. Causal discovery, quasi-experiments and statistical modelling: PC/FCI, LiNGAM, ordering and bivariate methods, time-series discovery, synthetic control, DiD, RDD, balancing, sensitivity; circuits, VB-GMM, LDS EM, reconciliation, intermittent demand, AFT survival (17)

Target: three new packages, `experimental.discovery` (11.3.1–11.3.5), `experimental.panel` (11.3.6–11.3.10) and `experimental.circuits` (11.3.11–11.3.12), plus additions to `experimental.ml` (11.3.13–11.3.14), `experimental.timeseries` (11.3.15–11.3.16) and `experimental.survival` (11.3.17). Two new themes: `experiments.discovery` with `DiscoveryShootout` (11.3.1–11.3.4) and `TimeSeriesDiscovery` (11.3.5), and `experiments.econometrics` with `PanelShootout` (11.3.6–11.3.7), `DiscontinuityDesigns` (11.3.8) and `ObservationalAdjustment` (11.3.9–11.3.10). 11.3.11–11.3.13 add `CircuitShootout` to `experiments.probabilistic`; 11.3.14–11.3.16 extend `experiments.timeseries` (`StateSpaceEm`, `HierarchicalForecasting`, `IntermittentDemand`); 11.3.17 extends `experiments.survival.SurvivalShootout`. Two new generators in `experimental.data`: `DiscoveryBenchmarks` (random DAGs with 5–20 nodes and expected degree 1–3; linear-Gaussian, linear non-Gaussian (uniform, Laplace) and nonlinear additive-noise SEMs; hidden confounders and selection for FCI; bivariate cause–effect pairs; lagged VAR and nonlinear time-series graphs with contemporaneous links), and `PanelStudies` (units × periods with two-way fixed effects plus low-rank factors, single and staggered adoption, heterogeneous and dynamic effects, a running variable with a cutoff and optional manipulation, and observational studies with known confounding strength). Every estimator has the true graph or effect available for scoring. Each test class finishes in about 10 s; most of these items are linear algebra and need no gradient training.

### 11.3.1 Constraint-based discovery with orientation rules: PC and FCI (Spirtes, Glymour & Scheines, 2000; Zhang, AIJ 2008)
Wave 6 compared NOTEARS with the PC skeleton only. The full PC algorithm removes edges by conditional-independence tests of growing conditioning sets, orients v-structures from the separating sets, then applies Meek's rules to get a CPDAG. FCI drops the causal-sufficiency assumption: it adds possible-d-separation tests and orientation rules that output a partial ancestral graph (PAG) whose circle marks express uncertainty caused by hidden confounders. Both need only a CI test (Fisher-z for Gaussian data, HSIC or a kernel CI test otherwise).
- **Block:** `experimental.discovery.PcAlgorithm` (stable PC, Meek rules, CPDAG output) and `experimental.discovery.Fci` (PAG output), with Fisher-z and `KernelTests` HSIC-based CI tests.
- **Experiment:** `experiments.discovery.DiscoveryShootout`.
- **Toy task:** `DiscoveryBenchmarks` linear-Gaussian and nonlinear SEMs with n = 500–2,000 samples, with and without hidden confounders.
- **Baselines:** the PC skeleton in `CausalAndFairness`; `Notears`, `Dagma` and `GreedyEquivalenceSearch` (GES, wave 7); an empty graph.
- **Expected metric:** structural Hamming distance against the true CPDAG/PAG, orientation precision/recall, and runtime. Hypothesis: PC with orientation beats NOTEARS on SHD at d = 10–20 for Gaussian data (NOTEARS exploits varsortability, which standardised data removes); with hidden confounders only FCI keeps orientation precision high.
- **Source:** [Causation, Prediction, and Search, MIT Press 2000](https://mitpress.mit.edu/9780262527927/causation-prediction-and-search/), [Zhang, AIJ 2008](https://doi.org/10.1016/j.artint.2008.08.001)

### 11.3.2 Linear non-Gaussian models: ICA-LiNGAM and DirectLiNGAM (Shimizu et al., JMLR 2006 and 2011)
With non-Gaussian noise, a linear SEM is identifiable beyond its Markov equivalence class. ICA-LiNGAM runs ICA on the data, permutes and scales the unmixing matrix so that its diagonal has no zeros, and reads the causal order from its lower-triangular form; DirectLiNGAM finds the order directly by repeatedly picking the variable most independent of its regression residuals (a kernel or entropy-based measure), regressing it out, and finally estimating the weights by adaptive lasso.
- **Block:** `experimental.discovery.Lingam` (ICA via `ComponentAnalysis`' FastICA, DIRECT with a pairwise likelihood-ratio measure, adaptive-lasso pruning).
- **Experiment:** `DiscoveryShootout`.
- **Toy task:** `DiscoveryBenchmarks` linear SEMs with uniform and Laplace noise, standardised and raw, d = 5–20.
- **Baselines:** `Notears`, `Dagma`; 11.3.1's PC (which can only return a CPDAG).
- **Expected metric:** SHD against the true DAG and the share of correctly ordered pairs. Hypothesis: DirectLiNGAM recovers the full DAG on non-Gaussian data at n ≥ 1,000 and beats NOTEARS on standardised data; on Gaussian data it falls to chance orientation, as theory predicts.
- **Source:** [JMLR 2006](https://www.jmlr.org/papers/v7/shimizu06a.html), [arXiv:1101.2489](https://arxiv.org/abs/1101.2489)

### 11.3.3 Ordering-based discovery: score-matching leaf removal (SCORE) and TriOpt (Rolland et al., ICML 2022; 2026 preprint)
For nonlinear additive-noise models with Gaussian noise, a variable is a leaf exactly when the corresponding diagonal entry of the Jacobian of the score ∇log p(x) has zero variance. SCORE estimates the score's Jacobian with a kernel (Stein) estimator, removes the leaf with the smallest variance, repeats to get a topological order, and prunes edges with a CAM-style regression. TriOpt (2026) applies the same leaf-removal idea in the linear case with an additive kernel and Sherman–Morrison downdates, then fits the weights by a convex, L1-penalised regression restricted by a strictly upper-triangular mask, so no acyclicity constraint is needed.
- **Block:** `experimental.discovery.ScoreOrdering` (Stein Jacobian estimator, leaf removal, pruning) and `experimental.discovery.TriOpt` (linear-kernel ordering with rank-1 downdates, masked lasso).
- **Experiment:** `DiscoveryShootout`.
- **Toy task:** `DiscoveryBenchmarks` nonlinear ANMs (SCORE) and linear SEMs (TriOpt), d = 5–30.
- **Baselines:** `Notears`, `Dagma`; 11.3.1's PC; 11.3.2's DirectLiNGAM (for the linear case).
- **Expected metric:** order divergence (edges pointing against the estimated order), SHD and runtime against d. Hypothesis: SCORE beats NOTEARS and DAGMA on nonlinear ANMs; TriOpt matches DAGMA's SHD on linear data at a fraction of its runtime for d = 30.
- **Source:** [arXiv:2203.04413](https://arxiv.org/abs/2203.04413), [arXiv:2605.17465](https://arxiv.org/abs/2605.17465)

### 11.3.4 Bivariate cause–effect inference: additive noise models, IGCI and RECI (Hoyer et al., NeurIPS 2008; Janzing et al., AIJ 2012; Blöbaum et al., AISTATS 2018)
With only two variables, conditional independence says nothing about direction. An additive noise model regresses Y on X and X on Y and prefers the direction whose residuals are independent of the input (by HSIC). IGCI assumes that the cause's density and the mechanism are independent and compares slope-based entropies for deterministic relations; RECI compares the regression errors of the two directions after rescaling, preferring the direction with the smaller error.
- **Block:** `experimental.discovery.CauseEffect` (ANM with GP or kernel-ridge regression and HSIC, IGCI slope and entropy estimators, RECI).
- **Experiment:** `DiscoveryShootout`.
- **Toy task:** `DiscoveryBenchmarks` bivariate pairs: nonlinear ANMs, multiplicative noise, nearly deterministic mechanisms, and linear-Gaussian controls (unidentifiable).
- **Baselines:** `KernelTests` HSIC used naively (direction of larger dependence); random guessing; LiNGAM (11.3.2) on the pair.
- **Expected metric:** direction accuracy and the area under the accuracy–decision-rate curve. Hypothesis: ANM is best on additive-noise pairs, IGCI on near-deterministic ones; every method sits at chance on the linear-Gaussian controls.
- **Source:** [NeurIPS 2008](https://papers.nips.cc/paper/3548-nonlinear-causal-discovery-with-additive-noise-models), [Janzing et al., AIJ 2012](https://doi.org/10.1016/j.artint.2012.01.002), [AISTATS 2018](https://proceedings.mlr.press/v84/bloebaum18a.html)

### 11.3.5 Causal discovery in time series: Granger tests, neural Granger causality and PCMCI (Granger, 1969; Tank et al., TPAMI 2021; Runge et al., Science Advances 2019)
In time series, the arrow of time orients lagged edges. Granger tests ask whether the past of X improves the prediction of Y in a VAR. Neural Granger causality fits one MLP or LSTM per target with a group-lasso penalty on the input weights of each source series, so a whole series drops out when it is not causal. PCMCI first prunes each variable's lagged parents with a PC-style condition selection, then runs momentary conditional-independence tests that control for both sides' parents, which keeps false positives low with strong autocorrelation.
- **Block:** `experimental.discovery.GrangerCausality` (VAR F-tests with lag selection), `experimental.discovery.NeuralGranger` (component-wise MLP with hierarchical group lasso) and `experimental.discovery.Pcmci` (partial-correlation and kernel CI tests).
- **Experiment:** `experiments.discovery.TimeSeriesDiscovery`.
- **Toy task:** `DiscoveryBenchmarks` lagged VAR(2) systems with 5–10 series, nonlinear lagged systems, strongly autocorrelated series, and Lorenz-96 from `ChaoticSeries`.
- **Baselines:** pairwise correlation of lagged series; `Notears` on stacked lags; `Sindy` (sparse regression of dynamics) read as a graph.
- **Expected metric:** lagged-edge AUROC and false-positive rate at α = 0.05. Hypothesis: PCMCI keeps the false-positive rate near α where pairwise Granger tests inflate it under autocorrelation; neural Granger wins on the nonlinear systems.
- **Source:** [Granger, Econometrica 1969](https://doi.org/10.2307/1912791), [arXiv:1802.05842](https://arxiv.org/abs/1802.05842), [arXiv:1702.07007](https://arxiv.org/abs/1702.07007)

### 11.3.6 Synthetic control, synthetic difference-in-differences and triply robust panel estimators (Abadie et al., JASA 2010; Arkhangelsky et al., AER 2021; 2025)
With one treated unit and a donor pool, synthetic control builds a convex combination of donors that tracks the treated unit before treatment and uses it as the counterfactual afterwards. Synthetic DiD adds time weights and unit fixed effects, so it is robust when either units or periods are imbalanced. TROP (2025) combines a two-way fixed-effects model with a nuclear-norm-penalised low-rank factor term, weighted by exponentially decaying unit weights (by pre-period distance) and time weights, with tuning by leave-one-out on controls; its bias is a product of unit imbalance, time imbalance and outcome-model error, so it is unbiased if any one vanishes.
- **Block:** `experimental.panel.SyntheticControl` (simplex-constrained weights by projected gradient or `LinearProgram`), `experimental.panel.SyntheticDid` and `experimental.panel.TriplyRobustPanel` (weighted nuclear-norm regression with soft-thresholded SVD from `ThinSvd`), with placebo-in-space inference.
- **Experiment:** `experiments.econometrics.PanelShootout`.
- **Toy task:** `PanelStudies` with 20–50 units and 30 periods, low-rank factors of rank 1–3, single treated unit or block of units, with and without pre-period fit.
- **Baselines:** two-way fixed-effects DiD (written as the reference); matrix-completion imputation (soft-impute with `ThinSvd`); wave 9's CUPED-adjusted difference (`OnlineExperiments`).
- **Expected metric:** RMSE of the estimated effect over 200 simulated panels and placebo-test coverage. Hypothesis: DiD is badly biased when factor loadings differ; SDID beats SC when pre-period fit is poor; TROP has the lowest RMSE in most designs, as its paper reports.
- **Source:** [Abadie et al., JASA 2010](https://doi.org/10.1198/jasa.2009.ap08746), [arXiv:1812.09970](https://arxiv.org/abs/1812.09970), [arXiv:2508.21536](https://arxiv.org/abs/2508.21536)

### 11.3.7 Staggered adoption: the two-way fixed-effects pitfall, Callaway–Sant'Anna and interaction-weighted event studies (Goodman-Bacon, J. Econometrics 2021; Callaway & Sant'Anna, J. Econometrics 2021; Sun & Abraham, J. Econometrics 2021)
When units adopt treatment at different times and effects change over time, the TWFE regression is a weighted average of all 2×2 comparisons, including "forbidden" ones that use already-treated units as controls, and some weights can be negative. Callaway and Sant'Anna estimate group-time effects ATT(g, t) against never-treated or not-yet-treated units and aggregate them; Sun and Abraham estimate cohort-specific event-study coefficients and average them with cohort shares.
- **Block:** `experimental.panel.StaggeredDid` (Goodman-Bacon decomposition, CS group-time ATTs with outcome-regression and IPW versions, SA interaction-weighted event study, bootstrap bands).
- **Experiment:** `PanelShootout`.
- **Toy task:** `PanelStudies` staggered adoption with 3–5 cohorts, constant, growing and heterogeneous effects, with and without anticipation.
- **Baselines:** TWFE DiD and TWFE event studies; 11.3.6's SDID applied per cohort.
- **Expected metric:** bias and coverage of the overall ATT and of the event-time profile. Hypothesis: TWFE can have the wrong sign with growing effects, while CS and SA stay unbiased; with constant effects all agree.
- **Source:** [Goodman-Bacon 2021](https://doi.org/10.1016/j.jeconom.2021.03.014), [arXiv:1803.09015](https://arxiv.org/abs/1803.09015), [arXiv:1804.05785](https://arxiv.org/abs/1804.05785)

### 11.3.8 Regression discontinuity: local linear estimation, MSE-optimal bandwidths, robust bias-corrected intervals and the density test (Imbens & Kalyanaraman, REStud 2012; Calonico, Cattaneo & Titiunik, Econometrica 2014; McCrary, J. Econometrics 2008)
When treatment switches at a cutoff of a running variable, the jump in the outcome at the cutoff identifies a local effect. Local linear regression with a triangular kernel on each side estimates the jump; the bandwidth that minimises MSE balances bias and variance, but at that bandwidth naive confidence intervals under-cover, which robust bias correction fixes by subtracting an estimated bias and widening the interval accordingly. McCrary's test checks for manipulation as a discontinuity in the running variable's density.
- **Block:** `experimental.panel.RegressionDiscontinuity` (local polynomial fits, IK and CCT bandwidths, robust bias-corrected CIs, fuzzy RD by local Wald ratio, McCrary density test).
- **Experiment:** `experiments.econometrics.DiscontinuityDesigns`.
- **Toy task:** `PanelStudies` running-variable designs with curved conditional means, sharp and fuzzy cutoffs, and planted manipulation.
- **Baselines:** global polynomial regressions of order 1–4 (the practice Gelman and Imbens warn against); difference of means in a fixed window.
- **Expected metric:** bias, RMSE and 95% CI coverage over 500 simulations, and the density test's power. Hypothesis: conventional intervals at the MSE-optimal bandwidth cover about 80–85%, robust bias-corrected ones about 95%; high-order global polynomials have the worst RMSE.
- **Source:** [Imbens & Kalyanaraman 2012](https://doi.org/10.1093/restud/rdr043), [CCT 2014](https://doi.org/10.3982/ECTA11757), [McCrary 2008](https://doi.org/10.1016/j.jeconom.2007.05.005)

### 11.3.9 Covariate balancing weights: entropy balancing, CBPS and stable balancing weights (Hainmueller, Political Analysis 2012; Imai & Ratkovic, JRSS-B 2014; Zubizarreta, JASA 2015)
Inverse propensity weights can be extreme and need not balance covariates. Entropy balancing finds the control weights closest in KL to uniform that exactly match the treated group's covariate moments (a convex dual solved by Newton); CBPS fits the propensity model so that its weights balance covariates as well as predict treatment; stable balancing weights minimise weight variance subject to approximate balance tolerances.
- **Block:** `experimental.panel.BalancingWeights` (ENTROPY_BALANCING, CBPS, STABLE_WEIGHTS via a QP with Frank–Wolfe or `LinearProgram`).
- **Experiment:** `experiments.econometrics.ObservationalAdjustment`.
- **Toy task:** `PanelStudies` observational studies with misspecified propensity (Kang–Schafer style) and limited overlap; `CausalScenarios` confounded data.
- **Baselines:** IPW and AIPW from `CausalEstimators`; `DoubleMl`; 1:1 nearest-neighbour matching (written as the reference).
- **Expected metric:** bias and RMSE of the ATT, standardised mean differences, and effective sample size. Hypothesis: entropy balancing and stable weights beat IPW under propensity misspecification, and AIPW stays competitive only when the outcome model is right.
- **Source:** [Hainmueller 2012](https://doi.org/10.1093/pan/mpr025), [Imai & Ratkovic 2014](https://doi.org/10.1111/rssb.12027), [Zubizarreta 2015](https://doi.org/10.1080/01621459.2015.1023805)

### 11.3.10 Sensitivity analysis for unobserved confounding: E-values, robustness values and Rosenbaum bounds (VanderWeele & Ding, Annals of Internal Medicine 2017; Cinelli & Hazlett, JRSS-B 2020; Rosenbaum, 2002)
Every observational estimate assumes no hidden confounder; sensitivity analysis asks how strong one would need to be to explain the result away. The E-value is the minimum risk-ratio association a confounder must have with treatment and outcome; Cinelli and Hazlett express omitted-variable bias through partial R² values and report the robustness value at which the estimate would vanish, benchmarked against observed covariates; Rosenbaum bounds give the largest odds ratio Γ of hidden bias under which a matched-pair test stays significant.
- **Block:** `experimental.panel.SensitivityAnalysis` (E-values, partial-R² bias bounds and robustness values with covariate benchmarking, Rosenbaum bounds for matched pairs).
- **Experiment:** `ObservationalAdjustment`.
- **Toy task:** `PanelStudies` observational studies with a planted hidden confounder of known strength, plus a "null" study where the effect is entirely confounding.
- **Baselines:** the naive adjusted estimate with its confidence interval; 11.3.9's weighted estimates.
- **Expected metric:** whether each method flags the confounded study, and how its sensitivity parameter tracks the planted confounder strength. Hypothesis: robustness values benchmarked against the strongest observed covariate correctly separate the robust and the confounded studies, where the confidence interval alone does not.
- **Source:** [VanderWeele & Ding 2017](https://doi.org/10.7326/M16-2607), [Cinelli & Hazlett 2020](https://doi.org/10.1111/rssb.12348), [Rosenbaum, Observational Studies, 2002](https://doi.org/10.1007/978-1-4757-3692-2)

### 11.3.11 Probabilistic circuits: LearnSPN, randomised tensorised circuits and EM (Gens & Domingos, ICML 2013; Peharz et al., UAI 2019 and ICML 2020)
Sum-product networks compute a density with sum and product nodes, and for smooth and decomposable circuits every marginal and conditional is exact in one pass. LearnSPN learns the structure top-down by clustering rows (sum nodes) and splitting independent column groups (product nodes) by a G-test or RDC; RAT-SPNs and Einsum networks fix a random region graph, vectorise the layers and fit the parameters by EM or gradient descent. The point of comparison is tractable marginal inference, which GMMs share and flows and VAEs lack.
- **Block:** `experimental.circuits.LearnSpn` and `experimental.circuits.TensorizedCircuit` (random binary-tree region graphs, Bernoulli/Gaussian/categorical leaves, EM and Adam fitting, exact marginals and MPE).
- **Experiment:** `experiments.probabilistic.CircuitShootout`.
- **Toy task:** binarised digits and `ToyTables` discrete tables (log-likelihood, marginal queries with 50% missing variables, and conditional classification).
- **Baselines:** `GaussianMixture`/Bernoulli mixtures by EM; `RealNvpFlow` and `PixelCnn` for likelihood; `FactorGraph` belief propagation for marginals; mean imputation for missing data.
- **Expected metric:** test log-likelihood, exact-marginal error against a brute-force reference on small tables, and accuracy of classification by conditional inference with missing inputs. Hypothesis: circuits lose to flows and PixelCNN on likelihood but give exact marginals in milliseconds and beat mixtures and imputation when inputs are missing.
- **Source:** [ICML 2013](https://proceedings.mlr.press/v28/gens13.html), [arXiv:1806.01910](https://arxiv.org/abs/1806.01910), [arXiv:2004.06231](https://arxiv.org/abs/2004.06231)

### 11.3.12 Beyond monotone circuits: squared circuits with negative weights and Inception PCs (Loconte et al., ICLR 2024; Wang & Van den Broeck, AAAI 2025)
Monotone circuits need non-negative weights, which limits how compactly they can represent some distributions. Squared circuits allow real (even complex) weights and define p(x) ∝ f(x)²; squaring a structured-decomposable circuit takes quadratic time and keeps the partition function tractable. Wang and Van den Broeck prove that neither class contains the other with compact size, and propose Inception PCs, which sum some latent variables outside the square and some inside it, p(x) ∝ Σ_u |Σ_w f(x, u, w)|², recovering both as special cases.
- **Block:** `experimental.circuits.SquaredCircuit` (real and complex weights, squared partition function) and `experimental.circuits.InceptionCircuit`, on 11.3.11's tensorised layers.
- **Experiment:** `CircuitShootout`.
- **Toy task:** as 11.3.11, at matched parameter counts; plus a planted distribution with cancellation structure where monotone circuits are known to need exponential size.
- **Baselines:** 11.3.11's monotone circuits at the same size.
- **Expected metric:** test log-likelihood against parameter count. Hypothesis: squared circuits win at small sizes on the planted distribution and tie on digits; Inception PCs are never worse than both.
- **Source:** [arXiv:2310.00724](https://arxiv.org/abs/2310.00724), [arXiv:2408.00876](https://arxiv.org/abs/2408.00876)

### 11.3.13 Variational Bayesian Gaussian mixtures with automatic component pruning, and a public `GaussianMixture` posterior (Corduneanu & Bishop, AISTATS 2001; Bishop, 2006)
A variational Bayesian GMM puts Dirichlet and Normal–Wishart priors on the mixture and fits a factorised posterior by coordinate ascent. With a small Dirichlet concentration, unneeded components get vanishing weight, so the number of clusters is chosen without BIC sweeps, and the evidence lower bound gives a model-comparison score. Wave 10 WP1 had to compute GMM posteriors by hand (`SampleSelection`); this item adds `GaussianMixture.posterior(x)` and `logLikelihood(x)` as public methods first and uses them in the VB code and its baselines.
- **Block:** `experimental.ml.VariationalGaussianMixture` (CAVI updates, ELBO, finite-Dirichlet and truncated stick-breaking priors) and a public `GaussianMixture.posterior`/`logLikelihood` (existing behaviour unchanged).
- **Experiment:** `CircuitShootout` (density rows) and `experiments.probabilistic.NonparametricClustering` (new rows; existing rows unchanged).
- **Toy task:** `NonparametricClustering`'s blobs with 3–10 true clusters, unequal sizes and an outlier group; streams from `DriftingStreams`.
- **Baselines:** `GaussianMixture` with BIC selection; `DirichletProcessMixture` collapsed Gibbs; k-means with the elbow rule.
- **Expected metric:** ARI, the estimated number of clusters and runtime. Hypothesis: VB-GMM matches GMM + BIC in ARI at one fit instead of ten, and beats DP Gibbs in runtime by an order of magnitude; it over-prunes small clusters at strong concentration priors.
- **Source:** [AISTATS 2001](https://proceedings.mlr.press/r3/corduneanu01a.html), [Bishop, PRML 2006, ch. 10](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)

### 11.3.14 Linear dynamical systems with missing data: a missing-aware Kalman smoother and EM (Shumway & Stoffer, J. Time Series Analysis 1982; Ghahramani & Hinton, 1996) — follow-up
`StateSpaceFilter.kalman` cannot take missing observations, so wave 10 10.5.11 wrote a private local-level smoother for its Gaussian-process comparison. The fix is standard: skip the update (or use only the observed rows of the observation matrix) at missing entries, and run the RTS smoother as usual. With smoothed second moments, EM re-estimates the transition, observation and noise matrices in closed form, so the LDS parameters are learned rather than assumed.
- **Block:** missing-observation support in `experimental.ml.StateSpaceFilter` (NaN-masked rows, unchanged defaults) and `experimental.ml.LinearDynamicalSystemEm` (E-step by the smoother, closed-form M-step, log-likelihood monitoring).
- **Experiment:** `experiments.timeseries.StateSpaceEm`, and a row in `experiments.events.IrregularSeriesShootout` that replaces its local smoother (wave 10 rows unchanged).
- **Toy task:** `IrregularSeries` linear-Gaussian dynamics with MAR, MNAR and block missingness at 20–80%; `SeasonalSeries` with a planted seasonal state.
- **Baselines:** `IrregularSeriesShootout`'s local-linear-trend smoother; the Gaussian-process imputer; BRITS (`TimeSeriesImputer`); linear interpolation.
- **Expected metric:** imputation MAE and one-step forecast error, and EM's parameter recovery error. Hypothesis: on linear-Gaussian series the EM-fitted LDS beats every deep imputer and matches the GP at a fraction of the cost; on chaotic series it loses to BRITS.
- **Source:** [Shumway & Stoffer 1982](https://doi.org/10.1111/j.1467-9892.1982.tb00349.x), [Ghahramani & Hinton 1996](https://mlg.eng.cam.ac.uk/zoubin/papers/tr-96-2.pdf)

### 11.3.15 Hierarchical forecast reconciliation: bottom-up, OLS, WLS, MinT-shrink and robust M-estimation (Hyndman et al., CSDA 2011; Wickramasuriya et al., JASA 2019; 2026)
When series form a hierarchy (totals and their parts), independent base forecasts do not add up. Reconciliation projects them onto the coherent subspace: bottom-up keeps only the leaf forecasts, OLS projects orthogonally, WLS weights by residual variances, and MinT uses a shrunk estimate of the full base-forecast error covariance, which minimises the trace of the reconciled error covariance. A 2026 paper replaces MinT's least squares by a Huber or LAD loss solved by iteratively reweighted reconciliation, so a few bad base forecasts do not spread through the hierarchy.
- **Block:** `experimental.timeseries.Reconciliation` (summing matrix builder, BU, OLS, WLS_STRUCT, WLS_VAR, MINT_SHRINK, ROBUST_M with Huber/LAD weights).
- **Experiment:** `experiments.timeseries.HierarchicalForecasting`.
- **Toy task:** a three-level hierarchy (1 total, 4 regions, 16 leaves) generated from `SeasonalSeries` with correlated leaf noise; a variant with outlier base forecasts.
- **Baselines:** unreconciled base forecasts from `ExponentialSmoothing` and `DLinear` (wave 8); bottom-up.
- **Expected metric:** MASE per level and in total, with and without outliers. Hypothesis: MinT-shrink beats BU and OLS at every level on clean data; with outliers the robust version keeps the gain where MinT degrades.
- **Source:** [Hyndman et al. 2011](https://doi.org/10.1016/j.csda.2011.03.006), [Wickramasuriya et al. 2019](https://doi.org/10.1080/01621459.2018.1448825), [arXiv:2602.22694](https://arxiv.org/abs/2602.22694)

### 11.3.16 Intermittent demand forecasting: Croston, SBA, TSB and temporal aggregation (Croston, ORQ 1972; Syntetos & Boylan, IJF 2005; Teunter et al., EJOR 2011; Nikolopoulos et al., JORS 2011)
Many series are mostly zeros. Croston's method smooths the non-zero demand sizes and the intervals between them separately and forecasts their ratio; SBA multiplies by (1 − α/2) to remove Croston's bias; TSB smooths the demand probability every period instead of the interval, so obsolescence (demand stopping) is tracked; ADIDA aggregates the series into buckets, forecasts the smoother aggregate and disaggregates.
- **Block:** `experimental.timeseries.IntermittentDemand` (CROSTON, SBA, TSB, ADIDA, IMAPA) with an intermittent-series generator option in `SeasonalSeries` (unchanged defaults).
- **Experiment:** `experiments.timeseries.IntermittentDemand`.
- **Toy task:** Poisson–gamma and Bernoulli–lognormal intermittent series with 50–95% zeros, including obsolescence and demand-rate changes.
- **Baselines:** `ExponentialSmoothing` (simple and Holt); DeepAR with a negative-binomial head (`DeepAr`); the zero forecast.
- **Expected metric:** scaled MAE, RMSSE and the bias of cumulative demand. Hypothesis: SBA and TSB beat simple exponential smoothing on RMSSE; TSB is the only one that tracks obsolescence; DeepAR does not beat SBA at test budget.
- **Source:** [Croston 1972](https://doi.org/10.1057/jors.1972.50), [Syntetos & Boylan 2005](https://doi.org/10.1016/j.ijforecast.2004.10.001), [Teunter et al. 2011](https://doi.org/10.1016/j.ejor.2011.05.018)

### 11.3.17 Parametric and deep accelerated failure time models, with a shared Weibull helper (Wei, Statistics in Medicine 1992; Nagpal et al., IEEE JBHI 2021)
Accelerated failure time (AFT) models assume covariates stretch or compress time: log T = xᵀβ + σε with Weibull, log-normal or log-logistic ε, fitted by maximum likelihood with right censoring. Deep Survival Machines replace the single distribution with a mixture of Weibulls or log-normals whose parameters come from a network, which captures non-proportional hazards and gives full survival curves. Wave 10 noted that the repo has no Weibull helper (`SurvivalData`, `OpenMax` and `PointProcessShootout` keep private copies), so this item adds `experimental.survival.Weibull` (density, survival, hazard, censored log-likelihood, MLE) and switches new code to it, without changing those three classes' behaviour.
- **Block:** `experimental.survival.Weibull` and `experimental.survival.AftModels` (WEIBULL, LOG_NORMAL, LOG_LOGISTIC AFT by Newton/L-BFGS; DEEP_SURVIVAL_MACHINES with k mixture components and an ELBO-style censored loss).
- **Experiment:** `experiments.survival.SurvivalShootout` (new rows; existing rows unchanged).
- **Toy task:** `SurvivalData` Weibull data with linear and nonlinear risks, and wave 9's crossing-hazards set where proportional hazards fails; 20–60% censoring.
- **Baselines:** Kaplan–Meier, Cox and DeepSurv (`SurvivalModels`), DeepHit (`DeepHit`) and random survival forests (`SurvivalForest`).
- **Expected metric:** Harrell's and Uno's C-index, integrated Brier score and the calibration of median survival times. Hypothesis: the Weibull AFT matches Cox on Weibull data and gives better calibrated times; Deep Survival Machines beat DeepSurv on crossing hazards and tie DeepHit.
- **Source:** [Wei 1992](https://doi.org/10.1002/sim.4780111409), [arXiv:2003.01176](https://arxiv.org/abs/2003.01176)

---

## WP4. Uncertainty, ensembles and classic-model gaps: recalibration, Venn–Abers, conformal follow-ups and training, cheap and diverse ensembles, stacking, TTA, distillation, multiple-choice learning, binary nets, RBMs, Tsetlin machines, monotonic nets (17)

Target: two new packages, `experimental.calibrate` (11.4.1–11.4.5) and `experimental.ensemble` (11.4.6–11.4.10, 11.4.15–11.4.16), plus additions to `experimental.nn` (11.4.11, 11.4.17) and `experimental.ml` (11.4.12–11.4.13). A new theme `experiments.uncertainty` with `RecalibrationShootout` (11.4.1–11.4.3), `ConformalSets` (11.4.4–11.4.5), `EnsembleShootout` (11.4.6–11.4.10, 11.4.16) and `MultipleHypotheses` (11.4.15); 11.4.11 adds rows to `experiments.compression.CompactModels`, 11.4.12–11.4.13 add `ClassicLearners` to `experiments.classic`, and 11.4.17 adds a monotonic table to `experiments.classic.ClassicMlShootout`. No new generator: the items use `ToyTasks` digits and spirals, `ShapeImages`, `ShiftSuites` (shifted test sets), `LabelCorruption` (imbalance and noise), `TabularStressSuite` and `ToyTables`. A shared fixture caches one trained digits CNN, one `ShapeImages` CNN and one tabular MLP per seed with their validation and test logits, so the post-hoc items 11.4.1–11.4.5 cost seconds. Training items stay at a few hundred steps per member.

### 11.4.1 Post-hoc recalibration maps: histogram binning, isotonic regression, Platt, vector and matrix scaling, beta and Dirichlet calibration (Zadrozny & Elkan, ICML 2001 and KDD 2002; Platt, 1999; Guo et al., ICML 2017; Kull et al., AISTATS 2017 and NeurIPS 2019)
Temperature scaling, the repo's only recalibration map, has one parameter and cannot fix class-wise or non-monotone miscalibration. Histogram binning replaces each confidence bin's score with its empirical accuracy; isotonic regression fits the best monotone step function (pool-adjacent-violators); Platt scaling fits a logistic map per score, vector scaling a per-class temperature and bias, and matrix scaling a full affine map on the logits. Beta calibration fits a three-parameter map that includes the identity, and Dirichlet calibration is matrix scaling on log-probabilities with off-diagonal and intercept regularisation, the multiclass generalisation of beta calibration.
- **Block:** `experimental.calibrate.Recalibrators` (HISTOGRAM, ISOTONIC one-vs-rest with renormalisation, PLATT, VECTOR, MATRIX, BETA, DIRICHLET with ODIR regularisation), all fitted on validation logits with a common `fit`/`apply` interface.
- **Experiment:** `experiments.uncertainty.RecalibrationShootout`.
- **Toy task:** the cached digits, `ShapeImages` and tabular models, in-distribution and on `ShiftSuites` shifts; validation sets of 100–2,000 samples.
- **Baselines:** `Calibration.fitTemperature`; the uncalibrated model; wave 9's debiased and smooth ECE estimators (`CalibrationEstimators`) as the metric.
- **Expected metric:** debiased ECE, class-wise ECE, NLL and accuracy change against validation size. Hypothesis: temperature scaling is hard to beat in top-label ECE at small validation sets; Dirichlet/matrix scaling win in class-wise ECE once validation exceeds about 1,000; binning and isotonic overfit at 100 samples; all maps lose their gains under shift.
- **Source:** [Zadrozny & Elkan 2002](https://doi.org/10.1145/775047.775151), [arXiv:1706.04599](https://arxiv.org/abs/1706.04599), [Beta calibration, AISTATS 2017](https://proceedings.mlr.press/v54/kull17a.html), [arXiv:1910.12656](https://arxiv.org/abs/1910.12656)

### 11.4.2 Structured matrix scaling, ensemble temperature scaling and spline calibration (Nov 2025; Zhang et al., ICML 2020; Gupta et al., ICLR 2021)
More expressive multiclass maps overfit small calibration sets. Structured matrix scaling (Nov 2025) argues from a logistic-regression view that matrix-scaling-type maps are well founded, and controls their variance through structured regularisation of the affine map, robust preprocessing of the logits and efficient optimisation. Ensemble temperature scaling (Mix-n-Match) mixes the temperature-scaled, uniform and uncalibrated predictions with learned weights, which keeps temperature scaling's data efficiency with more expressiveness. Spline calibration fits a smooth monotone spline to the cumulative accuracy against confidence (a Kolmogorov–Smirnov view), avoiding bins.
- **Block:** `experimental.calibrate.StructuredMatrixScaling` (low-rank-plus-diagonal and shrink-to-temperature parametrisations with logit clipping and centring), `experimental.calibrate.EnsembleTemperature` and `experimental.calibrate.SplineCalibration`.
- **Experiment:** `RecalibrationShootout`.
- **Toy task:** as 11.4.1.
- **Baselines:** 11.4.1's temperature, vector, matrix and Dirichlet maps.
- **Expected metric:** as 11.4.1, plus the number of validation samples needed to beat temperature scaling. Hypothesis: structured matrix scaling beats both temperature and Dirichlet calibration in class-wise ECE across validation sizes; ensemble temperature scaling ties temperature in top-label ECE at 100 samples and beats it at 1,000.
- **Source:** [arXiv:2511.03685](https://arxiv.org/abs/2511.03685), [arXiv:2003.07329](https://arxiv.org/abs/2003.07329), [arXiv:2006.12800](https://arxiv.org/abs/2006.12800)

### 11.4.3 Venn–Abers predictors: calibrated probability intervals with a validity guarantee (Vovk & Petej, UAI 2014; Vovk et al., NeurIPS 2015)
Isotonic calibration can be made distribution-free. For a test point, a Venn–Abers predictor fits isotonic regression twice, once with the test point added with label 0 and once with label 1, giving two probabilities p₀ ≤ p₁ that bracket a calibrated probability; under exchangeability one of them is perfectly calibrated. The interval width is itself an uncertainty measure, and p₁/(1 − p₀ + p₁) gives a single log-loss-optimal probability. Inductive and cross Venn–Abers make it cheap.
- **Block:** `experimental.calibrate.VennAbers` (inductive and cross versions, binary and one-vs-rest multiclass, merged probability).
- **Experiment:** `RecalibrationShootout`.
- **Toy task:** binary tasks from spirals and `TabularStressSuite`; one-vs-rest digits.
- **Baselines:** 11.4.1's isotonic and Platt maps; temperature scaling.
- **Expected metric:** NLL, Brier score, ECE and interval width against validation size, in distribution and under `ShiftSuites` shift. Hypothesis: Venn–Abers matches isotonic in Brier score and beats it in NLL at small validation sets, where isotonic produces extreme probabilities; its intervals widen under shift.
- **Source:** [arXiv:1211.0025](https://arxiv.org/abs/1211.0025), [NeurIPS 2015](https://papers.nips.cc/paper/5805-large-scale-probabilistic-predictors-with-and-without-guarantees-of-validity)

### 11.4.4 Conformal sets beyond LAC and APS: RAPS, class-conditional (Mondrian) coverage and conformal risk control (Angelopoulos et al., ICLR 2021 and ICLR 2024; Vovk, 2005) — follow-up
Every conformal experiment since wave 5 uses `ConformalClassifier`'s LAC and APS scores, which guarantee marginal coverage only. RAPS adds a penalty λ·(rank − k_reg)⁺ to APS, which shrinks the large sets that APS produces from noisy tail probabilities. Mondrian (class-conditional) conformal calibrates a separate threshold per class, so every class gets 1 − α coverage, at the cost of larger sets for rare classes. Conformal risk control generalises the guarantee from miscoverage to any bounded monotone loss, such as the false-negative rate in multilabel prediction.
- **Block:** `experimental.calibrate.ConformalSets` (RAPS with λ and k_reg selection, MONDRIAN per-class thresholds, RISK_CONTROL for monotone losses), using `ConformalClassifier`'s scores unchanged.
- **Experiment:** `experiments.uncertainty.ConformalSets`.
- **Toy task:** the cached models; `LabelCorruption` long-tailed digits (imbalance 10–100); a multilabel `ShapeScenes` task (which shapes are present) for risk control.
- **Baselines:** `ConformalClassifier` LAC and APS; top-k sets with the same average size.
- **Expected metric:** marginal and worst-class coverage, average set size and size-stratified coverage; false-negative rate against the target for risk control. Hypothesis: RAPS gives sets about 30–50% smaller than APS at equal coverage; on long-tailed data marginal methods under-cover tail classes by 10–30 points and Mondrian fixes it with larger tail sets.
- **Source:** [arXiv:2009.14193](https://arxiv.org/abs/2009.14193), [arXiv:2208.02814](https://arxiv.org/abs/2208.02814), [Vovk et al., Algorithmic Learning in a Random World, 2005](https://doi.org/10.1007/b106715)

### 11.4.5 Conformal training: learning with a differentiable set-size loss (Stutz et al., ICLR 2022)
Post-hoc conformal prediction takes the model as given. Conformal training (ConfTr) simulates split conformal prediction inside each mini-batch: half the batch calibrates a threshold with a differentiable (smooth-sorting) quantile, the other half forms soft prediction sets with sigmoid membership, and the loss is the expected set size plus an optional class-wise coverage term. The model learns to produce probabilities whose conformal sets are small, while the guarantee still comes from a post-hoc calibration step.
- **Block:** `experimental.calibrate.ConformalTraining` (smooth quantile, soft-set size loss, coverage loss, fine-tuning from a trained model).
- **Experiment:** `ConformalSets`.
- **Toy task:** digits and `ShapeImages`; fine-tuning the cached models for 100–200 steps.
- **Baselines:** cross-entropy training followed by LAC/APS/RAPS (11.4.4); label-smoothed and focal-loss training (`SmoothedCrossEntropy`, `RobustLosses`), which change confidence profiles.
- **Expected metric:** average set size at 90% and 95% coverage, and accuracy. Hypothesis: ConfTr cuts the average set size by 10–30% at equal coverage with less than a point of accuracy loss; the gain is largest where the base model is least accurate.
- **Source:** [arXiv:2110.09192](https://arxiv.org/abs/2110.09192)

### 11.4.6 Ensembles from a single run: snapshot ensembles and fast geometric ensembling (Huang et al., ICLR 2017; Garipov et al., NeurIPS 2018)
Deep ensembles train M models. Snapshot ensembles use a cyclic cosine learning-rate schedule and keep the model at the end of each cycle, so one run yields M members from different local minima; fast geometric ensembling (FGE) starts from a trained model and collects members along a low-loss path with short, high-frequency cycles, exploiting mode connectivity (wave 9's linear mode connectivity probe found these paths in the repo's models).
- **Block:** `experimental.ensemble.SnapshotEnsemble` (cyclic schedule, snapshot capture, uniform and validation-weighted averaging) and `experimental.ensemble.FastGeometricEnsemble`.
- **Experiment:** `experiments.uncertainty.EnsembleShootout`.
- **Toy task:** digits and `ShapeImages` CNNs and a tabular MLP; members at an equal total training budget; OOD sets from `ShiftSuites`.
- **Baselines:** deep ensembles and MC dropout (`PopulationEnsemble`, `UncertaintyBenchmark`); SWA and SWAG (`StochasticWeightAveraging`, `WeightPosterior`); a single model at the same budget.
- **Expected metric:** accuracy, NLL, ECE and OOD AUROC (mutual information) at equal training compute. Hypothesis: snapshot and FGE ensembles beat a single model and SWAG at equal compute and recover most of a deep ensemble's NLL gain, but less of its OOD AUROC, since their members are less diverse.
- **Source:** [arXiv:1704.00109](https://arxiv.org/abs/1704.00109), [arXiv:1802.10026](https://arxiv.org/abs/1802.10026)

### 11.4.7 Ensembles inside one network: MIMO and packed ensembles (Havasi et al., ICLR 2021; Laurent et al., ICLR 2023; 2026)
MIMO feeds M independent inputs concatenated into one network with M output heads during training, so the network learns M independent subnetworks; at test time the same input is repeated M times and the heads are averaged, giving an ensemble for one forward pass. Packed ensembles split every layer into M groups with grouped convolutions and an expansion factor α, so the members share nothing but run as one dense model. A 2026 study asks whether BatchEnsemble behaves like an ensemble at all, finding its members much less diverse than independently trained ones; the same check applies to these.
- **Block:** `experimental.ensemble.MimoNetwork` (input repetition ratio, batch repetition) and `experimental.ensemble.PackedEnsemble` (grouped layers with α and γ), plus a member-diversity probe (pairwise disagreement and functional distance).
- **Experiment:** `EnsembleShootout`.
- **Toy task:** as 11.4.6, at equal inference FLOPs.
- **Baselines:** `BatchEnsembleDense` and `TabM`; deep ensembles; a single model of the same width.
- **Expected metric:** accuracy, NLL, OOD AUROC, inference cost and member disagreement. Hypothesis: packed ensembles approach deep ensembles' NLL at about one model's inference cost; MIMO's gains need a wide network and vanish at the toy widths; BatchEnsemble's member disagreement is far below a deep ensemble's, as the 2026 paper reports.
- **Source:** [arXiv:2010.06610](https://arxiv.org/abs/2010.06610), [arXiv:2210.09184](https://arxiv.org/abs/2210.09184), [arXiv:2601.16936](https://arxiv.org/abs/2601.16936)

### 11.4.8 Stacking and the super learner (Wolpert, Neural Networks 1992; van der Laan et al., 2007)
Averaging treats ensemble members as equals. Stacking trains a meta-learner on out-of-fold predictions of diverse base learners; the super learner restricts the meta-learner to a convex combination chosen by cross-validated loss and comes with an oracle inequality: asymptotically it does as well as the best combination of the library. On tabular data, a library of trees, linear models, k-NN and MLPs often beats every member.
- **Block:** `experimental.ensemble.SuperLearner` (k-fold out-of-fold predictions, non-negative least squares or simplex-constrained log-loss weights, and a ridge/logistic stacker).
- **Experiment:** `EnsembleShootout`.
- **Toy task:** `TabularStressSuite` regression and classification tasks and `ToyTables`.
- **Baselines:** each library member alone (core `GradientBoostedTrees`, `RandomForest`, k-NN, ridge, an MLP, `PriorFittedNetwork`); the uniform average; picking the best by CV (the "discrete super learner"); wave 10's PFN-Boost.
- **Expected metric:** test RMSE or log-loss, and the rank against the best single member per task. Hypothesis: the super learner is never worse than the discrete super learner by more than noise and beats it on about half the tasks; the unconstrained stacker overfits on small tasks.
- **Source:** [Wolpert 1992](https://doi.org/10.1016/S0893-6080(05)80023-1), [van der Laan et al. 2007](https://doi.org/10.2202/1544-6115.1309)

### 11.4.9 Test-time augmentation and learned TTA aggregation (Krizhevsky et al., 2012; Shanmugam et al., ICCV 2021)
Averaging predictions over flips, shifts and crops of a test image is a cheap ensemble. Shanmugam et al. show that naive averaging can hurt: it changes many correct predictions to wrong ones when some augmentations are out of distribution for the model, and they learn per-augmentation (and per-class) aggregation weights on validation data instead. TTA also changes calibration, so it must be compared after recalibration.
- **Block:** `experimental.ensemble.TestTimeAugmentation` (augmentation policies from core `Augmentation`, MEAN, MAX and LEARNED aggregation with per-augmentation or per-class weights).
- **Experiment:** `EnsembleShootout`.
- **Toy task:** the cached digits and `ShapeImages` models; clean and `ShiftSuites`-corrupted test sets.
- **Baselines:** single-pass prediction; MC dropout at the same number of passes; 11.4.6's snapshot ensemble at the same number of passes.
- **Expected metric:** accuracy, NLL after temperature scaling, the counts of corrections and corruptions (right → wrong), and passes. Hypothesis: naive TTA with flips hurts digits (flipped digits are other classes) while learned weights recover a small gain; under corruption TTA gives more than MC dropout per pass.
- **Source:** [ImageNet classification, NeurIPS 2012](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks), [arXiv:2011.11156](https://arxiv.org/abs/2011.11156)

### 11.4.10 Diversity on purpose: negative correlation learning, DivDis and D-BAT (Liu & Yao, Neural Networks 1999; Lee et al., ICLR 2023; Pagliardini et al., ICLR 2023)
Members trained independently can all latch onto the same shortcut. Negative correlation learning penalises the correlation of each member's error with the ensemble's error during joint training. DivDis trains several heads and, on unlabelled target data, minimises the mutual information between their predictions, so the heads pick different features; D-BAT trains models sequentially, each agreeing with the labels on training data but disagreeing with its predecessors on out-of-distribution inputs. With spurious features, one head ends up using the causal feature, and a few target labels pick it.
- **Block:** `experimental.ensemble.DiverseEnsembles` (NCL penalty, DIVDIS mutual-information head loss with head selection from k target labels, DBAT sequential disagreement).
- **Experiment:** `EnsembleShootout`.
- **Toy task:** `CausalAndFairness.colouredShapes` with agreement 0.9–1.0 between colour and shape (wave 10's accessor) and `ShiftSuites` spurious splits; a no-shortcut control.
- **Baselines:** deep ensembles; `GroupRobustTraining` (JTT, GroupDRO) and DFR from wave 10, which use group labels; ERM.
- **Expected metric:** worst-group accuracy on the decorrelated test set, head diversity, and accuracy on the control task. Hypothesis: DivDis or D-BAT find a shape-based head at 100% spurious agreement, where deep ensembles and JTT cannot; NCL alone does not.
- **Source:** [Liu & Yao 1999](https://doi.org/10.1016/S0893-6080(99)00073-8), [arXiv:2202.03418](https://arxiv.org/abs/2202.03418), [D-BAT, ICLR 2023](https://openreview.net/forum?id=K7CbYQbyYhY)

### 11.4.11 Binary neural networks: BinaryConnect, BNNs, XNOR-Net and Bi-Real (Courbariaux et al., NeurIPS 2015 and 2016; Rastegari et al., ECCV 2016; Liu et al., ECCV 2018)
BinaryConnect keeps real-valued latent weights but uses their signs in the forward pass, with a straight-through estimator; BNNs binarise activations too, so matrix products become XNOR and popcount. XNOR-Net scales the binary weights by their mean absolute value per filter, and Bi-Real adds real-valued shortcuts around every binary convolution and a piecewise-polynomial approximation of the sign's derivative, which closes much of the accuracy gap. The repo has ternary (BitNet) and 2-bit layers but no 1-bit networks with binary activations.
- **Block:** `experimental.nn.BinaryLayers` (BINARY_CONNECT, BNN with clipped STE, XNOR scaling, BI_REAL shortcuts and ApproxSign) and an XNOR/popcount `long`-packed inference kernel.
- **Experiment:** `experiments.compression.CompactModels` (new rows; existing rows unchanged).
- **Toy task:** digits and `ShapeImages` MLPs and CNNs.
- **Baselines:** `BitLinear` ternary layers; `DoReFaQuantizer` at 1–2 bits; the full-precision model.
- **Expected metric:** accuracy, model bits and packed-inference speed against the float kernel. Hypothesis: BNNs lose 10+ points on `ShapeImages` CNNs while Bi-Real recovers half of the gap; the popcount kernel is several times faster than float matmul at batch 1.
- **Source:** [arXiv:1511.00363](https://arxiv.org/abs/1511.00363), [arXiv:1602.02830](https://arxiv.org/abs/1602.02830), [arXiv:1603.05279](https://arxiv.org/abs/1603.05279), [arXiv:1808.00278](https://arxiv.org/abs/1808.00278)

### 11.4.12 Restricted Boltzmann machines: CD-k, persistent CD, parallel tempering and AIS likelihoods (Hinton, Neural Computation 2002; Tieleman, ICML 2008; Desjardins et al., AISTATS 2010; Salakhutdinov & Murray, ICML 2008)
An RBM is a bipartite Boltzmann machine whose hidden units are conditionally independent given the visible ones, so block Gibbs sampling is cheap. Contrastive divergence approximates the log-likelihood gradient with k Gibbs steps from the data; persistent CD keeps the negative chains between updates, which gives better likelihoods; parallel tempering runs chains at several temperatures and swaps them, so the negative phase mixes between modes. Annealed importance sampling estimates the partition function, which is needed to compare RBMs by likelihood at all.
- **Block:** `experimental.ml.RestrictedBoltzmannMachine` (Bernoulli–Bernoulli and Gaussian–Bernoulli, CD_K, PCD, PARALLEL_TEMPERING), with log Z from `AnnealedImportanceSampler`.
- **Experiment:** `experiments.classic.ClassicLearners`.
- **Toy task:** binarised digits (12×12) and `ToyTables` binary tables; a planted bimodal distribution where CD-1 fails to mix.
- **Baselines:** the wave 6 p-bit Boltzmann chains (`DenoisingThermodynamicModel`, trained with CD); a Bernoulli mixture (11.3.11's leaves); a small VAE (`ImageAutoencoders`).
- **Expected metric:** AIS test log-likelihood, samples' mode coverage on the bimodal task, and linear-probe accuracy on hidden features. Hypothesis: PCD beats CD-1 by tens of nats per digit image; parallel tempering covers both modes where CD-1 and PCD collapse to one; the VAE has a better likelihood bound.
- **Source:** [Hinton 2002](https://doi.org/10.1162/089976602760128018), [Tieleman 2008](https://doi.org/10.1145/1390156.1390290), [Desjardins et al. 2010](https://proceedings.mlr.press/v9/desjardins10a.html), [Salakhutdinov & Murray 2008](https://doi.org/10.1145/1390156.1390266)

### 11.4.13 Tsetlin machines and the fuzzy-pattern Tsetlin machine (Granmo, 2018; Hnilov, Aug 2025)
A Tsetlin machine learns conjunctive clauses over Boolean literals, each literal's inclusion controlled by a Tsetlin automaton (a two-action counter), and classifies by summing the votes of positive and negative clauses; training gives type I feedback (reinforcing patterns for true positives) and type II feedback (combating false positives) with a specificity s and a vote threshold T. Its clauses are directly readable as rules. The fuzzy-pattern variant (2025) lets a clause still vote when some literals fail, with a score reduced in proportion to the number of mismatches, and reports far fewer clauses for the same accuracy.
- **Block:** `experimental.ml.TsetlinMachine` (bit-packed literals, type I/II feedback, multiclass, clause inspection) and a FUZZY_PATTERN evaluation mode; thermometer encoding for continuous features.
- **Experiment:** `ClassicLearners`.
- **Toy task:** binarised digits, `ToyTables` and `TabularStressSuite` (thermometer-encoded), and noisy XOR/parity rules with a known clause solution.
- **Baselines:** `LogicGateLayer` networks (wave 6), logistic regression, `RandomForest` and `HoeffdingTree`; the `HdcClassifier` hyperdimensional classifier.
- **Expected metric:** accuracy against the number of clauses, training time, and recovery of the planted rules. Hypothesis: the Tsetlin machine recovers the planted XOR rules exactly and trails random forests on digits by a few points; the fuzzy variant reaches the same accuracy with about 10x fewer clauses.
- **Source:** [arXiv:1804.01508](https://arxiv.org/abs/1804.01508), [arXiv:2508.08350](https://arxiv.org/abs/2508.08350)

### 11.4.14 Ensemble distillation: ensemble distribution distillation and Hydra multi-head students (Malinin et al., ICLR 2020; Tran et al., 2020)
Distilling an ensemble's mean prediction into one student keeps its accuracy but erases its disagreement, which is the part that signals epistemic uncertainty. Ensemble distribution distillation trains the student to output the parameters of a Dirichlet that fits the ensemble members' predictive distributions, so total and knowledge uncertainty stay separable; Hydra shares a trunk and gives each member a light head trained to imitate it, keeping member diversity at almost single-model cost.
- **Block:** `experimental.ensemble.EnsembleDistillation` (MEAN_KD, DIRICHLET (EnD²) with temperature annealing, HYDRA heads), reusing `DistillationLoss`.
- **Experiment:** `EnsembleShootout`.
- **Toy task:** a five-member deep ensemble on digits and `ShapeImages` from `PopulationEnsemble`; OOD sets from `ShiftSuites`.
- **Baselines:** the teacher ensemble; a single model; mean distillation; `EvidentialHead` (a Dirichlet head trained without a teacher).
- **Expected metric:** accuracy, NLL, and OOD AUROC by mutual information or Dirichlet precision. Hypothesis: mean distillation keeps accuracy but loses most of the ensemble's OOD AUROC; EnD² and Hydra keep most of it at one model's cost; evidential heads stay below both.
- **Source:** [arXiv:1905.00076](https://arxiv.org/abs/1905.00076), [arXiv:2001.04694](https://arxiv.org/abs/2001.04694)

### 11.4.15 Multiple-choice learning: winner-takes-all hypotheses for ambiguous targets (Guzman-Rivera et al., NeurIPS 2012; Lee et al., NeurIPS 2016; Rupprecht et al., ICCV 2017; Letzelter et al., NeurIPS 2023)
When a target is multimodal, a regressor trained on squared error predicts the mean, which may be impossible. Multiple-choice learning trains K hypothesis heads and back-propagates only the best head's loss per example (winner takes all), so the heads spread over the modes; relaxed WTA gives the other heads a small ε of the gradient to avoid dead heads, and resilient MCL adds a scoring head that estimates each hypothesis's probability, turning the set into a proper distribution estimate.
- **Block:** `experimental.ensemble.MultipleChoiceLearning` (WTA, RELAXED_WTA, EVOLVING_WTA, resilient scoring heads), for regression and classification.
- **Experiment:** `experiments.uncertainty.MultipleHypotheses`.
- **Toy task:** inverse kinematics of a two-link arm (two solutions), the inverse of sin (many branches), and future positions of a `ShapeScenes` object that may turn left or right.
- **Baselines:** a mixture density network (as in `SimulationBasedInference`); an ensemble of independent regressors; `NaturalGradientBoosting` (NGBoost) Gaussian outputs; quantile regression.
- **Expected metric:** oracle (best-of-K) error, distortion/coverage of the modes, and NLL for the scored version. Hypothesis: WTA heads find all modes where an ensemble collapses to the mean; resilient MCL matches the MDN's NLL with more stable training.
- **Source:** [NeurIPS 2012](https://papers.nips.cc/paper/4549-multiple-choice-learning-learning-to-produce-multiple-structured-outputs), [arXiv:1606.07839](https://arxiv.org/abs/1606.07839), [arXiv:1612.00197](https://arxiv.org/abs/1612.00197), [arXiv:2311.01052](https://arxiv.org/abs/2311.01052)

### 11.4.16 Deep ensembles under label noise and imbalance: a diversity–calibration audit across the ensemble zoo
Waves 5–10 built about a dozen ensemble and uncertainty methods (deep ensembles, MC dropout, SWAG, BatchEnsemble, TabM, SNGP, DUQ, evidential heads, Laplace) and this WP adds five more, but each was measured on clean, balanced data. This audit runs them all through one protocol on `LabelCorruption`'s noisy and long-tailed variants, where ensembles are expected to help most, and reports the diversity–accuracy–calibration triangle with wave 9's error bars, so later waves know which uncertainty method to default to.
- **Block:** `experimental.ensemble.EnsembleAudit` (a common `Member` interface with adapters for the existing blocks, ambiguity decomposition of the ensemble loss, disagreement and functional diversity).
- **Experiment:** `EnsembleShootout.audit`.
- **Toy task:** digits with 0/20/40% symmetric noise and imbalance ratios 1/10/100 from `LabelCorruption`; `ShiftSuites` OOD sets.
- **Baselines:** every existing method named above, at equal training compute, plus 11.4.6–11.4.7 and 11.4.14.
- **Expected metric:** accuracy, NLL, ECE after temperature scaling, OOD AUROC and member disagreement, with paired bootstrap intervals (`EvalStatistics`). Hypothesis: deep ensembles stay the best overall; under 40% noise the gap between deep ensembles and single-run methods widens; SNGP leads on OOD AUROC but not on noisy-label NLL.
- **Source:** [Lakshminarayanan et al., arXiv:1612.01474](https://arxiv.org/abs/1612.01474), [Ovadia et al., arXiv:1906.02530](https://arxiv.org/abs/1906.02530)

### 11.4.17 Monotonic networks: min–max networks, deep lattice networks and constrained monotonic MLPs (Sill, NeurIPS 1998; You et al., NeurIPS 2017; Runje & Shankaranarayana, ICML 2023)
In credit, pricing and medicine the output must be monotone in some inputs. Min–max networks take the minimum over groups of the maximum of linear units with non-negative weights on the monotone inputs, a universal approximator of monotone functions; deep lattice networks interleave calibrators and multilinear lattices with monotonicity constraints on the lattice parameters; constrained monotonic MLPs use non-negative weights with a mix of convex, concave and saturated activations so that monotonicity holds by construction without losing expressiveness.
- **Block:** `experimental.nn.MonotonicLayers` (MIN_MAX, LATTICE with piecewise-linear calibrators, CONSTRAINED_MLP) and a monotonicity verifier (pairwise checks on a grid).
- **Experiment:** `experiments.classic.ClassicMlShootout.monotonic` (new method).
- **Toy task:** `TabularStressSuite` tasks with a planted monotone relation in 1–4 features, small training sets (100–500 rows), and an extrapolation split beyond the training range.
- **Baselines:** an unconstrained MLP; core `GradientBoostedTrees` (no monotone constraints, written as an option next to it in the experiment only); isotonic regression for the one-feature case; `NeuralAdditiveModel`.
- **Expected metric:** test RMSE, the share of monotonicity violations, and extrapolation error. Hypothesis: the constrained models match the MLP's RMSE in distribution, have zero violations by construction, and beat it at 100 rows and on extrapolation.
- **Source:** [Sill 1998](https://papers.nips.cc/paper/1358-monotonic-networks), [arXiv:1709.06680](https://arxiv.org/abs/1709.06680), [arXiv:2205.11775](https://arxiv.org/abs/2205.11775)

---

## WP5. Perception and language gaps: pitch, beats, chords, fingerprints, sound events, word alignment, noisy-channel correction, SimCSE, grammar induction, image anomaly detection, action recognition, matching, edges, hashing, FFT convolution (17)

Target: two new packages, `experimental.music` (11.5.1–11.5.6) and `experimental.nlp` (11.5.7–11.5.9), plus additions to `experimental.parse` (11.5.10), `experimental.anomaly` (11.5.11–11.5.12) and `experimental.vision` (11.5.13–11.5.17). A new theme `experiments.music` with `PitchShootout` (11.5.1–11.5.2), `RhythmAndHarmony` (11.5.3–11.5.4) and `AudioRetrieval` (11.5.5–11.5.6); `experiments.language` gains `ClassicNlp` (11.5.7–11.5.9), `experiments.parsing` gains `GrammarInduction` (11.5.10), and `experiments.vision` gains `ImageAnomalyShootout` (11.5.11–11.5.12), `ActionRecognition` (11.5.13), `FeatureMatching` (11.5.14), `EdgeDetection` (11.5.15), `HashingRetrieval` (11.5.16) and `SpectralConvolution` (11.5.17). Four new generators in `experimental.data`: `ToyMusic` (monophonic and polyphonic note sequences rendered with `HarmonicNoiseSynth` and simple drum sounds, with a tempo of 60–180 BPM, a chord progression, onsets, beats, F0 tracks, voicing and event labels, and reverb/noise options; 2–5 s clips at 8 kHz), `ParallelText` (sentence pairs from `SyntheticGrammar` rendered in a source and a target "language" with a lexicon, reordering rules, fertility and null words, with gold alignments), `DefectImages` (`ShapeImages` objects as "products" with scratches, holes, stains and misplacements as anomalies, with pixel masks; five object categories for the multi-class setting) and `ShapeClips` (8–16-frame clips of moving shapes performing 6–10 actions — slide, bounce, rotate, grow, shrink, swap, orbit — with appearance-matched distractors so that single frames are not enough). Every test class trains small models for a few hundred steps so that each finishes in about 10 s.

### 11.5.1 Classical pitch estimation: autocorrelation, YIN and probabilistic YIN (de Cheveigné & Kawahara, JASA 2002; Mauch & Dixon, ICASSP 2014)
The repo synthesises speech and music from an F0 track but cannot estimate one. Autocorrelation picks the lag of the highest normalised peak; YIN computes the squared difference function, normalises it by its cumulative mean (which removes the zero-lag bias), takes the first dip below a threshold, and refines the lag by parabolic interpolation, which avoids octave errors. pYIN uses a distribution over thresholds to get several F0 candidates per frame with probabilities and decodes a smooth, voiced/unvoiced-aware track with an HMM.
- **Block:** `experimental.music.PitchTrackers` (AUTOCORRELATION, YIN, PYIN with Viterbi decoding through `HiddenMarkovModel`).
- **Experiment:** `experiments.music.PitchShootout`.
- **Toy task:** `ToyMusic` monophonic melodies and `SpeakerVoices` utterances (known F0 tracks), at 0–20 dB SNR, with vibrato and reverb options.
- **Baselines:** the spectral-peak pitch (argmax of the core `MelSpectrogram`/STFT magnitude below 1 kHz); harmonic-sum spectral pitch.
- **Expected metric:** raw pitch accuracy (within 50 cents), raw chroma accuracy, voicing F1 and octave-error rate. Hypothesis: YIN beats autocorrelation and spectral peaks above 10 dB SNR, mainly by fewer octave errors; pYIN adds voicing accuracy and wins at 0–5 dB.
- **Source:** [YIN, JASA 2002](https://doi.org/10.1121/1.1458024), [pYIN, ICASSP 2014](https://doi.org/10.1109/ICASSP.2014.6853678)

### 11.5.2 Neural pitch estimation: CREPE, self-supervised PESTO and lightweight self-supervised voicing (Kim et al., ICASSP 2018; Riou et al., ISMIR 2023; Bitra & Beigi, ICPRAM 2026)
CREPE classifies raw waveform frames into 360 pitch bins with a CNN trained on labelled F0. PESTO needs no labels: a small network on constant-Q frames with a Toeplitz-structured output layer is trained so that shifting the CQT along frequency shifts its predicted pitch distribution by the same number of bins (a transposition-equivariance loss), plus invariance to timbre-changing augmentations. The 2026 follow-up uses the same shift-consistency signal per frame (a shift cross-entropy) to down-weight noisy frames in an EM-like loop and to pseudo-label voicing for a linear voicing classifier.
- **Block:** `experimental.music.Crepe` (small waveform CNN, Gaussian-blurred bin targets) and `experimental.music.EquivariantPitch` (CQT front end, Toeplitz head, equivariance and invariance losses, shift-consistency frame weighting and a pseudo-labelled voicing classifier).
- **Experiment:** `PitchShootout`.
- **Toy task:** as 11.5.1; PESTO trained on unlabelled `ToyMusic` audio only; transfer to `SpeakerVoices`.
- **Baselines:** 11.5.1's YIN and pYIN; CREPE with labels as the supervised reference.
- **Expected metric:** as 11.5.1, plus training labels used. Hypothesis: supervised CREPE is best in distribution; the self-supervised model beats YIN at 0–10 dB SNR with no labels; its shift-consistency-weighted voicing classifier beats pYIN's voicing.
- **Source:** [arXiv:1802.06182](https://arxiv.org/abs/1802.06182), [arXiv:2309.02265](https://arxiv.org/abs/2309.02265), [arXiv:2601.11768](https://arxiv.org/abs/2601.11768)

### 11.5.3 Onset detection and beat tracking: spectral flux, dynamic-programming beat tracking and Beat This!'s shift-tolerant loss (Bello et al., IEEE TSAP 2005; Ellis, JNMR 2007; Foscarin et al., ISMIR 2024)
Onset detection functions measure how suddenly the spectrum changes: spectral flux sums the positive differences of the log-magnitude spectrum between frames, and peaks above an adaptive threshold are onsets. Ellis's beat tracker estimates a global tempo from the onset envelope's autocorrelation and then finds beat times by dynamic programming that trades onset strength against deviation from the tempo period. Beat This! replaces both with a network trained by a shift-tolerant loss — binary cross-entropy after max-pooling predictions over ±3 frames, ignoring negatives near annotations — and simple peak picking instead of a dynamic Bayesian network.
- **Block:** `experimental.music.OnsetDetection` (SPECTRAL_FLUX, superflux-style max-filtered flux, adaptive peak picking), `experimental.music.BeatTracker` (tempo estimation and Ellis DP) and `experimental.music.NeuralBeatTracker` (small conv-transformer on mel frames, shift-tolerant loss, peak picking).
- **Experiment:** `experiments.music.RhythmAndHarmony`.
- **Toy task:** `ToyMusic` clips with drums and without, at steady and changing tempi, with expressive timing jitter.
- **Baselines:** energy-envelope onsets; a frame classifier trained with plain BCE (the ablation of the shift-tolerant loss).
- **Expected metric:** onset F1 (±50 ms), beat F1 (±70 ms) and continuity (CMLt). Hypothesis: spectral flux beats the energy envelope on soft onsets; the DP tracker wins on steady tempi; the neural tracker with the shift-tolerant loss has the best F1 on tempo changes, and plain BCE produces smeared peaks.
- **Source:** [Bello et al. 2005](https://doi.org/10.1109/TSA.2005.851998), [Ellis 2007](https://doi.org/10.1080/09298210701653344), [arXiv:2407.21658](https://arxiv.org/abs/2407.21658)

### 11.5.4 Chroma features and chord recognition with an HMM (Fujishima, ICMC 1999; Sheh & Ellis, ISMIR 2003)
Chroma vectors fold the spectrum onto the 12 pitch classes, discarding octave and much of timbre. Chord recognition matches each frame's chroma to chord templates (major, minor; 24 classes plus no-chord) or Gaussian emission models, then smooths the frame decisions with an HMM whose transition matrix favours staying on the same chord; a small CNN on log-frequency spectra can replace the hand-made chroma.
- **Block:** `experimental.music.Chroma` (CQT-style pitch-class profiles, log compression) and `experimental.music.ChordRecognizer` (TEMPLATE, GAUSSIAN_HMM via `HiddenMarkovModel`, CNN_CRF via `LinearChainCrf`).
- **Experiment:** `RhythmAndHarmony`.
- **Toy task:** `ToyMusic` polyphonic clips with chord progressions, inversions and passing tones, with timbre variation between train and test.
- **Baselines:** frame-wise template matching without smoothing; an MLP on raw mel frames.
- **Expected metric:** weighted chord symbol recall (majmin) and segmentation quality. Hypothesis: HMM smoothing adds 10+ points over frame-wise templates; the CNN-CRF beats templates only when timbre differs between train and test less than in the hardest split.
- **Source:** [Fujishima 1999](https://quod.lib.umich.edu/i/icmc/bbp2372.1999.446), [Sheh & Ellis 2003](https://doi.org/10.7916/D8V12P4J)

### 11.5.5 Audio fingerprinting: spectral-peak landmarks against neural contrastive fingerprints (Wang, ISMIR 2003; Chang et al., ICASSP 2021)
A fingerprinting system finds which reference recording a short, degraded query comes from. The landmark method picks local maxima of the spectrogram, pairs each anchor peak with peaks in a target zone, and hashes (f₁, f₂, Δt); matching queries vote for (track, offset) pairs, and a consistent offset gives the match. Neural fingerprints train a small encoder contrastively so that segments and their degraded versions (noise, filtering, time shifts) map to nearby embeddings, then search with nearest neighbours.
- **Block:** `experimental.music.LandmarkFingerprint` (peak picking, target-zone pairing, hash index, offset voting) and `experimental.music.NeuralFingerprint` (segment encoder with `ContrastivePairLoss`' InfoNCE, `KnnIndex` search).
- **Experiment:** `experiments.music.AudioRetrieval`.
- **Toy task:** a database of 200–500 `ToyMusic` and `SpeakerVoices` recordings; 1–3 s queries with noise at 0–20 dB, low-pass filtering, speed change of ±2% and reverb.
- **Baselines:** MFCC + DTW template matching (`Mfcc`, `DynamicTimeWarping`, wave 7); mean-pooled MFCC nearest neighbour.
- **Expected metric:** top-1 identification rate against query length and degradation, and query time. Hypothesis: landmarks are near-perfect at ≥ 10 dB and fast, but fail under speed changes; neural fingerprints are more robust to speed and reverb; MFCC-DTW is slowest and least robust.
- **Source:** [Wang 2003](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf), [arXiv:2010.11910](https://arxiv.org/abs/2010.11910)

### 11.5.6 Sound event detection with a convolutional-recurrent network (Çakır et al., IEEE/ACM TASLP 2017)
Sound event detection labels which events (here: notes of given instruments, drum hits, speech, clicks) are active in each frame, with overlaps. A CRNN applies convolutions over the log-mel spectrogram, pools only over frequency, runs a bidirectional GRU over time and outputs per-frame sigmoids per class; median filtering and thresholds turn these into events, scored with event-based F1.
- **Block:** `experimental.music.SoundEventDetector` (CRNN, frame and event post-processing, event-based and segment-based metrics).
- **Experiment:** `AudioRetrieval` (detection table).
- **Toy task:** `ToyMusic` scenes mixing 1–4 overlapping event types with speech from `SpeakerVoices`, at varied SNR.
- **Baselines:** a frame-wise MLP; the wave 9 DS-CNN keyword spotter (`KeywordSpotters`) applied in sliding windows; wave 10's VAD for the speech class.
- **Expected metric:** event-based F1 (onset ±200 ms) and segment-based error rate. Hypothesis: the CRNN beats the frame MLP by 10+ points of event F1 through temporal context, and beats sliding-window keyword spotters on overlapping events.
- **Source:** [arXiv:1702.06286](https://arxiv.org/abs/1702.06286)

### 11.5.7 Statistical word alignment: IBM Models 1 and 2 and the HMM alignment model (Brown et al., Computational Linguistics 1993; Vogel et al., COLING 1996)
IBM Model 1 learns word translation probabilities t(f | e) from sentence pairs alone by EM, treating each target word as generated by a uniformly chosen source word (or NULL); Model 2 adds position-dependent alignment probabilities; the HMM model makes the alignment of each word depend on the previous word's alignment through a jump-width distribution, which captures local monotonicity. Viterbi alignments in both directions are combined by intersection or grow-diag.
- **Block:** `experimental.nlp.WordAlignment` (IBM1, IBM2, HMM with forward–backward EM, symmetrisation heuristics).
- **Experiment:** `experiments.language.ClassicNlp`.
- **Toy task:** `ParallelText` with 2k–20k sentence pairs, local and long-distance reordering, fertility-2 words and null words.
- **Baselines:** attention alignments of `Seq2SeqAttention` and `TransformerSeq2Seq` trained on the same pairs; co-occurrence (Dice) alignment.
- **Expected metric:** alignment error rate (AER) against data size, and training time. Hypothesis: the HMM model with grow-diag symmetrisation has the lowest AER; IBM1 beats seq2seq attention alignments at 2k pairs; attention catches up only with more data.
- **Source:** [Brown et al. 1993](https://aclanthology.org/J93-2003/), [Vogel et al. 1996](https://aclanthology.org/C96-2141/)

### 11.5.8 Noisy-channel spelling correction, with a `LogitsProcessor` hook for `WindowLm` (Kernighan et al., COLING 1990; Brill & Moore, ACL 2000)
The noisy-channel model chooses the correction c that maximises P(c)·P(typo | c): a language model supplies the prior, and an error model supplies the channel. Kernighan et al. estimate single-edit confusion probabilities by EM from unlabelled typos; Brill and Moore learn substring-to-substring edits (such as "ent"→"ant") with position, which captures realistic errors. Correction in context uses a word-level LM. Wave 10 WP2 found that `WindowLm` sampling has no `LogitsProcessor` hook; this item adds it (unchanged default behaviour) and uses it to decode corrections character by character from `WindowLm` constrained by the channel scores.
- **Block:** `experimental.nlp.NoisyChannelCorrector` (candidate generation within edit distance 2, single-edit and Brill–Moore substring channels learned by EM, `KneserNeyLm` priors) and a `LogitsProcessor` hook on `WindowLm` sampling.
- **Experiment:** `ClassicNlp`.
- **Toy task:** `SyntheticCorpus` text with planted typos from a known error model (keyboard-adjacent substitutions, transpositions, phonetic substring errors) at 5–20% word error.
- **Baselines:** nearest dictionary word by Levenshtein distance (the OCR metrics' code); a character-level `TransformerSeq2Seq` trained on (typo, word) pairs; uncorrected text.
- **Expected metric:** word correction accuracy, false-correction rate on correct words, and data needed. Hypothesis: the Brill–Moore channel with a context LM beats nearest-word correction by 10+ points and beats seq2seq at small data; seq2seq wins only with tens of thousands of pairs.
- **Source:** [Kernighan et al. 1990](https://aclanthology.org/C90-2036/), [Brill & Moore 2000](https://aclanthology.org/P00-1037/)

### 11.5.9 SimCSE: sentence embeddings from dropout noise and from entailment pairs (Gao et al., EMNLP 2021)
Unsupervised SimCSE passes the same sentence through an encoder twice with different dropout masks and trains with InfoNCE, using the two views as positives and other sentences in the batch as negatives; dropout alone is enough augmentation, and it prevents the anisotropy of raw LM embeddings. Supervised SimCSE uses entailment pairs as positives and contradictions as hard negatives.
- **Block:** `experimental.nlp.SimCse` (dropout-noise positives, hard negatives, alignment and uniformity diagnostics), over a small transformer encoder and `ContrastivePairLoss`' InfoNCE.
- **Experiment:** `ClassicNlp`.
- **Toy task:** `SyntheticGrammar` sentences with paraphrase groups (same meaning, different surface form) and contradiction pairs; a semantic-similarity test set with graded labels.
- **Baselines:** mean-pooled `StaticWordEmbeddings`; mean-pooled hidden states of `TinyLm`; the encoder trained with an MLM objective only.
- **Expected metric:** Spearman correlation on the similarity set, paraphrase retrieval recall@1, and the alignment/uniformity numbers. Hypothesis: unsupervised SimCSE beats both mean-pooled baselines by a wide margin; the supervised version adds a further gain from hard negatives.
- **Source:** [arXiv:2104.08821](https://arxiv.org/abs/2104.08821)

### 11.5.10 Unsupervised constituency parsing: neural and compound PCFGs (Kim et al., ACL 2019)
Grammar induction learns a PCFG from raw sentences by maximising their marginal likelihood with the inside algorithm. A neural PCFG parametrises rule probabilities with embeddings of the symbols, which shares statistical strength and lets gradient descent work where EM on a flat PCFG gets stuck; a compound PCFG adds a per-sentence continuous latent vector that modulates the rule probabilities, trained with amortised variational inference, which captures sentence-specific structure.
- **Block:** `experimental.parse.NeuralPcfg` (symbol embeddings, softmax rule heads, inside algorithm on `ChartParser`'s chart, MBR/Viterbi decoding) and a COMPOUND option with a sentence encoder and a KL term.
- **Experiment:** `experiments.parsing.GrammarInduction`.
- **Toy task:** `ToyTreebank` sentences without trees for training; the gold trees for scoring only; lengths 5–20.
- **Baselines:** right- and left-branching trees and random trees; EM on a flat PCFG (written as the reference); `TreebankPcfg` and `ChartParser` trained on gold trees (the supervised ceilings).
- **Expected metric:** unlabelled sentence-level F1 against gold constituents. Hypothesis: the neural PCFG beats right-branching, which is a strong baseline for English-like toy grammars, and flat EM; the compound PCFG adds a few points; both stay far below the supervised parsers.
- **Source:** [arXiv:1906.10225](https://arxiv.org/abs/1906.10225)

### 11.5.11 Image anomaly detection from pretrained features: PaDiM and PatchCore (Defard et al., ICPR-W 2020; Roth et al., CVPR 2022)
Industrial anomaly detection has only normal training images. PaDiM fits a multivariate Gaussian to the pretrained-CNN patch features at each spatial position and scores test patches by Mahalanobis distance. PatchCore stores locally aggregated mid-level patch features of all normal images in a memory bank, reduces it by greedy coreset subsampling, and scores each test patch by its distance to the nearest neighbour in the bank; image scores are the maximum over patches, and the patch scores form a segmentation map.
- **Block:** `experimental.anomaly.PatchDistribution` (PaDiM) and `experimental.anomaly.PatchCore` (feature aggregation, greedy coreset, `KnnIndex` scoring, anomaly maps), on a feature extractor from `SelfSupervisedBenchmark.pretrainEncoder` or a supervised `ShapeImages` CNN.
- **Experiment:** `experiments.vision.ImageAnomalyShootout`.
- **Toy task:** `DefectImages`: 100–300 normal images per category; test sets with scratches, holes, stains and misplaced parts, with pixel masks.
- **Baselines:** `DeepSvdd` and `Ecod` on image embeddings; a convolutional autoencoder's reconstruction error (`ImageAutoencoders`); global k-NN on image embeddings.
- **Expected metric:** image-level AUROC and pixel-level AUROC/PRO. Hypothesis: PatchCore has the best image and pixel AUROC; PaDiM is close on aligned objects and falls behind on misplacements; Deep SVDD and the autoencoder trail both.
- **Source:** [arXiv:2011.08785](https://arxiv.org/abs/2011.08785), [arXiv:2106.08265](https://arxiv.org/abs/2106.08265)

### 11.5.12 Dinomaly: reconstruction-based multi-class anomaly detection with a noisy bottleneck and loose reconstruction (Guo et al., CVPR 2025)
Training one model for all categories (multi-class unsupervised anomaly detection) usually costs accuracy, because a strong decoder learns to reconstruct anomalies too ("identity mapping"). Dinomaly uses only attention and MLP blocks: a frozen pretrained transformer encoder, a bottleneck MLP whose ordinary dropout serves as noise injection, a decoder of linear-attention blocks (which "cannot focus" and so do not copy local anomalies), and a loose reconstruction loss that matches groups of encoder layers to groups of decoder layers as a whole rather than layer by layer and point by point. Anomaly maps are cosine distances between encoder and decoder features.
- **Block:** `experimental.anomaly.Dinomaly` (frozen ViT features from `SelfSupervisedBenchmark`'s patch-transformer branch, dropout bottleneck, linear-attention decoder from `experimental.nn`, grouped loose cosine loss with hard-point mining).
- **Experiment:** `ImageAnomalyShootout`.
- **Toy task:** `DefectImages` in the multi-class setting (one model for all five categories) and per category.
- **Baselines:** 11.5.11's PatchCore and PaDiM (per category and pooled); a plain feature-reconstruction model with softmax attention, no dropout and layer-wise loss (the ablation).
- **Expected metric:** image and pixel AUROC, multi-class against per-category. Hypothesis: the ablated reconstruction model loses 5–15 AUROC points in the multi-class setting; Dinomaly closes most of the gap to per-category PatchCore; linear attention and the loose loss each contribute.
- **Source:** [arXiv:2405.14325](https://arxiv.org/abs/2405.14325)

### 11.5.13 Video action recognition: temporal shift, (2+1)-D convolutions and late fusion (Lin et al., ICCV 2019; Tran et al., CVPR 2018; Karpathy et al., CVPR 2014)
Recognising actions needs motion, not just appearance. Late fusion averages per-frame CNN predictions and cannot see order; (2+1)-D convolutions factorise 3-D convolutions into a spatial convolution and a temporal convolution with a non-linearity in between; the temporal shift module (TSM) shifts a fraction of the channels one frame forward and one backward inside a 2-D CNN, which gives temporal modelling at zero extra FLOPs.
- **Block:** `experimental.vision.VideoClassifiers` (LATE_FUSION, R2PLUS1D, TSM residual shift with a shift fraction, and a frame-difference input option).
- **Experiment:** `experiments.vision.ActionRecognition`.
- **Toy task:** `ShapeClips` with 6–10 actions, including time-reversed pairs (grow/shrink, approach/recede) that single frames cannot separate.
- **Baselines:** a single-frame CNN; `ConvLstmCell`-based ConvLSTM classifiers and the wave 8 `SimVp` encoder with a linear head.
- **Expected metric:** accuracy overall and on time-reversed pairs, and FLOPs per clip. Hypothesis: late fusion is at chance on reversed pairs; TSM matches (2+1)-D accuracy at the 2-D CNN's cost; ConvLSTM is slower and no better.
- **Source:** [arXiv:1811.08383](https://arxiv.org/abs/1811.08383), [arXiv:1711.11248](https://arxiv.org/abs/1711.11248), [Karpathy et al. 2014](https://doi.org/10.1109/CVPR.2014.223)

### 11.5.14 Feature matching and robust geometry: Harris corners, learned SuperPoint-style descriptors and RANSAC homographies (Harris & Stephens, 1988; DeTone et al., CVPR-W 2018; Fischler & Bolles, CACM 1981)
Classical matching detects corners (the Harris response from the structure tensor), describes patches, matches descriptors by mutual nearest neighbours with a ratio test, and fits a homography by RANSAC, which samples minimal sets of four matches and keeps the model with most inliers. SuperPoint learns keypoints and descriptors together, self-supervised by homographic adaptation: random homographies warp an image, and detections and descriptors are trained to be consistent under them.
- **Block:** `experimental.vision.FeatureMatching` (Harris detector, normalised-patch and BRIEF-like binary descriptors, mutual-NN ratio matching, DLT homography, RANSAC with adaptive iterations) and `experimental.vision.LearnedKeypoints` (small detector/descriptor heads trained by homographic adaptation).
- **Experiment:** `experiments.vision.FeatureMatching`.
- **Toy task:** `ShapeScenes` and `GlyphDocuments` images warped by random homographies with illumination change, blur and noise; ground-truth homographies.
- **Baselines:** `ClassicalFlow`'s pyramidal Lucas–Kanade tracking followed by the same RANSAC; least-squares homography on all matches (no RANSAC); the spatial transformer of wave 8 trained to regress the homography.
- **Expected metric:** homography accuracy (corner error < 3 px), matching precision and inlier ratio. Hypothesis: RANSAC is essential (least squares fails above 20% outliers); the learned keypoints beat Harris under blur and illumination change; Lucas–Kanade fails at large warps.
- **Source:** [Harris & Stephens 1988](https://doi.org/10.5244/C.2.23), [arXiv:1712.07629](https://arxiv.org/abs/1712.07629), [Fischler & Bolles 1981](https://doi.org/10.1145/358669.358692)

### 11.5.15 Edge detection: Sobel and Canny against holistically-nested edge detection (Canny, TPAMI 1986; Xie & Tu, ICCV 2015)
Canny smooths with a Gaussian, computes gradient magnitude and direction, thins edges by non-maximum suppression along the gradient, and links them with hysteresis between two thresholds. HED trains a fully convolutional network with deep supervision: side outputs at several depths each predict an edge map with a class-balanced cross-entropy, and a learned fusion combines them, so both fine and coarse boundaries are captured.
- **Block:** `experimental.vision.EdgeDetectors` (SOBEL, CANNY with hysteresis, HED with side outputs and balanced loss) and boundary metrics (ODS/OIS F-measure with distance-tolerant matching).
- **Experiment:** `experiments.vision.EdgeDetection`.
- **Toy task:** `ShapeScenes` and `DegradedImages` with true object boundaries (from the renderer's masks), textured backgrounds that create false edges, and noise.
- **Baselines:** the Sobel/Laplacian filters in `NeuralCellularAutomaton`'s perception kernels; the U-Net from `DensePrediction` trained on the same boundary masks.
- **Expected metric:** ODS and OIS F-measure, and average precision. Hypothesis: Canny beats Sobel by suppressing texture edges only partly; HED beats Canny by a wide margin on textured scenes and edges out the U-Net on thin boundaries.
- **Source:** [Canny 1986](https://doi.org/10.1109/TPAMI.1986.4767851), [arXiv:1504.06375](https://arxiv.org/abs/1504.06375)

### 11.5.16 Deep hashing for retrieval: deep supervised hashing and HashNet (Liu et al., CVPR 2016; Cao et al., ICCV 2017)
Learning-to-hash maps images to short binary codes so that Hamming distance reflects semantic similarity. Deep supervised hashing trains real-valued outputs with a pairwise contrastive loss plus a regulariser that pushes them toward ±1; HashNet trains with a continuation method, using tanh(βx) with β increasing over training so that the network converges to exact binary codes, and weights similar and dissimilar pairs to correct their imbalance.
- **Block:** `experimental.vision.DeepHashing` (DSH, HASHNET continuation, bit-packed Hamming search).
- **Experiment:** `experiments.vision.HashingRetrieval`.
- **Toy task:** digits and `ShapeImages` retrieval with 16–64-bit codes, seen and unseen classes.
- **Baselines:** `IvfPqIndex` product quantisation of float embeddings at the same bytes; random-hyperplane LSH and ITQ-style rotated PCA codes; `MetricLearningShootout`'s float embeddings (the uncompressed ceiling).
- **Expected metric:** mAP@all and precision@100 against code length, and query time. Hypothesis: HashNet beats DSH and LSH at 16–32 bits and approaches PQ at equal bytes with faster search; unsupervised codes lose most on unseen classes.
- **Source:** [Liu et al. 2016](https://doi.org/10.1109/CVPR.2016.227), [arXiv:1702.00758](https://arxiv.org/abs/1702.00758)

### 11.5.17 FFT convolutions and spectral pooling, with public convolution helpers and a 2-D FFT (Mathieu et al., ICLR 2014; Rippel et al., NeurIPS 2015)
Convolution in the spatial domain is a product in the Fourier domain, so large kernels are cheaper by FFT. Spectral pooling downsamples by truncating the Fourier spectrum, which keeps more information per parameter than max pooling at the same output size; spectral parametrisation of filters improves conditioning. Wave 10 WP6 reported that the convolution helpers are package-private (every package keeps a local copy) and that there is no 2-D FFT; this item adds `experimental.vision.Conv2dOps` (public im2col, direct and padded convolutions, consolidating the local copies for new code) and `experimental.vision.Fft2d` (radix-2 row–column FFT on core `Complex`), with unchanged behaviour in the existing classes.
- **Block:** `experimental.vision.Fft2d`, `experimental.vision.Conv2dOps` and `experimental.vision.SpectralConv` (FFT convolution forward and backward, spectral pooling, spectral filter parametrisation).
- **Experiment:** `experiments.vision.SpectralConvolution`.
- **Toy task:** timing over kernel sizes 3–15 at image sizes 16–64; digits and `ShapeImages` CNNs with spectral pooling in place of max pooling.
- **Baselines:** direct convolution and `WinogradConv` (wave 8); the core convolution kernel; max and average pooling.
- **Expected metric:** time per convolution against kernel size (the crossover point), numerical error, and accuracy with spectral pooling. Hypothesis: FFT convolution overtakes direct convolution above kernel size 7–9 at 32×32; spectral pooling matches max pooling's accuracy with smoother downsampling and no gain in speed on the CPU.
- **Source:** [arXiv:1312.5851](https://arxiv.org/abs/1312.5851), [arXiv:1506.03767](https://arxiv.org/abs/1506.03767)

---

## WP6. RL beyond the wave 10 recipes: bandits, deep and count-based exploration, skill discovery, MBPO, hierarchy, preference-based and average-reward RL, value factorisation, mean-field and heterogeneous MARL, zero-shot coordination, pixel follow-up (17)

Target: three new packages, `experimental.bandit` (11.6.1–11.6.5), `experimental.explore` (11.6.6–11.6.8, 11.6.10–11.6.11) and `experimental.marl` (11.6.13–11.6.16), plus additions to `experimental.deeprl` (11.6.9, 11.6.12, 11.6.17). A new theme `experiments.bandits` with `StochasticBandits` (11.6.1–11.6.2), `PureExploration` (11.6.3–11.6.4) and `RankingBandits` (11.6.5); `experiments.control` gains `DeepExploration` (11.6.6–11.6.8), `ModelBasedAndHierarchical` (11.6.9–11.6.10) and `RewardFormulations` (11.6.11–11.6.12); `experiments.multiagent` gains `CooperativeMarl` (11.6.13–11.6.16); and `experiments.world.WorldModelShootout` gains a pixel table (11.6.17). New in `experimental.data`: `BanditSuites` (Bernoulli, Gaussian and heavy-tailed arms with small gaps; switching and adversarial reward sequences; preference matrices with and without a Condorcet winner; cascade and position-based click models built on `ClickLogs`' click model), and in `experimental.explore` the `DeepSeaChain` environment (bsuite's Deep Sea of size N = 10–30 with a small action cost and one rewarding corner, plus a RiverSwim chain). RL items report through wave 10's `LearningCurveHarness` (equal environment steps, IQM and bootstrap intervals from `EvalStatistics`). Bandit items run thousands of arms-pulls in milliseconds and use 50–200 seeds; the deep RL items run 2k–20k environment steps on CartPole, MountainCar, `KeyDoorGrid`, `GoalEnvironment` and `DeepSeaChain` so that each test class finishes in about 10 s.

### 11.6.1 Optimal stochastic bandits: KL-UCB, Bernoulli Thompson sampling, MOSS and IMED (Garivier & Cappé, COLT 2011; Agrawal & Goyal, COLT 2012; Audibert & Bubeck, COLT 2009; Honda & Takemura, JMLR 2015)
UCB1's Hoeffding bonus is loose for Bernoulli rewards near 0 or 1. KL-UCB picks the arm whose largest plausible mean, the q with n·KL(μ̂, q) ≤ log t, is highest, which matches the Lai–Robbins lower bound; Bernoulli Thompson sampling draws each arm's mean from its Beta posterior; MOSS uses a horizon-dependent bonus that is minimax optimal; IMED plays the arm minimising n·KL(μ̂, μ̂*) + log n, asymptotically optimal and deterministic.
- **Block:** `experimental.bandit.StochasticBandits` (UCB1, KL_UCB, THOMPSON_BETA and Gaussian, MOSS, IMED), over a common `BanditPolicy` interface.
- **Experiment:** `experiments.bandits.StochasticBandits`.
- **Toy task:** `BanditSuites` Bernoulli instances with 2–100 arms, gaps 0.01–0.2 and means near 0 and 0.5; core `Bandit` Gaussian arms; horizons 10³–10⁵.
- **Baselines:** UCB1 and ε-greedy (`InContextBandits`, `ContextualBandit`); the Lai–Robbins regret lower bound.
- **Expected metric:** cumulative regret curves and regret at T, with intervals over 200 seeds. Hypothesis: KL-UCB, TS and IMED have 2–5x less regret than UCB1 on low-mean Bernoulli arms and track the lower bound's slope; MOSS is best at small T with many arms.
- **Source:** [arXiv:1102.2490](https://arxiv.org/abs/1102.2490), [arXiv:1111.1797](https://arxiv.org/abs/1111.1797), [MOSS, COLT 2009](https://www.learningtheory.org/colt2009/papers/024.pdf), [Honda & Takemura 2015](https://www.jmlr.org/papers/v16/honda15a.html)

### 11.6.2 Adversarial and best-of-both-worlds bandits: EXP3, EXP3-IX and Tsallis-INF (Auer et al., SIAM J. Computing 2002; Neu, NeurIPS 2015; Zimmert & Seldin, JMLR 2021)
When rewards are chosen by an adversary, deterministic index policies can be exploited. EXP3 samples arms from exponential weights on importance-weighted reward estimates; EXP3-IX adds implicit exploration to the importance weights, which gives high-probability regret bounds; Tsallis-INF uses online mirror descent with the ½-Tsallis entropy and is optimal in both the adversarial and the stochastic regime without knowing which one it faces.
- **Block:** `experimental.bandit.AdversarialBandits` (EXP3, EXP3_IX, TSALLIS_INF with the reduced-variance estimator).
- **Experiment:** `StochasticBandits`.
- **Toy task:** `BanditSuites` stochastic instances, switching instances (the best arm changes every 2,000 steps), and an adaptive adversary that punishes the currently favoured arm.
- **Baselines:** 11.6.1's KL-UCB and TS; `BanditMixer`'s EXP3 as used for data mixing (the same algorithm in its existing form); sliding-window UCB.
- **Expected metric:** regret against the best fixed arm in each regime. Hypothesis: UCB-type policies suffer linear regret against the adaptive adversary while EXP3 and Tsallis-INF do not; on stochastic instances Tsallis-INF comes within 2x of TS while EXP3 is far worse.
- **Source:** [Auer et al. 2002](https://doi.org/10.1137/S0097539701398375), [NeurIPS 2015](https://papers.nips.cc/paper/5732-explore-no-more-improved-high-probability-regret-bounds-for-non-stochastic-bandits), [arXiv:1807.07623](https://arxiv.org/abs/1807.07623)

### 11.6.3 Best-arm identification: successive elimination, LUCB, Track-and-Stop, top-two sampling and anytime Almost Tracking (Even-Dar et al., JMLR 2006; Kalyanakrishnan et al., ICML 2012; Garivier & Kaufmann, COLT 2016; Jourdan et al., NeurIPS 2022; Komiyama et al., Oct 2025)
Pure exploration asks for the best arm, not low regret. Successive elimination drops arms whose upper bound falls below the leader's lower bound; LUCB samples the empirical leader and its strongest challenger; Track-and-Stop tracks the optimal sampling proportions from the lower-bound optimisation and stops with a GLR test, which is asymptotically optimal at fixed confidence; top-two algorithms sample the leader with probability β and a challenger otherwise. Almost Tracking (Oct 2025) is a closed-form anytime algorithm: it computes target proportions for the H₁ risk from the empirical means and splits each batch among the under-sampled arms in proportion to their targets, needing no budget in advance.
- **Block:** `experimental.bandit.BestArmIdentification` (SUCCESSIVE_ELIMINATION, LUCB, TRACK_AND_STOP with GLR stopping, TOP_TWO with TS and EB leaders, ALMOST_TRACKING), in fixed-confidence and fixed-budget modes.
- **Experiment:** `experiments.bandits.PureExploration`; plus a hyperparameter-selection use on a `ToyTasks` grid where each pull is a short training run.
- **Toy task:** `BanditSuites` Bernoulli and Gaussian instances with 5–50 arms and "one good arm" and "linearly spaced" gap profiles.
- **Baselines:** uniform allocation; successive halving from core `HyperparameterSearch` (fixed-budget, the existing Hyperband component).
- **Expected metric:** sample complexity at δ = 0.05 (fixed confidence) and error probability against budget (fixed budget and anytime). Hypothesis: Track-and-Stop and top-two need about half the samples of successive elimination; Almost Tracking beats successive halving and uniform allocation at every budget without being told the budget.
- **Source:** [Even-Dar et al. 2006](https://www.jmlr.org/papers/v7/evendar06a.html), [LUCB, ICML 2012](https://icml.cc/2012/papers/359.pdf), [arXiv:1602.04589](https://arxiv.org/abs/1602.04589), [arXiv:2206.05979](https://arxiv.org/abs/2206.05979), [arXiv:2510.23199](https://arxiv.org/abs/2510.23199)

### 11.6.4 Dueling bandits: relative UCB and double Thompson sampling (Zoghi et al., ICML 2014; Wu & Liu, NeurIPS 2016)
When feedback is a preference between two arms (as in interleaved ranking or RLHF comparisons), regret is measured against the Condorcet winner. Relative UCB keeps optimistic estimates of pairwise win probabilities, picks a candidate that could beat every other arm, and duels it against its strongest optimistic challenger; double Thompson sampling draws two samples of the preference matrix, picks the first arm as the sampled Copeland winner and the second as its sampled strongest opponent, and also handles instances without a Condorcet winner.
- **Block:** `experimental.bandit.DuelingBandits` (RUCB, DOUBLE_TS, Copeland-winner variants, and a reduction to `PreferenceLosses`' Bradley–Terry fit).
- **Experiment:** `PureExploration` (dueling table).
- **Toy task:** `BanditSuites` preference matrices: Bradley–Terry utilities, cyclic preferences (no Condorcet winner), and comparisons from wave 9's toy preference oracle.
- **Baselines:** treating each duel as two independent bandit pulls with 11.6.1's TS; uniform random duels; a BT model refitted every 100 duels with greedy selection.
- **Expected metric:** cumulative weak and strong regret, and the rate of identifying the Condorcet or Copeland winner. Hypothesis: double TS has the lowest regret and is the only method that copes with cyclic preferences; the independent-pull reduction has linear regret on cyclic instances.
- **Source:** [arXiv:1312.3393](https://arxiv.org/abs/1312.3393), [arXiv:1604.07101](https://arxiv.org/abs/1604.07101)

### 11.6.5 Combinatorial semi-bandits and cascading bandits for ranking (Chen et al., ICML 2013; Kveton et al., ICML 2015)
Recommending a list of K items is a combinatorial action with feedback on each shown item. CUCB plays the best super-arm under optimistic item estimates and updates every observed item; under the cascade model the user scans the list top-down and clicks the first attractive item, so only items up to the click are observed, and CascadeUCB1/CascadeKL-UCB use exactly that partial feedback. Ignoring position bias or the cascade structure gives biased estimates.
- **Block:** `experimental.bandit.RankingBandits` (CUCB, CASCADE_UCB1, CASCADE_KL_UCB, a position-based-model variant, and a linear contextual cascade version on `ContextualBandit`'s ridge models).
- **Experiment:** `experiments.bandits.RankingBandits`.
- **Toy task:** `ClickLogs`' cascade click model and a position-based model over 50–500 items with lists of 5–10.
- **Baselines:** `ContextualBandit` LinUCB and linear TS scoring items independently (ignoring the cascade); ranking by `ClickLogs`' unbiased learning-to-rank estimators (wave 9) trained offline on logged clicks; random lists.
- **Expected metric:** cumulative regret against the best list and click-through rate over 10⁴–10⁵ rounds. Hypothesis: cascade-aware policies beat item-wise LinUCB/TS when the click model is a cascade and lose some of the edge under the position-based model; offline ULTR stays behind the online policies.
- **Source:** [CUCB, ICML 2013](https://proceedings.mlr.press/v28/chen13a.html), [arXiv:1502.02763](https://arxiv.org/abs/1502.02763)

### 11.6.6 Deep exploration: bootstrapped DQN with randomised prior functions and RLSVI (Osband et al., NeurIPS 2016 and 2018; Osband et al., ICML 2016)
ε-greedy and noisy networks dither and need exponentially many steps on problems like Deep Sea, where reward lies at the end of a long chain of costly actions. Bootstrapped DQN trains K Q-heads on bootstrapped data and follows one randomly chosen head for a whole episode, which gives temporally consistent exploration like Thompson sampling; adding a fixed random "prior" network to each head keeps them diverse where data is absent. RLSVI is the linear version: it samples value-function parameters from a Bayesian linear-regression posterior every episode.
- **Block:** `experimental.explore.BootstrappedDqn` (K heads, Bernoulli masks, episode-level head sampling, randomised prior scale β) and `experimental.explore.Rlsvi` (one-hot and linear features), plus the `DeepSeaChain` environment.
- **Experiment:** `experiments.control.DeepExploration`.
- **Toy task:** `DeepSeaChain` N = 10–30, RiverSwim, and sparse-reward `KeyDoorGrid`.
- **Baselines:** core `Dqn` with ε-greedy; `RainbowDqn` with `NoisyDense`; `RandomNetworkDistillation` and `IntrinsicCuriosityModule` bonuses on DQN.
- **Expected metric:** episodes to first reward and to solving Deep Sea against N (the scaling exponent). Hypothesis: ε-greedy and NoisyNet fail beyond N = 12 within the budget; bootstrapped DQN with priors and RLSVI scale roughly polynomially; without priors the heads collapse and the gain shrinks.
- **Source:** [arXiv:1602.04621](https://arxiv.org/abs/1602.04621), [arXiv:1806.03335](https://arxiv.org/abs/1806.03335), [arXiv:1402.0635](https://arxiv.org/abs/1402.0635), [bsuite, arXiv:1908.03568](https://arxiv.org/abs/1908.03568)

### 11.6.7 Count-based exploration in deep RL: SimHash counts and density-model pseudo-counts (Tang et al., NeurIPS 2017; Bellemare et al., NeurIPS 2016)
A bonus β/√N(s) for rarely visited states drives exploration in tabular RL. #Exploration makes counts work in continuous or pixel spaces by hashing states with SimHash (signs of a random projection, optionally of a learned autoencoder code) and counting hash codes; pseudo-counts derive N̂(s) from how much a density model's probability of s increases after one more update on it. Both are simple baselines that RND and ICM are usually compared against.
- **Block:** `experimental.explore.CountBonuses` (SIMHASH with k bits, learned-code SimHash, PSEUDO_COUNT from a small PixelCNN or Gaussian density model), as reward wrappers for any agent.
- **Experiment:** `DeepExploration`.
- **Toy task:** sparse MountainCar, `KeyDoorGrid` with pixel observations from `PixelObservations`, and `DeepSeaChain`.
- **Baselines:** `RandomNetworkDistillation`, `IntrinsicCuriosityModule` and `NeverGiveUp` bonuses on the same agent; ε-greedy; tabular counts where states are discrete (the oracle).
- **Expected metric:** state coverage and episodes to first success. Hypothesis: SimHash matches RND on MountainCar and `KeyDoorGrid` at much lower cost; on pixel grids the learned-code version is needed; pseudo-counts are noisier than both.
- **Source:** [arXiv:1611.04717](https://arxiv.org/abs/1611.04717), [arXiv:1606.01868](https://arxiv.org/abs/1606.01868)

### 11.6.8 Unsupervised skill discovery: DIAYN, METRA and contrastive successor features (Eysenbach et al., ICLR 2019; Park et al., ICLR 2024; Zheng et al., ICLR 2025)
Without any reward, an agent can learn a set of distinguishable skills. DIAYN maximises the mutual information between a latent skill z and visited states with a discriminator q(z | s) as reward; METRA instead maximises a Wasserstein-style objective with a temporal-distance constraint, which favours skills that cover distant states; contrastive successor features (CSF, 2025) show that METRA's gain can be reproduced inside the mutual-information framework with a contrastive bound on (φ(s′) − φ(s))ᵀz, negatives from the skill prior, and vector-valued successor-feature critics.
- **Block:** `experimental.explore.SkillDiscovery` (DIAYN discriminator reward, METRA with dual-gradient constraint, CSF contrastive representation and successor-feature critic), on `ModernSac`.
- **Experiment:** `DeepExploration` (skills table).
- **Toy task:** a 2-D continuous point maze (from `GoalEnvironment`), `ContinuousMountainCar` and `KeyDoorGrid` with continuous positions.
- **Baselines:** SAC with random skills (no intrinsic reward); RND-driven SAC; the existing `SuccessorFeatures` with GPI for downstream reuse.
- **Expected metric:** state coverage (occupied grid cells) after pretraining, and downstream goal-reaching return after fine-tuning a skill selector. Hypothesis: DIAYN learns static, clustered skills; METRA and CSF cover several times more of the maze and transfer better to goal reaching; CSF matches METRA with fewer hyperparameters, as its paper reports.
- **Source:** [arXiv:1802.06070](https://arxiv.org/abs/1802.06070), [arXiv:2310.08887](https://arxiv.org/abs/2310.08887), [arXiv:2412.08021](https://arxiv.org/abs/2412.08021)

### 11.6.9 Model-based policy optimisation: short model rollouts from an ensemble of probabilistic dynamics (Janner et al., NeurIPS 2019)
PETS plans with an ensemble model at every step (the repo's `CemPlanner`). MBPO learns a policy instead: it trains an ensemble of probabilistic dynamics models on real data, branches k-step rollouts (k = 1–5) from real states using a random ensemble member, adds them to a model buffer, and trains SAC on a mix of real and model data at a high update ratio. Short branched rollouts keep model error from compounding while still multiplying the data.
- **Block:** `experimental.deeprl.Mbpo` (Gaussian dynamics ensemble with elite selection, branched rollouts with a length schedule, mixed replay into `ModernSac`).
- **Experiment:** `experiments.control.ModelBasedAndHierarchical`.
- **Toy task:** Pendulum, `ContinuousMountainCar` and continuous CartPole at 2k–10k environment steps.
- **Baselines:** `CemPlanner` (PETS-style MPC); `ModernSac` and `HighUtdSac` (REDQ/DroQ, the high-UTD model-free rivals); `TdMpc` (TD-MPC2) and `DreamerAgent`.
- **Expected metric:** return at equal environment steps and wall-clock per step. Hypothesis: MBPO beats SAC by 2–5x in sample efficiency on Pendulum and matches DroQ; with long rollouts (k = 10) it degrades; PETS is competitive on Pendulum only.
- **Source:** [arXiv:1906.08253](https://arxiv.org/abs/1906.08253)

### 11.6.10 Goal-conditioned hierarchies: HIRO and hierarchical actor-critic (Nachum et al., NeurIPS 2018; Levy et al., ICLR 2019)
Hierarchical RL lets a high-level policy set subgoals every c steps for a goal-conditioned low-level policy rewarded for reaching them. HIRO trains both levels off-policy and corrects the high level's stale transitions by relabelling the subgoal with the one that best explains the low level's actual actions; HAC uses hindsight action and goal transitions at every level so that all levels learn in parallel even when the levels below are still poor.
- **Block:** `experimental.explore.Hiro` (two-level TD3 with off-policy subgoal correction) and a HAC option (hindsight action transitions and subgoal testing), on `Td3` and `HindsightReplay`.
- **Experiment:** `ModelBasedAndHierarchical`.
- **Toy task:** a continuous U-maze and four-rooms from `GoalEnvironment` with distant goals; `KeyDoorGrid` with a continuous agent.
- **Baselines:** flat TD3 with HER (`HindsightReplay`); `OptionCritic`; `GoExplore` on the discretised maze.
- **Expected metric:** success rate against environment steps. Hypothesis: HIRO with correction solves the U-maze where flat TD3 + HER does not within the budget; without correction the high level stalls; HAC learns fastest on the shortest horizon.
- **Source:** [arXiv:1805.08296](https://arxiv.org/abs/1805.08296), [arXiv:1712.00948](https://arxiv.org/abs/1712.00948)

### 11.6.11 Preference-based RL from segment comparisons: Christiano-style reward learning, PEBBLE and SURF (Christiano et al., NeurIPS 2017; Lee et al., ICML 2021; Park et al., ICLR 2022)
Instead of a reward, a teacher compares pairs of trajectory segments. Christiano et al. fit a Bradley–Terry reward model to the comparisons and run RL on it; PEBBLE adds unsupervised pre-training (state-entropy maximisation) so the first queried segments are diverse, and relabels the whole replay buffer every time the reward model changes, which makes off-policy SAC usable; SURF adds semi-supervised pseudo-labels for unlabelled segment pairs and temporal-crop augmentation to save feedback.
- **Block:** `experimental.explore.PreferenceRl` (segment sampler with uniform, disagreement and entropy query selection, ensemble BT reward model from `PreferenceLosses`, buffer relabelling, unsupervised pre-training, SURF pseudo-labels).
- **Experiment:** `experiments.control.RewardFormulations`.
- **Toy task:** Pendulum and `ContinuousMountainCar` with a scripted teacher that compares true returns, with 0–20% label noise; 50–500 queries.
- **Baselines:** SAC on the true reward (the ceiling); `GoalsImitationAndTeams`' behaviour cloning and GAIL from 10 expert demonstrations; random query selection.
- **Expected metric:** true return against the number of queries. Hypothesis: PEBBLE reaches 80% of the true-reward return with a few hundred queries where uniform-query Christiano needs several times more; SURF halves the queries again; label noise hurts all methods roughly equally.
- **Source:** [arXiv:1706.03741](https://arxiv.org/abs/1706.03741), [arXiv:2106.05091](https://arxiv.org/abs/2106.05091), [arXiv:2203.10050](https://arxiv.org/abs/2203.10050)

### 11.6.12 Average-reward RL for continuing tasks: R-learning and differential TD/Q-learning (Schwartz, ICML 1993; Wan, Naik & Sutton, ICML 2021)
In continuing tasks the discount factor is a hack: it makes optimal policies depend on γ and can prefer short-term gains. Average-reward methods learn differential values relative to an estimated reward rate r̄; R-learning and differential Q-learning update r̄ from the TD error itself, which Wan et al. prove convergent for the tabular off-policy case, and the same update works with function approximation in practice.
- **Block:** `experimental.deeprl.AverageReward` (tabular and DQN-style differential Q-learning, differential TD for prediction, reward-rate step size η).
- **Experiment:** `RewardFormulations`.
- **Toy task:** continuing access-control queuing (a classic average-reward task), a continuing version of `GridWorld` with a recurring reward, and CartPole without termination penalty.
- **Baselines:** core `Tabular` Q-learning and `Dqn` at γ ∈ {0.9, 0.99, 0.999}; the exact average-reward optimum by relative value iteration on the tabular tasks.
- **Expected metric:** long-run reward rate and its gap to the optimum. Hypothesis: differential Q-learning reaches the optimal rate on the queuing task, where discounted Q-learning at γ = 0.9 converges to a worse policy and γ = 0.999 learns slowly.
- **Source:** [Schwartz 1993](https://doi.org/10.1016/B978-1-55860-307-3.50045-9), [arXiv:2006.16318](https://arxiv.org/abs/2006.16318)

### 11.6.13 Complete value factorisation: QTRAN, QPLEX and QFIX (Son et al., ICML 2019; Wang et al., ICLR 2021; Baisero et al., RLJ 2026)
VDN and QMIX satisfy the individual-global-max (IGM) property but cannot represent every IGM-consistent joint value, so they fail on games with non-monotonic payoffs such as the repo's `TwoStepGame`. QTRAN adds soft constraints that make a factorised value match the joint one at the greedy action; QPLEX uses a duplex dueling architecture with attention-based advantage weights that spans the full IGM class; QFIX (2026) gives a simple description of the full IGM class and extends VDN or QMIX with a small "fixing" layer that rescales the fixee's (non-positive) advantage with a learned positive weight and adds a history-dependent bias.
- **Block:** `experimental.marl.CompleteFactorization` (QTRAN_BASE, QPLEX, QFIX_SUM, QFIX_MONO), plugging into `ValueDecomposition`'s training loop through a new mixer interface (the existing `Mixer` values unchanged).
- **Experiment:** `experiments.multiagent.CooperativeMarl`.
- **Toy task:** `TwoStepGame`, non-monotonic matrix games (the climbing and penalty games), and `MultiAgentEnvironment` navigation with 3–5 agents.
- **Baselines:** `ValueDecomposition` INDEPENDENT, VDN and QMIX; `MultiAgentPpo` (MAPPO).
- **Expected metric:** return and the share of seeds reaching the optimal joint action. Hypothesis: VDN and QMIX miss the optimum on the climbing game and `TwoStepGame`; QPLEX and QFIX find it; QFIX trains more stably than QPLEX with a smaller mixer; on navigation all tie.
- **Source:** [arXiv:1905.05408](https://arxiv.org/abs/1905.05408), [arXiv:2008.01062](https://arxiv.org/abs/2008.01062), [QFIX, RLJ 2026](https://rlj.cs.umass.edu/2026/papers/Paper121.pdf)

### 11.6.14 Mean-field multi-agent RL (Yang et al., ICML 2018)
With many agents, the joint action space explodes. Mean-field Q-learning approximates each agent's Q-function as depending on its own action and the mean action of its neighbours, Q(s, a_i, ā_i), and updates it with a Boltzmann policy that is consistent with that mean; the mean-field actor-critic is the policy-gradient analogue. Both scale to hundreds of agents where independent learners see a non-stationary environment.
- **Block:** `experimental.marl.MeanFieldQ` (neighbourhood mean actions, Boltzmann consistency iteration) and `experimental.marl.MeanFieldActorCritic`.
- **Experiment:** `CooperativeMarl`.
- **Toy task:** a 2-D Ising-style coordination lattice (agents rewarded for aligning with neighbours) with 16–400 agents, and a crowd-navigation grid with 20–100 agents sharing a corridor.
- **Baselines:** `ValueDecomposition` INDEPENDENT Q-learners; `MultiAgentPpo` with parameter sharing; the exact Ising ground state for the lattice.
- **Expected metric:** order parameter (alignment) and return against the number of agents. Hypothesis: mean-field Q reaches near-full alignment at 400 agents where independent Q-learners plateau; at 16 agents MAPPO ties it.
- **Source:** [arXiv:1802.05438](https://arxiv.org/abs/1802.05438)

### 11.6.15 Zero-shot coordination: other-play (Hu et al., ICML 2020)
Self-play agents converge on arbitrary conventions (call the left lever "1") and fail with new partners that broke the symmetry differently. Other-play trains against copies of the policy transformed by random symmetries of the game (permutations of equivalent actions or messages), so the agent can only rely on conventions that survive every relabelling; it is the strongest symmetric baseline for coordination with unseen partners.
- **Block:** `experimental.marl.OtherPlay` (symmetry groups for discrete games, symmetrised training for policy-gradient and Q-learning agents, cross-play evaluation).
- **Experiment:** `CooperativeMarl` (coordination table).
- **Toy task:** the lever game of Hu et al. (ten levers, one with a unique payoff), `ReferentialGame`'s signalling game with permutable messages, and a small cooperative card game.
- **Baselines:** self-play with the same agents; `ReferentialGame`'s trained senders and receivers; `MultiAgentPpo` with independent seeds.
- **Expected metric:** self-play return and cross-play return between independently trained seeds. Hypothesis: self-play reaches the highest self-play score but cross-play near chance; other-play picks the unique lever and keeps cross-play close to its self-play score.
- **Source:** [arXiv:2003.02979](https://arxiv.org/abs/2003.02979)

### 11.6.16 Heterogeneous-agent trust-region learning: HAPPO and HATRPO (Kuba et al., ICLR 2022)
MAPPO with parameter sharing assumes interchangeable agents and updates all of them simultaneously, which can lower the joint return even when each agent's update looks good. HAPPO updates agents one at a time in a random order, and each agent's clipped objective uses the advantage reweighted by the probability ratios of the agents already updated in this round (the multi-agent advantage decomposition), which guarantees monotonic improvement of the joint return for the trust-region version.
- **Block:** `experimental.marl.Happo` (sequential random-order updates, ratio-product advantage weighting, HATRPO's KL-constrained variant), on `MultiAgentPpo`'s networks without parameter sharing.
- **Experiment:** `CooperativeMarl`.
- **Toy task:** heterogeneous-role tasks in `MultiAgentEnvironment` (a carrier and a scout with different action sets), the climbing game with continuous actions, and a two-agent coupled Pendulum.
- **Baselines:** `MultiAgentPpo` (MAPPO with and without sharing, and IPPO); `Maddpg`; 11.6.13's QPLEX on the discrete tasks.
- **Expected metric:** return and its variance across seeds. Hypothesis: HAPPO beats MAPPO with sharing on heterogeneous roles and matches it on homogeneous ones; its seed variance is lower than IPPO's.
- **Source:** [arXiv:2109.11251](https://arxiv.org/abs/2109.11251)

### 11.6.17 World models and control from pixels: pixel CartPole, a pixel MountainCar and continuous-control rows (Hafner et al., 2023; Ha & Schmidhuber, 2018) — follow-up
Wave 9 9.5.6 cut pixel CartPole from its world-model comparison, and wave 10 WP4 reported that the core environments have no pixel renderer and that `ContinuousMountainCar` was built but never reported. This item adds 32×32 renderers for CartPole and MountainCar to `PixelObservations` (as wrappers; the environments are unchanged), with frame stacking, and runs the world-model and pixel agents on them at test budgets. With state and pixel versions of the same task, the cost of learning from pixels becomes measurable.
- **Block:** CartPole and MountainCar renderers in `experimental.deeprl.PixelObservations` (new methods, unchanged defaults) and a pixel-input configuration of `DreamerAgent` and wave 10's `DrQ`.
- **Experiment:** `experiments.world.WorldModelShootout.pixels` (new method; wave 9 rows unchanged), plus `ContinuousMountainCar` rows in `experiments.control.ModernOffPolicy` (new method; wave 10 rows unchanged).
- **Toy task:** pixel and state CartPole; pixel MountainCar; `ContinuousMountainCar` from states.
- **Baselines:** `DreamerAgent` and DQN on states (the same tasks); wave 10's DrQ and SPR on the pixel grids; `ModernSac`, `Td7` and `CrossQ` on continuous MountainCar.
- **Expected metric:** return at equal environment steps, pixels against states, and the share of seeds that reach the goal on continuous MountainCar. Hypothesis: Dreamer from pixels reaches at least half of its state-based CartPole return at the test budget and beats pixel DQN; on continuous MountainCar only TD7 and SAC with an exploration bonus from 11.6.7 reach the goal reliably.
- **Source:** [DreamerV3, arXiv:2301.04104](https://arxiv.org/abs/2301.04104), [World Models, arXiv:1803.10122](https://arxiv.org/abs/1803.10122)

---

## Notes on dependencies between items

**Security and privacy (WP1)**
- First build `experimental.data.PoisonedSets` (it must keep clean labels and poison flags, and must not change `ToyTasks`, `ShapeImages` or `ClientSplits` sampling), then create `experimental.security`, `experimental.dpmech` and the theme `experiments.security`.
- 11.1.1 → 11.1.3 share `RobustAggregation` and one federated loop that calls `FederatedAlgorithms`' client step unchanged; build 11.1.3's attacks together with 11.1.1, since every aggregator row needs them. 11.1.3's model-replacement attack reuses 11.1.4's trigger code.
- 11.1.4 → 11.1.6 share `BackdoorShootout` and one cached poisoned model per attack and seed. 11.1.5 uses `ThinSvd`, `KMeans` and `ComponentAnalysis` unchanged; 11.1.6 uses `LatentAdversarialTrainer` read-only.
- 11.1.7 → 11.1.10 share `AttackShootout` and a query-counting logits oracle; 11.1.10's AutoAttack reuses 11.1.7's Square attack, so build 11.1.7 first. 11.1.10's re-evaluation adds a method to `AdversarialRobustness` and leaves wave 8's rows unchanged.
- 11.1.13 comes before 11.1.14 (MWEM selects queries with the exponential mechanism). 11.1.15 reuses `DpSgd`'s clipping and accountant; the DP-FTRL rows go in `PrivateTraining` as new methods.
- 11.1.16 adds classes next to `MachineUnlearning` and rows to `UnlearningComparison`; wave 5 rows must stay reproducible. S3T reuses `LoraAdapterDense` unchanged.
- 11.1.17 uses core `Augmentation`'s operations read-only and must keep the test corruptions out of the training augmentations.

**Combinatorial optimisation and algorithms (WP2)**
- First build `experimental.data.CombinatorialInstances` (with exact optima from Held–Karp and B&B over `LinearProgram`) and `KeyStreams`, `ClassicHeuristics` and `ClassicCaches`, and create `experimental.combopt`, `experimental.sketch` and the themes `experiments.combinatorial` and `experiments.algorithms`.
- 11.2.1 → 11.2.7 share `RoutingEnv` and `RoutingShootout`. Build 11.2.2's `AttentionModel` before 11.2.3–11.2.5, which build on it; 11.2.4 builds on 11.2.3's multi-start rollouts. 11.2.1 wraps `PointerNetwork` without changing it. 11.2.7 reuses `EntropicTransport`'s Sinkhorn and `MinibatchOptimalTransport`'s Hungarian solver (import them, do not copy).
- 11.2.8 → 11.2.9 share `GraphCoShootout` and the QUBO definitions; they read `SimulatedBifurcation`, `PBitIsingMachine` and the wave 6 `GFlowNet` unchanged. 11.2.10 → 11.2.11 share `SolverLearning`; 11.2.11 reads `RandomSatFormulas`, `NeuroSat` and the WalkSAT in `GraphReasoning` unchanged (expose WalkSAT as a public method if it is private, rather than copying it).
- 11.2.12 → 11.2.13 add classes to `experimental.evolve` and `Method` rows to `ProgramSynthesisShootout`; 11.2.13 builds on 11.2.12's `BottomUpSynthesis`. The existing rows and their numbers must stay unchanged.
- 11.2.14 comes before 11.2.15 (the learned sketches compare at equal memory with the classical ones). 11.2.16 and 11.2.17 are independent of each other.

**Causal discovery, quasi-experiments and statistics (WP3)**
- First build `experimental.data.DiscoveryBenchmarks` and `PanelStudies` and create `experimental.discovery`, `experimental.panel`, `experimental.circuits` and the themes `experiments.discovery` and `experiments.econometrics`.
- 11.3.1 → 11.3.4 share `DiscoveryShootout`, one CI-test interface and the SHD/orientation metrics; 11.3.1's PC is the baseline for 11.3.2–11.3.3, so build it first. `Notears`, `Dagma`, `GreedyEquivalenceSearch` and `KernelTests` are used unchanged. 11.3.5 is independent and reads `ChaoticSeries` and `Sindy` unchanged.
- 11.3.6 → 11.3.7 share `PanelShootout` and the TWFE reference; 11.3.6's SDID is a baseline of 11.3.7. 11.3.8 is independent. 11.3.9 → 11.3.10 share `ObservationalAdjustment`; 11.3.10 runs on 11.3.9's weighted estimates. `CausalEstimators`, `DoubleMl` and `LinearProgram` are used unchanged.
- 11.3.11 comes before 11.3.12 (same tensorised layers). 11.3.13 adds a public `GaussianMixture.posterior`/`logLikelihood` first (wave 10 WP1's `SampleSelection` may then switch to it in a later PR; do not change it in this wave) and adds rows to `NonparametricClustering` without changing existing ones.
- 11.3.14 changes `StateSpaceFilter` only by accepting NaN-masked observations (existing calls behave identically) and adds a row to `IrregularSeriesShootout` that replaces wave 10's private smoother for the new row only. 11.3.15 and 11.3.16 are independent; 11.3.16 adds an option to `SeasonalSeries` with unchanged defaults.
- 11.3.17 adds `experimental.survival.Weibull`; `SurvivalData`, `OpenMax` and `PointProcessShootout` keep their private copies in this wave (their numbers must not move), and new code uses the helper.

**Uncertainty, ensembles and classic models (WP4)**
- First build the shared fixture of cached trained models and logits (digits, `ShapeImages`, tabular; one per seed) and create `experimental.calibrate`, `experimental.ensemble` and the theme `experiments.uncertainty`.
- 11.4.1 comes before 11.4.2 and 11.4.3 (common `fit`/`apply` interface and metrics from `CalibrationEstimators`, used read-only). 11.4.4 → 11.4.5 share `ConformalSets` and call `ConformalClassifier`'s scores unchanged; 11.4.5's sets are calibrated with 11.4.4's code.
- 11.4.6, 11.4.7, 11.4.9, 11.4.10, 11.4.14 and 11.4.16 share `EnsembleShootout` and one `Member` interface; build 11.4.16's adapters for the existing blocks (`PopulationEnsemble`, `WeightPosterior`, `BatchEnsembleDense`, `TabM`, `DistanceAwareHead`, `EvidentialHead`) early, read-only. 11.4.10 uses wave 10's `CausalAndFairness.colouredShapes` accessor and `GroupRobustTraining` unchanged. 11.4.8 is independent.
- 11.4.11 adds rows to `CompactModels`; 11.4.12 → 11.4.13 share `ClassicLearners`; 11.4.12 reads `AnnealedImportanceSampler` and `DenoisingThermodynamicModel` unchanged. 11.4.17 adds a method to `ClassicMlShootout`.

**Perception and language (WP5)**
- First build `ToyMusic` (it must reuse `HarmonicNoiseSynth` and `SpeakerVoices` synthesis unchanged), `ParallelText`, `DefectImages` and `ShapeClips`, and create `experimental.music`, `experimental.nlp` and the theme `experiments.music`.
- 11.5.1 comes before 11.5.2 (shared metrics and F0 ground truth). 11.5.3 → 11.5.4 share `RhythmAndHarmony` and the CQT/chroma front end; 11.5.4 uses `HiddenMarkovModel` and `LinearChainCrf` unchanged. 11.5.5 → 11.5.6 share `AudioRetrieval`; 11.5.5 uses `Mfcc`, `DynamicTimeWarping`, `KnnIndex` and `ContrastivePairLoss` unchanged.
- 11.5.7 → 11.5.9 share `ClassicNlp`. 11.5.8's `WindowLm` `LogitsProcessor` hook must leave default sampling bit-identical (wave 10's `WatermarkShootout` keeps its local sampler; switching it is a later clean-up). 11.5.10 reuses `ChartParser`'s chart and `ToyTreebank` unchanged.
- 11.5.11 comes before 11.5.12 (shared metrics and `DefectImages` splits); both read `SelfSupervisedBenchmark.pretrainEncoder` (exposed in wave 10) without changing the benchmark. 11.5.13–11.5.16 are independent.
- Build 11.5.17's public `Conv2dOps` and `Fft2d` early, because 11.5.11–11.5.16 can use them; existing classes keep their local helpers in this wave so that their numbers do not move.

**RL beyond the wave 10 recipes (WP6)**
- First build `BanditSuites` and `DeepSeaChain` and create `experimental.bandit`, `experimental.explore`, `experimental.marl` and the theme `experiments.bandits`. RL items report through wave 10's `LearningCurveHarness` and `EvalStatistics`, used read-only.
- 11.6.1 → 11.6.5 share the `BanditPolicy` interface; 11.6.1 is the baseline set for 11.6.2–11.6.4. 11.6.5 reuses `ClickLogs`' click model and `ContextualBandit`'s ridge models unchanged.
- 11.6.6 → 11.6.8 share `DeepExploration`; 11.6.7's bonuses are reused in 11.6.17's continuous MountainCar row. 11.6.9 → 11.6.10 share `ModelBasedAndHierarchical`; 11.6.9 uses `ModernSac` and 11.6.10 uses `Td3` and `HindsightReplay` unchanged. 11.6.11 → 11.6.12 share `RewardFormulations`; 11.6.11 reuses `PreferenceLosses`' Bradley–Terry loss.
- 11.6.13 → 11.6.16 share `CooperativeMarl`. 11.6.13 adds a mixer interface next to `ValueDecomposition.Mixer` without changing the existing values or `GoalsImitationAndTeams`' rows. 11.6.15 reads `ReferentialGame` unchanged.
- 11.6.17 adds renderers to `PixelObservations` as new methods and new methods to `WorldModelShootout` and `ModernOffPolicy`; wave 9 and wave 10 rows must stay reproducible.

**Cross-package**
- The fourteen new packages (`experimental.security`, `.dpmech`, `.combopt`, `.sketch`, `.discovery`, `.panel`, `.circuits`, `.calibrate`, `.ensemble`, `.music`, `.nlp`, `.bandit`, `.explore`, `.marl`) and the new generators in `experimental.data` (`PoisonedSets`, `CombinatorialInstances`, `KeyStreams`, `DiscoveryBenchmarks`, `PanelStudies`, `ToyMusic`, `ParallelText`, `DefectImages`, `ShapeClips`, `BanditSuites`) have no dependencies on each other, so all six work packages can start at once. Each new package and generator name belongs to exactly one WP.
- Existing packages that two WPs add classes to, without touching each other's classes: `experimental.ml` (WP3: `VariationalGaussianMixture`, `LinearDynamicalSystemEm`, `GaussianMixture`/`StateSpaceFilter` changes; WP4: `RestrictedBoltzmannMachine`, `TsetlinMachine`). Every other existing package gets additions from one WP only: `.robust` and `.train` (WP1), `.evolve` (WP2), `.timeseries` and `.survival` (WP3), `.nn` (WP4), `.parse`, `.anomaly`, `.vision` and `.llm`'s `WindowLm` hook (WP5), `.deeprl` (WP6).
- Cross-WP soft links, each with a local stand-in until the other lands: WP4 11.4.12 compares with WP3 11.3.11's Bernoulli-mixture leaves (use `GaussianMixture`-style EM locally until then); WP5 11.5.11–11.5.16 may use WP5's own 11.5.17 helpers; WP6 11.6.3's hyperparameter use reads core `HyperparameterSearch` only. No WP needs another WP's block to produce its verdict.
- Shared helpers to expose rather than copy: `GaussianMixture.posterior` (WP3 11.3.13), missing observations in `StateSpaceFilter` (WP3 11.3.14), `experimental.survival.Weibull` (WP3 11.3.17), the `WindowLm` logits hook (WP5 11.5.8), `Conv2dOps`/`Fft2d` (WP5 11.5.17), the CartPole/MountainCar renderers (WP6 11.6.17), and the WalkSAT in `GraphReasoning` (WP2 11.2.11). Whichever WP needs it owns the change, with unchanged defaults.
- Edits to shared experiment classes, each owned by exactly one WP in this wave: `AdversarialRobustness`, `PrivateTraining`, `UnlearningComparison` (WP1); `ProgramSynthesisShootout` (WP2); `NonparametricClustering`, `IrregularSeriesShootout`, `SurvivalShootout` (WP3); `CompactModels`, `ClassicMlShootout` (WP4); `WorldModelShootout`, `ModernOffPolicy` (WP6). WP5 edits no existing experiment class. No other WP edits them.
- Read-only by several WPs: core `Augmentation` (WP1 11.1.17, WP4 11.4.9), `LinearProgram` (WP2 11.2.10, WP3 11.3.6/11.3.9), `KnnIndex` (WP5 11.5.5/11.5.11), `PreferenceLosses` (WP2 11.2.4, WP6 11.6.4/11.6.11), `ClickLogs` (WP6 11.6.5), `ThinSvd` (WP1 11.1.5, WP2 11.2.15, WP3 11.3.6).
- Test budget: per the user's current rule, every hypothesis above is stated at the ~10 s test budget and no item's verdict depends on running a `main()`; builders run only their own test classes, one or two at a time.
- Core-worthy items, to land in a separate PR from the experiments once measured: 11.1.1–11.1.2 robust aggregation (`brainiac-train` federated server step), 11.1.12–11.1.13 local DP and private selection (`brainiac-train` privacy utilities), 11.2.14 sketches (`brainiac-data`), 11.3.14 missing-aware Kalman smoothing, 11.4.1–11.4.2 recalibration maps and 11.4.4 RAPS/Mondrian sets (`brainiac-train` `Evaluation`), 11.4.11 binary layers with the popcount kernel (`brainiac-nn`/`brainiac-tensor`), 11.5.17 the 2-D FFT and public convolution helpers (`brainiac-tensor`), 11.6.1 KL-UCB/Thompson bandit agents (`brainiac-rl`), and 11.6.17 pixel renderers for the core environments (`brainiac-rl`).
