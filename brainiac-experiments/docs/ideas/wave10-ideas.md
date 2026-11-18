# Brainiac: wave 10 ideas

*Compiled 2026-11-18 against `claude/project-thread-8zvcl0` (main tree, which now holds waves 1–9), plus every worktree under `.claude/worktrees/*` (48 worktrees). Waves 1–9 implemented about 1,000 techniques; see `experiment-results.md` (waves 1–8), the wave 9 builders' reports, `unimplemented-ideas.md` and `wave5-ideas.md` to `wave9-ideas.md`. A survey of the `experiments.*` themes shows which areas are still thin: `selfsupervised` (2 classes), `robustness` (3), `privacy`, `multiagent` and `synthesis` (3 each), and `learning` and `memory` (4–6). Several whole fields have no code at all: semi-supervised and noisy-label learning, out-of-distribution detection and domain adaptation, metric learning, deep clustering, multi-task gradient balancing, temporal point processes, temporal and relational graphs, parsing, image restoration and speech analysis. Wave 10 targets those areas and avoids the areas wave 9 has just covered, except for follow-ups:*

- *(a) learning from imperfect supervision: semi-supervised learning, noisy labels, long-tailed classes, positive-unlabelled data, weak supervision, bags and label proportions;*
- *(b) distribution shift and the open world: post-hoc out-of-distribution scores, open-set recognition, selective prediction and deferral, label-shift estimation, domain adaptation and generalisation, spurious correlations, and text watermarking;*
- *(c) representation learning: self-supervised methods not yet built (including LeJEPA's SIGReg), deep metric learning, deep clustering, multi-task gradient balancing, and a follow-up on contrast-consistent search;*
- *(d) deep RL recipes from 2023–2025 (TD7, CrossQ, BRO, SimbaV2, PQN, MR.Q, BBF, plasticity fixes, scaling studies) and follow-ups of the wave 8–9 RL and post-training negatives;*
- *(e) classic and structured ML: boosting variants, temporal point processes, irregular series and imputation, temporal, relational and hyperbolic graph models, and dependency and constituency parsing;*
- *(f) image restoration and speech analysis, plus the wave 9 perception and systems follow-ups (DETR, Donut, VQA bridges, Transfusion, once-for-all, pipeline memory, the GAN vocoder).*

*Twelve items are fair-test follow-ups of earlier negatives, marked "— follow-up" in their titles: 10.3.17 (CCS, R-wave9), 10.4.13 (GAIL, first-run negatives table), 10.4.14 (continuous control at 400 steps, first-run negatives table), 10.4.15 (reward-model ensembles and WARM, R-wave9), 10.4.16 (VinePPO, PRIME and Step-DPO, R-wave9), 10.6.10 (DETR on scenes, R-wave8/9), 10.6.11 (Donut, R-wave9), 10.6.12 (VQA bridges at the answer prior, R-wave9), 10.6.13 (Transfusion against VQ tokens, R-wave9), 10.6.14 (once-for-all supernets, R-wave9), 10.6.15 (ZB-H2 never beating ZB-H1, R-wave9) and 10.6.16 (the untrained GAN vocoder, R-wave9). Two items also lift wave 7/8 cuts: speaker verification (cut twice because the formant synthesiser had too few speaker factors; 10.6.5 adds a richer voice generator) and temporal graph networks (cut twice for lack of a fair baseline; 10.5.12 adds EdgeBank and a controlled probe suite).*

*It also adds the late-2024 to 2026 items found by a web sweep, each checked against the paper's abstract or text rather than a search snippet: LeJEPA/SIGReg (Nov 2025), JEPAMatch (2026), Joint Asymmetric Loss (ICCV 2025), multiplicative logit adjustment (ICLR 2025), AdaSCALE (2025/2026), likelihood-ratio selective classification (2025), online Bayesian label-shift estimation with Dirichlet priors (Nov 2025), 2-HNC for PU learning (2025), GradOPS (WSDM 2025), SimbaV2 (ICML 2025), MR.Q (ICLR 2025), PQN (ICLR 2025), multi-skip residuals with Kron for RL (2025), TD-overfitting and compute-optimal value-based RL (NeurIPS 2025), PFN-Boost (2025), the flexible-kernel Hawkes re-evaluation of neural TPPs (NeurIPS 2026), HyperEvent and "What do temporal graph models learn?" (2025), Median2Median (2025), and V-shape controllable-memory pipelines (NeurIPS 2024).*

*Every item was grepped over all `*.java` sources under `src/` in all modules, in the main tree and in all 48 worktrees: 82,384 files, 3,913 unique by content after de-duplication (3,041 distinct file names). The search was case-insensitive and used class names, paper names, author-coined terms and javadoc keywords. An item is listed only when nothing matched, or when the only matches were unrelated. Examples of unrelated matches:*

- *"mean teacher" and "MoCo" matching only the javadoc of `MomentumTeacher` (the EMA helper used by BYOL/DINO/I-JEPA; there is no semi-supervised consistency training and no MoCo queue); "pseudo-label" matching `SelfCertaintyReward` (an RL reward); "RandAugment" correctly matching core `brainiac-data` `Augmentation` (reused as FixMatch's strong augmentation, not re-implemented); "symmetric cross-entropy" matching `AsymmetricReplay`; "LDAM" matching "ldam"-substrings in `DifferentiableSim` and `HessianFree`; "long-tail" matching Zipf generators; "decoupled classifier/cRT" matching "cer"-like substrings in `GlyphOcrTest`.*
- *"ODIN" matching "coding"/"encoding" everywhere; "energy" matching `JointEnergyModel`, `EnergyMatching` and `EnergyRefiner` (generative energies, no energy OOD score); "ReAct" matching "reactivate"/"react" substrings; "out-of-distribution" correctly matching `UncertaintyBenchmark` and `DistanceAwareHead` (SNGP, DUQ, SWAG and Bayes-by-backprop OOD AUROCs, the baselines of 10.2.1–10.2.4); "abstain" correctly matching `PopulationEnsemble.predictOrAbstain` (a threshold on ensemble uncertainty, used as the 10.2.6 baseline); "defer" matching Vulkan deferred work and `SpeculativeCascade`'s deferral rule (a decoding cascade, not learning to defer); "label shift" matching one `package-info` sentence in `experimental.robust`; "covariate shift" matching `ClientSplits` and `TestTimeAdaptation` (no density-ratio weighting); "TENT" matching every "attention"/"content" (the real `TestTimeAdaptation` TENT is a baseline); "SAR" matching `NeatActivation`; "MMD" correctly matching `KernelTests` (two-sample tests, not a DA loss); "CORAL" matching "chat route"-substrings.*
- *"SwAV" matching "swap" substrings in `ChatServiceTest`; "prototypes" correctly matching DINO's prototype head in `SelfSupervisedBenchmark` (no Sinkhorn swapped prediction); "stop-gradient" matching VQ, MeanFlow and GRPO code (no SimSiam); "triplet" matching one comment in `RelationalKdLoss`; "InfoNCE" correctly matching `Functional.infoNce`, `SelfSupervisedLosses` and `ContrastivePairLoss` (CLIP/SigLIP; the 10.3.7–10.3.10 baselines); "angular" matching linear algebra; "DEC"/"IDEC" matching "decoder"/"JacobiDecoding"; "IIC"/"SCAN" matching `Dbscan`; "GradNorm" matching gradient-norm variables in `TrainingStabilityExperiment`, `ZeroCostProxies` and the RL trainers (no GradNorm loss balancing); "min-norm" matching quantisation code; "Proxy" matching reward proxies in `RewardOveroptimization` and `DataCuration`.*
- *"target network" correctly matching `Dqn`, `Td3` and `TurnLevelCritic` (the baselines that PQN and CrossQ remove them from); "dormant" matching `Brain`/`Plasticity` neurogenesis (a non-RL brain simulation); "LERP/unit sphere" matching model merging (SLERP); "Kron" correctly matching `PsgdKron` (reused by 10.4.11, not rebuilt); "BatchRenorm" correctly matching `experimental.nn.BatchRenorm` (reused by 10.4.2); "two-hot" correctly matching `DreamerAgent` (wave 9 symlog/two-hot, reused by MR.Q); "SPR" matching `LogitsProcessors`; "AIRL" matching "repAIRLoop"; "IQ-Learn" matching nothing but `MambaLayer` substrings; "value prefix" matching `HookedDecoder`; "BRO" matching one word in `FairnessInterventions`; "Riemannian" matching `Ivon`'s optimiser correction (no hyperbolic embeddings).*
- *"XGBoost" correctly matching core `GradientBoostedTrees` (second-order histogram boosting with quantile bins, so 10.5.1 adds only GOSS, EFB and leaf-wise growth); "histogram" matching calibration and graph code; "oblivious tree" correctly matching `ObliviousTreeEnsemble` (NODE's differentiable soft trees, not CatBoost's hard symmetric trees with ordered statistics); "natural gradient" matching PINN/TRPO code (no NGBoost); "shape function" correctly matching `NeuralAdditiveModel` (the 10.5.4 baseline); "intensity" matching diffractive optics; "irregular" matching `NeuralCde`, `LatentSde` and `OperatorLearning` (NeuralCde is the 10.5.10 baseline; there is no GRU-D, ODE-RNN or mTAN); "imputation" matching `MetaLearners` (X-learner effect imputation); "Bellman–Ford" correctly matching `GraphReasoning`'s CLRS traces (algorithm imitation, not NBFNet link prediction); "Edmonds" matching the Edmonds–Karp max-flow inside `AttentionRollout` (no Chu–Liu–Edmonds arborescence); "spanning tree" matching `PersistentHomology`; "span" matching `SpanCorruption`; "hyperbolic" matching `tanh`/`cosh` ops.*
- *"super-resolution/sub-pixel" correctly matching core `PixelShuffle` (reused as ESPCN's upsampler) and otherwise detection heads; "inpaint" matching video/action inpainting; "unroll" matching MuZero and V-trace unrolls; "speaker" correctly matching `SpokenDigitRecognition.Speaker` (two factors: formant scale and rate) and `SourceSeparation`'s two-speaker mixtures (no verification, VAD or diarisation); "ratio mask" correctly matching `SourceSeparation`'s ideal ratio mask (an oracle row, reused); "permutation invariant" correctly matching `PermutationInvariantLoss` (reused by EEND); "sandwich" correctly matching `ElasticSupernet` (wave 9 already trains the largest net every other step, so 10.6.14 adds in-place distillation and budget, not the sandwich rule); "progressive shrinking" correctly matching `ElasticSupernet`; "LLaVA" correctly matching core `VisionLanguageModel` and wave 9 `PatchProjector` (inference and a one-stage bridge; no two-stage alignment-then-instruction recipe); "tournament" matching evolutionary tournament selection (no SynthID tournament sampling); "watermark" correctly matching `GreenListWatermark` (the 10.2.17 baseline).*

## How this was checked

- **Already implemented, so left out.** These turned up in the systematic check but already have code:
  - Imperfect labels and shift: RandAugment and the other image augmentations (core `Augmentation`), the EMA `MomentumTeacher`, active learning (core-set, BADGE), EL2N/forgetting scores, label smoothing (`SmoothedCrossEntropy`), focal loss, class-incremental learning (LwF, iCaRL, DER++, GEM/A-GEM), test-time adaptation with BN statistics, TENT and MEMO (`TestTimeAdaptation`), IRM, V-REx, GroupDRO and JTT (`GroupRobustTraining`), conformal classification and forecasting (CQR, ACI, EnbPI), SNGP, DUQ, SWAG, Bayes by backprop, evidential heads, deep ensembles, MC dropout, last-layer Laplace, temperature scaling, the Kirchenbauer green-list watermark.
  - Representations: SimCLR, BYOL, DINO, Barlow Twins, VICReg, MAE, I-JEPA (`SelfSupervisedBenchmark`, `SelfSupervisedLosses`), CLIP/SigLIP (`ContrastivePairLoss`), prototypical, matching and relation networks, k-means, GMM, DBSCAN, spectral clustering, DP mixtures, SOM, growing neural gas, CCS and LEACE (wave 9), MMoE/PLE (wave 9).
  - RL: DQN, Rainbow, IQN, R2D2, PPO, TRPO, A3C, IMPALA, SAC, TD3, REDQ/DroQ (`HighUtdSac`), MPO, CQL, IQL, flow Q-learning, decision transformer, behaviour cloning, GAIL, MaxEnt IRL, Go-Explore, RND, ICM, NGU/Agent57, MuZero-lite, Gumbel MCTS, AlphaZero, PlaNet, Dreamer with the V3 tricks, TD-MPC2, DIAMOND, SimBa with HL-Gauss (`ResidualRlNetworks`, `HlGaussValue`), streaming actor-critic, PSGD Kron.
  - Classic and structured: second-order histogram GBDT (core `GradientBoostedTrees`), random forests, isolation forest, regression forest, kernel SVM, GPs, NAM, NODE, TabPFN-style `PriorFittedNetwork`, TabICL-lite, HMMs and linear-chain CRFs, seq2seq with attention and pointer networks, NeuralCde, latent SDEs, GCN/GAT/SAGE/GIN, GraphGPS, PNA, knowledge-graph embeddings (TransE, DistMult, ComplEx, RotatE), CLRS-lite, DiGress, survival models and causal estimators (wave 9).
  - Perception and systems: U-Net, DeepLab ASPP, core `PixelShuffle`, MFCC/mel front ends, SincNet/LEAF/PCEN, Conv-TasNet and DPRNN with permutation-invariant training, keyword spotters, CenterNet, RetinaNet, DETR/DN-DETR/H-DETR (`SetPredictionDetector`), Donut-style `ImageToMarkup`, VQA bridges (projector, Q-Former, gated cross-attention), Transfusion, once-for-all `ElasticSupernet` with the sandwich rule, GPipe/1F1B/interleaved/ZB-H1/ZB-H2/DualPipe (`PipelineSchedule`), Griffin–Lim, the HiFi-GAN-style `GanVocoder` and DiffWave/WaveGrad.
- **Considered but cut** (to stay near 6 × 17, or too close to an existing block):
  - GCG adversarial suffixes and refusal-direction ablation: the toy LMs have no refusal behaviour worth attacking, and `PersonaSteering` already ablates a learned trait direction.
  - RAG pipelines (HyDE, Self-RAG, CRAG): the toy LMs cannot write useful hypothetical documents; `KnnLm`, `ChunkedCrossAttention` and the retrieval shootout cover retrieval-augmented modelling.
  - Heterogeneous-graph attention (HAN, HGT): too close to 10.5.13's relation-typed message passing on toy graphs.
  - Monocular depth and colourisation: no renderer with depth or colour ground truth beyond wave 8's stereo task.
  - DivideMix as a separate item: folded into 10.1.8 as a row (it needs co-teaching's two networks and MixMatch, both in WP1).
  - AdaBN: the BN-statistics row of `TestTimeAdaptation` is already AdaBN at test time; it appears only as a baseline in 10.2.10.
  - Dawid–Skene crowd aggregation as its own item: folded into 10.1.14.
  - Neural ODE time-series beyond latent ODE (ODE-LSTM, CRU): close to 10.5.10's rows and `NeuralCde`.
  - Hypergraph networks: still no fair toy with a baseline.
  - Another PFN pre-training budget sweep: wave 9 added `TabularShootout.pretrainingBudget`; its 16x/64x rows only need to be run.
- **CPU-toy filter.** Every item is pure Java with no external libraries, runs as a test class in about 10 s on a CPU (larger budgets go in the experiment's `main` or a separate "Run"), and names a task that exists in `brainiac-experiments` or a tiny new generator next to it, plus a baseline already in the repo. The tasks used are:
  - `ToyTasks` digits/spirals, `ShapeImages`, `ShapeScenes`, `DigitGlyphs`, `GlyphDocuments`, `ToyImages`, `TabularPrior`, `TabularStressSuite`, `SyntheticRecsys`, `ClickLogs`, `SyntheticGraphs`, `FamilyTreeGraph`, `SeasonalSeries`, `ChaoticSeries`, `DriftingStreams`.
  - `SyntheticCorpus`, `SyntheticGrammar`, `StepChainTask`, `ArithmeticTask`, the toy LMs (`TinyLm`, `WindowLm`).
  - The `brainiac-rl` environments (CartPole, Pendulum, MountainCar, GridWorld) and the `experimental.rl` grids (`KeyDoorGrid`, `ObjectGrid`, `FrameStackEnvironment`), plus `OfflineDataset`.
  - `TalkingDigits`/`SpokenDigitRecognition`'s formant synthesiser and speakers.
  - New small generators named per work package: `LabelCorruption` (WP1), `ShiftSuites` (WP2), `EventStreams`, `IrregularSeries`, `TemporalGraphs` and `ToyTreebank` (WP5), `DegradedImages` and `SpeakerVoices` (WP6).

**Format.** Wave 10 keeps wave 9's structure (checks, summary, one section per work package, dependency notes), with each item as a short block:

- **Block**: the reusable class, in `io.github.zzuegg.brainiac.experimental.<pkg>`.
- **Experiment**: the runnable measurement, in `io.github.zzuegg.brainiac.experiments.<theme>`.
- **Toy task**, **Baselines** (all already in the repo) and **Expected metric** (the number to report, with the hypothesis to test).

**Where an item should live:**

- `experimental.<pkg>`: a new block in `brainiac-experiments`. Thirteen new packages are proposed, each owned by one work package so that no two agents create the same package (none of these names exists yet):
  - WP1: `experimental.semisup` (semi-supervised training) and `experimental.labels` (noisy, long-tailed, PU, weak, bag and proportion labels).
  - WP2: `experimental.ood` (OOD scores, open-set, selective prediction, deferral) and `experimental.shift` (label shift, domain adaptation and generalisation, density ratios). The watermark item adds to the existing `experimental.llm` next to `GreenListWatermark`.
  - WP3: `experimental.ssl`, `experimental.metric` and `experimental.multitask` (deep clustering adds to the existing `experimental.ml`; the CCS follow-up adds to the existing `experimental.interp`).
  - WP4: `experimental.deeprl` (the 2023–2025 off-policy and value-based recipes, plasticity and scaling tools). TD3+BC and the imitation follow-up add to the existing `experimental.rl`; the post-training follow-ups add to the existing `experimental.align`.
  - WP5: `experimental.boost`, `experimental.events` and `experimental.parse` (graph models add to the existing `experimental.graph`).
  - WP6: `experimental.restore` and `experimental.speech` (follow-ups add to the existing `experimental.vision`, `.document`, `.multimodal`, `.edge`, `.parallel` and `.audio`).
  - Existing blocks are not moved: `MomentumTeacher`, `TestTimeAdaptation`, `GroupRobustTraining`, `SelfSupervisedLosses`, `ContrastivePairLoss`, `ResidualRlNetworks`, `GradientBoostedTrees`, `NeuralCde`, `KnowledgeGraphEmbedding` and the others stay where they are, and the new packages import them.
  - New data generators go in `experimental.data` under the names listed above; each name belongs to one WP.
- `experiments.<theme>`: a new or extended runnable experiment. New themes: `experiments.labels` (WP1), `experiments.shift` (WP2), `experiments.metric` and `experiments.multitask` (WP3), `experiments.events` and `experiments.parsing` (WP5), `experiments.restoration` and `experiments.speech` (WP6). Other items extend `experiments.selfsupervised`, `experiments.classic`, `experiments.control`, `experiments.alignment`, `experiments.reasoning`, `experiments.interp`, `experiments.graphs`, `experiments.timeseries`, `experiments.language`, `experiments.vision`, `experiments.document`, `experiments.multimodal`, `experiments.edge`, `experiments.parallel` and `experiments.audio`.
- **core-worthy**: belongs in a core module once measured.

## Summary

| # | Work package | Items |
|---|---|---|
| WP1 | Learning from imperfect supervision: semi-supervised learning, noisy labels, long-tailed classes, PU, weak and bag-level labels | 17 |
| WP2 | Distribution shift and the open world: OOD detection, open-set recognition, selective prediction and deferral, label shift, domain adaptation and generalisation, watermarking | 17 |
| WP3 | Representation learning: self-supervised follow-ups and LeJEPA, deep metric learning, deep clustering, multi-task gradient balancing | 17 |
| WP4 | Deep RL recipes 2023–2025, plasticity and scaling, and RL/post-training follow-ups | 17 |
| WP5 | Classic and structured ML: boosting variants, temporal point processes, irregular series and imputation, temporal/relational/hyperbolic graphs, parsing | 17 |
| WP6 | Image restoration and speech analysis, plus the wave 9 perception and systems follow-ups | 16 |
| | **Total** | **101** |

---

## WP1. Learning from imperfect supervision: semi-supervised learning, noisy labels, long-tailed classes, PU, weak and bag-level labels (17)

Target: two new packages, `experimental.semisup` (10.1.1–10.1.6) and `experimental.labels` (10.1.7–10.1.17), and a new theme `experiments.labels` with `SemiSupervisedShootout` (10.1.1–10.1.6), `NoisyLabelShootout` (10.1.7–10.1.10), `LongTailShootout` (10.1.11–10.1.12) and `WeakLabelShootout` (10.1.13–10.1.17). One new generator in `experimental.data`: `LabelCorruption`, which wraps `ToyTasks` digits, `ShapeImages` and `TabularStressSuite` and produces (a) labelled/unlabelled splits with 4–40 labels per class, (b) symmetric, pair-flip and instance-dependent label noise at 20–60% with the clean labels kept for scoring, (c) long-tailed training sets with imbalance ratios 10–100 and a balanced test set, (d) positive-unlabelled splits with a known class prior, (e) noisy labelling functions with planted accuracies and correlations, (f) bags with bag-level labels and (g) candidate-label sets and group label proportions. Every test class trains small MLPs or the `VisionShootout` CNN for a few hundred steps so that it finishes in about 10 s; the curves at 4–10x the budget go in `main`. Baselines use the same network and step count, over at least 3 seeds.

### 10.1.1 Pseudo-labelling, the Π-model and Mean Teacher (Lee, 2013; Laine & Aila, ICLR 2017; Tarvainen & Valpola, NeurIPS 2017)
The three classic semi-supervised baselines add a loss on unlabelled data to the supervised loss. Pseudo-labelling trains on the argmax of the model's own prediction for unlabelled inputs, with a weight ramped up from zero. The Π-model penalises the squared difference between the predictions for two stochastic augmentations of the same input. Mean Teacher makes the target the prediction of an exponential moving average of the student's weights, which gives a smoother, more accurate target than the Π-model's second pass.
- **Block:** `experimental.semisup.ConsistencyTraining` (pseudo-label, Π-model and Mean Teacher modes with a sigmoid ramp-up), using `MomentumTeacher` for the EMA.
- **Experiment:** `experiments.labels.SemiSupervisedShootout`.
- **Toy task:** `LabelCorruption` digits and `ShapeImages` with 4, 10 and 40 labels per class and the rest unlabelled; plus two-moons for a picture of the decision boundary.
- **Baselines:** supervised-only training on the labelled subset; the same with core `Augmentation`; `ActiveLearner` picking the same number of labels (a different use of the budget); a fully labelled oracle.
- **Expected metric:** test accuracy against labels per class. Hypothesis: all three beat supervised-only at 4–10 labels per class; Mean Teacher is best of the three, and pseudo-labelling without a confidence threshold can fall below supervised-only through confirmation bias.
- **Source:** Lee, "Pseudo-Label", ICML 2013 Workshop on Challenges in Representation Learning; [arXiv:1610.02242](https://arxiv.org/abs/1610.02242), [arXiv:1703.01780](https://arxiv.org/abs/1703.01780)

### 10.1.2 Virtual adversarial training (Miyato et al., TPAMI 2018)
VAT replaces random augmentation with the worst-case one. For each input it finds, by one power-iteration step on a finite-difference Hessian-vector product, the small perturbation r of norm ε that changes the prediction most in KL divergence, and penalises KL(p(x) ‖ p(x + r)). It needs no hand-designed augmentations, so it also works on tabular data, and an entropy-minimisation term is usually added.
- **Block:** `experimental.semisup.VirtualAdversarial` (power iteration, ξ and ε, optional entropy term).
- **Experiment:** `SemiSupervisedShootout`.
- **Toy task:** as 10.1.1, plus `TabularStressSuite` with 20 labels per class, where image augmentations do not apply.
- **Baselines:** 10.1.1's Π-model and Mean Teacher; supervised-only; `AdversarialTraining`'s supervised PGD (adversarial perturbations on labelled data only).
- **Expected metric:** test accuracy and wall-clock per step. Hypothesis: VAT matches Mean Teacher on images and beats it on tabular data, where random noise is a poor augmentation.
- **Source:** [arXiv:1704.03976](https://arxiv.org/abs/1704.03976)

### 10.1.3 MixMatch (Berthelot et al., NeurIPS 2019)
MixMatch guesses a label for each unlabelled input by averaging the predictions over K augmentations and sharpening the average with a temperature T, then mixes labelled and unlabelled examples with MixUp and trains with cross-entropy on the labelled part and squared error on the unlabelled part. The sharpening acts as entropy minimisation, and MixUp regularises between the two sets.
- **Block:** `experimental.semisup.MixMatch` (label guessing, sharpening, MixUp across both sets), reusing `BatchMixing`'s mixup.
- **Experiment:** `SemiSupervisedShootout`.
- **Toy task:** as 10.1.1.
- **Baselines:** 10.1.1–10.1.2; supervised training with `BatchMixing` mixup on labelled data only.
- **Expected metric:** test accuracy at 4/10/40 labels per class, with ablations of K, T and MixUp. Hypothesis: MixMatch beats Mean Teacher at 4–10 labels per class; removing sharpening costs the most.
- **Source:** [arXiv:1905.02249](https://arxiv.org/abs/1905.02249)

### 10.1.4 FixMatch: weak-to-strong consistency with a confidence threshold (Sohn et al., NeurIPS 2020)
FixMatch takes the prediction on a weakly augmented view (flip and shift) and, only where its confidence exceeds τ = 0.95, uses its argmax as a hard pseudo-label for a strongly augmented view (RandAugment plus Cutout). One threshold and one strong augmentation replace MixMatch's label guessing and mixing, and the method reaches good accuracy with a handful of labels.
- **Block:** `experimental.semisup.FixMatch` (weak and strong pipelines from core `Augmentation`, threshold, unlabelled-to-labelled batch ratio μ).
- **Experiment:** `SemiSupervisedShootout`.
- **Toy task:** as 10.1.1; also a class-imbalanced unlabelled pool (from `LabelCorruption`'s long-tail mode), where a fixed threshold starves the rare classes.
- **Baselines:** 10.1.1–10.1.3; supervised-only with RandAugment.
- **Expected metric:** test accuracy, pseudo-label precision and the share of unlabelled data above threshold over training. Hypothesis: FixMatch is the strongest at 4 labels per class; on the imbalanced pool rare-class recall falls well below head-class recall.
- **Source:** [arXiv:2001.07685](https://arxiv.org/abs/2001.07685)

### 10.1.5 Adaptive thresholds: FlexMatch, FreeMatch and SoftMatch (Zhang et al., NeurIPS 2021; Wang et al., ICLR 2023; Chen et al., ICLR 2023)
A fixed 0.95 threshold wastes most unlabelled data early and favours easy classes. FlexMatch scales each class's threshold by its learning status (the share of confident predictions for that class), FreeMatch sets a self-adaptive global threshold from an EMA of the model's confidence plus per-class modulation and a fairness term, and SoftMatch replaces the hard cut with a truncated-Gaussian weight on confidence whose mean and variance are tracked by EMA. All three raise the quantity of pseudo-labels without lowering their quality much.
- **Block:** `experimental.semisup.AdaptiveThreshold` (FLEX, FREE and SOFT modes, plugged into 10.1.4's `FixMatch`).
- **Experiment:** `SemiSupervisedShootout`.
- **Toy task:** as 10.1.4, both balanced and imbalanced unlabelled pools.
- **Baselines:** FixMatch at τ ∈ {0.7, 0.95}; Mean Teacher.
- **Expected metric:** accuracy, worst-class accuracy, pseudo-label quantity and quality curves, and steps to 90% of final accuracy. Hypothesis: the adaptive thresholds converge 2–3x faster than FixMatch and raise worst-class accuracy on the imbalanced pool; final balanced accuracy ties.
- **Source:** [arXiv:2110.08263](https://arxiv.org/abs/2110.08263), [arXiv:2205.07246](https://arxiv.org/abs/2205.07246), [arXiv:2301.10921](https://arxiv.org/abs/2301.10921)

### 10.1.6 JEPAMatch: shaping the latent geometry of semi-supervised training with SIGReg (2026)
JEPAMatch adds LeJEPA's sketched isotropic Gaussian regulariser (SIGReg, 10.3.6) on the encoder's features to a FlexMatch-style loss. The authors argue that pushing the embeddings toward an isotropic Gaussian gives clearer class boundaries early in training, which reduces the damage from noisy early pseudo-labels and the dominance of majority classes, and report faster convergence than FixMatch pipelines.
- **Block:** `experimental.semisup.JepaMatch` (10.1.5's FLEX mode plus `experimental.ssl.Sigreg` on the penultimate features, weight λ).
- **Experiment:** `SemiSupervisedShootout`.
- **Toy task:** as 10.1.5, imbalanced pool included.
- **Baselines:** FlexMatch (10.1.5); FlexMatch plus VICReg's variance–covariance term from `SelfSupervisedLosses` (another anti-collapse regulariser); FixMatch.
- **Expected metric:** accuracy at 50%/100% of steps, worst-class accuracy, and early pseudo-label error. Hypothesis: SIGReg lowers early pseudo-label error and reaches FlexMatch's final accuracy in fewer steps; VICReg gives part of the gain.
- **Source:** [arXiv:2604.21046](https://arxiv.org/abs/2604.21046)

### 10.1.7 Noise-robust losses: GCE, SCE, active–passive losses and the Joint Asymmetric Loss (Zhang & Sabuncu, NeurIPS 2018; Wang et al., ICCV 2019; Ma et al., ICML 2020; Wang et al., ICCV 2025)
Cross-entropy fits wrong labels because its gradient is largest on the examples the model disagrees with. Generalised cross-entropy interpolates between CE and MAE with (1 − p_y^q)/q; symmetric CE adds reverse cross-entropy; the active–passive framework (APL) pairs a normalised "active" loss with a "passive" one that also pushes down non-target classes. The 2025 Joint Asymmetric Loss replaces APL's symmetric passive term with an asymmetric mean squared error (AMSE), whose asymmetry condition is derived exactly, so robustness no longer costs underfitting.
- **Block:** `experimental.labels.RobustLosses` (GCE, SCE, NCE+RCE, NCE+AMSE).
- **Experiment:** `experiments.labels.NoisyLabelShootout`.
- **Toy task:** `LabelCorruption` digits and `ShapeImages` with symmetric 20/40/60% and pair-flip 20/40% noise; accuracy on clean test labels.
- **Baselines:** cross-entropy; `SmoothedCrossEntropy` (label smoothing); focal loss; MAE.
- **Expected metric:** clean test accuracy at each noise rate, and the gap between best-epoch and final accuracy (memorisation). Hypothesis: CE's final accuracy collapses at 60% noise; GCE and APL keep most of it; JAL matches APL under symmetric noise and beats it under pair-flip noise; MAE underfits on clean data.
- **Source:** [arXiv:1805.07836](https://arxiv.org/abs/1805.07836), [arXiv:1908.06112](https://arxiv.org/abs/1908.06112), [arXiv:2006.13554](https://arxiv.org/abs/2006.13554), [arXiv:2507.17692](https://arxiv.org/abs/2507.17692)

### 10.1.8 Sample selection: co-teaching and DivideMix (Han et al., NeurIPS 2018; Li et al., ICLR 2020)
Networks learn clean examples before memorising noisy ones, so small-loss examples are probably clean. Co-teaching trains two networks, and each updates only on the small-loss share of the batch chosen by its peer, with the kept share shrinking to 1 − noise rate over a few epochs; the cross-update stops one network's selection errors from reinforcing themselves. DivideMix fits a two-component Gaussian mixture to per-example losses to split clean from noisy data, discards the noisy labels and trains the two networks with MixMatch (10.1.3) on the resulting labelled and unlabelled sets.
- **Block:** `experimental.labels.SampleSelection` (co-teaching, co-teaching+, loss-GMM split via `GaussianMixture`, DivideMix on top of `experimental.semisup.MixMatch`).
- **Experiment:** `NoisyLabelShootout`.
- **Toy task:** as 10.1.7, plus instance-dependent noise.
- **Baselines:** 10.1.7's best robust loss; CE; `DataPruner`'s EL2N pruning (drop high-score examples before training).
- **Expected metric:** clean test accuracy, precision and recall of the clean/noisy split. Hypothesis: co-teaching beats every robust loss at 40–60% symmetric noise; DivideMix is best overall but costs about 3x the compute; EL2N pruning flags noisy labels but removes hard clean ones too.
- **Source:** [arXiv:1804.06872](https://arxiv.org/abs/1804.06872), [arXiv:2002.07394](https://arxiv.org/abs/2002.07394)

### 10.1.9 Early-learning regularisation (Liu et al., NeurIPS 2020)
ELR keeps a temporal-ensemble target t_i (an EMA of each example's past predictions) and adds log(1 − ⟨p_i, t_i⟩) to the loss. Early in training the targets are mostly correct, so the term pulls the model toward its own early-learning predictions and its gradient cancels the push toward memorising wrong labels, without detecting the noisy examples explicitly.
- **Block:** `experimental.labels.EarlyLearningRegularizer` (per-example target memory, λ and β).
- **Experiment:** `NoisyLabelShootout`.
- **Toy task:** as 10.1.7.
- **Baselines:** CE with early stopping on a noisy validation set; 10.1.7's GCE; 10.1.8's co-teaching.
- **Expected metric:** final and best clean accuracy, and the share of noisy examples whose training prediction matches the true label at the end. Hypothesis: ELR keeps over 80% of noisy examples predicted correctly at 40% noise (CE under 20%) and matches co-teaching with one network.
- **Source:** [arXiv:2007.00151](https://arxiv.org/abs/2007.00151)

### 10.1.10 Loss correction with an estimated noise transition matrix (Patrini et al., CVPR 2017; Xia et al., NeurIPS 2019)
If labels are corrupted by a class-conditional transition matrix T (T_ij = P(noisy j | clean i)), training the model's output through T ("forward correction": loss on Tᵀp) is consistent for the clean posterior. T is estimated from anchor points, the examples with the most confident prediction for each class. T-revision fine-tunes T jointly with the classifier using importance reweighting, which fixes errors in the anchor estimate.
- **Block:** `experimental.labels.TransitionCorrection` (anchor estimator, forward and backward correction, T-revision).
- **Experiment:** `NoisyLabelShootout`.
- **Toy task:** pair-flip and asymmetric class-conditional noise (where the matrix assumption holds) and instance-dependent noise (where it does not).
- **Baselines:** forward correction with the true T (oracle); CE; 10.1.7 and 10.1.8.
- **Expected metric:** clean accuracy and ‖T̂ − T‖. Hypothesis: forward correction with estimated T comes within 2 points of the oracle on class-conditional noise and beats the robust losses there; under instance-dependent noise it loses its edge.
- **Source:** [arXiv:1609.03683](https://arxiv.org/abs/1609.03683), [arXiv:1906.00189](https://arxiv.org/abs/1906.00189)

### 10.1.11 Logit adjustment for long-tailed classes: additive, balanced softmax and multiplicative (Menon et al., ICLR 2021; Ren et al., NeurIPS 2020; Hasegawa & Sato, ICLR 2025)
A model trained on long-tailed data predicts the head classes too often. Additive logit adjustment either subtracts τ·log π_y from the logits after training or adds it inside the softmax during training (the logit-adjusted loss), which is Fisher-consistent for the balanced error; balanced softmax is the training-time form with τ = 1. Multiplicative logit adjustment scales each class's logit by n_y^(−γ) after training; the ICLR 2025 paper shows that, under neural-collapse assumptions, this approximates the optimal boundary shift that accounts for each class's feature spread, while the additive form depends on feature norms.
- **Block:** `experimental.labels.LogitAdjustment` (post-hoc additive, loss-time additive, balanced softmax, post-hoc multiplicative; τ/γ tuned on a balanced validation split).
- **Experiment:** `experiments.labels.LongTailShootout`.
- **Toy task:** `LabelCorruption` long-tailed digits and `ShapeImages` at imbalance ratios 10, 50 and 100, with a balanced test set.
- **Baselines:** CE; class-reweighted CE; random oversampling of tail classes; focal loss.
- **Expected metric:** balanced accuracy and head/medium/tail accuracy. Hypothesis: every logit adjustment beats reweighting and oversampling at ratio 100 by 3+ points; multiplicative and additive post-hoc tie within 1 point on this toy; loss-time adjustment is best.
- **Source:** [arXiv:2007.07314](https://arxiv.org/abs/2007.07314), [arXiv:2007.10740](https://arxiv.org/abs/2007.10740), [arXiv:2409.17582](https://arxiv.org/abs/2409.17582)

### 10.1.12 LDAM with deferred re-weighting, and classifier re-training (Cao et al., NeurIPS 2019; Kang et al., ICLR 2020)
LDAM gives each class a margin proportional to n_y^(−1/4) inside the cross-entropy, so rare classes need a larger margin; deferred re-weighting trains normally first and switches to class-balanced weights only in the last stage. Kang et al. found that representation and classifier can be decoupled: train the features with plain instance sampling, then re-train only the classifier with class-balanced sampling (cRT) or rescale its weight norms by ‖w_y‖^τ (τ-normalisation).
- **Block:** `experimental.labels.MarginAndDecoupling` (LDAM loss, DRW schedule, cRT and τ-norm on a frozen trunk).
- **Experiment:** `LongTailShootout`.
- **Toy task:** as 10.1.11.
- **Baselines:** 10.1.11's logit adjustments; CE; class-balanced re-sampling throughout.
- **Expected metric:** balanced and tail accuracy; head accuracy lost. Hypothesis: LDAM-DRW and cRT both beat CE by over 5 points on the tail at ratio 100; τ-norm, which needs no training, gets most of cRT's gain; re-sampling throughout hurts the features.
- **Source:** [arXiv:1906.07413](https://arxiv.org/abs/1906.07413), [arXiv:1910.09217](https://arxiv.org/abs/1910.09217)

### 10.1.13 Positive-unlabelled learning: uPU, nnPU, prior estimation and 2-HNC (Elkan & Noto, KDD 2008; Kiryo et al., NeurIPS 2017; 2025)
With only positive and unlabelled examples and the class prior π, the unbiased PU risk rewrites the negative-class risk as the unlabelled risk minus π times the positive risk. With a flexible model that estimate goes negative and the model overfits; nnPU clips the negative-risk term at zero and takes a gradient step that pushes it back when it would go negative. The prior itself can be estimated (Elkan–Noto's c = P(labelled | positive)). A 2025 alternative, 2-HNC, ranks unlabelled samples by a parametric minimum cut on a similarity graph (Hochbaum's normalised cut) and picks the nested partition whose positive share matches the prior.
- **Block:** `experimental.labels.PositiveUnlabeled` (biased PN, uPU, nnPU, Elkan–Noto prior estimation, 2-HNC using a max-flow solver; expose the Edmonds–Karp of `AttentionRollout` rather than copying it).
- **Experiment:** `experiments.labels.WeakLabelShootout`.
- **Toy task:** `LabelCorruption` PU splits of binary digits (even vs odd) and `TabularStressSuite`, π ∈ {0.3, 0.5}, 10% of positives labelled.
- **Baselines:** treating unlabelled as negative (biased PN) with `Calibration`'s threshold tuning; `IsolationForest` and Deep SVDD one-class detectors trained on positives only; a fully supervised PN oracle.
- **Expected metric:** test accuracy/AUC, the training risk curve, and prior-estimation error. Hypothesis: uPU overfits (its risk goes negative) and nnPU fixes it, coming within 3 points of PN; 2-HNC is competitive on tabular data with few labels.
- **Source:** [doi:10.1145/1401890.1401920](https://doi.org/10.1145/1401890.1401920), [arXiv:1703.00593](https://arxiv.org/abs/1703.00593), [arXiv:2505.08212](https://arxiv.org/abs/2505.08212)

### 10.1.14 Weak supervision with labelling functions: data programming and Dawid–Skene (Ratner et al., NeurIPS 2016 and VLDB 2018; Dawid & Skene, 1979)
Instead of hand labels, users write noisy heuristic labelling functions that vote or abstain. A generative label model learns each function's accuracy (and, in Snorkel, their correlations) without ground truth, from their agreements and disagreements, and produces probabilistic labels that train a discriminative end model with a noise-aware loss. Dawid–Skene does the same for crowd annotators with per-annotator confusion matrices fitted by EM.
- **Block:** `experimental.labels.LabelModel` (majority vote, Dawid–Skene EM, the data-programming factor model with learned accuracies and an optional correlation structure, matrix-completion estimator).
- **Experiment:** `WeakLabelShootout`.
- **Toy task:** `LabelCorruption` labelling functions over `TabularStressSuite` and digits features: 5–15 functions with accuracies 0.55–0.9, coverage 0.2–0.8 and one correlated pair.
- **Baselines:** majority vote; training on the few gold labels only; `IrtEvaluator`'s item-response model used as a label model (it also estimates annotator ability).
- **Expected metric:** label-model accuracy on covered points, accuracy-estimation error per function, and end-model accuracy. Hypothesis: the learned label model beats majority vote by 3–8 points when accuracies vary; modelling the correlated pair matters when it is in the majority.
- **Source:** [arXiv:1605.07723](https://arxiv.org/abs/1605.07723), [arXiv:1711.10160](https://arxiv.org/abs/1711.10160), [doi:10.2307/2346806](https://doi.org/10.2307/2346806)

### 10.1.15 Attention-based multiple-instance learning (Ilse et al., ICML 2018)
In multiple-instance learning only bags of instances are labelled: a bag is positive if at least one instance is. Attention-based MIL embeds each instance, pools the bag with learned (optionally gated) attention weights a_k ∝ exp(wᵀ tanh(V h_k) ⊙ sigm(U h_k)), and classifies the pooled vector; the attention weights then localise the positive instances without instance labels.
- **Block:** `experimental.labels.AttentionMil` (mean, max, attention and gated-attention pooling).
- **Experiment:** `WeakLabelShootout`.
- **Toy task:** MNIST-bags-style bags of `LabelCorruption` digits (positive if a "9" is present), bag sizes 5–50 and 50–400 training bags.
- **Baselines:** mean and max pooling MIL; `SetLearning`'s Deep Sets and Set Transformer (`SetEncoder`); an instance-level oracle trained on instance labels.
- **Expected metric:** bag AUC and instance-localisation AUC of the attention weights. Hypothesis: gated attention beats mean and max pooling with few training bags and localises the key instance with AUC above 0.9; the Set Transformer matches it only with many bags.
- **Source:** [arXiv:1802.04712](https://arxiv.org/abs/1802.04712)

### 10.1.16 Partial-label learning: PRODEN (Lv et al., ICML 2020)
Each training example comes with a set of candidate labels, one of them correct. PRODEN keeps a per-example weight vector over its candidates, trains on the weighted cross-entropy, and after each epoch re-sets the weights to the model's normalised predictions restricted to the candidate set, so the model progressively identifies the true label. The loss is shown to be classifier-consistent.
- **Block:** `experimental.labels.PartialLabels` (uniform-candidate loss, PRODEN re-weighting, and the "CC" classifier-consistent loss).
- **Experiment:** `WeakLabelShootout`.
- **Toy task:** `LabelCorruption` digits with 1–5 distractor candidates per example, uniform or confusion-biased.
- **Baselines:** uniform weights over candidates (no re-weighting); training on examples with a single candidate only; 10.1.7's GCE on a random candidate.
- **Expected metric:** test accuracy against candidate-set size. Hypothesis: PRODEN stays within 3 points of supervised accuracy with 3 distractors, while uniform weights lose 10+; confusion-biased distractors hurt every method.
- **Source:** [arXiv:2002.08053](https://arxiv.org/abs/2002.08053)

### 10.1.17 Learning from label proportions (Quadrianto et al., JMLR 2009; Ardehaly & Culotta, ICDM 2017)
Sometimes only the class proportions of groups are known (aggregate statistics, privacy-preserving releases). Deep LLP (DLLP) averages the model's predicted class distribution over each bag and minimises cross-entropy to the bag's known proportions; an EM-style variant then assigns instance pseudo-labels consistent with the proportions and retrains on them.
- **Block:** `experimental.labels.LabelProportions` (DLLP proportion loss, proportion-constrained pseudo-labelling by optimal transport with `EntropicTransport`).
- **Experiment:** `WeakLabelShootout`.
- **Toy task:** `LabelCorruption` digits grouped into bags of 8–128 with random or skewed proportions.
- **Baselines:** the mean-map (Quadrianto) linear estimator; training on bag-majority labels; supervised training on the same number of instances.
- **Expected metric:** instance-level test accuracy against bag size. Hypothesis: DLLP is near supervised accuracy at bag size 8 and degrades gracefully to 128; bag-majority labels collapse when proportions are near uniform; OT pseudo-labelling helps at large bags.
- **Source:** [JMLR 2009](https://jmlr.org/papers/v10/quadrianto09a.html), [arXiv:1709.04108](https://arxiv.org/abs/1709.04108)

---

## WP2. Distribution shift and the open world: OOD detection, open-set recognition, selective prediction and deferral, label shift, domain adaptation and generalisation, watermarking (17)

Target: two new packages, `experimental.ood` (10.2.1–10.2.7) and `experimental.shift` (10.2.8–10.2.16); 10.2.17 adds to the existing `experimental.llm`. A new theme `experiments.shift` holds `OodShootout` (10.2.1–10.2.5), `SelectivePrediction` (10.2.6–10.2.7), `LabelShift` (10.2.8), `DomainAdaptation` (10.2.9–10.2.11, 10.2.15) and `DomainGeneralization` (10.2.12–10.2.14, 10.2.16); 10.2.17 extends `experiments.language` (`WatermarkShootout`). One new generator in `experimental.data`: `ShiftSuites`, built on `ShapeImages`, `ToyTasks` digits, `DigitGlyphs` and `TabularStressSuite`: (a) in-distribution classes with near-OOD (held-out shape or digit classes) and far-OOD (noise, `ToyImages` textures) sets; (b) source/target domain pairs (colour, rotation, stroke width, background texture, glyph font); (c) several training domains plus a held-out test domain for generalisation; (d) label-shifted test sets with Dirichlet priors; (e) a spurious-colour split in the style of `CausalAndFairness`' coloured bars (reuse its generator through a public accessor, so wave 6–7 rows are unchanged). One shared classifier per task (the `VisionShootout` CNN or an MLP) is trained once per seed and cached, so the post-hoc OOD items stay well under 10 s per test class.

### 10.2.1 Logit-based OOD scores: maximum softmax probability, max logit, energy and ODIN (Hendrycks & Gimpel, ICLR 2017; Hendrycks et al., ICML 2022; Liu et al., NeurIPS 2020; Liang et al., ICLR 2018)
A trained classifier can flag inputs from unseen classes without retraining. The baseline score is the maximum softmax probability; the max logit avoids softmax saturation; the energy score −T·log Σ exp(f_k/T) is aligned with the input density and separates OOD inputs better. ODIN sharpens the gap with a temperature T ≈ 1000 and a small input perturbation in the direction that increases the softmax score, which helps in-distribution inputs more than OOD ones.
- **Block:** `experimental.ood.LogitScores` (MSP, max logit, energy, ODIN with gradient-sign input perturbation).
- **Experiment:** `experiments.shift.OodShootout`.
- **Toy task:** `ShiftSuites` near-OOD and far-OOD sets for a `ShapeImages` CNN and a digits MLP.
- **Baselines:** `UncertaintyBenchmark`'s OOD AUROCs for deep ensembles, MC dropout, SWAG, SNGP and DUQ (`DistanceAwareHead`), and `LastLayerLaplace`.
- **Expected metric:** AUROC and FPR at 95% TPR for near and far OOD, and cost in forward passes. Hypothesis: energy beats MSP by 2–5 AUROC points at one pass; ODIN helps on far OOD only; ensembles stay best on near OOD at 5x the cost.
- **Source:** [arXiv:1610.02136](https://arxiv.org/abs/1610.02136), [arXiv:1911.11132](https://arxiv.org/abs/1911.11132), [arXiv:2010.03759](https://arxiv.org/abs/2010.03759), [arXiv:1706.02690](https://arxiv.org/abs/1706.02690)

### 10.2.2 Feature-space OOD scores: Mahalanobis, deep k-nearest neighbours and ViM (Lee et al., NeurIPS 2018; Sun et al., ICML 2022; Wang et al., CVPR 2022)
OOD inputs often look confident in the logits but sit far from the training features. The Mahalanobis score fits class-conditional Gaussians with a shared covariance to penultimate features and scores the minimum Mahalanobis distance. Deep k-NN scores the distance to the k-th nearest normalised training feature, with no distribution assumption. ViM (virtual-logit matching) adds a "virtual logit" built from the norm of the feature's residual outside the principal subspace of the training features, scaled to match the logits, and scores its softmax probability.
- **Block:** `experimental.ood.FeatureScores` (Mahalanobis with tied covariance, k-NN on normalised features via `KnnIndex`, ViM).
- **Experiment:** `OodShootout`.
- **Toy task:** as 10.2.1.
- **Baselines:** 10.2.1's scores; `DistanceAwareHead` (DUQ/SNGP, trained to be distance-aware); `GaussianMixture` on raw inputs.
- **Expected metric:** AUROC and FPR95 on near and far OOD. Hypothesis: k-NN and ViM beat every logit score on near OOD; Mahalanobis is good on far OOD but brittle on near OOD; the post-hoc scores close most of the gap to DUQ without retraining.
- **Source:** [arXiv:1807.03888](https://arxiv.org/abs/1807.03888), [arXiv:2204.06507](https://arxiv.org/abs/2204.06507), [arXiv:2203.10807](https://arxiv.org/abs/2203.10807)

### 10.2.3 Activation shaping: ReAct, ASH, SCALE and AdaSCALE (Sun et al., NeurIPS 2021; Djurisic et al., ICLR 2023; Xu et al., ICLR 2024; Regmi, 2025)
OOD inputs cause a few unusually large penultimate activations. ReAct clips activations at a percentile of their training distribution before the last layer. ASH prunes all but the top p% of a sample's activations and rescales the rest by the ratio of total to kept sum; SCALE keeps the rescaling only, which helps OOD detection without hurting accuracy. AdaSCALE makes the percentile per-sample: it perturbs the least-attributed input features slightly, measures how much the top activations shift (OOD inputs shift more), turns that into an ID-likelihood percentile with a small ID validation set, and scales more strongly for likely-ID inputs before the energy score.
- **Block:** `experimental.ood.ActivationShaping` (ReAct, ASH-P/B/S, SCALE, AdaSCALE-A/L), wrapping any `Model` whose penultimate layer is exposed.
- **Experiment:** `OodShootout`.
- **Toy task:** as 10.2.1.
- **Baselines:** 10.2.1–10.2.2; the energy score without shaping.
- **Expected metric:** AUROC/FPR95, in-distribution accuracy after shaping, and extra forward passes (AdaSCALE needs one gradient and one perturbed pass). Hypothesis: SCALE improves energy by 2–4 points with no accuracy loss and ASH loses some accuracy; AdaSCALE adds a further gain on near OOD at 2–3x the cost.
- **Source:** [arXiv:2111.12797](https://arxiv.org/abs/2111.12797), [arXiv:2209.09858](https://arxiv.org/abs/2209.09858), [arXiv:2310.00227](https://arxiv.org/abs/2310.00227), [arXiv:2503.08023](https://arxiv.org/abs/2503.08023)

### 10.2.4 Outlier exposure (Hendrycks et al., ICLR 2019; Liu et al., NeurIPS 2020)
If some auxiliary outliers are available at training time (not the test OOD set), training the classifier to give them a uniform softmax (cross-entropy to uniform) teaches it to be unconfident off the data manifold. The energy-bounded variant instead adds hinge losses pushing in-distribution energy below m_in and outlier energy above m_out.
- **Block:** `experimental.ood.OutlierExposure` (uniform-target and energy-margin losses).
- **Experiment:** `OodShootout`.
- **Toy task:** as 10.2.1, with auxiliary outliers drawn from `ToyImages` noise and textures, and the test OOD sets disjoint from them (held-out shape classes, other textures).
- **Baselines:** 10.2.1–10.2.3 without outlier training; `JointEnergyModel` (JEM, another way to shape energy); mixup-trained classifiers (`BatchMixing`).
- **Expected metric:** AUROC on near and far OOD, and accuracy. Hypothesis: OE gives the largest gain of the WP on far OOD and a smaller one on near OOD, which the auxiliary set does not resemble.
- **Source:** [arXiv:1812.04606](https://arxiv.org/abs/1812.04606), [arXiv:2010.03759](https://arxiv.org/abs/2010.03759)

### 10.2.5 Open-set recognition: OpenMax against a good closed-set classifier (Bendale & Boult, CVPR 2016; Vaze et al., ICLR 2022)
Open-set recognition must classify known classes and reject unknown ones in one model. OpenMax fits a Weibull model (extreme value theory) to the distances of each class's correct activation vectors from their mean and uses it to move probability mass from the top classes to an explicit "unknown" class. Vaze et al. showed that the open-set AUROC of a plain classifier's max logit tracks its closed-set accuracy, so better training (longer schedules, label smoothing, augmentation) closes most of the gap to dedicated methods.
- **Block:** `experimental.ood.OpenMax` (per-class mean activation vectors, Weibull tail fitting, recalibrated softmax with an unknown class).
- **Experiment:** `OodShootout` (`openSet`).
- **Toy task:** `ShiftSuites` with 6 known and 4 unknown shape or digit classes; open-set classification rate and AUROC.
- **Baselines:** max logit (10.2.1) of the same classifier trained briefly and trained well (4x steps, label smoothing, RandAugment); `ConformalClassifier` prediction sets (empty or large sets as rejection).
- **Expected metric:** AUROC and open-set classification rate against closed-set accuracy. Hypothesis: OpenMax beats max logit at the short budget; the well-trained max logit matches or beats OpenMax, as Vaze et al. found.
- **Source:** [arXiv:1511.06233](https://arxiv.org/abs/1511.06233), [arXiv:2110.06207](https://arxiv.org/abs/2110.06207)

### 10.2.6 Selective classification: SelectiveNet, Deep Gamblers and likelihood-ratio rejection (Geifman & El-Yaniv, ICML 2019; Liu et al., NeurIPS 2019; 2025)
A selective classifier may abstain and is judged by its risk–coverage curve. SelectiveNet trains a selection head jointly with the classifier under a target coverage constraint (a quadratic penalty), plus an auxiliary head trained on all data. Deep Gamblers adds an "abstain" output and trains with a doubling-rate (gambling) loss with payoff o, so the model learns when to reserve. A 2025 paper casts the optimal selection rule as a Neyman–Pearson likelihood-ratio test, which unifies post-hoc scores and suggests ratio-based rejection that holds up under covariate shift.
- **Block:** `experimental.ood.SelectiveClassifier` (SelectiveNet heads and coverage penalty, the gambler's loss, and a post-hoc likelihood-ratio rejector from class-conditional feature densities).
- **Experiment:** `experiments.shift.SelectivePrediction`.
- **Toy task:** `ShapeImages` and digits with label noise and ambiguous examples; in-distribution and covariate-shifted test sets from `ShiftSuites`.
- **Baselines:** softmax-response thresholding; `PopulationEnsemble.predictOrAbstain`; MC dropout variance; `ConformalClassifier` set size.
- **Expected metric:** area under the risk–coverage curve, and risk at 80% coverage, with and without shift. Hypothesis: softmax response is hard to beat in distribution (as in the literature); the trained methods help a little; likelihood-ratio rejection degrades least under shift.
- **Source:** [arXiv:1901.09192](https://arxiv.org/abs/1901.09192), [arXiv:1907.00208](https://arxiv.org/abs/1907.00208), [arXiv:2505.15008](https://arxiv.org/abs/2505.15008)

### 10.2.7 Learning to defer to an expert (Mozannar & Sontag, ICML 2020; Verma & Nalisnick, ICML 2022)
When a human or larger model can take over, the system should defer exactly the inputs on which the expert beats the classifier. Mozannar and Sontag add a "defer" class to the softmax and train with a consistent surrogate loss that rewards deferral where the expert is right. Verma and Nalisnick replace the softmax with one-vs-all sigmoids, which gives calibrated estimates of the expert's correctness and so a calibrated deferral rule.
- **Block:** `experimental.ood.LearningToDefer` (softmax surrogate, one-vs-all surrogate, confidence-threshold baseline, deferral budget).
- **Experiment:** `SelectivePrediction` (`defer`).
- **Toy task:** digits where a simulated expert is accurate on some classes (or on noisy images) and random on others, with a per-query cost.
- **Baselines:** defer when the classifier's confidence is below a threshold (ignoring the expert); a two-stage "confidence vs separately learned expert-correctness model" rule; the classifier alone; the expert alone.
- **Expected metric:** system accuracy against deferral rate. Hypothesis: both learned rules beat confidence thresholding by deferring where the expert is strong rather than where the classifier is weak; the one-vs-all rule's expert-correctness estimates are better calibrated (lower ECE).
- **Source:** [arXiv:2006.01862](https://arxiv.org/abs/2006.01862), [arXiv:2202.03673](https://arxiv.org/abs/2202.03673)

### 10.2.8 Label-shift estimation and correction: BBSE, MLLS, and online Dirichlet-prior estimation (Lipton et al., ICML 2018; Saerens et al., Neural Computation 2002; Garg et al., NeurIPS 2020; Hu & Barria, 2025)
Under label shift, p(x | y) stays fixed but the class priors change, so a classifier can be corrected by reweighting its posteriors with w_y = q(y)/p(y). Black-box shift estimation solves C·w = μ̂, where C is the confusion matrix on held-out source data and μ̂ the mean prediction on target data. MLLS runs EM on the target's prior with the classifier's (calibrated) posteriors and is usually more accurate. A November 2025 paper puts a Dirichlet prior on the target priors and, for streams, replaces the E-step by a stochastic approximation step, cutting the per-batch cost from O(NK) to O(K).
- **Block:** `experimental.shift.LabelShiftEstimator` (BBSE, RLLS regularised BBSE, MLLS with bias-corrected temperature calibration from `Calibration`, MAP with a Dirichlet prior, an online variant).
- **Experiment:** `experiments.shift.LabelShift`.
- **Toy task:** `ShiftSuites` label-shifted digit and shape test sets: tweak-one, long-tailed and Dirichlet(α ∈ {0.1, 1, 10}) priors; a stream whose prior drifts.
- **Baselines:** no correction; correction with the true weights (oracle); `TestTimeAdaptation`'s BN statistics and TENT, which wave 8 found to fail under label shift.
- **Expected metric:** weight-estimation MSE, accuracy after correction, and for the stream, accuracy and estimation error over time. Hypothesis: MLLS with calibration beats BBSE at small target samples; correction recovers most of the oracle gain; TENT lowers accuracy here, as in wave 8; the online estimator tracks a drifting prior at a fraction of the batch cost.
- **Source:** [arXiv:1802.03916](https://arxiv.org/abs/1802.03916), [doi:10.1162/089976602753284446](https://doi.org/10.1162/089976602753284446), [arXiv:2003.07554](https://arxiv.org/abs/2003.07554), [arXiv:2511.18615](https://arxiv.org/abs/2511.18615)

### 10.2.9 Adversarial domain adaptation: DANN and CDAN (Ganin et al., JMLR 2016; Long et al., NeurIPS 2018)
With labelled source data and unlabelled target data, DANN trains a domain discriminator on the features and passes its gradient back through a gradient-reversal layer, so the features become domain-invariant while staying discriminative for the source labels. CDAN conditions the discriminator on the outer product of features and predicted class probabilities (and weights examples by prediction entropy), which aligns the class-conditional distributions rather than only the marginal one.
- **Block:** `experimental.shift.AdversarialAdaptation` (gradient-reversal op with a λ schedule, DANN and CDAN(+E) discriminators).
- **Experiment:** `experiments.shift.DomainAdaptation`.
- **Toy task:** `ShiftSuites` source→target pairs: digits in one `DigitGlyphs` font → another font, `ShapeImages` plain → textured background, and rotated two-moons.
- **Baselines:** source-only training; `TestTimeAdaptation` (BN statistics = AdaBN, TENT); a target-supervised oracle.
- **Expected metric:** target accuracy and the share of the source-only→oracle gap closed. Hypothesis: DANN closes 30–60% of the gap; CDAN more when the class balance differs between domains; both are unstable without the λ ramp.
- **Source:** [arXiv:1505.07818](https://arxiv.org/abs/1505.07818), [arXiv:1705.10667](https://arxiv.org/abs/1705.10667)

### 10.2.10 Discrepancy-based adaptation: Deep CORAL and multi-kernel MMD (Sun & Saenko, ECCV-W 2016; Long et al., ICML 2015)
Instead of an adversary, align statistics directly. Deep CORAL penalises the squared Frobenius distance between the source and target feature covariances. DAN penalises a multi-kernel maximum mean discrepancy between source and target features in several layers, with an unbiased linear-time estimator.
- **Block:** `experimental.shift.DiscrepancyAdaptation` (CORAL loss, linear-time MK-MMD reusing `KernelTests`' kernels).
- **Experiment:** `DomainAdaptation`.
- **Toy task:** as 10.2.9.
- **Baselines:** 10.2.9's DANN and CDAN; source-only; AdaBN (`TestTimeAdaptation`'s BN-statistics row).
- **Expected metric:** target accuracy, stability across seeds, and wall-clock. Hypothesis: CORAL is the most stable and nearly as accurate as DANN on appearance shifts; MMD helps on the rotated moons; neither handles a class-prior change between domains.
- **Source:** [arXiv:1607.01719](https://arxiv.org/abs/1607.01719), [arXiv:1502.02791](https://arxiv.org/abs/1502.02791)

### 10.2.11 Source-free adaptation: SHOT (Liang et al., ICML 2020)
Sometimes the source data cannot be shared, only the source model. SHOT freezes the source classifier head and adapts the feature extractor on unlabelled target data with information maximisation (low per-sample entropy, high entropy of the batch-average prediction, which prevents collapse) plus self-supervised pseudo-labels from class centroids computed in feature space and refined by one k-means step.
- **Block:** `experimental.shift.SourceFreeAdaptation` (frozen head, IM loss, centroid pseudo-labelling).
- **Experiment:** `DomainAdaptation`.
- **Toy task:** as 10.2.9, without access to source data at adaptation time.
- **Baselines:** `TestTimeAdaptation` TENT (entropy only) and MEMO; source-only; 10.2.9–10.2.10 (which do see the source data).
- **Expected metric:** target accuracy and the collapse rate (share of runs predicting fewer than half the classes). Hypothesis: SHOT beats TENT clearly because the diversity term prevents collapse, and comes within a few points of DANN without source data.
- **Source:** [arXiv:2002.08546](https://arxiv.org/abs/2002.08546)

### 10.2.12 Domain generalisation: MixStyle and SWAD (Zhou et al., ICLR 2021; Cha et al., NeurIPS 2021)
In domain generalisation the target domain is unseen during training. MixStyle mixes the per-channel feature means and standard deviations (the "style") of instances from different domains inside early CNN layers, synthesising new styles for free. SWAD averages weights densely along the training trajectory, starting and stopping where validation loss is flat, to find flat minima that transfer better than a single checkpoint.
- **Block:** `experimental.shift.DomainGeneralization` (MixStyle layer with domain-aware or random shuffling; SWAD's overfit-aware averaging window, reusing `StochasticWeightAveraging`).
- **Experiment:** `experiments.shift.DomainGeneralization`.
- **Toy task:** `ShiftSuites` leave-one-domain-out on 4 `ShapeImages` styles (colour, texture, stroke, noise) and 4 glyph fonts.
- **Baselines:** ERM; `GroupRobustTraining`'s IRM, V-REx and GroupDRO (which lost to ERM in waves 6–7); plain SWA; core `Augmentation` RandAugment.
- **Expected metric:** held-out-domain accuracy averaged over the leave-one-out splits. Hypothesis: ERM with augmentation is a strong baseline (as DomainBed found); SWAD beats it by 1–3 points; MixStyle helps on texture shifts only; IRM/V-REx stay below ERM.
- **Source:** [arXiv:2104.02008](https://arxiv.org/abs/2104.02008), [arXiv:2102.08604](https://arxiv.org/abs/2102.08604)

### 10.2.13 Last-layer retraining for spurious correlations: DFR (Kirichenko et al., ICLR 2023)
Kirichenko et al. found that ERM features already encode the core attribute well even when the classifier relies on a spurious one; the problem is in the last layer. Deep feature reweighting retrains only the final linear layer, with ℓ1 regularisation, on a small group-balanced held-out set, and matches GroupDRO's worst-group accuracy at a fraction of the cost and without group labels on the training set.
- **Block:** `experimental.shift.LastLayerRetraining` (group-balanced subsampling, ℓ1 logistic last layer, ensembling over subsamples).
- **Experiment:** `DomainGeneralization` (`spurious`).
- **Toy task:** `ShiftSuites`' spurious-colour split (`CausalAndFairness` coloured bars) and a Waterbirds-like `ShapeImages` split with correlated backgrounds.
- **Baselines:** ERM; `GroupRobustTraining`'s GroupDRO and JTT; 10.2.12's IRM row.
- **Expected metric:** worst-group and average accuracy, and the number of group labels used. Hypothesis: DFR matches or beats GroupDRO's worst-group accuracy with group labels only on the small held-out set, and beats JTT.
- **Source:** [arXiv:2204.02937](https://arxiv.org/abs/2204.02937)

### 10.2.14 Covariate-shift correction by density-ratio estimation: KLIEP, uLSIF and classifier odds (Sugiyama et al., AISM 2008; Kanamori et al., JMLR 2009)
Under covariate shift p(y | x) is fixed and only p(x) changes, so importance-weighting the training loss by w(x) = q(x)/p(x) is unbiased. KLIEP fits w as a kernel expansion by maximising the target log-likelihood under a normalisation constraint; uLSIF fits it by least squares in closed form; the simplest estimator trains a domain classifier and uses its odds. Weighted conformal prediction (Tibshirani et al., 2019) uses the same weights to keep coverage under covariate shift.
- **Block:** `experimental.shift.DensityRatio` (KLIEP, uLSIF with cross-validated bandwidth, classifier odds, weight clipping) and a weighted split-conformal wrapper around `ConformalClassifier`.
- **Experiment:** `DomainGeneralization` (`covariate`).
- **Toy task:** 1-D and 5-D regression with a known shift of the input density (the classic Shimodaira setup), and `TabularStressSuite` with a shifted feature.
- **Baselines:** unweighted training; weighting with the true density ratio (oracle); split conformal without weights.
- **Expected metric:** target MSE or accuracy, ratio-estimation error, effective sample size, and conformal coverage on the target. Hypothesis: uLSIF is the most stable estimator; weighting helps only for misspecified models; weighted conformal restores nominal coverage where plain split conformal under-covers.
- **Source:** [doi:10.1007/s10463-008-0197-x](https://doi.org/10.1007/s10463-008-0197-x), [JMLR 2009](https://jmlr.org/papers/v10/kanamori09a.html), [arXiv:1904.06019](https://arxiv.org/abs/1904.06019)

### 10.2.15 Continual and robust test-time adaptation: EATA, SAR and CoTTA (Niu et al., ICML 2022 and ICLR 2023; Wang et al., CVPR 2022)
TENT fails on long or non-stationary streams: it collapses, forgets and breaks on small or single-class batches (wave 8 measured the last). EATA adapts only on reliable, non-redundant samples (low entropy, prediction unlike recent ones) and adds a Fisher regulariser against forgetting. SAR removes high-gradient samples, uses sharpness-aware entropy minimisation and resets the model when it collapses. CoTTA uses a weight-averaged teacher with augmentation-averaged pseudo-labels and stochastically restores a few weights to the source values.
- **Block:** `experimental.shift.RobustTestTimeAdaptation` (EATA, SAR on `SharpnessAware`, CoTTA with `MomentumTeacher`), on top of `TestTimeAdaptation`.
- **Experiment:** `DomainAdaptation` (`continual`).
- **Toy task:** a `ShiftSuites` stream cycling through 6 corruption types for 10 rounds, with batch sizes 1–64 and label-shifted (single-class) batches.
- **Baselines:** no adaptation; BN statistics; TENT; MEMO.
- **Expected metric:** accuracy over the stream, accuracy in the final round versus the first, and collapse events. Hypothesis: TENT degrades over rounds and collapses on small batches; EATA and SAR stay stable, SAR best at batch size 1; CoTTA forgets least at about 3x the compute.
- **Source:** [arXiv:2204.02610](https://arxiv.org/abs/2204.02610), [arXiv:2302.12400](https://arxiv.org/abs/2302.12400), [arXiv:2203.13591](https://arxiv.org/abs/2203.13591)

### 10.2.16 Invariance by fine-tuning a robust start: WiSE-FT and LP-FT (Wortsman et al., CVPR 2022; Kumar et al., ICLR 2022)
Fine-tuning a pretrained model can lose its out-of-distribution robustness. LP-FT first fits a linear probe on frozen features and only then fine-tunes everything, so the head does not distort the features early on. WiSE-FT linearly interpolates the zero-shot (or probed) and fine-tuned weights, which keeps much of the fine-tuned in-distribution gain and recovers the robustness.
- **Block:** `experimental.shift.RobustFineTuning` (LP-FT schedule, weight interpolation with α sweep).
- **Experiment:** `DomainGeneralization` (`fineTuning`).
- **Toy task:** a `ShapeImages` encoder pretrained with `SelfSupervisedBenchmark`'s SimCLR (and `ZeroShotDigits`' CLIP for digits), fine-tuned on one domain of `ShiftSuites` and tested on the shifted domains.
- **Baselines:** full fine-tuning; linear probing; `ModelMerging`'s plain averaging of fine-tuned seeds (model soup).
- **Expected metric:** in-distribution and shifted accuracy across α. Hypothesis: LP-FT beats full fine-tuning on the shifted domains with no in-distribution loss; WiSE-FT at α ≈ 0.5 beats both endpoints on the shifted domains.
- **Source:** [arXiv:2109.01903](https://arxiv.org/abs/2109.01903), [arXiv:2202.10054](https://arxiv.org/abs/2202.10054)

### 10.2.17 Tournament-sampling text watermarks: SynthID-Text (Dathathri et al., Nature 2024)
The green-list watermark biases logits toward a keyed half of the vocabulary, which changes the output distribution and is weakened by low-entropy text. SynthID-Text instead samples 2^m candidate tokens from the unchanged LM distribution and runs an m-layer knockout tournament, where each layer keeps the candidate with the higher pseudorandom g-value (keyed by a hash of the preceding tokens); the winner is emitted. Detection averages the g-values of the text. In its non-distortionary configuration the per-token distribution is preserved in expectation.
- **Block:** `experimental.llm.TournamentWatermark` (g-value functions, tournament sampler as a `LogitsProcessor`-compatible sampler, mean and weighted-mean detectors).
- **Experiment:** `experiments.language.WatermarkShootout`.
- **Toy task:** generations from the toy LMs (`WindowLm`/`TinyLm` on `SyntheticCorpus`) at temperatures 0.5–1.0, lengths 25–200 tokens, plus paraphrase-like edits (random token substitutions and deletions at 10–30%).
- **Baselines:** `GreenListWatermark` (Kirchenbauer et al.) at δ ∈ {1, 2}; unwatermarked text.
- **Expected metric:** detection TPR at 1% FPR against length, quality (held-out LM log-likelihood and distinct-n diversity), and robustness to edits. Hypothesis: the tournament watermark matches the green list's detectability with no measurable quality loss, and is more detectable on low-entropy text at equal quality; both degrade similarly under edits.
- **Source:** [doi:10.1038/s41586-024-08025-4](https://doi.org/10.1038/s41586-024-08025-4)

---

## WP3. Representation learning: self-supervised follow-ups and LeJEPA, deep metric learning, deep clustering, multi-task gradient balancing (17)

Target: three new packages, `experimental.ssl` (10.3.1–10.3.6), `experimental.metric` (10.3.7–10.3.10) and `experimental.multitask` (10.3.13–10.3.16); deep clustering (10.3.11–10.3.12) adds to the existing `experimental.ml`, and the CCS follow-up (10.3.17) adds to the existing `experimental.interp`. Measurements: `experiments.selfsupervised.SelfSupervisedBenchmark` gains rows (10.3.1–10.3.6) through new methods, without changing its existing `Method` rows; new theme `experiments.metric` (`MetricLearningShootout` for 10.3.7–10.3.10, `DeepClustering` for 10.3.11–10.3.12); new theme `experiments.multitask` (`GradientBalancing` for 10.3.13–10.3.16); `experiments.interp.LatentKnowledge` gains a follow-up method (10.3.17). No new generator: the tasks are `SelfSupervisedBenchmark`'s shape views, `ToyTasks` digits, `DigitGlyphs`, `ShapeImages` with held-out classes, wave 9's `ClickLogs` two-label task, and a multi-label digits task (class, parity, magnitude, stroke count) built inside `GradientBalancing`. Test classes run 150–400 steps of a small MLP or CNN so that each finishes in about 10 s; the SSL comparisons report linear-probe and k-NN accuracy at matched encoder steps and matched wall-clock.

### 10.3.1 SimSiam: Siamese learning without negatives or momentum (Chen & He, CVPR 2021)
SimSiam shows that a Siamese network with a predictor MLP on one branch and a stop-gradient on the other does not collapse, even without negatives (SimCLR), a momentum encoder (BYOL) or large batches. The loss is the negative cosine between the predictor output of one view and the stop-gradient projection of the other, symmetrised. The stop-gradient is the essential piece; the authors interpret it as an EM-like alternation.
- **Block:** `experimental.ssl.SimSiam` (projector, predictor, stop-gradient, symmetric loss), next to `SelfSupervisedLosses`.
- **Experiment:** `SelfSupervisedBenchmark` (new `siamese` method).
- **Toy task:** `SelfSupervisedBenchmark`'s shape views and digits.
- **Baselines:** SimCLR, BYOL, Barlow Twins, VICReg (existing rows); SimSiam without stop-gradient and without predictor (ablations); a random encoder.
- **Expected metric:** linear-probe and k-NN accuracy, embedding rank (effective dimensionality) over training, and collapse rate across seeds. Hypothesis: SimSiam matches BYOL at batch 64 without a momentum encoder; removing stop-gradient collapses every seed.
- **Source:** [arXiv:2011.10566](https://arxiv.org/abs/2011.10566)

### 10.3.2 SwAV: online clustering with swapped prediction (Caron et al., NeurIPS 2020)
SwAV assigns each view's embedding to K learnable prototypes with a balanced soft assignment from three Sinkhorn–Knopp iterations (the equipartition constraint prevents collapse), then trains each view to predict the other view's assignment. Multi-crop adds several low-resolution crops cheaply.
- **Block:** `experimental.ssl.SwappedPrediction` (prototypes, Sinkhorn–Knopp codes reusing `EntropicTransport`, multi-crop sampler).
- **Experiment:** `SelfSupervisedBenchmark` (`swav`).
- **Toy task:** as 10.3.1, with 2 + 4 multi-crop views.
- **Baselines:** DINO (existing row, which uses centring and sharpening instead of Sinkhorn); SimCLR; 10.3.1.
- **Expected metric:** linear-probe and k-NN accuracy; cluster purity of the prototype assignments; gain from multi-crop. Hypothesis: SwAV beats SimCLR at small batch because it needs no negatives; multi-crop adds 1–3 points to both SwAV and DINO.
- **Source:** [arXiv:2006.09882](https://arxiv.org/abs/2006.09882)

### 10.3.3 MoCo: a queue of negatives with a momentum key encoder (He et al., CVPR 2020; Chen et al., ICCV 2021)
SimCLR needs large batches for many negatives. MoCo keeps a FIFO queue of keys from past batches as negatives, encoded by a slowly updated momentum copy of the encoder so that queued keys stay consistent. MoCo v3 drops the queue for ViTs, uses in-batch negatives with a predictor and a symmetrised loss, and freezes the patch projection for stability.
- **Block:** `experimental.ssl.MomentumContrast` (key queue, momentum encoder through `MomentumTeacher`, v2 and v3 modes).
- **Experiment:** `SelfSupervisedBenchmark` (`moco`).
- **Toy task:** as 10.3.1, at batch sizes 16, 64 and 256.
- **Baselines:** SimCLR at the same batch sizes; BYOL.
- **Expected metric:** linear-probe accuracy against batch size, and queue length sweep. Hypothesis: MoCo v2 holds its accuracy at batch 16 where SimCLR drops several points; v3 matches v2 at large batch.
- **Source:** [arXiv:1911.05722](https://arxiv.org/abs/1911.05722), [arXiv:2104.02057](https://arxiv.org/abs/2104.02057)

### 10.3.4 Masked prediction of teacher features: data2vec and iBOT (Baevski et al., ICML 2022; Zhou et al., ICLR 2022)
MAE predicts pixels; I-JEPA predicts target-encoder features of masked blocks. data2vec has a student predict, for the masked positions, the average of the top-K layer outputs of an EMA teacher that sees the unmasked input, with the same recipe for images, speech and text. iBOT combines a DINO-style class-token self-distillation loss with masked patch-token self-distillation against the online teacher's patch tokens (an "online tokenizer").
- **Block:** `experimental.ssl.MaskedFeaturePrediction` (data2vec layer-averaged targets with instance norm; iBOT patch-level DINO loss).
- **Experiment:** `SelfSupervisedBenchmark` (patch-transformer branch next to MAE and I-JEPA).
- **Toy task:** the existing 2x2-patch shape images used by MAE and I-JEPA-lite; also `SpokenDigitRecognition` frames for data2vec's audio claim (one row).
- **Baselines:** MAE and I-JEPA-lite (existing rows, which wave 7 found weak); DINO.
- **Expected metric:** linear-probe accuracy at matched steps; on audio, digit error after a linear probe against `SpokenDigitRecognition`'s HuBERT/wav2vec rows. Hypothesis: data2vec beats MAE at the same budget because contextual targets are easier than pixels; iBOT beats DINO by adding the patch loss.
- **Source:** [arXiv:2202.03555](https://arxiv.org/abs/2202.03555), [arXiv:2111.07832](https://arxiv.org/abs/2111.07832)

### 10.3.5 The KoLeo regulariser from DINOv2 (Sablayrolles et al., ICLR 2019; Oquab et al., TMLR 2024)
DINOv2 adds the Kozachenko–Leonenko differential-entropy estimator as a regulariser: for each embedding in the batch, −log of the distance to its nearest neighbour, averaged. Maximising it spreads features uniformly over the sphere, which improves nearest-neighbour retrieval without changing the main loss.
- **Block:** `experimental.ssl.KoLeo` (batch nearest-neighbour entropy loss).
- **Experiment:** `SelfSupervisedBenchmark` (a KoLeo option on DINO, BYOL and SimSiam rows).
- **Toy task:** as 10.3.1; also as a regulariser in 10.3.7's supervised metric learning.
- **Baselines:** the same methods without KoLeo; VICReg's variance term (another anti-concentration term).
- **Expected metric:** k-NN accuracy and retrieval Recall@1, plus the uniformity metric of Wang & Isola. Hypothesis: KoLeo improves k-NN and retrieval by 1–3 points and uniformity clearly, with linear-probe accuracy unchanged.
- **Source:** [arXiv:1806.03198](https://arxiv.org/abs/1806.03198), [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)

### 10.3.6 LeJEPA: a JEPA with sketched isotropic Gaussian regularisation (Balestriero & LeCun, Nov 2025)
LeJEPA argues that the optimal embedding distribution for downstream linear and k-NN tasks is an isotropic Gaussian, and enforces it with SIGReg: project the batch of embeddings onto M random unit directions (resampled each step), compare each 1-D projection's empirical characteristic function to the standard normal's with the Epps–Pulley statistic (a Gaussian-weighted L2 distance, integrated by the trapezoid rule on 17 points over [−5, 5]), and average over directions. The total loss is λ·SIGReg + (1 − λ)·prediction loss, where each view's embedding is pulled toward the mean embedding of the global views; λ ≈ 0.05. Stop-gradient, teacher–student EMA, predictors, negatives and schedulers are all removed, and the cost is linear in batch size.
- **Block:** `experimental.ssl.Sigreg` (slicing, Epps–Pulley statistic, trapezoid quadrature) and `experimental.ssl.LeJepa` (multi-view prediction to the global-view mean plus SIGReg). `Sigreg` is public: WP1's 10.1.6 reuses it.
- **Experiment:** `SelfSupervisedBenchmark` (`leJepa`, both on the MLP encoder and the patch transformer).
- **Toy task:** as 10.3.1 and 10.3.4.
- **Baselines:** I-JEPA-lite and the I-JEPA full recipe (existing rows); VICReg (another explicit regulariser); BYOL; DINO; LeJEPA with VICReg's variance–covariance term in place of SIGReg.
- **Expected metric:** linear-probe and k-NN accuracy, stability across λ ∈ [0.01, 0.2] and seeds, collapse rate, and whether training loss tracks probe accuracy. Hypothesis: LeJEPA beats both I-JEPA rows on this toy without any EMA teacher, never collapses, and its loss correlates with probe accuracy better than BYOL's or DINO's; VICReg in its place is less stable.
- **Source:** [arXiv:2511.08544](https://arxiv.org/abs/2511.08544)

### 10.3.7 Supervised contrastive learning (Khosla et al., NeurIPS 2020)
SupCon extends the InfoNCE loss to labels: every sample with the same label in the batch is a positive, all others negatives, with the loss averaged over positives outside the log. Representations trained this way and then probed linearly can beat cross-entropy and are more robust to hyperparameters and corruptions.
- **Block:** `experimental.metric.SupervisedContrastive` (multi-positive InfoNCE with temperature, two augmented views per sample).
- **Experiment:** `experiments.metric.MetricLearningShootout`.
- **Toy task:** (a) classification on `ShapeImages` and digits; (b) retrieval of unseen classes: train on 6 of 10 digit or glyph classes, retrieve on the other 4.
- **Baselines:** cross-entropy with the penultimate layer as embedding; SimCLR (`SelfSupervisedLosses`); `PrototypicalNetworks` episodic training.
- **Expected metric:** accuracy, Recall@1/@4 and NMI on unseen classes, and accuracy under `ShiftSuites`-style corruptions. Hypothesis: SupCon ties cross-entropy on accuracy but beats it on unseen-class retrieval and under corruption.
- **Source:** [arXiv:2004.11362](https://arxiv.org/abs/2004.11362)

### 10.3.8 Pair-based losses with mining: triplet with semi-hard negatives and the multi-similarity loss (Schroff et al., CVPR 2015; Wang et al., CVPR 2019)
The triplet loss requires an anchor to be closer to a positive than to a negative by a margin; which triplets are used matters more than the loss. FaceNet's semi-hard mining picks negatives farther than the positive but still within the margin. The multi-similarity loss mines informative pairs by comparing each pair to the hardest pairs of the opposite kind and weights them with a soft log-sum-exp over self- and relative similarities.
- **Block:** `experimental.metric.PairLosses` (contrastive, triplet with random/hard/semi-hard mining, multi-similarity with mining).
- **Experiment:** `MetricLearningShootout`.
- **Toy task:** as 10.3.7 (b).
- **Baselines:** 10.3.7; cross-entropy embeddings; `RelationalKdLoss`-style distance preservation (no labels).
- **Expected metric:** Recall@1 and NMI on unseen classes, with the fair-comparison protocol of Musgrave et al. (same embedding size, same steps, same augmentation). Hypothesis: with equal tuning, the methods land within a few points of each other, as Musgrave et al. found; semi-hard mining beats random triplets clearly.
- **Source:** [arXiv:1503.03832](https://arxiv.org/abs/1503.03832), [arXiv:1904.06627](https://arxiv.org/abs/1904.06627), [arXiv:2003.08505](https://arxiv.org/abs/2003.08505)

### 10.3.9 Angular-margin softmax: CosFace and ArcFace (Wang et al., CVPR 2018; Deng et al., CVPR 2019)
Margin softmax losses normalise both features and class weights, so logits are cosines scaled by s, and subtract a margin for the true class: CosFace subtracts m from the cosine, ArcFace adds m to the angle (cos(θ + m)). This pulls same-class embeddings into tight angular clusters and is the standard for face and speaker verification. Sub-center ArcFace gives each class several centres to absorb noisy labels.
- **Block:** `experimental.metric.MarginSoftmax` (normalised softmax, CosFace, ArcFace with the easy-margin fix, sub-center option). WP6's 10.6.5 reuses it for speakers.
- **Experiment:** `MetricLearningShootout`.
- **Toy task:** as 10.3.7 (b), plus a verification protocol (same/different pairs, equal error rate).
- **Baselines:** cross-entropy embeddings; normalised softmax without margin; 10.3.7–10.3.8.
- **Expected metric:** EER and Recall@1 on unseen classes; intra/inter-class angle statistics. Hypothesis: ArcFace and CosFace beat plain softmax on EER by a clear margin and match SupCon at lower cost (no pairs).
- **Source:** [arXiv:1801.09414](https://arxiv.org/abs/1801.09414), [arXiv:1801.07698](https://arxiv.org/abs/1801.07698)

### 10.3.10 Proxy-based losses: Proxy-NCA and Proxy-Anchor (Movshovitz-Attias et al., ICCV 2017; Kim et al., CVPR 2020)
Proxy methods learn one proxy vector per class and compare samples with proxies instead of other samples, which converges in far fewer steps and needs no mining. Proxy-NCA applies NCA between samples and proxies; Proxy-Anchor treats each proxy as an anchor and pulls all its positive samples (and pushes negatives) with a log-sum-exp that weights hard samples by their relative difficulty, combining proxy speed with pair-based data–data relations.
- **Block:** `experimental.metric.ProxyLosses` (Proxy-NCA, Proxy-NCA++, Proxy-Anchor).
- **Experiment:** `MetricLearningShootout`.
- **Toy task:** as 10.3.7 (b).
- **Baselines:** 10.3.7–10.3.9.
- **Expected metric:** Recall@1 against training steps (convergence speed) and final Recall@1. Hypothesis: Proxy-Anchor reaches its final Recall@1 in a third of the steps of the pair losses and ties or beats them at the end.
- **Source:** [arXiv:1703.07464](https://arxiv.org/abs/1703.07464), [arXiv:2003.13911](https://arxiv.org/abs/2003.13911)

### 10.3.11 Deep embedded clustering: DEC and IDEC (Xie et al., ICML 2016; Guo et al., IJCAI 2017)
DEC pretrains an autoencoder, initialises cluster centres with k-means in its latent space, then refines encoder and centres together by minimising the KL divergence between soft assignments q_ij (a Student-t kernel on the distance to each centre) and a sharpened target distribution p_ij ∝ q_ij² / f_j, which emphasises confident assignments and normalises cluster sizes. IDEC keeps the reconstruction loss during refinement so the latent space is not distorted.
- **Block:** `experimental.ml.DeepEmbeddedClustering` (Student-t soft assignment, target distribution, DEC and IDEC modes).
- **Experiment:** `experiments.metric.DeepClustering`.
- **Toy task:** digits, `ShapeImages` and `DigitGlyphs` without labels; scored against the true classes.
- **Baselines:** `KMeans`, `GaussianMixture` and `SpectralClustering` on raw pixels; k-means on the autoencoder latents (DEC's initialisation); `DirichletProcessMixture`.
- **Expected metric:** clustering accuracy (Hungarian-matched, reusing the solver from `MinibatchOptimalTransport`), NMI and ARI. Hypothesis: DEC beats k-means on latents by 5–10 points; IDEC is more stable across seeds.
- **Source:** [arXiv:1511.06335](https://arxiv.org/abs/1511.06335), [doi:10.24963/ijcai.2017/243](https://doi.org/10.24963/ijcai.2017/243)

### 10.3.12 Clustering from self-supervised features: SCAN and invariant information clustering (Van Gansbeke et al., ECCV 2020; Ji et al., ICCV 2019)
IIC trains a network with a softmax "cluster head" to maximise the mutual information between the cluster assignments of two augmented views of the same image, which rewards confident, balanced and augmentation-invariant clusters. SCAN first learns self-supervised features (SimCLR), mines each sample's k nearest neighbours in that space, and trains a cluster head so neighbours get the same assignment, with an entropy term for balance; a self-labelling stage then fine-tunes on confident samples.
- **Block:** `experimental.ml.NeighbourClustering` (IIC mutual-information loss with over-clustering head; SCAN neighbour-consistency loss plus self-labelling).
- **Experiment:** `DeepClustering`.
- **Toy task:** as 10.3.11, with SCAN using the frozen SimCLR encoder from `SelfSupervisedBenchmark`.
- **Baselines:** 10.3.11; `KMeans` on SimCLR features (SCAN's natural baseline); `SpectralClustering` on the same features.
- **Expected metric:** clustering accuracy, NMI and ARI. Hypothesis: SCAN beats k-means on the same SSL features by 5+ points and beats DEC; IIC is competitive on digits but sensitive to augmentation strength.
- **Source:** [arXiv:2005.12320](https://arxiv.org/abs/2005.12320), [arXiv:1807.06653](https://arxiv.org/abs/1807.06653)

### 10.3.13 Loss weighting for multi-task learning: uncertainty weighting, GradNorm and dynamic weight averaging (Kendall et al., CVPR 2018; Chen et al., ICML 2018; Liu et al., CVPR 2019)
Summing task losses lets the task with the largest loss or gradient dominate. Uncertainty weighting learns a homoscedastic noise σ_t per task and minimises Σ L_t/(2σ_t²) + log σ_t. GradNorm adjusts the weights so the gradient norms on the last shared layer track each task's relative training rate, with an asymmetry α. Dynamic weight averaging sets weights from the ratio of each task's last two losses, with no extra gradients.
- **Block:** `experimental.multitask.LossWeighting` (equal, uncertainty, GradNorm, DWA).
- **Experiment:** `experiments.multitask.GradientBalancing`.
- **Toy task:** (a) multi-label digits (class, parity, magnitude, stroke count; classification and regression heads on a shared trunk); (b) `ClickLogs`' two-label task at correlation −0.5 from wave 9; (c) a synthetic two-regression problem with a 100x loss-scale mismatch.
- **Baselines:** equal weights; single-task models; wave 9's `MultiGateExperts` MMoE and PLE (architectural answers to the same conflict).
- **Expected metric:** per-task metrics and the average relative improvement Δm over single-task models. Hypothesis: with a scale mismatch, every weighting beats equal weights; on (a)–(b) uncertainty weighting and DWA are close to GradNorm at lower cost; MMoE and weighting gains add up.
- **Source:** [arXiv:1705.07115](https://arxiv.org/abs/1705.07115), [arXiv:1711.02257](https://arxiv.org/abs/1711.02257), [arXiv:1803.10704](https://arxiv.org/abs/1803.10704)

### 10.3.14 Gradient surgery: PCGrad and GradOPS (Yu et al., NeurIPS 2020; Alibaba, WSDM 2025)
When task gradients conflict (negative cosine), summing them can harm one task. PCGrad projects each task gradient onto the normal plane of every gradient it conflicts with, in random order. GradOPS projects each conflicting gradient onto the subspace orthogonal to the span of all other task gradients (via Gram–Schmidt), so every pair of modified gradients is non-conflicting regardless of order, and then reweights them with one hyperparameter α that trades off dominated and dominating tasks.
- **Block:** `experimental.multitask.GradientSurgery` (PCGrad, GradVac-style cosine targets, GradOPS), acting on per-task gradients of the shared parameters.
- **Experiment:** `GradientBalancing`.
- **Toy task:** as 10.3.13, plus the classic two-objective toy loss surface of Yu et al. (where plain summation gets stuck).
- **Baselines:** 10.3.13's weightings; equal weights; single-task models.
- **Expected metric:** Δm, the share of steps with conflicting gradients, and cost per step (one backward per task). Hypothesis: on the toy surface PCGrad and GradOPS reach the Pareto front where summing stalls; on the real tasks the gain is small (1–2 points of Δm) and GradOPS is less order-dependent than PCGrad with 4 tasks.
- **Source:** [arXiv:2001.06782](https://arxiv.org/abs/2001.06782), [arXiv:2503.03438](https://arxiv.org/abs/2503.03438)

### 10.3.15 Multi-objective descent: MGDA-UB, CAGrad and Nash-MTL (Sener & Koltun, NeurIPS 2018; Liu et al., NeurIPS 2021; Navon et al., ICML 2022)
MGDA finds the min-norm convex combination of task gradients, a common descent direction that converges to a Pareto-stationary point (the "upper bound" variant uses gradients of the shared representation to need one backward pass). CAGrad maximises the worst task's local improvement within a ball around the average gradient, so it still makes progress on the average loss. Nash-MTL treats the update as a bargaining game and finds the Nash bargaining solution, which is scale-invariant across tasks.
- **Block:** `experimental.multitask.ParetoDescent` (Frank–Wolfe min-norm solver, CAGrad's dual, Nash-MTL's iterative solution), sharing the per-task gradient plumbing of 10.3.14.
- **Experiment:** `GradientBalancing`.
- **Toy task:** as 10.3.14.
- **Baselines:** 10.3.13–10.3.14; equal weights.
- **Expected metric:** Δm, worst-task metric, and cost per step. Hypothesis: MGDA over-favours the easiest task and lowers Δm; CAGrad and Nash-MTL give the best Δm in this lineup, at about T times the backward cost.
- **Source:** [arXiv:1810.04650](https://arxiv.org/abs/1810.04650), [arXiv:2110.14048](https://arxiv.org/abs/2110.14048), [arXiv:2202.01017](https://arxiv.org/abs/2202.01017)

### 10.3.16 FAMO: fast adaptive multitask optimisation (Liu et al., NeurIPS 2023)
The gradient-manipulation methods of 10.3.14–10.3.15 need one backward pass per task. FAMO aims for the same balanced outcome (an equal rate of improvement in every task's log-loss) with O(1) space and time per step: it keeps task logits ξ, takes one step on the weighted sum of losses, and then updates ξ from the observed change in each log-loss, a cheap online proxy for the per-task gradients.
- **Block:** `experimental.multitask.Famo` (weight logits, single-backward update, log-loss change update).
- **Experiment:** `GradientBalancing`.
- **Toy task:** as 10.3.14, with 4 and 8 tasks to show the cost scaling.
- **Baselines:** 10.3.13–10.3.15.
- **Expected metric:** Δm against wall-clock per step. Hypothesis: FAMO comes within 1 point of Nash-MTL's Δm at the cost of equal weighting; the gap in cost grows linearly with the number of tasks.
- **Source:** [arXiv:2306.03792](https://arxiv.org/abs/2306.03792)

### 10.3.17 Unsupervised truth directions revisited: CCS with normalisation, prompt ensembles and CRC (Burns et al., ICLR 2023; Farquhar et al., 2023) — follow-up
Wave 9's CCS scored 0.575 on plain statements, the same as a random direction (0.570), while a supervised probe reached 0.780 and zero-shot answers 0.930. Two parts of the original recipe were missing: per-template mean normalisation of the contrast pair's hidden states (which removes the "is this the yes-version" direction that otherwise dominates) and averaging over several prompt templates; Burns et al. also report contrastive representation clustering (CRC, the top principal component of normalised pair differences) as a simpler alternative. Farquhar et al. argue that CCS finds whatever binary feature is most salient, so a distractor-feature split is the key control.
- **Block:** additions to `experimental.interp.ContrastConsistentSearch` (per-template normalisation, prompt-ensemble averaging, CRC; defaults unchanged).
- **Experiment:** `experiments.interp.LatentKnowledge.followUp` (wave 9 rows unchanged).
- **Toy task:** wave 9's verdict-token task, trained 4x longer, with 1, 4 and 8 templates; plain, sycophantic and distractor splits.
- **Baselines:** wave 9's CCS, contrastive PCA, supervised probe, zero-shot answers and random direction.
- **Expected metric:** accuracy on each split, and agreement with the distractor feature. Hypothesis: normalisation lifts CCS clearly above the random direction on plain statements (towards the supervised probe); on the distractor split CCS and CRC follow the distractor, as Farquhar et al. predict, so the method stays unreliable.
- **Source:** [arXiv:2212.03827](https://arxiv.org/abs/2212.03827), [arXiv:2312.10029](https://arxiv.org/abs/2312.10029)

---

## WP4. Deep RL recipes 2023–2025, plasticity and scaling, and RL/post-training follow-ups (17)

Target: one new package, `experimental.deeprl` (10.4.2–10.4.12, 10.4.17); TD3+BC (10.4.1) and the imitation follow-up (10.4.13) add to the existing `experimental.rl`, and the post-training follow-ups (10.4.15–10.4.16) add to the existing `experimental.align`. Measurements: new `experiments.control.ModernOffPolicy` (10.4.2–10.4.4, 10.4.6–10.4.7, 10.4.14), `experiments.control.ValueBasedRecipes` (10.4.5, 10.4.8–10.4.9), `experiments.control.PlasticityAndScale` (10.4.10–10.4.12, 10.4.17), and extensions of `OfflineRlComparison` (10.4.1), `GoalsImitationAndTeams` (10.4.13), `experiments.alignment.RewardOveroptimization` (10.4.15) and `experiments.reasoning.ProcessSupervision` (10.4.16). Environments are the `brainiac-rl` ones (CartPole, Pendulum, MountainCar continuous, GridWorld) plus a pixel version of GridWorld/`KeyDoorGrid` rendered as `DreamingAgent` does, and `OfflineDataset`'s Pendulum and grid datasets. Test classes use short budgets (2–5k environment steps, about 10 s) and assert only sanity; the learning-curve comparisons at 20–100k steps run in each experiment's `main` and are reported as a Run, as wave 8 did for its long budgets. Every comparison is at equal environment steps and reports gradient updates and wall-clock too, over at least 5 seeds with interquartile means.

### 10.4.1 TD3+BC: a minimalist offline RL baseline (Fujimoto & Gu, NeurIPS 2021)
TD3+BC adds one behaviour-cloning term to TD3's actor loss, λ·Q(s, π(s))/mean|Q| − (π(s) − a)², and normalises states. The Q normalisation makes λ scale-free. With this two-line change TD3 matches far more complex offline methods, which makes it the standard sanity baseline.
- **Block:** `experimental.rl.Td3Bc` (on `Td3`; Q-normalised BC term, state normalisation).
- **Experiment:** `experiments.control.OfflineRlComparison` (new rows; existing rows unchanged).
- **Toy task:** `OfflineDataset`'s Pendulum random, medium, medium-replay and expert datasets.
- **Baselines:** `Cql`, `Iql`, `FlowQLearning` (FQL), `BehaviorCloning`, `DecisionTransformer`.
- **Expected metric:** normalised return per dataset and training wall-clock. Hypothesis: TD3+BC matches IQL within noise on medium and expert data at a fraction of CQL's cost, and trails CQL on random data, where BC is a bad anchor.
- **Source:** [arXiv:2106.06860](https://arxiv.org/abs/2106.06860)

### 10.4.2 CrossQ: batch normalisation instead of target networks (Bhatt et al., ICLR 2024)
Batch normalisation was thought to break off-policy critics because the statistics of (s, a) and (s′, π(s′)) batches differ. CrossQ concatenates the current and next state–action batches and passes them through the critic in one forward pass, so BN normalises both jointly; it then drops the target network entirely and widens the critic. At an update-to-data ratio of 1 it matches REDQ and DroQ, which need about 20 updates per step.
- **Block:** `experimental.deeprl.CrossQ` (joint-batch BN critic using `BatchRenorm`, no target network, wider critic), on `Sac`.
- **Experiment:** `experiments.control.ModernOffPolicy`.
- **Toy task:** Pendulum and continuous MountainCar; learning curves to 20k steps (Run) and a 3k-step test.
- **Baselines:** `Sac`; `HighUtdSac` (REDQ and DroQ at UTD 10–20); `Td3`.
- **Expected metric:** return at 5k/10k/20k steps and gradient updates per environment step. Hypothesis: CrossQ at UTD 1 matches DroQ at UTD 20 in return per environment step, at about a tenth of the compute; SAC with naive BN in the critic diverges.
- **Source:** [arXiv:1902.05605](https://arxiv.org/abs/1902.05605)

### 10.4.3 BRO: bigger, regularised, optimistic critics (Nauman et al., NeurIPS 2024)
BRO shows that critic size helps sample efficiency only with strong regularisation. Its BroNet critic is a residual MLP with LayerNorm after every dense layer (default width 512, two residual blocks), trained with weight decay and periodic full-parameter resets; clipped double-Q is turned off in favour of the ensemble mean, the critic predicts return quantiles, and exploration uses a separate optimistic actor trained toward an upper bound (mean plus an epistemic term from the critic ensemble), held near the pessimistic actor by a KL penalty with a target of 0.05. It uses a replay ratio of 10 (BRO-Fast: 2).
- **Block:** `experimental.deeprl.Bro` (BroNet critic and actor, quantile heads, dual optimistic/pessimistic actors with adaptive KL weight, reset schedule).
- **Experiment:** `ModernOffPolicy`.
- **Toy task:** as 10.4.2, with critic widths 64, 256 and 512.
- **Baselines:** `Sac`; `ResidualRlNetworks` (SimBa with HL-Gauss, wave 5); `HighUtdSac`; 10.4.2.
- **Expected metric:** return against critic width and against replay ratio; ablations of optimism and resets. Hypothesis: plain SAC gets worse or no better as the critic grows, while BRO improves with width; optimism helps on MountainCar (sparse reward) and is neutral on Pendulum.
- **Source:** [arXiv:2405.16158](https://arxiv.org/abs/2405.16158)

### 10.4.4 SimbaV2: hyperspherical normalisation for scalable RL (Lee et al., ICML 2025)
SimbaV2 replaces SimBa's LayerNorm with ℓ2 normalisation that keeps features on the unit hypersphere and projects weights back onto the unit sphere after every update (removing the need for weight decay and keeping effective learning rates stable). Inputs get an extra constant coordinate before normalisation so magnitude is not lost; each linear layer has a learnable per-channel scaler and no bias; residuals are a learnable linear interpolation (LERP) followed by re-normalisation. The critic is categorical (KL loss), rewards are scaled by running return statistics and bounded, and the learning rate decays linearly.
- **Block:** `experimental.deeprl.SimbaV2` (hypersphere-normalised linear, scaler, LERP residual, weight projection step, categorical critic, return-based reward scaling).
- **Experiment:** `ModernOffPolicy`.
- **Toy task:** as 10.4.2, at UTD 1, 2 and 8 and widths 64–512.
- **Baselines:** `ResidualRlNetworks` (SimBa), 10.4.3 BRO, 10.4.2 CrossQ, `Sac`.
- **Expected metric:** return against UTD and width; the growth of weight and feature norms and of the effective learning rate. Hypothesis: SimbaV2 keeps improving as UTD and width grow where SimBa plateaus or degrades, and its norms stay flat.
- **Source:** [arXiv:2502.15280](https://arxiv.org/abs/2502.15280)

### 10.4.5 PQN: parallelised Q-learning with LayerNorm and no replay buffer (Gallici et al., ICLR 2025)
Gallici et al. prove that LayerNorm (plus ℓ2 regularisation) makes TD learning convergent even with off-policy data and nonlinear function approximation, and then remove the stabilisers DQN needed: there is no target network and no replay buffer. Instead PQN collects short rollouts from many vectorised environments, computes λ-returns over them, and takes a few epochs of minibatch updates, like PPO but with a Q-network.
- **Block:** `experimental.deeprl.Pqn` (LayerNorm Q-network, vectorised environment runner on threads, Q(λ) targets, ε-greedy schedule).
- **Experiment:** `experiments.control.ValueBasedRecipes`.
- **Toy task:** CartPole and GridWorld with 8–64 parallel environments; also the pixel GridWorld.
- **Baselines:** core `Dqn` with replay and target network; `RainbowDqn`; core `Ppo` at the same environment steps.
- **Expected metric:** return against environment steps and wall-clock; ablations of LayerNorm and λ. Hypothesis: PQN matches DQN per environment step and wins clearly per wall-clock second; removing LayerNorm without a target network diverges.
- **Source:** [arXiv:2407.04811](https://arxiv.org/abs/2407.04811)

### 10.4.6 MR.Q: model-based representations for model-free Q-learning (Fujimoto et al., ICLR 2025)
MR.Q learns a state encoder z_s and a state–action encoder z_sa whose features make the value function approximately linear. The encoders are trained over a short unrolled horizon with three model-based losses on z_sa through a linear predictor: next-state embedding (MSE to a target encoder), reward (two-hot cross-entropy with symexp-spaced bins) and termination. The value network is TD3-style (two critics, clipped target-policy noise) on multi-step returns with a Huber loss, rewards are normalised by the replay buffer's mean absolute reward, sampling uses LAP prioritisation, and discrete actions use Gumbel-softmax, all with one hyperparameter set across benchmarks.
- **Block:** `experimental.deeprl.MrQ` (encoders, linear MDP predictor, two-hot reward head reusing `DreamerAgent`'s symlog/two-hot helpers, LAP replay, TD3 value and policy).
- **Experiment:** `ModernOffPolicy` (continuous) and `ValueBasedRecipes` (discrete, pixel GridWorld).
- **Toy task:** Pendulum, MountainCar continuous, CartPole and pixel GridWorld with one shared configuration.
- **Baselines:** `Td3`, `Sac`, core `Dqn`, `TdMpc` (TD-MPC2) and `DreamerAgent` (model-based with planning or imagination).
- **Expected metric:** return per environment step on every task with one configuration; ablation without the model-based losses. Hypothesis: MR.Q is never the worst and is best or second on most tasks with one hyperparameter set, while each baseline needs per-task tuning; without the model losses it falls to TD3's level.
- **Source:** [arXiv:2501.16142](https://arxiv.org/abs/2501.16142)

### 10.4.7 TD7: SALE embeddings, LAP and policy checkpoints (Fujimoto et al., NeurIPS 2023)
TD7 is MR.Q's predecessor. It adds to TD3 state–action learned embeddings (SALE: an encoder trained to predict the next state's embedding, whose outputs are concatenated to the critic's and actor's inputs), loss-adjusted prioritised replay (LAP), value clipping from replay-buffer bounds, and policy checkpoints, which evaluate a frozen policy for several episodes and keep the best one, to remove the instability of deploying the latest actor.
- **Block:** `experimental.deeprl.Td7` (SALE encoder, LAP buffer shared with 10.4.6, checkpoint scheduler).
- **Experiment:** `ModernOffPolicy`.
- **Toy task:** as 10.4.2.
- **Baselines:** `Td3`; 10.4.6 MR.Q.
- **Expected metric:** return and its variance across evaluations; ablation of each component. Hypothesis: SALE gives most of the gain over TD3; checkpoints mainly cut the variance of the deployed policy.
- **Source:** [arXiv:2306.02451](https://arxiv.org/abs/2306.02451)

### 10.4.8 Self-predictive representations and BBF (Schwarzer et al., ICLR 2021; ICML 2023)
SPR adds to a DQN-style agent an auxiliary loss: a transition model, rolled forward k steps in latent space from the online encoder and the actions taken, must predict (by cosine similarity after projection) the EMA target encoder's embeddings of the future observations. BBF scales this up for sample efficiency: a wider network, a high replay ratio with periodic shrink-and-perturb resets, an update horizon annealed from 10 to 3, a discount annealed from 0.97 to 0.997, and weight decay.
- **Block:** `experimental.deeprl.SelfPredictive` (latent transition model, k-step cosine loss with `MomentumTeacher` targets; BBF's resets, n-step and γ schedules).
- **Experiment:** `ValueBasedRecipes` (`sampleEfficient`).
- **Toy task:** pixel GridWorld and `KeyDoorGrid` at a 10k-step budget ("Atari 100k" in miniature).
- **Baselines:** `RainbowDqn`; core `Dqn`; 10.4.9's DrQ; `DreamerAgent`.
- **Expected metric:** return at 10k steps; ablations of the SPR loss and of resets. Hypothesis: SPR beats Rainbow at 10k steps by improving the encoder; BBF's resets and schedules add more at high replay ratios, and without resets a high replay ratio hurts.
- **Source:** [arXiv:2007.05929](https://arxiv.org/abs/2007.05929), [arXiv:2305.19452](https://arxiv.org/abs/2305.19452)

### 10.4.9 DrQ and DrQ-v2: data-augmented RL from pixels (Kostrikov et al., ICLR 2021; Yarats et al., ICLR 2022)
DrQ applies random-shift augmentation (pad by 4 pixels, random crop) to observations and averages the Q-target over K augmentations and the Q-estimate over M augmentations, regularising the critic without any auxiliary loss. DrQ-v2 moves to DDPG with n-step returns, a scheduled exploration noise and bilinear-interpolated shifts.
- **Block:** `experimental.deeprl.DrQ` (random-shift augmentation as an observation transform, K/M target averaging), usable with `Dqn`, `Sac` and 10.4.6.
- **Experiment:** `ValueBasedRecipes` (`sampleEfficient`).
- **Toy task:** as 10.4.8; also a pixel Pendulum rendered as an image for the continuous DrQ-v2 row.
- **Baselines:** the same agents without augmentation; `ShortcutDreaming`/`DreamerAgent` on pixels; 10.4.8.
- **Expected metric:** return at 10k steps. Hypothesis: augmentation alone gives most of SPR's gain; SPR plus DrQ adds a little more.
- **Source:** [arXiv:2004.13649](https://arxiv.org/abs/2004.13649), [arXiv:2107.09645](https://arxiv.org/abs/2107.09645)

### 10.4.10 Plasticity loss: ReDo, plasticity injection and periodic resets (Sokar et al., ICML 2023; Nikishin et al., NeurIPS 2023 and ICML 2022)
Deep RL networks gradually lose their ability to fit new targets. ReDo detects dormant neurons, whose normalised mean activation falls below τ, and recycles them: incoming weights are re-initialised and outgoing weights zeroed, so the function is unchanged. Plasticity injection freezes the network and adds a fresh copy minus a frozen copy of the head, which adds trainable capacity without changing the output. Periodic resets re-initialise the last layers (or all of them) every N steps while keeping the replay buffer.
- **Block:** `experimental.deeprl.Plasticity` (dormant-neuron measurement and ReDo, plasticity injection, reset schedules, plus diagnostics: feature rank and weight norm).
- **Experiment:** `experiments.control.PlasticityAndScale`.
- **Toy task:** CartPole and pixel GridWorld at replay ratios 1, 4 and 16; a task-switch variant (rewards change mid-training) as the clearest test.
- **Baselines:** the same agents without intervention; `WarmStartGap`'s shrink-and-perturb; 10.4.8's BBF resets.
- **Expected metric:** return against replay ratio, dormant-neuron share and feature rank over training, and return after the task switch. Hypothesis: dormant neurons grow with the replay ratio; ReDo and resets recover return at high replay ratios and after the task switch; injection is the cheapest fix.
- **Source:** [arXiv:2302.12902](https://arxiv.org/abs/2302.12902), [arXiv:2305.15555](https://arxiv.org/abs/2305.15555), [arXiv:2205.07802](https://arxiv.org/abs/2205.07802)

### 10.4.11 Stable gradients at scale: multi-skip residuals and Kronecker-factored optimisation for deep RL (Castanyer et al., 2025)
Castanyer et al. find that deep RL agents degrade as networks grow deeper or wider because gradients become unstable under non-stationary targets. They broadcast the encoder's features directly to every subsequent MLP layer (multi-skip residuals) and optimise with the Kron Kronecker-factored preconditioner; only the combination lets value-based, policy-gradient, actor-critic and offline agents keep improving with scale.
- **Block:** `experimental.deeprl.MultiSkipNetwork` (encoder-to-every-layer skips), with `PsgdKron` as the optimiser.
- **Experiment:** `PlasticityAndScale` (`depth`).
- **Toy task:** CartPole (DQN and PPO) and Pendulum (SAC) at depths 2, 4, 8 and 16.
- **Baselines:** plain MLP and SimBa residual MLP (`ResidualRlNetworks`) with Adam; each intervention alone.
- **Expected metric:** return against depth; gradient-norm spread across layers. Hypothesis: plain MLPs degrade beyond depth 4; multi-skip plus Kron holds or improves with depth; either alone gives part of it.
- **Source:** [arXiv:2506.15544](https://arxiv.org/abs/2506.15544)

### 10.4.12 EfficientZero: sample-efficient MuZero (Ye et al., NeurIPS 2021)
EfficientZero adds three things to MuZero. A self-supervised consistency loss (SimSiam-style, with stop-gradient) makes the dynamics model's predicted latent match the encoder's latent of the real next observation. A value-prefix LSTM predicts the cumulative reward over the unrolled horizon instead of each step's reward, which removes the hard problem of predicting exactly when a reward arrives. Off-policy correction shortens the n-step bootstrap horizon for old trajectories and re-runs MCTS on them for fresh value targets.
- **Block:** `experimental.deeprl.EfficientZero` (on `MuZeroLite`; consistency loss, value-prefix head with `RlGruCell`, adaptive horizon reanalysis).
- **Experiment:** `ValueBasedRecipes` (`modelBased`).
- **Toy task:** pixel GridWorld and CartPole at 5–10k steps.
- **Baselines:** `MuZeroLite`; `GumbelMcts`; 10.4.8 SPR; `DreamerAgent`.
- **Expected metric:** return at 10k steps; ablations of each component. Hypothesis: the consistency loss gives most of the gain over MuZero-lite; the value prefix matters on the sparse-reward grid.
- **Source:** [arXiv:2111.00210](https://arxiv.org/abs/2111.00210)

### 10.4.13 Adversarial imitation that works: AIRL, IQ-Learn and GAIL with a regularised discriminator (Fu et al., ICLR 2018; Garg et al., NeurIPS 2021; Kostrikov et al., ICLR 2019) — follow-up
Wave 1's GAIL scored −0.20 against −0.09 for random and 0.84 for behaviour cloning in `GoalsImitationAndTeams`, though it passed its block-level test. Three recipes address the usual failure modes. A gradient penalty on the discriminator and absorbing-state handling (DAC) stop the discriminator from saturating and fix the survival bias of its reward. AIRL restricts the discriminator to the form exp(f)/(exp(f) + π) with f = g(s) + γh(s′) − h(s), which recovers a shaped reward that transfers. IQ-Learn avoids the adversary altogether by learning a single soft Q-function with an inverse soft-Bellman objective, from which policy and reward follow.
- **Block:** `experimental.rl.AdversarialImitation` (GAIL with gradient penalty and absorbing states, AIRL discriminator, IQ-Learn objective); `Gail` is unchanged.
- **Experiment:** `experiments.control.GoalsImitationAndTeams.imitationFollowUp` (wave 1 rows unchanged).
- **Toy task:** the same grid and CartPole demonstration sets, with 1, 5 and 20 demonstrations, and a transfer variant (changed dynamics) for AIRL's reward.
- **Baselines:** `Gail` as it is; `BehaviorCloning`; `MaxEntIrl`; random.
- **Expected metric:** return against number of demonstrations; return of a policy re-trained on the recovered reward under changed dynamics. Hypothesis: regularised GAIL and IQ-Learn beat BC at 1–5 demonstrations, where BC suffers from compounding errors; AIRL's reward transfers where GAIL's does not.
- **Source:** [arXiv:1710.11248](https://arxiv.org/abs/1710.11248), [arXiv:2106.12142](https://arxiv.org/abs/2106.12142), [arXiv:1809.02925](https://arxiv.org/abs/1809.02925)

### 10.4.14 Continuous control at a fair budget, with the modern recipes — follow-up
In `ContinuousControlComparison` at 400 steps, PPO (−322), CEM-MPC (−323), SAC (−334), DDPG (−373) and TD3 (−391) were all near random (−334): the budget was far too short to rank anything. This item re-runs the lineup on Pendulum and continuous MountainCar at 5k, 20k and 50k steps with fixed seeds and interquartile means, and adds 10.4.2–10.4.7 to the same harness, so wave 10's recipes and the old baselines are compared on one curve. It also reports the rliable-style statistics (stratified bootstrap confidence intervals, probability of improvement).
- **Block:** none new; a `LearningCurveHarness` helper in `experimental.deeprl` (equal environment steps, IQM, bootstrap intervals; it can reuse wave 9's `EvalStatistics`).
- **Experiment:** `ModernOffPolicy.fairBudget` (the wave 1 class is not changed).
- **Toy task:** Pendulum and continuous MountainCar.
- **Baselines:** `ContinuousControlComparison`'s PPO, SAC, TD3, DDPG-style and CEM-MPC agents.
- **Expected metric:** IQM return with 95% intervals at each budget, and ranking stability across budgets (Kendall τ). Hypothesis: SAC and TD3 clearly beat random by 20k steps; the modern recipes lead at 5k steps and the gap narrows at 50k.
- **Source:** [arXiv:2108.13264](https://arxiv.org/abs/2108.13264)

### 10.4.15 Reward-model ensembles and WARM with stronger proxies (Coste et al., ICLR 2024; Ramé et al., ICML 2024) — follow-up
Wave 9's single proxy reward model reached 0.586 preference accuracy and the WARM members 0.521, so the ensembles averaged near-random members and lost gold reward (mean ensemble 0.333 after GRPO, single proxy 0.487, WARM 0.491). The over-optimisation result held, but the ensemble claims were not tested fairly. This follow-up trains proxies to 0.70–0.80 accuracy (8x the preference labels and 4x the steps, shared pretrained trunk fine-tuned from one start for WARM, as the paper requires linear mode connectivity), sweeps the number of labels so that proxy quality is a controlled variable, and adds uncertainty-weighted optimisation with a KL penalty.
- **Block:** options on `experimental.align.RewardEnsemble` and `WeightAveragedReward` (shared-trunk fine-tuning from one start, label-budget sweep); defaults unchanged.
- **Experiment:** `experiments.alignment.RewardOveroptimization.strongProxies` (wave 9 rows unchanged).
- **Toy task:** wave 9's `GoldRewardHarness` (gold = `RubricRewards`' true score) at label budgets 1x, 4x and 8x.
- **Baselines:** wave 9's single proxy, mean/worst/mean-minus-std ensembles, WARM and the gold oracle.
- **Expected metric:** proxy accuracy, gold reward against KL from the reference for best-of-N and GRPO, and the peak-to-final drop. Hypothesis: once members exceed about 0.7 accuracy, worst-case and uncertainty-penalised ensembles and WARM delay over-optimisation and end with higher gold than a single proxy, as in the papers.
- **Source:** [arXiv:2310.02743](https://arxiv.org/abs/2310.02743), [arXiv:2401.12187](https://arxiv.org/abs/2401.12187)

### 10.4.16 VinePPO, PRIME and Step-DPO at a larger budget on a task that generalises (Kazemnejad et al., 2024; Cui et al., 2025; Lai et al., 2024) — follow-up
In wave 9 all three lost to their baselines: VinePPO 0.200 against PPO 0.308 (although its value estimates were three times more accurate), PRIME 0.277 against RLOO 0.275, and Step-DPO 0.180 against solution-level DPO 0.187 and STaR 0.312. The builder noted that the toy `WindowLm` memorised rather than generalised running sums (0.3 held-out against 0.8 on training problems) and that greedy accuracy dipped under RL. This follow-up uses a 4x larger step budget and a start model that generalises (pre-trained on 4x more `StepChainTask` problems until held-out accuracy is above 0.5), reports held-out accuracy, and matches token budgets across methods.
- **Block:** none new; options on `MonteCarloValuePpo`, `ImplicitProcessReward` and `StepPreferenceBuilder` (rollout counts, start checkpoint), defaults unchanged.
- **Experiment:** `experiments.reasoning.ProcessSupervision.largerBudget` (wave 9 rows unchanged).
- **Toy task:** `StepChainTask` with 3–4 operands and a held-out split.
- **Baselines:** wave 9's PPO with a learned critic, GRPO, RLOO, solution-level DPO, rejection-sampling fine-tuning and STaR, at the same token budget.
- **Expected metric:** held-out sampled accuracy against tokens generated; value-estimate error for VinePPO. Hypothesis: with a start that generalises, VinePPO's better values turn into a policy gain over PPO, and Step-DPO beats solution-level DPO on longer chains; PRIME stays tied with RLOO.
- **Source:** [arXiv:2410.01679](https://arxiv.org/abs/2410.01679), [arXiv:2502.01456](https://arxiv.org/abs/2502.01456), [arXiv:2406.18629](https://arxiv.org/abs/2406.18629)

### 10.4.17 Value-based RL scaling laws: TD-overfitting, batch size and model size (Rybkin et al., ICML 2025; Fu et al., NeurIPS 2025)
Two 2025 papers argue that value-based RL scales predictably once the right hyperparameters are scaled together. Rybkin et al. fit power laws for the data and compute needed to reach a return, and for the best batch size and learning rate as functions of the update-to-data ratio. Fu et al. identify TD-overfitting: for small critics, larger batches make Q-accuracy worse because low-variance gradients fit noisy, non-stationary TD targets; with large critics, large batches are fine. A "passive critic" experiment shows the culprit is target quality, not memorisation capacity.
- **Block:** `experimental.deeprl.ValueScalingStudy` (grids over critic width, batch size, UTD ratio and learning rate; held-out TD error; passive-critic training on another critic's targets; power-law fitting via wave 9's `CurveExtrapolator`).
- **Experiment:** `PlasticityAndScale` (`scaling`).
- **Toy task:** SAC on Pendulum and DQN on CartPole with critic widths 32–512 and batch sizes 32–1024 (the full grid is a Run; the test uses a 3x3 corner).
- **Baselines:** the default configurations of `Sac` and `Dqn`; 10.4.3 BRO and 10.4.4 SimbaV2 as "regularised" points.
- **Expected metric:** held-out TD error and return across the grid; the fitted best batch size against width; passive-critic results. Hypothesis: for narrow critics held-out TD error rises with batch size (TD-overfitting) while wide critics benefit; targets from a wide critic let a narrow passive critic fit well.
- **Source:** [arXiv:2502.04327](https://arxiv.org/abs/2502.04327), [arXiv:2508.14881](https://arxiv.org/abs/2508.14881)

---

## WP5. Classic and structured ML: boosting variants, temporal point processes, irregular series and imputation, temporal/relational/hyperbolic graphs, parsing (17)

Target: three new packages, `experimental.boost` (10.5.1–10.5.6), `experimental.events` (10.5.7–10.5.11) and `experimental.parse` (10.5.16–10.5.17); graph models (10.5.12–10.5.15) add to the existing `experimental.graph`. Measurements: `experiments.classic.BoostingShootout` (new, 10.5.1–10.5.6), new theme `experiments.events` (`PointProcessShootout` for 10.5.7–10.5.9, `IrregularSeriesShootout` for 10.5.10–10.5.11), `experiments.graphs.TemporalGraphs` (10.5.12) and `RelationalGraphs` (10.5.13–10.5.15), and new theme `experiments.parsing` (`ParsingShootout`, 10.5.16–10.5.17). Four new generators in `experimental.data`: `EventStreams` (multivariate Hawkes processes with exponential, power-law and non-monotone kernels, self-correcting and inhibitory processes, all simulated by Ogata thinning so the true log-likelihood is known), `IrregularSeries` (`SeasonalSeries` and `ChaoticSeries` sampled at random times with 30–90% of values missing, missing at random or not at random), `TemporalGraphs` (timestamped edge streams with controlled recency, periodicity, preferential attachment and homophily, following the probe design of Hayes et al., 2025) and `ToyTreebank` (a small probabilistic context-free grammar with nested clauses, prepositional-phrase attachment ambiguity and coordination, giving sentences with gold constituency trees and head-rule dependency trees). Tabular items use `TabularStressSuite` and `TabularPrior`. Test classes stay near 10 s by using a few hundred rows, sequences or sentences.

### 10.5.1 LightGBM's GOSS, exclusive feature bundling and leaf-wise growth (Ke et al., NeurIPS 2017)
Core `GradientBoostedTrees` already does second-order boosting on quantile bins, level by level. LightGBM adds three things. Gradient-based one-side sampling keeps the top a% of rows by gradient magnitude and a random b% of the rest, up-weighting the sampled small-gradient rows by (1 − a)/b, which keeps the gain estimate nearly unbiased at a fraction of the rows. Exclusive feature bundling merges sparse features that are rarely non-zero together into one histogram feature (greedy graph colouring on the conflict graph). Leaf-wise (best-first) growth splits the leaf with the largest gain anywhere in the tree under a leaf budget, rather than a whole level at a time.
- **Block:** `experimental.boost.LightBoost` (GOSS, EFB, leaf-wise growth with max-leaves, histogram subtraction), reusing `GradientBoostedTrees.Loss`.
- **Experiment:** `experiments.classic.BoostingShootout`.
- **Toy task:** `TabularStressSuite` tables at 2k–50k rows (the 50k rows only in `main`), plus a one-hot-heavy sparse variant from wave 9's `ClickLogs` features.
- **Baselines:** core `GradientBoostedTrees` (level-wise, all rows); `RandomForest`; `RegressionForest`.
- **Expected metric:** accuracy/AUC or RMSE against training time and number of leaves. Hypothesis: GOSS with a = 0.2, b = 0.1 matches full-data accuracy at about a third of the time; EFB speeds up the sparse variant several-fold; leaf-wise growth reaches the same accuracy with fewer leaves but overfits small tables without a depth cap.
- **Source:** [NeurIPS 2017](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)

### 10.5.2 CatBoost: ordered boosting and ordered target statistics (Prokhorenkova et al., NeurIPS 2018)
Target encoding of categorical features (replacing a category by its mean target) leaks the label and so does standard boosting, whose residuals at a row come from trees that were fit on that row: both cause a prediction shift. CatBoost draws random permutations and, for each row, computes the target statistic and the residual only from rows earlier in the permutation (ordered TS and ordered boosting). It also uses oblivious (symmetric) trees, where every node at one depth uses the same split, which are fast to evaluate and regularise.
- **Block:** `experimental.boost.OrderedBoosting` (ordered target statistics with a prior, ordered boosting with several permutations, oblivious trees, combinations of categorical features).
- **Experiment:** `BoostingShootout` (`categorical`).
- **Toy task:** `TabularStressSuite` with high-cardinality categorical columns (100–5,000 levels, Zipf frequencies) and a planted interaction between two categoricals.
- **Baselines:** core GBT with one-hot encoding; GBT with naive (leaky) target encoding; GBT with out-of-fold target encoding; wave 9's `CompactEmbedding` hashed embeddings in an MLP; `RandomForest`.
- **Expected metric:** test log-loss/AUC and the train–test gap. Hypothesis: naive target encoding overfits badly on high-cardinality columns; ordered TS removes the gap and beats one-hot and out-of-fold encoding; oblivious trees cost little accuracy.
- **Source:** [arXiv:1706.09516](https://arxiv.org/abs/1706.09516)

### 10.5.3 NGBoost: natural-gradient boosting for probabilistic prediction (Duan et al., ICML 2020)
NGBoost predicts the parameters of a full conditional distribution (for example a Gaussian's mean and log-scale) with one boosted ensemble per parameter. Each stage fits base learners to the natural gradient of a proper scoring rule (log score or CRPS), which is the ordinary gradient premultiplied by the inverse Fisher information; this makes the steps invariant to the parameterisation and keeps the parameters on comparable scales.
- **Block:** `experimental.boost.NaturalGradientBoosting` (Normal, LogNormal and Laplace families, log score and CRPS, Fisher matrices in closed form, regression trees from `RegressionForest`'s grower).
- **Experiment:** `BoostingShootout` (`probabilistic`).
- **Toy task:** heteroscedastic regression from `TabularPrior` and the UCI-like synthetic sets of `GaussianProcessUncertainty`.
- **Baselines:** core GBT with a quantile `Loss` (two quantile models); `GaussianProcess`; a Gaussian-head MLP and deep ensemble (`UncertaintyBenchmark`); wave 9's `SurvivalForest`-style forest variance; NGBoost with the ordinary gradient (ablation).
- **Expected metric:** test NLL, CRPS, RMSE and interval coverage. Hypothesis: NGBoost beats quantile GBT on NLL and coverage and is competitive with deep ensembles on small tables; with the ordinary gradient training is unstable or slow.
- **Source:** [arXiv:1910.03225](https://arxiv.org/abs/1910.03225)

### 10.5.4 Explainable boosting machines: cyclic GA²M (Lou et al., KDD 2013; Nori et al., 2019)
An explainable boosting machine is a generalised additive model with a few pairwise interactions, f(x) = Σ f_i(x_i) + Σ f_ij(x_i, x_j), learned by cyclic gradient boosting: tiny trees on one feature at a time, round-robin, with a very low learning rate so that feature order does not matter. Pairs are chosen by the FAST heuristic, which ranks all pairs by the gain of a quick 2-D split on the residuals. Every term can be plotted, so the model is exactly interpretable.
- **Block:** `experimental.boost.ExplainableBoosting` (cyclic single-feature boosting, FAST pair ranking, pairwise terms, term importance).
- **Experiment:** `BoostingShootout` (`interpretable`).
- **Toy task:** `TabularStressSuite` with a planted additive structure plus two planted pairwise interactions, and the `CausalScenarios` covariates.
- **Baselines:** `NeuralAdditiveModel` (NAM); core GBT (black box); logistic regression; KernelSHAP explanations of GBT (`ExplainabilityBenchmark`).
- **Expected metric:** accuracy, recovery of the planted shape functions (L2 error) and of the planted pairs (rank). Hypothesis: EBM matches GBT within 1–2 points, beats NAM in accuracy and shape recovery at a fraction of the training time, and FAST ranks both planted pairs in its top 3.
- **Source:** [doi:10.1145/2487575.2487579](https://doi.org/10.1145/2487575.2487579), [arXiv:1909.09223](https://arxiv.org/abs/1909.09223)

### 10.5.5 PFN-Boost: boosting on top of a prior-fitted network (Jayawardhana et al., 2025)
Jayawardhana et al. fuse a pretrained in-context tabular model with gradient-boosted trees: the transformer's per-class scores, multiplied by a scale s, replace the constant starting prediction of the boosting, and the trees then fit the residuals of the transformer. With s = 0 it is plain GBT and with large s it follows the transformer; intermediate s is best, the transformer helping on small tables and the trees taking over on large ones. The LLM-Boost variant uses an LLM's label likelihoods; the toy version uses the repo's PFN.
- **Block:** `experimental.boost.PriorBoost` (base margins from any scorer, scale s tuned on validation data, binary and multiclass).
- **Experiment:** `BoostingShootout` (`priorBoost`) and new rows in `experiments.recsys.TabularShootout` (existing rows unchanged).
- **Toy task:** `TabularPrior` tables at 30, 100, 300, 1,000 and 3,000 rows.
- **Baselines:** `PriorFittedNetwork` alone and wave 9's TabICL-lite (`ColumnRowIcl`) alone; core GBT alone; a stacking baseline (logistic regression on PFN and GBT outputs).
- **Expected metric:** accuracy against training-set size. Hypothesis: PFN-Boost matches the PFN on the smallest tables and GBT on the largest, and beats both in between; stacking gets part of the gain. (Wave 7–9 found the PFN weak at CPU pretraining budgets, so the crossover point is what is measured.)
- **Source:** [arXiv:2502.02672](https://arxiv.org/abs/2502.02672)

### 10.5.6 Quantile regression forests (Meinshausen, JMLR 2006)
A random forest's leaves hold training rows; instead of averaging their targets, a quantile regression forest keeps them and weights each training target by how often it shares a leaf with the query point, giving the full conditional distribution and any quantile from one fitted forest. Conformalised quantile regression on top gives finite-sample coverage.
- **Block:** `experimental.boost.QuantileForest` (leaf-membership weights over `RegressionForest`'s trees, weighted quantiles, CQR wrapper).
- **Experiment:** `BoostingShootout` (`probabilistic`).
- **Toy task:** as 10.5.3, plus a bimodal-noise variant where Gaussian families fail.
- **Baselines:** 10.5.3 NGBoost; quantile GBT; `ConformalForecaster`'s CQR on an MLP; `RegressionForest` with a Gaussian from the tree spread.
- **Expected metric:** pinball loss at 5/50/95%, interval coverage and width, and the CRPS of the bimodal variant. Hypothesis: QRF beats NGBoost on the bimodal variant and ties elsewhere; CQR on top fixes its slight under-coverage.
- **Source:** [JMLR 2006](https://jmlr.org/papers/v7/meinshausen06a.html)

### 10.5.7 Hawkes processes with flexible kernels (Hawkes, Biometrika 1971; a 2026 re-evaluation of neural TPPs)
A multivariate Hawkes process models event streams with intensity λ_k(t) = μ_k + Σ_{t_i < t} φ_{k,k_i}(t − t_i): each past event excites future events of each type through a kernel φ. With exponential kernels the log-likelihood is computed recursively and fitted by EM (the branching structure is the latent variable) or gradient ascent. A NeurIPS 2026 re-evaluation found that once the kernels are flexible (learnable splines or sums of exponentials), most of the reported gap between Hawkes processes and neural TPPs disappears, except on data with genuinely higher-order interactions.
- **Block:** `experimental.events.HawkesProcess` (exponential and sum-of-exponentials kernels fitted by EM, B-spline kernels fitted by penalised likelihood, Ogata thinning simulation, compensator-based goodness of fit).
- **Experiment:** `experiments.events.PointProcessShootout`.
- **Toy task:** `EventStreams` with 1–5 event types: exponential, power-law and non-monotone (delayed) kernels, inhibition, and a non-Hawkes self-correcting process.
- **Baselines:** a homogeneous Poisson process; a renewal process with a Weibull inter-event time (from wave 9's `SurvivalModels`); `BayesianOnlineChangePoint` as a rate tracker.
- **Expected metric:** held-out log-likelihood per event against the generator's true log-likelihood, next-type accuracy, next-time RMSE, and a time-rescaling KS statistic. Hypothesis: exponential Hawkes is near-optimal on its own data but misfits power-law and delayed kernels; spline kernels close most of that gap; none fits inhibition.
- **Source:** [doi:10.1093/biomet/58.1.83](https://doi.org/10.1093/biomet/58.1.83), [NeurIPS 2026](https://neurips.cc/virtual/2026/poster/139473)

### 10.5.8 Neural temporal point processes: RMTPP, the neural Hawkes process and the transformer Hawkes process (Du et al., KDD 2016; Mei & Eisner, NeurIPS 2017; Zuo et al., ICML 2020)
Neural TPPs encode the event history with a network. RMTPP feeds (type, time gap) into an RNN and uses an exponential-of-linear conditional intensity whose integral is closed-form. The neural Hawkes process uses a continuous-time LSTM whose cells decay exponentially between events, giving intensities that can rise, fall or inhibit; its likelihood integral needs Monte Carlo. The transformer Hawkes process uses self-attention over past events with a softplus intensity interpolated between events.
- **Block:** `experimental.events.NeuralPointProcess` (RMTPP, continuous-time LSTM intensity, transformer intensity; Monte Carlo compensator; thinning sampler).
- **Experiment:** `PointProcessShootout`.
- **Toy task:** as 10.5.7.
- **Baselines:** 10.5.7 with exponential and spline kernels; the Poisson and renewal baselines.
- **Expected metric:** as 10.5.7, plus training time. Hypothesis: the neural models win on inhibition and the self-correcting process, but on the Hawkes-generated streams they do not beat spline-kernel Hawkes, which reproduces the 2026 finding at toy scale.
- **Source:** [doi:10.1145/2939672.2939875](https://doi.org/10.1145/2939672.2939875), [arXiv:1612.09328](https://arxiv.org/abs/1612.09328), [arXiv:2002.09291](https://arxiv.org/abs/2002.09291)

### 10.5.9 Intensity-free point processes: log-normal mixtures (Shchur et al., ICLR 2020)
Modelling the intensity forces a Monte Carlo integral or a restrictive form. Shchur et al. model the conditional density of the next inter-event time directly, as a mixture of log-normals whose weights, means and scales come from the RNN history embedding. The likelihood, sampling and the expected next time are all closed-form, and the mixture is a universal approximator of densities on the positive half-line.
- **Block:** `experimental.events.IntensityFreeTpp` (log-normal mixture head on a GRU history encoder, mark head, exact sampling).
- **Experiment:** `PointProcessShootout`.
- **Toy task:** as 10.5.7, with sequence lengths 50–500.
- **Baselines:** 10.5.8's RMTPP and transformer Hawkes; 10.5.7's spline Hawkes.
- **Expected metric:** held-out NLL, next-time MAE, sampling speed (events per second), and the calibration of predicted inter-event quantiles. Hypothesis: the log-normal mixture matches the best intensity models on NLL, beats RMTPP on multimodal inter-event times, and samples orders of magnitude faster than thinning.
- **Source:** [arXiv:1909.12127](https://arxiv.org/abs/1909.12127)

### 10.5.10 Irregularly sampled series: GRU-D, ODE-RNN and latent ODEs, and mTAN (Che et al., Scientific Reports 2018; Rubanova et al., NeurIPS 2019; Shukla & Marlin, ICLR 2021)
Most forecasters assume a regular grid. GRU-D feeds the mask and the time since the last observation into a GRU and decays both the inputs (toward the empirical mean) and the hidden state with learned rates. ODE-RNN evolves the hidden state between observations with a neural ODE and updates it with a GRU at observations; the latent ODE uses an ODE-RNN encoder for a VAE whose latent trajectory is a single ODE solution. mTAN embeds time with learned sinusoids and uses attention from a fixed set of reference times to the observed ones, giving a fixed-size representation of any sampling pattern.
- **Block:** `experimental.events.IrregularSequenceModels` (GRU-D cell, ODE-RNN and latent-ODE encoder/decoder using `NeuralOde`'s solver, mTAN encoder).
- **Experiment:** `experiments.events.IrregularSeriesShootout`.
- **Toy task:** `IrregularSeries`: interpolation (fill in held-out observation times) and extrapolation (forecast the next window) at 30–90% missingness, MAR and MNAR; plus classification of `SeriesClassificationTasks` series after random subsampling.
- **Baselines:** `NeuralCde` (wave 8); a GRU on forward-filled values with a mask channel; linear interpolation followed by DLinear; `LiquidCell`.
- **Expected metric:** interpolation and extrapolation MSE and classification accuracy against missingness. Hypothesis: GRU-D is a strong cheap baseline; latent ODEs win on interpolation of smooth dynamics; mTAN is the fastest and best on classification; forward-fill plus GRU degrades sharply above 70% missingness.
- **Source:** [arXiv:1606.01865](https://arxiv.org/abs/1606.01865), [arXiv:1907.03907](https://arxiv.org/abs/1907.03907), [arXiv:2101.10318](https://arxiv.org/abs/2101.10318)

### 10.5.11 Deep imputation: BRITS, SAITS and CSDI (Cao et al., NeurIPS 2018; Du et al., ESWA 2023; Tashiro et al., NeurIPS 2021)
BRITS treats missing values as variables of a bidirectional recurrent graph: at each step a history-based and a feature-correlation-based estimate are combined, and the imputation errors at observed points are backpropagated through time in both directions. SAITS replaces the RNN with two diagonally-masked self-attention blocks trained with a joint observed-reconstruction and masked-imputation loss. CSDI is a conditional score-based diffusion model: it learns to denoise the missing entries conditioned on the observed ones, with temporal and feature transformers, and gives probabilistic imputations.
- **Block:** `experimental.events.TimeSeriesImputer` (BRITS, SAITS with diagonal masking, CSDI with self-supervised masking).
- **Experiment:** `IrregularSeriesShootout` (`imputation`).
- **Toy task:** multivariate `IrregularSeries` on a regular grid with 10–80% of entries missing at random, in blocks, and not at random.
- **Baselines:** mean, last-observation-carried-forward and linear interpolation; `StateSpaceFilter` (Kalman smoother); `GaussianProcess` per channel; 10.5.10's latent ODE.
- **Expected metric:** imputation MAE and, for CSDI, CRPS; the accuracy of a downstream forecaster trained on the imputed data. Hypothesis: SAITS beats BRITS at lower cost; CSDI has the best CRPS and MAE on block-missing data but is the slowest; the Kalman smoother is strong on the linear-Gaussian series.
- **Source:** [arXiv:1805.10572](https://arxiv.org/abs/1805.10572), [arXiv:2202.08516](https://arxiv.org/abs/2202.08516), [arXiv:2107.03502](https://arxiv.org/abs/2107.03502)

### 10.5.12 Temporal graph networks against EdgeBank, with a controlled probe suite (Rossi et al., 2020; Xu et al., ICLR 2020; Cong et al., ICLR 2023; Poursafaei et al., NeurIPS 2022; Gao et al., 2025; Hayes et al., 2025)
Waves 7 and 8 cut temporal graph networks because there was no fair toy baseline; there is now one. TGN keeps a memory vector per node, updated by a GRU from messages at each interaction, and embeds nodes with temporal graph attention (TGAT's attention over recent neighbours with a functional time encoding). GraphMixer shows that an MLP-Mixer over the most recent links plus mean-pooled neighbour features does as well. EdgeBank, which memorises past edges and predicts "seen before", is a strong non-learned baseline under random negatives and weak under historical negatives. HyperEvent (2025) instead encodes each query event's relation to the recent events of both endpoints by 0/1/2-hop co-occurrence counts and classifies the sequence with a transformer. Hayes et al. (2025) show that such models learn preferential attachment but often miss recency and density, which the generator here plants on purpose.
- **Block:** `experimental.graph.TemporalGraphModels` (TGN memory and messages, TGAT attention with time encoding, GraphMixer, EdgeBank variants, HyperEvent-style relative structural encoding).
- **Experiment:** `experiments.graphs.TemporalGraphs`.
- **Toy task:** `TemporalGraphs` streams with one planted property at a time (recency, periodicity, preferential attachment, homophily, repeated edges) and a mixed stream; wave 9 `ClickLogs` user–item interactions as a bipartite case.
- **Baselines:** EdgeBank; a static GCN on the aggregated graph (`GcnLayer`); wave 7's `SequentialRecommendation` SASRec on each user's sequence (the bipartite case).
- **Expected metric:** MRR and AUC under random and historical negatives, per planted property. Hypothesis: EdgeBank beats every learned model under random negatives on repeated-edge streams and loses under historical negatives; TGN and GraphMixer capture preferential attachment and periodicity but not recency; HyperEvent's structural encoding wins on the mixed stream.
- **Source:** [arXiv:2006.10637](https://arxiv.org/abs/2006.10637), [arXiv:2002.07962](https://arxiv.org/abs/2002.07962), [arXiv:2302.11636](https://arxiv.org/abs/2302.11636), [arXiv:2207.10128](https://arxiv.org/abs/2207.10128), [arXiv:2507.11836](https://arxiv.org/abs/2507.11836), [arXiv:2510.09416](https://arxiv.org/abs/2510.09416)

### 10.5.13 Relational message passing on knowledge graphs: R-GCN and CompGCN (Schlichtkrull et al., ESWC 2018; Vashishth et al., ICLR 2020)
R-GCN extends graph convolution to multi-relational graphs with one weight matrix per relation (and direction), regularised by basis decomposition (each relation's matrix is a learned combination of a few shared bases), and is used as an encoder in front of a DistMult decoder. CompGCN embeds relations as well as entities and composes them in each message (subtraction, multiplication or circular correlation, as in TransE, DistMult and HolE), so the number of parameters does not grow with the relations.
- **Block:** `experimental.graph.RelationalGcn` (R-GCN layer with basis and block-diagonal decomposition; CompGCN layer with the three compositions).
- **Experiment:** `experiments.graphs.RelationalGraphs`.
- **Toy task:** `FamilyTreeGraph` link prediction, plus inductive splits where test entities are new, and node classification by relation pattern.
- **Baselines:** `KnowledgeGraphEmbedding` TransE, DistMult, ComplEx and RotatE (shallow embeddings); `GcnLayer` ignoring relation types.
- **Expected metric:** filtered MRR and Hits@10, transductive and inductive. Hypothesis: shallow embeddings win transductively on this small graph; R-GCN and CompGCN win on relation-pattern node classification; none handles inductive test entities well, which motivates 10.5.14.
- **Source:** [arXiv:1703.06103](https://arxiv.org/abs/1703.06103), [arXiv:1911.03082](https://arxiv.org/abs/1911.03082)

### 10.5.14 Neural Bellman–Ford networks for link prediction (Zhu et al., NeurIPS 2021)
NBFNet represents a pair (u, v) by the generalised sum over all paths from u to v of the product of edge representations, computed by a learned Bellman–Ford iteration: the source node is initialised with a query-relation embedding (the "indicator"), messages compose the current representation with each edge's relation (as in CompGCN), and aggregation (sum, PNA) replaces the min. Because nothing is tied to node identities, it is inductive by construction and its paths are interpretable.
- **Block:** `experimental.graph.BellmanFordNetwork` (indicator initialisation, relation-conditioned messages, T iterations, path-importance read-out by gradients).
- **Experiment:** `RelationalGraphs`.
- **Toy task:** as 10.5.13, transductive and inductive, and `SyntheticGraphs` homogeneous link prediction.
- **Baselines:** 10.5.13; `KnowledgeGraphEmbedding`; `GraphReasoning`'s Bellman–Ford CLRS processor (algorithm imitation rather than link prediction); common-neighbour and Adamic–Adar heuristics.
- **Expected metric:** filtered MRR and Hits@10, and the share of top-ranked paths that match the planted family rules (for example parent ∘ parent = grandparent). Hypothesis: NBFNet is the only model that keeps its accuracy on inductive splits, and its top paths recover the planted rules.
- **Source:** [arXiv:2106.06935](https://arxiv.org/abs/2106.06935)

### 10.5.15 Hyperbolic embeddings of hierarchies: Poincaré and Lorentz models (Nickel & Kiela, NeurIPS 2017 and ICML 2018)
Trees grow exponentially with depth, and so does the volume of hyperbolic space, so hierarchies embed in very few hyperbolic dimensions with low distortion where Euclidean embeddings need many. Poincaré embeddings place nodes in the open unit ball, train them by Riemannian SGD (the Euclidean gradient rescaled by (1 − ‖x‖²)²/4, then projected back into the ball) on a softmax ranking loss over hyperbolic distances. The Lorentz (hyperboloid) model has the same geometry with a numerically stabler exponential map.
- **Block:** `experimental.graph.HyperbolicEmbedding` (Poincaré and Lorentz models, Riemannian SGD, hyperbolic distance), proposed for the hyperbolic-graph wave suggested in wave 8.
- **Experiment:** `RelationalGraphs` (`hierarchy`).
- **Toy task:** transitive closures of random trees (depth 4–8) and of `FamilyTreeGraph`'s ancestor relation; reconstruction and link prediction.
- **Baselines:** Euclidean embeddings with the same loss and dimension; `RandomWalkEmbedding` (node2vec); `KnowledgeGraphEmbedding` RotatE.
- **Expected metric:** mean rank and MAP of reconstruction at 2, 5 and 10 dimensions; correlation of norm with depth. Hypothesis: 2-D hyperbolic embeddings beat 10-D Euclidean ones on mean rank; the Lorentz model trains more stably than the Poincaré ball.
- **Source:** [arXiv:1705.08039](https://arxiv.org/abs/1705.08039), [arXiv:1806.03417](https://arxiv.org/abs/1806.03417)

### 10.5.16 Graph-based dependency parsing: biaffine attention with maximum spanning trees (Dozat & Manning, ICLR 2017; Eisner, 1996; Chu & Liu, 1965; Edmonds, 1967)
The biaffine parser encodes a sentence with a BiLSTM (or transformer), projects each word into a "head" and a "dependent" vector with separate MLPs, and scores every possible arc with a biaffine product s_ij = h_iᵀ U d_j + wᵀ h_i; a second biaffine classifier labels the arcs. At test time the highest-scoring well-formed tree is found by the Chu–Liu–Edmonds maximum spanning arborescence (non-projective) or Eisner's O(n³) dynamic programme (projective). It is still a standard strong baseline.
- **Block:** `experimental.parse.BiaffineParser` (head/dependent MLPs, biaffine arc and label scorers, Chu–Liu–Edmonds and Eisner decoders, head-selection fallback).
- **Experiment:** `experiments.parsing.ParsingShootout`.
- **Toy task:** `ToyTreebank` dependency trees, 2k training sentences of length 4–20, with test sentences up to length 30 and PP-attachment ambiguities resolved by planted lexical preferences.
- **Baselines:** `PointerNetwork` pointing to each word's head; `TransformerSeq2Seq` generating a linearised head sequence; `LinearChainCrf` tagging relative head offsets; a right-branching heuristic.
- **Expected metric:** unlabelled and labelled attachment scores (UAS/LAS), the share of well-formed trees, and accuracy on the PP-attachment cases. Hypothesis: the biaffine parser beats the seq2seq and offset-tagging baselines by several UAS points and always outputs trees; MST decoding adds a little over greedy head selection, most on long sentences.
- **Source:** [arXiv:1611.01734](https://arxiv.org/abs/1611.01734), [ACL Anthology C96-1058](https://aclanthology.org/C96-1058/)

### 10.5.17 Span-based constituency parsing with a CKY chart (Stern et al., ACL 2017; Kitaev & Klein, ACL 2018)
A span-based parser scores every span (i, j) and label from the difference of encoder states at its boundaries (or a self-attentive encoder's span vectors) and finds the highest-scoring binarised tree with the CKY algorithm, training with a structured margin loss (or a tree-CRF likelihood whose partition function is the inside algorithm). It needs no grammar and no transition system. Kitaev and Klein's self-attentive encoder factors content and position attention.
- **Block:** `experimental.parse.ChartParser` (span scorer, CKY decoder, margin and tree-CRF losses via the inside algorithm, transformer encoder with content/position factoring).
- **Experiment:** `ParsingShootout`.
- **Toy task:** `ToyTreebank` constituency trees (with nested clauses and coordination), same splits as 10.5.16.
- **Baselines:** `TransformerSeq2Seq` and `Seq2SeqAttention` generating linearised bracketings ("grammar as a foreign language"); a PCFG estimated from the training trees with CKY (the generator's own grammar family); right-branching trees.
- **Expected metric:** labelled bracketing F1 (EVALB-style), exact-match rate and the share of well-formed outputs, against sentence length. Hypothesis: the chart parser beats seq2seq on F1 and always yields trees, and its advantage grows with sentence length; the tree-CRF loss beats the margin loss slightly; the treebank PCFG is strong on in-distribution lengths but cannot use lexical preferences.
- **Source:** [arXiv:1705.03919](https://arxiv.org/abs/1705.03919), [arXiv:1805.01052](https://arxiv.org/abs/1805.01052)

---

## WP6. Image restoration and speech analysis, plus the wave 9 perception and systems follow-ups (16)

Target: two new packages, `experimental.restore` (10.6.1–10.6.4) and `experimental.speech` (10.6.5–10.6.8); 10.6.9 adds to the existing `experimental.vision`, and the follow-ups add to the existing `experimental.vision` (10.6.10), `experimental.document` (10.6.11), `experimental.multimodal` (10.6.12–10.6.13), `experimental.edge` (10.6.14), `experimental.parallel` (10.6.15) and `experimental.audio` (10.6.16), with options whose defaults keep the wave 9 rows unchanged. Measurements: new theme `experiments.restoration` (`RestorationShootout`, 10.6.1–10.6.4), new theme `experiments.speech` (`SpeakerShootout` for 10.6.5 and 10.6.8, `SpeechFrontEnds` for 10.6.6–10.6.7), `experiments.vision.VisionShootout` (10.6.9), and follow-up methods on `DetectionShootout`, `DocumentParsing`, `FusionShootout`, `EdgeDeployment`, `ParallelismShootout` and `AudioGeneration`. Two new generators in `experimental.data`: `DegradedImages` (clean `ShapeImages`, `DigitGlyphs`, `ToyImages` and `GlyphDocuments` crops with Gaussian, Poisson–Gaussian, spatially correlated and salt-and-pepper noise at known levels, Gaussian and motion blur with known kernels, and ×2/×4 bicubic downsampling) and `SpeakerVoices` (extends `SpokenDigitRecognition.speak` with more speaker factors so that speakers are separable: pitch and pitch range, per-formant offsets, spectral tilt, breathiness, jitter and speaking rate, for 50–200 speakers; it also produces multi-speaker conversations with known speaker turns, silence and background noise). The wave 7 and 8 lists cut speaker verification because the formant synthesiser had only two speaker factors (formant scale and rate); `SpeakerVoices` exists to remove that obstacle, and its speaker factors are checked first with an i-vector-like baseline (the item's first row). Test classes stay near 10 s with 16×16 to 32×32 images and 0.5–2 s audio; the follow-ups' long budgets run in `main` as Runs, which is what their wave 9 negatives were missing.

### 10.6.1 Single-image super-resolution: SRCNN, ESPCN, EDSR and SPAN (Dong et al., ECCV 2014; Shi et al., CVPR 2016; Lim et al., CVPRW 2017; Wan et al., 2024)
SRCNN upsamples bicubically and then refines with three convolutions. ESPCN keeps all convolutions at low resolution and upsamples at the end with a sub-pixel convolution (a convolution to r²·C channels followed by pixel shuffle), which is much cheaper. EDSR stacks residual blocks without batch norm, with residual scaling, before the sub-pixel upsampler. SPAN, a 2024 efficient-SR winner, adds a parameter-free attention: each block's output is multiplied by an odd, origin-symmetric activation of its own convolution features (so negative features of large magnitude also count), with a residual path to keep weak features.
- **Block:** `experimental.restore.SuperResolution` (SRCNN, ESPCN with core `PixelShuffle`, EDSR blocks, SPAN blocks).
- **Experiment:** `experiments.restoration.RestorationShootout`.
- **Toy task:** `DegradedImages` ×2 and ×4 bicubic downsampling of `ShapeImages`, `ToyImages` and `GlyphDocuments` crops.
- **Baselines:** bicubic and nearest upsampling; wave 6's U-Net (`ShapeSegmentation`) adapted to regression; wave 8's SIREN/Fourier-feature `NeuralFields` fitted per image (a per-image implicit-function upsampler).
- **Expected metric:** PSNR and SSIM, parameters and multiply-adds; on glyph crops, the character error rate of wave 9's OCR (`LineRecognizer`) on the upsampled text. Hypothesis: ESPCN matches SRCNN at a fraction of the compute; EDSR is best at equal training steps; SPAN gains over a same-size ESPCN; super-resolution lowers OCR error on ×4 glyphs clearly.
- **Source:** [arXiv:1501.00092](https://arxiv.org/abs/1501.00092), [arXiv:1609.05158](https://arxiv.org/abs/1609.05158), [arXiv:1707.02921](https://arxiv.org/abs/1707.02921), [arXiv:2311.12770](https://arxiv.org/abs/2311.12770)

### 10.6.2 Supervised denoising with residual learning: DnCNN and FFDNet (Zhang et al., TIP 2017 and TIP 2018)
DnCNN predicts the noise rather than the clean image (residual learning) with a stack of conv–BN–ReLU layers; one network handles a range of noise levels. FFDNet works on a pixel-unshuffled (downsampled) image and takes a noise-level map as an extra input, so a single model can be steered to any noise level, including spatially varying noise, and runs faster.
- **Block:** `experimental.restore.Denoisers` (DnCNN residual denoiser, FFDNet with noise-level map and pixel-(un)shuffle).
- **Experiment:** `RestorationShootout` (`denoising`).
- **Toy task:** `DegradedImages` Gaussian noise σ ∈ {10, 25, 50}/255 (blind and non-blind), plus spatially varying noise.
- **Baselines:** Gaussian and median filtering; non-local means (implemented as a reference filter in the experiment); the U-Net; wave 7's diffusion denoiser network (`TinyDdpm` at the matching timestep) used as a one-step denoiser.
- **Expected metric:** PSNR/SSIM per noise level and on spatially varying noise. Hypothesis: residual learning beats direct prediction at the same network; FFDNet with the true noise map wins on spatially varying noise; the diffusion network used in one step is competitive at high noise.
- **Source:** [arXiv:1608.03981](https://arxiv.org/abs/1608.03981), [arXiv:1710.04026](https://arxiv.org/abs/1710.04026)

### 10.6.3 Denoising without clean targets: Noise2Noise, Noise2Void, Neighbor2Neighbor, zero-shot N2N and Median2Median (Lehtinen et al., ICML 2018; Krull et al., CVPR 2019; Huang et al., CVPR 2021; Mansour & Heckel, CVPR 2023; 2025)
If noise is zero-mean and independent, training to map one noisy image to another noisy copy of the same scene (Noise2Noise) has the same minimiser as training on clean targets. Noise2Void needs only single noisy images: it masks pixels and predicts each from its neighbours (a blind spot), which cannot learn independent noise. Neighbor2Neighbor builds training pairs by sub-sampling neighbouring pixels from one image, with a regulariser for the small offset. Zero-shot N2N trains a tiny network on 2×2 diagonal-mean sub-images of the single test image. Median2Median (2025) targets spatially correlated noise: it builds sub-image pairs from 3×3 blocks using directional interpolation and keeps the candidates nearest the median, since structured-noise outliers sit far from it.
- **Block:** `experimental.restore.SelfSupervisedDenoising` (N2N pairs, blind-spot masking for N2V, neighbour sub-sampler for Ne2Ne, the ZS-N2N per-image trainer, M2M sampling with directional candidates and median selection).
- **Experiment:** `RestorationShootout` (`selfSupervised`).
- **Toy task:** `DegradedImages` i.i.d. Gaussian and Poisson–Gaussian noise, and spatially correlated (filtered) noise.
- **Baselines:** 10.6.2's DnCNN trained on clean targets (the supervised ceiling); Gaussian and median filtering; non-local means.
- **Expected metric:** PSNR/SSIM on i.i.d. and correlated noise, and per-image time for the zero-shot methods. Hypothesis: N2N is within 0.5 dB of supervised; N2V and Ne2Ne lose 1–2 dB on i.i.d. noise; on correlated noise every blind-spot and sub-sampling method fails except M2M, which keeps a clear margin over filtering.
- **Source:** [arXiv:1803.04189](https://arxiv.org/abs/1803.04189), [arXiv:1811.10980](https://arxiv.org/abs/1811.10980), [arXiv:2101.02824](https://arxiv.org/abs/2101.02824), [arXiv:2303.11253](https://arxiv.org/abs/2303.11253), [arXiv:2510.01666](https://arxiv.org/abs/2510.01666)

### 10.6.4 Deblurring with learned priors: plug-and-play ADMM and deep unrolling (Venkatakrishnan et al., GlobalSIP 2013; Zhang et al., TPAMI 2021)
Non-blind deblurring inverts a known blur. Plug-and-play ADMM (or half-quadratic splitting) alternates a closed-form data step (an FFT-domain Wiener-like solve) with a denoising step in which any denoiser stands in for the proximal operator of the image prior; DPIR uses a noise-level-conditioned CNN denoiser (10.6.2's FFDNet-style network) with a decreasing schedule. Deep unrolling instead trains a fixed number of such iterations end to end, with per-iteration denoisers and step sizes.
- **Block:** `experimental.restore.PlugAndPlay` (FFT data step for convolutional blur, HQS and ADMM loops, a denoiser interface, an unrolled trainable variant).
- **Experiment:** `RestorationShootout` (`deblurring`).
- **Toy task:** `DegradedImages` Gaussian and motion blur with known kernels, with 1% and 5% noise.
- **Baselines:** Wiener deconvolution; Richardson–Lucy; total-variation-regularised deconvolution solved by gradient descent; a U-Net trained directly on blurred-to-sharp pairs.
- **Expected metric:** PSNR/SSIM per kernel and noise level, and generalisation to a kernel not seen in training. Hypothesis: PnP with the 10.6.2 denoiser beats Wiener and TV clearly and, unlike the direct U-Net, generalises to an unseen kernel; the unrolled variant is best on seen kernels.
- **Source:** [doi:10.1109/GlobalSIP.2013.6737048](https://doi.org/10.1109/GlobalSIP.2013.6737048), [arXiv:2008.13751](https://arxiv.org/abs/2008.13751)

### 10.6.5 Speaker verification: x-vectors with AAM-softmax, and GE2E (Snyder et al., ICASSP 2018; Wan et al., ICASSP 2018)
Speaker verification decides whether two utterances come from the same speaker. The x-vector network applies time-delay (dilated 1-D) convolutions to frame features, pools them with mean and standard deviation over time (statistics pooling), and takes an embedding from a layer after pooling; training is speaker classification, today usually with additive angular margin softmax, and scoring is cosine or PLDA. GE2E trains on batches of N speakers × M utterances so that each utterance's embedding is close to its own speaker's centroid (computed without that utterance) and far from the other centroids, which directly optimises the verification geometry.
- **Block:** `experimental.speech.SpeakerEmbedding` (TDNN with statistics pooling, attentive statistics pooling option, GE2E loss, cosine and two-covariance PLDA scoring), using WP3's `experimental.metric.MarginSoftmax` for AAM.
- **Experiment:** `experiments.speech.SpeakerShootout`.
- **Toy task:** `SpeakerVoices` with 100 training speakers and 30 held-out speakers, 0.5–2 s digit strings, at 20 and 5 dB SNR.
- **Baselines:** mean MFCC vectors with cosine scoring; a GMM-UBM-style baseline built from `GaussianMixture` (per-speaker MAP-adapted means); `SpokenDigitRecognition`'s encoder trained on digits and used as a speaker embedding (a content model).
- **Expected metric:** equal error rate and minDCF on held-out speakers; EER against utterance length. Hypothesis: x-vector plus AAM gives the lowest EER, GE2E comes close with less tuning, and the digit encoder is poor (its features discard speaker information).
- **Source:** [doi:10.1109/ICASSP.2018.8461375](https://doi.org/10.1109/ICASSP.2018.8461375), [arXiv:1710.10467](https://arxiv.org/abs/1710.10467)

### 10.6.6 Voice activity detection: statistical and small neural detectors (Sohn et al., IEEE SPL 1999; Hughes & Mierle, ICASSP 2013)
Voice activity detection marks which frames contain speech; it gates keyword spotting, recognition and diarisation. The classic statistical VAD models speech and noise spectra as Gaussian, computes a per-frequency likelihood ratio with a decision-directed a-priori SNR estimate, and smooths the decisions with a hidden Markov model (hangover). Small neural detectors (an RNN or a CRNN on log-mel frames) are now standard and far more robust to non-stationary noise.
- **Block:** `experimental.speech.VoiceActivity` (energy and zero-crossing detector, Sohn likelihood-ratio VAD with HMM hangover via `HiddenMarkovModel`, CRNN/GRU detector, frame smoothing).
- **Experiment:** `experiments.speech.SpeechFrontEnds`.
- **Toy task:** `SpeakerVoices` conversations with silences, at 20/10/0 dB SNR in white, pink and babble-like noise (other synthesised speakers in the background).
- **Baselines:** energy thresholding; wave 9's keyword spotters (`KeywordSpotters`, which have a silence class) used frame-wise; PCEN front end (`Pcen`) under the energy detector.
- **Expected metric:** frame F1 and AUC against SNR, model size in kB, and latency. Hypothesis: the statistical VAD beats energy thresholding at 10 dB; the CRNN beats both at 0 dB and in babble noise with under 10 kB of weights.
- **Source:** [doi:10.1109/97.736233](https://doi.org/10.1109/97.736233); Hughes & Mierle, "Recurrent neural networks for voice activity detection", ICASSP 2013

### 10.6.7 Speech enhancement: spectral subtraction, Wiener filtering and complex ratio masks (Boll, 1979; Ephraim & Malah, 1984; Williamson et al., IEEE/ACM TASLP 2016)
Single-channel speech enhancement removes noise from one speaker's recording. Spectral subtraction subtracts an estimated noise magnitude spectrum (from speech-free frames given by a VAD) with flooring; Wiener and Ephraim–Malah MMSE estimators apply a gain from the a-priori SNR. Learned enhancers predict a time–frequency mask; the complex ratio mask also corrects the phase, which matters at low SNR, and is trained on the complex spectrum.
- **Block:** `experimental.speech.Enhancement` (spectral subtraction with oversubtraction and floor, decision-directed Wiener/MMSE-STSA gains, a small CRNN predicting magnitude or complex ratio masks).
- **Experiment:** `SpeechFrontEnds` (`enhancement`).
- **Toy task:** `SpeakerVoices` utterances plus stationary and non-stationary noise at −5 to 15 dB.
- **Baselines:** the noisy input; `SourceSeparation`'s STFT-mask MLP (wave 7) and ideal ratio mask (oracle); `ConvTasNet` trained as a one-source enhancer.
- **Expected metric:** SI-SNR improvement, a spectral-distance proxy for quality, and the downstream digit error rate of `SpokenDigitRecognition`'s recogniser on the enhanced audio. Hypothesis: classic methods help in stationary noise only; the complex ratio mask beats the magnitude mask by 1–2 dB at −5 dB SNR; enhancement lowers digit error at low SNR but can raise it at high SNR (artefacts).
- **Source:** [doi:10.1109/TASSP.1979.1163209](https://doi.org/10.1109/TASSP.1979.1163209), [doi:10.1109/TASSP.1984.1164453](https://doi.org/10.1109/TASSP.1984.1164453), [doi:10.1109/TASLP.2015.2512042](https://doi.org/10.1109/TASLP.2015.2512042)

### 10.6.8 Speaker diarisation: embedding clustering against end-to-end neural diarisation (Park et al., 2022 review; Fujita et al., Interspeech 2019)
Diarisation answers "who spoke when". The modular pipeline runs VAD, cuts speech into short windows, embeds each with a speaker model (10.6.5) and clusters the embeddings (agglomerative or spectral clustering with an eigengap estimate of the number of speakers). End-to-end neural diarisation (EEND) instead outputs, for every frame, an independent speech-activity probability per speaker slot, trained with a permutation-invariant loss, which handles overlapping speech that clustering cannot.
- **Block:** `experimental.speech.Diarization` (windowing, embedding clustering with AHC and `SpectralClustering` plus eigengap, EEND with a small transformer encoder and `PermutationInvariantLoss`).
- **Experiment:** `SpeakerShootout` (`diarization`).
- **Toy task:** `SpeakerVoices` two- and three-speaker conversations, with 0%, 10% and 30% overlapping speech.
- **Baselines:** one-speaker-for-everything; clustering of mean-MFCC windows; oracle-VAD plus oracle-number-of-speakers versions of the clustering pipeline.
- **Expected metric:** diarisation error rate (missed speech, false alarm, speaker confusion) with and without overlap. Hypothesis: the clustering pipeline with 10.6.5's embeddings wins without overlap; EEND wins at 30% overlap; the eigengap speaker count is the weak point of clustering.
- **Source:** [arXiv:2101.09624](https://arxiv.org/abs/2101.09624), [arXiv:1909.06247](https://arxiv.org/abs/1909.06247)

### 10.6.9 FlexiViT: one ViT for every patch size (Beyer et al., CVPR 2023)
A ViT is trained for one patch size, which fixes its cost. FlexiViT randomises the patch size during training and resizes the patch-embedding weights (and position embeddings) on the fly with a pseudo-inverse resize that preserves the token embeddings' inner products, so a single model can be deployed at any patch size, trading accuracy for compute after training.
- **Block:** `experimental.vision.FlexiblePatchEmbedding` (PI-resize of patch weights, bilinear position-embedding resize, random patch-size sampling).
- **Experiment:** `experiments.vision.VisionShootout` (`flexiViT`; existing rows unchanged).
- **Toy task:** `ShapeImages` at 32×32 with patch sizes 2, 4 and 8.
- **Baselines:** separate ViTs trained at each patch size (same total steps split three ways, and full steps each); one ViT trained at patch 4 and evaluated at other sizes with naive weight resizing; wave 9's `ElasticSupernet` as the CNN analogue.
- **Expected metric:** accuracy against multiply-adds at each patch size. Hypothesis: FlexiViT is within 1–2 points of the separately trained ViTs at every size, while naive resizing collapses away from the training size.
- **Source:** [arXiv:2212.08013](https://arxiv.org/abs/2212.08013)

### 10.6.10 DETR on scenes: two-stage query selection and a long budget (Zhu et al., ICLR 2021; Zhao et al., CVPR 2024) — follow-up
Both DETR and H-DETR stayed at mAP 0 on `ShapeScenes` even at 700 steps in wave 9, so scenes were dropped from the test; on layout pages H-DETR overtook DETR only after 700 steps (0.49 against 0.34). A large part of DETR's slow convergence is that its object queries start as learned constants. Two-stage Deformable DETR and RT-DETR initialise the decoder queries from the encoder: every encoder token predicts a class score and a box, and the top-k tokens become the queries (with their boxes as reference points), so the decoder starts near the objects.
- **Block:** a `querySelection(k)` option on `experimental.vision.SetPredictionDetector` (encoder class/box heads, top-k selection, reference boxes; defaults unchanged).
- **Experiment:** `experiments.vision.DetectionShootout.detrFollowUp` (wave 8–9 rows unchanged).
- **Toy task:** `ShapeScenes` and `GlyphDocuments` layouts, learning curves to 5k steps (Run) and a 500-step test.
- **Baselines:** DETR, DN-DETR and H-DETR (`SetPredictionDetector` options); CenterNet and RetinaNet-lite on the same curve.
- **Expected metric:** mAP against steps on both datasets. Hypothesis: with query selection DETR leaves mAP 0 on scenes within a few hundred steps and approaches CenterNet by 5k steps; combined with H-DETR's one-to-many matching it is the fastest-converging set predictor.
- **Source:** [arXiv:2010.04159](https://arxiv.org/abs/2010.04159), [arXiv:2304.08069](https://arxiv.org/abs/2304.08069)

### 10.6.11 Donut with a reading-pretraining stage and a learning curve (Kim et al., ECCV 2022) — follow-up
Wave 9's Donut-style `ImageToMarkup` scored field F1 0.000 at 50 steps, while OCR plus rules reached 0.838. The paper's model is pretrained to read all text in synthetic documents (SynthDoG) before it is fine-tuned to emit structured fields; without that stage, the decoder must learn reading and parsing at once from a few hundred forms. The follow-up adds a reading-pretraining stage on `GlyphDocuments` pages (target: the full text in reading order), then fine-tunes on form parsing, and runs a learning curve.
- **Block:** a `pretrainReading(steps)` stage and checkpoint reuse on `experimental.document.ImageToMarkup` (defaults unchanged).
- **Experiment:** `experiments.document.DocumentParsing.donutFollowUp` (wave 9 rows unchanged).
- **Toy task:** `GlyphDocuments` forms; reading pretraining on 2k synthetic pages; curves to 3k fine-tuning steps (Run), 300 in the test.
- **Baselines:** wave 9's OCR plus rules and OCR plus layout tagger (with the oracle text detector); Donut without reading pretraining at the same total steps.
- **Expected metric:** field F1 and tree-edit accuracy against steps; reading CER after pretraining. Hypothesis: with reading pretraining Donut's F1 rises well above 0 within a few hundred fine-tuning steps and approaches the OCR pipelines at the long budget, while the same total steps without pretraining stay near 0.
- **Source:** [arXiv:2111.15664](https://arxiv.org/abs/2111.15664)

### 10.6.12 VQA bridges with LLaVA's two-stage recipe and an unfrozen LM (Liu et al., NeurIPS 2023 and CVPR 2024) — follow-up
In wave 9 every bridge (projector 0.650, Q-Former 0.640, gated cross-attention 0.650, early fusion 0.635) sat at the majority-answer prior of about 0.65, below the 0.77 ceiling given by feeding the frozen LM the true scene description. LLaVA trains in two stages: first only the projector, on caption-style alignment data (here: the image to its 16-slot description), then projector and LM together (or LM with LoRA) on question answering. The LM is never fully frozen in stage 2. The follow-up adds the alignment stage, LoRA on the LM in stage 2, and a 4x budget, and checks the text-only ability of the LM afterwards.
- **Block:** a `TwoStageBridgeTrainer` in `experimental.multimodal` (stage 1 description alignment, stage 2 QA with optional `LoraAdapterDense` on the LM), reusing wave 9's `PatchProjector`, `QueryTransformer` and `GatedCrossAttentionAdapter`.
- **Experiment:** `experiments.multimodal.FusionShootout.vqaFollowUp` (wave 9 rows unchanged).
- **Toy task:** wave 9's balanced VQA scenes with the `VqaFixture` frozen parts.
- **Baselines:** wave 9's bridges, the text-only LM, and the DESCRIPTION ceiling.
- **Expected metric:** VQA accuracy, description-alignment accuracy after stage 1, and the LM's text loss after training. Hypothesis: the alignment stage lifts every bridge above the answer prior (toward the 0.77 ceiling), LoRA in stage 2 adds a few points with little text-loss damage, and the projector remains as good as the Q-Former, as LLaVA-1.5 found.
- **Source:** [arXiv:2304.08485](https://arxiv.org/abs/2304.08485), [arXiv:2310.03744](https://arxiv.org/abs/2310.03744)

### 10.6.13 Transfusion with a VAE latent and U-Net patch layers, at matched FLOPs (Zhou et al., ICLR 2025) — follow-up
Wave 9's Transfusion used raw 2×2 pixel patches with no VAE or U-Net, matched compute by steps, and generated at chance (0.133 and 0.067), while the VQ-token models reached 0.567 and 0.433. The paper encodes images with a VAE into a latent, embeds latent patches with small U-Net down and up blocks (which it reports matter a lot at small scale), trains with a λ-weighted diffusion loss, and compares at matched FLOPs. The follow-up adds those pieces, using wave 6's latent-diffusion autoencoder.
- **Block:** a `latentPatches` option on `experimental.multimodal.TransfusionModel` (VAE encoder/decoder from `LatentDiffusion`'s autoencoder, U-Net patch embedder and un-embedder, loss weight λ, FLOP counter; defaults unchanged).
- **Experiment:** `experiments.multimodal.FusionShootout.transfusionFollowUp` (wave 9 rows unchanged).
- **Toy task:** wave 9's `DigitGlyphs` caption-and-generate task, plus `ToyImages`.
- **Baselines:** wave 9's pixel-patch Transfusion and VQ-token `EarlyFusion` (shared and separate), compared at matched training FLOPs.
- **Expected metric:** generation class accuracy (frozen classifier), caption accuracy, and FLOPs. Hypothesis: with latent patches and U-Net layers Transfusion generates far above chance and matches or beats VQ tokens on generation at matched FLOPs, while caption accuracy stays comparable, which is the paper's claim.
- **Source:** [arXiv:2408.11039](https://arxiv.org/abs/2408.11039)

### 10.6.14 Once-for-all with in-place distillation and a longer budget: BigNAS (Yu et al., ECCV 2020) — follow-up
At 160 steps, wave 9's progressive shrinking reached 0.625 for the largest subnet and 0.330 for the smallest, against 0.860 and 0.570 for networks trained from scratch, and its accuracy predictor was unreliable (Kendall τ from −0.3 to 0.7 with 24 points). Wave 9 already added the sandwich rule. BigNAS trains all subnets at once with in-place distillation: in each step the largest network is trained on labels and every sampled subnet (always including the smallest) on the largest network's soft predictions, with initialisation and learning-rate tweaks for stability; no progressive phases and no retraining are needed. The follow-up adds in-place distillation, runs 4x and 16x longer, and fits the predictor on 200 measured subnets.
- **Block:** an `inplaceDistillation` mode on `experimental.edge.ElasticSupernet`, and more sample points for `SubnetSearch`'s predictor (defaults unchanged).
- **Experiment:** `experiments.edge.EdgeDeployment.onceForAllFollowUp` (wave 9 rows unchanged).
- **Toy task:** wave 9's `ShapeImages` supernet at 640 and 2,560 steps (Run), 300 in the test.
- **Baselines:** wave 9's progressive shrinking, largest-only, random sampling and per-subnet from-scratch rows at the same budgets.
- **Expected metric:** largest- and smallest-subnet accuracy against budget, accuracy across the MAC range, and predictor τ. Hypothesis: at 16x the supernet's subnets come within 2 points of from-scratch training, in-place distillation gets there sooner than progressive shrinking, and the predictor reaches τ above 0.7 with 200 points.
- **Source:** [arXiv:2003.11142](https://arxiv.org/abs/2003.11142)

### 10.6.15 Memory-aware pipeline schedules: V-shape schedules and an exact scheduler for ZB-H2 (Qi et al., NeurIPS 2024) — follow-up
Wave 9's greedy scheduler never let ZB-H2 beat ZB-H1 (both 0.150 bubble at p = 4, m = 8) because it did not use ZB-H2's extra activation memory, and DualPipe got worse at p = 8, m = 32 once its limits were relaxed. Qi et al. build schedules by repeating one building block: the lifespan of a stage (from forward to the end of its backward passes) sets the activation memory, and "V-shape" placements, where each device holds one early and one late stage, balance lifespans. V-Min, V-Half and V-ZB reach about 1/3, 1/2 and the full 1F1B activation memory, with V-ZB nearly bubble-free. The follow-up adds the building-block constructor, a memory-constrained exact (branch-and-bound) scheduler for small p and m, and a sweep over the memory limit.
- **Block:** `experimental.parallel.BuildingBlockSchedules` (V-Min, V-Half, V-ZB from offsets δ0, δ1) and a `MemoryAwareScheduler` (exact search under a per-device activation limit), both producing `PipelineSchedule` instances checked by its validator and runnable by `PipelineRunner`.
- **Experiment:** `experiments.parallel.ParallelismShootout.pipelineMemoryFollowUp` (wave 9 rows unchanged).
- **Toy task:** wave 9's simulated pipelines at p ∈ {4, 8}, m ∈ {8, 16, 32}, with unequal forward, input-gradient and weight-gradient times taken from `ParallelCostModel`.
- **Baselines:** wave 9's GPipe, 1F1B, interleaved, ZB-H1, ZB-H2 and DualPipe from the greedy generator.
- **Expected metric:** bubble fraction against peak activation memory (a Pareto plot), and agreement of real threaded runs with single-device gradients. Hypothesis: the exact scheduler finds ZB-H2-memory schedules with a bubble near 0 where the greedy generator stays at ZB-H1's; V-Half matches 1F1B's bubble at half its memory; V-Min's bubble grows with m when pass times differ, as the paper reports.
- **Source:** [arXiv:2405.15362](https://arxiv.org/abs/2405.15362), [arXiv:2401.10241](https://arxiv.org/abs/2401.10241)

### 10.6.16 A GAN vocoder trained to convergence, with mel-loss warm-up (Kong et al., NeurIPS 2020) — follow-up
Wave 9's GAN and diffusion vocoders were trained for only 6 and 10 steps inside tests (digit error 0.67–1.0), so 9.5.11 and 9.5.12 vocoded with Griffin–Lim and the vocoder hypotheses were never tested. HiFi-GAN needs many updates; a common recipe warms the generator up on the mel-spectrogram L1 loss alone before switching on the multi-period and multi-scale discriminators, which shortens adversarial training. The follow-up trains `GanVocoder` with warm-up for 2k–20k steps outside the test suite (a Run, with a checkpoint saved for reuse), keeps a short smoke test, and re-vocodes 9.5.11 and 9.5.12's outputs with it.
- **Block:** a `melWarmup(steps)` phase and checkpoint save/load on `experimental.audio.GanVocoder` (defaults unchanged); the same checkpointing for `DiffusionVocoder`.
- **Experiment:** `experiments.audio.AudioGeneration.vocoderFollowUp` (wave 9 rows unchanged).
- **Toy task:** wave 9's `TalkingDigits` mel-to-waveform task.
- **Baselines:** Griffin–Lim with 32 and 100 iterations; wave 9's short-trained GAN and diffusion vocoders; the ground-truth waveform.
- **Expected metric:** multi-resolution STFT distance and the frozen recogniser's digit error against training steps; generation speed. Hypothesis: after warm-up and 10–20k steps the GAN vocoder beats Griffin–Lim on STFT distance at equal digit error and is much faster at inference than DiffWave; re-vocoding lowers the digit error of the flow-matching TTS of 9.5.11.
- **Source:** [arXiv:2010.05646](https://arxiv.org/abs/2010.05646)

---

## Notes on dependencies between items

**Imperfect supervision (WP1)**
- First build `experimental.data.LabelCorruption` (it must keep the clean labels for scoring and must not change `ToyTasks`, `ShapeImages` or `TabularStressSuite` sampling), then create `experimental.semisup` and `experimental.labels` and the theme `experiments.labels`.
- 10.1.1 → 10.1.6 share `SemiSupervisedShootout` and one training loop with labelled and unlabelled batches; build 10.1.1 first. 10.1.3's `MixMatch` is reused by 10.1.8's DivideMix, and 10.1.4's `FixMatch` is the base of 10.1.5 and 10.1.6.
- 10.1.6 needs WP3's `experimental.ssl.Sigreg` (10.3.6). Agree its signature early (a static loss on a [batch, dim] `Var`, with the number of slices and quadrature points as arguments); until it lands, WP1 uses a local copy that is replaced afterwards.
- 10.1.7 → 10.1.10 share `NoisyLabelShootout`; 10.1.8 uses `GaussianMixture` and `DataPruner` unchanged. 10.1.11 → 10.1.12 share `LongTailShootout`; the balanced validation split comes from `LabelCorruption`.
- 10.1.13 needs a max-flow solver: expose the Edmonds–Karp currently inside `AttentionRollout` (wave 9) as a public helper rather than copying it. 10.1.17 reuses `EntropicTransport`.

**Shift and the open world (WP2)**
- First build `experimental.data.ShiftSuites` and a cached trained classifier per task (`OodShootout` trains once per seed and shares it across 10.2.1–10.2.6). The spurious-colour split needs a public accessor on `CausalAndFairness`' coloured-bar generator (wave 9 regenerated it privately in `ConceptErasure`); add the accessor without changing its sampling, so wave 6–7 rows are unchanged.
- 10.2.1 → 10.2.3 build on each other (10.2.3's AdaSCALE ends in 10.2.1's energy score); 10.2.2 reuses `KnnIndex`. 10.2.4–10.2.5 retrain classifiers and are independent of the post-hoc items.
- 10.2.6 → 10.2.7 share `SelectivePrediction` and the risk–coverage code; 10.2.6 compares with `PopulationEnsemble.predictOrAbstain` read-only.
- 10.2.8 uses `Calibration`'s temperature scaling unchanged. 10.2.9 → 10.2.11 and 10.2.15 share `DomainAdaptation`; the gradient-reversal op lives in `experimental.shift` (core-worthy). 10.2.15 wraps `TestTimeAdaptation` and `SharpnessAware` without changing them.
- 10.2.12 → 10.2.14 and 10.2.16 share `DomainGeneralization`; 10.2.12–10.2.13 use `GroupRobustTraining` as-is. 10.2.16 needs `SelfSupervisedBenchmark`'s SimCLR encoder through a public method (WP3 adds rows to that class; coordinate so that only WP3 edits it, and WP2 calls a public `pretrainEncoder` method that WP3 exposes first).
- 10.2.17 adds a sampler next to `GreenListWatermark` in `experimental.llm` and changes neither it nor `LogitsProcessor`.

**Representation learning (WP3)**
- First create `experimental.ssl`, `experimental.metric` and `experimental.multitask`. `SelfSupervisedBenchmark` gains new methods; its `Method` enum rows and their numbers stay unchanged (new methods may use a second enum). Expose `pretrainEncoder` for WP2 10.2.16 and WP3 10.3.12 early.
- Build 10.3.6's `Sigreg` first: WP1 10.1.6 depends on it. 10.3.1–10.3.3 share the two-view pipeline; 10.3.4 and 10.3.6 also run on the patch-transformer branch next to MAE and I-JEPA.
- 10.3.7 → 10.3.10 share `MetricLearningShootout` and the unseen-class retrieval protocol (same embedding size, steps and augmentation for every loss). 10.3.9's `MarginSoftmax` is reused by WP6 10.6.5; make it public early.
- 10.3.11 → 10.3.12 add to `experimental.ml` and reuse `MinibatchOptimalTransport`'s Hungarian solver for clustering accuracy; 10.3.12 needs `pretrainEncoder`.
- 10.3.13 → 10.3.16 share a per-task-gradient helper in `experimental.multitask` (one backward per task on the shared parameters); build it first. The `ClickLogs` task and `MultiGateExperts` (wave 9) are used read-only.
- 10.3.17 adds options to `ContrastConsistentSearch` and a method to `LatentKnowledge`; wave 9's rows must stay reproducible with defaults.

**Deep RL and follow-ups (WP4)**
- First create `experimental.deeprl` and the `LearningCurveHarness` of 10.4.14 (equal environment steps, IQM, bootstrap intervals; reuse wave 9's `EvalStatistics`), which every RL item in this WP reports through. Add the pixel GridWorld/`KeyDoorGrid` renderer as a wrapper (do not change the environments).
- 10.4.2 → 10.4.4 share `ModernOffPolicy` and the critic-width/UTD sweeps; 10.4.2 reuses `BatchRenorm`, 10.4.3–10.4.4 compare with `ResidualRlNetworks`/`HlGaussValue` read-only. 10.4.6 and 10.4.7 share the LAP replay buffer; build 10.4.7's first. 10.4.6 reuses `DreamerAgent`'s symlog/two-hot helpers (make them public if they are not).
- 10.4.5, 10.4.8, 10.4.9 and 10.4.12 share `ValueBasedRecipes` and the pixel grids; 10.4.12 builds on `MuZeroLite` without changing it.
- 10.4.10, 10.4.11 and 10.4.17 share `PlasticityAndScale`; 10.4.11 uses `PsgdKron` as-is. 10.4.17 reuses wave 9's `CurveExtrapolator` for power-law fits.
- 10.4.1 adds rows to `OfflineRlComparison` and 10.4.13 to `GoalsImitationAndTeams`; their existing rows must stay unchanged.
- 10.4.15 and 10.4.16 touch wave 9's `experimental.align` classes only through new options with unchanged defaults; `RewardOveroptimization` and `ProcessSupervision` keep their wave 9 test classes and numbers. These two are the most compute-hungry items in the WP; run their Runs in the background from day one.

**Classic and structured ML (WP5)**
- First build the four generators (`EventStreams`, `IrregularSeries`, `TemporalGraphs`, `ToyTreebank`) and create `experimental.boost`, `experimental.events` and `experimental.parse` and the themes `experiments.events` and `experiments.parsing`.
- 10.5.1 → 10.5.6 share `BoostingShootout`; all reuse `GradientBoostedTrees.Loss`, and 10.5.3 and 10.5.6 reuse `RegressionForest`'s tree grower (wave 9 exposed it). Core `GradientBoostedTrees` is not changed; GOSS and EFB live in `experimental.boost` until measured (core-worthy). 10.5.5 adds rows to `TabularShootout` without changing its existing rows (wave 7 and 9 numbers stay).
- 10.5.7 → 10.5.9 share `PointProcessShootout`, the thinning simulator and the time-rescaling test; build 10.5.7 first (its simulator generates the data). 10.5.10 → 10.5.11 share `IrregularSeriesShootout`; 10.5.10 uses `NeuralOde`'s solver unchanged.
- 10.5.12 is independent. 10.5.13 → 10.5.14 share `RelationalGraphs` and the filtered-ranking metric of `KnowledgeGraphEmbedding`; 10.5.14 uses 10.5.13's composition functions. 10.5.15 is independent.
- 10.5.16 → 10.5.17 share `ToyTreebank` and an encoder; build the treebank's head rules so that both trees come from one derivation.

**Restoration, speech and follow-ups (WP6)**
- First build `DegradedImages` and `SpeakerVoices` (the latter reuses `SpokenDigitRecognition.speak`'s synthesis; it must not change that method, which `SourceSeparation`, `KeywordSpotters` and the audio tests depend on), and create `experimental.restore` and `experimental.speech` and the themes `experiments.restoration` and `experiments.speech`.
- 10.6.1 → 10.6.4 share `RestorationShootout`; 10.6.2's FFDNet is the denoiser plugged into 10.6.4, so build 10.6.2 before 10.6.4. 10.6.1's OCR metric uses wave 9's `LineRecognizer` read-only.
- 10.6.5 → 10.6.8: 10.6.6's VAD feeds 10.6.7 (noise estimation) and 10.6.8 (segmentation); 10.6.5's embeddings feed 10.6.8. Build 10.6.6 and 10.6.5 first. 10.6.5 uses WP3's `MarginSoftmax` (10.3.9) or a local copy until it lands. 10.6.8 reuses `PermutationInvariantLoss` and `SpectralClustering`.
- 10.6.9 adds a row to `VisionShootout`; existing rows unchanged.
- The follow-ups 10.6.10–10.6.16 each add options with unchanged defaults and a new experiment method; the wave 9 test classes and numbers must stay reproducible. Their long budgets are Runs outside the test suite, which is the point of each follow-up. 10.6.12 reuses wave 9's `VqaFixture` from the test sources; move its builder into main sources if the Run needs it. 10.6.13 reuses wave 6's `LatentDiffusion` autoencoder read-only. 10.6.15's exact scheduler must emit schedules that wave 9's validator accepts.

**Cross-package**
- The thirteen new packages (`experimental.semisup`, `.labels`, `.ood`, `.shift`, `.ssl`, `.metric`, `.multitask`, `.deeprl`, `.boost`, `.events`, `.parse`, `.restore`, `.speech`) and the new generators in `experimental.data` (`LabelCorruption`, `ShiftSuites`, `EventStreams`, `IrregularSeries`, `TemporalGraphs`, `ToyTreebank`, `DegradedImages`, `SpeakerVoices`) have no dependencies on each other, so all six work packages can start at once.
- Cross-WP interfaces to agree early, each with a local stub until the owner lands it: `experimental.ssl.Sigreg` (WP3 → WP1 10.1.6), `experimental.metric.MarginSoftmax` (WP3 → WP6 10.6.5), `SelfSupervisedBenchmark.pretrainEncoder` (WP3 → WP2 10.2.16).
- Shared helpers to expose rather than copy: the Edmonds–Karp max-flow inside `AttentionRollout` (WP1 10.1.13), `CausalAndFairness`' coloured-bar generator (WP2 10.2.13), `DreamerAgent`'s symlog/two-hot helpers (WP4 10.4.6), the Hungarian solver of `MinibatchOptimalTransport` (WP3 10.3.11), `KnnIndex` (WP2 10.2.2), and `RegressionForest`'s grower (WP5 10.5.3, 10.5.6). Whichever WP lands first makes the method public; the others import it.
- Edits to shared experiment classes, each owned by exactly one WP in this wave: `SelfSupervisedBenchmark` (WP3), `LatentKnowledge` (WP3), `OfflineRlComparison`, `GoalsImitationAndTeams`, `RewardOveroptimization`, `ProcessSupervision` (WP4), `TabularShootout` (WP5), `VisionShootout`, `DetectionShootout`, `DocumentParsing`, `FusionShootout`, `EdgeDeployment`, `ParallelismShootout`, `AudioGeneration` (WP6). No other WP edits them.
- WP2 10.2.15 and wave 8's `ShiftAdaptation` both use `TestTimeAdaptation`; WP2 adds new classes and does not change TENT's or MEMO's defaults.
- Core-worthy items, to land in a separate PR from the experiments once measured: 10.1.4/10.1.5 FixMatch-style training (`brainiac-train`), 10.1.7 robust losses and 10.1.11 logit adjustment (`brainiac-nn` losses), 10.2.1–10.2.3 OOD scores and 10.2.8 label-shift correction (`brainiac-train` `Evaluation`), the gradient-reversal op (10.2.9, `brainiac-autograd`), 10.3.6 SIGReg and 10.3.9 margin softmax (`brainiac-nn`), 10.3.16 FAMO (`brainiac-train`), 10.4.5 PQN and 10.4.6 MR.Q (`brainiac-rl`), 10.5.1 GOSS/EFB and 10.5.2 ordered target statistics (core `GradientBoostedTrees`), 10.6.1 ESPCN (`brainiac-nn`, with `PixelShuffle`) and 10.6.5 statistics pooling (`brainiac-audio`).
