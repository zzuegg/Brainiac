# Brainiac: wave 9 ideas

*Compiled 2026-11-18 against `claude/project-thread-8zvcl0` (main tree), plus every worktree under `.claude/worktrees/*` (42 worktrees). Waves 1–8 implemented about 900 techniques; see `experiment-results.md`, `unimplemented-ideas.md`, `wave5-ideas.md`, `wave6-ideas.md`, `wave7-ideas.md` and `wave8-ideas.md`. Wave 9 targets the areas that are still thin after eight waves:*

- *(a) recommender, ads and search ranking at depth (multi-task ranking, unbiased learning to rank from clicks, diversity, feature-interaction networks, budget pacing), causal inference and uplift, online experimentation, and survival analysis;*
- *(b) post-training: preference and RLHF variants not yet covered, reward-model robustness, process rewards and verifiers for mathematical reasoning, and code models (execution prediction, repair, execution-based reranking);*
- *(c) data and evaluation: online data mixing and curricula, data selection, synthetic data generation, continual pretraining, and evaluation methodology (calibration estimators, LLM-as-judge biases and their statistical correction, contamination, error bars, adaptive overfitting, rankings against budget);*
- *(d) interpretability tools not yet built, document/OCR and table understanding toys, and multimodal fusion;*
- *(e) world models and video generation toys, audio generation, and fair-test follow-ups of the wave 8 scientific-ML negatives;*
- *(f) hardware-aware training (pipeline, tensor, ZeRO and expert parallelism simulated on threads, recomputation schedules, asynchronous SGD), streaming and online learning, and edge/TinyML.*

*Seven items are fair-test follow-ups of earlier negatives, marked "— follow-up" in their titles: 9.1.18 (prior-fitted networks with more pretraining, R-wave7), 9.2.17 (ArCHer tuning, R-wave8), 9.3.16 (long-horizon transformer rankings against budget, R-wave8), 9.4.14 (DETR at a long budget, R-wave8), 9.5.14 and 9.5.15 (hard-PDE PINN fixes, R-wave8) and 9.5.16 (Transolver against FNO at a scale sweep, R-wave8).*

*It also adds the late-2025/2026 items found by a web sweep, each checked against the paper's abstract or text rather than a search snippet: RankMixer (ByteDance, 2025), TabICL (ICML 2025), SquaredPO (ICLR 2026), PRIME implicit process rewards (2025), process advantage verifiers (ICLR 2025), 1-shot RLVR (2025), the Code World Model's execution-trace mid-training (Meta, 2025), Aioli data mixing (2025), the stability gap in continual pretraining (2024/2025), Rogan–Gladen and PPI corrections for LLM judges (2025; ICML 2026), the Kernel Divergence Score for contamination (ICML 2025), the local learning coefficient (AISTATS 2025), Transfusion (ICLR 2025), Self Forcing (NeurIPS 2025), history-guided video diffusion (ICML 2025) and DINO-WM (ICML 2025).*

*Every item was grepped over all `*.java` sources under `src/` in all modules, in the main tree and in all 42 worktrees: 64,595 files, 3,555 unique by content after de-duplication (2,754 distinct file names). The search was case-insensitive and used class names, paper names, author-coined terms and javadoc keywords. An item is listed only when nothing matched, or when the only matches were unrelated. Examples of unrelated matches:*

- *"PLE" (progressive layered extraction) matching Gemma 3n's per-layer embeddings (`PerLayerEmbeddingBlock`, `Gemma3nBlocks`), "position bias" matching T5's relative position bias and `HstuBlock`, "pacing" matching `Curriculum`'s pacing function and the noise schedules (not ad budget pacing), and "auction/bid" matching nothing.*
- *"sampled softmax" matching core `Functional`'s sampled softmax (Jean et al.) and `TwoTowerRetrieval`'s in-batch softmax, which has no logQ correction (that block is the 9.1.8 baseline); "hazard/survival" matching `HazardGrid` (constrained RL) and the hazard rate of `BayesianOnlineChangePoint` (no time-to-event model); "meta-learner" matching MAML, Reptile and SNAIL, while "T-learner/S-learner" correctly matched `CausalEstimators` and `CausalAndFairness` (the 9.1.10 baselines); "e-value" matching `DType`.*
- *"WARM" matching every warm-up schedule, "BOND" matching the bond dimension of `MatrixProductState`, "XPO" matching "export"/"exponent", "R-DPO" matching nothing but `PreferenceTuning`'s `RDPO` case, which is robust DPO for noisy labels (Chowdhury et al.), not length-regularised DPO (Park et al.), "PRIME" matching prime moduli in `SecureAggregation` and `MinHashDeduplicator`, "SLiC" matching "slice", and "repair" matching the regrowth of `NeuralCellularAutomaton`.*
- *"Elo/arena" matching `SelfPlayArena` and `EloLeague` (agent leagues), "judge" matching the rubric and self-rewarding judges (the 9.3.11 baselines), "paraphrase" matching `SyntheticCorpus` and `DataCuration`'s paraphrase arm (so WRAP-style rephrasing is left out), and "distribution matching" matching `DistributionMatchingDistiller` (DMD one-step distillation, not dataset distillation).*
- *"tuned lens" correctly matching `LogitLens` and "control task/selectivity" correctly matching `LinearProbe` (both left out); "interchange intervention" matching `LowRankIntervention`'s javadoc (ReFT trains a task intervention; nothing learns an alignment from base/source pairs as DAS does); "CCS" matching `SecureAggregation`; "layout" matching tensor memory layouts in the backends; "projector" matching core `VisionLanguageModel`/`Mmproj` (inference of a pretrained LLaVA-style projector, no fusion training); "FiLM" matching `Cnaps` and `FluxTransformer`.*
- *"ZeRO" correctly matching `Offloaded` (ZeRO-Offload, left out) and otherwise every "zero"; "pipeline" matching `DecodingPipeline`; "interleaved" matching `ToolInterleavedRollout` and `InterleavedRopeStack`; "recompute" matching `Recompute` (uniform per-block checkpointing; 9.6.5 adds selective and optimal schedules); "online softmax" correctly matching `AttentionKernels` (FlashAttention-style tiling, left out); "auxiliary-loss-free" correctly matching `FineGrainedMoe.updateBalanceBias` (left out); "Oja" correctly matching `PlasticityRule` (streaming PCA, left out); "keyword spotter" matching a javadoc line in `MelSpectrogram`; "supernet" matching `WeightSharingNas` (ENAS, no elastic once-for-all network); "requantized" matching `Llama`'s KV-cache requantisation; "staleness" matching `LevelReplay`.*
- *"causal training" correctly matching `PinnTraining` (so 9.5.14 adds time-marching and curriculum regularisation, not causal weights); "Gauss–Newton" matching `CurvatureProducts`, `LevenbergMarquardt` and `HessianFree` (generic, nothing for PINN function-space natural gradients); "HiFi-GAN" matching core `HifiGanBlock` (an inference-only generator: no discriminators, no training); "vocoder" matching the F5, VITS and Vocos inference code; "free bits" matching `ImageAutoencoders`' VAE (not DreamerV3's KL balancing in `DreamerAgent`); "V-JEPA" matching a javadoc mention in `JepaWorldModel`; "rollout" matching every RL rollout (no attention rollout).*

## How this was checked

- **Already implemented, so left out.** These turned up in the systematic check but already have code:
  - Recsys and causal: DCN-V2 (`CrossNetwork`), factorization machines, Wide & Deep, DeepFM, NeuMF, LightGCN, Mult-VAE, EASE, SASRec, BERT4Rec, DIN (`TargetAttentionPooling`), HSTU, TIGER, two-tower retrieval with an in-batch softmax, BM25, DPR, ColBERT, SPLADE, IVF-PQ, Matryoshka embeddings, RankNet/LambdaRank/LambdaMART/ListNet (`RankingLosses`, `LambdaMart`), off-policy estimators (IPS, SNIPS, DR, Switch), LinUCB and Thompson sampling, TabPFN-style `PriorFittedNetwork`, FT-Transformer, TabM, RealMLP, NODE, NAM; IPW and AIPW (`CausalEstimators`), a linear T-learner, an S-learner MLP, TARNet and DragonNet (`TreatmentEffectNet`), NOTEARS, DAGMA, GES, PC, IRM, GroupDRO, V-REx, JTT.
  - Post-training: PPO for LMs, DPO, IPO, KTO, ORPO, SimPO, cDPO, robust DPO, DPOP, β-DPO, iterative/online DPO (`SelfImprovement`), SPIN, self-rewarding, Constitutional AI, GRPO and its twelve variants (Dr. GRPO, DAPO, RLOO, ReMax, REINFORCE++, GSPO, CISPO, GMPO, PSR/NSR, W-REINFORCE, pass@k), Clip-Cov/KL-Cov, forking tokens, Intuitor, TTRL, STaR, Quiet-STaR, rejection-sampling fine-tuning, a Bradley–Terry `RewardModel`, `ProcessRewardModel` (trained on gold step labels), GenRM (`GenerativeVerifier`), rubric rewards, step-level MCTS, ToolRL/ReTool, ArCHer, RobustFill and DeepCoder (`ProgramGenerator`, `GuidedEnumerator`).
  - Data and evaluation: temperature mixing, staged annealing and DoReMi (`DataMixture`), RegMix (`RegressionMixer`), curriculum with pacing functions, self-paced learning, sequence-length warm-up, mixup/CutMix (`BatchMixing`), Rho-1, quality classifiers, MinHash de-duplication, EntiGraph, model collapse, EL2N/forgetting scores, In-Run Shapley, KNN-Shapley, Data-OOB, prototype pruning, active learning; ECE/MCE/NLL/Brier and temperature scaling (`Calibration`), conformal classification, Min-K%++ (`ContaminationDetector`), Fluid Benchmarking (`IrtEvaluator`), pass@k, Hyperband/BOHB, population-based training.
  - Interpretability: SAEs (TopK, BatchTopK, JumpReLU, Matryoshka), transcoders, cross-layer transcoders and attribution graphs, crosscoders, activation patching, AtP/AtP*, SPD, weight-sparse transformers, logit lens and tuned lens, linear probes with control tasks, steering and persona vectors, ROME and AlphaEdit, ReFT, Grad-CAM, TCAV, integrated gradients, KernelSHAP, LIME, counterfactuals, influence functions and TracIn.
  - Multimodal and vision: CLIP/SigLIP, Chameleon-style early fusion, a VLM inference path (core `VisionLanguageModel`), DETR-lite with DN queries, CenterNet, RetinaNet-lite, `ShapeScenes`.
  - World models and audio: RSSM/PlaNet, a Dreamer-style agent, TD-MPC2, Genie latent actions, DIAMOND, Dreamer 4 shortcut forcing, JEPA world model, diffusion forcing, ConvLSTM and SimVP, MaskGIT, VAR, WaveNet, DDSP, RVQ codec, a MusicGen-style codec LM, Conv-TasNet, DPRNN; core inference for Whisper, VITS, Kokoro, F5-TTS, Vocos and HiFi-GAN blocks.
  - Systems: data parallelism (`DataParallel`, `TcpCollective`), ZeRO-Offload (`Offloaded`), per-block recomputation (`Recompute`), ring attention, FlashAttention-style tiling (`AttentionKernels`), DiLoCo/Streaming DiLoCo/DeMo, PowerSGD, top-k with error feedback, QSGD, ActNN, reversible blocks, 4-bit optimizer states, auxiliary-loss-free balancing and expert-choice routing with a capacity factor, streaming deep RL, reservoir replay, DER++, EWC, BOCPD and PELT, Oja's rule, BitNet, LSQ, PACT/DoReFa, QAT recipes, BitFit, early exit, ENAS.
- **Considered but cut** (to stay near 6 × 17, or too close to an existing block):
  - ThinkPRM (Khalifa et al., 2025): a long-CoT generative PRM, too close to `GenerativeVerifier` (GenRM-CoT) at toy scale.
  - WRAP/BeyondWeb rephrasing and Self-Instruct/Evol-Instruct: the paraphrase arm of `DataCuration` and EntiGraph cover rephrasing, and the toy LMs cannot write instructions.
  - Synthetic control and difference-in-differences: no fair toy beyond hand-made panels, and nothing in the repo to compare with.
  - CTR calibration (isotonic, field-aware): small next to 9.3.10's calibration estimators; fold in as a row there if wanted.
  - Arena-style Bradley–Terry rankings: `EloLeague` already ranks players from pairwise games.
  - Contrastive preference learning (CPL): needs regret labels the toy preference data do not have.
  - VALL-E: too close to `CodecLanguageModel` with a phoneme prompt.
  - XNOR-Net binary networks and ProxylessNAS latency predictors: close to BitNet and folded into 9.6.15's predictor respectively.
  - A second data-assimilation follow-up (score-based DA lost 5.02 vs 0.92) and Koopman follow-ups: the gaps were too large to expect a budget change to close them.
- **CPU-toy filter.** Every item is pure Java with no external libraries, runs in seconds to minutes on a CPU, and names a task that exists in `brainiac-experiments` or a tiny new generator next to it, plus a baseline already in the repo. The tasks used are:
  - `SyntheticRecsys`, `TabularPrior`, `TabularStressSuite`, `CausalAndFairness`' confounded data, `ToyTasks` digits/spirals, `ShapeImages`, `ShapeScenes`, `DigitGlyphs`, `ToyImages`, `SeasonalSeries`, `ChaoticSeries`.
  - `ArithmeticTask`, `ChainTask`, `ListPrograms`, `CountingTasks`, `SyntheticCorpus`, `SyntheticGrammar`, `SimulatedUser`, the toy LMs (`TinyLm`, `WindowLm`, `ContextLm`).
  - `TalkingDigits`' formant synthesiser and `ToyAudioCodec`, the `brainiac-rl` environments and the `DreamingAgent` grid frames, `PdeProblems` and `MechanicalSystems`.
  - New small generators named per work package: `ClickLogs`, `CausalScenarios` and `SurvivalData` (WP1), `SkillTasks` (WP3), `GlyphDocuments` and `ToyTables` (WP4), `DriftingStreams` (WP6).

**Format.** Wave 9 keeps wave 8's structure (checks, summary, one section per work package, dependency notes), with each item as a short block:

- **Block**: the reusable class, in `io.github.zzuegg.brainiac.experimental.<pkg>`.
- **Experiment**: the runnable measurement, in `io.github.zzuegg.brainiac.experiments.<theme>`.
- **Toy task**, **Baselines** (all already in the repo) and **Expected metric** (the number to report, with the hypothesis to test).

**Where an item should live:**

- `experimental.<pkg>`: a new block in `brainiac-experiments`. Fourteen new packages are proposed, each owned by one work package so that no two agents create the same package:
  - WP1: `experimental.causal` and `experimental.survival` (ranking blocks stay in the existing `experimental.recsys`).
  - WP2: `experimental.align` and `experimental.code`.
  - WP3: `experimental.eval` and `experimental.synth` (mixing and selection stay in `experimental.data`).
  - WP4: `experimental.interp`, `experimental.document` and `experimental.multimodal`.
  - WP5: `experimental.world` and `experimental.audio` (PINN and operator follow-ups extend the existing `experimental.sciml`).
  - WP6: `experimental.parallel`, `experimental.stream` and `experimental.edge`.
  - Existing blocks are not moved: `CausalEstimators`, `TreatmentEffectNet`, `RewardModel`, `ProcessRewardModel`, `LogitLens`, `LinearProbe`, `DreamerAgent`, `Recompute`, `DataParallel` and the others stay where they are, and the new packages import them.
  - New data generators go in `experimental.data` under the names listed above; each name belongs to one WP.
- `experiments.<theme>`: a new or extended runnable experiment. New themes: `experiments.causal` and `experiments.survival` (WP1), `experiments.alignment` and `experiments.code` (WP2), `experiments.evaluation` (WP3), `experiments.document` (WP4), `experiments.world` (WP5), `experiments.parallel`, `experiments.streaming` and `experiments.edge` (WP6). Other items extend `experiments.recsys`, `experiments.reasoning`, `experiments.data`, `experiments.interp`, `experiments.multimodal`, `experiments.vision`, `experiments.audio` and `experiments.sciml`.
- **core-worthy**: belongs in a core module once measured.

## Summary

| # | Work package | Items |
|---|---|---|
| WP1 | Ranking, ads and search at depth; causal inference, uplift and online experimentation; survival analysis | 18 |
| WP2 | Post-training: preference and RLHF variants, reward robustness, process rewards and verifiers for math, and code models | 17 |
| WP3 | Data and evaluation: online mixing and curricula, data selection, synthetic data, continual pretraining, and evaluation methodology | 17 |
| WP4 | Interpretability tools, document/OCR and table understanding, and multimodal fusion | 17 |
| WP5 | World models and video generation, audio generation, and scientific-ML follow-ups | 16 |
| WP6 | Hardware-aware and parallel training, streaming and online learning, and edge/TinyML | 17 |
| | **Total** | **102** |

---

## WP1. Ranking, ads and search at depth; causal inference, uplift and online experimentation; survival analysis (18)

Target: ranking blocks in the existing `experimental.recsys`; two new packages, `experimental.causal` and `experimental.survival`. Measurements go in new experiments `experiments.recsys.MultiTaskRanking` (9.1.1–9.1.2), `experiments.recsys.ClickLogLearning` (9.1.3), `experiments.recsys.RerankingDiversity` (9.1.4), `experiments.recsys.ClickModels` (extended, 9.1.5–9.1.7), `experiments.recsys.CollaborativeFiltering` (extended, 9.1.8), `experiments.recsys.BudgetPacing` (9.1.9), the new theme `experiments.causal` (`HeterogeneousEffects` for 9.1.10–9.1.13, `InstrumentalVariables` for 9.1.14, `OnlineExperiments` for 9.1.15), the new theme `experiments.survival` (`SurvivalShootout`, 9.1.16–9.1.17) and `experiments.recsys.TabularShootout` (9.1.18). Three new generators in `experimental.data`: `ClickLogs` (ranked lists from `SyntheticRecsys` with hidden graded relevance, clicks from a position-based or cascade model, impression/click/conversion labels, ad bids and budgets), `CausalScenarios` (confounded observational data with known per-unit effects, uplift campaigns, an instrument, and A/B tests with a pre-period covariate) and `SurvivalData` (covariates, a known hazard, right censoring, competing risks). Baselines are the existing recsys and causal blocks at the same parameter count, over at least 3 seeds.

### 9.1.1 Multi-task ranking with expert gates: MMoE and PLE (Ma et al., KDD 2018; Tang et al., RecSys 2020)
Ranking systems predict several labels at once (click, like, finish, convert), and a shared-bottom network suffers "negative transfer" when the tasks conflict. MMoE replaces the shared bottom with a pool of expert MLPs and gives each task its own softmax gate over them, so each task mixes the experts it needs. PLE (progressive layered extraction) separates task-specific experts from shared experts and stacks several such extraction layers, which fixes MMoE's "seesaw" where one task improves only at another's expense.
- **Block:** `experimental.recsys.MultiGateExperts` (MMoE and PLE layers; task towers).
- **Experiment:** `experiments.recsys.MultiTaskRanking`.
- **Toy task:** `ClickLogs` with two labels whose correlation is swept from +0.9 to −0.5 (same hidden interaction, or opposite-sign interactions).
- **Baselines:** shared-bottom `WideAndDeep` and `CrossNetwork` with two heads; two single-task models; the hard-routed `FineGrainedMoe` with task-indexed routing.
- **Expected metric:** AUC per task across the correlation sweep. Hypothesis: all methods tie at high correlation; at negative correlation shared-bottom loses > 0.03 AUC on one task, MMoE recovers part and PLE matches the single-task models on both.
- **Source:** [doi:10.1145/3219819.3220007](https://doi.org/10.1145/3219819.3220007), [doi:10.1145/3383313.3412236](https://doi.org/10.1145/3383313.3412236)

### 9.1.2 ESMM: conversion modelling over the entire impression space (Ma et al., SIGIR 2018)
Conversion rate (CVR) is normally trained only on clicked impressions but served on all impressions: sample selection bias, plus very sparse labels. ESMM never trains CVR directly. It trains pCTR on all impressions and pCTCVR = pCTR × pCVR on all impressions (the label is "clicked and converted"), with the CVR tower sharing the embeddings of the CTR tower, so pCVR is learned implicitly over the whole space.
- **Block:** `experimental.recsys.EntireSpaceModel` (reuses 9.1.1's towers).
- **Experiment:** `MultiTaskRanking`.
- **Toy task:** `ClickLogs` with a hidden conversion probability that depends on features differently for unclicked items (so click-space training is biased); report on all impressions, where the true CVR is known.
- **Baselines:** a CVR model trained on clicked samples only; the same with inverse-propensity weights from a CTR model (`OffPolicyEstimators`-style); 9.1.1 MMoE.
- **Expected metric:** CVR AUC and calibration error on all impressions. Hypothesis: ESMM beats click-space training by > 0.05 AUC on the full space and is better calibrated than IPW, whose weights are high-variance.
- **Source:** [arXiv:1804.07931](https://arxiv.org/abs/1804.07931)

### 9.1.3 Unbiased learning to rank from clicks: IPW, regression EM and the dual learning algorithm (Joachims et al., WSDM 2017; Wang et al., WSDM 2018; Ai et al., SIGIR 2018)
Clicks confound relevance with position: under the position-based model P(click) = P(examined at rank k) × P(relevant). Counterfactual LTR reweights each click by 1/propensity(k), with propensities from a swap-intervention experiment. Regression EM fits the click model itself, alternating an E-step over the hidden "examined" and "relevant" variables with a regression of relevance on features. The dual learning algorithm learns the ranker and a propensity model jointly, each serving as the other's inverse-propensity weight.
- **Block:** `experimental.recsys.ClickDebiasing` (PBM and cascade click models fitted by EM, IPW weights, DLA).
- **Experiment:** `experiments.recsys.ClickLogLearning`.
- **Toy task:** `ClickLogs`: rankings produced by a logging ranker, clicks from a PBM with η ∈ {0.5, 1, 2} position decay, plus a cascade-model split.
- **Baselines:** `RankingLosses` (RankNet/LambdaRank) on naive clicks; LambdaMART on naive clicks; an oracle trained on true relevance.
- **Expected metric:** NDCG@5 on true relevance, and propensity-curve error. Hypothesis: naive clicks lose > 0.1 NDCG at strong position bias; IPW with true propensities closes most of the gap, regression EM and DLA close most of it without intervention data; under a cascade model the PBM-based fixes degrade.
- **Source:** [arXiv:1608.04468](https://arxiv.org/abs/1608.04468), [doi:10.1145/3159652.3159732](https://doi.org/10.1145/3159652.3159732), [arXiv:1804.05938](https://arxiv.org/abs/1804.05938)

### 9.1.4 Diversity re-ranking: MMR and determinantal point processes (Carbonell & Goldstein, SIGIR 1998; Chen et al., NeurIPS 2018)
A relevance-sorted list is often redundant. Maximal marginal relevance greedily adds the item maximising λ·relevance − (1−λ)·max similarity to the items already chosen. A DPP instead scores a whole set by the determinant of a kernel L = diag(q)·S·diag(q) (quality times similarity), which rewards sets that are both relevant and spread out. MAP inference is NP-hard, but the fast greedy algorithm adds items by incremental Cholesky updates in O(k²) per step.
- **Block:** `experimental.recsys.DiverseReranker` (MMR and greedy DPP MAP).
- **Experiment:** `experiments.recsys.RerankingDiversity`.
- **Toy task:** `SyntheticRecsys` preferences with items assigned to latent topics and users with multi-topic tastes (a user is satisfied per covered topic).
- **Baselines:** relevance-sorted top-k from `TwoTowerRetrieval` and `LightGcn`; random re-ranking of the top 3k.
- **Expected metric:** NDCG@10, α-NDCG and intra-list diversity, as a λ trade-off curve. Hypothesis: the DPP dominates MMR's curve (higher α-NDCG at equal NDCG); both beat relevance-only on α-NDCG by > 0.1.
- **Source:** [doi:10.1145/290941.291025](https://doi.org/10.1145/290941.291025), [arXiv:1709.05135](https://arxiv.org/abs/1709.05135)

### 9.1.5 Feature-interaction networks: AutoInt and xDeepFM (Song et al., CIKM 2019; Lian et al., KDD 2018)
Two ways of learning high-order feature crosses explicitly. AutoInt embeds every field and runs multi-head self-attention across fields with residual connections, so each layer adds one order of interaction and the attention maps show which crosses matter. xDeepFM's compressed interaction network (CIN) computes vector-wise crosses: each layer takes outer products of the previous layer's feature maps with the field embeddings and compresses them with 1×1 convolutions, then sums each layer into the output.
- **Block:** `experimental.recsys.FieldAttention` (AutoInt), `experimental.recsys.CompressedInteraction` (CIN).
- **Experiment:** `experiments.recsys.ClickModels` (new rows).
- **Toy task:** `SyntheticRecsys.clicks` extended with planted 2nd-, 3rd- and 4th-order field interactions.
- **Baselines:** `FactorizationMachine`, `WideAndDeep`, DeepFM, `CrossNetwork` (DCN-V2), `FtTransformer`.
- **Expected metric:** log loss and AUC, sliced by the order of the planted interaction. Hypothesis: all tie on 2nd order; AutoInt and CIN beat DeepFM on 3rd/4th order but not DCN-V2.
- **Source:** [arXiv:1810.11921](https://arxiv.org/abs/1810.11921), [arXiv:1803.05170](https://arxiv.org/abs/1803.05170)

### 9.1.6 RankMixer: hardware-friendly token mixing for ranking (ByteDance, 2025)
Feature groups are concatenated and split into T fixed-width feature tokens. Each block applies parameter-free multi-head token mixing (head h of every token is gathered into new token h, replacing self-attention) and then a per-token FFN, a separate FFN for each token so that heterogeneous feature subspaces do not share parameters, with residuals and layer norm; the outputs are mean-pooled. To scale, the per-token FFNs become a sparse MoE with ReLU routing and a regulariser on the number of active experts, trained densely and served sparsely (DTSI). Checked against arXiv:2507.15551.
- **Block:** `experimental.recsys.RankMixerBlock` (token mixing, per-token FFN, optional ReLU-routed sparse experts reusing `ReluRoutedMoe`).
- **Experiment:** `ClickModels`, as a parameter/FLOP sweep.
- **Toy task:** 9.1.5's task with 16 fields grouped into 4–8 tokens.
- **Baselines:** 9.1.5 AutoInt, `CrossNetwork`, `FtTransformer`, `HstuBlock`, at equal parameters and equal multiply-adds.
- **Expected metric:** AUC against parameters and against wall-clock per batch. Hypothesis: RankMixer matches AutoInt's AUC at a fraction of the time (no attention), and its loss keeps falling with width where DCN-V2 flattens.
- **Source:** [arXiv:2507.15551](https://arxiv.org/abs/2507.15551)

### 9.1.7 Compact embeddings for huge ID vocabularies: the hashing trick and quotient–remainder embeddings (Weinberger et al., ICML 2009; Shi et al., KDD 2020)
ID embedding tables dominate ranking-model memory. The hashing trick maps IDs into fewer buckets and accepts collisions. Compositional embeddings give each ID a unique vector built from smaller tables: the quotient-remainder trick looks up row ⌊i/m⌋ in one table and row i mod m in another and combines them (product, sum or concatenation), so two IDs never share both rows; mixed-dimension variants give frequent IDs wider vectors.
- **Block:** `experimental.recsys.CompactEmbedding` (hashed, QR, mixed-dimension).
- **Experiment:** `ClickModels`, as a memory sweep.
- **Toy task:** `SyntheticRecsys.clicks` with a 100k-value Zipf-distributed ID field carrying signal.
- **Baselines:** full embedding table; frequency thresholding (rare IDs → one OOV row); `LowRankDense`-factored table.
- **Expected metric:** AUC against embedding bytes (1x, 1/4, 1/16, 1/64). Hypothesis: QR keeps AUC within 0.005 of full at 1/16 where hashing loses > 0.02; the gap concentrates on tail IDs.
- **Source:** [arXiv:0902.2206](https://arxiv.org/abs/0902.2206), [arXiv:1909.02107](https://arxiv.org/abs/1909.02107)

### 9.1.8 Sampling-bias-corrected in-batch negatives and mixed negative sampling (Yi et al., RecSys 2019; Yang et al., WWW 2020)
An in-batch softmax uses other positives as negatives, so popular items appear as negatives far more often than uniform sampling would, and the model is pushed to under-score them. The logQ correction subtracts log of each item's sampling probability (estimated by streaming frequency counts: the average gap between an item's appearances) from its logit. Mixed negative sampling adds a set of uniformly sampled items to the in-batch negatives, which covers tail items that rarely appear in batches.
- **Block:** `experimental.recsys.CorrectedInBatchSoftmax` (streaming frequency estimator, logQ, mixed negatives).
- **Experiment:** `experiments.recsys.CollaborativeFiltering` (new rows).
- **Toy task:** `SyntheticRecsys.preferences` with Zipf item popularity (exponent 0.8 and 1.2).
- **Baselines:** `TwoTowerRetrieval` with plain in-batch softmax; `BprMatrixFactorization` with uniform negatives; full softmax (feasible at toy size) as the oracle.
- **Expected metric:** Recall@20 overall and on head/tail items; frequency-estimate error. Hypothesis: logQ closes most of the gap to full softmax on head items; mixed negatives give the best tail recall.
- **Source:** [doi:10.1145/3298689.3346996](https://doi.org/10.1145/3298689.3346996), [doi:10.1145/3366424.3386195](https://doi.org/10.1145/3366424.3386195)

### 9.1.9 Budget pacing for ad auctions: throttling and Lagrangian bid shading (Agarwal et al., KDD 2014; Balseiro & Gur, Management Science 2019)
An advertiser with budget B over T auctions wants maximal value without running out early. Throttling pacing participates in each auction with a probability adjusted by a feedback controller tracking a target spend curve. Dual-gradient pacing bids v/(1+μ), shading the value by a Lagrange multiplier μ that is updated after each auction by μ ← max(0, μ − ε(B/T − spend)); this is asymptotically optimal in second-price auctions with stochastic competition.
- **Block:** `experimental.recsys.BudgetPacer` (PID throttling, dual-gradient bid shading) and a second-price auction simulator.
- **Experiment:** `experiments.recsys.BudgetPacing`.
- **Toy task:** `ClickLogs` auctions: per-impression value = pCTR × value per click from a 9.1.5 model, competing bids lognormal with a daily traffic cycle.
- **Baselines:** bid truthfully until the budget runs out; uniform throttling with a fixed rate; the offline optimum (knapsack with hindsight, solved by `LinearProgram`).
- **Expected metric:** value obtained as a share of the hindsight optimum, and budget-exhaustion time. Hypothesis: dual-gradient pacing reaches > 0.9 of the optimum; greedy exhausts early in the day and loses the cheap evening traffic.
- **Source:** [doi:10.1145/2623330.2623366](https://doi.org/10.1145/2623330.2623366), [arXiv:1703.06223](https://arxiv.org/abs/1703.06223)

### 9.1.10 Meta-learners for heterogeneous effects: X-, R- and DR-learners (Künzel et al., PNAS 2019; Nie & Wager, Biometrika 2021; Kennedy, 2023)
Ways of turning any regression model into a conditional average treatment effect (CATE) estimator. The X-learner fits outcome models per arm, imputes each unit's individual effect from the other arm's model, regresses those on x, and blends the two estimates by the propensity; it shines when one arm is small. The R-learner minimises the Robinson-residual loss Σ((y − m(x)) − (t − e(x))τ(x))² with cross-fitted nuisances m and e. The DR-learner regresses the AIPW pseudo-outcome on x.
- **Block:** `experimental.causal.MetaLearners` (S, T, X, R, DR on any `experimental.ml` regressor or MLP; cross-fitting helper).
- **Experiment:** `experiments.causal.HeterogeneousEffects`.
- **Toy task:** `CausalScenarios`: Künzel's simulation set (unbalanced arms, a complex outcome with a simple effect, confounding) with known τ(x).
- **Baselines:** `CausalAndFairness`' S-learner MLP and linear T-learner, TARNet and DragonNet (`TreatmentEffectNet`).
- **Expected metric:** PEHE (√ mean squared CATE error). Hypothesis: X-learner wins with 10% treated; R- and DR-learners win when the effect is simpler than the outcome; TARNet wins only with balanced arms.
- **Source:** [arXiv:1706.03461](https://arxiv.org/abs/1706.03461), [arXiv:1712.04912](https://arxiv.org/abs/1712.04912), [arXiv:2004.14497](https://arxiv.org/abs/2004.14497)

### 9.1.11 Double/debiased machine learning with cross-fitting (Chernozhukov et al., Econometrics Journal 2018)
Plugging flexible ML nuisance models into an effect estimate gives a regularisation bias that does not shrink at the √n rate. DML uses a Neyman-orthogonal score, so first-order errors in the nuisances cancel, and cross-fitting: nuisances are trained on K−1 folds and evaluated on the held-out fold, so the score is not computed on data the nuisances overfit. In the partially linear model the estimate is the regression of the outcome residual y − ĝ(x) on the treatment residual t − ê(x), with an analytic standard error.
- **Block:** `experimental.causal.DoubleMl` (partially linear and interactive models, K-fold cross-fitting, sandwich standard errors).
- **Experiment:** `HeterogeneousEffects` (ATE section).
- **Toy task:** `CausalScenarios` with a nonlinear confounding function, n from 500 to 8,000.
- **Baselines:** `CausalEstimators` naive, IPW and AIPW (same nuisance models, no cross-fitting); DragonNet.
- **Expected metric:** ATE bias, RMSE and 95%-interval coverage over 50 replications. Hypothesis: AIPW without cross-fitting under-covers (< 0.85) with random-forest nuisances; DML holds ≈ 0.95 coverage.
- **Source:** [arXiv:1608.00060](https://arxiv.org/abs/1608.00060)

### 9.1.12 Causal forests and generalized random forests (Wager & Athey, JASA 2018; Athey, Tibshirani & Wager, Annals of Statistics 2019)
A random forest whose splits maximise the heterogeneity of the treatment effect between children rather than outcome fit, built with honesty: each tree uses half its subsample to choose splits and the other half to estimate leaf effects, which makes the estimates asymptotically normal with valid confidence intervals. GRF generalises this to any moment condition by using the forest as an adaptive kernel (the weights are how often a training point shares a leaf with the query) and solving the local moment equation.
- **Block:** `experimental.causal.CausalForest` (honest effect-splitting trees, out-of-bag weights, variance by the bootstrap of little bags; reuses `RandomForest` utilities).
- **Experiment:** `HeterogeneousEffects`.
- **Toy task:** `CausalScenarios` with a step-shaped effect in two of ten covariates.
- **Baselines:** 9.1.10 meta-learners on `RandomForest`; TARNet; T-learner forests.
- **Expected metric:** PEHE and CATE-interval coverage. Hypothesis: the causal forest gives the only calibrated intervals (≈ 0.9 coverage for 90%) and matches the R-learner's PEHE.
- **Source:** [arXiv:1510.04342](https://arxiv.org/abs/1510.04342), [arXiv:1610.01271](https://arxiv.org/abs/1610.01271)

### 9.1.13 Uplift modelling, Qini curves and targeting policies (Radcliffe, 2007; Jaśkowski & Jaroszewicz, 2012; Gutierrez & Gérardy, 2017)
Marketing does not want to know who buys, but whose buying the treatment changes: "persuadables" versus "sure things" and "do-not-disturbs". The class-transformation trick (with 50/50 randomisation, z = y·t + (1−y)(1−t) has P(z=1|x) = ½(1 + uplift(x))) turns uplift into ordinary classification. Uplift models are judged by Qini and uplift curves (incremental conversions when targeting the top fraction by predicted uplift) and by the value of the policy "treat if uplift > cost".
- **Block:** `experimental.causal.UpliftModels` (class transformation, two-model, uplift trees with a KL/ΔΔP split criterion) and `experimental.causal.UpliftMetrics` (Qini, AUUC, policy value).
- **Experiment:** `HeterogeneousEffects` (uplift section).
- **Toy task:** `CausalScenarios` campaign: four latent customer types with planted proportions, randomised treatment, cost per contact.
- **Baselines:** a response model (target the most likely buyers, the usual mistake); 9.1.10 T-/X-learners; random targeting.
- **Expected metric:** Qini coefficient and profit at the optimal targeting fraction. Hypothesis: the response model is close to random on Qini because it targets sure things; uplift trees and class transformation beat it by a wide margin.
- **Source:** [proceedings.mlr.press/v67/gutierrez17a](https://proceedings.mlr.press/v67/gutierrez17a.html)

### 9.1.14 Instrumental variables: two-stage least squares and DeepIV (Angrist & Pischke; Hartford et al., ICML 2017)
When an unobserved confounder drives both treatment and outcome, no adjustment on observed covariates works, but an instrument z that moves the treatment and affects the outcome only through it does. 2SLS regresses treatment on z, then outcome on the fitted treatment. DeepIV does this nonparametrically: stage one fits a mixture density network for p(t | x, z), stage two trains the outcome network h(t, x) so that E over t ~ p(t|x,z) of h(t, x) matches y, with Monte Carlo gradients.
- **Block:** `experimental.causal.InstrumentalVariables` (2SLS, DeepIV with a `GaussianHead` mixture first stage).
- **Experiment:** `experiments.causal.InstrumentalVariables`.
- **Toy task:** `CausalScenarios`: Hartford et al.'s airline-demand simulation (price confounded by an unobserved demand shock, fuel cost as instrument).
- **Baselines:** an MLP regressing outcome on (t, x) directly; `CausalEstimators` AIPW on observed covariates; linear 2SLS.
- **Expected metric:** out-of-sample counterfactual MSE of h(t, x) under price interventions. Hypothesis: the direct MLP has the lowest observational loss and the worst counterfactual error; DeepIV beats linear 2SLS when the demand curve is nonlinear.
- **Source:** [arXiv:1612.09596](https://arxiv.org/abs/1612.09596)

### 9.1.15 Online experimentation: CUPED variance reduction and always-valid sequential tests (Deng et al., WSDM 2013; Johari et al., 2017)
Two fixes for A/B tests. CUPED subtracts θ(X − E[X]) from each user's metric, with X a pre-experiment covariate (last month's spend) and θ = cov(Y, X)/var(X); the estimate stays unbiased and its variance falls by a factor 1 − ρ². Peeking at a fixed-horizon t-test inflates false positives far above α; the mixture sequential probability ratio test gives always-valid p-values that may be monitored continuously and stopped at any time.
- **Block:** `experimental.causal.OnlineExperiments` (CUPED and its regression form CUPAC with an ML predictor, mSPRT, a fixed-horizon t-test).
- **Experiment:** `experiments.causal.OnlineExperiments`.
- **Toy task:** `CausalScenarios` A/B streams: heavy-tailed revenue, pre-period correlation ρ ∈ {0.3, 0.7}, true lifts of 0 and 1–3%.
- **Baselines:** the plain difference in means with a t-test; the same t-test checked daily (peeking).
- **Expected metric:** false-positive rate under the null with daily peeking; power and sample size to detect a 2% lift. Hypothesis: peeking pushes false positives from 0.05 to > 0.2 while mSPRT stays ≤ 0.05; CUPED at ρ = 0.7 halves the sample size needed.
- **Source:** [doi:10.1145/2433396.2433413](https://doi.org/10.1145/2433396.2433413), [arXiv:1512.04922](https://arxiv.org/abs/1512.04922)

### 9.1.16 Survival analysis: Kaplan–Meier, Cox proportional hazards and DeepSurv (Kaplan & Meier, 1958; Cox, 1972; Katzman et al., 2018)
Time-to-event data (churn, failure, death) are right-censored: for some units we only know the event had not happened yet. Kaplan–Meier estimates the survival curve as a product of conditional survival over event times. Cox's model assumes h(t|x) = h₀(t)·exp(f(x)) and fits f by maximising the partial likelihood, which cancels the unknown baseline hazard; Breslow's estimator recovers h₀. DeepSurv makes f an MLP trained on the negative log partial likelihood.
- **Block:** `experimental.survival.SurvivalModels` (Kaplan–Meier, Cox with Breslow ties and baseline, DeepSurv loss) and `experimental.survival.SurvivalMetrics` (Harrell's C-index, integrated Brier score with IPCW).
- **Experiment:** `experiments.survival.SurvivalShootout`.
- **Toy task:** `SurvivalData`: Weibull event times with linear and nonlinear risk functions, 20–60% censoring.
- **Baselines:** treating censored rows as events (or dropping them) with an MLP regressing log time; logistic regression on "event before t*"; `GradientBoostedTrees` on time.
- **Expected metric:** C-index and integrated Brier score. Hypothesis: the censoring-naive regressions lose > 0.05 C-index at 60% censoring; DeepSurv beats linear Cox only on the nonlinear risk.
- **Source:** [arXiv:1606.00931](https://arxiv.org/abs/1606.00931)

### 9.1.17 Beyond proportional hazards: discrete-time neural hazards, DeepHit and random survival forests (Gensheimer & Narasimhan, 2019; Lee et al., AAAI 2018; Ishwaran et al., 2008)
When hazards cross over time, Cox's proportionality assumption breaks. A discrete-time model splits time into bins and predicts a conditional hazard per bin with a sigmoid, with a likelihood that uses censored rows up to their censoring bin. DeepHit predicts the full distribution of the first hitting time with a softmax over bins, trained by log-likelihood plus a ranking loss on comparable pairs, and handles competing risks with one output head per cause. Random survival forests split nodes by the log-rank statistic and average cumulative-hazard estimates over trees.
- **Block:** `experimental.survival.DiscreteHazardNet`, `experimental.survival.DeepHit`, `experimental.survival.SurvivalForest`.
- **Experiment:** `SurvivalShootout`.
- **Toy task:** `SurvivalData` with crossing hazards (one subgroup fails early, another late) and a two-cause competing-risk split.
- **Baselines:** 9.1.16 Cox and DeepSurv; Kaplan–Meier per subgroup; cause-specific Cox for competing risks.
- **Expected metric:** time-dependent C-index, integrated Brier score, and cumulative-incidence error per cause. Hypothesis: on crossing hazards Cox/DeepSurv lose > 0.05 time-dependent C-index to the discrete-time net and DeepHit; on proportional hazards everything ties.
- **Source:** [arXiv:1805.00917](https://arxiv.org/abs/1805.00917), [ojs.aaai.org DeepHit](https://ojs.aaai.org/index.php/AAAI/article/view/11842), [arXiv:0811.1645](https://arxiv.org/abs/0811.1645)

### 9.1.18 Prior-fitted networks with a longer pretraining budget and TabICL's column-then-row architecture (Qu et al., ICML 2025) — follow-up
Wave 7's `PriorFittedNetwork` scored 0.720 / 0.602 / 0.498 against 0.807 / 0.842 / 0.855 for the best baselines "at a CPU pre-training budget". This item re-runs it as a budget sweep (1x, 4x, 16x, 64x the wave 7 synthetic datasets) and adds TabICL's architecture: a shared induced set transformer embeds each column as a set (so cell embeddings depend on the column's distribution), a small row-wise transformer with RoPE and CLS tokens mixes a row's features into one vector, and a dataset-wise transformer does in-context learning where test rows attend only to training rows. Pretraining follows a curriculum that grows the synthetic dataset size. Checked against arXiv:2502.05564.
- **Block:** `experimental.recsys.ColumnRowIcl` (TabICL-lite); `PriorFittedNetwork` unchanged.
- **Experiment:** `experiments.recsys.TabularShootout` (budget sweep rows).
- **Toy task:** `TabularPrior` for pretraining; the three wave 7 evaluation tables plus `TabularStressSuite`.
- **Baselines:** wave 7's `PriorFittedNetwork` at its original budget; `GradientBoostedTrees`, `RandomForest`, `RealMlp`, `TabM`.
- **Expected metric:** accuracy against pretraining datasets seen (log scale), and the budget at which the PFN line crosses the best baseline, if it does. Hypothesis: accuracy rises steadily with budget; TabICL-lite reaches the PFN's 64x accuracy at about 16x; whether either passes GBDT within one CPU-hour is the open question.
- **Source:** [arXiv:2502.05564](https://arxiv.org/abs/2502.05564), [arXiv:2207.01848](https://arxiv.org/abs/2207.01848)

---

## WP2. Post-training: preference and RLHF variants, reward robustness, process rewards and verifiers for math, and code models (17)

Target: two new packages, `experimental.align` (preference losses beyond `PreferenceLosses`, reward-model robustness, process supervision) and `experimental.code` (execution traces, repair and execution-based reranking). Measurements go in the new theme `experiments.alignment` (`RewardOveroptimization` for 9.2.1–9.2.2, `GeneralPreferences` for 9.2.3–9.2.4, `PreferenceShootout` for 9.2.5–9.2.7), `experiments.reasoning` (`ProcessSupervision` for 9.2.8–9.2.12, `ArithmeticRl` for 9.2.13), the new theme `experiments.code` (`CodeExecution` for 9.2.14–9.2.16) and `experiments.agents.ToolAgentBench` (9.2.17). Preference items use the toy LMs and the `PreferenceTuning` task (and the noisy-label setup of `NoisyPreferenceTest`) with a known gold reward, so over-optimisation can be measured against the truth; math items use `ArithmeticTask`/`ChainTask` (step-structured solutions with exact checking); code items use `ListPrograms`, whose programs can be executed and traced. Baselines are the existing post-training blocks at equal samples and equal LM calls, over at least 3 seeds.

### 9.2.1 Reward-model ensembles and weight-averaged reward models against over-optimisation (Gao et al., ICML 2023; Coste et al., ICLR 2024; Ramé et al., ICML 2024)
Optimising a learned proxy reward first raises and then lowers the gold reward as the policy drifts (Goodhart); Gao et al. fitted this as a function of √KL. Ensembles of reward models trained from different seeds, combined conservatively (the mean minus a multiple of the standard deviation, or the worst member), slow the decline. WARM fine-tunes several reward models from one shared initialisation and averages their weights (they stay linearly mode-connected), which gives most of an ensemble's robustness at the cost of one model at inference.
- **Block:** `experimental.align.RewardEnsemble` (mean, worst-case, uncertainty-penalised) and `experimental.align.WeightAveragedReward`.
- **Experiment:** `experiments.alignment.RewardOveroptimization`.
- **Toy task:** a gold reward over toy-LM outputs (`RubricRewards`' satisfaction score); proxies trained on 1–4k noisy preference pairs labelled by it; best-of-n and PPO against each proxy.
- **Baselines:** a single `RewardModel`; a KL penalty sweep in `PpoLmTrainer`; `RubricReward`.
- **Expected metric:** gold reward against KL from the reference (the over-optimisation curve) and its peak. Hypothesis: the single proxy peaks early and collapses; worst-case ensembles and WARM push the peak to higher KL, with WARM matching the ensemble at one model's cost.
- **Source:** [arXiv:2210.10760](https://arxiv.org/abs/2210.10760), [arXiv:2310.02743](https://arxiv.org/abs/2310.02743), [arXiv:2401.12187](https://arxiv.org/abs/2401.12187)

### 9.2.2 BOND: best-of-N distillation (Sessa et al., 2024)
Best-of-N sampling is a strong, simple alignment method but costs N samples per query. BOND trains the policy to match the best-of-N distribution itself. That distribution has a closed form in terms of the reference policy's reward quantiles, p_BoN(y) ∝ p_ref(y)·F(r(y))^(N−1), so the policy can be fitted by distribution matching with a Jeffreys divergence (a mix of forward and reverse KL). The iterative variant repeatedly distils best-of-2 against a moving anchor, reaching large effective N.
- **Block:** `experimental.align.BestOfNDistiller` (quantile estimation from reference samples, Jeffreys loss, iterated anchor).
- **Experiment:** `RewardOveroptimization`.
- **Toy task:** 9.2.1's task, N ∈ {4, 16, 64}.
- **Baselines:** best-of-N at inference with `RewardModel`; `RejectionSamplingFineTuning` on best-of-N outputs; DPO on best/worst pairs; PPO.
- **Expected metric:** gold reward and KL against inference cost (samples per query). Hypothesis: BOND reaches the gold reward of best-of-16 at 1 sample per query with lower KL than PPO; rejection-sampling fine-tuning lags it.
- **Source:** [arXiv:2407.14622](https://arxiv.org/abs/2407.14622)

### 9.2.3 Nash learning from human feedback and self-play preference optimisation (Munos et al., ICML 2024; Wu et al., 2024)
Bradley–Terry reward models assume preferences are transitive; real ones (and mixtures of annotators) need not be. NLHF learns a preference model P(y ≻ y′ | x) directly and seeks the Nash equilibrium of the two-player game in which each player picks a response, a policy no other policy is preferred to, by mirror descent towards a regularised opponent. SPPO approximates this with iterated self-play: sample responses, estimate each one's win rate against the current policy with the preference model, and regress the log-ratio log(π/π_t) onto η(win rate − ½).
- **Block:** `experimental.align.SelfPlayPreference` (SPPO loss, Nash-MD update), with a pairwise preference model.
- **Experiment:** `experiments.alignment.GeneralPreferences`.
- **Toy task:** toy-LM responses scored by three annotator types whose majority preference is cyclic (rock–paper–scissors over styles), plus a transitive control.
- **Baselines:** DPO and iterative DPO (`SelfImprovement`); `RewardModel` + PPO.
- **Expected metric:** worst-case win rate against a panel of fixed policies, and against the other methods. Hypothesis: on transitive data all tie; on cyclic data DPO collapses onto one style (low worst-case win rate) while SPPO keeps a mixed policy with worst-case win rate near ½.
- **Source:** [arXiv:2312.00886](https://arxiv.org/abs/2312.00886), [arXiv:2405.00675](https://arxiv.org/abs/2405.00675)

### 9.2.4 General preference models with preference embeddings (Zhang et al., 2024)
A scalar reward can represent only transitive preferences. The general preference model embeds each response as a vector v(y) in 2k dimensions and scores P(y ≻ y′) = σ(⟨R·v(y), v(y′)⟩) with R a fixed block-diagonal skew-symmetric matrix of 2×2 rotations. Skew symmetry guarantees P(y ≻ y′) + P(y′ ≻ y) = 1, cyclic preferences become representable, and each response is embedded once, so scoring n candidates is linear rather than quadratic in n, unlike a pairwise model.
- **Block:** `experimental.align.PreferenceEmbeddingModel`.
- **Experiment:** `GeneralPreferences`.
- **Toy task:** 9.2.3's cyclic and transitive preference sets; held-out pairs.
- **Baselines:** Bradley–Terry `RewardModel`; a pairwise classifier on concatenated responses (the quadratic-cost upper bound).
- **Expected metric:** held-out preference accuracy, and accuracy on the cyclic triples. Hypothesis: BT cannot exceed 2/3 on cyclic triples; GPM matches the pairwise classifier at linear cost; as the reward source for 9.2.3 it beats BT.
- **Source:** [arXiv:2410.02197](https://arxiv.org/abs/2410.02197)

### 9.2.5 Displacement- and over-optimisation-resistant DPO: χ²PO and SquaredPO (Huang et al., ICLR 2025; ICLR 2026)
Two one-line changes to DPO's link function. χ²PO replaces the log-ratio in DPO with φ(z) = z + log z, z = π/π_ref, which corresponds to χ²-plus-KL regularisation; its heavier penalty on leaving the reference's support makes it provably robust to over-optimisation. SquaredPO comes from f-DPO with the non-convex f(t) = ½(log t)², whose minimum sits at t = 1: in training it acts like DPO with an effective β divided by the policy-to-reference ratio, so when the chosen response's likelihood falls, regularisation strengthens and pushes it back up. This targets likelihood displacement, where DPO lowers both responses. Checked against arXiv:2407.13399 and the ICLR 2026 paper (arXiv:2602.06788).
- **Block:** `experimental.align.LinkDpoLosses` (χ²PO, SquaredPO, general f-DPO).
- **Experiment:** `experiments.alignment.PreferenceShootout`.
- **Toy task:** `PreferenceTuning` pairs with high-similarity chosen/rejected responses (where displacement is strongest), plus 9.2.1's gold reward.
- **Baselines:** DPO, IPO, DPOP (also designed against displacement), SimPO (`PreferenceLosses`).
- **Expected metric:** change in chosen log-likelihood over training, win rate, and gold reward against KL. Hypothesis: DPO lowers the chosen likelihood; SquaredPO and DPOP keep it near or above the start, and χ²PO keeps the highest gold reward at large step counts.
- **Source:** [arXiv:2407.13399](https://arxiv.org/abs/2407.13399), [arXiv:2602.06788](https://arxiv.org/abs/2602.06788)

### 9.2.6 Length-regularised DPO (Park et al., ACL Findings 2024)
Preference data often prefer longer answers, and DPO exploits this: response length grows without quality. R-DPO adds a length-difference term to the DPO margin, β·log ratios − α(|y_w| − |y_l|), which removes the incentive to win on length alone; the derivation is a length penalty in the underlying RLHF objective.
- **Block:** `experimental.align.LengthRegularizedDpo`.
- **Experiment:** `PreferenceShootout`.
- **Toy task:** `PreferenceTuning` with a synthetic judge that prefers the correct answer and, independently, the longer one (length bias 0.2–0.4).
- **Baselines:** DPO; SimPO (length-normalised margin); `LengthControlledReward` used as a GRPO reward.
- **Expected metric:** mean length, length-controlled win rate (9.3.11's regression), and accuracy. Hypothesis: DPO grows the length 2x with a falling controlled win rate; R-DPO holds length with no accuracy loss and beats SimPO on controlled win rate.
- **Source:** [arXiv:2403.19159](https://arxiv.org/abs/2403.19159)

### 9.2.7 XPO: exploratory preference optimisation (Xie et al., ICLR 2025)
Online DPO samples new pairs from the current policy, but with no incentive to explore it can miss responses the reference rarely produces. XPO adds one term to the online DPO objective, an optimism bonus α·Σ log π(y′) on responses sampled from the reference, which encourages the policy to move probability to regions outside the reference's support. With it, online preference learning has sample-efficiency guarantees like those of optimistic RL.
- **Block:** `experimental.align.ExploratoryPreferenceOptimizer` (online loop with a preference oracle).
- **Experiment:** `PreferenceShootout`.
- **Toy task:** toy-LM task where the best response has reference probability below 1e-3 (a rare digit pattern), with a gold oracle labelling online pairs.
- **Baselines:** iterative/online DPO (`SelfImprovement`); offline DPO; `SpinTrainer`.
- **Expected metric:** gold reward against preference queries. Hypothesis: online DPO plateaus on the reference's mode; XPO finds the rare best response within the budget.
- **Source:** [arXiv:2405.21046](https://arxiv.org/abs/2405.21046)

### 9.2.8 VinePPO: Monte Carlo values in place of a critic (Kazemnejad et al., 2024)
PPO for LMs learns a value network, which for reasoning tasks is badly inaccurate at intermediate steps, so advantages are noisy. Because a language environment can be reset to any prefix, VinePPO estimates V(prefix) by sampling K completions from the prefix and averaging their final rewards, at each reasoning step. The resulting step-level advantages replace GAE's; the extra rollouts are used only for value estimation.
- **Block:** `experimental.align.MonteCarloValuePpo` (wraps `PpoLmTrainer` with MC step values).
- **Experiment:** `experiments.reasoning.ProcessSupervision`.
- **Toy task:** `ChainTask` multi-step arithmetic with step separators.
- **Baselines:** `PpoLmTrainer` with a learned critic; GRPO and RLOO (`GrpoTrainer`); 9.2.10 PAV rewards.
- **Expected metric:** test accuracy against total sampled tokens, and value-estimate error at each step against exhaustive MC. Hypothesis: the learned critic's step-value error is near that of a constant; VinePPO reaches GRPO's final accuracy in fewer policy updates but more tokens.
- **Source:** [arXiv:2410.01679](https://arxiv.org/abs/2410.01679)

### 9.2.9 Math-Shepherd: process labels from Monte Carlo rollouts (Wang et al., ACL 2024)
Human step labels (PRM800K) are expensive. Math-Shepherd labels each step automatically by its potential to reach the right answer: from the prefix ending at step s, a completer samples N continuations; the "hard" label is 1 if any reaches the gold answer, the "soft" label is the fraction that do. A PRM trained on these labels is then used for reranking and as a step-level reward for PPO.
- **Block:** `experimental.align.RolloutStepLabeler` (hard and soft labels; feeds `ProcessRewardModel.fit`).
- **Experiment:** `ProcessSupervision`.
- **Toy task:** `ChainTask` solutions with injected step errors whose positions are known (for scoring the labels).
- **Baselines:** `ProcessRewardModel` trained on gold step labels; an outcome reward model; `GenerativeVerifier`.
- **Expected metric:** step-label accuracy against gold, and best-of-16 accuracy when reranking with each PRM. Hypothesis: soft labels from N = 8 rollouts reach > 0.85 step agreement; the resulting PRM is within 0.03 of the gold-label PRM at best-of-16, and above the outcome model.
- **Source:** [arXiv:2312.08935](https://arxiv.org/abs/2312.08935)

### 9.2.10 Process advantage verifiers: rewarding progress (Setlur et al., ICLR 2025)
A PRM that predicts "is the solution still correct" (a Q-value) gives little signal to RL, because it scores steps that are correct but do not help. PAVs score each step by its advantage: the change in the probability of eventually succeeding, measured under a separate prover policy rather than the base policy. Provers that are complementary to the base policy, even weaker ones, give the most useful progress signal. The step rewards are used for beam search at test time and as dense rewards added to the outcome reward in online RL. Checked against arXiv:2410.08146.
- **Block:** `experimental.align.ProgressReward` (prover-policy advantages, trained like a PRM on rollout-difference targets).
- **Experiment:** `ProcessSupervision`.
- **Toy task:** `ChainTask`; the prover is a smaller toy LM or a best-of-4 version of the base.
- **Baselines:** outcome-reward GRPO; GRPO plus a Q-value PRM (`ProcessRewardModel`) as dense reward; `TestTimeCompute.stepBeamSearch` with the PRM.
- **Expected metric:** accuracy against RL samples, and beam-search accuracy at equal compute. Hypothesis: the Q-value PRM as dense reward does not beat outcome-only RL; PAV rewards improve sample efficiency by > 2x and lift beam search.
- **Source:** [arXiv:2410.08146](https://arxiv.org/abs/2410.08146)

### 9.2.11 PRIME: process rewards from an implicit PRM trained on outcomes (Cui et al., 2025)
PRIME gets dense token-level rewards without step labels. An implicit PRM, itself a language model, defines the reward of token t as β·log(π_φ(y_t|y<t)/π_ref(y_t|y<t)), and is trained only with a cross-entropy loss on outcome correctness of on-policy rollouts, so it is updated online and stays in step with the policy. The token rewards get a leave-one-out baseline and are summed as returns; the outcome reward gets its own leave-one-out term; the two are added and drive a PPO-clip update. Checked against arXiv:2502.01456.
- **Block:** `experimental.align.ImplicitProcessReward` (implicit PRM, online update, combined RLOO advantages).
- **Experiment:** `ProcessSupervision`.
- **Toy task:** `ChainTask` and `ArithmeticTask`.
- **Baselines:** RLOO with outcome rewards (`GrpoTrainer.Variant.RLOO`); RLOO plus a frozen `ProcessRewardModel`; 9.2.10.
- **Expected metric:** accuracy against RL steps, and reward hacking of the frozen PRM (PRM score rising while accuracy falls). Hypothesis: the frozen PRM is hacked late in training; PRIME's online implicit PRM is not, and beats outcome-only RLOO in sample efficiency.
- **Source:** [arXiv:2502.01456](https://arxiv.org/abs/2502.01456)

### 9.2.12 Step-DPO: preferences over the first wrong step (Lai et al., 2024)
DPO on whole solutions gives one coarse signal for a long chain. Step-DPO localises it: given a prefix of correct steps, the chosen continuation is a correct next step and the rejected one is the first erroneous step, both generated by the model and checked automatically (here, exactly). The DPO loss is applied to the single step conditioned on the shared prefix.
- **Block:** `experimental.align.StepPreferenceBuilder` (pair construction from verified rollouts; reuses `PreferenceLosses.dpo`).
- **Experiment:** `ProcessSupervision`.
- **Toy task:** `ChainTask` with 4–8 steps.
- **Baselines:** solution-level DPO on the same number of pairs; `RejectionSamplingFineTuning`; `StarTrainer`.
- **Expected metric:** accuracy against preference pairs, sliced by chain length. Hypothesis: Step-DPO beats solution-level DPO, more so on long chains; it matches STaR with fewer generated tokens.
- **Source:** [arXiv:2406.18629](https://arxiv.org/abs/2406.18629)

### 9.2.13 RLVR from one training example (Wang et al., 2025)
Wang et al. ran GRPO with a verifiable reward on a single training problem, chosen by the historical variance of its training accuracy, and found test accuracy matching training on 1.2k examples. Test accuracy kept rising after training accuracy saturated ("post-saturation generalisation"); the gain came mainly from the policy-gradient loss, with an entropy bonus adding to it. This item tests whether that happens with a toy LM, and what a single example teaches. Checked against arXiv:2504.20571.
- **Block:** `experimental.align.SingleExampleSelector` (historical-variance ranking); runs on `GrpoTrainer` unchanged.
- **Experiment:** `experiments.reasoning.ArithmeticRl` (new rows).
- **Toy task:** `ArithmeticTask`: train on 1, 2, 16 or all problems; test on the full held-out set.
- **Baselines:** GRPO on the full set; GRPO on 1 random example; `SpuriousRewards`' format-only and random-reward controls; entropy bonus on and off.
- **Expected metric:** held-out accuracy against steps, after the training example is solved. Hypothesis: with a toy LM, one high-variance example gives most of the format gain but not the full-set accuracy; the random-reward control separates format learning from real learning.
- **Source:** [arXiv:2504.20571](https://arxiv.org/abs/2504.20571)

### 9.2.14 Execution prediction from traces: scratchpads and Code-World-Model mid-training (Zaremba & Sutskever, 2014; Nye et al., 2021; Meta CWM, 2025)
Can a model predict what a program outputs? Direct prediction maps program and input to output. A scratchpad has the model write intermediate states first. CWM mid-trains on execution traces at scale: each step pairs the local-variable state before a line with the executed line and the resulting state, formatted as frames with special tokens for call, line and return events, and output prediction then runs as trace generation. Checked against arXiv:2510.02387.
- **Block:** `experimental.code.ExecutionTracer` (step-by-step traces of `ListPrograms` programs, a trace tokeniser) and `experimental.code.TraceLm` (trains on trace text).
- **Experiment:** `experiments.code.CodeExecution`.
- **Toy task:** `ListPrograms` programs of 2–8 primitives on lists of length 3–8; test on longer programs than training.
- **Baselines:** a toy LM predicting output directly; a toy LM with a final-state-only scratchpad; the `NeuralGpu` and a `TransformerSeq2Seq`.
- **Expected metric:** exact-output accuracy by program length, and as an auxiliary pretraining task for 9.2.15. Hypothesis: direct prediction collapses beyond the training length; full traces generalise to 2x the length at the cost of 5–10x more tokens.
- **Source:** [arXiv:1410.4615](https://arxiv.org/abs/1410.4615), [arXiv:2112.00114](https://arxiv.org/abs/2112.00114), [arXiv:2510.02387](https://arxiv.org/abs/2510.02387)

### 9.2.15 Self-repair with execution feedback (Chen et al., ICLR 2024; Olausson et al., ICLR 2024)
Self-debugging feeds a failed program back with its execution result (wrong outputs on the examples, or an error) and asks the model to repair it. Olausson et al. showed the fair comparison is at equal sample budget: repair helps only when feedback quality is good, and plain resampling (pass@k) is often as good. With a toy generator, the feedback is the diff between expected and actual outputs and the trace from 9.2.14.
- **Block:** `experimental.code.RepairLoop` (generate, run, feed back failure, repair; budget accounting).
- **Experiment:** `CodeExecution`.
- **Toy task:** `ListPrograms` input/output tasks.
- **Baselines:** `ProgramGenerator` (RobustFill) resampling at equal total samples; `ReflexionLoop` with a text-only reflection; `GuidedEnumerator`.
- **Expected metric:** solve rate against total samples (the pass@t curve). Hypothesis: repair with traces beats resampling only after a first-sample solve rate above about 0.3; with output-diff-only feedback it ties resampling.
- **Source:** [arXiv:2304.05128](https://arxiv.org/abs/2304.05128), [arXiv:2306.09896](https://arxiv.org/abs/2306.09896)

### 9.2.16 Execution-based reranking: AlphaCode clustering, CodeT dual agreement and LEVER (Li et al., Science 2022; Chen et al., ICLR 2023; Ni et al., ICML 2023)
Given many sampled programs, which to submit? AlphaCode filters by the given examples, then runs the survivors on extra inputs and clusters them by identical outputs, picking from the largest clusters. CodeT has the model also generate test cases and scores each program by dual agreement: the number of programs agreeing with it times the number of tests it passes. LEVER trains a small verifier on (program, its execution result) to predict correctness and reranks by verifier probability × generation probability. Wave 7 cut CodeT for needing a strong code LM; here the extra inputs come from `ListPrograms`' input sampler, so only the clustering and the verifier are learned.
- **Block:** `experimental.code.ExecutionReranker` (behaviour clustering, dual agreement, learned execution verifier).
- **Experiment:** `CodeExecution`.
- **Toy task:** `ListPrograms` tasks with 2 given examples (under-specified, so many programs fit them) and 5 hidden ones.
- **Baselines:** pick by generation likelihood; filter-by-examples then random; `GenerativeVerifier`-style reranking without execution.
- **Expected metric:** hidden-test accuracy of the top-1 submission at 16–256 samples. Hypothesis: clustering and LEVER beat likelihood ranking by > 0.15 at 64 samples; they tie when the examples fully specify the task.
- **Source:** [arXiv:2203.07814](https://arxiv.org/abs/2203.07814), [arXiv:2207.10397](https://arxiv.org/abs/2207.10397), [arXiv:2302.08468](https://arxiv.org/abs/2302.08468)

### 9.2.17 ArCHer with a tuned critic across interaction budgets (Zhou et al., ICML 2024) — follow-up
Wave 8's ArCHer was "fastest early, then falls below a random player (0.68)", ending at 0.51 at 6k interactions against 0.98 for trajectory GRPO. The paper's recipe has details the wave 8 run did not tune: a critic warm-up before any actor update, a much smaller actor learning rate than critic rate, a Polyak target with τ ≈ 0.01, a value baseline from several sampled utterances, an entropy bonus, and double-Q clipping against over-estimation. This item adds them, sweeps them, and reports the curve out to 24k interactions, logging the critic's TD error and Q over-estimation (against MC returns) to see whether the collapse is critic divergence.
- **Block:** options on `HierarchicalAgentTrainer` and `TurnLevelCritic` (warm-up, double Q, target rate, actor/critic rate ratio, entropy); defaults unchanged.
- **Experiment:** `experiments.agents.ToolAgentBench` (new rows; wave 8 rows kept).
- **Toy task:** the wave 8 `SimulatedUser` dialogue task.
- **Baselines:** wave 8 ArCHer as run; trajectory-level GRPO; `MultiturnRewardTrainer` (CollabLLM); a random player.
- **Expected metric:** success rate at 2k, 6k, 12k and 24k interactions; Q over-estimation. Hypothesis: the wave 8 collapse coincides with growing over-estimation; double Q plus warm-up removes it, and ArCHer then leads GRPO up to 6k interactions; whether it holds at 24k is the open question.
- **Source:** [arXiv:2402.19446](https://arxiv.org/abs/2402.19446)

---

## WP3. Data and evaluation: online mixing and curricula, data selection, synthetic data, continual pretraining, and evaluation methodology (17)

Target: two new packages, `experimental.eval` (evaluation statistics and audits) and `experimental.synth` (synthetic-data generators and their training loops); mixing, curricula and selection stay in the existing `experimental.data`. Measurements go in `experiments.data.DataCuration` (extended, 9.3.1–9.3.5), a new `experiments.data.SyntheticTraining` (9.3.6–9.3.8), a new `experiments.data.ContinualPretraining` (9.3.9) and the new theme `experiments.evaluation` (`CalibrationEstimators` 9.3.10, `JudgeAudit` 9.3.11–9.3.12, `ContaminationAudit` 9.3.13, `StatisticalReporting` 9.3.14–9.3.15, `BudgetRankings` 9.3.16–9.3.17). One new generator, `experimental.data.SkillTasks`: arithmetic and string skills with a known prerequisite graph (digit addition before carrying before multi-digit multiplication). Baselines are DoReMi, RegMix, the curriculum and calibration blocks, Min-K%++ and the IRT evaluator, over at least 3 seeds; evaluation items report coverage and error over 100+ replications.

### 9.3.1 Online data mixing as a multi-armed bandit (Albalak et al., 2023)
DoReMi needs a reference model and a proxy run before training. Online data mixing treats each domain as an arm of an EXP3 bandit and sets the mixture during the one real run: the reward of sampling a domain is the training loss on that domain's batch (high loss means more to learn), importance-weighted by the sampling probability, and weights are smoothed towards uniform for exploration.
- **Block:** `experimental.data.BanditMixer` (EXP3 with loss rewards; plugs into `DataMixture`).
- **Experiment:** `experiments.data.DataCuration` (new rows).
- **Toy task:** `DataCuration`'s four-domain toy corpus (one easy, one hard, one noise-heavy, one target-like).
- **Baselines:** uniform; temperature mixing; DoReMi (`DataMixture.updateWithExcessLoss`); RegMix (`RegressionMixer`).
- **Expected metric:** target-domain loss and average loss at equal tokens, counting DoReMi's proxy run. Hypothesis: ODM beats uniform and DoReMi with no extra run, but the noise-heavy domain (always high loss) attracts too much weight unless reward clipping is used.
- **Source:** [arXiv:2312.02406](https://arxiv.org/abs/2312.02406)

### 9.3.2 Aioli: data mixing with a mixing law fitted during the run (Chen et al., 2025)
Aioli's linear mixing optimisation framework writes every mixing method as minimising average loss under a method-specific "mixing law" linking per-group loss to proportions. Aioli fits the law's parameters A^t online from the history of observed losses and proportions in the current run (no extra training runs), then updates the proportions by exponentiated gradient descent, multiplying each group's weight by an exponential of the estimated gradient and renormalising. Checked against arXiv:2411.05735.
- **Block:** `experimental.data.AioliMixer`.
- **Experiment:** `DataCuration`.
- **Toy task:** 9.3.1's corpus; also a 2-domain case where the optimum is known by grid search.
- **Baselines:** 9.3.1 ODM; DoReMi; RegMix; the grid-searched static optimum.
- **Expected metric:** average and target loss at equal tokens, and distance from the grid optimum. Hypothesis: Aioli never loses to uniform (Chen et al.'s central claim), while DoReMi and ODM each lose in at least one setting.
- **Source:** [arXiv:2411.05735](https://arxiv.org/abs/2411.05735)

### 9.3.3 Skill-It: ordered skills for data selection (Chen et al., NeurIPS 2023)
Skills have prerequisites, and training on a prerequisite lowers the loss on later skills faster than training on them directly. Skill-It first estimates a skill graph A (A_ij > 0 if training on skill i helps skill j), from short probe runs, then samples skills online by mirror descent with weights proportional to exp(η Σ_j A_ij L_j), so a skill's sampling weight rises with the remaining loss of the skills it helps.
- **Block:** `experimental.data.SkillGraphSampler` (graph estimation by probe runs, online sampler).
- **Experiment:** `DataCuration` (skills section).
- **Toy task:** `SkillTasks` with a planted prerequisite chain and a decoy skill unrelated to the target.
- **Baselines:** uniform over skills; target-skill only; `Curriculum` with difficulty pacing; 9.3.4.
- **Expected metric:** target-skill loss at equal tokens; recovered graph against the planted one. Hypothesis: Skill-It recovers the chain and reaches the target loss with 30–50% fewer tokens than target-only training.
- **Source:** [arXiv:2307.14430](https://arxiv.org/abs/2307.14430)

### 9.3.4 Learning-progress curricula: automated curriculum learning and teacher–student curricula (Graves et al., ICML 2017; Matiisen et al., 2017)
A teacher picks which task to train on next by the learner's progress, not by fixed difficulty. Graves et al. use a non-stationary bandit (EXP3.S) whose reward is a progress signal such as the loss drop on the chosen task (prediction gain). Matiisen et al. use the absolute slope of each task's recent learning curve, so tasks being learned or being forgotten get attention and mastered or impossible tasks do not.
- **Block:** `experimental.data.ProgressCurriculum` (EXP3.S teacher; slope-based Thompson and ε-greedy teachers).
- **Experiment:** `DataCuration` (skills section).
- **Toy task:** `SkillTasks` plus `ArithmeticTask` digit lengths 1–8 with an unlearnable (random-label) length mixed in.
- **Baselines:** `Curriculum` (fixed pacing), `SelfPacedLearning`, uniform sampling, `LevelReplay` (prioritised level replay, adapted).
- **Expected metric:** steps to reach 0.9 accuracy at the longest learnable length; share of samples wasted on the unlearnable task. Hypothesis: progress teachers ignore the random task and beat fixed pacing; uniform wastes 1/8 of its budget.
- **Source:** [arXiv:1704.03003](https://arxiv.org/abs/1704.03003), [arXiv:1707.00183](https://arxiv.org/abs/1707.00183)

### 9.3.5 RHO-LOSS: train on points that are learnable, worth learning and not yet learnt (Mindermann et al., ICML 2022)
From each large candidate batch, select the points with the highest reducible holdout loss: the current training loss minus the loss of a small model trained on a holdout set ("irreducible loss"). High training loss alone selects noisy and unlearnable points; subtracting the irreducible loss filters them, while still skipping points already learnt.
- **Block:** `experimental.data.ReducibleLossSelector` (irreducible-loss model, per-batch selection).
- **Experiment:** `DataCuration` (selection section).
- **Toy task:** `ToyTasks` digits with 20% label noise and redundant duplicates; the counting LM with noise tokens (as for Rho-1).
- **Baselines:** uniform; highest-loss selection; EL2N pruning (`DataPruner`); Rho-1 `SelectiveTokenLoss` (token-level).
- **Expected metric:** steps to target accuracy, and the share of noisy points selected. Hypothesis: highest-loss selection picks mostly noise and is worse than uniform; RHO-LOSS reaches the target in 2–3x fewer steps.
- **Source:** [arXiv:2206.07137](https://arxiv.org/abs/2206.07137)

### 9.3.6 Dataset distillation: gradient matching and distribution matching (Wang et al., 2018; Zhao et al., ICLR 2021; Zhao & Bilen, WACV 2023)
Synthesise a tiny training set (one to ten images per class) that trains a fresh network nearly as well as the full set. Gradient matching (dataset condensation) optimises the synthetic images so that the gradients they induce in a network match those of real batches, over many random initialisations. Distribution matching is cheaper: it matches the mean embeddings of real and synthetic images under randomly initialised networks, with no inner training loop.
- **Block:** `experimental.synth.DatasetCondenser` (gradient and distribution matching, with differentiable augmentation).
- **Experiment:** `experiments.data.SyntheticTraining`.
- **Toy task:** `ToyTasks` digits and `ShapeImages`, 1 / 10 images per class.
- **Baselines:** random real images per class; k-means centroids (`KMeans`); iCaRL herding exemplars (`DistillationReplay`); core-set selection (`ActiveLearner`).
- **Expected metric:** accuracy of a fresh CNN trained only on the synthetic set, and across architectures. Hypothesis: gradient matching beats random real images by > 0.15 at 1 image per class; transfer to a different architecture drops it by about half.
- **Source:** [arXiv:1811.10959](https://arxiv.org/abs/1811.10959), [arXiv:2006.05929](https://arxiv.org/abs/2006.05929), [arXiv:2110.04181](https://arxiv.org/abs/2110.04181)

### 9.3.7 Synthetic tabular data: TabDDPM and CTGAN, judged by train-on-synthetic, test-on-real (Kotelnikov et al., ICML 2023; Xu et al., NeurIPS 2019)
For privacy or augmentation, generate a synthetic table and train downstream models on it. CTGAN uses mode-specific normalisation of continuous columns (a Gaussian mixture per column) and a conditional generator with training-by-sampling for imbalanced categories. TabDDPM uses Gaussian diffusion for numeric columns and multinomial diffusion for categorical ones, with an MLP denoiser. Utility is judged by training on synthetic and testing on real data; privacy by the distance of each synthetic row to its closest real record.
- **Block:** `experimental.synth.TabularDiffusion`, `experimental.synth.ConditionalTabularGan`, `experimental.synth.SyntheticDataAudit` (utility, closest-record distance).
- **Experiment:** `SyntheticTraining`.
- **Toy task:** `TabularStressSuite` tables with mixed types and an imbalanced class; `SyntheticRecsys.clicks`.
- **Baselines:** Gaussian copula (`GaussianMixture` per column + correlation); SMOTE-style interpolation; real data (the upper bound).
- **Expected metric:** `GradientBoostedTrees` accuracy trained on synthetic and tested on real; closest-record distance ratio. Hypothesis: TabDDPM is closest to real-data utility but copies more rows (low closest-record distance); CTGAN wins on the rare class only.
- **Source:** [arXiv:2209.15421](https://arxiv.org/abs/2209.15421), [arXiv:1907.00503](https://arxiv.org/abs/1907.00503)

### 9.3.8 Training classifiers on diffusion-generated images (Azizi et al., TMLR 2023)
A class-conditional diffusion model, fine-tuned on the training set and sampled with tuned guidance, produces synthetic images that improve a classifier when mixed with real data. Sample quality metrics (FID) are not the right target: lower guidance (more diversity) helps classification even when it hurts FID.
- **Block:** `experimental.synth.GenerativeAugmenter` (class-conditional sampling with a guidance sweep, real/synthetic mixing ratio).
- **Experiment:** `SyntheticTraining`.
- **Toy task:** `ShapeImages` and digits at 50, 200 and 1000 real images per class; generator: the existing conditional denoisers with classifier-free guidance as in `GuidanceComparison`.
- **Baselines:** real only; real with standard augmentation and mixup (`BatchMixing`); real + VAE samples (`ImageAutoencoders`).
- **Expected metric:** test accuracy against the real-data budget and against guidance weight. Hypothesis: synthetic data helps most at 50 images per class; the best guidance for accuracy is lower than the best for sample quality.
- **Source:** [arXiv:2304.08466](https://arxiv.org/abs/2304.08466)

### 9.3.9 Continual pretraining: re-warming, re-decaying, replay and the stability gap (Ibrahim et al., TMLR 2024; Guo et al., 2024)
Adding a new domain to a pretrained model raises two questions: the learning-rate schedule and forgetting. Ibrahim et al. show that re-warming and re-decaying the learning rate plus replaying a few percent of old data matches retraining on the union. Guo et al. describe a "stability gap": performance on the target first drops at the start of continual pretraining before recovering. Their fixes are several epochs on a properly sized high-quality subset instead of one pass over everything, and a mixture that mirrors the original pretraining mixture. Checked against arXiv:2406.14833.
- **Block:** `experimental.data.ContinualPretrainer` (re-warm/re-decay schedules on `TrainingSchedules`, replay ratio, subset selection by perplexity, gap tracking).
- **Experiment:** `experiments.data.ContinualPretraining`.
- **Toy task:** a toy LM pretrained on `SyntheticCorpus` domain A, continued on domain B (a new grammar from `SyntheticGrammar`), with frequent evaluation.
- **Baselines:** constant-LR continuation; no replay; retraining on A ∪ B (compute upper bound); `ElasticWeightConsolidation`; `SparseMemoryFinetuning`.
- **Expected metric:** final A and B loss, the depth and length of the stability-gap dip on B, and compute. Hypothesis: 5% replay with re-warming matches retraining within 0.02 nats; the multi-epoch high-quality subset shortens the dip.
- **Source:** [arXiv:2403.08763](https://arxiv.org/abs/2403.08763), [arXiv:2406.14833](https://arxiv.org/abs/2406.14833)

### 9.3.10 Calibration estimators: debiased and adaptive ECE, smooth ECE and proper scores (Kumar et al., NeurIPS 2019; Roelofs et al., AISTATS 2022; Błasiok & Nakkiran, ICLR 2024)
The usual binned ECE is a biased, bin-dependent estimator: with few samples a perfectly calibrated model scores well above zero, and rankings of methods flip with the bin count. Fixes: the debiased plug-in estimator, equal-mass (adaptive) bins with the bin count chosen by the monotone sweep, and smooth ECE, which kernel-smooths the residuals with a bandwidth chosen so the estimate is consistent. This item measures their bias and variance where the true calibration error is known.
- **Block:** `experimental.eval.CalibrationEstimators` (debiased, equal-mass, monotone sweep, smooth ECE, classwise ECE, reliability-diagram bootstrap).
- **Experiment:** `experiments.evaluation.CalibrationEstimators`.
- **Toy task:** synthetic predictors with a known calibration curve (so the true error is known), at n = 100–10,000; then the `UncertaintyBenchmark` models.
- **Baselines:** `Calibration.expectedCalibrationError` (15 equal-width bins); Brier and NLL (proper scores).
- **Expected metric:** bias and RMSE against true calibration error over 200 replications; how often method rankings in `UncertaintyBenchmark` flip. Hypothesis: equal-width ECE overstates error 2–5x at n = 200; smooth ECE and the sweep stay within 20%; at least one `UncertaintyBenchmark` ranking flips.
- **Source:** [arXiv:1909.10155](https://arxiv.org/abs/1909.10155), [arXiv:2012.08668](https://arxiv.org/abs/2012.08668), [arXiv:2309.12236](https://arxiv.org/abs/2309.12236)

### 9.3.11 LLM-as-judge biases and their mitigation: position, verbosity and self-preference (Zheng et al., NeurIPS 2023; Dubois et al., 2024; Panickssery et al., NeurIPS 2024)
Pairwise LM judges prefer the first (or second) position, longer answers, and outputs from their own model family. Mitigations: judging both orders and averaging (or counting inconsistent verdicts as ties), and length control by regression, as in length-controlled AlpacaEval, which fits a GLM of the judge's preference on the model identity plus the length difference and reports the win rate with the length term set to zero.
- **Block:** `experimental.eval.JudgeAudit` (bias probes: order swap, length padding with no content, self/other source; swap averaging; length-controlled GLM).
- **Experiment:** `experiments.evaluation.JudgeAudit`.
- **Toy task:** toy-LM judges built from `RewardModel` and `SelfRewardingTrainer`'s judge, trained on preference data with planted order and length biases; candidate outputs with known gold quality.
- **Baselines:** raw judge verdicts; `RubricReward` scores; gold labels.
- **Expected metric:** each bias's size (verdict change under the probe), and Spearman correlation of judge rankings with gold before and after mitigation. Hypothesis: swap averaging removes the position effect but not verbosity; length control removes most of the verbosity effect and raises the rank correlation.
- **Source:** [arXiv:2306.05685](https://arxiv.org/abs/2306.05685), [arXiv:2404.04475](https://arxiv.org/abs/2404.04475), [arXiv:2404.13076](https://arxiv.org/abs/2404.13076)

### 9.3.12 Statistically correcting judge-based scores: Rogan–Gladen and prediction-powered inference (Angelopoulos et al., Science 2023; Lee et al., 2025; Chen et al., ICML 2026)
A judge's pass rate is biased when its sensitivity and specificity are imperfect. With a small human-labelled calibration set, two corrections give unbiased estimates with valid intervals. Rogan–Gladen estimates the judge's sensitivity and specificity and inverts the misclassification: θ = (p̂ + spec − 1)/(sens + spec − 1). Prediction-powered inference adds the mean judge-versus-human residual on the labelled set to the judge mean on the large unlabelled set; PPI++ tunes the weight of the judge term for minimum variance. Chen et al. show that PPI has strictly smaller asymptotic variance than Rogan–Gladen. Lee et al. build intervals covering both data sources and allocate calibration labels adaptively. Checked against arXiv:2511.21140 and arXiv:2601.05420.
- **Block:** `experimental.eval.JudgeCorrection` (Rogan–Gladen with delta-method intervals, PPI, PPI++, the human-only estimate).
- **Experiment:** `JudgeAudit` (estimation section).
- **Toy task:** 9.3.11's judges scoring 10,000 outputs; 50–800 gold labels; also a shift where the calibration and test sets differ in true pass rate.
- **Baselines:** the raw judge mean; the mean of the gold labels alone (unbiased but wide).
- **Expected metric:** bias, interval width and coverage over 500 replications. Hypothesis: the raw judge mean misses its 95% interval almost always; PPI++ gives ≈ 0.95 coverage with intervals 30–60% narrower than human-only; Rogan–Gladen stays valid under the label-rate shift.
- **Source:** [arXiv:2301.09633](https://arxiv.org/abs/2301.09633), [arXiv:2311.01453](https://arxiv.org/abs/2311.01453), [arXiv:2511.21140](https://arxiv.org/abs/2511.21140), [arXiv:2601.05420](https://arxiv.org/abs/2601.05420)

### 9.3.13 Contamination audits: the Kernel Divergence Score and rephrased contamination (Choi et al., ICML 2025; Yang et al., 2023)
Two questions Min-K%++ does not answer. How much of a benchmark leaked, as a set-level fraction? The Kernel Divergence Score builds RBF kernel matrices of the model's embeddings of the benchmark samples before and after briefly fine-tuning on the benchmark; pairs involving unseen samples move more, so the (similarity-weighted) divergence falls as the seen fraction rises. And does contamination survive rephrasing? Yang et al. show that paraphrased test items in training inflate scores yet pass n-gram decontamination; they propose an LLM-based decontaminator, here an embedding-similarity detector. Checked against arXiv:2502.00678.
- **Block:** `experimental.eval.KernelDivergenceScore`, `experimental.eval.Decontaminator` (n-gram overlap and embedding-nearest-neighbour detectors).
- **Experiment:** `experiments.evaluation.ContaminationAudit`.
- **Toy task:** `BenchmarkMethodology`'s toy test set planted into training at 0–100% (verbatim) and as `SyntheticCorpus` paraphrases.
- **Baselines:** `ContaminationDetector` (Min-K%++), Min-K%, the loss threshold, averaged into a set-level estimate.
- **Expected metric:** Spearman correlation of each set-level score with the true contamination rate; score inflation and detection rate for paraphrased contamination. Hypothesis: KDS tracks the rate monotonically (ρ > 0.9) where averaged Min-K%++ is noisier; n-gram decontamination misses most paraphrases, which still inflate accuracy.
- **Source:** [arXiv:2502.00678](https://arxiv.org/abs/2502.00678), [arXiv:2311.04850](https://arxiv.org/abs/2311.04850)

### 9.3.14 Error bars for evaluations: clustered standard errors, paired tests and power (Miller, 2024; Dror et al., ACL 2018)
Most model comparisons report point accuracies. Miller's recipe: report standard errors from the central limit theorem; cluster them when questions come in groups (several questions per passage), which can widen them several-fold; compare two models with paired differences on the same questions, which narrows them; reduce variance by resampling answers or using token probabilities; and compute in advance how many questions a given difference needs.
- **Block:** `experimental.eval.EvalStatistics` (clustered SEs, paired-difference SEs, McNemar, bootstrap, power analysis).
- **Experiment:** `experiments.evaluation.StatisticalReporting`.
- **Toy task:** the reasoning and `BenchmarkMethodology` evaluations re-run 200 times with fresh seeds, with question clusters (several questions sharing a context).
- **Baselines:** naive i.i.d. standard errors; unpaired comparisons; the existing "mean of 3 seeds" reporting.
- **Expected metric:** coverage of each interval over reruns; false "wins" per 100 comparisons of identical models. Hypothesis: i.i.d. SEs under-cover on clustered questions (< 0.8); paired tests detect a 2-point difference with half the questions; some published wave rows fail a paired test.
- **Source:** [arXiv:2411.00640](https://arxiv.org/abs/2411.00640), [aclanthology.org/P18-1128](https://aclanthology.org/P18-1128/)

### 9.3.15 Adaptive overfitting to a reused test set: the Ladder and Thresholdout (Blum & Hardt, ICML 2015; Dwork et al., Science 2015)
Every hyperparameter chosen by looking at a test score leaks information, and a leaderboard can be climbed without improving true accuracy, for example by boosting random submissions. The Ladder reports a new score only if it beats the best so far by a margin, rounded. Thresholdout answers queries from the training set unless it differs from the holdout by more than a noisy threshold, in which case it returns the holdout value plus Laplace noise, which preserves the holdout's validity for exponentially many adaptive queries.
- **Block:** `experimental.eval.ReusableHoldout` (Ladder, Thresholdout) and an adaptive "attacker" analyst (boosting attack, greedy hyperparameter search).
- **Experiment:** `StatisticalReporting`.
- **Toy task:** random-label classification (true accuracy 0.5) and `HyperparameterSearchComparison`'s searches run against a holdout of 500 points.
- **Baselines:** the plain reused holdout; a fresh test set (truth).
- **Expected metric:** reported versus true accuracy after k adaptive queries. Hypothesis: the boosting attack reaches > 0.6 reported accuracy on random labels with a plain holdout and ≈ 0.5 with the Ladder or Thresholdout; honest hyperparameter search loses little under them.
- **Source:** [arXiv:1502.04585](https://arxiv.org/abs/1502.04585), [arXiv:1506.02629](https://arxiv.org/abs/1506.02629)

### 9.3.16 Do method rankings survive the training budget? Long-horizon forecasting and the architecture lineups (wave 8 R-run) — follow-up
Wave 8's transformers lost to DLinear at the test budget (Autoformer 0.212, FEDformer 0.098, TimesNet 0.076, transformer 0.155 against 0.036 at H = 24) "though they led in the report's longer run (R-wave8)", and the wave 4/5 attention and mixer lineups were noted as "smoke runs only". This item makes ranking-against-budget a measured quantity: each lineup is trained at budgets 1x, 2x, ..., 32x the test budget, with seeds, and the ranking at each budget is compared with the ranking at the largest budget.
- **Block:** `experimental.eval.BudgetRankingStudy` (budget ladder runner, Kendall's τ with bootstrap intervals, crossover detection).
- **Experiment:** `experiments.evaluation.BudgetRankings`.
- **Toy task:** `LongHorizonForecasting` (DLinear, PatchTST, Autoformer, FEDformer, iTransformer, TimesNet, TiDE, N-HiTS), plus the `MixerShootout` and `AttentionShootout` lineups.
- **Baselines:** the rankings published in `experiment-results.md` at the test budget.
- **Expected metric:** Kendall's τ between rankings at budget b and at 32x; budget at which each pair crosses. Hypothesis: τ at the test budget is below 0.4 for the forecasting lineup; DLinear's lead holds up to 4x and transformers overtake by 16x; several wave 4–5 "losses" are budget artefacts.
- **Source:** `experiment-results.md` (R-wave8, LongHorizonForecasting); [arXiv:2205.13504](https://arxiv.org/abs/2205.13504)

### 9.3.17 Predicting final rankings early: learning-curve extrapolation and LC-PFN (Domhan et al., IJCAI 2015; Adriaensen et al., NeurIPS 2023)
9.3.16 is expensive; can the final ranking be predicted from the first few percent of each curve? Domhan et al. fit an ensemble of parametric curve families (power law, exponential, log-power) by MCMC and stop runs that are unlikely to beat the best. LC-PFN is a prior-fitted network trained on curves sampled from that parametric prior; it does Bayesian curve extrapolation in one forward pass.
- **Block:** `experimental.eval.CurveExtrapolator` (parametric ensemble fitted with `HamiltonianMonteCarlo` or `Lbfgs`; LC-PFN reusing `PriorFittedNetwork`'s in-context transformer).
- **Experiment:** `BudgetRankings`.
- **Toy task:** the curves logged by 9.3.16, truncated at 5–25% of the 32x budget.
- **Baselines:** rank by last observed value; successive halving (`HyperparameterSearch`); power-law fit by least squares.
- **Expected metric:** Kendall's τ between the predicted and true 32x rankings at each truncation point, and compute saved. Hypothesis: last-value ranking does no better than 9.3.16's test-budget rankings; extrapolation at 10% recovers τ > 0.7 on the forecasting lineup.
- **Source:** [ijcai.org/Proceedings/15/Papers/487](https://www.ijcai.org/Proceedings/15/Papers/487.pdf), [arXiv:2310.20447](https://arxiv.org/abs/2310.20447)

---

## WP4. Interpretability tools, document/OCR and table understanding, and multimodal fusion (17)

Target: three new packages, `experimental.interp` (the existing interpretability blocks in `experimental.nn` and `experimental.llm`, such as `SparseAutoencoder`, `AttributionPatching`, `LogitLens` and `LinearProbe`, stay where they are and are imported), `experimental.document` and `experimental.multimodal`. Measurements go in `experiments.interp` (new `CircuitDiscovery` for 9.4.1–9.4.2, `ConceptErasure` for 9.4.3, `RepresentationComparison` for 9.4.4 and 9.4.7, `LatentKnowledge` for 9.4.5, `ExplainabilityBenchmark` extended for 9.4.6, `DevelopmentalInterp` for 9.4.8, `TrainingAttributionExperiment` in `experiments.data` extended for 9.4.9), the new theme `experiments.document` (`GlyphOcr` for 9.4.10, `DocumentParsing` for 9.4.11–9.4.12, `TableQa` for 9.4.13), `experiments.vision.DetectionShootout` (9.4.14) and `experiments.multimodal` (new `FusionShootout` for 9.4.15–9.4.17). Two new generators in `experimental.data`: `GlyphDocuments` (text lines and 2-D form/receipt pages rendered from `DigitGlyphs`' 3×5 font extended to letters, with word boxes, key/value labels, reading order and layout-region boxes) and `ToyTables` (small tables of numbers and names with questions that need lookup, comparison, count or sum). Interpretability items use toy models with planted, known mechanisms (as `SuperpositionRecovery`, `CircuitTracing` and `WeightSparseCircuits` do), so each tool is scored against ground truth.

### 9.4.1 Automated circuit discovery: ACDC and edge attribution patching (Conmy et al., NeurIPS 2023; Syed et al., 2023; Hanna et al., COLM 2024)
Find the subgraph of a network (edges between heads, MLPs and the residual stream) that implements a behaviour. ACDC iterates over edges from the output backwards, patching each edge with its value on a corrupted input and pruning it if the task metric (KL to the clean output) changes less than a threshold. Edge attribution patching replaces the per-edge forward passes with a first-order estimate, (corrupt − clean activation) × gradient, so all edges are scored in two forward passes and one backward; EAP-IG integrates the gradient along the path between corrupt and clean inputs, which is more faithful.
- **Block:** `experimental.interp.CircuitDiscovery` (edge-level computational graph over `HookedDecoder`, ACDC, EAP, EAP-IG, faithfulness and completeness scores).
- **Experiment:** `experiments.interp.CircuitDiscovery`.
- **Toy task:** a toy transformer trained on an indirect-object-style copy task and on `WeightSparseCircuits`' task, where the circuit is known from construction or from the weight-sparse model.
- **Baselines:** node-level `AttributionPatching` (AtP*) and `ActivationPatching`; `AttributionGraph` from the cross-layer transcoder.
- **Expected metric:** ROC-AUC of recovered edges against the known circuit, and runtime. Hypothesis: ACDC and EAP-IG reach AUC > 0.9; plain EAP misses edges whose gradient saturates; EAP is 50–100x faster than ACDC.
- **Source:** [arXiv:2304.14997](https://arxiv.org/abs/2304.14997), [arXiv:2310.10348](https://arxiv.org/abs/2310.10348), [arXiv:2403.17806](https://arxiv.org/abs/2403.17806)

### 9.4.2 Distributed alignment search (Geiger et al., CLeaR 2024)
Test whether a hidden representation encodes a high-level causal variable (such as "the carry bit"), even when it is spread over many neurons. DAS learns an orthogonal rotation of a layer's activations and a k-dimensional subspace by interchange-intervention training: run the model on a base input, replace the subspace's coordinates with those from a source input, and train the rotation so the model's output matches what the high-level causal model predicts after setting the variable to its source value. Interchange-intervention accuracy measures how well the variable is localised.
- **Block:** `experimental.interp.AlignmentSearch` (orthogonal rotation via a Cayley or Householder parametrisation, interchange-intervention training, IIA). It is distinct from `LowRankIntervention`, which trains a ReFT task intervention.
- **Experiment:** `CircuitDiscovery` (alignment section).
- **Toy task:** a toy MLP or transformer on two-digit addition (`ArithmeticTask`) with high-level variables "ones sum" and "carry".
- **Baselines:** a `LinearProbe` direction used for interchange; the top PCA components; single-neuron interchange.
- **Expected metric:** interchange-intervention accuracy against subspace size k. Hypothesis: DAS reaches IIA > 0.9 at k = 1–2 for the carry; probe directions are decodable but not causal (IIA < 0.6).
- **Source:** [arXiv:2303.02536](https://arxiv.org/abs/2303.02536)

### 9.4.3 Linear concept erasure: INLP and LEACE (Ravfogel et al., ACL 2020; Belrose et al., NeurIPS 2023)
Remove a concept (a protected attribute, a spurious feature) from representations so no linear classifier can recover it, while changing the representations as little as possible. INLP repeatedly trains a linear probe and projects onto its nullspace. LEACE gives the closed-form optimum: whiten, project out the span of the class-conditional mean differences (the cross-covariance with the concept), and un-whiten. This provably prevents every linear probe from beating a constant predictor, with minimal mean-squared change.
- **Block:** `experimental.interp.ConceptErasure` (INLP, LEACE, oracle-free LEACE fitted on a subset).
- **Experiment:** `experiments.interp.ConceptErasure`.
- **Toy task:** `CausalAndFairness`' spurious-feature images and `PersonaSteering`'s fine-tuned toy LM (trait concept).
- **Baselines:** mean-difference direction ablation (`PersonaVector`); `SteeringVector` subtraction; group-robust training (`GroupRobustTraining`).
- **Expected metric:** best linear-probe accuracy after erasure (lower is better), MLP-probe accuracy, task accuracy, and worst-group accuracy downstream. Hypothesis: LEACE drives linear probes to chance with the least task damage; INLP needs many iterations and damages more; nonlinear probes still recover part of the concept.
- **Source:** [arXiv:2004.07667](https://arxiv.org/abs/2004.07667), [arXiv:2306.03819](https://arxiv.org/abs/2306.03819)

### 9.4.4 Comparing representations: CKA, SVCCA and model stitching (Kornblith et al., ICML 2019; Raghu et al., NeurIPS 2017; Bansal et al., NeurIPS 2021)
Are two networks (different seeds, widths or training methods) computing the same thing layer by layer? Linear CKA is HSIC between centred Gram matrices, normalised, and invariant to orthogonal transforms and isotropic scaling. SVCCA reduces each layer with an SVD, then measures canonical correlations. Model stitching is the functional test: freeze the bottom of network A and the top of network B, train only a linear "stitching" layer between them, and see how much accuracy is lost.
- **Block:** `experimental.interp.RepresentationSimilarity` (linear and RBF CKA, SVCCA, PWCCA; reuses `ThinSvd`, `KernelTests`' HSIC) and `experimental.interp.ModelStitcher`.
- **Experiment:** `experiments.interp.RepresentationComparison`.
- **Toy task:** `VisionShootout` CNNs and ViTs on `ShapeImages`, 5 seeds and 3 widths; SSL encoders from `SelfSupervisedBenchmark`.
- **Baselines:** raw activation correlation; linear-probe accuracy per layer (`LinearProbe`).
- **Expected metric:** the CKA "layer correspondence" test (the share of layers whose most similar layer in another seed is the same layer), and stitching penalty against CKA. Hypothesis: CKA passes the correspondence test (> 0.9) where SVCCA does not; stitching penalties are small even where CKA is low, so similarity metrics and function disagree.
- **Source:** [arXiv:1905.00414](https://arxiv.org/abs/1905.00414), [arXiv:1706.05806](https://arxiv.org/abs/1706.05806), [arXiv:2106.07682](https://arxiv.org/abs/2106.07682)

### 9.4.5 Unsupervised latent knowledge: contrast-consistent search (Burns et al., ICLR 2023)
Find a "truth" direction without labels. For each yes/no question, build both completions (x⁺: "... Yes", x⁻: "... No") and take normalised hidden states. CCS trains a linear probe p so that p(x⁺) ≈ 1 − p(x⁻) (consistency) while avoiding p = ½ for both (confidence). Follow-up work showed CCS can latch onto any salient binary feature, so the test includes a distractor feature.
- **Block:** `experimental.interp.ContrastConsistentSearch` (CCS, plus the contrastive-PCA baseline from the paper).
- **Experiment:** `experiments.interp.LatentKnowledge`.
- **Toy task:** a toy LM trained on true and false arithmetic statements, with a salient distractor (a random prefix word that correlates with nothing), and a "sycophantic" split where a user prompt asserts the wrong answer.
- **Baselines:** a supervised `LinearProbe` (upper bound); zero-shot output probabilities; random direction.
- **Expected metric:** probe accuracy on held-out statements, on the sycophantic split, and with the distractor present. Hypothesis: CCS reaches > 0.85 of the supervised probe and beats the model's own outputs on the sycophantic split; with the distractor, CCS finds the distractor in some seeds.
- **Source:** [arXiv:2212.03827](https://arxiv.org/abs/2212.03827)

### 9.4.6 Attention rollout and attention flow (Abnar & Zuidema, ACL 2020)
Raw attention in one layer is a poor explanation, because information mixes through the residual stream across layers. Attention rollout adds the identity to each layer's head-averaged attention (for the residual), renormalises, and multiplies the matrices from the input upwards, giving token-to-input attributions. Attention flow treats the layers as a flow network with attention weights as capacities and computes the max flow from each input token to the output.
- **Block:** `experimental.interp.AttentionRollout` (rollout, flow via a max-flow solver, gradient-weighted rollout).
- **Experiment:** `experiments.interp.ExplainabilityBenchmark` (new rows).
- **Toy task:** `VisionTransformer` on `ShapeImages` and its spurious-marker variant; a toy-LM copy task with known source positions.
- **Baselines:** raw last-layer attention; `GradCam`, integrated gradients and saliency (`Attribution`).
- **Expected metric:** pointing game, deletion-curve AUC, and agreement with known source positions on the copy task. Hypothesis: rollout beats raw attention; flow is more faithful on the copy task but 100x slower; neither beats integrated gradients on deletion AUC.
- **Source:** [arXiv:2005.00928](https://arxiv.org/abs/2005.00928)

### 9.4.7 Linear mode connectivity and Git Re-Basin (Frankle et al., ICML 2020; Ainsworth et al., ICLR 2023)
Two networks trained from different seeds usually have a loss barrier on the straight line between them. Much of the barrier is due to permutation symmetry, since hidden units can be reordered without changing the function. Git Re-Basin permutes model B's units to align with A's, by weight matching (a linear assignment per layer, iterated by coordinate descent) or activation matching, after which linear interpolation often has almost no barrier and the two models can be averaged.
- **Block:** `experimental.interp.PermutationAlignment` (weight and activation matching via `MinibatchOptimalTransport`'s Hungarian solver; barrier measurement).
- **Experiment:** `RepresentationComparison` (connectivity section).
- **Toy task:** MLPs and small CNNs on digits and `ShapeImages`, widths 16–256, and the two-skill merge task of `MergingMethodsTest`.
- **Baselines:** naive weight averaging and `ModelMerging` (TIES, SLERP, Fisher); ensembling outputs (upper bound).
- **Expected metric:** loss barrier along the interpolation path, before and after alignment, against width; merged-model accuracy. Hypothesis: the barrier after alignment shrinks with width and nearly vanishes at 256; aligned averaging beats naive averaging by > 0.3 accuracy and approaches the ensemble.
- **Source:** [arXiv:1912.05671](https://arxiv.org/abs/1912.05671), [arXiv:2209.04836](https://arxiv.org/abs/2209.04836)

### 9.4.8 The local learning coefficient as a developmental probe (Lau et al., AISTATS 2025; Hoogland et al., TMLR 2025)
Singular learning theory's learning coefficient λ measures the effective dimensionality of a minimum: degenerate, simpler solutions have lower λ. The local learning coefficient at weights w* is estimated as λ̂ = nβ(E[L_n(w)] − L_n(w*)), with the expectation under a localised tempered posterior ∝ exp(−nβL_n(w) − γ/2‖w − w*‖²) sampled by SGLD. Tracked over training, plateaus and drops in λ̂ mark developmental stages. Checked against arXiv:2402.02364.
- **Block:** `experimental.interp.LocalLearningCoefficient` (localised SGLD sampler on `WeightPosterior`'s SGLD; per-layer refined LLC).
- **Experiment:** `experiments.interp.DevelopmentalInterp`.
- **Toy task:** `GrokkingExperiment` (modular addition, run long enough to grok), the toy induction-head transformer, and MLPs on the superposition toy.
- **Baselines:** the Hessian trace and top eigenvalue (`CurvatureProducts`), weight norm, parameter count.
- **Expected metric:** λ̂ over training and its alignment with known transitions (grokking step, induction-head formation); estimator stability across SGLD settings. Hypothesis: λ̂ drops at grokking while the Hessian trace does not move cleanly; λ̂ ranks models of equal size by generalisation better than weight norm.
- **Source:** [arXiv:2308.12108](https://arxiv.org/abs/2308.12108), [arXiv:2402.02364](https://arxiv.org/abs/2402.02364)

### 9.4.9 TRAK: data attribution with random projections (Park et al., ICML 2023)
Influence functions need inverse Hessians, and leave-one-out retraining is too expensive. TRAK linearises the model around its trained weights, randomly projects the per-example gradients to k dimensions, and computes attribution scores as φ(x)ᵀ(ΦᵀΦ)⁻¹Φᵀ·Q, the influence formula for the linearised logistic model, averaged over a few independently trained models. Attributions are judged by the linear datamodeling score: the rank correlation between predicted and actual outputs of models retrained on random subsets.
- **Block:** `experimental.interp.Trak` (projected gradients, ensemble averaging, LDS evaluation harness).
- **Experiment:** `experiments.data.TrainingAttributionExperiment` (new rows).
- **Toy task:** digits MLP and a small CNN on `ShapeImages`; 100 retrained subset models per seed for LDS.
- **Baselines:** influence functions and TracIn (`TrainingDataAttribution`), In-Run Shapley, KNN-Shapley (`DataValuation`), gradient similarity.
- **Expected metric:** LDS, mislabelled-example detection, and cost in model-equivalents. Hypothesis: TRAK with 5 models beats influence functions and TracIn on LDS at a fraction of the cost; on label-noise detection KNN-Shapley stays competitive.
- **Source:** [arXiv:2303.14186](https://arxiv.org/abs/2303.14186)

### 9.4.10 Line OCR with a convolutional-recurrent network and CTC (Shi et al., TPAMI 2017)
CRNN reads a whole text line without segmenting characters: a CNN turns the line image into a sequence of column features, a bidirectional RNN adds context, and CTC aligns the per-column predictions with the label string, decoded by best path or a small beam with a lexicon. This is the vision counterpart of `TalkingDigits`' audio CTC recogniser.
- **Block:** `experimental.document.LineRecognizer` (CNN + BiGRU + `CtcLoss`, best-path and lexicon decoding).
- **Experiment:** `experiments.document.GlyphOcr`.
- **Toy task:** `GlyphDocuments` lines of 4–20 glyphs with variable spacing, touching characters, noise and slight skew.
- **Baselines:** segment by blank columns, then classify each glyph with the `VisionShootout` CNN; a transformer encoder + CTC.
- **Expected metric:** character and word error rate; the touching-character split. Hypothesis: segmentation fails on touching glyphs (CER > 0.2) while CRNN stays below 0.05; the transformer needs more steps to match CRNN.
- **Source:** [arXiv:1507.05717](https://arxiv.org/abs/1507.05717)

### 9.4.11 OCR-free document parsing: a Donut-style image-to-sequence transformer (Kim et al., ECCV 2022)
Instead of running OCR and then parsing its tokens, Donut encodes the page image with a vision transformer and decodes the structured output (a JSON-like token sequence of fields and values) with an autoregressive text decoder, trained first to read all the text (pseudo-OCR pretraining) and then on the parsing task. No OCR engine, so no OCR error propagation, but the model must learn to read.
- **Block:** `experimental.document.ImageToMarkup` (`PatchEmbedding` encoder + `TransformerSeq2Seq` decoder with a reading pretraining phase).
- **Experiment:** `experiments.document.DocumentParsing`.
- **Toy task:** `GlyphDocuments` receipts and forms (20×80 to 48×96 pixels), with target field/value sequences.
- **Baselines:** the pipeline 9.4.10 OCR + rule-based field extraction; 9.4.10 OCR + 9.4.12 layout model.
- **Expected metric:** field-level F1 and tree-edit accuracy against training steps. Hypothesis: the pipeline wins at small budgets; Donut overtakes once its reading phase converges, and degrades less with heavy noise because there is no OCR error to propagate.
- **Source:** [arXiv:2111.15664](https://arxiv.org/abs/2111.15664)

### 9.4.12 Layout-aware token encoders: LayoutLM-style 2-D position embeddings (Xu et al., KDD 2020)
Forms carry meaning in their layout: a value sits to the right of or below its key. LayoutLM adds embeddings of each token's bounding box (x0, y0, x1, y1, plus width and height, each a learned table) to the usual token and 1-D position embeddings, and pretrains with masked visual-language modelling (predict masked tokens using their layout). Key-value extraction is then token tagging (BIO) on the OCR output.
- **Block:** `experimental.document.LayoutEncoder` (2-D box embeddings on a core `TransformerBlock` encoder; reuses `LinearChainCrf` for tagging).
- **Experiment:** `DocumentParsing`.
- **Toy task:** `GlyphDocuments` forms with OCR tokens and boxes (from ground truth and from 9.4.10's noisy OCR), labels key/value/other; shuffled reading order to test layout use.
- **Baselines:** the same encoder with 1-D positions only; BiGRU-CRF tagger (`SequenceLabelling`); rule-based nearest-key matching.
- **Expected metric:** entity-level F1, with and without shuffled reading order. Hypothesis: 2-D embeddings add > 0.1 F1, and almost all of it when reading order is shuffled; masked pretraining helps at 100 labelled forms.
- **Source:** [arXiv:1912.13318](https://arxiv.org/abs/1912.13318)

### 9.4.13 Table question answering by cell selection and aggregation: TAPAS (Herzig et al., ACL 2020)
TAPAS answers questions over tables without generating logical forms. The table is flattened into tokens with extra embeddings for row index, column index and numeric rank within the column; a BERT-style encoder then predicts, per cell, whether it is selected, plus an aggregation operator (none, count, sum, average) over the selected cells. Training uses only the final answer: the expected result over the soft selection is differentiable for sum and average.
- **Block:** `experimental.document.TableEncoder` (row/column/rank embeddings, cell-selection head, aggregation head, weakly supervised answer loss).
- **Experiment:** `experiments.document.TableQa`.
- **Toy task:** `ToyTables`: 3–8 rows × 3–5 columns, question templates for lookup, comparison, count and sum; test on larger tables than training.
- **Baselines:** a toy LM reading the linearised table and generating the answer; a semantic parser into a tiny SQL-like DSL found by `GuidedEnumerator` with execution against the answer.
- **Expected metric:** answer accuracy by question type and table size. Hypothesis: the seq2seq LM fails at sums and on larger tables; TAPAS generalises to larger tables; the semantic parser is best on count/sum when its search finds the program.
- **Source:** [arXiv:2004.02349](https://arxiv.org/abs/2004.02349)

### 9.4.14 DETR at a long budget with one-to-many hybrid matching, on scenes and document layouts (Carion et al., 2020; Jia et al., CVPR 2023) — follow-up
Wave 8's DETR-lite scored mAP 0.020 at 900 steps and DN-DETR 0.005, against 0.505 for CenterNet at 300 steps, with the note "far too slow to converge". DETR's slow convergence is known: the original trains for 500 epochs. This item runs the budget the method needs (up to 20k steps) as a learning curve, and adds the convergence fixes that do not need deformable attention. H-DETR adds a second group of queries trained with one-to-many matching (each ground-truth box is repeated k times), which gives far more positive gradients, while the one-to-one branch keeps NMS-free inference. DAB-DETR makes queries explicit 4-D anchor boxes refined layer by layer. It also adds a document-layout split, where DETR is the standard detector (Table Transformer).
- **Block:** options on `SetPredictionDetector` (hybrid matching branch, anchor-box queries, auxiliary per-layer losses); defaults unchanged.
- **Experiment:** `experiments.vision.DetectionShootout` (learning-curve rows; wave 8 rows kept).
- **Toy task:** `ShapeScenes`; `GlyphDocuments` pages with text-block, table and figure regions.
- **Baselines:** wave 8 DETR-lite and DN-DETR as run; CenterNet; RetinaNet-lite, all at equal steps on the same curve.
- **Expected metric:** mAP@0.5 against steps (300 to 20k). Hypothesis: plain DETR passes 0.3 mAP only after more than 5k steps; hybrid matching reaches CenterNet's 300-step mAP within 2k steps; on layouts (few large boxes) DETR is competitive earlier.
- **Source:** [arXiv:2005.12872](https://arxiv.org/abs/2005.12872), [arXiv:2207.13080](https://arxiv.org/abs/2207.13080), [arXiv:2201.12329](https://arxiv.org/abs/2201.12329)

### 9.4.15 Bridging a frozen LM and a vision encoder: gated cross-attention with a perceiver resampler, Q-Former, and a linear projector (Alayrac et al., NeurIPS 2022; Li et al., ICML 2023; Liu et al., NeurIPS 2023)
Three ways to connect pretrained unimodal models. Flamingo inserts new gated cross-attention layers into a frozen LM, each with a tanh gate initialised at zero so the LM starts unchanged, and compresses image features to a fixed number of tokens with a perceiver resampler. BLIP-2's Q-Former is a small transformer with learned queries that extract text-relevant visual tokens, pretrained with contrastive, matching and captioning losses. LLaVA simply projects patch features with a linear or MLP layer into the LM's input embeddings.
- **Block:** `experimental.multimodal.GatedCrossAttentionAdapter`, `experimental.multimodal.QueryTransformer`, `experimental.multimodal.PatchProjector`.
- **Experiment:** `experiments.multimodal.FusionShootout`.
- **Toy task:** `DigitGlyphs` images and `ShapeScenes` with captions and VQA questions ("how many squares?", "what is left of the circle?"); frozen toy LM and frozen `VisionTransformer`/SimCLR encoder.
- **Baselines:** Chameleon-style early fusion (`EarlyFusion`, trained from scratch); CLIP features with a classifier (`ZeroShotDigits`); a text-only LM (to measure language-prior answering).
- **Expected metric:** VQA accuracy, trainable parameters, and the text-only LM's held-out perplexity after training (does fusion damage the LM?). Hypothesis: gated cross-attention leaves the LM exactly intact and wins on counting; the linear projector is cheapest and ties on single-object questions; early fusion needs the most compute.
- **Source:** [arXiv:2204.14198](https://arxiv.org/abs/2204.14198), [arXiv:2301.12597](https://arxiv.org/abs/2301.12597), [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)

### 9.4.16 Transfusion: next-token prediction and diffusion in one transformer (Zhou et al., ICLR 2025)
One transformer is trained on mixed sequences of text tokens and continuous image patches: text positions get the language-modelling loss, image patches get a diffusion (noise-prediction) loss, and the two are summed with a weight. Attention is causal over the whole sequence but bidirectional among patches of the same image, with BOI/EOI tokens marking image spans; patches come from a VAE latent through a linear layer or U-Net down/up blocks. The same model captions images and generates them. Checked against arXiv:2408.11039.
- **Block:** `experimental.multimodal.TransfusionModel` (mixed attention mask, dual loss, image span sampling; reuses `ImageAutoencoders`' VAE and `DiffusionTransformer` pieces).
- **Experiment:** `FusionShootout` (generation section).
- **Toy task:** `DigitGlyphs` and `ToyImages` with captions: caption → image and image → caption.
- **Baselines:** Chameleon-style early fusion over VQ tokens (`EarlyFusion` with `VectorQuantizer`), separate text LM plus class-conditional diffusion model at equal total compute.
- **Expected metric:** caption accuracy, generated-image classifier accuracy and sample distance, at equal FLOPs. Hypothesis: Transfusion beats VQ-token early fusion on image quality at equal compute (no quantisation loss), and ties on captioning.
- **Source:** [arXiv:2408.11039](https://arxiv.org/abs/2408.11039)

### 9.4.17 Fusion bottleneck tokens and modality dropout for missing modalities (Nagrani et al., NeurIPS 2021; Neverova et al., 2016)
In multimodal bottleneck transformers (MBT), each modality has its own transformer layers, and cross-modal information flows only through a few shared bottleneck tokens that both streams attend to and update, from a fusion layer upwards. This forces each modality to condense what it shares, at a fraction of full cross-attention cost. Modality dropout zeroes whole modalities during training so the fused model degrades gracefully when one is missing at test time.
- **Block:** `experimental.multimodal.BottleneckFusion` (per-modality streams, shared bottleneck tokens, fusion-layer choice, modality dropout).
- **Experiment:** `FusionShootout` (audio-visual section).
- **Toy task:** audio-visual digits: `TalkingDigits` audio paired with `DigitGlyphs` images, each noisy enough that neither alone is perfect; test with one modality removed or corrupted.
- **Baselines:** late fusion (average of unimodal classifiers); early fusion (concatenated tokens, full attention); each unimodal model.
- **Expected metric:** accuracy with both modalities, with each missing, and FLOPs. Hypothesis: bottleneck fusion matches full attention at < 1/3 of its cost; without modality dropout, fused models fall below unimodal ones when a modality is missing.
- **Source:** [arXiv:2107.00135](https://arxiv.org/abs/2107.00135), [arXiv:1501.00102](https://arxiv.org/abs/1501.00102)

---

## WP5. World models and video generation, audio generation, and scientific-ML follow-ups (16)

Target: two new packages, `experimental.world` (video generators and world models; `DreamerAgent`, `RecurrentStateSpaceModel`, `LatentActionWorldModel`, `DiffusionWorldModel`, `JepaWorldModel` and `DiffusionForcing` stay where they are) and `experimental.audio` (trainable audio generators; the core `brainiac-audio` inference models are not touched). The scientific follow-ups extend the existing `experimental.sciml`. Measurements go in the new theme `experiments.world` (`VideoGeneration` for 9.5.1–9.5.4, `WorldModelShootout` for 9.5.5–9.5.8), `experiments.audio` (new `AudioGeneration` for 9.5.9–9.5.13), and `experiments.sciml.PdeShootout` / `OperatorLearning` (9.5.14–9.5.16). Video tasks use `ShapeScenes`' moving-shape videos (with flow and track ids) at 16×16–32×32 and 8–32 frames; world-model tasks use the `DreamingAgent` grid frames, `CartPole` and the `brainiac-rl` environments; audio tasks use `TalkingDigits`' formant synthesiser at 8 kHz (waveforms, log-mels and digit-string transcripts). Generation quality is measured with the `GenerativeScores` helpers plus task-specific checks: a frozen `VideoTracking` detector counts objects and checks their motion; the `SpokenDigitRecognition` recogniser transcribes generated speech.

### 9.5.1 Video diffusion with factorised space-time attention (Ho et al., NeurIPS 2022)
Video diffusion models extend an image denoiser to a 4-D tensor (frames × height × width × channels). To keep attention affordable, each block alternates spatial attention (within a frame) and temporal attention (across frames at the same position, with relative position bias), and the spatial layers can be shared with an image model so image and video data are trained jointly. Long videos are extended by reconstruction-guided conditional sampling on earlier frames.
- **Block:** `experimental.world.SpaceTimeDenoiser` (factorised attention blocks over `DiffusionTransformer` parts; joint image/video training).
- **Experiment:** `experiments.world.VideoGeneration`.
- **Toy task:** `ShapeScenes` videos (1–3 shapes, constant velocity and bounces), 8 frames at 16×16, unconditional and first-frame-conditioned.
- **Baselines:** a frame-wise image DDPM (no temporal attention); `VideoTracking`'s SimVP and ConvLSTM (deterministic predictors) for the conditional task; full 3-D attention at small size.
- **Expected metric:** per-frame quality (classifier real share), temporal consistency (identity switches and velocity error from the frozen tracker) and FVD-proxy (feature MMD on clips). Hypothesis: frame-wise DDPM has good frames and no consistency; factorised attention matches full 3-D attention at a quarter of the cost; deterministic predictors blur at frame 8.
- **Source:** [arXiv:2204.03458](https://arxiv.org/abs/2204.03458)

### 9.5.2 MAGVIT-style 3-D video tokenizer with masked token modelling (Yu et al., CVPR 2023)
Tokenise video in space-time: a 3-D convolutional VQ autoencoder compresses a clip by 4× in time and 4–8× in space into discrete tokens, with the first frame encoded so image-only conditioning works. A bidirectional transformer is then trained with MaskGIT's masked-token objective and generates a clip in a few parallel decoding steps. Different tasks (prediction, interpolation, inpainting) become different masks over a shared token canvas.
- **Block:** `experimental.world.SpaceTimeTokenizer` (3-D conv VQ, causal first frame) and a multi-task masking scheme on `MaskGitDecoder`.
- **Experiment:** `VideoGeneration`.
- **Toy task:** 9.5.1's videos; prediction, interpolation and inpainting tasks.
- **Baselines:** per-frame `PatchVqTokenizer` + MaskGIT (2-D tokens, many more of them); 9.5.1 video diffusion; an autoregressive token model.
- **Expected metric:** reconstruction PSNR at equal token count; generation quality per decoding step. Hypothesis: 3-D tokens give 4x fewer tokens at equal PSNR because static background is shared; MaskGIT in 8 steps trails 9.5.1 slightly at a tenth of the time.
- **Source:** [arXiv:2212.05199](https://arxiv.org/abs/2212.05199)

### 9.5.3 Self Forcing: training autoregressive video diffusion on its own rollouts (Huang et al., NeurIPS 2025)
Autoregressive video diffusion trained with teacher forcing (clean past frames as context) or diffusion forcing (noisy past frames) never sees its own generated context during training, so errors accumulate at inference (exposure bias). Self Forcing performs the autoregressive rollout during training, with a KV cache as at inference, using a few-step generator, and scores the whole generated video with a holistic distribution-matching loss (DMD, SiD or a GAN loss). Memory is bounded by back-propagating only through each frame's final denoising step, chosen at random, and stopping gradients into earlier frames' KV cache. Checked against arXiv:2506.08009.
- **Block:** `experimental.world.SelfForcingTrainer` (training-time rollout with a KV cache, gradient truncation, DMD loss reusing `DistributionMatchingDistiller`).
- **Experiment:** `VideoGeneration` (long-rollout section).
- **Toy task:** 9.5.1's videos generated frame by frame to 4x the training length.
- **Baselines:** teacher forcing; `DiffusionForcing` (noisy context); a bidirectional 9.5.1 model sampled in sliding windows.
- **Expected metric:** quality and object-count error against frame index, out to 32 frames. Hypothesis: teacher-forced models degrade after the training length; Self Forcing keeps the object count stable to 4x, with a one-step generator per frame.
- **Source:** [arXiv:2506.08009](https://arxiv.org/abs/2506.08009)

### 9.5.4 History guidance for video diffusion with the diffusion forcing transformer (Song et al., ICML 2025)
The diffusion forcing transformer is trained with an independent noise level per frame, so at sampling time any subset of past frames can serve as clean history, and history can be dropped by setting its noise to maximum. That makes classifier-free guidance on history possible: the score conditioned on history is extrapolated away from the score without it ("vanilla history guidance"). The paper's stronger variants guide across time (subsets of history) and frequency (low-frequency components of history), improving motion and long-rollout stability. Checked against arXiv:2502.06764.
- **Block:** `experimental.world.HistoryGuidance` (vanilla and temporal-subset guidance on top of `DiffusionForcing`).
- **Experiment:** `VideoGeneration` (long-rollout section).
- **Toy task:** 9.5.3's long-rollout task with a variable history length (1–6 frames).
- **Baselines:** `DiffusionForcing` without guidance; ordinary CFG on a class label; 9.5.3.
- **Expected metric:** velocity error and object-count drift against rollout length, and against guidance weight. Hypothesis: guidance weight 1.5–3 halves velocity error at 4x length; too much guidance freezes motion.
- **Source:** [arXiv:2502.06764](https://arxiv.org/abs/2502.06764)

### 9.5.5 IRIS: a discrete-token transformer world model (Micheli et al., ICLR 2023)
IRIS learns behaviours entirely in imagination, like Dreamer, but with a different world model. A discrete autoencoder turns each frame into a few VQ tokens, and a GPT-style transformer predicts the next frame's tokens, the reward and termination autoregressively from the history of frame tokens and actions. Imagined rollouts are decoded and the actor-critic trains on them.
- **Block:** `experimental.world.TokenWorldModel` (frame VQ via `VectorQuantizer`, transformer dynamics with reward and done heads).
- **Experiment:** `experiments.world.WorldModelShootout`.
- **Toy task:** `DreamingAgent` grid frames with keys and doors; `CartPole` from pixels rendered at 16×16.
- **Baselines:** `DreamerAgent` (MLP world model), `RecurrentStateSpaceModel` (PlaNet RSSM), `DiscreteWorldModel`, `DiffusionWorldModel` (DIAMOND), model-free PPO, all at equal real-environment steps.
- **Expected metric:** return at 20k–100k environment steps; 10-step imagined-rollout token accuracy. Hypothesis: IRIS beats the MLP world model on the key–door task, where long memory matters, and ties DIAMOND at lower imagination cost.
- **Source:** [arXiv:2209.00588](https://arxiv.org/abs/2209.00588)

### 9.5.6 DreamerV3's robustness tricks: symlog, two-hot returns, percentile return normalisation and KL balancing (Hafner et al., Nature 2025)
DreamerV3 runs across domains with one set of hyperparameters thanks to a handful of tricks the repo's simplified `DreamerAgent` lacks. Symlog squashes reward and observation targets, sign(x)·log(1+|x|). The critic predicts a two-hot distribution over exponentially spaced bins. Advantages are divided by max(1, P95 − P5 of returns), so small rewards are not amplified. KL balancing with free bits splits the world-model KL into dynamics and representation terms of different weights, floored at 1 nat. Unimix mixes 1% uniform into categorical distributions.
- **Block:** options on `DreamerAgent` and `MlpWorldModel` (symlog targets, two-hot critic reusing `HlGaussValue`'s binning, percentile normalisation, KL balancing with free bits, unimix); defaults unchanged.
- **Experiment:** `WorldModelShootout` (ablation section).
- **Toy task:** `DreamingAgent` tasks with rewards rescaled by 0.01, 1 and 100; `CartPole` and `Pendulum`.
- **Baselines:** the current `DreamerAgent` (tuned per reward scale, and fixed); each trick removed one at a time.
- **Expected metric:** return across reward scales with one fixed configuration. Hypothesis: the current agent needs per-scale tuning and fails at 0.01 and 100; with all tricks one configuration works at all three; percentile normalisation and symlog matter most.
- **Source:** [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)

### 9.5.7 DINO-WM: planning in a frozen pretrained feature space (Zhou et al., ICML 2025)
Instead of learning the encoder jointly with the dynamics (and needing reconstruction or collapse-preventing losses), DINO-WM uses frozen patch features from a pretrained self-supervised encoder (DINOv2 in the paper) and trains only a ViT predictor with a causal mask to predict the next frame's patch features from past features and actions, on offline trajectories. At test time it plans with CEM/MPC for actions whose predicted final features are closest to the goal image's features, zero-shot for new goals. Checked against arXiv:2411.04983.
- **Block:** `experimental.world.FeatureSpacePlanner` (frozen encoder, causal patch-feature predictor, CEM planning with `CemPlanner`).
- **Experiment:** `WorldModelShootout` (goal-reaching section).
- **Toy task:** the point-mass reacher of `WindyReach`/`ControlSystems` rendered to 16×16 images, plus a small block-pushing variant, with offline random-play data and unseen goal images.
- **Baselines:** `JepaWorldModel` (encoder trained jointly with EMA targets); a pixel-reconstruction world model with CEM; goal-conditioned behaviour cloning (`BehaviorCloning`); frozen encoders from `SelfSupervisedBenchmark` (SimCLR, MAE, I-JEPA) compared as the feature space.
- **Expected metric:** goal-reaching success and final distance. Hypothesis: frozen SSL features with a learned predictor beat the jointly trained JEPA on unseen goals; MAE features beat SimCLR features because they keep position.
- **Source:** [arXiv:2411.04983](https://arxiv.org/abs/2411.04983)

### 9.5.8 Object-centric dynamics: SlotFormer (Wu et al., ICLR 2023)
Learn video dynamics over object slots instead of pixels. A pretrained slot-attention encoder decomposes each frame into K slots; a transformer over the slots of the last T frames predicts the next frame's slots autoregressively, with temporal position encodings; a slot decoder renders them. Because dynamics act per object, rollouts keep object identity and extrapolate interactions such as bounces better than pixel predictors.
- **Block:** `experimental.world.SlotDynamics` (reuses `SlotAttention` with a spatial-broadcast decoder, transformer dynamics over slot sequences).
- **Experiment:** `WorldModelShootout` (video section).
- **Toy task:** `ShapeScenes` videos with 2–4 bouncing shapes; test with more objects than training.
- **Baselines:** SimVP and ConvLSTM (`VideoTracking`); 9.5.1 conditional video diffusion; a slot model with an MLP (per-slot, no interaction) dynamics.
- **Expected metric:** rollout PSNR and object-position error at 10 and 20 frames, and FG-ARI of predicted masks. Hypothesis: SlotFormer keeps object identity at 20 frames where SimVP blurs; the no-interaction ablation fails on collisions.
- **Source:** [arXiv:2210.05861](https://arxiv.org/abs/2210.05861)

### 9.5.9 Training a GAN vocoder: HiFi-GAN with multi-period and multi-scale discriminators, against Griffin–Lim (Kong et al., NeurIPS 2020; Griffin & Lim, 1984)
A vocoder turns a mel-spectrogram into a waveform. Griffin–Lim, the classical baseline, iteratively estimates a phase consistent with the magnitude spectrogram (STFT, replace magnitude, inverse STFT). HiFi-GAN's generator upsamples mels with transposed convolutions and multi-receptive-field residual blocks; it is trained adversarially against a multi-period discriminator (1-D signals reshaped to 2-D with periods 2, 3, 5, 7, 11, catching periodic structure) and a multi-scale discriminator, plus feature-matching and mel-reconstruction losses. Core `HifiGanBlock` only runs a pretrained generator; nothing in the repo trains one.
- **Block:** `experimental.audio.GanVocoder` (generator from core blocks, MPD and MSD discriminators, losses) and `experimental.audio.GriffinLim` (on core `Signal` STFT/iSTFT).
- **Experiment:** `experiments.audio.AudioGeneration`.
- **Toy task:** `TalkingDigits` utterances at 8 kHz, 40-band log-mels.
- **Baselines:** Griffin–Lim (32 and 100 iterations); `WaveNet` conditioned on mels; `ToyAudioCodec` decoder; `HarmonicNoiseSynth` (DDSP) resynthesis.
- **Expected metric:** multi-resolution STFT distance, recogniser word accuracy on resynthesised speech, and generation speed. Hypothesis: HiFi-GAN-lite beats Griffin–Lim on recogniser accuracy and STFT distance and runs 100x faster than WaveNet; dropping the MPD costs most of the gain.
- **Source:** [arXiv:2010.05646](https://arxiv.org/abs/2010.05646), [doi:10.1109/TASSP.1984.1164317](https://doi.org/10.1109/TASSP.1984.1164317)

### 9.5.10 Diffusion vocoders: DiffWave and WaveGrad (Kong et al., ICLR 2021; Chen et al., ICLR 2021)
Vocoding as diffusion in the waveform. DiffWave is a non-autoregressive WaveNet-like network with dilated convolutions, conditioned on the mel and the diffusion step, trained with the ε-prediction loss; a fast schedule needs about 6 steps. WaveGrad conditions on a continuous noise level instead of a discrete step, which lets the inference schedule be chosen after training (down to 6 iterations).
- **Block:** `experimental.audio.DiffusionVocoder` (DiffWave network, discrete and continuous noise conditioning; reuses `experimental.nn.diffusion` schedules and `DpmSolver`).
- **Experiment:** `AudioGeneration`.
- **Toy task:** 9.5.9's task.
- **Baselines:** 9.5.9 HiFi-GAN-lite and Griffin–Lim; `WaveNet`.
- **Expected metric:** quality metrics as in 9.5.9 against the number of sampling steps (6, 12, 50). Hypothesis: DiffWave at 50 steps matches HiFi-GAN-lite; at 6 steps it trails; it is easier to train (no adversarial instability).
- **Source:** [arXiv:2009.09761](https://arxiv.org/abs/2009.09761), [arXiv:2009.00713](https://arxiv.org/abs/2009.00713)

### 9.5.11 Text-to-speech alignment for flow-matching TTS: monotonic alignment search (Matcha-TTS) against filler-token padding (F5-TTS) (Mehta et al., ICASSP 2024; Kim et al., NeurIPS 2020; Chen et al., 2024)
A TTS model must learn which text symbol each mel frame belongs to. Matcha-TTS encodes text, predicts a per-symbol mean mel and duration, and finds the alignment during training with monotonic alignment search (Glow-TTS's Viterbi-like dynamic programme for the most likely monotonic path); a conditional flow-matching decoder then generates mels from the aligned means in a few ODE steps. F5-TTS removes durations and alignment: it pads the character sequence with filler tokens to the mel length and trains a flow-matching DiT to infill masked mel frames, so alignment emerges implicitly.
- **Block:** `experimental.audio.MonotonicAlignmentSearch`, `experimental.audio.FlowMatchingTts` (Matcha-style encoder, duration predictor and flow decoder; F5-style padded infilling variant, on `RectifiedFlow` pieces).
- **Experiment:** `AudioGeneration` (TTS section).
- **Toy task:** `TalkingDigits` digit strings, 1–8 digits, with variable speaking rate; vocoded with 9.5.9.
- **Baselines:** an autoregressive mel model (Tacotron-lite: attention seq2seq from `Seq2SeqAttention`); a model trained with the synthesiser's ground-truth durations (upper bound); the formant synthesiser itself.
- **Expected metric:** recogniser digit-error rate on generated speech, alignment error against true durations, and ODE steps. Hypothesis: MAS converges in far fewer steps and generalises to longer strings than F5-style padding at toy scale; the autoregressive model skips or repeats digits on long strings.
- **Source:** [arXiv:2309.03199](https://arxiv.org/abs/2309.03199), [arXiv:2005.11129](https://arxiv.org/abs/2005.11129), [arXiv:2410.06885](https://arxiv.org/abs/2410.06885)

### 9.5.12 Latent audio diffusion: AudioLDM-style generation in a learned audio latent (Liu et al., ICML 2023; Evans et al., 2024)
Generate audio by diffusion in the compact latent of an audio autoencoder (a mel VAE in AudioLDM, a waveform VAE in Stable Audio) rather than over tokens, conditioned on a label or a text embedding through cross-attention, then decode with the autoencoder and a vocoder. Continuous latents avoid the quantisation loss of codec tokens.
- **Block:** `experimental.audio.LatentAudioDiffusion` (mel VAE, conditioned latent denoiser reusing `VectorConditionedDenoiser` / `DiffusionTransformer`).
- **Experiment:** `AudioGeneration` (generation section).
- **Toy task:** class-conditional generation of spoken digits and of `ToyAudioCodec`'s synthetic sound classes.
- **Baselines:** `CodecLanguageModel` (MusicGen-style tokens over `ResidualVectorQuantizer`); diffusion directly on mels; on waveforms.
- **Expected metric:** class accuracy of generated audio by a frozen classifier, a feature-MMD (FAD proxy), and generation time. Hypothesis: latent diffusion matches the codec LM's quality with 4x fewer network evaluations; mel-space diffusion is slower at equal quality.
- **Source:** [arXiv:2301.12503](https://arxiv.org/abs/2301.12503), [arXiv:2407.14358](https://arxiv.org/abs/2407.14358)

### 9.5.13 Finite scalar quantization for audio and video tokenizers (Mentzer et al., ICLR 2024)
VQ-VAEs need codebook losses, commitment terms and tricks against codebook collapse. FSQ drops the codebook: project the latent to a few dimensions (such as 4–6), bound each with tanh and round it to a small fixed number of levels (such as 8, 5, 5, 5), with the straight-through estimator. The implicit codebook is the product grid, every code is used, and there are no auxiliary losses.
- **Block:** `experimental.audio.FiniteScalarQuantizer` (shared with 9.5.2's video tokenizer).
- **Experiment:** `AudioGeneration` (codec section) and `VideoGeneration` (tokenizer row).
- **Toy task:** `ToyAudioCodec`'s signals and 9.5.2's videos, at matched code-space sizes (2^10 to 2^14).
- **Baselines:** `VectorQuantizer` (with EMA updates and restarts), `ResidualVectorQuantizer`, `MultiScaleVqTokenizer`.
- **Expected metric:** reconstruction error and codebook utilisation (perplexity) against code-space size; quality of 9.5.2's MaskGIT on top. Hypothesis: FSQ uses about 100% of codes at every size where VQ falls below 50% at 2^14, and matches VQ's reconstruction; generation quality on top ties.
- **Source:** [arXiv:2309.15505](https://arxiv.org/abs/2309.15505)

### 9.5.14 Hard PDEs for PINNs: curriculum regularisation, time-marching and exact periodic boundaries (Krishnapriyan et al., NeurIPS 2021; Dong & Ni, 2021) — follow-up
Wave 8's plain PINN lost everywhere to a coarse solver (Burgers 0.496, Allen–Cahn 0.993, advection 0.893 against 0.23, 0.11 and 0.0034), and at advection β = 10 every recipe stayed near 1.0. Krishnapriyan et al. traced these failures to an ill-conditioned loss, not to the network's capacity. Two fixes: curriculum regularisation (start at β = 1 and raise β gradually, warm-starting each stage) and sequence-to-sequence time-marching (train on short time windows one after another, each using the previous window's end as its initial condition). A third fix removes the boundary loss: periodic boundary conditions built in exactly with Fourier-feature inputs (sin/cos of 2πx/L).
- **Block:** options in `experimental.sciml.PinnTraining` (β/coefficient curriculum, time-marching windows, exact periodic embedding); defaults unchanged.
- **Experiment:** `experiments.sciml.PdeShootout` (new rows; wave 8 rows kept).
- **Toy task:** advection at β = 10, 20, 30 and 40; viscous Burgers; Allen–Cahn (`PdeProblems`).
- **Baselines:** wave 8 plain PINN and the four `PinnTraining` fixes (SA weights, NTK balancing, causal training, RAR-D); PirateNet; the coarse finite-difference solver.
- **Expected metric:** relative L2 error, wall time, against β. Hypothesis: time-marching plus the periodic embedding brings advection at β = 30 below 0.05 (from about 1.0); Allen–Cahn still needs causal weighting as well; the classical solver keeps the lowest error per second.
- **Source:** [arXiv:2109.01050](https://arxiv.org/abs/2109.01050), [arXiv:2007.07442](https://arxiv.org/abs/2007.07442)

### 9.5.15 Second-order optimisers for PINNs: energy natural gradient and Newton–CG after L-BFGS (Müller & Zeinhofer, ICML 2023; Rathore et al., ICML 2024) — follow-up
The PINN loss is badly conditioned because of the differential operator, so first-order methods stall. Energy natural gradient descent preconditions with the Gram matrix of the PDE operator applied to the network's tangent space (a Gauss–Newton matrix in function space) and takes a line-searched step; on small networks it reaches errors several orders of magnitude below Adam. Rathore et al. analyse the conditioning and recommend Adam, then L-BFGS, then a damped Newton–CG with a Nyström preconditioner (NNCG) to finish.
- **Block:** `experimental.sciml.PinnNaturalGradient` (ENGD with the Gram matrix from the residual Jacobian via `CurvatureProducts`, damping and line search) and `experimental.sciml.NystromNewtonCg`.
- **Experiment:** `PdeShootout` (optimiser rows).
- **Toy task:** Poisson 2-D, heat, and the 9.5.14 PDEs at sizes where a dense Gram matrix fits (≤ 2,000 parameters).
- **Baselines:** Adam then L-BFGS (`PinnTraining` default); `LevenbergMarquardt`; `HessianFree`; K-FAC (`Kfac`).
- **Expected metric:** relative L2 error against wall time. Hypothesis: ENGD reaches 1e-5–1e-6 on Poisson where Adam + L-BFGS stops near 1e-3; on advection β = 30 better optimisation alone does not fix the failure without 9.5.14's curriculum.
- **Source:** [arXiv:2302.13163](https://arxiv.org/abs/2302.13163), [arXiv:2402.01868](https://arxiv.org/abs/2402.01868)

### 9.5.16 Transolver and LANO against FNO across model size and data (Wu et al., ICML 2024) — follow-up
Wave 8's physics-attention operators lost clearly on Burgers (Transolver 0.576, LANO 0.555 against 0.167 for FNO) "at this size". Transolver's advantage in the paper is on irregular meshes and at scale; on a regular 1-D grid FNO's spectral bias is a strong prior. This item sweeps width (16–128), training samples (256–4,096) and grid regularity (uniform, and randomly subsampled point clouds, where FNO needs interpolation) to find where, if anywhere, the ranking flips.
- **Block:** no new block; `OperatorLearning` gains a sweep harness and an irregular-grid variant of `PdeProblems`' Burgers data.
- **Experiment:** `experiments.sciml.OperatorLearning` (sweep rows; wave 8 rows kept).
- **Toy task:** Burgers and Darcy-1D on uniform and irregular grids.
- **Baselines:** `FourierNeuralOperator1d` (with interpolation onto a uniform grid for irregular inputs); DeepONet; the wave 8 Transolver and LANO.
- **Expected metric:** relative L2 error across the sweep. Hypothesis: FNO keeps its lead on uniform grids at every size; on irregular grids with 50% of the points dropped Transolver overtakes FNO plus interpolation.
- **Source:** [arXiv:2402.02366](https://arxiv.org/abs/2402.02366), [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)

---

## WP6. Hardware-aware and parallel training, streaming and online learning, and edge/TinyML (17)

Target: three new packages, `experimental.parallel` (parallelism strategies simulated with Java threads as devices and a counting collective layer; `DataParallel`, `TcpCollective`, `Offloaded`, `Recompute` and `RingAttention` stay where they are), `experimental.stream` (online learners, drift detectors and prequential evaluation) and `experimental.edge` (integer inference, memory-aware scheduling and microcontroller-budget training). Measurements go in the new themes `experiments.parallel` (`ParallelismShootout` for 9.6.1–9.6.4 and 9.6.6, `MemorySchedules` for 9.6.5, `AsyncTraining` for 9.6.7), `experiments.streaming` (`StreamShootout` for 9.6.8–9.6.12) and `experiments.edge` (`EdgeDeployment` for 9.6.13–9.6.17). Every parallel item must reproduce single-device gradients to float tolerance and reports three numbers from the simulator: per-rank peak memory, bytes communicated, and the idle ("bubble") fraction in a cost model calibrated with measured per-op CPU times. One new generator, `experimental.data.DriftingStreams`: SEA, rotating hyperplane and RBF-drift streams with abrupt, gradual and recurring drift at known positions, plus a drifting version of `SyntheticRecsys.clicks`. Edge items report accuracy against peak RAM, model bytes and multiply-adds counted from the layer graph, with a 256 kB / 1 MB budget as in the TinyML papers.

### 9.6.1 Pipeline-parallel schedules: GPipe, 1F1B and interleaved 1F1B (Huang et al., NeurIPS 2019; Narayanan et al., SOSP 2019 and SC 2021)
Split a model into p stages on p devices and a batch into m micro-batches. GPipe runs all forwards, then all backwards: the bubble fraction is (p−1)/(m+p−1) and every stage holds m micro-batches of activations. 1F1B (PipeDream-Flush) alternates one forward and one backward after a warm-up, with the same bubble but at most p micro-batches of activations in flight. Interleaved 1F1B gives each device v non-adjacent chunks of layers, which shrinks the bubble by v at the cost of v times more point-to-point messages.
- **Block:** `experimental.parallel.PipelineSchedule` (schedule generator and validator), `experimental.parallel.PipelineRunner` (threads as stages, blocking queues as links, exact gradient check) and `experimental.parallel.ParallelCostModel`.
- **Experiment:** `experiments.parallel.ParallelismShootout`.
- **Toy task:** an 8–16-block MLP or toy transformer (`TunableTransformerBlock`) on `CountingTasks`, p = 2–8, m = 1–32.
- **Baselines:** single-device training; `DataParallel` at equal devices.
- **Expected metric:** bubble fraction (simulated and measured), peak activation memory per stage, and messages per step; gradient match with single device. Hypothesis: measured bubbles follow (p−1)/(m+p−1); 1F1B cuts peak activation memory by m/p against GPipe; interleaving with v = 2 halves the bubble.
- **Source:** [arXiv:1811.06965](https://arxiv.org/abs/1811.06965), [arXiv:1806.03377](https://arxiv.org/abs/1806.03377), [arXiv:2104.04473](https://arxiv.org/abs/2104.04473)

### 9.6.2 Zero-bubble pipelines and DualPipe: splitting the backward pass (Qi et al., ICLR 2024; DeepSeek-AI, 2024)
The backward pass of a layer has two parts: B, the gradient with respect to its input, which the previous stage waits for, and W, the gradient with respect to its weights, which nobody waits for. Zero-bubble schedules (ZB-H1, ZB-H2) run B as early as possible and use deferred W computations to fill what would be bubbles, reaching near-zero bubble at the memory of 1F1B (H1) or a little more (H2), with an optimizer-step synchronisation bypass. DualPipe (DeepSeek-V3) feeds micro-batches from both ends of the pipeline at once and overlaps the computation of one direction with the communication of the other.
- **Block:** `experimental.parallel.SplitBackward` (separate input-gradient and weight-gradient passes for `Dense` and attention blocks via the autograd tape) and ZB-H1/ZB-H2/DualPipe generators in `PipelineSchedule`.
- **Experiment:** `ParallelismShootout`.
- **Toy task:** 9.6.1's models and settings.
- **Baselines:** 9.6.1 1F1B and interleaved 1F1B.
- **Expected metric:** bubble fraction and peak memory at p = 4 and 8 for m = 8 and 32. Hypothesis: ZB-H1 cuts the 1F1B bubble by about two-thirds at the same memory; ZB-H2 reaches about zero with roughly 2x activation memory; DualPipe's gain needs the comm/compute overlap the cost model includes.
- **Source:** [arXiv:2401.10241](https://arxiv.org/abs/2401.10241), [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

### 9.6.3 Tensor and sequence parallelism: Megatron column/row splits (Shoeybi et al., 2019; Korthikanti et al., MLSys 2023)
Megatron splits each transformer MLP into a column-parallel first matrix (each rank holds a slice of the hidden units, no communication) and a row-parallel second matrix (each rank produces a partial sum), so one all-reduce per MLP in the forward pass and one in the backward pass suffice; attention is split by heads in the same way. Sequence parallelism additionally shards the layer-norm and dropout regions along the sequence, turning each all-reduce into a reduce-scatter plus an all-gather and dividing those activations by t.
- **Block:** `experimental.parallel.TensorParallelBlock` (column/row-parallel `Dense`, head-parallel attention, sequence-parallel norm regions) and `experimental.parallel.Collectives` (all-reduce, reduce-scatter, all-gather and all-to-all with byte counting).
- **Experiment:** `ParallelismShootout`.
- **Toy task:** the toy transformer on `CountingTasks`, t = 1, 2, 4, 8.
- **Baselines:** single device; `DataParallel`; 9.6.1 pipeline at equal devices.
- **Expected metric:** bytes communicated per step, per-rank parameter and activation memory, and gradient match. Hypothesis: tensor parallelism divides parameter memory by t at about 4 all-reduces per layer per step; sequence parallelism cuts the remaining replicated activation memory by t at the same communication volume.
- **Source:** [arXiv:1909.08053](https://arxiv.org/abs/1909.08053), [arXiv:2205.05198](https://arxiv.org/abs/2205.05198)

### 9.6.4 ZeRO stages 1–3 and fully sharded data parallelism (Rajbhandari et al., SC 2020; Zhao et al., VLDB 2023)
Data parallelism replicates optimizer states, gradients and parameters on every rank. ZeRO-1 shards optimizer states (each rank updates 1/N of the parameters, then all-gathers them); ZeRO-2 also shards gradients (reduce-scatter instead of all-reduce); ZeRO-3/FSDP also shards the parameters themselves, all-gathering each layer just in time for its forward and backward and freeing it afterwards. Memory falls towards 1/N, while communication rises from 2Ψ (stages 1–2, the same as all-reduce) to 3Ψ (stage 3).
- **Block:** `experimental.parallel.ShardedDataParallel` (stages 1, 2, 3 with prefetch; wraps any core `Optimizer`).
- **Experiment:** `ParallelismShootout`.
- **Toy task:** a toy LM with AdamW on `SyntheticCorpus`, N = 2–8 ranks.
- **Baselines:** `DataParallel`; `Offloaded` (ZeRO-Offload); `LowBitOptimizerStates` (4-bit states) as an orthogonal memory saving.
- **Expected metric:** per-rank memory by category, bytes communicated, simulated step time; exact loss match with `DataParallel`. Hypothesis: memory per rank follows the paper's 4Ψ + 12Ψ/N style accounting; ZeRO-3 adds 50% communication; combining ZeRO-1 with 4-bit states gives the lowest memory at no extra communication.
- **Source:** [arXiv:1910.02054](https://arxiv.org/abs/1910.02054), [arXiv:2304.11277](https://arxiv.org/abs/2304.11277)

### 9.6.5 Selective activation recomputation and optimal checkpoint schedules (Chen et al., 2016; Griewank & Walther, 2000; Korthikanti et al., MLSys 2023; Jain et al., MLSys 2020)
`Recompute` checkpoints whole blocks, about a third more compute for one stored tensor per block. Chen et al.'s √n schedule stores every √n-th layer for O(√n) memory at one extra forward pass. Selective recomputation keeps the cheap-to-store, expensive-to-compute tensors (matmul outputs) and recomputes only the attention-score region, which is large in memory but cheap in FLOPs. Revolve (binomial checkpointing) and Checkmate (an integer linear programme over the graph) give optimal schedules under a memory budget.
- **Block:** `experimental.parallel.CheckpointPlanner` (√n, selective by op type, Revolve dynamic programme, a small ILP or DP over a linear chain; drives `Recompute`).
- **Experiment:** `experiments.parallel.MemorySchedules`.
- **Toy task:** toy transformers at depth 8–32 and sequence length 64–512 on `CountingTasks`.
- **Baselines:** no checkpointing; `Recompute` on every block; `CompressedActivations` (ActNN 2-bit) and `ReversibleBlock` from `MemoryEfficientTraining`.
- **Expected metric:** peak activation memory against recompute overhead (a Pareto curve), measured on the CPU backend. Hypothesis: selective recomputation keeps about 70% of the memory saving of full recomputation at under 5% extra compute; the optimal schedule dominates √n at every budget.
- **Source:** [arXiv:1604.06174](https://arxiv.org/abs/1604.06174), [arXiv:2205.05198](https://arxiv.org/abs/2205.05198), [arXiv:1910.02653](https://arxiv.org/abs/1910.02653)

### 9.6.6 Expert parallelism with all-to-all dispatch, capacity limits and token dropping (Lepikhin et al., ICLR 2021; Fedus et al., JMLR 2022)
With experts placed on different ranks, each MoE layer needs two all-to-alls: dispatch tokens to their experts' ranks, and combine the results. Static buffers need a capacity per expert, capacity factor × tokens/experts; tokens beyond it are dropped (they pass through the residual only). The capacity factor trades dropped tokens against padding and communication; load imbalance makes the slowest rank set the step time.
- **Block:** `experimental.parallel.ExpertParallelMoe` (expert placement, capacity buffers, drop/no-drop modes, all-to-all via 9.6.3's collectives; wraps `FineGrainedMoe` routing, including its auxiliary-loss-free balancing).
- **Experiment:** `ParallelismShootout` (MoE section).
- **Toy task:** a toy MoE LM (8–16 experts on 2–8 ranks) on `SyntheticCorpus`.
- **Baselines:** single-device `FineGrainedMoe`; `ExpertChoiceMoe` (perfect balance by construction); balance by auxiliary loss against auxiliary-loss-free bias.
- **Expected metric:** dropped-token rate, all-to-all bytes, slowest-rank load (simulated step time), and validation loss, against capacity factor 1.0–2.0. Hypothesis: at capacity 1.0 the auxiliary-loss model drops about 10% of tokens and loses loss; loss-free balancing drops fewer at the same capacity; expert choice has zero drops but leaves tokens unprocessed.
- **Source:** [arXiv:2006.16668](https://arxiv.org/abs/2006.16668), [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)

### 9.6.7 Asynchronous SGD with stale gradients: Hogwild! and delay-compensated ASGD (Niu et al., NeurIPS 2011; Zheng et al., ICML 2017)
Asynchronous workers compute gradients on parameters that are several updates old. Hogwild! lets threads update shared parameters without locks, which is nearly race-free when updates are sparse. DC-ASGD compensates for a gradient computed at w_old and applied at w_now with a cheap diagonal Hessian approximation, g + λ·g ⊙ g ⊙ (w_now − w_old), since the outer product of gradients approximates the Hessian.
- **Block:** `experimental.parallel.AsyncSgd` (lock-free shared parameters on real threads, injected delay distributions, DC-ASGD compensation, staleness-aware learning-rate scaling).
- **Experiment:** `experiments.parallel.AsyncTraining`.
- **Toy task:** sparse logistic regression on `SyntheticRecsys.clicks` (Hogwild's regime) and a dense MLP on digits, with delays τ = 0–32.
- **Baselines:** synchronous `DataParallel`; `LocalWorker`/DiLoCo-style local SGD; `AsyncActorCritic`'s update scheme.
- **Expected metric:** loss against wall-clock and against updates, across τ. Hypothesis: Hogwild! matches synchronous accuracy on sparse data with near-linear speed-up on 8 threads; on the dense MLP plain ASGD degrades at τ ≥ 16 and DC-ASGD recovers most of it.
- **Source:** [arXiv:1106.5730](https://arxiv.org/abs/1106.5730), [arXiv:1609.08326](https://arxiv.org/abs/1609.08326)

### 9.6.8 FTRL-Proximal for online click prediction (McMahan et al., KDD 2013)
Google's production algorithm for online logistic regression in ad click prediction. Follow-the-regularised-leader with per-coordinate adaptive learning rates (AdaGrad-style, from the sum of squared gradients) and an L1 term handled in closed form, so most weights are exactly zero (sparse models) while accuracy matches online gradient descent. The paper also gives memory tricks: probabilistic feature inclusion, 16-bit coefficients, and shared counts.
- **Block:** `experimental.stream.FtrlProximal` (per-coordinate FTRL with L1/L2, hashed features from 9.1.7, Bloom-filter feature inclusion).
- **Experiment:** `experiments.streaming.StreamShootout`.
- **Toy task:** `DriftingStreams`' drifting click stream, a million impressions with hashed sparse features.
- **Baselines:** online SGD and AdaGrad logistic regression; batch logistic regression retrained daily; `FactorizationMachine` trained online.
- **Expected metric:** progressive (test-then-train) log loss, number of non-zero weights, and memory. Hypothesis: FTRL matches AdaGrad's log loss with 5–10x fewer non-zeros; daily batch retraining loses after each drift.
- **Source:** [doi:10.1145/2487575.2488200](https://doi.org/10.1145/2487575.2488200)

### 9.6.9 Hoeffding trees and adaptive random forests (Domingos & Hulten, KDD 2000; Gomes et al., Machine Learning 2017)
A Hoeffding tree (VFDT) grows a decision tree from a stream in one pass and constant memory per leaf: a leaf splits once the Hoeffding bound guarantees, with probability 1−δ, that the best split attribute observed so far is truly better than the second best. Adaptive random forests run online bagging (Poisson(6) sample weights) over Hoeffding trees with random feature subsets, and attach a drift detector to each tree, which grows a background tree on a warning and replaces the old tree on a drift.
- **Block:** `experimental.stream.HoeffdingTree` (information-gain splits with the Hoeffding bound, Gaussian numeric observers) and `experimental.stream.AdaptiveRandomForest` (uses 9.6.10's ADWIN).
- **Experiment:** `StreamShootout`.
- **Toy task:** `DriftingStreams` SEA (abrupt), rotating hyperplane (gradual) and RBF (recurring).
- **Baselines:** `RandomForest` retrained on a sliding window; an online MLP; Naive Bayes; a no-change and majority-class predictor.
- **Expected metric:** prequential accuracy over time, recovery time after each drift, and memory. Hypothesis: ARF has the highest prequential accuracy on drifting streams; the windowed forest is competitive only with a well-tuned window and costs 20x the compute.
- **Source:** [doi:10.1145/347090.347107](https://doi.org/10.1145/347090.347107), [doi:10.1007/s10994-017-5642-8](https://doi.org/10.1007/s10994-017-5642-8)

### 9.6.10 Concept-drift detectors: ADWIN, DDM and Page–Hinkley (Bifet & Gavaldà, SDM 2007; Gama et al., 2004; Page, 1954)
Detect when a model's error stream changes. ADWIN keeps a variable-length window of recent values in exponential-histogram buckets and drops the oldest part whenever two sub-windows have means differing by more than a Hoeffding-style bound, so the window adapts to the rate of change. DDM tracks the error rate p and its standard deviation s and warns or alarms when p + s exceeds the recorded minimum by 2 or 3 standard deviations. Page–Hinkley is a cumulative-sum test on deviations from the running mean.
- **Block:** `experimental.stream.DriftDetectors` (ADWIN, DDM, EDDM, Page–Hinkley, a common interface).
- **Experiment:** `StreamShootout` (detection section).
- **Toy task:** error streams of a fixed classifier on `DriftingStreams` (known drift points), plus `AnomalySeries` mean shifts.
- **Baselines:** `BayesianOnlineChangePoint` and `Pelt` from wave 8's `ChangePoints`; a fixed-window two-sample test.
- **Expected metric:** detection delay, false alarms per 10k points, and missed drifts. Hypothesis: ADWIN has the best delay/false-alarm trade-off on gradual drift; DDM is quickest on abrupt drift; BOCPD matches them but costs O(run length) per step.
- **Source:** [doi:10.1137/1.9781611972771.42](https://doi.org/10.1137/1.9781611972771.42), [doi:10.1007/978-3-540-28645-5_29](https://doi.org/10.1007/978-3-540-28645-5_29)

### 9.6.11 Online continual learning with asymmetric cross-entropy: ER-ACE (Caccia et al., ICLR 2022)
In online class-incremental learning (one pass, small replay buffer), new-class samples push old-class logits down, which causes abrupt representation drift when new classes arrive. ER-ACE computes the loss on incoming data with a softmax over only the classes present in the incoming batch (asymmetric), while replayed samples use the full softmax. This stops new data from pushing down the logits of old classes.
- **Block:** `experimental.stream.AsymmetricReplay` (ER-ACE loss, reservoir buffer from `ReservoirReplay`; also ER-AML).
- **Experiment:** `StreamShootout` (continual section).
- **Toy task:** online split-digits and split-`ShapeImages` (two classes per task, one pass, buffer of 20–200 samples).
- **Baselines:** experience replay (`ReservoirReplay`), DER++ and iCaRL (`ClassIncrementalDigits`), A-GEM, `ContinualBackprop`.
- **Expected metric:** final average accuracy and forgetting, and accuracy just after each task switch. Hypothesis: ER-ACE removes most of the post-switch accuracy drop and beats ER by > 0.05 at a buffer of 20; DER++ is close but costs stored logits.
- **Source:** [arXiv:2104.05025](https://arxiv.org/abs/2104.05025)

### 9.6.12 Parameter-free online learning by coin betting: COCOB (Orabona & Tommasi, NeurIPS 2017)
Learning-rate-free optimisation from online learning theory. Each coordinate is treated as a gambler betting on the sign of the negative gradient; the bet is a fraction of the accumulated "wealth" (the reward from past bets), as in Krichevsky–Trofimov betting. The resulting update needs no learning rate and adapts to the unknown distance to the optimum, with optimal regret guarantees. In a stream with drift a schedule-free, tuning-free learner is attractive.
- **Block:** `experimental.stream.CoinBetting` (COCOB-Backprop and KT betting as core `Optimizer`s).
- **Experiment:** `StreamShootout` (optimiser section).
- **Toy task:** the 9.6.8 click stream; digits MLP; `TrainingTricksExperiment`'s regression.
- **Baselines:** `DAdaptation`, `Prodigy`, `ScheduleFree`, AdamW at its best and its 10x-off learning rate.
- **Expected metric:** loss at a fixed budget, and its spread across tasks without tuning. Hypothesis: COCOB is never far from tuned AdamW on convex streams but trails D-Adaptation and Prodigy on the MLP; the parameter-free methods beat AdamW whenever its rate is 10x off.
- **Source:** [arXiv:1705.07795](https://arxiv.org/abs/1705.07795)

### 9.6.13 Integer-only inference with fixed-point requantisation (Jacob et al., CVPR 2018)
Microcontrollers lack floating point, so the whole network must run in integers. With affine quantisation r = S(q − Z), a layer computes in int8 × int8 → int32 accumulators, adds an int32 bias, and requantises to int8 by multiplying by M = S_in·S_w/S_out, represented as an int32 fixed-point multiplier and a right shift, with ReLU folded into the clamp and batch norm folded into the weights. Training with fake quantisation (simulated rounding in the forward pass, straight-through in the backward) makes the integer model match.
- **Block:** `experimental.edge.IntegerInference` (int8 kernels with int32 accumulation, fixed-point multiplier/shift requantisation, BN folding, a bit-exact reference) and a converter from trained float or QAT models.
- **Experiment:** `experiments.edge.EdgeDeployment`.
- **Toy task:** the `ShapeImages` CNN, the digits MLP and 9.6.17's keyword spotter.
- **Baselines:** float32; post-training fake quantisation (core `Quantization`); `LsqDense` and PACT QAT models run in float.
- **Expected metric:** accuracy, agreement with the fake-quant model (bit-exactness), model bytes and integer-op latency on the CPU. Hypothesis: the integer path matches fake-quant outputs exactly and loses < 0.5% accuracy against float with QAT, but 2–5% with post-training quantisation on the small CNN.
- **Source:** [arXiv:1712.05877](https://arxiv.org/abs/1712.05877)

### 9.6.14 Peak-memory-aware inference: MCUNet's patch-based inference and operator reordering (Lin et al., NeurIPS 2020 and 2021)
On a microcontroller, the binding constraint is peak SRAM for activations, which is dominated by the first high-resolution layers. MCUNetV2 runs those layers patch by patch: the input is split into overlapping spatial patches and the first stages are computed per patch up to a downsampled point, so only one patch's activations are alive at a time; receptive-field redistribution shrinks the overlap cost. In-place depthwise convolution and memory-aware operator ordering lower peak memory further.
- **Block:** `experimental.edge.MemoryPlanner` (tensor lifetime analysis over a layer graph, peak-memory calculator, operator reordering) and `experimental.edge.PatchBasedExecutor`.
- **Experiment:** `EdgeDeployment`.
- **Toy task:** `ShapeImages` and `ShapeScenes` at 32×32–64×64 with `MbConvBlock`/`ClassicCnnShootout` networks.
- **Baselines:** layer-by-layer execution; lower input resolution at the same peak memory; a smaller network at the same peak memory.
- **Expected metric:** peak activation memory, extra multiply-adds from patch overlap, and accuracy at a fixed peak-memory budget. Hypothesis: patch-based inference cuts peak memory 4–8x at 10–20% extra compute, which buys higher resolution and more accuracy than shrinking the input at the same budget.
- **Source:** [arXiv:2007.10319](https://arxiv.org/abs/2007.10319), [arXiv:2110.15352](https://arxiv.org/abs/2110.15352)

### 9.6.15 Once-for-all networks: one elastic supernet, many deployments (Cai et al., ICLR 2020)
Train once, then deploy subnetworks of many sizes without retraining. The OFA supernet supports elastic kernel size, depth and width. Progressive shrinking trains the largest network first, then fine-tunes so that sampled smaller kernels (through kernel transformation matrices), shallower depths (keeping the first layers) and narrower widths (keeping the most important channels by L1 norm) also work, with distillation from the largest model. For a hardware budget, an accuracy predictor plus an evolutionary search picks the subnet.
- **Block:** `experimental.edge.ElasticSupernet` (elastic kernel, depth and width on conv blocks, progressive shrinking with KD), `experimental.edge.SubnetSearch` (MLP accuracy predictor, evolutionary search via `experimental.evolve`, a latency/memory table).
- **Experiment:** `EdgeDeployment`.
- **Toy task:** `ShapeImages` CNNs, with budgets from 9.6.14's peak-memory planner and multiply-add counts.
- **Baselines:** each subnet trained from scratch; `WeightSharingNas` (ENAS); compound and width-only scaling from `ClassicCnnShootout`.
- **Expected metric:** accuracy-against-MACs Pareto front and total training cost. Hypothesis: OFA subnets match from-scratch training at 1/k of the cost for k deployments; without progressive shrinking, small subnets lose > 0.1 accuracy.
- **Source:** [arXiv:1908.09791](https://arxiv.org/abs/1908.09791)

### 9.6.16 On-device training under 256 kB: sparse updates, quantisation-aware scaling and lite residuals (Lin et al., NeurIPS 2022; Cai et al., NeurIPS 2020)
Training on a microcontroller must fit activations, gradients and updates in a few hundred kilobytes with an int8 graph. Lin et al. make int8 training work with quantisation-aware scaling (QAS), which rescales the gradients of quantised weights by s⁻² to fix the gradient-to-weight ratio, and save memory with a sparse update, a selection of layers and channels (found by contribution analysis) that are updated while everything else is frozen. TinyTL freezes the weights, updates biases, and adds small "lite residual" branches that run at reduced resolution, since bias updates need no stored activations.
- **Block:** `experimental.edge.SparseUpdateTrainer` (contribution analysis, layer and channel sparse updates, QAS on int8 weights from 9.6.13) and `experimental.edge.LiteResidual`.
- **Experiment:** `EdgeDeployment` (on-device adaptation section).
- **Toy task:** adapt a `ShapeImages` CNN pretrained on 4 glyph classes to new glyph classes or a new noise type, under a 256 kB training-memory budget counted by 9.6.14's planner.
- **Baselines:** full fine-tuning (memory-infeasible, the upper bound); last-layer only; `BitFit` (bias only); `LoraAdapterDense`.
- **Expected metric:** accuracy after adaptation against training memory. Hypothesis: the sparse update reaches within 1–2% of full fine-tuning at a tenth of the memory; without QAS int8 training diverges or lags by > 5%; lite residuals beat BitFit.
- **Source:** [arXiv:2206.15472](https://arxiv.org/abs/2206.15472), [arXiv:2007.11622](https://arxiv.org/abs/2007.11622)

### 9.6.17 Keyword spotting at microcontroller budgets: DS-CNN and its rivals (Zhang et al., "Hello Edge", 2017)
The canonical TinyML task. Zhang et al. compared DNN, CNN, LSTM, GRU, CRNN and depthwise-separable CNN keyword spotters on MFCC features under memory and operation limits for small, medium and large microcontrollers, and found the DS-CNN best at every budget. This item repeats that comparison on the repo's synthetic speech, with the full edge pipeline (9.6.13 integer inference, 9.6.14 peak-memory accounting).
- **Block:** `experimental.edge.KeywordSpotters` (DS-CNN and the other architecture families, sized to budgets).
- **Experiment:** `EdgeDeployment` (keyword-spotting section).
- **Toy task:** `TalkingDigits` single-digit utterances as keywords plus "silence" and "unknown" classes, with noise at 0–20 dB; `Mfcc` features (and PCEN from wave 8 as an option).
- **Baselines:** `SpokenDigitRecognition`'s frame-wise MLP and Conformer (unconstrained); MFCC + DTW templates.
- **Expected metric:** accuracy against model bytes and multiply-adds at 3 budgets (small/medium/large), after integer conversion. Hypothesis: DS-CNN gives the best accuracy at every budget as in the paper; the Conformer does not fit the small budget; int8 conversion costs < 1%.
- **Source:** [arXiv:1711.07128](https://arxiv.org/abs/1711.07128)

---

## Notes on dependencies between items

**Ranking, causal and survival (WP1)**
- First build the three generators in `experimental.data`: `ClickLogs` (built on `SyntheticRecsys`, so `ClickModels` and `CollaborativeFiltering` keep their numbers), `CausalScenarios` (reusing `CausalAndFairness`' confounded generator as its first scenario, so wave 6 rows are unchanged) and `SurvivalData`. Then create `experimental.causal` and `experimental.survival`.
- 9.1.1 → 9.1.2 share `MultiTaskRanking` and the expert towers; 9.1.3 is self-contained on `ClickLogs` and reuses `RankingLosses`/`RankingMetrics` unchanged.
- 9.1.5 → 9.1.6 share `ClickModels`' harness and the planted-order task; 9.1.6 reuses `ReluRoutedMoe` for its sparse variant. 9.1.7's hashed features are reused by WP6 9.6.8 (whichever lands first makes the hashing helper public in `experimental.recsys`).
- 9.1.9 needs a trained 9.1.5 CTR model only as a value source; any `ClickModels` model will do, so it can start in parallel. It uses `LinearProgram` for the hindsight optimum.
- 9.1.10 provides the cross-fitting helper used by 9.1.11–9.1.13; build it first. 9.1.12 reuses `RandomForest` internals (expose the tree builder rather than copying it). 9.1.14 uses `GaussianHead` for the mixture first stage. 9.1.15 is independent.
- 9.1.16 → 9.1.17 share `SurvivalMetrics`. 9.1.18 must keep wave 7's `PriorFittedNetwork` rows reproducible; its pretraining budget sweep is the slowest item in WP1, so start it early.

**Post-training and code (WP2)**
- First create `experimental.align` and a gold-reward harness on the toy LMs (`RubricRewards`' satisfaction score as the gold, proxies trained from it), used by 9.2.1, 9.2.2, 9.2.5 and 9.2.6. 9.2.6's length-controlled win rate uses WP3 9.3.11's GLM; until it lands, report raw win rate plus length.
- 9.2.3 and 9.2.4 share `GeneralPreferences` and the cyclic-preference annotators; 9.2.4's model is 9.2.3's preference source, so build 9.2.4 first.
- 9.2.8–9.2.12 share `ProcessSupervision` and a step-segmented `ChainTask` with error injection. 9.2.9's labeller feeds `ProcessRewardModel.fit` unchanged and its PRM is a baseline for 9.2.10–9.2.11. 9.2.8 wraps `PpoLmTrainer` without changing it; 9.2.11 adds advantages next to `GrpoTrainer`'s RLOO variant without changing its defaults. Schedule 9.2.13 after any open `GrpoTrainer` work, since it adds rows to `ArithmeticRl`.
- 9.2.14 → 9.2.15 → 9.2.16 share `experimental.code.ExecutionTracer` and `ListPrograms`' interpreter (`AbsoluteZero.execute`); build the tracer first. One agent should do 9.2.14–9.2.15.
- 9.2.17 adds options to `HierarchicalAgentTrainer`/`TurnLevelCritic` and must keep wave 8's `ToolAgentBench` rows reproducible with defaults.

**Data and evaluation (WP3)**
- First build `experimental.data.SkillTasks` and create `experimental.eval` and `experimental.synth`. 9.3.1 → 9.3.2 add mixers next to `DataMixture` and `RegressionMixer` and must keep `DataCuration`'s rows; one agent should do both. 9.3.3 and 9.3.4 share `SkillTasks`.
- 9.3.5 reuses `DataPruner` and `SelectiveTokenLoss` as rows. 9.3.6–9.3.8 share `SyntheticTraining`; 9.3.8 reuses the conditional denoisers and guidance of `GuidanceComparison`.
- 9.3.9 reuses `TrainingSchedules`, `ElasticWeightConsolidation` and `SparseMemoryFinetuning` as-is.
- 9.3.10 must not change `Calibration`'s default ECE (other experiments report it); the new estimators are separate methods, and `UncertaintyBenchmark` gets new columns only.
- 9.3.11 → 9.3.12 share the judges and `JudgeAudit`; 9.3.12 needs 9.3.11's judges. 9.3.13 reuses `BenchmarkMethodology`'s planted test set and `ContaminationDetector`.
- 9.3.14 → 9.3.15 share `StatisticalReporting`. 9.3.16 is the most compute-hungry item in the wave (it re-runs whole lineups at up to 32x budget); run it in the background from day one, saving every learning curve, which 9.3.17 then consumes. 9.3.16 must not edit the lineups' experiment classes; it drives them through their public entry points.

**Interpretability, documents and multimodal (WP4)**
- First create `experimental.interp`, `experimental.document` and `experimental.multimodal`, and build `GlyphDocuments` (extending `DigitGlyphs`' font to letters; `ZeroShotDigits` and `EarlyFusion` must keep their numbers) and `ToyTables`.
- 9.4.1 needs edge-level hooks on `HookedDecoder`; if the core `Model.Hook` cannot address edges, add an accessor in a separate PR. 9.4.2 reuses `LowRankIntervention`'s orthonormalisation helpers without changing that class.
- 9.4.3 uses `LinearProbe` and `PersonaVector` unchanged. 9.4.4 and 9.4.7 share `RepresentationComparison`; 9.4.7 reuses the Hungarian solver of `MinibatchOptimalTransport` (already public since wave 8).
- 9.4.8 reuses `WeightPosterior`'s SGLD; it needs `GrokkingExperiment` run long enough to grok, which its test does not do (see the negative-results table), so it runs its own longer schedule.
- 9.4.9 adds rows to `TrainingAttributionExperiment` (owned by `experiments.data`); coordinate with WP3 only if WP3 edits that class (it does not plan to).
- 9.4.10 → 9.4.11 → 9.4.12 share `GlyphDocuments`; 9.4.10's recogniser is the OCR front end of 9.4.11's pipeline baseline and of 9.4.12. 9.4.14 extends `SetPredictionDetector` with options and adds the layout split from `GlyphDocuments`; wave 8 `DetectionShootout` rows stay as they are.
- 9.4.15 → 9.4.16 → 9.4.17 share `FusionShootout`. 9.4.17 needs paired `TalkingDigits` audio and `DigitGlyphs` images; it reads both generators without changing them.

**World models, video and audio (WP5)**
- First create `experimental.world` and `experimental.audio`, and add a video-clip API to `ShapeScenes` if its existing video output lacks one (an additive change, coordinated with nobody else in wave 9).
- 9.5.1 → 9.5.2 → 9.5.3 → 9.5.4 share `VideoGeneration` and the frozen-tracker metrics from `VideoTracking`. 9.5.3 reuses `DistributionMatchingDistiller`; 9.5.4 builds on `DiffusionForcing` without changing it. 9.5.13's FSQ is used by 9.5.2's tokenizer, so build it early.
- 9.5.5–9.5.8 share `WorldModelShootout` and an equal-environment-steps protocol. 9.5.6 adds options to `DreamerAgent`/`MlpWorldModel` whose defaults stay unchanged, so `DreamingAgent` keeps its numbers. 9.5.7 consumes the encoders of `SelfSupervisedBenchmark` read-only. 9.5.8 reuses `SlotAttention` from `ObjectDiscovery`.
- 9.5.9 → 9.5.10 → 9.5.11 → 9.5.12 share `AudioGeneration` and the frozen recogniser of `SpokenDigitRecognition` as a judge. 9.5.9's vocoder is the waveform stage of 9.5.11 and 9.5.12, so build it first. The core `brainiac-audio` module is not touched.
- 9.5.14 → 9.5.15 add options to `PinnTraining` and must keep wave 8's `PdeShootout` rows; one agent should do both. 9.5.15 uses `CurvatureProducts` for Jacobian products. 9.5.16 adds a sweep harness and an irregular-grid data variant to `OperatorLearning` without changing the wave 8 rows.

**Parallel, streaming and edge (WP6)**
- First build `experimental.parallel.Collectives` and `ParallelCostModel` (thread-backed devices, byte and time accounting, calibrated once per machine). 9.6.1 → 9.6.2 share `PipelineSchedule`/`PipelineRunner`; one agent should do both. 9.6.2's split backward needs separate input- and weight-gradient passes; if the core autograd cannot express this, implement it for `Dense` and attention in `experimental.parallel` and propose the core change in a separate PR.
- 9.6.3 → 9.6.6 share the collectives (all-to-all for 9.6.6); 9.6.6 wraps `FineGrainedMoe` and `ExpertChoiceMoe` without changing them. 9.6.4 wraps core `Optimizer`s; it must match `DataParallel`'s loss exactly.
- 9.6.5 drives `Recompute` and compares with `MemoryEfficientTraining`'s ActNN and reversible rows; it changes no core class.
- First build `experimental.data.DriftingStreams` and a prequential evaluation loop in `experimental.stream`. 9.6.9 uses 9.6.10's ADWIN; build 9.6.10 first. 9.6.8 uses WP1 9.1.7's hashing helper (or its own copy until that lands, replaced afterwards). 9.6.10 compares with wave 8's `BayesianOnlineChangePoint` and `Pelt` read-only. 9.6.11 reuses `ReservoirReplay` and `ClassIncrementalDigits`' rows.
- 9.6.13 provides the integer engine for 9.6.16 and 9.6.17; 9.6.14's memory planner provides the budgets for 9.6.15–9.6.17. Build 9.6.13 and 9.6.14 first.

**Cross-package**
- The fourteen new packages (`experimental.causal`, `.survival`, `.align`, `.code`, `.eval`, `.synth`, `.interp`, `.document`, `.multimodal`, `.world`, `.audio`, `.parallel`, `.stream` and `.edge`), plus the `experimental.data` generators (`ClickLogs`, `CausalScenarios`, `SurvivalData`, `SkillTasks`, `GlyphDocuments`, `ToyTables`, `DriftingStreams`), have no dependencies on each other, so all six work packages can start at once.
- Shared helpers to expose rather than copy: feature hashing (WP1 9.1.7 → WP6 9.6.8), the cross-fitting helper (WP1 9.1.10, also usable by WP3's statistics), the Hungarian solver (`MinibatchOptimalTransport`; WP4 9.4.7), `CurvatureProducts` (WP4 9.4.8 baseline, WP5 9.5.15), `WeightPosterior`'s SGLD (WP4 9.4.8), the frozen digit recogniser of `SpokenDigitRecognition` (WP5 9.5.9–9.5.12, WP6 9.6.17), and `VideoTracking`'s detector and tracker (WP5 9.5.1–9.5.4, 9.5.8). Whichever WP lands first makes the method public; the others import it.
- WP2 9.2.6 uses WP3 9.3.11's length-controlled win rate; WP2 9.2.1's gold-reward harness is reused by WP3 9.3.11–9.3.12 as their source of known-quality outputs. Agree the interface early (a `GoldScorer` functional interface in `experimental.align`), or each side uses a local stub until the other lands.
- WP3 9.3.16 re-runs lineups owned by no wave 9 package (timeseries, sequence, attention); it only reads them. WP5 9.5.16 is a separate sweep of `OperatorLearning`; the two share no code.
- WP4 9.4.14 and WP5 9.5.1–9.5.8 both use `ShapeScenes`; WP5 may add a clip API but must not change scene sampling, which WP4's detection rows depend on.
- WP4 9.4.17 and WP6 9.6.17 both use `TalkingDigits`; neither changes the synthesiser.
- WP6 9.6.13–9.6.17 and the wave 8 compression packages (`experimental.compress`) share no code; 9.6.13 converts models trained with `LsqDense`/PACT but does not modify those blocks.
- Core-worthy items: 9.1.7 compact embeddings and 9.1.8 logQ-corrected in-batch softmax (`brainiac-nn` retrieval), 9.1.16 Cox/C-index (`brainiac-train` metrics), 9.3.10 calibration estimators and 9.3.14 evaluation statistics (`brainiac-train` `Evaluation`), 9.4.4 CKA, 9.5.9 a trainable vocoder (`brainiac-audio`), 9.6.4 ZeRO sharding and 9.6.5 checkpoint planning (`brainiac-train`, next to `DataParallel` and `Recompute`), 9.6.8 FTRL (`brainiac-train` optimizers) and 9.6.13 integer-only inference (`brainiac-tensor`/CPU backend). Land them in a separate PR from the experiments.
