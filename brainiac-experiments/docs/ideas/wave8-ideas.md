# Brainiac: wave 8 ideas

*Compiled 2026-11-18 against `claude/project-thread-8zvcl0` (main tree), plus every worktree under `.claude/worktrees/*` (36 worktrees). Waves 1–7 implemented about 800 techniques; see `experiment-results.md`, `unimplemented-ideas.md`, `wave5-ideas.md`, `wave6-ideas.md` and `wave7-ideas.md`. Wave 8 targets the areas that are still thin after seven waves:*

- *(a) computer vision beyond classification: detection, dense prediction (segmentation, optical flow, stereo depth, pose), tracking, video prediction, 3D scene toys (NeRF, hash grids, SDFs), point clouds and group-equivariant convolutions;*
- *(b) time series and signal processing: classical and modern forecasters, pretrained zero-shot forecasting, conformal forecasting, series classification, anomaly and change-point detection, learnable audio front ends;*
- *(c) scientific machine learning: PINNs and their training fixes, neural operators, learned PDE solvers, Hamiltonian/Lagrangian networks, equation discovery, Koopman methods, neural SDEs, equivariant and molecular models, data assimilation;*
- *(d) LLM serving and inference tricks, tokenisation, long context, and agents and tool use;*
- *(e) model compression and pruning beyond what exists, plus compilers and autotuning for kernels;*
- *(f) privacy and federated learning, robustness and certification, active learning and data valuation, spiking and quantum-inspired computing.*

*Wave 7's cut list asked for a dedicated scientific-ML wave (Hamiltonian/Lagrangian networks, PINNs, DeepONet, SINDy); WP3 is that package. Wave 6's cut of "detection and segmentation beyond the U-Net toy" (no dataset) is resolved by one new scene generator (WP1). Four items are fair-test follow-ups of earlier negatives, marked "— follow-up" in their titles (8.2.5, 8.2.16, 8.2.18 and 8.6.15).*

*It also adds the late-2025/2026 items found by a web sweep: Chronos-2 group attention, deep physics-informed KANs, LANO linear-attention neural operators, SpecKV/SpecPC draft-based approximate inference (ICLR 2026), the Length-MAX tokenizer, StochasTok (ICLR 2026), Sculptor active context management, ToolRL/ReTool, FASP and 2SSP structured pruning, geometric data valuation by leverage scores (NeurIPS 2025) and A²SG adaptive surrogate gradients (ICML 2026).*

*Every item was grepped over all `*.java` sources under `src/` in all modules, in the main tree and in all 36 worktrees: 48,875 files, 3,153 unique by content after de-duplication (2,414 distinct file names). The search was case-insensitive and used class names, paper names, author-coined terms and javadoc keywords. An item is listed only when nothing matched, or when the only matches were unrelated. Examples of unrelated matches:*

- *"DETR" matching `GaussianProcessKernels` ("…de**tr**end…"), "YOLO" matching `AgentRunsTest`, "RAFT" matching "d**raft**" in every speculative decoder, and "SORT" matching hundreds of sort calls.*
- *"TimesNet" matching a javadoc mention in `RevIn` ("nearly every modern forecaster … uses it"), and "Autoformer/auto-correlation" matching the pitch tracker of `HarmonicNoiseSynth` and the effective sample size in `McmcDiagnostics`.*
- *"Holt" matching Espeholt (V-trace) and Holtzman (nucleus sampling), and "Huffman" matching the Huffman decoder of the checkpoint reader `Zstd` (file decompression, not weight coding).*
- *"Change point" matching only the coin-bias change-point model of `ProbabilisticPrograms` (one offline posterior, not online BOCPD or PELT).*
- *"PINN" matching "**pinn**ed" in `BridgeMatching` and a javadoc remark in `Lbfgs` ("PINN-like problems"), "Euler–Maruyama" matching only the samplers of `BridgeMatching` and `ScoreSde` (no learned SDE), and "symplectic/leapfrog" matching only `HamiltonianMonteCarlo`.*
- *"Lookahead" matching the Lookahead optimizer (and, correctly, `JacobiDecoding`'s lookahead mode, so lookahead decoding is left out), "H2O" matching `EvictingKvCache.heavyHitters` (correctly: left out), "prompt compression" matching `SnapKv`'s javadoc, and "pushdown" matching core `Grammar` (the automaton exists; 8.4.9 only adds its mask cache).*
- *"SNIP/SynFlow" matching `ZeroCostProxies` (scores for architecture ranking, not pruning masks), "tile size" matching `EvolvedKernels` (evolved loop nests scored on a simulated cache, not timed on the real backend), "2:4" matching `WeightPruning` (masks only, no sparse kernel), "OBS" matching `SparseGPT`'s javadoc and the word "obs", and "e-graph" matching `GraphMath`.*
- *"Lipschitz" matching `SpectralNorm`/`GradientPenalty` (GAN critics), "membership" matching graph block membership and the loss-threshold `membershipAuc` of `OneRunAudit`/`MachineUnlearning` (which is the 8.6.6 baseline), "Sorscher" matching a citation in `DataPruner` (EL2N/forgetting scores only; no prototype metric), "rate coding" matching `SpikeEncoding` (input coding, not ANN-to-SNN conversion), "SpecTr" matching every "spectr…" identifier, "ArCHer" matching `archError`, "MEMO" matching the pickle memo of the checkpoint readers, "hyperbolic" matching tanh, and "spatial transformer" matching only the diffusion `SpatialTransformer` (a pixel transformer) and a javadoc mention of Genie's spatial transformer in `LatentActionWorldModel`, not Jaderberg et al.'s warping network.*

## How this was checked

- **Already implemented, so left out.** These turned up in the systematic check but already have code:
  - Vision: core `Boxes` (IoU, GIoU and its loss, anchor grids, Faster R-CNN box encoding, per-class NMS), core `Loss.dice` and `Functional.lovaszSoftmax`, dilated convolution (`Conv` `.dilation`), `UNet2d`, `SlotAttention`, `RelationNetwork`, `DeformableConv2d`, `GaussianSplats2d`/`GaussianSplats3d` (3DGS in `NovelViewSynthesis`), `FourierFeatures` (NeRF positional encoding and random Fourier features), `EquivariantGraphLayer` (EGNN), `SetEncoder` (Deep Sets, Set Transformer), `PerceiverEncoder` and the diffusion `SpatialTransformer`.
  - Time series and signals: DLinear, PatchTST, N-BEATS (generic and interpretable), TCN, RevIN, DeepAR, quantile MLP (`QuantileLoss`), `DiffusionForcing`, ESN and NG-RC, Kalman/EKF/particle filters (`StateSpaceFilter`), `HiddenMarkovModel`, `DynamicTimeWarping`, `IsolationForest`, core `Signal` (STFT, mel, SpecAugment), `Mfcc`, `ConvTasNet` with uPIT, `HarmonicNoiseSynth` (DDSP).
  - Scientific: `NeuralOde`, `NeuralCde`, `OdeSolver`, `FourierNeuralOperator1d`, `DifferentiableSim`, `KanLayer`, `SymbolicRegression` (GP with lexicase), `LiquidCell`, higher-order autograd (`Var`'s exact second derivatives, `HigherOrderTest`), `Lbfgs`, `HamiltonianMonteCarlo`.
  - Serving and inference: prefix caching, paged KV, continuous batching, chunked prefill, KV offload, Hydragen shared prefixes, prefill/decode disaggregation, EAGLE/EAGLE-3 trees, Medusa, lookahead/Jacobi decoding, prompt lookup, suffix decoding, speculative cascades, SnapKV, H2O (`EvictingKvCache.heavyHitters`), Quest, DuoAttention, PyramidKV/Ada-KV, KVzip, KIVI, MLA conversion, StreamingLLM sinks, YaRN/NTK/dynamic NTK (`RopeScaling`), MemGPT paging (`VirtualContextPager`), MemAgent, recursive LMs, cartridges, grammar-constrained decoding with a pushdown automaton (core `Grammar`, `GrammarConstraint`, `JsonSchemaLite`).
  - Tokenisation: BPE, WordPiece, Unigram, SentencePiece, BPE-dropout, SuperBPE, BLT entropy patches, H-Net dynamic chunking, evolved tokenizers.
  - Agents: core `Agent` (Hermes tool-call loop), Toolformer (`ToolRuntime`, `ToolformerFilter`), Reflexion, Tree of Thoughts (`TestTimeCompute` beam/DFS), Graph of Thoughts, step-level MCTS, debate and voting, skill libraries, generative-agent memory, ACE playbooks, CollabLLM multiturn reward.
  - Compression: magnitude/Wanda/SparseGPT/2:4 (`WeightPruning`), lottery tickets, RigL, ShortGPT (`BlockImportance`), SliceGPT, GPTQ, AWQ, SmoothQuant, HQQ, Hadamard rotation, QTIP, AQLM, SVDQuant, LSQ, ParetoQ, BitNet, logit/feature/sequence KD, GKD, MiniLLM, `LowRankDense`, `ZeroCostProxies` (SNIP/SynFlow scores), `EvolvedKernels` (AlphaEvolve-style softmax and loop-nest evolution).
  - Privacy and robustness: DP-SGD with an RDP accountant, the one-run privacy audit, FedAvg, DiLoCo/DeMo, unlearning (with a loss-threshold membership AUC), latent adversarial training, circuit breakers, spectral normalisation, SNGP/DUQ, conformal classification, temperature scaling, BALD.
  - Data: EL2N and forgetting scores, In-Run Data Shapley, influence functions and TracIn, quality classifiers.
  - Exotic: LIF with surrogate gradients, SpikeGPT, e-prop, STDP, p-bit Ising machine, DTM, MPS classifier and Born machine, simulated variational quantum circuits, stochastic computing.
- **Considered but cut** (to stay near 6 × 17, or too close to an existing block):
  - Mask R-CNN / RoIAlign: two-stage detection is heavy on a CPU and adds little over 8.1.1–8.1.3 on single-scale toy scenes.
  - Self-supervised monocular depth (Monodepth2, SfMLearner): needs a renderer with camera ego-motion; 8.1.9 stereo covers depth.
  - Hypergraph and temporal graph networks (cut again from wave 7): still no fair toy baseline.
  - Speaker verification (GE2E, AAM-softmax): the formant synthesiser has too few speaker factors.
  - BoundlessBPE: too close to `SuperBpeTokenizer` (both merge across pre-token boundaries).
  - Reverso and other 2026 time-series foundation models: their gains are at pretraining scale; 8.2.7 tests the Chronos recipe at toy scale instead.
  - CrossAD and ScatterAD (NeurIPS 2025 anomaly detection): 8.2.13 tests the Anomaly Transformer lineage and the point-adjustment pitfall first; revisit if it wins.
  - LATS and agent workflow memory: too close to `StepMctsSearch`, `ReflexionLoop` and `SkillLibrary`.
  - Poincaré/hyperbolic embeddings: graph material, would belong in a later graph wave.
  - Clifford/geometric-algebra layers and spherical-harmonic (tensor-field) networks: expensive to implement for little gain over EGNN on toy molecules; 8.3.14–8.3.15 cover equivariance.
- **CPU-toy filter.** Every item is pure Java with no external libraries, runs in seconds to minutes on a CPU, and names a task that exists in `brainiac-experiments` or a tiny new generator next to it, plus a baseline already in the repo. The tasks used are:
  - `ShapeImages`, the two-shape scenes of `ShapeSegmentation` (with masks), `ToyImages`, `ToyPoints`, `SeasonalSeries`, `ChaoticSeries` (Lorenz, Mackey–Glass), `ToyTasks` digits/spirals, `TalkingDigits`' formant synthesiser.
  - `SyntheticCorpus`, `CountingTasks`, `ArithmeticTask`, `ChainTask` and the toy LMs (`TinyLm`, `WindowLm`), `LongContextNeedle`, `ServingThroughput`.
  - New small generators named per WP (scene and video generator, series-classification and anomaly generators, PDE reference solvers and mechanical systems, Dirichlet non-IID client splits).

**Format.** Wave 8 keeps wave 7's structure (checks, summary, one section per work package, dependency notes), with each item as a short block:

- **Block**: the reusable class, in `io.github.zzuegg.brainiac.experimental.<pkg>`.
- **Experiment**: the runnable measurement, in `io.github.zzuegg.brainiac.experiments.<theme>`.
- **Toy task**, **Baselines** (all already in the repo) and **Expected metric** (the number to report, with the hypothesis to test).

**Where an item should live:**

- `experimental.<pkg>`: a new block in `brainiac-experiments`. Ten new packages are proposed, each owned by one work package so that no two agents create the same package:
  - WP1: `experimental.vision`.
  - WP2: `experimental.timeseries` and `experimental.anomaly`.
  - WP3: `experimental.sciml`.
  - WP4: `experimental.serving` and `experimental.agent` (tokenizers stay in `experimental.data`, long-context blocks in `experimental.llm`).
  - WP5: `experimental.compress` and `experimental.kernels`.
  - WP6: `experimental.privacy` and `experimental.robust` (active learning and data valuation go in `experimental.data`; spiking and quantum-inspired blocks in `experimental.nn` and `experimental.ml`).
  - Existing blocks are not moved: `UNet2d`, `DLinear`, `PatchTst`, `WeightPruning`, `FederatedAveraging` and the others stay where they are, and the new packages import them.
- `experiments.<theme>`: a new or extended runnable experiment. New themes: `experiments.sciml` (WP3), `experiments.serving` (WP4), `experiments.compression` and `experiments.kernels` (WP5), `experiments.privacy` and `experiments.robustness` (WP6). WP1 and WP2 extend `experiments.vision`, `experiments.timeseries` and `experiments.audio`.
- **core-worthy**: belongs in a core module once measured.

## Summary

| # | Work package | Items |
|---|---|---|
| WP1 | Computer vision beyond classification: detection, dense prediction, tracking, video, 3D scene toys, point clouds and equivariant convolutions | 17 |
| WP2 | Time series, forecasting, anomaly and change-point detection, and signal processing | 18 |
| WP3 | Scientific machine learning: PINNs, neural operators, learned simulators, equation discovery, equivariant and molecular models, data assimilation | 17 |
| WP4 | LLM serving and inference, tokenisation, long context, agents and tool use | 18 |
| WP5 | Compression and pruning beyond the existing set, and compilers/autotuning for kernels | 18 |
| WP6 | Privacy and federated learning, robustness and certification, active learning and data valuation, spiking and quantum-inspired computing | 18 |
| | **Total** | **106** |

---

## WP1. Computer vision beyond classification: detection, dense prediction, tracking, video, 3D scene toys, point clouds and equivariant convolutions (17)

Target: a new `experimental.vision` package. Measurements go in new experiments `experiments.vision.DetectionShootout` (8.1.1–8.1.4), `experiments.vision.DensePrediction` (8.1.5–8.1.9), `experiments.vision.VideoTracking` (8.1.10–8.1.11), `experiments.vision.SceneFields` (8.1.12–8.1.14, next to `generative.NeuralFields` and `NovelViewSynthesis`) and `experiments.vision.GeometricVision` (8.1.15–8.1.17). Baselines: the plain CNN and ViT of `VisionShootout`, `UNet2d` (`ShapeSegmentation`), `SlotAttention`, `FourierFeatures` MLPs, `GaussianSplats3d`, EGNN and Deep Sets, at the same parameter count and step budget, over at least 3 seeds. One new generator, `experimental.data.ShapeScenes`, extends the two-shape scenes of `ShapeSegmentation` to 1–5 shapes at 3 scales with boxes, instance masks, stick-figure keypoints, a stereo pair with known disparity, and short videos of moving shapes with ground-truth optical flow and track ids.

### 8.1.1 Single-stage anchor detector with focal loss and a feature pyramid: RetinaNet-lite (Lin et al., 2017)
A small CNN backbone feeds a feature pyramid (top-down upsampling plus lateral 1×1 connections), and each pyramid level predicts, for every anchor in core `Boxes.anchors`, a class score and four box deltas (`Boxes.encode`). Anchors are matched to ground truth by IoU; the class head is trained with the focal loss (core `Loss.focal`) so the thousands of easy background anchors do not swamp the gradient, the box head with smooth-L1 or GIoU (`Boxes.giouLoss`). Detections are decoded and filtered by per-class `Boxes.nms`.
- **Block:** `experimental.vision.AnchorDetector`, `experimental.vision.FeaturePyramid`, `experimental.vision.DetectionMetrics` (mAP@0.5 and mAP@[.5:.95], shared by the whole package).
- **Experiment:** `experiments.vision.DetectionShootout`.
- **Toy task:** `ShapeScenes` with 1–5 shapes at 3 scales, 32×32.
- **Baselines:** the `VisionShootout` CNN applied as a sliding-window classifier plus `Boxes.nms`; the U-Net mask plus connected components; ablations: no pyramid, cross-entropy instead of focal, smooth-L1 instead of GIoU.
- **Expected metric:** mAP@0.5 and multiply-adds per image. Hypothesis: focal loss is worth > 0.1 mAP over cross-entropy; the pyramid matters only on the small-shape slice.
- **Source:** [arXiv:1708.02002](https://arxiv.org/abs/1708.02002), [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)

### 8.1.2 CenterNet: objects as points (Zhou et al., 2019)
An anchor-free detector: a fully convolutional net outputs a per-class heatmap of object centres (trained with a penalty-reduced focal loss against Gaussian-splatted targets), plus a per-pixel width/height and a sub-pixel offset. Inference is a 3×3 max-pool peak extraction, so no anchors and no NMS are needed.
- **Block:** `experimental.vision.CenterNetHead` (reusable heatmap utilities, also used by 8.1.9).
- **Experiment:** `DetectionShootout`.
- **Toy task:** `ShapeScenes`, including a crowded split where shapes overlap.
- **Baselines:** 8.1.1 RetinaNet-lite; the sliding-window CNN.
- **Expected metric:** mAP@0.5 and decode time. Hypothesis: matches 8.1.1 at lower cost, and loses on heavy overlap where two centres fall in one cell.
- **Source:** [arXiv:1904.07850](https://arxiv.org/abs/1904.07850)

### 8.1.3 DETR-lite: set prediction with Hungarian matching, plus denoising queries (Carion et al., 2020; Li et al., 2022)
A transformer decoder with N learned object queries cross-attends to CNN features and emits N (class, box) pairs, one of them "no object". The loss first finds the optimal one-to-one assignment between predictions and ground truth with the Hungarian algorithm on a class + L1 + GIoU cost, then trains on the matched pairs, so duplicates are penalised and no NMS is needed. DETR converges slowly; DN-DETR adds noised ground-truth boxes as extra queries that must be denoised, which stabilises the matching early on.
- **Block:** `experimental.vision.SetPredictionDetector` (reuses the Hungarian solver of `MinibatchOptimalTransport` and the transformer blocks).
- **Experiment:** `DetectionShootout`, with a learning curve.
- **Toy task:** `ShapeScenes`.
- **Baselines:** 8.1.1, 8.1.2; DETR without denoising queries.
- **Expected metric:** mAP@0.5 against training steps. Hypothesis: plain DETR is the slowest to converge of the three; denoising queries close much of the gap.
- **Source:** [arXiv:2005.12872](https://arxiv.org/abs/2005.12872), [arXiv:2203.01305](https://arxiv.org/abs/2203.01305)

### 8.1.4 Instance segmentation with a discriminative embedding loss (De Brabandere et al., 2017)
A U-Net predicts a per-pixel embedding; the loss pulls pixels of one instance towards their mean (a hinged variance term), pushes instance means apart (a hinged distance term) and keeps means near the origin. Instances are read out by mean-shift or by thresholding distances to seeds, so the number of objects need not be known in advance.
- **Block:** `experimental.vision.DiscriminativeEmbeddingLoss`, `experimental.vision.EmbeddingClustering`.
- **Experiment:** `DetectionShootout` (instance rows).
- **Toy task:** `ShapeScenes` instance masks, including touching shapes of the same class.
- **Baselines:** U-Net semantic segmentation + connected components (`ShapeSegmentation`); `SlotAttention` (`ObjectDiscovery`).
- **Expected metric:** mask AP@0.5 and FG-ARI. Hypothesis: beats connected components exactly on touching same-class shapes, and beats slot attention (FG-ARI 0.303) because it is supervised.
- **Source:** [arXiv:1708.02551](https://arxiv.org/abs/1708.02551)

### 8.1.5 DeepLab: atrous spatial pyramid pooling, with a segmentation-loss sweep (Chen et al., 2017, 2018; Berman et al., 2018)
ASPP runs parallel 3×3 convolutions at several dilation rates (core `Conv` `.dilation`) plus a global-pooling branch and concatenates them, so one layer sees several receptive fields without downsampling; DeepLabv3+ adds a light decoder with one skip. The item also measures, for the first time, the core `Loss.dice` and `Functional.lovaszSoftmax` losses against cross-entropy, which exist in core but have no experiment row.
- **Block:** `experimental.vision.AtrousPyramidPooling`.
- **Experiment:** `experiments.vision.DensePrediction`.
- **Toy task:** `ShapeSegmentation` masks and the multi-scale `ShapeScenes` masks.
- **Baselines:** `UNet2d` (mIoU 0.458), the no-skip network (0.051).
- **Expected metric:** mIoU and multiply-adds; per-loss mIoU. Hypothesis: ASPP matches the U-Net with fewer parameters; Lovász gives the best mIoU, since it optimises it directly.
- **Source:** [arXiv:1706.05587](https://arxiv.org/abs/1706.05587), [arXiv:1802.02611](https://arxiv.org/abs/1802.02611), [arXiv:1705.08790](https://arxiv.org/abs/1705.08790)

### 8.1.6 Classical optical flow: Horn–Schunck and pyramidal Lucas–Kanade (Horn & Schunck, 1981; Lucas & Kanade, 1981; Bouguet, 2000)
Both solve the brightness-constancy equation I_x u + I_y v + I_t = 0. Horn–Schunck adds a global smoothness term and solves it by Jacobi iterations; Lucas–Kanade assumes constant flow in a window and solves a 2×2 least-squares system per pixel, run coarse-to-fine on an image pyramid for large motions. These are the training-free references for 8.1.7.
- **Block:** `experimental.vision.ClassicalFlow` (`hornSchunck`, `lucasKanade`), plus `experimental.vision.FlowMetrics` (end-point error, warping).
- **Experiment:** `DensePrediction` (flow rows).
- **Toy task:** `ShapeScenes` videos: frame pairs with ground-truth flow, displacements 0–4 px, with and without noise.
- **Baselines:** zero flow; the `VisionShootout` CNN regressing flow directly.
- **Expected metric:** average end-point error (EPE). Hypothesis: both classical methods beat the CNN regressor at this data size; Lucas–Kanade fails in textureless interiors, Horn–Schunck fills them in.
- **Source:** [Horn & Schunck 1981 (DOI)](https://doi.org/10.1016/0004-3702(81)90024-2), [Lucas & Kanade (IJCAI 1981)](https://www.ijcai.org/Proceedings/81-2/Papers/017.pdf)

### 8.1.7 RAFT-lite: correlation volume and recurrent refinement (Teed & Deng, 2020)
A shared CNN encodes both frames; the dot product of every feature pair gives a 4-D all-pairs correlation volume, pooled into a small pyramid. A convolutional GRU starts at zero flow and, at each iteration, looks up correlations around the current estimate and predicts an update, so the estimate is refined over 4–12 steps with tied weights. Training supervises every iteration with exponentially increasing weights.
- **Block:** `experimental.vision.CorrelationVolume`, `experimental.vision.RecurrentFlowRefiner` (reuses `GruCell`).
- **Experiment:** `DensePrediction`.
- **Toy task:** as 8.1.6, plus large displacements (6–10 px).
- **Baselines:** 8.1.6 Horn–Schunck and Lucas–Kanade; a FlowNet-style `UNet2d` regressing flow from the stacked frames.
- **Expected metric:** EPE against refinement iterations. Hypothesis: RAFT-lite beats the U-Net, and only it handles the large-displacement split.
- **Source:** [arXiv:2003.12039](https://arxiv.org/abs/2003.12039)

### 8.1.8 Stereo depth by a cost volume and soft-argmin: GC-Net-lite (Kendall et al., 2017)
Left and right features are compared at every candidate disparity d (concatenation or correlation of the left feature with the right feature shifted by d), giving a cost volume; a few 3-D convolutions regularise it, and a soft-argmin over d (the softmax-weighted mean disparity) makes the output differentiable and sub-pixel.
- **Block:** `experimental.vision.StereoCostVolume`.
- **Experiment:** `DensePrediction` (depth rows).
- **Toy task:** `ShapeScenes` stereo pairs: shapes at random depths, disparity 0–6 px.
- **Baselines:** classical block matching (SAD, winner-take-all, implemented in the block as a reference row); `UNet2d` regressing disparity from the stacked pair.
- **Expected metric:** end-point error and share of pixels with error > 1 px. Hypothesis: the cost volume beats the U-Net clearly, because the geometry is built in.
- **Source:** [arXiv:1703.04309](https://arxiv.org/abs/1703.04309)

### 8.1.9 Keypoint heatmaps with integral regression versus direct regression (Toshev & Szegedy, 2014; Sun et al., 2018)
DeepPose regresses keypoint coordinates directly from a CNN. Heatmap methods predict one map per keypoint; integral regression takes the softmax-normalised map's expected coordinate (soft-argmax), which is differentiable, sub-pixel and trained end to end with an L1 loss on coordinates.
- **Block:** `experimental.vision.KeypointHeads` (`direct`, `heatmapArgmax`, `integral`).
- **Experiment:** `DensePrediction` (pose rows).
- **Toy task:** a stick-figure generator in `ShapeScenes`: 5–9 joints, random limb angles, occluding bars.
- **Baselines:** direct regression with the `VisionShootout` CNN (DeepPose); argmax heatmaps.
- **Expected metric:** PCK@2px and mean joint error. Hypothesis: integral regression beats argmax at low resolution (quantisation) and direct regression on rare poses.
- **Source:** [arXiv:1312.4659](https://arxiv.org/abs/1312.4659), [arXiv:1711.08229](https://arxiv.org/abs/1711.08229)

### 8.1.10 Multi-object tracking: SORT and ByteTrack (Bewley et al., 2016; Zhang et al., 2022)
Tracking by detection. SORT keeps a constant-velocity Kalman filter per track, predicts each track into the next frame, and assigns detections by Hungarian matching on IoU; unmatched tracks age out. ByteTrack also uses the low-confidence detections in a second association round with the tracks left over, which recovers occluded objects that a score threshold would drop.
- **Block:** `experimental.vision.TrackingByDetection` (`sort`, `byteTrack`), reusing `StateSpaceFilter.kalman` and the Hungarian solver; `experimental.vision.TrackingMetrics` (MOTA, IDF1, ID switches).
- **Experiment:** `experiments.vision.VideoTracking`.
- **Toy task:** `ShapeScenes` videos with crossing trajectories and short occlusions; detections from 8.1.2 (or from ground truth plus noise and dropouts, to separate detector and tracker errors).
- **Baselines:** nearest-centroid greedy association; SORT without the Kalman prediction.
- **Expected metric:** MOTA, IDF1 and ID switches. Hypothesis: the Kalman prediction halves ID switches at crossings; ByteTrack's second round lifts IDF1 under occlusion.
- **Source:** [arXiv:1602.00763](https://arxiv.org/abs/1602.00763), [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)

### 8.1.11 Video prediction: ConvLSTM and SimVP (Shi et al., 2015; Gao et al., 2022)
ConvLSTM replaces the matrix products of an LSTM with convolutions, so its state is a feature map and it can predict the next frames of a video. SimVP shows that a purely convolutional encoder–translator–decoder that stacks the input frames along channels and predicts all future frames at once matches recurrent predictors at a fraction of the cost.
- **Block:** `experimental.vision.ConvLstmCell`, `experimental.vision.SimVp`.
- **Experiment:** `VideoTracking` (prediction rows).
- **Toy task:** `ShapeScenes` videos: 4 context frames, predict 4–8 frames of bouncing shapes.
- **Baselines:** copy the last frame; a `UNet2d` frame predictor; the latent dynamics of `LatentActionWorldModel` without actions.
- **Expected metric:** per-frame MSE and SSIM against horizon. Hypothesis: both beat copy-last; SimVP ties ConvLSTM at ≥ 3× lower training time.
- **Source:** [arXiv:1506.04214](https://arxiv.org/abs/1506.04214), [arXiv:2206.05099](https://arxiv.org/abs/2206.05099)

### 8.1.12 NeRF: volume rendering of a radiance field (Mildenhall et al., 2020)
An MLP maps a 3-D point (with `FourierFeatures.positional` encoding) and a view direction to a density σ and a colour. A pixel is rendered by marching along its camera ray, accumulating colour with weights T_i·(1 − exp(−σ_i δ_i)), where T_i is the transmittance so far; hierarchical sampling places a second set of samples where the first pass found mass. Only images and poses are needed. The repo has 3DGS but no volume renderer.
- **Block:** `experimental.vision.RadianceField`, `experimental.vision.VolumeRenderer`.
- **Experiment:** `experiments.vision.SceneFields` (shares its scene and cameras with `NovelViewSynthesis`).
- **Toy task:** the `NovelViewSynthesis` toy scene, 16×16 or 24×24 renders, 20 training views, 10 held-out views.
- **Baselines:** `GaussianSplats3d` (3DGS); NeRF without positional encoding.
- **Expected metric:** held-out PSNR, training time and render time. Hypothesis: NeRF matches 3DGS on PSNR but renders ≥ 10× slower; without the encoding it blurs.
- **Source:** [arXiv:2003.08934](https://arxiv.org/abs/2003.08934)

### 8.1.13 Multiresolution hash encoding: Instant-NGP (Müller et al., 2022)
The input coordinate is looked up in L grids of increasing resolution; each grid's corner features live in a fixed-size hash table (collisions are allowed, and gradients sort them out), and trilinearly interpolated features from all levels are concatenated and fed to a tiny MLP. Most capacity sits in the tables, so a small MLP suffices and training is much faster than with Fourier features.
- **Block:** `experimental.vision.HashGridEncoding`.
- **Experiment:** `SceneFields` and `generative.NeuralFields` (2-D image fitting).
- **Toy task:** the `NeuralFields` checkerboard-and-disc image; the 8.1.12 scene.
- **Baselines:** `FourierFeatures` MLP and raw MLP (`NeuralFields`); 2-D Gaussian splats; 8.1.12 NeRF with positional encoding.
- **Expected metric:** PSNR against wall-clock seconds and against table size (2¹⁰–2¹⁶). Hypothesis: reaches the Fourier-feature PSNR in a fraction of the steps; collisions start to hurt below 2¹².
- **Source:** [arXiv:2201.05989](https://arxiv.org/abs/2201.05989)

### 8.1.14 SIREN and neural signed distance functions with the eikonal loss (Sitzmann et al., 2020; Park et al., 2019; Gropp et al., 2020)
SIREN uses sin(ω₀·Wx + b) activations with a special initialisation, so the network and all its derivatives are well behaved. A neural SDF f(x) is fitted to points sampled on a surface (f = 0 there), and the eikonal loss (‖∇f‖ − 1)² everywhere makes f a true distance; the gradient comes from autograd. DeepSDF conditions one network on a per-shape latent code (auto-decoder), so a family of shapes shares one model.
- **Block:** `experimental.vision.SineLayer` (SIREN), `experimental.vision.SignedDistanceField` (eikonal loss, auto-decoder latents).
- **Experiment:** `SceneFields`.
- **Toy task:** 2-D outlines of the `ShapeImages` shapes and of random polygons, from 200 surface points each; image fitting as in `NeuralFields`.
- **Baselines:** ReLU MLP and `FourierFeatures` MLP with the same losses.
- **Expected metric:** Chamfer distance and IoU of the zero level set; image PSNR. Hypothesis: SIREN fits images like Fourier features but gives much cleaner gradients, so the eikonal residual is lowest.
- **Source:** [arXiv:2006.09661](https://arxiv.org/abs/2006.09661), [arXiv:1901.05103](https://arxiv.org/abs/1901.05103), [arXiv:2002.10099](https://arxiv.org/abs/2002.10099)

### 8.1.15 PointNet and DGCNN for point clouds (Qi et al., 2017; Wang et al., 2019)
PointNet applies a shared MLP to every point and max-pools over points, so it is permutation-invariant; a small T-Net predicts an input alignment. DGCNN adds EdgeConv: at each layer it rebuilds a k-nearest-neighbour graph in feature space and applies an MLP to (x_i, x_j − x_i) for every neighbour, then max-pools, capturing local geometry that PointNet misses.
- **Block:** `experimental.vision.PointNet`, `experimental.vision.EdgeConv`.
- **Experiment:** `experiments.vision.GeometricVision` (next to `exotic.EquivariantPointClouds`).
- **Toy task:** `ToyPoints` shape classification (sphere, cube, torus, …) with noise and random rotations; per-point part segmentation.
- **Baselines:** EGNN (`EquivariantGraphLayer`), Deep Sets (`SetEncoder`), an MLP on sorted coordinates.
- **Expected metric:** accuracy, part mIoU, and accuracy under unseen rotations. Hypothesis: DGCNN > PointNet ≈ Deep Sets; only EGNN is robust to rotation.
- **Source:** [arXiv:1612.00593](https://arxiv.org/abs/1612.00593), [arXiv:1801.07829](https://arxiv.org/abs/1801.07829)

### 8.1.16 Group-equivariant CNNs (Cohen & Welling, 2016)
A G-CNN for the group p4 (translations and 90° rotations) first lifts the image to a function on the group by convolving with 4 rotated copies of each filter, then applies group convolutions whose filters are rotated and cyclically shifted along the rotation axis; pooling over rotations at the end gives exact rotation invariance with the same parameter count as a CNN.
- **Block:** `experimental.vision.GroupConv2d` (lifting and p4 group convolution, rotation pooling).
- **Experiment:** `GeometricVision`.
- **Toy task:** `ShapeImages` with diagonals and bars (classes related by rotation removed or merged), trained upright and tested rotated by 90°/180°/270°; a low-data split (64 images).
- **Baselines:** the `VisionShootout` CNN with and without rotation augmentation, at equal parameters.
- **Expected metric:** test accuracy on rotated images, and in the low-data split. Hypothesis: the G-CNN is exactly invariant (no drop) and beats augmentation in the low-data split.
- **Source:** [arXiv:1602.07576](https://arxiv.org/abs/1602.07576)

### 8.1.17 Spatial transformer networks (Jaderberg et al., 2015)
A localisation network regresses the six parameters of an affine transform from the input; a differentiable grid generator and bilinear sampler then warp the image (or a feature map) with it, so the classifier sees a canonicalised input. Trained end to end from the classification loss only. (The diffusion `SpatialTransformer` is a pixel transformer, not this.)
- **Block:** `experimental.vision.AffineSampler` (grid generation and bilinear sampling, also reused by 8.1.7's warping and 8.1.8's shifts), `experimental.vision.SpatialTransformerNetwork`.
- **Experiment:** `GeometricVision`.
- **Toy task:** `ShapeImages` with random rotation (±45°), scale (0.6–1.2) and translation, plus clutter.
- **Baselines:** the `VisionShootout` CNN, `DeformableConv2d` CNN (accuracy 1.0 on clean shapes), and 8.1.16.
- **Expected metric:** accuracy under distortion, and the variance of the learned transforms per class. Hypothesis: the STN helps on scale and rotation where the plain CNN degrades; the deformable CNN is a strong rival.
- **Source:** [arXiv:1506.02025](https://arxiv.org/abs/1506.02025)

## WP2. Time series, forecasting, anomaly and change-point detection, and signal processing (18)

Target: two new packages, `experimental.timeseries` (forecasters, series classifiers, conformal wrappers and signal front ends) and `experimental.anomaly` (anomaly and change-point detectors). Measurements go in:

- `experiments.timeseries.ForecastingBenchmark` (extended) and a new `experiments.timeseries.LongHorizonForecasting` (multivariate, long horizons, covariates);
- `experiments.timeseries.ProbabilisticForecasting` (extended, for 8.2.7–8.2.8);
- a new `experiments.timeseries.SeriesClassification`;
- a new `experiments.timeseries.AnomalyBenchmark` and `experiments.timeseries.ChangePoints`;
- `experiments.audio.SpokenDigitRecognition` and `experiments.audio.SourceSeparation` (extended, for 8.2.16–8.2.18).

Baselines: naive and seasonal naive (MSE 0.026 on `SeasonalSeries`), DLinear (0.019), PatchTST (0.019), MLP + RevIN, N-BEATS, TCN, DeepAR (80% coverage 0.75), the quantile MLP (0.77), `DynamicTimeWarping`, `IsolationForest`, the autoencoder of `ForestBenchmarks`, `StateSpaceFilter`, `HiddenMarkovModel`, core `Signal` log-mel and `Mfcc`, all at equal budget over at least 3 seeds. New generators: multivariate `SeasonalSeries` with cross-channel lags and known future covariates (holiday spikes, price steps); `experimental.data.SeriesClassificationTasks` (the classic cylinder–bell–funnel task of Saito, 1994, plus shapelet and frequency classes); `experimental.data.AnomalySeries` with planted point, contextual, collective and seasonal-shift anomalies and labelled change points.

### 8.2.1 Exponential smoothing (Holt–Winters ETS) and ARIMA in state-space form (Holt, 1957; Winters, 1960; Box & Jenkins, 1970; Hyndman et al., 2008)
ETS keeps exponentially smoothed level, trend and seasonal components with additive or multiplicative errors; ARIMA models the differenced series as autoregressive plus moving-average noise. Both are written as linear Gaussian state-space models, so the existing Kalman filter gives the likelihood for fitting their few parameters (by L-BFGS) and exact prediction intervals. No classical statistical forecaster exists in the repo; deep forecasters are compared only with naive baselines.
- **Block:** `experimental.timeseries.ExponentialSmoothing`, `experimental.timeseries.Arima` (on `StateSpaceFilter`, fitted by `Lbfgs`), with AICc model selection.
- **Experiment:** `ForecastingBenchmark`, `ProbabilisticForecasting`.
- **Toy task:** `SeasonalSeries` (trend + season + noise) and the new covariate series.
- **Baselines:** seasonal naive, DLinear, PatchTST, DeepAR.
- **Expected metric:** test MSE and 80% interval coverage. Hypothesis: ETS ties DLinear on `SeasonalSeries` (it is the true generative family) and gives the best-calibrated intervals.
- **Source:** [Hyndman et al., *Forecasting with Exponential Smoothing* (2008)](https://doi.org/10.1007/978-3-540-71918-2), [Box–Jenkins method](https://en.wikipedia.org/wiki/Box%E2%80%93Jenkins_method)

### 8.2.2 Autoformer and FEDformer: decomposition with auto-correlation or frequency attention (Wu et al., 2021; Zhou et al., 2022)
Autoformer builds series decomposition (moving-average trend, residual season) into every layer, and replaces self-attention by auto-correlation: it finds the top-k lags with the highest autocorrelation (via FFT) and aggregates time-delayed copies of the values. FEDformer instead attends over a random subset of Fourier modes. DLinear was written to show such models lose to a linear map; this item gives the repo its first test of that claim.
- **Block:** `experimental.timeseries.AutoCorrelationLayer`, `experimental.timeseries.FrequencyEnhancedBlock`, `experimental.timeseries.SeriesDecomposition`.
- **Experiment:** `LongHorizonForecasting`.
- **Toy task:** `SeasonalSeries` at horizons 24/96/192, and multivariate series.
- **Baselines:** DLinear, PatchTST, a vanilla transformer forecaster.
- **Expected metric:** test MSE per horizon. Hypothesis: both beat the vanilla transformer, neither beats DLinear (reproducing Zeng et al., 2023).
- **Source:** [arXiv:2106.13008](https://arxiv.org/abs/2106.13008), [arXiv:2201.12740](https://arxiv.org/abs/2201.12740), [arXiv:2205.13504](https://arxiv.org/abs/2205.13504)

### 8.2.3 iTransformer: attention across variates instead of time (Liu et al., ICLR 2024)
The transformer is "inverted": each variate's whole lookback window is embedded as one token, self-attention mixes variates (capturing cross-channel correlation), and the feed-forward network acts on each variate's temporal representation. With few variates it is cheap, and it handles misaligned channels well.
- **Block:** `experimental.timeseries.InvertedTransformer`.
- **Experiment:** `LongHorizonForecasting`.
- **Toy task:** multivariate `SeasonalSeries` with planted cross-channel lags (channel j copies channel i with a delay), and independent channels as a control.
- **Baselines:** PatchTST (channel-independent), DLinear, 8.2.2.
- **Expected metric:** test MSE. Hypothesis: iTransformer wins only when channels are coupled; on independent channels PatchTST is at least as good.
- **Source:** [arXiv:2310.06625](https://arxiv.org/abs/2310.06625)

### 8.2.4 TimesNet: periods as 2-D images (Wu et al., ICLR 2023)
TimesNet finds the dominant periods by FFT, folds the 1-D series into a 2-D tensor per period (rows = periods, columns = phase), applies a 2-D inception convolution that captures both intra-period and inter-period variation, unfolds, and sums the branches weighted by their spectral amplitudes.
- **Block:** `experimental.timeseries.TimesBlock`.
- **Experiment:** `LongHorizonForecasting`; also `SeriesClassification` (TimesNet is a general backbone).
- **Toy task:** `SeasonalSeries` with two incommensurate periods; CBF classification.
- **Baselines:** PatchTST, TCN, DLinear.
- **Expected metric:** MSE; classification accuracy. Hypothesis: wins on the two-period series, ties elsewhere at higher cost.
- **Source:** [arXiv:2210.02186](https://arxiv.org/abs/2210.02186)

### 8.2.5 N-HiTS: hierarchical interpolation (Challu et al., AAAI 2023) — follow-up
N-BEATS lost to seasonal naive here (0.030 generic, 0.071 interpretable vs 0.026). N-HiTS changes N-BEATS in two ways: each stack max-pools its input at a different rate (multi-rate sampling), and each block predicts only a few coefficients that are interpolated up to the full horizon, so early stacks model low frequencies and later ones high frequencies with far fewer parameters. The follow-up also gives N-BEATS its paper recipe (ensembling over lookbacks and losses), so the earlier negative gets a fair test.
- **Block:** `experimental.timeseries.NHits` (reuses `NBeats` block plumbing).
- **Experiment:** `ForecastingBenchmark`, `LongHorizonForecasting`.
- **Toy task:** `SeasonalSeries`, horizons 24 and 192.
- **Baselines:** N-BEATS generic and interpretable (single and ensembled), seasonal naive, DLinear.
- **Expected metric:** test MSE and parameters. Hypothesis: N-HiTS beats seasonal naive and N-BEATS at long horizons; ensembling alone lifts N-BEATS below seasonal naive.
- **Source:** [arXiv:2201.12886](https://arxiv.org/abs/2201.12886), [arXiv:1905.10437](https://arxiv.org/abs/1905.10437)

### 8.2.6 TiDE: a dense encoder–decoder with covariates (Das et al., 2023)
TiDE flattens the lookback, the past covariates and a projection of the future covariates into one vector, encodes it with residual MLP blocks, decodes to a per-step representation, and combines it with each future step's covariates in a "temporal decoder", plus a linear skip from the lookback. It is an MLP that uses known future inputs, which DLinear and PatchTST ignore.
- **Block:** `experimental.timeseries.Tide`.
- **Experiment:** `LongHorizonForecasting` (covariate rows).
- **Toy task:** the new covariate series: demand driven by a known future price step and holiday flags.
- **Baselines:** DLinear and PatchTST without covariates; DeepAR with covariates; MLP + RevIN.
- **Expected metric:** test MSE overall and on holiday steps. Hypothesis: TiDE wins mainly on steps where a covariate changes.
- **Source:** [arXiv:2304.08424](https://arxiv.org/abs/2304.08424)

### 8.2.7 Chronos-style pretrained zero-shot forecasting, with KernelSynth data and Chronos-2 group attention (Ansari et al., 2024; Amazon, 2025)
Chronos mean-scales a series, quantises the values into ~4k bins, and trains a language model on the tokens with cross-entropy, so forecasting becomes next-token prediction and samples give probabilistic forecasts. Much of its training data is KernelSynth: series drawn from Gaussian-process priors with randomly composed kernels. Chronos-2 (2025) adds "group attention" across the series in a batch group (related series and covariates), so it can do in-context multivariate and covariate-aware forecasting zero-shot. The CPU version pretrains a tiny model on KernelSynth only and evaluates zero-shot on the repo's series.
- **Block:** `experimental.timeseries.ValueTokenizer`, `experimental.timeseries.KernelSynth` (on `KernelFunction`/`GaussianProcess`), `experimental.timeseries.GroupAttentionForecaster`.
- **Experiment:** `ProbabilisticForecasting` (zero-shot rows).
- **Toy task:** pretrain on KernelSynth; test on `SeasonalSeries`, Mackey–Glass and the covariate series, never seen in training.
- **Baselines:** seasonal naive; DeepAR and the quantile MLP trained per dataset; PatchTST trained per dataset.
- **Expected metric:** MSE, CRPS and 80% coverage. Hypothesis: zero-shot beats seasonal naive but trails per-dataset models; group attention helps only on the covariate series.
- **Source:** [arXiv:2403.07815](https://arxiv.org/abs/2403.07815), [arXiv:2510.15821](https://arxiv.org/abs/2510.15821)

### 8.2.8 Conformal forecasting under drift: CQR, ACI and EnbPI (Romano et al., 2019; Gibbs & Candès, 2021; Xu & Xie, 2021)
Conformalised quantile regression widens a quantile model's interval by the calibration-set quantile of its errors, giving finite-sample coverage under exchangeability. Time series break exchangeability, so adaptive conformal inference updates the target miscoverage online, α_{t+1} = α_t + γ(α − err_t), which guarantees long-run coverage even under arbitrary drift; EnbPI uses leave-one-out bootstrap ensembles and a sliding window of residuals. Only conformal classification exists in the repo.
- **Block:** `experimental.timeseries.ConformalForecaster` (`cqr`, `aci`, `enbpi`).
- **Experiment:** `ProbabilisticForecasting`.
- **Toy task:** `SeasonalSeries` with a variance jump and a level shift halfway through the test period.
- **Baselines:** DeepAR (coverage 0.75), the quantile MLP (0.77), split conformal (the `ConformalClassifier` recipe applied to residuals), 8.2.1 ETS intervals.
- **Expected metric:** rolling coverage at 80% and mean width. Hypothesis: only ACI recovers 0.80 after the shift; CQR is valid before it.
- **Source:** [arXiv:1905.03222](https://arxiv.org/abs/1905.03222), [arXiv:2106.00170](https://arxiv.org/abs/2106.00170), [arXiv:2010.09107](https://arxiv.org/abs/2010.09107)

### 8.2.9 ROCKET and MiniRocket: random convolutional kernels (Dempster et al., 2020, 2021)
Thousands of random 1-D convolution kernels (random length, weights, bias, dilation and padding) are applied to the series; each yields two features, the maximum and the proportion of positive values (PPV), and a ridge classifier is trained on them. MiniRocket makes the kernels almost deterministic and uses PPV only. No training of the convolutions is needed, and it is a very strong baseline in series classification.
- **Block:** `experimental.timeseries.RandomConvolutionFeatures` (`rocket`, `miniRocket`; ridge from `RidgeRegression`).
- **Experiment:** `experiments.timeseries.SeriesClassification`.
- **Toy task:** `SeriesClassificationTasks` (CBF, shapelets, frequency classes); spoken-digit band energies as a second task.
- **Baselines:** 1-NN with `DynamicTimeWarping`, MLP, TCN.
- **Expected metric:** accuracy and fit time. Hypothesis: ROCKET ≥ DTW-1NN at a fraction of the time, and beats the trained TCN at this data size.
- **Source:** [arXiv:1910.13051](https://arxiv.org/abs/1910.13051), [arXiv:2012.08791](https://arxiv.org/abs/2012.08791)

### 8.2.10 TS2Vec: hierarchical contrastive representations for series (Yue et al., AAAI 2022)
A dilated-convolution encoder produces per-timestamp representations; two overlapping random crops of a series are encoded, and contrastive losses are applied at every scale of a max-pooling hierarchy, both across instances and across timestamps (temporal contrast). The representations are then pooled for classification or used for forecasting with a linear head.
- **Block:** `experimental.timeseries.Ts2Vec`.
- **Experiment:** `SeriesClassification` (linear probe) and `ForecastingBenchmark` (linear head).
- **Toy task:** `SeriesClassificationTasks` with 10% labels; `SeasonalSeries`.
- **Baselines:** raw window + logistic regression; SimCLR on series crops (`SelfSupervisedLosses`); 8.2.9 ROCKET; supervised TCN.
- **Expected metric:** linear-probe accuracy at 10% labels; forecasting MSE. Hypothesis: beats SimCLR and raw windows at low label rates, ties ROCKET.
- **Source:** [arXiv:2106.10466](https://arxiv.org/abs/2106.10466)

### 8.2.11 Deep SVDD and Deep SAD (Ruff et al., ICML 2018, ICLR 2020)
Deep SVDD trains a bias-free network to map normal data into a small hypersphere around a fixed centre c, minimising the mean ‖φ(x) − c‖²; the anomaly score is that distance. Deep SAD adds a few labelled anomalies, whose distances are pushed up with an inverse-distance term, a semi-supervised variant that helps when a few anomalies are known.
- **Block:** `experimental.anomaly.DeepSvdd` (with `semiSupervised` for Deep SAD).
- **Experiment:** `experiments.timeseries.AnomalyBenchmark` (tabular and image rows).
- **Toy task:** the `ForestBenchmarks` anomaly task; one-class `ShapeImages` (one shape normal, others anomalous); windows of `AnomalySeries`.
- **Baselines:** `IsolationForest` (AUROC 0.967), the autoencoder (0.994).
- **Expected metric:** AUROC, and AUROC with 1% labelled anomalies. Hypothesis: Deep SVDD alone ties the autoencoder; Deep SAD clearly beats both with a few labels.
- **Source:** [Deep SVDD, PMLR 80 (2018)](https://proceedings.mlr.press/v80/ruff18a.html), [arXiv:1906.02694](https://arxiv.org/abs/1906.02694)

### 8.2.12 Classical outlier detectors: LOF, one-class SVM and ECOD (Breunig et al., 2000; Schölkopf et al., 2001; Li et al., 2022)
LOF compares each point's local reachability density with its neighbours'; the one-class SVM finds the smallest kernel region holding a ν share of the data; ECOD estimates each feature's empirical CDF and scores a point by its summed tail probabilities, a parameter-free, very fast detector. Only the isolation forest exists as a classical detector.
- **Block:** `experimental.anomaly.LocalOutlierFactor`, `experimental.anomaly.OneClassSvm` (on `KernelSvm`'s solver and `KernelFunction`), `experimental.anomaly.Ecod`.
- **Experiment:** `AnomalyBenchmark`.
- **Toy task:** the `ForestBenchmarks` task, plus local-density anomalies (a sparse cluster next to a dense one) where global methods fail, and high-dimensional noise features.
- **Baselines:** `IsolationForest`, the autoencoder, `KnnIndex` distance to the k-th neighbour.
- **Expected metric:** AUROC and time. Hypothesis: LOF wins on local-density anomalies; ECOD is the fastest and holds up with noise features.
- **Source:** [LOF (SIGMOD 2000)](https://doi.org/10.1145/335191.335388), [one-class SVM (Neural Computation 2001)](https://doi.org/10.1162/089976601750264965), [arXiv:2201.00382](https://arxiv.org/abs/2201.00382)

### 8.2.13 Anomaly Transformer and the point-adjustment pitfall (Xu et al., ICLR 2022; Kim et al., AAAI 2022)
Anomaly Transformer gives each attention layer two branches: a learnable Gaussian "prior association" over nearby time steps and the usual "series association". Normal points attend widely, anomalies mostly to their neighbours, so the KL divergence between the two (the association discrepancy, trained with a min-max strategy) is the anomaly score. Kim et al. showed that the common "point-adjusted F1" (crediting a whole anomalous segment if any point is detected) lets even random scores look state-of-the-art, so this item reports both metrics plus a random-score row.
- **Block:** `experimental.anomaly.AnomalyTransformer`, `experimental.anomaly.SeriesAnomalyMetrics` (point-wise F1, point-adjusted F1, range-based AUC).
- **Experiment:** `AnomalyBenchmark` (series rows).
- **Toy task:** `AnomalySeries`.
- **Baselines:** windowed autoencoder reconstruction error; `IsolationForest` on windows; 8.2.14 matrix profile; random scores.
- **Expected metric:** point-wise F1 and point-adjusted F1. Hypothesis: point adjustment inflates every method including random scores; on point-wise F1 the matrix profile is competitive.
- **Source:** [arXiv:2110.02642](https://arxiv.org/abs/2110.02642), [arXiv:2109.05257](https://arxiv.org/abs/2109.05257)

### 8.2.14 Matrix profile discords: STOMP (Yeh et al., ICDM 2016; Zhu et al., 2016)
The matrix profile stores, for every subsequence of length m, the z-normalised Euclidean distance to its nearest non-trivial match elsewhere in the series; STOMP computes it exactly in O(n²) using the sliding dot-product recurrence. Its maxima are discords (the most unusual subsequences, i.e. anomalies) and its minima are motifs. It is parameter-light (only m) and training-free.
- **Block:** `experimental.anomaly.MatrixProfile` (STOMP, discords, motifs).
- **Experiment:** `AnomalyBenchmark`.
- **Toy task:** `AnomalySeries` collective anomalies; motif discovery in planted repeated patterns.
- **Baselines:** `IsolationForest` on windows; windowed autoencoder; 1-NN `DynamicTimeWarping` distances.
- **Expected metric:** top-k discord hit rate and point-wise F1. Hypothesis: best on collective anomalies with no training at all.
- **Source:** [The UCR Matrix Profile page (Yeh et al. 2016; Zhu et al. 2016)](https://www.cs.ucr.edu/~eamonn/MatrixProfile.html)

### 8.2.15 Change-point detection: Bayesian online (BOCPD) and PELT (Adams & MacKay, 2007; Killick et al., 2012)
BOCPD keeps a posterior over the "run length" since the last change; at each step it updates every run length's predictive probability (with a conjugate model) and a hazard rate, so changes are detected online with uncertainty. PELT finds the exact offline segmentation minimising cost + penalty per change in near-linear time by pruning candidate change points that can never be optimal.
- **Block:** `experimental.anomaly.BayesianOnlineChangePoint`, `experimental.anomaly.Pelt`.
- **Experiment:** `experiments.timeseries.ChangePoints`.
- **Toy task:** `AnomalySeries` with labelled mean, variance and seasonality changes.
- **Baselines:** thresholded Kalman innovations (`StateSpaceFilter`); `HiddenMarkovModel` Viterbi segmentation; the coin-bias change-point model of `ProbabilisticPrograms` (single change only).
- **Expected metric:** F1 within a 5-step tolerance and mean detection delay. Hypothesis: PELT is the most accurate offline; BOCPD detects variance changes that the Kalman innovation test misses.
- **Source:** [arXiv:0710.3742](https://arxiv.org/abs/0710.3742), [arXiv:1101.1438](https://arxiv.org/abs/1101.1438)

### 8.2.16 Learnable audio front ends: SincNet, LEAF and PCEN (Ravanelli & Bengio, 2018; Zeghidour et al., 2021; Wang et al., 2017) — follow-up
Wave 7's recognisers all failed at 0–10 dB SNR (DER 0.55 or worse) with fixed log-mel inputs. SincNet's first layer is a bank of band-pass filters parameterised only by their two cut-off frequencies (sinc functions), so it learns a filterbank with few parameters. LEAF learns Gabor filters, Gaussian lowpass pooling and per-channel energy normalisation (PCEN), an adaptive gain control (a smoothed AGC followed by root compression) that suppresses stationary noise far better than a log.
- **Block:** `experimental.timeseries.SincFilterbank`, `experimental.timeseries.LearnableFrontEnd` (Gabor + pooling + PCEN); PCEN also as a drop-in after core `Signal` mel.
- **Experiment:** `audio.SpokenDigitRecognition` (noise sweep 0/5/10/20 dB).
- **Toy task:** `TalkingDigits` isolated words and strings.
- **Baselines:** log-mel (core `Signal`), `Mfcc`, the frame-wise MLP and Conformer encoders of wave 7.
- **Expected metric:** DER per SNR. Hypothesis: PCEN alone cuts the 0–10 dB DER the most; the learned filterbanks help a little more.
- **Source:** [arXiv:1808.00158](https://arxiv.org/abs/1808.00158), [arXiv:2101.08596](https://arxiv.org/abs/2101.08596), [arXiv:1607.05666](https://arxiv.org/abs/1607.05666)

### 8.2.17 The wavelet scattering transform (Mallat, 2012; Bruna & Mallat, 2013; Andén & Mallat, 2014)
A fixed cascade: wavelet convolution, complex modulus, then low-pass averaging, repeated for two orders. First-order coefficients resemble mel spectra; second-order coefficients recover the amplitude modulations that averaging destroys. The result is translation-invariant and stable to small deformations, with no learned parameters, which suits small data.
- **Block:** `experimental.timeseries.ScatteringTransform` (Morlet filter banks, 1-D and 2-D).
- **Experiment:** `SeriesClassification`, `audio.SpokenDigitRecognition`, and a `ShapeImages` low-data row in `vision.VisionShootout`.
- **Toy task:** CBF; spoken digits with 20 examples per class; 64-image `ShapeImages`.
- **Baselines:** `Mfcc`, log-mel, 8.2.9 ROCKET, the `VisionShootout` CNN.
- **Expected metric:** accuracy against training-set size. Hypothesis: scattering + linear classifier beats MFCC and the CNN at the smallest sizes; the gap closes with data.
- **Source:** [arXiv:1203.1513](https://arxiv.org/abs/1203.1513), [arXiv:1304.6763](https://arxiv.org/abs/1304.6763)

### 8.2.18 Dual-path RNN separation at a learning-curve budget (Luo et al., ICASSP 2020) — follow-up
Wave 7's Conv-TasNet was under-trained (SI-SNRi 2.1 dB vs 4.8 dB for an STFT-mask MLP at 150 steps). DPRNN splits the encoded mixture into overlapping chunks and alternates an intra-chunk RNN (local structure) with an inter-chunk RNN (global structure), modelling long sequences with short recurrences and far fewer parameters than a TCN. The follow-up trains DPRNN and Conv-TasNet to a learning-curve budget (150/600/1500 steps) so the earlier negative is tested fairly.
- **Block:** `experimental.timeseries.DualPathRnn` (reuses `ConvTasNet`'s encoder/decoder and `PermutationInvariantLoss`).
- **Experiment:** `audio.SourceSeparation`.
- **Toy task:** two-speaker `TalkingDigits` mixtures at 0/5 dB.
- **Baselines:** `ConvTasNet` (2.1 dB at 150 steps), the STFT-mask MLP with PIT (4.8 dB), FastICA on two microphones, the ideal ratio mask.
- **Expected metric:** SI-SNR improvement against steps. Hypothesis: both time-domain models pass the mask MLP by 1500 steps, DPRNN first.
- **Source:** [arXiv:1910.06379](https://arxiv.org/abs/1910.06379), [arXiv:1809.07454](https://arxiv.org/abs/1809.07454)

## WP3. Scientific machine learning: PINNs, neural operators, learned simulators, equation discovery, equivariant and molecular models, data assimilation (17)

Target: a new `experimental.sciml` package, measured in a new `experiments.sciml` theme: `PdeShootout` (8.3.1–8.3.3, 8.3.17), `OperatorLearning` (8.3.4–8.3.7), `DynamicsLearning` (8.3.8–8.3.13), `SymmetricModels` (8.3.14–8.3.15) and `DataAssimilation` (8.3.16). Baselines: `FourierNeuralOperator1d`, `NeuralOde` / `OdeSolver`, `KanLayer`, `SymbolicRegression`, `EchoStateReservoir` and NG-RC, EGNN, `StateSpaceFilter`, `DeepAr` and plain MLPs, at equal parameters and steps over at least 3 seeds; second-order optimisers (`Lbfgs`) where the papers use them. Two new generators come first:

- `experimental.sciml.PdeProblems`: reference solvers (finite differences and a pseudo-spectral FFT solver) for 1-D heat, advection, viscous Burgers, Allen–Cahn and KdV, 1-D/2-D Poisson, with random initial conditions and coefficient fields, giving train/test pairs for operators and ground truth for PINNs.
- `experimental.sciml.MechanicalSystems`: pendulum, mass–spring, double pendulum, damped oscillators, Lotka–Volterra, Lorenz-63 and Lorenz-96, n particles on springs, and Lennard-Jones clusters (energies and forces), all with exact integrators.

Every PINN item needs second derivatives of the network output with respect to its inputs; the core autograd's higher-order support (`Var`, `HigherOrderTest`) provides them, and `DualTensor` forward mode can be compared for speed.

### 8.3.1 Physics-informed neural networks (Raissi et al., 2019)
A coordinate MLP u_θ(x, t) is trained so that the PDE residual (for example u_t + u·u_x − ν·u_xx for Burgers), computed by automatic differentiation at random collocation points, is zero, together with initial- and boundary-condition losses; no solution data is needed. The usual recipe is Adam, then L-BFGS. The repo has none of this.
- **Block:** `experimental.sciml.PhysicsInformedNetwork` (PDE residuals as lambdas over autograd derivatives), `experimental.sciml.CollocationSampler`.
- **Experiment:** `experiments.sciml.PdeShootout`.
- **Toy task:** `PdeProblems` 1-D heat, Poisson, Burgers (ν = 0.01/π) and Allen–Cahn.
- **Baselines:** the reference solver (error floor); a data-fit MLP trained on 100 noisy solution samples; the same PINN trained with Adam only.
- **Expected metric:** relative L² error against the reference, and time. Hypothesis: the PINN solves heat and Poisson to < 1%, struggles on Burgers' shock and fails on Allen–Cahn without the fixes of 8.3.2.
- **Source:** [arXiv:1711.10561](https://arxiv.org/abs/1711.10561), [JCP 2019](https://doi.org/10.1016/j.jcp.2018.10.045)

### 8.3.2 PINN training fixes: self-adaptive weights, NTK loss balancing, causal training and residual-based adaptive sampling (McClenny & Braga-Neto, 2023; Wang et al., 2022, 2024; Wu et al., 2023)
PINNs fail mostly from optimisation, not capacity. Four fixes: SA-PINN gives each collocation point a trainable weight that is maximised (gradient ascent) while the network minimises, focusing on stubborn points; NTK balancing sets each loss term's weight from the trace of its neural-tangent-kernel block; causal training weights the residual at time t by exp(−ε·Σ residuals before t), so the network solves early times first; RAR-D resamples collocation points with probability proportional to the residual.
- **Block:** `experimental.sciml.PinnTraining` (`selfAdaptive`, `ntkBalancing`, `causalWeights`, `residualResampling`).
- **Experiment:** `PdeShootout`, one row per fix and all together.
- **Toy task:** Allen–Cahn and Burgers from 8.3.1; advection at high speed (a known failure case).
- **Baselines:** 8.3.1 plain PINN (Adam + L-BFGS).
- **Expected metric:** relative L² error. Hypothesis: causal training is what fixes Allen–Cahn and advection; RAR-D helps most on Burgers' shock.
- **Source:** [arXiv:2009.04544](https://arxiv.org/abs/2009.04544), [arXiv:2007.14527](https://arxiv.org/abs/2007.14527), [arXiv:2203.07404](https://arxiv.org/abs/2203.07404), [arXiv:2207.10289](https://arxiv.org/abs/2207.10289)

### 8.3.3 PirateNets and deep physics-informed KANs (Wang et al., 2024; 2025)
PirateNets make deep PINNs trainable: random Fourier-feature embeddings feed residual blocks whose skip connections are gated by a trainable α initialised at zero, so the network starts as a shallow model and grows depth during training; the last layer is initialised by least squares to fit the initial condition. The 2025 follow-up builds deep physics-informed Kolmogorov–Arnold networks with the same residual-gating idea and reports they train stably where plain deep PIKANs fail.
- **Block:** `experimental.sciml.PirateNet`, `experimental.sciml.PhysicsInformedKan` (on `KanLayer`).
- **Experiment:** `PdeShootout`, depth sweep 2/8/16 blocks.
- **Toy task:** Allen–Cahn, KdV and Burgers.
- **Baselines:** 8.3.1 MLP PINN at matched depth and parameters (with the 8.3.2 fixes), shallow `KanLayer` PINN.
- **Expected metric:** relative L² error against depth. Hypothesis: the plain MLP PINN gets worse with depth; PirateNet and the gated PIKAN get better.
- **Source:** [arXiv:2402.00326](https://arxiv.org/abs/2402.00326), [arXiv:2510.23501](https://arxiv.org/abs/2510.23501)

### 8.3.4 DeepONet (Lu et al., 2021)
An operator network: a branch net encodes the input function sampled at fixed sensor points, a trunk net encodes the query coordinate y, and the output is the dot product of the two embeddings, G(u)(y) ≈ Σ_k b_k(u)·t_k(y). It can be queried at any point, unlike grid-based operators.
- **Block:** `experimental.sciml.DeepOnet`.
- **Experiment:** `experiments.sciml.OperatorLearning`.
- **Toy task:** `PdeProblems`: antiderivative operator; Burgers initial condition → solution at t = 1; Darcy-like 1-D coefficient → solution.
- **Baselines:** `FourierNeuralOperator1d`; an MLP from grid to grid.
- **Expected metric:** relative L² test error, and error at query points off the training grid. Hypothesis: FNO wins on-grid; DeepONet wins off-grid and on irregular sensors.
- **Source:** [arXiv:1910.03193](https://arxiv.org/abs/1910.03193)

### 8.3.5 Transolver physics-attention and the linear attention neural operator (Wu et al., ICML 2024; 2025)
Transolver softly assigns the N mesh points to M ≪ N learned "slices" (physical states), averages each slice into a token, runs attention among the M tokens, and scatters the result back to the points, so attention cost is linear in N and geometry-agnostic. "Transolver is a linear transformer" (AAAI 2026) and LANO (2025) recast this as linear attention with a few agent tokens and report better accuracy at the same cost.
- **Block:** `experimental.sciml.PhysicsAttention` (`transolver`, `lano`).
- **Experiment:** `OperatorLearning`.
- **Toy task:** as 8.3.4, plus 1-D problems on irregular, randomly sampled points.
- **Baselines:** `FourierNeuralOperator1d`; 8.3.4 DeepONet; `LinearAttention` over all points.
- **Expected metric:** relative L² error and time per sample against N. Hypothesis: physics-attention matches FNO on regular grids and beats it on irregular points.
- **Source:** [arXiv:2402.02366](https://arxiv.org/abs/2402.02366), [arXiv:2510.16816](https://arxiv.org/abs/2510.16816)

### 8.3.6 Message-passing PDE solvers with the pushforward trick (Brandstetter et al., ICLR 2022)
The PDE grid is treated as a graph; an encoder–processor–decoder GNN predicts the next K time steps at once (temporal bundling), conditioned on PDE coefficients. The pushforward trick trains on inputs produced by the model's own one-step prediction (gradients stopped through the first step), so the model learns to correct the distribution shift of its own rollouts.
- **Block:** `experimental.sciml.MessagePassingPdeSolver` (reuses `experimental.graph` helpers).
- **Experiment:** `OperatorLearning` (rollout rows).
- **Toy task:** Burgers and KdV autoregressive rollouts of 100–200 steps.
- **Baselines:** autoregressive `FourierNeuralOperator1d`; the same GNN without pushforward or bundling; the reference solver at coarse resolution.
- **Expected metric:** rollout error over time and "survival time" until correlation < 0.8. Hypothesis: pushforward doubles survival time; bundling adds less.
- **Source:** [arXiv:2202.03376](https://arxiv.org/abs/2202.03376)

### 8.3.7 PDE-Refiner: diffusion-style refinement for long rollouts (Lippe et al., NeurIPS 2023)
A one-step neural solver neglects low-amplitude high-frequency components, and these errors grow over a rollout. PDE-Refiner predicts the next state, then refines it for a few steps: noise of decreasing scale is added and the network predicts it, as in a denoising diffusion model, which forces it to model all frequency bands.
- **Block:** `experimental.sciml.PdeRefiner` (reuses `experimental.nn.diffusion` noise schedules).
- **Experiment:** `OperatorLearning` (rollout rows).
- **Toy task:** KdV and Kuramoto–Sivashinsky (added to `PdeProblems`) rollouts.
- **Baselines:** autoregressive FNO; 8.3.6 MP-PDE; `DiffusionForcing` on the same rollouts.
- **Expected metric:** correlation time and spectrum error by frequency band. Hypothesis: longer correlation time than FNO at 3–4× inference cost.
- **Source:** [arXiv:2308.05732](https://arxiv.org/abs/2308.05732)

### 8.3.8 Hamiltonian and Lagrangian neural networks (Greydanus et al., 2019; Cranmer et al., 2020)
An HNN learns a scalar H_θ(q, p) and predicts dynamics by Hamilton's equations, q̇ = ∂H/∂p, ṗ = −∂H/∂q, so the learned flow conserves the learned energy exactly; integrated with a symplectic leapfrog, long rollouts do not drift. An LNN learns a Lagrangian L_θ(q, q̇) and gets accelerations from the Euler–Lagrange equations (a Hessian solve), so it works in arbitrary coordinates without canonical momenta.
- **Block:** `experimental.sciml.HamiltonianNetwork`, `experimental.sciml.LagrangianNetwork`, `experimental.sciml.SymplecticIntegrator` (leapfrog, Störmer–Verlet).
- **Experiment:** `experiments.sciml.DynamicsLearning`.
- **Toy task:** `MechanicalSystems` mass–spring, pendulum and double pendulum from noisy trajectories.
- **Baselines:** MLP predicting derivatives (`NeuralOde`); the same with RK4.
- **Expected metric:** rollout MSE and relative energy drift over 1,000 steps. Hypothesis: HNN/LNN energy drift is orders of magnitude below the MLP's; on the damped oscillator (non-conservative) the HNN is wrong.
- **Source:** [arXiv:1906.01563](https://arxiv.org/abs/1906.01563), [arXiv:2003.04630](https://arxiv.org/abs/2003.04630)

### 8.3.9 SINDy: sparse identification of nonlinear dynamics (Brunton et al., 2016)
Build a library Θ(X) of candidate terms (polynomials, trigonometric functions) evaluated on the measured states, estimate derivatives Ẋ by finite differences (or total-variation regularisation for noisy data), and solve Ẋ = Θ(X)Ξ for a sparse Ξ by sequentially thresholded least squares, so the governing equations come out in closed form.
- **Block:** `experimental.sciml.Sindy` (STLSQ, library builder, noise-robust derivatives).
- **Experiment:** `DynamicsLearning` (equation-discovery rows; also in `synthesis.SymbolicRegressionBenchmark`).
- **Toy task:** Lorenz-63, Lotka–Volterra and the damped oscillator at noise levels 0–5%.
- **Baselines:** `SymbolicRegression` (genetic programming, 4 of 4 equations with tournament selection); an MLP derivative model.
- **Expected metric:** exact structure recovery (share of correct terms), coefficient error, fit time. Hypothesis: SINDy recovers all three systems at ≤ 1% noise in milliseconds; GP is slower but more robust when the library lacks a term.
- **Source:** [arXiv:1509.03580](https://arxiv.org/abs/1509.03580)

### 8.3.10 Koopman methods: DMD, EDMD and deep Koopman autoencoders (Schmid, 2010; Williams et al., 2015; Lusch et al., 2018)
Koopman theory says a nonlinear system evolves linearly on a space of observables. DMD fits the best linear map between consecutive snapshots via a truncated SVD and gives its eigenvalues (frequencies, growth rates) and modes; EDMD does the same on a dictionary of nonlinear observables; the deep Koopman autoencoder learns the observables with an encoder, a linear latent step K and a decoder, trained on reconstruction, linear-prediction and multi-step losses.
- **Block:** `experimental.sciml.DynamicModeDecomposition` (`dmd`, `edmd`; uses `ThinSvd`), `experimental.sciml.KoopmanAutoencoder`.
- **Experiment:** `DynamicsLearning` (forecast rows, shared with `exotic.ReservoirForecast`).
- **Toy task:** pendulum (nonlinear but periodic), Van der Pol limit cycle, Lorenz-63 short-term, and a 1-D advection–diffusion field.
- **Baselines:** `EchoStateReservoir`, NG-RC, MLP one-step model.
- **Expected metric:** N-step forecast error and valid-prediction time; spectrum error on the periodic systems. Hypothesis: DMD is exact on the linear PDE; the Koopman autoencoder beats NG-RC on Van der Pol but not on Lorenz.
- **Source:** [DMD (JFM 2010)](https://doi.org/10.1017/S0022112010001217), [arXiv:1408.4408](https://arxiv.org/abs/1408.4408), [arXiv:1712.09707](https://arxiv.org/abs/1712.09707)

### 8.3.11 Universal differential equations: known physics plus a learned residual (Rackauckas et al., 2020)
A UDE writes the right-hand side as the part of the physics that is known plus a neural network for the unknown part, ẋ = f_known(x; p) + NN_θ(x), trains both through the ODE solver, and can then pass the learned term to SINDy (8.3.9) to recover a closed form. Data efficiency is the point: the network only has to learn what the physics misses.
- **Block:** `experimental.sciml.UniversalDifferentialEquation` (on `NeuralOde`).
- **Experiment:** `DynamicsLearning` (data-efficiency rows).
- **Toy task:** Lotka–Volterra with an unknown interaction term; a pendulum with unknown friction; training sets of 5–100 points.
- **Baselines:** a pure `NeuralOde`; the known physics alone (fitted parameters, no network).
- **Expected metric:** extrapolation error beyond the training window against training-set size, and recovery of the missing term after SINDy. Hypothesis: the UDE extrapolates with 10× less data than the pure neural ODE.
- **Source:** [arXiv:2001.04385](https://arxiv.org/abs/2001.04385)

### 8.3.12 Neural SDEs and latent SDEs (Li et al., 2020; Kidger et al., 2021)
A neural SDE has learned drift and diffusion networks, dX = f_θ(X)dt + g_θ(X)dW, integrated by Euler–Maruyama or a Milstein step. The latent SDE of Li et al. trains a posterior SDE against a prior SDE by a pathwise KL (an ELBO), with gradients through the solver (adjoint or direct); it models irregularly sampled stochastic series with calibrated uncertainty. The repo integrates SDEs only for diffusion sampling, not as learned models.
- **Block:** `experimental.sciml.NeuralSde`, `experimental.sciml.LatentSde`, `experimental.sciml.SdeSolver`.
- **Experiment:** `DynamicsLearning` (stochastic rows).
- **Toy task:** Ornstein–Uhlenbeck, a double-well potential with noise-driven switching, and irregularly sampled `SeasonalSeries`.
- **Baselines:** `NeuralOde` (deterministic), `DeepAr`, the quantile MLP.
- **Expected metric:** NLL, 1-D Wasserstein distance between simulated and true marginals (`EntropicTransport`), switching-rate error. Hypothesis: only the SDE reproduces the double-well switching rate.
- **Source:** [arXiv:2001.01328](https://arxiv.org/abs/2001.01328), [arXiv:2102.03657](https://arxiv.org/abs/2102.03657)

### 8.3.13 Graph network simulators (Sanchez-Gonzalez et al., ICML 2020)
Particles are nodes connected within a radius; an encoder–processor–decoder GNN with 10 or so message-passing steps predicts accelerations, which are integrated by a semi-implicit Euler step. Training adds random-walk noise to input positions so the model learns to correct its own rollout errors; it generalises to more particles than seen in training.
- **Block:** `experimental.sciml.GraphNetworkSimulator`.
- **Experiment:** `DynamicsLearning` (particle rows).
- **Toy task:** `MechanicalSystems` n particles on springs with walls; train on 8 particles, test on 16.
- **Baselines:** EGNN (`EquivariantGraphLayer`); an MLP on all particle states (fixed n); GNS without input noise.
- **Expected metric:** rollout MSE over 100 steps, at train n and at 2n. Hypothesis: input noise is worth most of the rollout gain; the MLP cannot change n at all.
- **Source:** [arXiv:2002.09405](https://arxiv.org/abs/2002.09405)

### 8.3.14 Symmetry by construction: frame averaging and equivariant MLPs (Puny et al., ICLR 2022; Finzi et al., ICML 2021)
Frame averaging makes any backbone equivariant: compute a small input-dependent set of frames (for E(3), from the PCA axes of the point cloud with sign ambiguities, at most 8 frames), canonicalise the input in each frame, run the network and average the outputs mapped back. EMLP instead solves for the linear layers that commute with a given group's representation (a null-space computation) for arbitrary matrix groups, such as O(3), SO(2) or Lorentz.
- **Block:** `experimental.sciml.FrameAveraging`, `experimental.sciml.EquivariantMlp` (constraint solver on `SymmetricEigen`/`ThinSvd`).
- **Experiment:** `experiments.sciml.SymmetricModels` (next to `exotic.EquivariantPointClouds`).
- **Toy task:** O(3)-invariant regression (the inertia-tensor and three-body energy tasks of the EMLP paper) and n-body spring rollouts.
- **Baselines:** EGNN; an MLP with and without random-rotation augmentation.
- **Expected metric:** test MSE on rotated inputs, and data needed to reach a target error. Hypothesis: both are exactly equivariant and match EGNN; augmentation needs ~10× more data.
- **Source:** [arXiv:2110.03336](https://arxiv.org/abs/2110.03336), [arXiv:2104.09459](https://arxiv.org/abs/2104.09459)

### 8.3.15 SchNet: continuous-filter convolutions with force matching (Schütt et al., 2017)
Atoms are nodes; each interaction block expands pair distances in radial basis functions, maps them by a filter network to per-pair filters, and multiplies them with neighbour features (a continuous-filter convolution), which is rotation-invariant by construction. Forces are the negative gradient of the predicted energy with respect to positions (autograd), and training on energies plus forces (force matching) is far more data-efficient than energies alone.
- **Block:** `experimental.sciml.ContinuousFilterConv`, `experimental.sciml.SchNet`.
- **Experiment:** `SymmetricModels` (molecular rows).
- **Toy task:** `MechanicalSystems` Lennard-Jones clusters of 6–13 atoms: energy and force regression, then a short molecular-dynamics run with the learned potential.
- **Baselines:** EGNN; an MLP on sorted pair distances; SchNet trained on energies only.
- **Expected metric:** energy MAE, force MAE and MD energy drift. Hypothesis: force matching cuts energy error at equal data; SchNet ≈ EGNN on energies.
- **Source:** [arXiv:1706.08566](https://arxiv.org/abs/1706.08566)

### 8.3.16 Data assimilation: ensemble Kalman filter and score-based data assimilation (Evensen, 1994; Rozet & Louppe, NeurIPS 2023)
The EnKF propagates an ensemble through a nonlinear model and updates every member with a Kalman gain estimated from the ensemble covariance (with localisation and inflation), which scales to high dimensions where the EKF and particle filter do not. Score-based data assimilation trains a diffusion model on short windows of trajectories and samples whole posterior trajectories given sparse, noisy observations by guiding the score with the observation likelihood, without needing the dynamics at inference.
- **Block:** `experimental.sciml.EnsembleKalmanFilter`, `experimental.sciml.ScoreBasedAssimilation` (reuses `experimental.nn.diffusion` denoisers).
- **Experiment:** `experiments.sciml.DataAssimilation` (next to `classic.TrackingFilters`).
- **Toy task:** Lorenz-96 with 40 variables, observing every 4th variable with noise; Lorenz-63.
- **Baselines:** `StateSpaceFilter` EKF and particle filter; no assimilation (free run).
- **Expected metric:** state RMSE over time and ensemble spread–error ratio. Hypothesis: the particle filter collapses on Lorenz-96 and the EnKF does not; SDA matches the EnKF without access to the model equations.
- **Source:** [Evensen 1994 (JGR)](https://doi.org/10.1029/94JC00572), [arXiv:2306.10574](https://arxiv.org/abs/2306.10574)

### 8.3.17 Hard constraints: divergence-free fields and neural conservation laws (Richter-Powell et al., NeurIPS 2022)
Instead of penalising a constraint, build it in: a 2-D velocity field parameterised as the curl of a learned stream function, u = (∂ψ/∂y, −∂ψ/∂x), is divergence-free for any ψ; more generally, a density and flux derived from a learned antisymmetric field satisfy the continuity equation ∂ρ/∂t + ∇·J = 0 exactly. The comparison is with soft penalties, which only drive the violation small.
- **Block:** `experimental.sciml.ConstrainedFields` (`streamFunction`, `conservationLaw`).
- **Experiment:** `PdeShootout` (constraint rows).
- **Toy task:** reconstruct an incompressible 2-D flow (Taylor–Green vortices plus noise) from sparse velocity samples; a 1-D advected density.
- **Baselines:** an MLP with a soft divergence penalty (weights 0.1–10); an unconstrained MLP; a `FourierFeatures` MLP.
- **Expected metric:** velocity error and maximum divergence. Hypothesis: the hard constraint has zero divergence and lower error with few samples; the penalty trades accuracy against violation.
- **Source:** [arXiv:2210.01741](https://arxiv.org/abs/2210.01741)

## WP4. LLM serving and inference, tokenisation, long context, agents and tool use (18)

Target: two new packages, `experimental.serving` (KV-cache approximation, speculative decoding, scheduling and adapter serving) and `experimental.agent` (agent loops, context tools, multi-agent orchestration). Tokenizers go in `experimental.data` next to `BpeDropoutTokenizer` and `SuperBpeTokenizer`; long-context blocks in `experimental.llm`. Measurements go in:

- a new `experiments.serving.ServingShootout` (8.4.1–8.4.9; it reuses `language.ServingThroughput`'s request generator and must keep that experiment's rows reproducible);
- `experiments.context.LongContextNeedle` and `HeadBudgetComparison` (extended, 8.4.10–8.4.11);
- a new `experiments.language.TokenizerShootout` (8.4.12–8.4.14), next to `DynamicChunking`;
- a new `experiments.agents.ToolAgentBench` (8.4.15–8.4.18).

Baselines: `SnapKv`, heavy hitters (`EvictingKvCache.heavyHitters`), `QuestKvStore`, `DuoHeadCache`, `QuantizedKvCache` (KIVI), `CrossLayerKvBlock`, `EagleDraft`/`Eagle3Draft`, `SequenceDistillation`, `ChunkedPrefillScheduler`, `PrefixCache`, `LoraAdapterDense`, `GrammarConstraint`, BPE/Unigram/SuperBPE/BPE-dropout, `VirtualContextPager`, `MemoryOverwriteAgent`, `RecursiveLm`, `ToolRuntime` + `ToolformerFilter`, `GrpoTrainer`, `AnswerDebate` and `MultiturnRewardTrainer`, at equal compute over at least 3 seeds. All "LLMs" here are the repo's toy LMs (`TinyLm`, `WindowLm`, the toy Llama of `PagedLlama`).

### 8.4.1 Draft-based approximate inference: SpecKV and SpecPC (2025; ICLR 2026)
KV eviction and prompt compression usually estimate token importance from the target model's own attention to the prompt, which cannot see what the future output will need. SpecKV first generates a short draft answer with a small draft model, then scores every prompt KV entry by the attention the draft's tokens pay to it (in the target model), and keeps the top entries; SpecPC uses the draft model's attention to drop whole prompt tokens before the target model runs at all.
- **Block:** `experimental.serving.DraftGuidedKvSelector` (`specKv`, `specPc`).
- **Experiment:** `ServingShootout` (approximate-inference rows), plus `context.HeadBudgetComparison`.
- **Toy task:** the `LongContextNeedle` key–value retrieval and `CartridgeCorpusQa` questions at KV budgets of 10–50%.
- **Baselines:** `SnapKv`, heavy hitters, `QuestKvStore`, the full cache.
- **Expected metric:** answer accuracy against KV budget, and extra draft cost. Hypothesis: SpecKV beats SnapKV at the smallest budgets because the draft "knows" the question's answer region.
- **Source:** [arXiv:2506.08373](https://arxiv.org/abs/2506.08373)

### 8.4.2 LLMLingua: budgeted prompt compression by a small LM (Jiang et al., EMNLP 2023; LongLLMLingua, ACL 2024)
A small LM computes the perplexity of every prompt token; tokens it finds predictable carry little information and are dropped, under a budget controller that gives instructions and questions a larger budget than demonstrations, and with iterative token-level compression conditioned on what was kept. LongLLMLingua makes the scores question-aware (contrastive perplexity given the question) and reorders documents. Unlike KV eviction, the compressed prompt is plain text, so it works with any target model.
- **Block:** `experimental.serving.PerplexityPromptCompressor`.
- **Experiment:** `ServingShootout` (prompt-compression rows, shared with 8.4.1).
- **Toy task:** as 8.4.1, plus few-shot `ArithmeticTask` prompts with redundant demonstrations.
- **Baselines:** truncation (keep the end); `SnapKv`; `SelfSummarizingContext`; 8.4.1 SpecPC.
- **Expected metric:** accuracy against compression ratio, and prefill time. Hypothesis: question-aware compression beats truncation clearly; SpecPC beats LLMLingua because its draft is closer to the target.
- **Source:** [arXiv:2310.05736](https://arxiv.org/abs/2310.05736), [arXiv:2310.06839](https://arxiv.org/abs/2310.06839)

### 8.4.3 MiniCache: merging the KV cache across adjacent layers (Liu et al., NeurIPS 2024)
In the middle-to-deep layers, the KV states of adjacent layers are very similar in direction. MiniCache stores, for each pair of layers, one shared direction obtained by SLERP between the two layers' normalised states plus each layer's magnitude, and keeps unmergeable "retention" tokens (those with large angular distance) separately. It is training-free, unlike cross-layer attention (CLA), which must be trained in.
- **Block:** `experimental.serving.CrossLayerKvMerger`.
- **Experiment:** `ServingShootout` (KV-memory rows).
- **Toy task:** the toy Llama on `LongContextNeedle` and validation loss on `SyntheticCorpus`.
- **Baselines:** the full cache; `CrossLayerKvBlock` (CLA, trained); KIVI 2-bit (`QuantizedKvCache`); MiniCache + KIVI combined.
- **Expected metric:** KV bytes per token against validation loss and recall. Hypothesis: training-free merging halves the deep layers' KV for a small loss increase; trained CLA is better at equal memory.
- **Source:** [arXiv:2405.14366](https://arxiv.org/abs/2405.14366)

### 8.4.4 MagicPIG: sampling attention with locality-sensitive hashing (Chen et al., ICLR 2025)
Top-k attention is biased when attention is not very sparse. MagicPIG instead samples keys with SimHash LSH tables (keys colliding with the query in enough tables are sampled), and forms an unbiased importance-sampled estimate of the attention output using the known collision probabilities; the hash tables can live in CPU memory.
- **Block:** `experimental.serving.LshSampledAttention`.
- **Experiment:** `ServingShootout` and `context.HeadBudgetComparison`.
- **Toy task:** `LongContextNeedle` and an aggregation task (sum of many values), where attention is dense.
- **Baselines:** exact attention, top-k selection (`QuestKvStore`), `DuoHeadCache`.
- **Expected metric:** output error relative to exact attention, task accuracy, keys touched per query. Hypothesis: top-k wins on the needle task, MagicPIG on the aggregation task.
- **Source:** [arXiv:2410.16179](https://arxiv.org/abs/2410.16179)

### 8.4.5 DistillSpec: distilling the draft for acceptance (Zhou et al., ICLR 2024)
The speedup of speculative decoding depends on how often the target accepts draft tokens, which is set by the total variation distance between draft and target distributions. DistillSpec trains the draft on its own samples (on-policy) with a divergence chosen for acceptance (TVD or a skew of forward and reverse KL), rather than on ground-truth text.
- **Block:** `experimental.serving.SpeculativeDraftDistiller` (reuses `OnPolicyDistiller`).
- **Experiment:** `ServingShootout` (speculative rows, next to `language.Eagle3Speculation`).
- **Toy task:** the toy LM on `SyntheticCorpus` and `ArithmeticTask`, greedy and T = 1 sampling.
- **Baselines:** a draft trained on data only; `SequenceDistillation` (offline sequence-level KD); `EagleDraft`.
- **Expected metric:** acceptance rate and tokens per target pass. Hypothesis: on-policy TVD distillation lifts acceptance most at T = 1, less under greedy.
- **Source:** [arXiv:2310.08461](https://arxiv.org/abs/2310.08461)

### 8.4.6 SpecTr: multi-draft speculative sampling by optimal transport (Sun et al., NeurIPS 2023)
With k independent draft candidates for the next token, the best lossless acceptance rule is an optimal-transport coupling between the draft's k-sample distribution and the target. SpecTr gives a near-optimal, cheap selection rule (k-sequential draft selection), so more drafts raise acceptance while the output distribution stays exactly the target's.
- **Block:** `experimental.serving.MultiDraftVerifier`.
- **Experiment:** `ServingShootout` (speculative rows).
- **Toy task:** as 8.4.5, at k = 1/2/4/8 drafts.
- **Baselines:** single-draft speculative sampling; the EAGLE tree (`EagleDraft`, 2.76 tokens per pass); naive "accept if any draft matches" (to show it biases the distribution).
- **Expected metric:** tokens per target pass, and a distribution test (total variation of 10⁵ samples against the target). Hypothesis: SpecTr gains over one draft with no bias; the naive rule is visibly biased.
- **Source:** [arXiv:2310.15141](https://arxiv.org/abs/2310.15141)

### 8.4.7 Multi-adapter serving: S-LoRA and Punica (Sheng et al., MLSys 2024; Chen et al., MLSys 2024)
Serving many LoRA adapters at once: the base model runs once for the whole batch, and each request's low-rank delta xAᵢBᵢ is applied by a segmented gather matmul (Punica's SGMV) over the requests that use adapter i; S-LoRA pages adapter weights and KV in one unified memory pool. The alternative is merging an adapter into the weights and batching only same-adapter requests.
- **Block:** `experimental.serving.MultiLoraBatcher`.
- **Experiment:** `ServingShootout` (throughput rows).
- **Toy task:** the toy Llama with 4–64 adapters (`LoraAdapterDense`), requests with Zipf-distributed adapter popularity.
- **Baselines:** merged adapters with per-adapter batches; unmerged adapters with no cross-adapter batching.
- **Expected metric:** tokens per second and p95 latency against the number of adapters. Hypothesis: SGMV batching keeps throughput nearly flat in the number of adapters; per-adapter batching collapses beyond ~8.
- **Source:** [arXiv:2311.03285](https://arxiv.org/abs/2311.03285), [arXiv:2310.18547](https://arxiv.org/abs/2310.18547)

### 8.4.8 Fair scheduling with a virtual token counter (Sheng et al., OSDI 2024)
FCFS continuous batching lets a heavy client starve light ones. VTC keeps a counter of weighted service (input and output tokens) per client and always admits the backlogged client with the smallest counter; counters of newly active clients are lifted to the current minimum, so idle time is not banked. It gives provable fairness within a bounded gap while keeping the batch full.
- **Block:** `experimental.serving.FairScheduler` (`vtc`, `fcfs`, `requestRateLimit`).
- **Experiment:** `ServingShootout` (scheduling rows).
- **Toy task:** a simulated multi-client workload on `ChunkedPrefillScheduler`'s engine: one client sending 10× the load of the others.
- **Baselines:** FCFS (`ChunkedPrefillScheduler`); per-client request-rate limiting.
- **Expected metric:** maximum service difference between clients, light-client p95 latency, and total throughput. Hypothesis: VTC bounds the service gap with < 5% throughput loss; rate limiting wastes capacity when the heavy client is alone.
- **Source:** [arXiv:2401.00588](https://arxiv.org/abs/2401.00588)

### 8.4.9 XGrammar: a token-mask cache for grammar-constrained decoding (Dong et al., MLSys 2025)
Checking every vocabulary token against a pushdown automaton at every step is slow. XGrammar splits the vocabulary per grammar position into context-independent tokens (whose validity depends only on the current automaton position, precomputed once into a bitmask cache) and the few context-dependent ones checked at runtime against the stack; it also overlaps mask computation with the model's forward pass.
- **Block:** `experimental.serving.GrammarMaskCache` (on core `Grammar`'s pushdown automaton).
- **Experiment:** `ServingShootout` (structured-output rows, next to `language.DecodingShowdown`).
- **Toy task:** JSON-schema and arithmetic-grammar outputs from the toy LM with byte-level and BPE vocabularies.
- **Baselines:** `GrammarConstraint` and `JsonSchemaLite` (per-token scan).
- **Expected metric:** mask time per token and the share of context-independent tokens; outputs must be identical. Hypothesis: > 10× faster masks with the BPE vocabulary, much less with bytes.
- **Source:** [arXiv:2411.15100](https://arxiv.org/abs/2411.15100)

### 8.4.10 Activation Beacon and gist tokens: learned context compression (Zhang et al., 2024; Mu et al., NeurIPS 2023)
Activation Beacon interleaves special beacon tokens into each chunk of a long input; the beacons' KV activations are trained to summarise the chunk, and only the beacons' KV is kept as the window slides, giving a fixed compression ratio with a fine-tuned plug-in. Gist tokens do the same for a prompt: an attention mask forces later tokens to see the instruction only through k gist tokens, whose KV can then be cached instead of the prompt.
- **Block:** `experimental.llm.BeaconCompressor` (`beacon`, `gist` masking modes).
- **Experiment:** `context.LongContextNeedle`; gist rows in `ServingShootout` next to `PrefixCache`.
- **Toy task:** needle retrieval and running-sum tasks at 4–16× the training window; instruction-following prompts for gists.
- **Baselines:** `CompressiveKvMemory`, `RecurrentMemoryTransformer`, `InfiniAttention`, sliding window; `PrefixCache` (exact) and `SoftPrompts` for gists.
- **Expected metric:** accuracy against compression ratio and memory. Hypothesis: beacons beat compressive memory at 8×; gists keep most of the accuracy at 10–25× prompt compression.
- **Source:** [arXiv:2401.03462](https://arxiv.org/abs/2401.03462), [arXiv:2304.08467](https://arxiv.org/abs/2304.08467)

### 8.4.11 Lost in the middle, and calibrating it away: found-in-the-middle and Ms-PoE (Liu et al., TACL 2024; Hsieh et al., 2024; Zhang et al., 2024)
LMs use information at the start and end of a long context better than in the middle (a U-shaped accuracy curve). Hsieh et al. trace this to a positional attention bias, estimate it by inserting a dummy document at every position, and subtract it from the attention to get calibrated relevance; Ms-PoE rescales position indices with a different ratio per head, training-free, so some heads see the middle at a "nearer" scale.
- **Block:** `experimental.llm.PositionBiasCalibration` (`foundInTheMiddle`, `multiScalePositions`).
- **Experiment:** `context.LongContextNeedle` (position sweep).
- **Toy task:** multi-document key–value QA with the answer document at 10 positions.
- **Baselines:** the uncalibrated model; `RopeScaling` (YaRN) at the same length; `DuoHeadCache`.
- **Expected metric:** accuracy by answer position, and middle-minus-ends gap. Hypothesis: a U-curve appears even in toy LMs, and calibration shrinks the gap by half.
- **Source:** [arXiv:2307.03172](https://arxiv.org/abs/2307.03172), [arXiv:2406.16008](https://arxiv.org/abs/2406.16008), [arXiv:2403.04797](https://arxiv.org/abs/2403.04797)

### 8.4.12 The Length-MAX tokenizer (2025)
BPE picks merges by pair frequency. Length-MAX picks the vocabulary to minimise the average number of tokens per character of the corpus (equivalently, maximise frequency × length of the chosen tokens), solved as a graph-partitioning style optimisation with a greedy approximation; the paper reports fewer tokens per text, faster training and inference, and equal or better downstream accuracy.
- **Block:** `experimental.data.LengthMaxTokenizer`.
- **Experiment:** `experiments.language.TokenizerShootout`.
- **Toy task:** `SyntheticCorpus` and the byte corpus of `DynamicChunking`, vocabulary sizes 128–2048; a toy LM trained on each tokenisation.
- **Baselines:** BPE (bits per byte 0.679 in `DynamicChunking`), `UnigramTokenizer`, `SuperBpeTokenizer`.
- **Expected metric:** tokens per character, bits per byte at equal training FLOPs, and accuracy on `CountingTasks`. Hypothesis: fewer tokens than BPE at equal vocabulary; bits per byte about equal.
- **Source:** [arXiv:2511.20849](https://arxiv.org/abs/2511.20849)

### 8.4.13 Segmentation at inference: PathPiece and optimal versus greedy segmentation (Schmidt et al., EMNLP 2024; Uzan et al., 2024)
With a fixed vocabulary there are many ways to segment a word. PathPiece segments into the minimum number of tokens (a shortest path), and its authors show that fewer tokens does not mean a better LM. Uzan et al. compare inference methods for the same vocabulary (BPE merge order, longest-prefix greedy, least tokens, likelihood-based) and find the choice matters as much as the vocabulary.
- **Block:** `experimental.data.SegmentationStrategies` (`mergeOrder`, `longestPrefix`, `leastTokens`, `maxLikelihood`) as a wrapper around any vocabulary.
- **Experiment:** `TokenizerShootout`.
- **Toy task:** as 8.4.12; morphology probes (planted prefixes and suffixes in `SyntheticCorpus`).
- **Baselines:** each existing tokenizer with its own native segmentation.
- **Expected metric:** bits per byte, tokens per character and morpheme-boundary F1. Hypothesis: least-tokens segmentation lowers token count but not bits per byte, reproducing the paper.
- **Source:** [arXiv:2402.18376](https://arxiv.org/abs/2402.18376), [arXiv:2403.01289](https://arxiv.org/abs/2403.01289)

### 8.4.14 StochasTok: stochastic token splitting for subword understanding (Sims et al., ICLR 2026)
During training, StochasTok randomly splits some tokens into pairs of shorter in-vocabulary tokens that concatenate to the same string (the vocabulary is unchanged), so the model sees how its tokens are spelled. It can be added at pretraining or post-training, and it improves character-level tasks such as letter counting and multi-digit arithmetic. BPE-dropout perturbs the merge process instead.
- **Block:** `experimental.data.StochasticTokenSplitter`.
- **Experiment:** `TokenizerShootout` (subword-understanding rows).
- **Toy task:** letter counting and spelling questions on `SyntheticCorpus` words; `ArithmeticTask` with multi-digit tokens.
- **Baselines:** plain BPE; `BpeDropoutTokenizer`; character-level tokenisation (upper bound).
- **Expected metric:** letter-count and spelling accuracy, arithmetic accuracy, and LM loss. Hypothesis: StochasTok gets most of the character-level gain at no loss cost; BPE-dropout less.
- **Source:** [arXiv:2506.01687](https://arxiv.org/abs/2506.01687)

### 8.4.15 Sculptor: active context management tools (2025)
Instead of paging (MemGPT) or rewriting memory (MemAgent), the model is given tools to manage its own working context: fragment the context into labelled pieces, hide and restore fragments, summarise a fragment in place, and search the hidden ones. Irrelevant or distracting material is hidden early, which counters proactive interference from earlier content.
- **Block:** `experimental.agent.ContextSculptor` (fragment, hide, restore, summarise, search as tool calls on `ToolRuntime`).
- **Experiment:** `experiments.agents.ToolAgentBench` (context rows, next to `context.LongContextNeedle`).
- **Toy task:** `LongContextNeedle` with planted distractor keys that are updated later (the latest value is correct), and multi-hop chains in long contexts.
- **Baselines:** `VirtualContextPager` (MemGPT), `MemoryOverwriteAgent` (MemAgent), `RecursiveLm`, the full context.
- **Expected metric:** accuracy and tokens processed. Hypothesis: wins on the overwritten-key task, where interference hurts the other methods.
- **Source:** [arXiv:2508.04664](https://arxiv.org/abs/2508.04664)

### 8.4.16 RL for tool use: ToolRL and ReTool (Qian et al., 2025; Feng et al., 2025)
Tool use is usually taught by supervised traces (Toolformer). ToolRL trains with GRPO and a structured reward (format, correct tool name, correct arguments, and outcome); ReTool interleaves code-interpreter calls inside the rollout, feeds the results back, and trains on the outcome reward only, so the model learns when to call the tool. Both report that RL generalises better than SFT on traces.
- **Block:** `experimental.agent.ToolUseRewards`, `experimental.agent.ToolInterleavedRollout` (on `GrpoTrainer` and `ToolRuntime`).
- **Experiment:** `ToolAgentBench`.
- **Toy task:** `ArithmeticTask` with large operands where a calculator tool helps, plus questions where calling the tool is wasteful.
- **Baselines:** `ToolformerFilter` SFT; GRPO without tools (`ArithmeticRl`); always-call and never-call policies.
- **Expected metric:** accuracy, tool calls per question and share of unnecessary calls. Hypothesis: RL matches Toolformer accuracy with fewer unnecessary calls, and beats it on held-out operand sizes.
- **Source:** [arXiv:2504.13958](https://arxiv.org/abs/2504.13958), [arXiv:2504.11536](https://arxiv.org/abs/2504.11536)

### 8.4.17 Mixture-of-Agents (Wang et al., 2024)
Several proposer models answer; in each following layer, every agent sees all previous answers and writes an improved one, and a final aggregator synthesises the last layer's answers. Unlike voting it can combine partial solutions, and unlike debate it has no adversarial framing.
- **Block:** `experimental.agent.MixtureOfAgents`.
- **Experiment:** `ToolAgentBench` (orchestration rows, next to `agents.DebatePanel`).
- **Toy task:** `ChainTask` multi-hop questions and `ArithmeticTask` with three differently trained toy LMs (each strong on a different sub-skill).
- **Baselines:** majority vote (0.52 in `DebatePanel`), debate (0.53, `AnswerDebate`), the best single model.
- **Expected metric:** accuracy against total tokens. Hypothesis: MoA beats voting only when the models have complementary skills.
- **Source:** [arXiv:2406.04692](https://arxiv.org/abs/2406.04692)

### 8.4.18 ArCHer: hierarchical multi-turn RL with a turn-level critic (Zhou et al., ICML 2024)
Trajectory-level policy gradients (as in GRPO) give one reward to a whole multi-turn conversation, so credit is assigned poorly. ArCHer trains an off-policy, turn-level critic (Q over whole utterances, with TD learning across turns) and uses its advantages to train the token-level policy within each turn by policy gradient, combining sample-efficient value learning with token-level generation.
- **Block:** `experimental.agent.TurnLevelCritic`, `experimental.agent.HierarchicalAgentTrainer`.
- **Experiment:** `ToolAgentBench` (multi-turn rows, next to `reasoning.ClarifyingQuestions`).
- **Toy task:** 20-questions-style guessing against `SimulatedUser`, and multi-turn tool use where an early wrong call must be corrected.
- **Baselines:** `MultiturnRewardTrainer` (CollabLLM); trajectory-level GRPO; filtered behaviour cloning.
- **Expected metric:** success rate and turns to success against environment interactions. Hypothesis: the turn-level critic reaches GRPO's success with several times fewer interactions.
- **Source:** [arXiv:2402.19446](https://arxiv.org/abs/2402.19446)

## WP5. Compression and pruning beyond the existing set, and compilers/autotuning for kernels (18)

Target: two new packages, `experimental.compress` (pruning schedules and criteria, structured pruning, distillation losses, weight sharing, mixed precision, decompositions, token merging) and `experimental.kernels` (autotuning, fast algorithms, sparse kernels and graph rewriting on the CPU backend). Measurements go in a new `experiments.compression.PruningShootout` (8.5.1–8.5.7), `experiments.compression.CompactModels` (8.5.8–8.5.13) and a new `experiments.kernels.KernelTuning` (8.5.14–8.5.18). Existing compression experiments (`language.CompressionComparison`, `techniques.QuantizationShootout`, `language.SliceCompression`) keep their rows; new rows that reuse their models are added in the new experiments. Baselines: `WeightPruning` (magnitude, Wanda, SparseGPT, n:m), `LotteryTicket`, `RigL`, `BlockImportance`, `SliceCompressor`, `Gptq`, round-to-nearest, `LsqDense`, `ParetoQDense`, `DistillationLoss`, `FeatureDistillationLoss`, `LowRankDense`, `FineGrainedMoe`, `VisionTransformer`, `EvolvedKernels`, and the core CPU backend's dense matmul and convolution, at equal sparsity, bits or FLOPs over at least 3 seeds. Every kernel item reports wall-clock time measured with JMH-style warm-up in the test, plus a correctness check against the core kernel.

### 8.5.1 Gradual magnitude pruning (Zhu & Gupta, 2017)
Instead of pruning once after training, sparsity is raised during training along a cubic schedule, s_t = s_f + (s_i − s_f)(1 − (t − t₀)/(nΔt))³, removing the smallest-magnitude weights every Δt steps while the rest recover. It is the standard strong baseline that many later pruning papers fail to beat.
- **Block:** `experimental.compress.GradualPruningSchedule` (wraps any `WeightPruning` criterion).
- **Experiment:** `experiments.compression.PruningShootout`.
- **Toy task:** `ToyTasks` digits MLP and the `VisionShootout` CNN at 50/80/90/95% sparsity; the toy LM on `SyntheticCorpus`.
- **Baselines:** one-shot magnitude pruning + fine-tuning (`WeightPruning`), `LotteryTicket` (iterative rewinding), `RigL` (dynamic sparse).
- **Expected metric:** accuracy or loss against sparsity at equal total steps. Hypothesis: GMP ≈ RigL ≈ lottery tickets at ≤ 90%, and beats one-shot pruning clearly at 95%.
- **Source:** [arXiv:1710.01878](https://arxiv.org/abs/1710.01878)

### 8.5.2 Movement pruning (Sanh et al., NeurIPS 2020)
During fine-tuning, a weight's importance is how far it moves away from zero, S = −Σ_t W·∂L/∂W (first-order), not its magnitude; soft movement pruning learns scores with a straight-through top-k mask. Magnitude pruning keeps weights that were large in pretraining even when fine-tuning is shrinking them; movement keeps what the new task needs.
- **Block:** `experimental.compress.MovementPruning` (hard and soft variants).
- **Experiment:** `PruningShootout` (transfer rows).
- **Toy task:** pretrain the toy LM on `SyntheticCorpus`, fine-tune and prune on `ArithmeticTask` or a classification head on `ToyTasks` digits from a pretrained MLP.
- **Baselines:** magnitude pruning during fine-tuning; Wanda; 8.5.1 GMP.
- **Expected metric:** fine-tuned accuracy against sparsity. Hypothesis: movement beats magnitude at high sparsity (≥ 90%) in transfer, and loses at low sparsity, as in the paper.
- **Source:** [arXiv:2005.07683](https://arxiv.org/abs/2005.07683)

### 8.5.3 WoodFisher: optimal brain surgeon with an empirical-Fisher inverse (Singh & Alistarh, NeurIPS 2020)
Optimal Brain Surgeon removes the weight with the smallest saliency w_q²/(2[H⁻¹]_qq) and updates all remaining weights by −w_q H⁻¹e_q/[H⁻¹]_qq to compensate. WoodFisher approximates H by a block-diagonal empirical Fisher and computes its inverse with Woodbury rank-one updates over gradients, which makes OBS feasible for whole networks (SparseGPT's layer-wise variant uses the input Hessian XXᵀ instead, which ignores the loss).
- **Block:** `experimental.compress.WoodFisherPruning` (reuses `CurvatureProducts` for gradient capture).
- **Experiment:** `PruningShootout` (one-shot rows).
- **Toy task:** the digits MLP and the `VisionShootout` CNN, one-shot at 50–90% with no fine-tuning, then with 100 recovery steps.
- **Baselines:** magnitude, Wanda, SparseGPT (`WeightPruning`); OBD (diagonal Fisher) as an ablation.
- **Expected metric:** accuracy right after pruning and after recovery. Hypothesis: WoodFisher > SparseGPT > magnitude one-shot; the gap shrinks after recovery.
- **Source:** [arXiv:2004.14340](https://arxiv.org/abs/2004.14340)

### 8.5.4 Pruning at initialisation: SNIP, GraSP and SynFlow as pruners (Lee et al., 2019; Wang et al., 2020; Tanaka et al., 2020)
These score weights before any training and prune once: SNIP by |w·∂L/∂w| on one batch, GraSP by keeping weights whose removal would reduce gradient flow (a Hessian-gradient product), SynFlow data-free and iteratively, by the synaptic-flow product of absolute weights, which avoids layer collapse at extreme sparsity. `ZeroCostProxies` already computes SNIP and SynFlow scores, but only to rank architectures; no pruner uses them. Frankle et al. (2021) found these methods barely beat random per-layer pruning at the same layer-wise ratios, which this item can test.
- **Block:** `experimental.compress.PruningAtInit` (`snip`, `grasp`, `synflow`, `randomLayerwise`).
- **Experiment:** `PruningShootout` (at-init rows).
- **Toy task:** the digits MLP and the `VisionShootout` CNN at 90/98/99.5% sparsity.
- **Baselines:** random pruning (uniform and with the same per-layer ratios), `LotteryTicket` (pruning after training, as an upper bound).
- **Expected metric:** accuracy after training against sparsity; per-layer survival. Hypothesis: only SynFlow avoids collapse at 99.5%; at moderate sparsity none beats random pruning with matched layer ratios.
- **Source:** [arXiv:1810.02340](https://arxiv.org/abs/1810.02340), [arXiv:2002.07376](https://arxiv.org/abs/2002.07376), [arXiv:2006.05467](https://arxiv.org/abs/2006.05467), [arXiv:2009.08576](https://arxiv.org/abs/2009.08576)

### 8.5.5 Structured LLM pruning: LLM-Pruner, FASP and 2SSP (Ma et al., NeurIPS 2023; 2025; Sandri et al., TMLR 2025)
LLM-Pruner finds groups of coupled structures (a head with its output columns, an FFN neuron with its row and column) by dependency tracing, scores each group by a first-order Taylor estimate of the loss change, removes them and recovers with LoRA. FASP (2025) links the pruning of one layer's columns to the next layer's rows, so whole channels go at once, and restores the remaining weights by a closed-form least-squares update. 2SSP (2025) first prunes FFN neurons by output-activation magnitude (width), then whole attention modules (depth), balancing the two stages by a target sparsity. Only depth pruning (ShortGPT) and SliceGPT exist.
- **Block:** `experimental.compress.StructuredLmPruner` (`llmPruner`, `fasp`, `twoStage`).
- **Experiment:** `PruningShootout` (LM rows, next to `language.SliceCompression`).
- **Toy task:** the toy LM of `SliceCompression` (width 32) at 20/30/50% parameters removed; 40 recovery steps.
- **Baselines:** SliceGPT (2.618 at 75% width), Wanda unstructured (2.085), dropping one block via `BlockImportance` (2.139; 1.962 after recovery).
- **Expected metric:** validation loss before and after recovery, at equal parameter count, and real CPU speed-up. Hypothesis: FASP beats SliceGPT and LLM-Pruner without recovery; after recovery all structured methods are close to block dropping.
- **Source:** [arXiv:2305.11627](https://arxiv.org/abs/2305.11627), [arXiv:2501.09412](https://arxiv.org/abs/2501.09412), [arXiv:2501.17771](https://arxiv.org/abs/2501.17771)

### 8.5.6 Network slimming: channel pruning by batch-norm scales (Liu et al., ICCV 2017)
Train a CNN with an L1 penalty on every batch-norm scale γ; channels whose γ go to zero contribute nothing and are removed together with the matching filters of the previous layer, giving a thinner dense network that needs no sparse kernels. Liu et al. (2019) then showed that the pruned architecture trained from scratch often matches, so pruning mainly finds a good width per layer.
- **Block:** `experimental.compress.NetworkSlimming` (L1 on `BatchNorm` scales, channel surgery).
- **Experiment:** `PruningShootout` (CNN rows).
- **Toy task:** the `ClassicCnnShootout` shapes CNN and ResNet at 30/50/70% fewer channels.
- **Baselines:** filter-magnitude pruning (L1 norm of filters); the width-only scaled CNN from `ClassicCnnShootout` trained from scratch at the same multiply-adds; the slimmed architecture re-trained from scratch.
- **Expected metric:** accuracy against multiply-adds and measured CPU latency. Hypothesis: slimming beats uniform width reduction; re-training the slimmed shape from scratch is as good as fine-tuning.
- **Source:** [arXiv:1708.06519](https://arxiv.org/abs/1708.06519), [arXiv:1810.05270](https://arxiv.org/abs/1810.05270)

### 8.5.7 MoE expert pruning and merging (Lu et al., ACL 2024; Li et al., ICLR 2024)
Expert pruning drops the experts whose removal changes the layer output least on calibration data (searched over expert subsets) and can skip low-weight experts per token at inference. MC-SMoE groups experts by the similarity of their router logits, and merges each group into one expert after permutation-aligning their neurons, weighting by usage frequency, then optionally compresses the merged expert by low rank.
- **Block:** `experimental.compress.ExpertPruning`, `experimental.compress.ExpertMerging`.
- **Experiment:** `PruningShootout` (MoE rows, next to `attention.MoeRefinements`).
- **Toy task:** the `FineGrainedMoe` regression and LM tasks with 8–16 experts, reduced to 50% of the experts.
- **Baselines:** the full MoE; dropping the least-used experts; a dense FFN of equal active parameters; `SparseUpcycling` reversed (average all experts).
- **Expected metric:** validation loss at 50% experts. Hypothesis: merging beats pruning when experts are redundant (fine-grained), and loses when experts specialise.
- **Source:** [arXiv:2402.14800](https://arxiv.org/abs/2402.14800), [arXiv:2310.01334](https://arxiv.org/abs/2310.01334)

### 8.5.8 Decoupled and relational knowledge distillation (Zhao et al., CVPR 2022; Park et al., CVPR 2019)
DKD splits the classic KD loss into target-class KD (how confident the teacher is in the true class) and non-target-class KD (the teacher's distribution over wrong classes, the "dark knowledge"), and shows the classic loss suppresses the second whenever the teacher is confident; reweighting them separately helps. RKD distils relations instead of outputs: pairwise distances and triplet angles between the embeddings of examples in a batch.
- **Block:** `experimental.compress.DecoupledKdLoss`, `experimental.compress.RelationalKdLoss`.
- **Experiment:** `experiments.compression.CompactModels` (next to `language.DistillationShootout`).
- **Toy task:** a large CNN teacher and a 4× smaller student on `ShapeImages` and `ToyTasks` digits; embedding students for `KnnIndex` retrieval.
- **Baselines:** logit KD (`DistillationLoss`), `FeatureDistillationLoss`, student trained alone.
- **Expected metric:** student accuracy; recall@1 for embeddings. Hypothesis: DKD beats KD when the teacher is very confident; RKD wins only on the retrieval task.
- **Source:** [arXiv:2203.08679](https://arxiv.org/abs/2203.08679), [arXiv:1904.05068](https://arxiv.org/abs/1904.05068)

### 8.5.9 Deep Compression: pruning, weight sharing and Huffman coding (Han et al., ICLR 2016)
Three stages: prune small weights and retrain; cluster the remaining weights of each layer by k-means into 2ᵇ shared values (trained centroids, with gradients summed per cluster); then Huffman-code the cluster indices and the sparse index gaps. The storage size, not the arithmetic, is what is compressed, often 35–49× on classic nets.
- **Block:** `experimental.compress.WeightSharingQuantizer` (k-means with centroid fine-tuning, reuses `experimental.ml.KMeans`), `experimental.compress.HuffmanCoder`.
- **Experiment:** `CompactModels`.
- **Toy task:** the digits MLP and the shapes CNN; the toy LM.
- **Baselines:** round-to-nearest and `Gptq` at the same bits per weight; `AdditiveQuant`; pruning only.
- **Expected metric:** accuracy against total bits stored (including indices and codebooks). Hypothesis: at 3–4 bits shared-weight clustering matches GPTQ; Huffman adds 20–30% extra compression for free.
- **Source:** [arXiv:1510.00149](https://arxiv.org/abs/1510.00149)

### 8.5.10 Hessian-aware mixed-precision quantisation: HAWQ (Dong et al., ICCV 2019; HAWQ-V2, NeurIPS 2020)
Layers differ in sensitivity. HAWQ-V2 estimates each layer's sensitivity by the average Hessian trace (Hutchinson's estimator with Hessian-vector products) times its quantisation perturbation ‖Q(W) − W‖², and allocates bits per layer under a total-size budget by a Pareto-frontier search, quantising sensitive layers at higher precision.
- **Block:** `experimental.compress.HessianAwareBitAllocation` (uses `CurvatureProducts` HVPs, quantisers from `QuantMath`).
- **Experiment:** `CompactModels` (mixed-precision rows, next to `techniques.QuantizationShootout`).
- **Toy task:** the digits MLP, the shapes CNN and the toy LM at an average of 3 bits.
- **Baselines:** uniform 3-bit and 4-bit `Gptq`/round-to-nearest; random bit allocation at the same budget; allocation by weight variance.
- **Expected metric:** accuracy or loss at equal average bits. Hypothesis: Hessian-aware allocation at 3 bits average matches uniform 4 bits; random allocation is worse than uniform.
- **Source:** [arXiv:1905.03696](https://arxiv.org/abs/1905.03696), [arXiv:1911.03852](https://arxiv.org/abs/1911.03852)

### 8.5.11 PACT and DoReFa: learning to quantise activations (Choi et al., 2018; Zhou et al., 2016)
DoReFa quantises weights, activations and gradients to k bits with straight-through estimators, mapping weights through tanh into [0, 1] first. PACT replaces ReLU by a clipped ReLU with a learned clipping level α (trained through the straight-through estimator), so the activation range adapts to the bit width; it was the standard 4-bit activation recipe before LSQ.
- **Block:** `experimental.compress.PactActivation`, `experimental.compress.DoReFaQuantizer`.
- **Experiment:** `CompactModels` (QAT rows, next to `techniques.ParetoQat`).
- **Toy task:** the shapes CNN and digits MLP with 2/3/4-bit weights and activations.
- **Baselines:** `LsqDense` (learned step size), `ParetoQDense`, fixed-range clipping.
- **Expected metric:** accuracy against bits. Hypothesis: LSQ ≥ PACT > DoReFa at 2 bits; all tie at 4 bits.
- **Source:** [arXiv:1805.06085](https://arxiv.org/abs/1805.06085), [arXiv:1606.06160](https://arxiv.org/abs/1606.06160)

### 8.5.12 Tensor decompositions of layers: CP, Tucker and tensor-train (Lebedev et al., 2015; Kim et al., 2016; Novikov et al., 2015)
A conv kernel (out × in × k × k) is factorised by CP into a sum of R rank-one terms, which runs as four cheap convolutions (1×1, k×1, 1×k, 1×1), or by Tucker-2 along the channel modes (1×1 → small k×k core → 1×1). Tensor-train layers reshape a dense matrix into a high-order tensor stored as a chain of small cores, compressing it by orders of magnitude. The repo only has matrix SVD (`LowRankDense`); `MatrixProductState` is a classifier, not a layer compressor.
- **Block:** `experimental.compress.TensorDecomposition` (`cp` by ALS, `tucker2` by HOSVD, `tensorTrain` by TT-SVD), `experimental.compress.TensorTrainDense`.
- **Experiment:** `CompactModels` (decomposition rows).
- **Toy task:** the shapes CNN (conv layers) and the digits MLP (dense layers), decomposed post-training and fine-tuned.
- **Baselines:** SVD `LowRankDense` at equal parameters; channel pruning (8.5.6); training the small model from scratch.
- **Expected metric:** accuracy against parameters and measured latency. Hypothesis: Tucker beats SVD on conv layers; TT gives the highest compression of dense layers with a latency cost on a CPU.
- **Source:** [arXiv:1412.6553](https://arxiv.org/abs/1412.6553), [arXiv:1511.06530](https://arxiv.org/abs/1511.06530), [arXiv:1509.06569](https://arxiv.org/abs/1509.06569)

### 8.5.13 Token merging for vision transformers: ToMe (Bolya et al., ICLR 2023)
Between the attention and the MLP of each block, ToMe splits the tokens into two alternating sets, matches every token in one set to its most similar token (by key cosine similarity) in the other (bipartite soft matching), and merges the r most similar pairs by size-weighted averaging; proportional attention keeps merged tokens' weight. It needs no training and removes r tokens per layer.
- **Block:** `experimental.compress.TokenMerging` (a wrapper over `VisionTransformer` blocks).
- **Experiment:** `CompactModels` (ViT rows).
- **Toy task:** `VisionTransformer` on `ShapeImages` at 16×16 with 4×4 patches (16–64 tokens), r = 1–8 per layer.
- **Baselines:** the dense ViT (accuracy 0.996 at 8×8); token pruning by CLS-attention (keep top tokens); `PerceiverEncoder` with few latents (0.387).
- **Expected metric:** accuracy against multiply-adds and latency, with and without fine-tuning. Hypothesis: ToMe halves the FLOPs for < 1 point without training; pruning loses more.
- **Source:** [arXiv:2210.09461](https://arxiv.org/abs/2210.09461)

### 8.5.14 Learned cost models for kernel autotuning: AutoTVM and Ansor (Chen et al., OSDI 2018 and NeurIPS 2018; Zheng et al., OSDI 2020)
Autotuning searches a kernel's schedule space (tile sizes, loop order, unrolling, vector width, parallel split). Measuring every candidate is slow, so AutoTVM fits a cost model (gradient-boosted trees on loop-structure features) to the measurements so far and uses it to choose the next batch by simulated annealing; Ansor generates sketches hierarchically and fine-tunes them with evolutionary search guided by the learned model. Here the space is the core CPU backend's tiled matmul and convolution parameters, timed for real; `EvolvedKernels` only evolves loop nests against a simulated cache.
- **Block:** `experimental.kernels.ScheduleSpace` (matmul and conv tilings over the CPU backend), `experimental.kernels.LearnedCostModel` (core `GradientBoostedTrees` on loop features), `experimental.kernels.AutoTuner` (`random`, `grid`, `evolution`, `costModelGuided`).
- **Experiment:** `experiments.kernels.KernelTuning`.
- **Toy task:** matmul shapes from the toy LM (64–512) and conv shapes from the shapes CNN.
- **Baselines:** the backend's default tiling; `EvolvedKernels`' evolutionary search with a simulated cache; random and grid search at equal measurement budgets.
- **Expected metric:** GFLOP/s of the best schedule against number of real measurements; cost-model rank correlation. Hypothesis: the cost model reaches 95% of the best found in a third of the measurements; the simulated-cache optimum is not the real optimum.
- **Source:** [arXiv:1802.04799](https://arxiv.org/abs/1802.04799), [arXiv:1805.08166](https://arxiv.org/abs/1805.08166), [arXiv:2006.06762](https://arxiv.org/abs/2006.06762)

### 8.5.15 Fast algorithms: Winograd convolution and Strassen matmul (Lavin & Gray, CVPR 2016; Strassen, 1969)
Winograd's minimal filtering F(2×2, 3×3) computes a 2×2 output tile of a 3×3 convolution with 16 instead of 36 multiplications, by transforming the input tile and filter (Bᵀ d B, G g Gᵀ), multiplying element-wise and transforming back; batched over channels it becomes a set of matmuls. Strassen multiplies 2×2 block matrices with 7 instead of 8 block products, recursively. Both trade multiplications for additions and numerical error, which matters for low-precision inference.
- **Block:** `experimental.kernels.WinogradConv` (F(2,3) and F(4,3)), `experimental.kernels.StrassenMatmul` (with a cut-over size).
- **Experiment:** `KernelTuning` (algorithm rows).
- **Toy task:** the shapes CNN's 3×3 layers and square matmuls 64–1024; inference of the CNN in fp32 and after 8-bit quantisation.
- **Baselines:** the core direct and im2col convolutions; the core matmul.
- **Expected metric:** wall-clock speed-up, maximum relative error, and accuracy of the quantised CNN. Hypothesis: Winograd F(2,3) gives a real speed-up with negligible error; F(4,3) is faster but its error hurts 8-bit inference; Strassen only pays off above ~512.
- **Source:** [arXiv:1509.09308](https://arxiv.org/abs/1509.09308), [Strassen 1969](https://doi.org/10.1007/BF02165411)

### 8.5.16 AlphaTensor-style search for low-rank matmul decompositions (Fawzi et al., Nature 2022)
Multiplying n×n matrices with R scalar multiplications is equivalent to writing the n×n×n matrix-multiplication tensor as a sum of R rank-one tensors u⊗v⊗w. AlphaTensor plays a single-player game that subtracts rank-one terms until the tensor is zero, with an AlphaZero agent, and rediscovered Strassen's rank 7 for 2×2 and found new decompositions over GF(2). A CPU toy searches the 2×2 and 2×2×3 cases with coefficients in {−1, 0, 1}.
- **Block:** `experimental.kernels.TensorDecompositionGame` (environment for `AlphaZero`/`GumbelMcts`), plus an alternating-least-squares-with-rounding search for comparison.
- **Experiment:** `KernelTuning` (discovery rows, next to `evolution.EvolvedKernels`).
- **Toy task:** decompose the 2×2×2 matmul tensor (target rank 7) and the 2×2×3 tensor (target rank 11).
- **Baselines:** `ProgramEvolution`/`CmaEs` over factor entries; random search; the naive rank (8 and 12).
- **Expected metric:** best rank found and episodes or evaluations to find it; correctness of the resulting algorithm on random matrices. Hypothesis: MCTS and ALS-with-rounding both find rank 7; only the search methods find rank 11 reliably.
- **Source:** [AlphaTensor (Nature 2022)](https://doi.org/10.1038/s41586-022-05172-4)

### 8.5.17 Tensor-graph superoptimisation with rewrite rules and equality saturation (Jia et al., SOSP 2019; Willsey et al., POPL 2021)
TASO automatically generates and verifies graph substitutions (for example, fusing two matmuls that share an input into one wider matmul, or folding a scale into a weight) and searches for the cheapest graph by a cost-guided backtracking search. Equality saturation (the egg library's e-graphs) instead applies all rewrites non-destructively to a compact e-graph until saturation, then extracts the cheapest equivalent program, avoiding phase-ordering problems.
- **Block:** `experimental.kernels.TensorExpressionGraph`, `experimental.kernels.EGraph` (rewrite rules, saturation, cost extraction), `experimental.kernels.GreedyRewriter`.
- **Experiment:** `KernelTuning` (graph rows).
- **Toy task:** forward graphs of the toy transformer block (separate Q/K/V projections, scale-then-matmul, bias-add chains) and the shapes CNN (conv + batch norm).
- **Baselines:** the unrewritten graph executed by core `Model`; a fixed set of hand-written fusions applied greedily.
- **Expected metric:** FLOPs and measured latency of the extracted graph, number of rules applied, and numerical equivalence. Hypothesis: equality saturation finds the QKV fusion and batch-norm folding that greedy rewriting misses when rule order matters.
- **Source:** [TASO (SOSP 2019)](https://doi.org/10.1145/3341301.3359630), [arXiv:2004.03082](https://arxiv.org/abs/2004.03082)

### 8.5.18 Sparse kernels that run faster: block-sparse and n:m matmul on a CPU (Gray et al., 2017; Mishra et al., 2021)
`WeightPruning` produces masks, but nothing in the repo turns sparsity into speed (core `SparseMatrix` is a COO gather/scatter). Block-sparse kernels store only non-zero b×b blocks and run dense micro-kernels on them; n:m kernels store the n kept values per group of m plus 2-bit indices and multiply with a compact gather. The item measures at what sparsity each format beats the dense kernel, and which pruning criterion gives blocks that survive.
- **Block:** `experimental.kernels.BlockSparseMatmul` (BSR), `experimental.kernels.NmSparseMatmul`, `experimental.kernels.CsrMatmul`.
- **Experiment:** `KernelTuning` (sparse rows), with an end-to-end row in `PruningShootout`.
- **Toy task:** matmuls 256–1024 at 50–95% sparsity; the toy LM pruned to 2:4 and to 8×8 blocks.
- **Baselines:** the core dense matmul; core `SparseMatrix` (COO); unstructured CSR.
- **Expected metric:** speed-up against sparsity per format, and LM loss of block-pruned versus unstructured models. Hypothesis: unstructured CSR needs > 90% sparsity to win, 2:4 gives a modest constant speed-up, and 8×8 blocks win from ~70% at some loss cost.
- **Source:** [Block-sparse GPU kernels (OpenAI, 2017)](https://openai.com/index/block-sparse-gpu-kernels/), [arXiv:2104.08378](https://arxiv.org/abs/2104.08378)

## WP6. Privacy and federated learning, robustness and certification, active learning and data valuation, spiking and quantum-inspired computing (18)

Target: two new packages, `experimental.privacy` (federated algorithms, secure and private aggregation, privacy attacks, PATE) and `experimental.robust` (adversarial training, certification, Lipschitz layers, test-time adaptation). Active learning and data valuation go in `experimental.data` (next to `DataPruner` and `InRunShapley`); spiking blocks in `experimental.nn` (next to `LifLayer`); the Ising solver in `experimental.ml`. Measurements go in:

- a new `experiments.privacy.FederatedShootout` (8.6.1–8.6.4) and `experiments.privacy.PrivacyAttacks` (8.6.5–8.6.7), next to `safety.PrivateTraining`;
- a new `experiments.robustness.AdversarialRobustness` (8.6.8–8.6.11) and `experiments.robustness.ShiftAdaptation` (8.6.12);
- a new `experiments.data.ActiveLearning` and `experiments.data.DataValuation` (8.6.13–8.6.15);
- `experiments.exotic.SpikingClassifier` and `experiments.exotic.ProbabilisticBits` (extended, 8.6.16–8.6.18).

Baselines: `FederatedAveraging`, `DpSgd` with `RdpAccountant`, `OneRunAudit`, the loss-threshold membership AUC of `MachineUnlearning`, `GradientCompressor`/`QuantizedGradients`, standard training, `SpectralNormDense`, `CnnNormalizationShootout`'s BN models, `Calibration` (entropy, BALD), `InRunShapley`, `DataPruner` (EL2N, forgetting), `TrainingDataAttribution` (influence, TracIn), `LifLayer` with `SurrogateSpike`, `SpikeEncoding`, `PBitIsingMachine`, over at least 3 seeds. One new helper, `experimental.data.ClientSplits`, partitions `ToyTasks` digits, `ShapeImages` and `SyntheticRecsys.tabular` across 10–100 clients by a Dirichlet(α) label skew (α = 0.1/1/100) plus feature-shift splits. Robustness rows use ℓ∞ budgets suited to 8×8 images (ε = 1/255 to 8/255 of the range, scaled to the toy data) and are reported with clean accuracy.

### 8.6.1 FedProx and SCAFFOLD: correcting client drift (Li et al., MLSys 2020; Karimireddy et al., ICML 2020)
With non-IID clients, local SGD steps drift towards each client's optimum, and FedAvg's average is biased. FedProx adds a proximal term (μ/2)‖w − w_global‖² to each local objective, limiting drift. SCAFFOLD keeps control variates (estimates of the server and each client's update direction) and corrects every local gradient by c − cᵢ, which removes drift and provably converges independent of heterogeneity.
- **Block:** `experimental.privacy.FederatedAlgorithms` (`fedProx`, `scaffold`), built on `FederatedAveraging`'s client simulation.
- **Experiment:** `experiments.privacy.FederatedShootout`.
- **Toy task:** `ClientSplits` digits and shapes at α = 0.1/1/100, 10–100 clients, 10% participation, 1–20 local steps.
- **Baselines:** `FederatedAveraging`; centralised training (upper bound).
- **Expected metric:** test accuracy against communication rounds. Hypothesis: SCAFFOLD needs far fewer rounds than FedAvg at α = 0.1 with many local steps; FedProx helps a little; all tie at α = 100.
- **Source:** [arXiv:1812.06127](https://arxiv.org/abs/1812.06127), [arXiv:1910.06378](https://arxiv.org/abs/1910.06378)

### 8.6.2 Adaptive server optimisers and normalised averaging: FedOpt and FedNova (Reddi et al., ICLR 2021; Wang et al., NeurIPS 2020)
FedOpt treats the averaged client update as a pseudo-gradient and applies a server optimiser to it (FedAdam, FedYogi, FedAdagrad, server momentum), which needs less learning-rate tuning and copes with heterogeneity. FedNova normalises each client's update by its number of local steps before averaging, fixing the objective inconsistency that arises when clients do different amounts of work.
- **Block:** `experimental.privacy.ServerOptimizer` (`fedAdam`, `fedYogi`, `fedAvgM`, any core `Optimizer` as the server step), `experimental.privacy.FedNova`.
- **Experiment:** `FederatedShootout`.
- **Toy task:** as 8.6.1, with clients drawing 1–20 local steps at random.
- **Baselines:** `FederatedAveraging`; 8.6.1 SCAFFOLD.
- **Expected metric:** accuracy against rounds and the spread over server learning rates. Hypothesis: FedAdam is the most robust to the learning rate; FedNova fixes the bias only when local step counts vary.
- **Source:** [arXiv:2003.00295](https://arxiv.org/abs/2003.00295), [arXiv:2007.07481](https://arxiv.org/abs/2007.07481)

### 8.6.3 Personalised federated learning: Ditto, FedRep and Per-FedAvg (Li et al., ICML 2021; Collins et al., ICML 2021; Fallah et al., NeurIPS 2020)
When clients really differ, one global model is the wrong goal. Ditto trains a personal model per client regularised towards the global one, λ‖v − w‖²; FedRep learns a shared representation by federated averaging and a local head per client; Per-FedAvg learns a global initialisation that adapts well after one local gradient step (MAML in federated form).
- **Block:** `experimental.privacy.PersonalizedFederated` (`ditto`, `fedRep`, `perFedAvg`, reusing `Maml`'s inner loop).
- **Experiment:** `FederatedShootout` (personalisation rows).
- **Toy task:** `ClientSplits` with label skew α = 0.1 and with client-specific label permutations (concept shift).
- **Baselines:** FedAvg; FedAvg + local fine-tuning; local training only.
- **Expected metric:** mean and worst-decile client accuracy on their own test data. Hypothesis: FedRep wins under concept shift; FedAvg + fine-tuning is a strong baseline under label skew.
- **Source:** [arXiv:2012.04221](https://arxiv.org/abs/2012.04221), [arXiv:2102.07078](https://arxiv.org/abs/2102.07078), [arXiv:2002.07948](https://arxiv.org/abs/2002.07948)

### 8.6.4 User-level DP-FedAvg with secure aggregation (McMahan et al., ICLR 2018; Bonawitz et al., CCS 2017)
DP-FedAvg clips each client's whole update to norm S and adds Gaussian noise to the sum, giving user-level differential privacy (protecting all of a client's data, not one example), accounted by the moments/RDP accountant. Secure aggregation lets the server learn only the sum: each pair of clients agrees on a random mask that one adds and the other subtracts, so masks cancel in the sum; secret-shared seeds let the sum be recovered if clients drop out.
- **Block:** `experimental.privacy.UserLevelDpFedAvg` (uses `RdpAccountant`), `experimental.privacy.SecureAggregation` (pairwise masking over a finite field with dropout recovery).
- **Experiment:** `FederatedShootout` (privacy rows, next to `safety.PrivateTraining`).
- **Toy task:** `ClientSplits` digits with 100–1000 simulated clients.
- **Baselines:** `DpSgd` example-level DP at the same ε (centralised); non-private FedAvg; the `OneRunAudit` estimate of empirical ε.
- **Expected metric:** accuracy against user-level ε; audited ε lower bound; exactness of the masked sum and its cost against dropout rate. Hypothesis: with many clients user-level DP costs little accuracy; secure aggregation is exact at up to 30% dropout.
- **Source:** [arXiv:1710.06963](https://arxiv.org/abs/1710.06963), [Bonawitz et al. (ePrint 2017/281)](https://eprint.iacr.org/2017/281)

### 8.6.5 Gradient inversion: deep leakage from gradients and its defences (Zhu et al., NeurIPS 2019; Geiping et al., NeurIPS 2020)
A server that sees one client's gradient can reconstruct its training data: DLG optimises a dummy input and label so that their gradient matches the observed one (L2 distance); Geiping et al. use cosine similarity plus total-variation regularisation and recover images even from averaged updates of small batches. The item measures the attack and how far the repo's existing defences (noise, compression, larger batches, secure aggregation) stop it.
- **Block:** `experimental.privacy.GradientInversionAttack` (`dlg`, `cosineTv`).
- **Experiment:** `experiments.privacy.PrivacyAttacks`.
- **Toy task:** `ShapeImages` and digits gradients from an MLP and the shapes CNN, batch sizes 1–16.
- **Baselines (defences):** none; `DpSgd` noise; top-k `GradientCompressor` and `QuantizedGradients`; batch averaging; 8.6.4 secure aggregation (only the sum is visible).
- **Expected metric:** reconstruction PSNR and label accuracy per defence and batch size, with the defence's accuracy cost. Hypothesis: single-example gradients leak almost perfectly; top-k at 1% stops most of it at little accuracy cost, as does DP noise at moderate ε.
- **Source:** [arXiv:1906.08935](https://arxiv.org/abs/1906.08935), [arXiv:2003.14053](https://arxiv.org/abs/2003.14053)

### 8.6.6 Membership inference from first principles: LiRA (Carlini et al., IEEE S&P 2022)
The likelihood-ratio attack trains many shadow models, half including the target example and half not, fits a Gaussian to the target's (logit-scaled) confidence in each group, and scores membership by the likelihood ratio, per example. Carlini et al. argue attacks should be judged by true-positive rate at very low false-positive rates, where loss thresholds are close to useless.
- **Block:** `experimental.privacy.LikelihoodRatioAttack` (online and offline variants, shadow-model training).
- **Experiment:** `PrivacyAttacks` (also an auditor in `safety.PrivateTraining` and `interp.UnlearningComparison`).
- **Toy task:** small MLPs on digits subsets of 500, 16–64 shadow models; models trained with `DpSgd` at several ε; unlearned models from `MachineUnlearning`.
- **Baselines:** the loss-threshold attack (the `membershipAuc` already reported by `OneRunAudit` and `MachineUnlearning`).
- **Expected metric:** AUC and TPR at 0.1% and 1% FPR. Hypothesis: equal AUC to the loss attack but ≥ 10× its TPR at 0.1% FPR; unlearning methods that "pass" the loss attack fail LiRA.
- **Source:** [arXiv:2112.03570](https://arxiv.org/abs/2112.03570)

### 8.6.7 PATE: private aggregation of teacher ensembles (Papernot et al., ICLR 2017, 2018)
Split the private data into disjoint parts and train one teacher per part; a student is trained on public unlabelled data labelled by the teachers' noisy-max vote. Privacy is spent only on the queries the student makes, and the data-dependent analysis (and Confident-GNMax, which only answers when teachers agree strongly) gives small ε when teachers agree.
- **Block:** `experimental.privacy.TeacherEnsembleAggregation` (`laplaceMax`, `gnMax`, `confidentGnMax`, with data-dependent RDP via `RdpAccountant`).
- **Experiment:** `PrivacyAttacks` (utility–privacy rows, next to `safety.PrivateTraining`).
- **Toy task:** digits with 10–50 teachers, 200–1000 student queries, the rest as public unlabelled data.
- **Baselines:** `DpSgd` at the same ε; non-private training.
- **Expected metric:** student accuracy against ε. Hypothesis: PATE beats DP-SGD at small ε when public unlabelled data exist.
- **Source:** [arXiv:1610.05755](https://arxiv.org/abs/1610.05755), [arXiv:1802.08908](https://arxiv.org/abs/1802.08908)

### 8.6.8 Adversarial training: PGD-AT, TRADES and fast FGSM training (Madry et al., ICLR 2018; Zhang et al., ICML 2019; Wong et al., ICLR 2020)
PGD adversarial training solves min_θ E max_{‖δ‖∞≤ε} L(x + δ), with the inner max found by projected gradient ascent from a random start. TRADES trains on clean cross-entropy plus β·KL(f(x) ‖ f(x + δ)), trading accuracy for robustness explicitly. Wong et al. show a single FGSM step from a random start ("fast is better than free") matches PGD-AT at a fraction of the cost, if catastrophic overfitting is avoided. The repo has no input-space adversarial attack or training at all; `LatentAdversarialTrainer` perturbs hidden states of an LM.
- **Block:** `experimental.robust.AdversarialAttacks` (`fgsm`, `pgd`, with restarts), `experimental.robust.AdversarialTraining` (`pgdAt`, `trades`, `fastFgsm`).
- **Experiment:** `experiments.robustness.AdversarialRobustness`.
- **Toy task:** `ShapeImages` and digits with the shapes CNN and an MLP; two-moons for visualising boundaries.
- **Baselines:** standard training; training with random noise of the same norm.
- **Expected metric:** clean and PGD-20 robust accuracy at several ε, and training time. Hypothesis: standard models fall to near zero at the larger ε; PGD-AT and TRADES hold most of it at a clean-accuracy cost; fast FGSM matches PGD-AT at ~3× less time.
- **Source:** [arXiv:1706.06083](https://arxiv.org/abs/1706.06083), [arXiv:1901.08573](https://arxiv.org/abs/1901.08573), [arXiv:2001.03994](https://arxiv.org/abs/2001.03994)

### 8.6.9 Randomized smoothing: certified ℓ₂ robustness (Cohen et al., ICML 2019)
Classify by majority vote of the base classifier over Gaussian-noised copies of the input, g(x) = argmax_c P(f(x + η) = c), η ~ N(0, σ²I). If the top class has probability at least p_A (a Clopper–Pearson lower bound from Monte Carlo samples), g is certifiably constant within ℓ₂ radius σ·Φ⁻¹(p_A). Training the base model with noise is essential.
- **Block:** `experimental.robust.RandomizedSmoothing` (prediction with abstention, certification).
- **Experiment:** `AdversarialRobustness` (certified rows).
- **Toy task:** as 8.6.8, σ = 0.12/0.25/0.5 (scaled to the data range).
- **Baselines:** 8.6.8 PGD-AT's empirical ℓ₂ robust accuracy (an upper bound on what is certifiable); standard training under smoothing.
- **Expected metric:** certified accuracy against radius; abstention rate; samples per prediction. Hypothesis: certified accuracy tracks PGD-AT's empirical accuracy at small radii and needs noise training.
- **Source:** [arXiv:1902.02918](https://arxiv.org/abs/1902.02918)

### 8.6.10 Bound propagation and certified training: IBP, CROWN and CROWN-IBP (Gowal et al., 2018; Zhang et al., NeurIPS 2018; Zhang et al., ICLR 2020)
Interval bound propagation pushes an input box through the network layer by layer (|W|·radius for linear layers, monotone activations clamp the bounds), giving guaranteed output bounds; training on the worst-case logits within these bounds yields certifiably robust networks. CROWN propagates linear lower and upper bounds for each ReLU backwards and is much tighter; CROWN-IBP mixes the two for stable certified training.
- **Block:** `experimental.robust.BoundPropagation` (`ibp`, `crown` for dense, conv and ReLU layers), `experimental.robust.CertifiedTraining`.
- **Experiment:** `AdversarialRobustness` (certified rows).
- **Toy task:** as 8.6.8, ℓ∞ radii.
- **Baselines:** PGD-AT (8.6.8), certified by IBP and CROWN after training; 8.6.9 (a different norm, reported side by side); standard training.
- **Expected metric:** verified accuracy (and the bound gap of IBP versus CROWN on the same model); clean accuracy. Hypothesis: PGD-trained models are robust empirically but almost uncertifiable by IBP; IBP-trained models are certifiable at a large clean-accuracy cost; CROWN-IBP sits between.
- **Source:** [arXiv:1810.12715](https://arxiv.org/abs/1810.12715), [arXiv:1811.00866](https://arxiv.org/abs/1811.00866), [arXiv:1906.06316](https://arxiv.org/abs/1906.06316)

### 8.6.11 1-Lipschitz networks: GroupSort with orthonormal weights, and AOL (Anil et al., ICML 2019; Prach & Lampert, ECCV 2022)
If every layer is 1-Lipschitz, a classification margin m certifies robustness within radius m/√2 at no extra cost. Anil et al. show ReLU with norm-constrained weights loses expressivity, and use the GroupSort activation (sort pairs of units) with weights made orthonormal by Björck iterations, which are gradient-norm preserving. AOL rescales each column of a weight matrix by a factor computed from |WᵀW| so the layer is 1-Lipschitz without any iterative orthogonalisation.
- **Block:** `experimental.robust.LipschitzLayers` (`groupSort`, `bjorckDense`, `aolDense`, `aolConv`).
- **Experiment:** `AdversarialRobustness` (Lipschitz rows).
- **Toy task:** as 8.6.8; two-moons for the margin picture.
- **Baselines:** `SpectralNormDense` + ReLU (Lipschitz but not gradient-norm preserving); 8.6.10 IBP training; 8.6.9 randomized smoothing.
- **Expected metric:** certified ℓ₂ accuracy against radius, clean accuracy, training time. Hypothesis: GroupSort + Björck > spectral norm + ReLU at equal size; AOL matches Björck at lower cost.
- **Source:** [arXiv:1811.05381](https://arxiv.org/abs/1811.05381), [arXiv:2208.03160](https://arxiv.org/abs/2208.03160)

### 8.6.12 Test-time adaptation: BN statistics, TENT and MEMO (Schneider et al., NeurIPS 2020; Wang et al., ICLR 2021; Zhang et al., NeurIPS 2022)
Under covariate shift, simply re-estimating batch-norm statistics on the test batch recovers much of the accuracy. TENT goes further: it updates only the BN affine parameters online to minimise the entropy of the test predictions. MEMO adapts on a single test point by minimising the entropy of the marginal prediction over augmentations of that point, so it needs no batch.
- **Block:** `experimental.robust.TestTimeAdaptation` (`bnStats`, `tent`, `memo`).
- **Experiment:** `experiments.robustness.ShiftAdaptation`.
- **Toy task:** `ShapeImages` with test-time corruptions (Gaussian noise, blur, contrast, inversion), at 5 severities; a label-shift control where TENT is known to fail.
- **Baselines:** no adaptation; the BN and GroupNorm models of `CnnNormalizationShootout`; training with augmentation.
- **Expected metric:** accuracy per corruption and severity, and over a long stream (to catch collapse). Hypothesis: BN statistics recover most of the gap; TENT adds a few points but collapses on long or label-shifted streams; MEMO helps with batch size 1.
- **Source:** [arXiv:2006.16971](https://arxiv.org/abs/2006.16971), [arXiv:2006.10726](https://arxiv.org/abs/2006.10726), [arXiv:2110.09506](https://arxiv.org/abs/2110.09506)

### 8.6.13 Active learning: core-set and BADGE (Sener & Savarese, ICLR 2018; Ash et al., ICLR 2020)
Uncertainty sampling picks redundant points in batches. The core-set approach picks a batch by greedy k-centre over embeddings, covering the pool. BADGE embeds each unlabelled point by its hypothetical last-layer gradient (using the predicted label), whose norm reflects uncertainty, and picks a batch with k-means++ seeding in that space, which balances uncertainty and diversity. Only the acquisition scores (entropy, BALD in `Calibration`) exist, with no active-learning loop.
- **Block:** `experimental.data.ActiveLearner` (loop with pluggable acquisition: `random`, `entropy`, `bald`, `coreSet`, `badge`, `leverage` from 8.6.14).
- **Experiment:** `experiments.data.ActiveLearning`.
- **Toy task:** digits and `ShapeImages` with a pool of 2000, starting from 20 labels, batches of 10 and 50; a pool with 30% redundant near-duplicates.
- **Baselines:** random sampling; entropy and BALD (`Calibration`).
- **Expected metric:** accuracy against labels used (learning curve area). Hypothesis: BADGE ≥ core-set > entropy at large batches; entropy wins at batch 1; on the duplicate pool, entropy falls below random.
- **Source:** [arXiv:1708.00489](https://arxiv.org/abs/1708.00489), [arXiv:1906.03671](https://arxiv.org/abs/1906.03671)

### 8.6.14 Cheap data valuation: KNN-Shapley, Data-OOB and leverage scores (Jia et al., VLDB 2019; Kwon & Zou, ICML 2023; NeurIPS 2025)
KNN-Shapley computes exact Shapley values for a k-nearest-neighbour classifier on embeddings by a closed-form recursion over sorted distances, in O(N log N) per test point. Data-OOB trains a bagging ensemble and values each point by how well the models that did not see it predict it (out-of-bag accuracy), with no retraining. Geometric data valuation (NeurIPS 2025) values a point by its statistical leverage score in feature space, x_iᵀ(XᵀX + λI)⁻¹x_i, a training-free measure of how much it extends the span of the data, and uses it for selection.
- **Block:** `experimental.data.DataValuation` (`knnShapley`, `dataOob` on core `RandomForest`-style bagging, `leverageScores`).
- **Experiment:** `experiments.data.DataValuation` (next to `data.ShapleyPruning` and `data.TrainingAttributionExperiment`).
- **Toy task:** digits with 20% flipped labels (as in `ShapleyPruning`); point-removal and point-addition curves.
- **Baselines:** In-Run Shapley (flipped-label detection 0.35/0.43), EL2N (0.56), TracIn and influence functions, random.
- **Expected metric:** share of flipped labels among the lowest-valued 20%, accuracy after removing them, and compute time. Hypothesis: KNN-Shapley and Data-OOB beat In-Run Shapley at detecting flipped labels in a fraction of the time; leverage scores do not detect noise but are the best for selecting diverse subsets.
- **Source:** [arXiv:1908.08619](https://arxiv.org/abs/1908.08619), [arXiv:2304.07718](https://arxiv.org/abs/2304.07718), [Geometric Data Valuation via Leverage Scores (NeurIPS 2025)](https://neurips.cc/virtual/2025/135741)

### 8.6.15 Self-supervised prototype pruning across a dataset-size sweep (Sorscher et al., NeurIPS 2022) — follow-up
Waves 4 and 5 found that pruning by EL2N, forgetting or In-Run Shapley does not beat random pruning (keeping 50%: 0.582–0.647 vs 0.645 random). Sorscher et al. predict exactly this: with little data one should keep easy examples and with much data hard ones, and the crossover depends on the data size. Their self-supervised metric clusters embeddings by k-means and scores each example by its distance to its prototype, needing no labels. The follow-up sweeps the dataset size and the kept fraction, so the earlier negatives get a fair test.
- **Block:** `experimental.data.PrototypePruning` (embeddings from an existing `SelfSupervisedLosses` encoder, `KMeans` prototypes).
- **Experiment:** `experiments.data.DataValuation` (pruning rows; adds rows to `vision.DataPruningExperiment` without changing its existing ones).
- **Toy task:** digits and `ShapeImages` at 500/2000/8000 examples, keeping 30–90%, keeping easiest versus hardest.
- **Baselines:** random pruning; EL2N and forgetting (`DataPruner`); In-Run Shapley.
- **Expected metric:** test accuracy against kept fraction per dataset size. Hypothesis: at 500 examples keeping easy examples is best; at 8000 keeping hard (prototype-far) ones beats random, reproducing the crossover.
- **Source:** [arXiv:2206.14486](https://arxiv.org/abs/2206.14486)

### 8.6.16 ANN-to-SNN conversion with quantisation-clip-floor activations (Rueckauer et al., 2017; Bu et al., ICLR 2022)
Train an ordinary ReLU network, then replace each ReLU by an integrate-and-fire neuron whose firing rate over T time steps approximates the activation; weights are rescaled by data-based normalisation (a high percentile of each layer's activations) so neurons neither saturate nor stay silent, and reset-by-subtraction keeps the residual charge. Plain conversion needs hundreds of steps; QCFS trains the ANN with a quantised, clipped, shifted activation matched to the SNN's T, so conversion works in 4–16 steps.
- **Block:** `experimental.nn.SnnConverter` (data-based normalisation, reset by subtraction), `experimental.nn.QuantClipFloorActivation`.
- **Experiment:** `exotic.SpikingClassifier` (conversion rows).
- **Toy task:** digits and `ShapeImages` MLP and small CNN.
- **Baselines:** the LIF network trained directly with surrogate gradients (`LifLayer`, `SurrogateSpike`); the source ANN.
- **Expected metric:** accuracy against time steps T (1–256) and spike counts. Hypothesis: plain conversion needs T ≥ 64 to match the ANN; QCFS gets within a point at T = 8, beating direct surrogate training at equal T.
- **Source:** [Rueckauer et al. (Frontiers in Neuroscience 2017)](https://doi.org/10.3389/fnins.2017.00682), [arXiv:2303.04347](https://arxiv.org/abs/2303.04347)

### 8.6.17 Better direct SNN training: TET and adaptive asymmetric surrogate gradients (Deng et al., ICLR 2022; ICML 2026)
TET (temporal efficient training) applies the loss to the output at every time step and averages, instead of to the time-averaged output, which gives flatter minima and better generalisation, plus a regulariser keeping membrane potentials near a target. A²SG (ICML 2026) makes the surrogate gradient's width and asymmetry adaptive during training, so early training gets wide, informative gradients and late training sharp, accurate ones, addressing the mismatch between a fixed surrogate and the true spike function.
- **Block:** `experimental.nn.TemporalEfficientLoss`, `experimental.autograd.AdaptiveSurrogateSpike` (learnable/scheduled width and asymmetry, next to `SurrogateSpike`).
- **Experiment:** `exotic.SpikingClassifier` (training rows), and `exotic.SpikingLanguageModel`.
- **Toy task:** digits and `ShapeImages` with `SpikeEncoding` rate inputs, T = 4/8.
- **Baselines:** `LifLayer` with the fixed surrogate and loss on mean output; 8.6.16 conversion at equal T.
- **Expected metric:** accuracy at T = 4 and 8, and the gradient-mismatch measure (cosine between surrogate and finite-difference gradients). Hypothesis: TET adds a few points at T = 4; the adaptive surrogate adds more on deeper networks.
- **Source:** [arXiv:2202.11946](https://arxiv.org/abs/2202.11946), [A²SG (ICML 2026)](https://icml.cc/virtual/2026/poster/66327)

### 8.6.18 Simulated bifurcation: a quantum-inspired Ising solver (Goto et al., Science Advances 2019, 2021)
Inspired by networks of Kerr-nonlinear parametric oscillators, simulated bifurcation integrates classical Hamiltonian equations for continuous positions x_i and momenta y_i with a pump amplitude that ramps up, so each x_i bifurcates to ±1 according to the Ising couplings. It updates all spins in parallel with simple vector operations; the ballistic and discrete variants (2021) add walls at |x| = 1 and sign-discretised couplings for better solutions. It solves MaxCut and other QUBO problems.
- **Block:** `experimental.ml.SimulatedBifurcation` (`adiabatic`, `ballistic`, `discrete`), plus a plain simulated-annealing reference.
- **Experiment:** `exotic.ProbabilisticBits` (Ising rows).
- **Toy task:** MaxCut on random and planted-partition graphs of 64–1024 nodes (from `SyntheticGraphs`); Sherrington–Kirkpatrick instances; the bars-and-stripes Boltzmann task as a sampling check.
- **Baselines:** `PBitIsingMachine` (Gibbs/p-bit annealing); simulated annealing; the spectral-clustering cut (`SpectralClustering`).
- **Expected metric:** best and mean cut value against wall-clock time; time to reach 99% of the best known. Hypothesis: discrete SB reaches the best cuts fastest on dense graphs; p-bits are competitive on sparse ones.
- **Source:** [Goto et al. 2019 (Science Advances)](https://doi.org/10.1126/sciadv.aav2372), [Goto et al. 2021 (Science Advances)](https://doi.org/10.1126/sciadv.abe7953)

---

## Notes on dependencies between items

**Vision (WP1)**
- First create the `experimental.vision` package, `experimental.data.ShapeScenes` (multi-object scenes with boxes, instance masks, keypoints, stereo pairs and videos with flow and track ids) and `DetectionMetrics`. Every WP1 item uses the generator; build it as an extension of `ShapeSegmentation`'s two-shape scenes so that experiment keeps its numbers.
- 8.1.1 → 8.1.2 → 8.1.3 share `DetectionShootout`, the backbone and core `Boxes`; 8.1.3 reuses the Hungarian solver of `MinibatchOptimalTransport` (do not copy it; expose it). 8.1.2's heatmap code is reused by 8.1.9.
- 8.1.17's `AffineSampler` (bilinear sampling) is also the warping step of 8.1.6–8.1.7 and the shift of 8.1.8; build it first in that group. 8.1.6 is the reference for 8.1.7; one agent should do both.
- 8.1.10 consumes 8.1.2's detections, `StateSpaceFilter.kalman` and the Hungarian solver. 8.1.11 shares the video generator.
- 8.1.12 → 8.1.13 → 8.1.14 share `SceneFields` and `FourierFeatures`; 8.1.12 must reuse `NovelViewSynthesis`'s scene and cameras so 3DGS is a like-for-like row.
- 8.1.15 and 8.1.16 are independent; 8.1.15 should land next to `exotic.EquivariantPointClouds` without moving it.

**Time series and signals (WP2)**
- First extend `SeasonalSeries` (multivariate, covariates) and add `SeriesClassificationTasks` and `AnomalySeries` in `experimental.data`. Keep `ForecastingBenchmark`'s existing rows reproducible.
- 8.2.1 fits by `Lbfgs` on `StateSpaceFilter`; 8.2.8 uses 8.2.1's intervals as a row and 8.2.7's samples as an input. 8.2.7 is the slowest item (pretraining on KernelSynth); start it early. 8.2.7's KernelSynth uses `KernelFunction` and `GaussianProcess`; coordinate with no one else (WP3 does not touch kernels).
- 8.2.2 → 8.2.3 → 8.2.4 → 8.2.6 share `LongHorizonForecasting`; 8.2.5 touches `NBeats` plumbing and must keep its existing numbers.
- 8.2.9 and 8.2.10 share `SeriesClassification`; 8.2.17 adds rows there and in the audio experiment.
- 8.2.11–8.2.15 share the new `experimental.anomaly` package and `AnomalySeries`; 8.2.13's `SeriesAnomalyMetrics` is used by 8.2.11, 8.2.12 and 8.2.14 (build it first). 8.2.12's one-class SVM reuses `KernelSvm`'s solver.
- 8.2.16 and 8.2.18 change the audio experiments (`SpokenDigitRecognition`, `SourceSeparation`); one agent should do both, keeping wave 7's rows.

**Scientific ML (WP3)**
- First build `PdeProblems` and `MechanicalSystems` (reference solvers and exact integrators) in `experimental.sciml`. Check that the core autograd gives second input-derivatives through `Var` for every activation used (tanh, sin, GELU); this is the only core dependency, and any fix belongs in a separate core PR.
- 8.3.1 → 8.3.2 → 8.3.3 → 8.3.17 share `PhysicsInformedNetwork` and `PdeShootout`; one agent should do 8.3.1–8.3.2.
- 8.3.4 → 8.3.5 → 8.3.6 → 8.3.7 share `OperatorLearning` and must reuse `FourierNeuralOperator1d` unchanged. 8.3.6 imports `experimental.graph` helpers; 8.3.7 imports `experimental.nn.diffusion` schedules.
- 8.3.8 builds the symplectic integrators used by 8.3.13 and 8.3.15's MD run. 8.3.9 SINDy is reused by 8.3.11 (closed-form recovery). 8.3.10 uses `ThinSvd`; 8.3.14 uses `SymmetricEigen`/`ThinSvd`.
- 8.3.12 reuses `EntropicTransport` for its marginal metric. 8.3.16 reuses `StateSpaceFilter` and `experimental.nn.diffusion`.

**Serving, tokenisation, long context and agents (WP4)**
- First create `experiments.serving.ServingShootout` on `language.ServingThroughput`'s workload generator, keeping that experiment's rows. 8.4.1 and 8.4.2 share the compression harness; 8.4.4 and 8.4.11 add rows to `context.HeadBudgetComparison` and `LongContextNeedle`; coordinate the edits with whoever owns those experiments.
- 8.4.5 and 8.4.6 share the speculative harness and the draft models of `language.Eagle3Speculation`; one agent should do both.
- 8.4.7 and 8.4.8 share the multi-request engine (`ChunkedPrefillScheduler`); 8.4.9 needs read access to core `Grammar`'s automaton state (add an accessor in a separate core PR if needed).
- 8.4.12 → 8.4.13 → 8.4.14 share `TokenizerShootout` and its LM-training loop; 8.4.13's segmentation wrapper also applies to 8.4.12's vocabulary.
- 8.4.15–8.4.18 share `ToolAgentBench`. 8.4.16 changes no existing `GrpoTrainer` behaviour (it adds a rollout wrapper); schedule it after any open `GrpoTrainer` work. 8.4.18 uses `SimulatedUser`.

**Compression and kernels (WP5)**
- First create `experiments.compression.PruningShootout` with fixed trained reference models (digits MLP, shapes CNN, toy LM) saved once per seed, so every pruning and quantisation row starts from the same weights.
- 8.5.1, 8.5.2 and 8.5.4 add criteria and schedules around `WeightPruning`; they do not change it. 8.5.3 and 8.5.10 share `CurvatureProducts` (from wave 7) for gradients and Hessian-vector products.
- 8.5.5 reuses `language.SliceCompression`'s toy LM; 8.5.7 reuses `FineGrainedMoe`; 8.5.13 reuses `VisionTransformer`.
- 8.5.9 reuses `experimental.ml.KMeans`; 8.5.12's decompositions reuse `ThinSvd`.
- 8.5.14 → 8.5.15 → 8.5.18 share `KernelTuning` and one timing harness (warm-up, median of repeats, correctness check). They may only add kernels under `experimental.kernels`; any change to the core CPU backend's kernels goes in a separate core PR after measurement. 8.5.18's end-to-end row needs 8.5.1's block-pruned LM.
- 8.5.16 uses `AlphaZero`/`GumbelMcts` from `experimental.rl` as-is; 8.5.17 needs a read-only view of a core `Model`'s forward graph.

**Privacy, robustness, data, spiking and Ising (WP6)**
- First build `experimental.data.ClientSplits`. 8.6.1 → 8.6.2 → 8.6.3 → 8.6.4 share `FederatedShootout` and `FederatedAveraging`'s client simulation; one agent should do 8.6.1–8.6.2. 8.6.3's Per-FedAvg reuses `Maml`.
- 8.6.4's secure aggregation is the last defence row of 8.6.5. 8.6.6's LiRA is added as an auditor to `safety.PrivateTraining` and `interp.UnlearningComparison`; existing rows stay. 8.6.4 and 8.6.7 reuse `RdpAccountant`.
- 8.6.8's attacks are used by 8.6.9–8.6.12 as the empirical-robustness reference; build them first. 8.6.10 and 8.6.11 share certification code (margins, bound checks).
- 8.6.13 uses 8.6.14's leverage scores as one acquisition function. 8.6.15 adds rows to `vision.DataPruningExperiment` without changing the existing ones.
- 8.6.16 and 8.6.17 share `exotic.SpikingClassifier`; 8.6.17 adds an autograd surrogate next to `SurrogateSpike` and must not change its default.
- 8.6.18 is independent; it reads graphs from `SyntheticGraphs`.

**Cross-package**
- The ten new packages (`experimental.vision`, `.timeseries`, `.anomaly`, `.sciml`, `.serving`, `.agent`, `.compress`, `.kernels`, `.privacy` and `.robust`, plus the `experimental.data` additions) have no dependencies on each other, so all six work packages can start at once.
- Shared helpers to expose rather than copy: the Hungarian solver (`MinibatchOptimalTransport`; used by WP1 8.1.3 and 8.1.10), `StateSpaceFilter` (WP1 8.1.10, WP2 8.2.1/8.2.15, WP3 8.3.16), `CurvatureProducts` (WP5 8.5.3/8.5.10), `ThinSvd`/`SymmetricEigen` (WP3, WP5) and `KMeans` (WP5 8.5.9, WP6 8.6.15). Whichever WP lands first makes the method public; the others import it.
- WP1 8.1.6/8.1.7 and WP3 8.3.10 both produce flow-like fields; they share no code.
- WP2 8.2.17 (scattering) and WP1 8.1.16 (G-CNN) both add a low-data `ShapeImages` row; use the same 64-image split.
- WP4 8.4.4 and WP5 8.5.18 both add sparse compute paths; they share no code.
- WP6 8.6.12 (test-time adaptation) and WP1's dense tasks are independent; WP6 8.6.8–8.6.11 use only the classification models.
- Core-worthy items: 8.2.1 ETS/ARIMA (`brainiac-train` or a forecasting utility), 8.2.9 MiniRocket, 8.3.1 PINN residuals (needs the core autograd), 8.4.7 multi-LoRA batching and 8.4.9 the grammar mask cache (core `brainiac-llm` serving), 8.5.15 Winograd convolution and 8.5.18 block-sparse matmul (core CPU backend), 8.6.8 PGD/FGSM attacks (`brainiac-train`), and 8.6.9 randomized smoothing. Land them in a separate PR from the experiments.
