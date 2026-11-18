# Brainiac: wave 6 ideas

*Compiled 2026-11-17 against `claude/project-thread-8zvcl0` (main tree), plus every worktree under `.claude/worktrees/*`. Waves 1–5 implemented about 590 techniques; see `experiment-results.md`, `unimplemented-ideas.md` and `wave5-ideas.md`. Wave 5 was a 2024–2026 literature sweep. Wave 6 fills three remaining gaps:*

- *(a) well-known classic and foundational techniques that never got a class: classic ML, classic DL, classic RL, classic generative models, classic NLP pretraining, meta-learning, continual learning, AutoML, uncertainty, explainability, causal ML and fairness;*
- *(b) speculative or unconventional computing ideas that still run as a CPU toy;*
- *(c) 2025–2026 frontier items that wave 5 missed.*

*Every item was grepped over all `*.java` sources under `src/` in all modules, in the main tree and in all 24 worktrees: 2,075 unique files after de-duplication by content. The search was case-insensitive and used class names, paper names, author-coined terms and javadoc keywords. An item is listed only when nothing matched, or when the only matches were unrelated. Examples of unrelated matches: "GEM" matching `Generator`, "HAT" matching `ChatRoutes`, "SHAP" matching `shape`, "MADE" matching "made", "EM" matching `Embedding`, "Retrace" matching nothing but "trace", and "Spiral" matching the `SpiralExample` toy task.*

## How this was checked

- **Already implemented, so left out.** These classic techniques turned up in the systematic check but already have code:
  - Classic ML:
    - Gradient-boosted trees: core `GradientBoostedTrees`.
    - Hyperband and TPE: core `HyperparameterSearch`.
    - Logistic regression, factorization machines, DCN-V2, TabNet-lite and two-tower retrieval: `RecsysTabularBenchmarks`.
    - k-means, but only as codebook initialisation inside `VectorQuantizer` and `AdditiveQuant`. Clustering as a technique is still listed (1.6).
    - PCA, but only inside `SliceCompressor` (still listed as part of 1.7).
    - Boltzmann machines and Ising sampling: `PBitIsingMachine` and `DenoisingThermodynamicModel`.
    - Hopfield networks: `ModernHopfield`.
    - Hyperdimensional computing and Kanerva item memory: `Hypervector` and `HdcItemMemory`.
  - Classic DL:
    - Dropout, spatial dropout, DropPath and FractalNet: `Dropout`, `DropPath` and `FractalBlock`.
    - Capsules: `CapsuleLayer`.
    - NTM and DNC: `NeuralTuringMachine` and `DifferentiableNeuralComputer`.
    - ConvNeXt, ViT and MLP-Mixer: `VisionShootout`.
    - GroupNorm and RevIN, plus the normalisation zoo: `NormalizationShootout`.
    - FiLM: `Functional.film`.
    - Gumbel-softmax: `Functional.gumbelSoftmax`.
    - LSTM, GRU, sLSTM and mLSTM.
    - Focal loss, label smoothing, mixup and CutMix.
    - SAM, SWA, lottery tickets, Lookahead, Shampoo and SOAP.
  - Classic RL:
    - A2C: core `Ppo.a2c`.
    - C51 and QR-DQN: `RainbowDqn`.
    - Go-Explore, MuZero-lite, AlphaZero, QMIX/VDN (`ValueDecomposition`), GAIL, MaxEnt IRL, CQL, IQL, Decision Transformer, HER, SAC, TD3 and DDPG.
    - DreamerV3-style agent: `DreamerAgent`, which uses an MLP world model with no RSSM.
  - Classic generative:
    - WGAN-GP, spectral norm, hinge loss and the R1 penalty: core `Adversarial` and `GradientPenalty`.
    - RealNVP, PixelCNN and WaveNet.
    - VAE (free bits) and VQ-VAE.
    - DDPM/DDIM, flow matching and consistency models.
  - Classic NLP:
    - BPE, WordPiece, Unigram and SentencePiece tokenizers.
    - MLM: the baseline in `ElectraVsMaskedLm`.
    - Fill-in-the-middle (FIM).
  - Self-supervised:
    - SimCLR, BYOL, DINO, Barlow Twins, VICReg, MAE and I-JEPA-lite.
  - Graphs:
    - GCN, GAT, GraphSAGE, GIN and the graph autoencoder.
  - Meta-learning: MAML (first- and second-order), Reptile and prototypical networks.
  - Continual learning:
    - EWC and SI (`SynapticIntelligence`).
    - Progressive networks: `ContinualBrain.Protection.GROW_FROZEN`, which cites Rusu et al.
    - Reservoir replay: `ReservoirReplay`.
    - ContinualBackprop.
  - AutoML: DARTS (`DartsSearch`), AutoML-Zero, NEAT, HyperNEAT, CMA-ES and MAP-Elites.
  - Uncertainty:
    - Deep ensembles and MC-dropout ensembles: `PopulationEnsemble`.
    - Temperature scaling, ECE and BALD: `Calibration`.
    - Conformal prediction: `ConformalClassifier`.
    - Last-layer Laplace: `LastLayerLaplace`.
  - Explainability:
    - Saliency, gradient×input, integrated gradients and SmoothGrad: `Attribution`.
    - Activation patching, logit lens, SAEs, transcoders and attribution graphs.
  - Speculative computing:
    - Optical D2NN (`DiffractiveLayer`).
    - Analog noise-aware training (`AnalogDense`).
    - p-bits, logic-gate networks and reservoir computing (ESN and NG-RC).
    - LIF spiking with surrogate gradients, SpikeGPT and e-prop.
    - NCA and Lenia, KAN, neural ODE and CDE, FNO and liquid time-constant cells.
    - Predictive coding, active inference, Forward-Forward and equilibrium propagation.
  - 2025–26 frontier:
    - DAPO (dynamic sampling, token-level loss, overlong shaping) and Dr. GRPO: `GrpoTrainer` and `ArithmeticRl`.
    - V-JEPA 2-style planning: `JepaWorldModel`.
    - StreamingLLM attention sinks: `KvCache` and `DuoHeadCache`.
- **Considered but cut** (to stay within 6 × 18, or too close to an existing block):
  - Ape-X: its gain is wall-clock throughput across machines, and 3.2 A3C already covers asynchronous actors on a CPU.
  - Dion: its benefit is communication in sharded training, which `LowCommunicationTraining` already measures with DiLoCo/DeMo.
  - AdaMuon: too close to `NorMuon` / `AdaptiveMuon`.
  - Hamiltonian neural networks, PINNs and DeepONet: worth a later "scientific ML" package, but they are not foundational for this repo's themes.
  - Counterfactual explanations (Wachter et al.): folded into 5.14 as a third row.
  - Naive Bayes and plain kNN classification: trivial baselines, and `KnnIndex` exists.
  - IndRNN, Clockwork RNN and QRNN: superseded by the xLSTM / Mamba / delta-rule lineups.
  - Detection (DETR/YOLO) and segmentation beyond the U-Net toy: they need datasets the repo does not have.
  - Wav2vec, HuBERT and TTS: `TalkingDigits` covers the toy speech path.
  - ALBERT: its cross-layer sharing is close to `UniversalTransformer`.
- **CPU-toy filter.** Every item names a small task that already exists in `brainiac-experiments` (or a tiny new one) and a baseline already in the repo. The tasks used are:
  - `ToyTasks` spirals/digits, `ShapeImages`, `ToyImages`, `ToyPoints`, `SeasonalSeries` and `ChaoticSeries`.
  - `CountingTasks`, `ArithmeticTask`, `ChainTask` and `CuedTMaze`.
  - The `brainiac-rl` environments.
  - `SyntheticGraphs`, `SyntheticRecsys` and `SafetyToyLm`.

**Where an item should live:**

- `experimental.<pkg>`: a new block in `brainiac-experiments`.
- `experimental.ml`: a **new** package proposed for the classic ML estimators. They are non-neural, run on the host, and sit next to core `GradientBoostedTrees`.
- `experiments.<theme>`: a new or extended runnable experiment. `experiments.classic` is a **new** theme for WP1.
- **core-worthy**: belongs in a core module once measured.

## Summary

| # | Work package | Items |
|---|---|---|
| WP1 | Classic machine learning: kernels, probabilistic models, clustering, embeddings and filtering | 15 |
| WP2 | Classic deep learning: regularisers, normalisers, CNN/RNN architectures, set/memory models and NLP pretraining | 18 |
| WP3 | Classic and modern RL missing from `brainiac-rl` / `experimental.rl` | 18 |
| WP4 | Generative models: GAN/flow/score/EBM classics and the 2025–26 one-step frontier | 18 |
| WP5 | Meta-learning, continual learning, AutoML, uncertainty, explainability, causal ML and fairness | 18 |
| WP6 | Unconventional computing on a CPU, and the 2025–26 LLM frontier that wave 5 missed | 18 |
| | **Total** | **105** |

---

## WP1. Classic machine learning: kernels, probabilistic models, clustering, embeddings and filtering (15)

Target: a new `experimental.ml` package (host-side, double precision, no autograd needed), measured in a new `experiments.classic.ClassicMlShootout` and in the existing `RecsysTabularBenchmarks`, `ProbabilisticForecasting` and `SelfSupervisedBenchmark`. Baselines: logistic regression and a small MLP from the repo, core `GradientBoostedTrees`, and `LastLayerLaplace` / `PopulationEnsemble` for uncertainty rows.

| # | Item | One line | Lives in | Experiment vs baseline | Source |
|---|---|---|---|---|---|
| 1.1 | **Kernel SVM (SMO) + random Fourier features** | Soft-margin SVM solved by sequential minimal optimisation with RBF/polynomial kernels; RFF approximates the RBF kernel so a linear SVM scales. | `experimental.ml.KernelSvm`, `experimental.ml.RandomFourierFeatures` | ClassicMlShootout on `ToyTasks` spirals/circles: accuracy and train time against support-vector count, compared with logistic regression, an MLP and RFF + linear. | [Cortes & Vapnik 1995](https://link.springer.com/article/10.1007/BF00994018), [Rahimi & Recht 2007](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines) |
| 1.2 | **Gaussian-process regression and classification** | Exact GP with RBF/Matérn kernels, marginal-likelihood hyperparameter fitting, and Laplace-approximate GP classification. | `experimental.ml.GaussianProcess` | ProbabilisticForecasting-style 1-D regression with a gap: NLL, CRPS and coverage inside and outside the gap, compared with `LastLayerLaplace`, a deep ensemble and MC dropout. | [Rasmussen & Williams, GPML](http://gaussianprocess.org/gpml/) |
| 1.3 | **Sparse variational GP + deep kernel learning** | Inducing-point SVGP trained by minibatch ELBO; DKL puts an RBF GP on top of a learned MLP feature map. | `experimental.ml.SparseVariationalGp` (on 1.2) | 1.2's task scaled to 5k points, plus `SeasonalSeries`: NLL against inducing-point count, compared with the exact GP and with the MLP alone. | [arXiv:1309.6835](https://arxiv.org/abs/1309.6835), [arXiv:1511.02222](https://arxiv.org/abs/1511.02222) |
| 1.4 | **GP Bayesian optimisation + BOHB** | GP surrogate with expected improvement / UCB; BOHB replaces Hyperband's random sampling with a TPE model per budget. | core `HyperparameterSearch` (new `bayesOpt`, `bohb` methods); core-worthy | The existing hyperparameter test objectives: best score against trials and against total budget, compared with random search, `tpe` and `hyperband`. | [arXiv:1206.2944](https://arxiv.org/abs/1206.2944), [arXiv:1807.01774](https://arxiv.org/abs/1807.01774) |
| 1.5 | **Random forest, extra trees and isolation forest** | Bagged CART trees with feature subsampling; extremely randomised splits; isolation path length as an anomaly score. | `experimental.ml.RandomForest`, `experimental.ml.IsolationForest` | RecsysTabularBenchmarks `tabular`: accuracy and OOB error, compared with `GradientBoostedTrees` and an MLP. A planted-outlier set: AUROC compared with a k-NN distance score and an autoencoder reconstruction score. | [Breiman 2001](https://link.springer.com/article/10.1023/A:1010933404324), [Geurts et al. 2006](https://link.springer.com/article/10.1007/s10994-006-6226-1), [Liu et al. 2008](https://ieeexplore.ieee.org/document/4781136) |
| 1.6 | **k-means++ and Gaussian mixture via EM** | k-means++ seeding and Lloyd steps as a standalone clusterer; full-covariance GMM fit by EM with BIC model selection. | `experimental.ml.KMeans`, `experimental.ml.GaussianMixture` | `ToyPoints` blobs and anisotropic mixtures: adjusted Rand index and log-likelihood, compared with random-init k-means. The GMM is also used as a density baseline for `RealNvpFlow`. | [Arthur & Vassilvitskii 2007](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf), [Dempster et al. 1977](https://www.jstor.org/stable/2984875) |
| 1.7 | **PCA, kernel PCA and FastICA** | Standalone PCA (SVD) and RBF kernel PCA; FastICA recovers independent non-Gaussian sources. | `experimental.ml.ComponentAnalysis` (reuses core `ThinSvd`) | A cocktail-party mix of three `SeasonalSeries` signals: source-recovery correlation for ICA compared with PCA. Concentric circles: linear separability after kernel PCA compared with PCA. | [Hyvärinen 1999](https://ieeexplore.ieee.org/document/761722), [Schölkopf et al. 1998](https://direct.mit.edu/neco/article/10/5/1299/6193) |
| 1.8 | **t-SNE and UMAP** | Non-linear 2-D embeddings: t-SNE with Student-t affinities and early exaggeration; UMAP's fuzzy k-NN graph with cross-entropy layout. | `experimental.ml.NeighbourEmbedding` | SelfSupervisedBenchmark features of `ShapeImages`: trustworthiness and k-NN accuracy in 2-D, compared with PCA-2. Also gives the studio a projection view. | [van der Maaten & Hinton 2008](https://jmlr.org/papers/v9/vandermaaten08a.html), [arXiv:1802.03426](https://arxiv.org/abs/1802.03426) |
| 1.9 | **Hidden Markov models** | Forward–backward, Viterbi and Baum–Welch for discrete and Gaussian emissions. | `experimental.ml.HiddenMarkovModel` | A regime-switching series and the "dishonest casino": state-decoding accuracy and held-out likelihood, compared with a GRU tagger trained on the true states. | [Rabiner 1989](https://ieeexplore.ieee.org/document/18626) |
| 1.10 | **Linear-chain CRF (+ BiLSTM-CRF)** | A structured output layer with transition scores, trained by exact log-partition (forward algorithm) and decoded by Viterbi. | `experimental.nn.LinearChainCrf` | A toy NER/chunking task with hard label constraints (I- must follow B-): span F1 and illegal-transition rate, compared with per-token softmax on the same BiLSTM. | [Lafferty et al. 2001](https://repository.upenn.edu/cis_papers/159/), [arXiv:1508.01991](https://arxiv.org/abs/1508.01991) |
| 1.11 | **Bayesian networks: variable elimination, loopy BP and Gibbs** | Discrete factor graphs with exact inference on trees, loopy belief propagation on cycles, and Gibbs sampling. | `experimental.ml.FactorGraph` | Sprinkler / alarm networks and a 2-D Ising grid: marginal error against time, comparing the three methods with each other and with `PBitIsingMachine` sampling. | [arXiv:1301.6725](https://arxiv.org/abs/1301.6725) |
| 1.12 | **LDA topic model (collapsed Gibbs) + NMF** | Latent Dirichlet allocation by collapsed Gibbs sampling; non-negative matrix factorisation by multiplicative updates. | `experimental.ml.TopicModel` | A synthetic corpus with planted topics: topic-recovery cosine and held-out perplexity, compared with NMF and with k-means on bag-of-words. | [Blei et al. 2003](https://jmlr.org/papers/v3/blei03a.html), [Lee & Seung 1999](https://www.nature.com/articles/44565) |
| 1.13 | **Self-organising map + growing neural gas** | Kohonen's topology-preserving map, and Fritzke's incrementally grown graph of prototypes. | `experimental.ml.SelfOrganizingMap` | `ToyPoints` rings/S-curve: quantisation error and topographic error, compared with k-means (1.6) at equal prototypes. | [Kohonen 1990](https://ieeexplore.ieee.org/document/58325) |
| 1.14 | **Spectral clustering + DBSCAN** | Normalised-Laplacian spectral clustering; density-based clustering with noise points. | `experimental.ml.SpectralClustering`, `experimental.ml.Dbscan` | Two moons, rings and a noisy blob set: ARI and noise-point recall, compared with k-means and the GMM (1.6). | [arXiv:0711.0189](https://arxiv.org/abs/0711.0189), [Ester et al. 1996](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) |
| 1.15 | **Kalman filter, EKF and particle filter** | Linear-Gaussian filtering and smoothing; an extended KF for mild non-linearity; a bootstrap particle filter for multimodal posteriors. | `experimental.ml.StateSpaceFilter` | A noisy pendulum / bearing-only tracking toy: RMSE and NLL, compared with a GRU trained on the same trajectories. Also usable as a `ForecastingBenchmark` row. | [Welch & Bishop, Kalman filter intro](https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf), [Gordon et al. 1993](https://ieeexplore.ieee.org/document/210672) |

## WP2. Classic deep learning: regularisers, normalisers, CNN/RNN architectures, set/memory models and NLP pretraining (18)

Target: `experimental.nn`. Measurements go in `experiments.techniques.TrainingTricksExperiment` / `NormalizationShootout`, `experiments.vision.VisionShootout`, `experiments.memory`, `experiments.sequence` and `experiments.language`. The baseline is the plain CNN or MLP of the shootout, or core `TransformerBlock`, at the same parameter count and step budget.

| # | Item | One line | Lives in | Experiment vs baseline | Source |
|---|---|---|---|---|---|
| 2.1 | **DropConnect, concrete dropout and variational RNN dropout** | Drop weights rather than units; learn the dropout rate per layer by a concrete relaxation; tie the dropout mask across time steps in RNNs. | `experimental.nn.DropConnectDense`, `experimental.nn.ConcreteDropout` | TrainingTricksExperiment: few-shot noisy `ShapeImages` accuracy and ECE, compared with plain `Dropout`. A GRU LM on `CountingTasks`, compared with per-step masks. | [Wan et al. 2013](http://proceedings.mlr.press/v28/wan13.html), [arXiv:1705.07832](https://arxiv.org/abs/1705.07832), [arXiv:1512.05287](https://arxiv.org/abs/1512.05287) |
| 2.2 | **DropBlock + Shake-Shake / ShakeDrop** | Drop contiguous spatial blocks; mix residual branches with random per-sample coefficients forward and backward. | `experimental.nn.DropBlock2d`, `experimental.nn.ShakeResidualBlock` | VisionShootout CNN: test accuracy at a small data size, compared with spatial dropout and `DropPath`. | [arXiv:1810.12890](https://arxiv.org/abs/1810.12890), [arXiv:1705.07485](https://arxiv.org/abs/1705.07485), [arXiv:1802.02375](https://arxiv.org/abs/1802.02375) |
| 2.3 | **Batch renormalisation, weight normalisation and filter response norm** | BN with r/d correction for small or non-iid batches; the w = g·v/‖v‖ reparameterisation; batch-free FRN + TLU. | `experimental.nn.BatchRenorm`, `experimental.nn.WeightNormDense`, `experimental.nn.FilterResponseNorm` | A new CNN arm of NormalizationShootout at batch 2, 8 and 64: accuracy compared with `BatchNorm` and `GroupNorm`. | [arXiv:1702.03275](https://arxiv.org/abs/1702.03275), [arXiv:1602.07868](https://arxiv.org/abs/1602.07868), [arXiv:1911.09737](https://arxiv.org/abs/1911.09737) |
| 2.4 | **DenseNet + highway networks** | Dense blocks that concatenate every earlier feature map; gated carry/transform layers for very deep MLPs. | `experimental.nn.DenseBlock2d`, `experimental.nn.HighwayDense` | VisionShootout: accuracy against parameters, compared with the plain CNN and `ConvNeXtBlock`. A 30-layer MLP on spirals: trainability, compared with `SwishResidualBlock`. | [arXiv:1608.06993](https://arxiv.org/abs/1608.06993), [arXiv:1505.00387](https://arxiv.org/abs/1505.00387) |
| 2.5 | **Squeeze-excitation + MBConv (EfficientNet-lite)** | Channel attention by global pooling and gating; inverted-residual depthwise blocks; compound width/depth/resolution scaling. | `experimental.nn.SqueezeExcitation`, `experimental.nn.MbConvBlock` | VisionShootout: accuracy against FLOPs, compared with the plain CNN and ConvNeXt. A compound-scaling sweep compared with width-only scaling. | [arXiv:1709.01507](https://arxiv.org/abs/1709.01507), [arXiv:1905.11946](https://arxiv.org/abs/1905.11946), [arXiv:1801.04381](https://arxiv.org/abs/1801.04381) |
| 2.6 | **U-Net** | An encoder–decoder with skip connections at each scale, used for dense prediction and as the classic diffusion denoiser. | `experimental.nn.UNet2d` | A new `ShapeImages` segmentation mask task: IoU compared with an encoder–decoder without skips. `TinyDdpm`: sample distance with the U-Net compared with `MlpDenoiser` / `DiffusionTransformer`. | [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) |
| 2.7 | **Seq2seq with Bahdanau / Luong attention** | A GRU encoder–decoder with additive or multiplicative attention and teacher forcing. | `experimental.nn.Seq2SeqAttention` | Sequence reversal and toy date normalisation: exact-match accuracy against length, compared with seq2seq without attention and a 2-layer transformer encoder–decoder. | [arXiv:1409.0473](https://arxiv.org/abs/1409.0473), [arXiv:1508.04025](https://arxiv.org/abs/1508.04025) |
| 2.8 | **Pointer networks + pointer-generator copying** | Attention over input positions as the output vocabulary; a mixture of generate and copy. | `experimental.nn.PointerNetwork` | Sorting, convex hull of `ToyPoints`, and copy-heavy summaries on `CountingTasks`: accuracy at longer lengths than in training, compared with 2.7 and a transformer. | [arXiv:1506.03134](https://arxiv.org/abs/1506.03134), [arXiv:1704.04368](https://arxiv.org/abs/1704.04368) |
| 2.9 | **End-to-end memory networks + key-value memory networks** | Multi-hop soft attention over a sentence memory; separate key and value embeddings. | `experimental.nn.MemoryNetwork` | A bAbI-like 1/2/3-supporting-fact toy in `experiments.memory`: accuracy by hop count, compared with `NeuralTuringMachine`, `DifferentiableNeuralComputer` and a transformer. | [arXiv:1503.08895](https://arxiv.org/abs/1503.08895), [arXiv:1606.03126](https://arxiv.org/abs/1606.03126) |
| 2.10 | **NALU / neural arithmetic units** | Gated additive (NAC) and log-space multiplicative paths that extrapolate arithmetic outside the training range. | `experimental.nn.NeuralArithmeticUnit` | Arithmetic extrapolation (train on [0, 10], test on [10, 1000]): relative error, compared with MLPs using ReLU/SiLU and with `KanLayer`. | [arXiv:1808.00508](https://arxiv.org/abs/1808.00508) |
| 2.11 | **Deep Sets, Set Transformer and relation networks** | Permutation-invariant sum pooling; induced-set attention with pooling by attention; pairwise relation MLPs. | `experimental.nn.SetEncoder` (DeepSets, ISAB/PMA), `experimental.nn.RelationNetwork` | Set tasks (max, unique count, amortised clustering on `ToyPoints`) and Sort-of-CLEVR-lite on `ShapeImages`: accuracy, compared with an MLP on the sorted input and with `EquivariantGraphLayer`. | [arXiv:1703.06114](https://arxiv.org/abs/1703.06114), [arXiv:1810.00825](https://arxiv.org/abs/1810.00825), [arXiv:1706.01427](https://arxiv.org/abs/1706.01427) |
| 2.12 | **Perceiver / Perceiver IO** | A small latent array cross-attends to a large input, then self-attends in latent space; outputs come from query cross-attention. | `experimental.nn.PerceiverEncoder` | VisionShootout on raw pixels and a long-sequence copy task: accuracy against FLOPs and latent size, compared with ViT and `ChunkedCrossAttention`. | [arXiv:2103.03206](https://arxiv.org/abs/2103.03206), [arXiv:2107.14795](https://arxiv.org/abs/2107.14795) |
| 2.13 | **Slot attention** | Iterative competitive attention binds input features to K slots, trained by reconstruction. | `experimental.nn.SlotAttention` | Multi-shape `ShapeImages` scenes: object-discovery ARI and reconstruction, compared with a plain autoencoder and with `CapsuleLayer` routing (`OverlappingShapeCapsules`). | [arXiv:2006.15055](https://arxiv.org/abs/2006.15055) |
| 2.14 | **word2vec (SGNS), GloVe and fastText** | Static word vectors: skip-gram with negative sampling; weighted least squares on log co-occurrence; character n-gram subwords. | `experimental.llm.StaticWordEmbeddings` | A synthetic-grammar corpus with planted analogies: analogy accuracy, similarity correlation and OOV robustness (fastText). Downstream: warm-starting core `Embedding` (which already accepts pretrained vectors), compared with random init. | [arXiv:1310.4546](https://arxiv.org/abs/1310.4546), [Pennington et al. 2014](https://nlp.stanford.edu/pubs/glove.pdf), [arXiv:1607.04606](https://arxiv.org/abs/1607.04606) |
| 2.15 | **ELMo-style biLM features** | Forward and backward LSTM LMs whose layer-mixed hidden states (learned scalar mix) feed a downstream tagger. | `experiments.language.ContextualFeatures` | A downstream tagging task from 1.10: F1 with frozen ELMo features, compared with 2.14 static vectors and with the frozen MLM encoder of `ElectraVsMaskedLm`. | [arXiv:1802.05365](https://arxiv.org/abs/1802.05365) |
| 2.16 | **T5 span corruption** | Replace random spans with sentinel tokens; an encoder–decoder (or decoder) predicts the spans. | `experimental.llm.SpanCorruption` (beside `FimTransform`) | ElectraVsMaskedLm: downstream probe accuracy at equal compute, compared with MLM (15% tokens), ELECTRA and FIM. | [arXiv:1910.10683](https://arxiv.org/abs/1910.10683) |
| 2.17 | **XLNet permutation LM** | Autoregressive training over random factorisation orders with two-stream (content/query) attention. | `experimental.llm.PermutationLm` | ElectraVsMaskedLm: probe accuracy and infill accuracy, compared with MLM, ELECTRA and FIM. | [arXiv:1906.08237](https://arxiv.org/abs/1906.08237) |
| 2.18 | **ULMFiT fine-tuning recipe** | Discriminative per-layer learning rates, slanted triangular schedule and gradual unfreezing. | `experimental.train.DiscriminativeFineTuning` | PeftShootout / SentimentExample transfer: target accuracy and source forgetting, compared with full fine-tuning at one rate and with LoRA. | [arXiv:1801.06146](https://arxiv.org/abs/1801.06146) |

## WP3. Classic and modern RL missing from `brainiac-rl` / `experimental.rl` (18)

Target: `experimental.rl`. Measurements go in `experiments.control` (`RainbowAblation`, `ContinuousControlComparison`, `OfflineRlComparison`, `GoalsImitationAndTeams`) and `experiments.agents` (`CuriosityExploration`, `HardExploration`, `DreamingAgent`). Baselines are core `Ppo` / `Ppo.a2c` and `Dqn`, `RainbowDqn`, `Sac` and `DreamerAgent` at equal environment steps, over at least 3 seeds.

| # | Item | One line | Lives in | Experiment vs baseline | Source |
|---|---|---|---|---|---|
| 3.1 | **TRPO** | KL-constrained policy step via a Fisher-vector product, conjugate gradient and backtracking line search. | `experimental.rl.Trpo` | ContinuousControlComparison (Pendulum) and CartPole: return and per-update KL, compared with PPO and A2C. | [arXiv:1502.05477](https://arxiv.org/abs/1502.05477) |
| 3.2 | **A3C (asynchronous actor-learners)** | Several threads each run an environment and apply Hogwild gradient updates to shared parameters. | `experimental.rl.AsyncActorCritic` | CartPole / GridWorld: return against wall-clock time and steps, compared with synchronous `Ppo.a2c` on the same core count. | [arXiv:1602.01783](https://arxiv.org/abs/1602.01783) |
| 3.3 | **IMPALA V-trace (+ Retrace)** | Decoupled actors with stale policies; truncated-importance-weighted V-trace targets correct the lag. Retrace(λ) is the off-policy Q counterpart. | `experimental.rl.VTraceLearner` | Actors that lag the learner by 0 to 32 updates: return against lag, compared with uncorrected A2C and with 1-step IS. Retrace is compared with n-step Q in `RainbowAblation`. | [arXiv:1802.01561](https://arxiv.org/abs/1802.01561), [arXiv:1606.02647](https://arxiv.org/abs/1606.02647) |
| 3.4 | **R2D2 (recurrent replay with burn-in)** | LSTM Q-network trained from replayed sequences, with stored recurrent state and a burn-in prefix. | `experimental.rl.RecurrentReplayDqn` | `CuedTMaze` and a flickering CartPole (POMDP): success rate, compared with frame-stacked `Dqn` and with zero-state replay without burn-in. | [Kapturowski et al. 2019](https://openreview.net/forum?id=r1lyTjAqYX) |
| 3.5 | **IQN (+ CVaR risk-sensitive acting)** | Quantile embedding with sampled τ gives a full return distribution; acting on CVaR_α gives risk aversion. | `experimental.rl.RainbowDqn` (new `Distribution.IMPLICIT`) | RainbowAblation: return compared with C51 and QR-DQN. A cliff-walk variant: rate of falling off the cliff at α = 0.25, compared with risk-neutral acting. | [arXiv:1806.06923](https://arxiv.org/abs/1806.06923) |
| 3.6 | **NGU / Agent57** | An episodic k-NN novelty bonus modulated by RND lifelong novelty; a family of (β, γ) policies chosen by a meta-controller bandit. | `experimental.rl.NeverGiveUp` (reuses `RandomNetworkDistillation`, `Bandit`) | CuriosityExploration / HardExploration: cells visited and success, compared with RND, ICM and `GoExplore`. | [arXiv:2002.06038](https://arxiv.org/abs/2002.06038), [arXiv:2003.13350](https://arxiv.org/abs/2003.13350) |
| 3.7 | **PlaNet (RSSM + latent CEM)** | Recurrent state-space model with deterministic and stochastic paths, trained by ELBO with latent overshooting; CEM planning in latent space. | `experimental.rl.RecurrentStateSpaceModel`, `experimental.rl.LatentPlanner` | DreamingAgent on pixel-like `GridWorld` / CartPole observations: return and multi-step prediction error, compared with `MlpWorldModel` + `CemPlanner` and `DreamerAgent`. | [arXiv:1811.04551](https://arxiv.org/abs/1811.04551) |
| 3.8 | **TD-MPC2** | A decoder-free latent model trained by TD + consistency losses, planned with MPPI and a learned terminal value. | `experimental.rl.TdMpc` | ContinuousControlComparison and `DifferentiablePhysicsControl` reach: return at 5k–20k steps, compared with SAC, `CemPlanner` and 3.7. | [arXiv:2310.16828](https://arxiv.org/abs/2310.16828) |
| 3.9 | **Dreamer 4 (shortcut-forcing world model)** | A transformer world model trained with shortcut forcing for few-step generation; the agent learns its policy entirely in imagination from offline data. | `experimental.rl.ShortcutWorldModel` (reuses `MeanFlow` shortcut targets) | DreamingAgent / DiffusionDreaming with an offline dataset only: real-env return and rollout fidelity against denoising steps, compared with `DiffusionWorldModel` (DIAMOND) and `DreamerAgent`. | [arXiv:2509.24527](https://arxiv.org/abs/2509.24527) |
| 3.10 | **REDQ / DroQ (high update-to-data ratio)** | An ensemble of Q-functions with a random min-subset (REDQ), or dropout plus layer-norm Qs (DroQ), so that UTD 10–20 does not overfit. | `experimental.rl.HighUtdSac` (extends core `Sac`) | ContinuousControlComparison: return at 2k–10k steps and Q bias, compared with SAC at UTD 1 and 20 and with `ResidualRlNetworks` SimBa. | [arXiv:2101.05982](https://arxiv.org/abs/2101.05982), [arXiv:2110.02034](https://arxiv.org/abs/2110.02034) |
| 3.11 | **MPO (maximum a-posteriori policy optimisation)** | An EM-style policy improvement: a non-parametric E-step reweights sampled actions by Q, then a KL-constrained M-step fit. | `experimental.rl.Mpo` | ContinuousControlComparison: return and stability across seeds, compared with SAC and PPO. | [arXiv:1806.06920](https://arxiv.org/abs/1806.06920) |
| 3.12 | **Option-critic** | Intra-option policies, termination functions and a policy over options, all learned end to end by the option-critic gradient theorems. | `experimental.rl.OptionCritic` | Four-rooms `GridWorld` with a goal switch: steps to re-adapt and option duration, compared with A2C and flat Q-learning. | [arXiv:1609.05140](https://arxiv.org/abs/1609.05140) |
| 3.13 | **Successor features + generalised policy improvement** | ψ(s, a) learns expected discounted features; Q_w = ψ·w transfers instantly to new rewards, and GPI picks the best of several policies. | `experimental.rl.SuccessorFeatures` | GridWorld with new reward weights at test time: zero-shot return, compared with retraining `Dqn` and with `ContrastiveGoalRl`. | [arXiv:1606.05312](https://arxiv.org/abs/1606.05312) |
| 3.14 | **Constrained RL: CPO and PPO-Lagrangian** | Maximise return subject to E[cost] ≤ d, by a trust-region projection (CPO) or a learned Lagrange multiplier. | `experimental.rl.ConstrainedPolicyOptimization` | A GridWorld / reach variant with hazard cells: return and constraint violation over training, compared with PPO with fixed reward shaping. | [arXiv:1705.10528](https://arxiv.org/abs/1705.10528), [Ray et al. 2019](https://cdn.openai.com/safexp-short.pdf) |
| 3.15 | **Reward machines (QRM / CRM)** | The reward is a finite-state machine over labelled events; Q-learning on the product with counterfactual experience sharing. | `experimental.rl.RewardMachine` | "Get coffee then mail then office" in GridWorld: success and sample efficiency, compared with Q-learning given the history-free reward and an LSTM `Dqn`. | [arXiv:2010.03950](https://arxiv.org/abs/2010.03950) |
| 3.16 | **Meta-RL: RL² and VariBAD** | A recurrent policy whose hidden state learns the RL algorithm across episodes; VariBAD adds a VAE belief over the task. | `experimental.rl.MetaRlAgent` | Bandit and goal-in-unknown-corner GridWorld task distributions: return in the first episodes of a new task, compared with `Maml`-adapted PPO and with PPO from scratch. | [arXiv:1611.02779](https://arxiv.org/abs/1611.02779), [arXiv:1910.08348](https://arxiv.org/abs/1910.08348) |
| 3.17 | **Algorithm distillation (in-context RL)** | A causal transformer trained on whole learning histories of a source RL algorithm improves in context, with no weight updates. | `experimental.rl.AlgorithmDistillation` (reuses `DecisionTransformer` code) | Bandit and dark-room tasks: in-context return curve, compared with the source learner and with an expert-only Decision Transformer. | [arXiv:2210.14215](https://arxiv.org/abs/2210.14215) |
| 3.18 | **Gumbel MuZero / Gumbel AlphaZero** | Policy improvement guaranteed with few simulations: Gumbel-top-k root sampling, sequential halving and a completed-Q policy target. | `experimental.rl.Mcts` (new Gumbel root) | SelfPlayArena (TicTacToe, ConnectThree): Elo against simulations n = 2–32, compared with PUCT `AlphaZero` / `MuZeroLite`. | [Danihelka et al. 2022](https://openreview.net/forum?id=bERaNdoegnO) |

## WP4. Generative models: GAN/flow/score/EBM classics and the 2025–26 one-step frontier (18)

Target: `experimental.nn` and `experiments.generative`. Measurements use the existing `TinyDdpm` / `RectifiedFlowExperiment` / `OneStepGenerators` harnesses on `ToyImages`, `ShapeImages` and `ToyPoints`: sliced Wasserstein, real share and distinct count. Baselines: DDPM/DDIM, flow matching with Heun, `MeanFlow`, `ConsistencyModel` and `RealNvpFlow`.

| # | Item | One line | Lives in | Experiment vs baseline | Source |
|---|---|---|---|---|---|
| 4.1 | **pix2pix + CycleGAN** | Paired translation with a U-Net generator and a PatchGAN discriminator; unpaired translation with two generators and cycle consistency. | `experiments.generative.ImageTranslation` (uses 2.6 U-Net and core `Adversarial`) | ShapeImages edges → filled shapes, and squares ↔ circles: pixel L1 (paired), shape-classifier accuracy (unpaired), compared with an L1-only U-Net. | [arXiv:1611.07004](https://arxiv.org/abs/1611.07004), [arXiv:1703.10593](https://arxiv.org/abs/1703.10593) |
| 4.2 | **StyleGAN-lite** | A mapping network to w, modulated/demodulated convolutions (or AdaIN), noise injection, style mixing and path-length regularisation. | `experimental.nn.StyleGenerator` | ToyImages: sliced distance and perceptual path length, compared with a DCGAN-style generator under the same `Adversarial` + R1 recipe. | [arXiv:1812.04948](https://arxiv.org/abs/1812.04948), [arXiv:1912.04958](https://arxiv.org/abs/1912.04958) |
| 4.3 | **VQ-GAN** | A VQ autoencoder trained with perceptual and patch-adversarial losses so codes stay sharp at high compression. | `experimental.nn.VqGan` (extends `VectorQuantizer`) | ImageAutoencoders: reconstruction sharpness and code usage at 4× fewer codes, compared with VQ-VAE. Downstream `MaskGit` quality on its tokens. | [arXiv:2012.09841](https://arxiv.org/abs/2012.09841) |
| 4.4 | **Glow + MAF / neural spline flows** | ActNorm, invertible 1×1 convolutions and affine couplings; masked autoregressive flows; monotone rational-quadratic spline transforms. | `experimental.nn.GlowFlow`, `experimental.nn.SplineFlow` | ToyPoints (two moons, rings) and ToyImages: test NLL in bits/dim, compared with `RealNvpFlow`, `TransformerFlow` (TarFlow) and the GMM (1.6). | [arXiv:1807.03039](https://arxiv.org/abs/1807.03039), [arXiv:1705.07057](https://arxiv.org/abs/1705.07057), [arXiv:1906.04032](https://arxiv.org/abs/1906.04032) |
| 4.5 | **FFJORD (continuous normalising flow)** | A neural-ODE flow with exact density via the instantaneous change of variables and Hutchinson trace estimation. | `experimental.nn.ContinuousFlow` (on core `NeuralOde` / `OdeSolver`) | ToyPoints: NLL and function evaluations, compared with 4.4 and with flow matching (whose likelihood is computed the same way). | [arXiv:1810.01367](https://arxiv.org/abs/1810.01367) |
| 4.6 | **Score SDE (VP/VE, predictor–corrector, probability-flow ODE)** | Continuous-time denoising score matching; reverse-SDE sampling with Langevin corrector steps; exact likelihood via the ODE. | `experimental.nn.ScoreSde` | TinyDdpm harness: quality against function evaluations for PC, the ODE and DDPM/DDIM, plus likelihood compared with 4.5. | [arXiv:2011.13456](https://arxiv.org/abs/2011.13456) |
| 4.7 | **EDM preconditioning + DPM-Solver++** | Karras c_skip/c_out/c_in/c_noise scaling, log-normal σ training and a 2nd-order Heun sampler; a multistep exponential-integrator solver. | `experimental.nn.EdmDenoiser`, `experimental.nn.DpmSolver` | DiffusionTargets / TinyDdpm: quality at 5–20 NFE, compared with DDIM and flow-matching Heun. | [arXiv:2206.00364](https://arxiv.org/abs/2206.00364), [arXiv:2211.01095](https://arxiv.org/abs/2211.01095) |
| 4.8 | **JEM (classifier as energy model, SGLD)** | Reinterpret a classifier's logits as an energy over x; train with persistent SGLD samples plus cross-entropy. | `experimental.train.JointEnergyModel` | ShapeImages: accuracy, ECE (`Calibration`), OOD AUROC and sample quality, compared with the plain classifier and with `EnergyRefiner`. | [arXiv:1912.03263](https://arxiv.org/abs/1912.03263) |
| 4.9 | **Latent diffusion vs representation-autoencoder latents (RAE)** | Diffuse in a VAE latent (LDM), or in the frozen latent of a pretrained representation encoder with a trained decoder (RAE). | `experiments.generative.LatentDiffusion` | ToyImages: quality and training steps to a target, compared with pixel-space `TinyDdpm`. RAE uses a frozen `SelfSupervisedBenchmark` encoder; compare with `RepaExperiment`. | [arXiv:2112.10752](https://arxiv.org/abs/2112.10752), [arXiv:2510.11690](https://arxiv.org/abs/2510.11690) |
| 4.10 | **ControlNet (zero-convolution conditioning)** | A trainable copy of the denoiser encoder, fed a control map and joined through zero-initialised layers into a frozen base model. | `experimental.nn.ControlBranch` | TinyDdpm with an edge or position map as control: control adherence (IoU) and base-quality retention, compared with fine-tuning the full model and with concatenating the condition. | [arXiv:2302.05543](https://arxiv.org/abs/2302.05543) |
| 4.11 | **Distribution matching distillation (DMD)** | A one-step generator trained by the difference between real and fake score estimates, plus a regression loss. | `experimental.train.DistributionMatchingDistiller` | OneStepGenerators: 1-step quality, compared with `MeanFlow`, shortcut models, `ConsistencyModel` and reflow. | [arXiv:2311.18828](https://arxiv.org/abs/2311.18828) |
| 4.12 | **IWAE, β-VAE and ladder (hierarchical) VAE** | A tighter importance-weighted bound; KL-weighted disentanglement; top-down hierarchical latents. | `experimental.nn.HierarchicalVae` (extends `ImageAutoencoders`' VAE) | ImageAutoencoders: test log-likelihood (IWAE-5000), active units, and a disentanglement score on ShapeImages factors (position, shape), compared with the plain VAE. | [arXiv:1509.00519](https://arxiv.org/abs/1509.00519), [β-VAE](https://openreview.net/forum?id=Sy2fzU9gl), [arXiv:1602.02282](https://arxiv.org/abs/1602.02282) |
| 4.13 | **Discrete flow matching** | Probability paths over discrete tokens with learned per-token velocities (corrector schedulers), generalising masked diffusion. | `experimental.llm.DiscreteFlowMatching` | TextDiffusionVsAutoregressive / `BlockDiffusionText`: perplexity bound and quality at fixed steps, compared with `MaskedDiffusion` and the AR LM. | [arXiv:2407.15595](https://arxiv.org/abs/2407.15595) |
| 4.14 | **Bayesian flow networks** | Generation by iteratively updating the parameters of a factorised distribution with noisy observations, in continuous form for both discrete and continuous data. | `experimental.nn.BayesianFlowNetwork` | Binarised ToyImages and CountingTasks text: NLL and quality at 10–100 steps, compared with `MaskedDiffusion` and DDPM. | [arXiv:2308.07037](https://arxiv.org/abs/2308.07037) |
| 4.15 | **Stochastic interpolants / diffusion Schrödinger bridge matching** | Learn a drift between two arbitrary distributions (not noise to data) with a stochastic interpolant, or by iterative Markovian fitting (DSBM). | `experimental.nn.BridgeMatching` (beside core `FlowMatching`) | ToyPoints moons ↔ rings, and ShapeImages squares ↔ circles: transport cost and target fit, compared with rectified flow + reflow and with CycleGAN (4.1). | [arXiv:2303.08797](https://arxiv.org/abs/2303.08797), [arXiv:2303.16852](https://arxiv.org/abs/2303.16852) |
| 4.16 | **Inductive moment matching + sCM** | IMM: a single-stage few-step generator trained by an MMD between model outputs at two times. sCM: a stable continuous-time consistency objective (TrigFlow). | `experimental.nn.MomentMatchingGenerator`, `experimental.nn.ContinuousConsistency` | OneStepGenerators: 1/2/4-step quality and training stability across seeds, compared with `MeanFlow`, discrete `ConsistencyModel` and DMD (4.11). | [arXiv:2503.07565](https://arxiv.org/abs/2503.07565), [arXiv:2410.11081](https://arxiv.org/abs/2410.11081) |
| 4.17 | **Energy matching** | One time-independent scalar potential whose gradient acts as an OT flow far from data and as a Boltzmann energy near it. | `experimental.nn.EnergyMatching` | ToyPoints and ToyImages: quality and likelihood proxy, compared with flow matching, `EquilibriumMatching` and JEM (4.8). | [arXiv:2504.10612](https://arxiv.org/abs/2504.10612) |
| 4.18 | **Drifting models** | A one-step generator trained so that a drift field (attraction to data, repulsion between samples) reaches equilibrium, without an ODE or SDE at inference. | `experimental.train.DriftingGenerator` | OneStepGenerators: 1-step quality and mode coverage on ToyPoints mixtures, compared with `MeanFlow`, IMM (4.16) and a GAN. | [arXiv:2602.04770](https://arxiv.org/abs/2602.04770) |

## WP5. Meta-learning, continual learning, AutoML, uncertainty, explainability, causal ML and fairness (18)

Target: `experimental.train` and `experimental.nn`. Measurements go in:

- `experiments.techniques.MetaLearningExperiment`, with MAML, Reptile and prototypical networks as baselines.
- `experiments.learning.ContinualLearningExperiment` (permuted digits) and `SparseMemoryFacts`, with EWC, SI, `GROW_FROZEN` and `ReservoirReplay` as baselines.
- `HyperparameterSearch` / `DartsSearch`.
- A new `experiments.interp.ExplainabilityBenchmark`, which plants a known ground-truth feature.
- A new `experiments.safety.CausalAndFairness`.

| # | Item | One line | Lives in | Experiment vs baseline | Source |
|---|---|---|---|---|---|
| 5.1 | **SNAIL** | A meta-learner made of temporal convolutions interleaved with causal attention over the episode history. | `experimental.nn.SnailLearner` | MetaLearningExperiment few-shot shapes and sine: accuracy and MSE, compared with prototypical networks and MAML. | [arXiv:1707.03141](https://arxiv.org/abs/1707.03141) |
| 5.2 | **LEO (latent embedding optimisation)** | Inner-loop adaptation in a low-dimensional latent that a relation-net decoder maps to classifier weights. | `experimental.train.LatentEmbeddingOptimization` | MetaLearningExperiment classification: 1/5-shot accuracy and inner-loop stability, compared with MAML (2nd order) and ANIL (5.4). | [arXiv:1807.05960](https://arxiv.org/abs/1807.05960) |
| 5.3 | **CNAPs (conditional neural adaptive processes)** | A task encoder produces FiLM parameters for a shared feature extractor, plus an amortised linear head; no gradient steps at test time. | `experimental.nn.Cnaps` (uses `Functional.film`) | MetaLearningExperiment, with a mixed-domain episode set (shapes + digits): accuracy and adaptation time, compared with MAML and prototypical networks. | [arXiv:1906.07697](https://arxiv.org/abs/1906.07697) |
| 5.4 | **Matching networks, ANIL and Meta-SGD** | Attention over support embeddings; MAML adapting only the head; MAML with learned per-parameter inner learning rates. | `experimental.train.Maml` (ANIL and Meta-SGD flags), `experimental.nn.MatchingNetwork` | MetaLearningExperiment: accuracy and compute, compared with MAML, Reptile and prototypical networks. | [arXiv:1606.04080](https://arxiv.org/abs/1606.04080), [arXiv:1909.09157](https://arxiv.org/abs/1909.09157), [arXiv:1707.09835](https://arxiv.org/abs/1707.09835) |
| 5.5 | **Conditional and attentive neural processes** | Amortised function regression: encode the context set, then decode target predictive distributions; ANP adds cross-attention. | `experimental.nn.NeuralProcess` | MetaLearningExperiment sine regression: NLL and calibration, compared with MAML and with the GP (1.2) as the gold standard. | [arXiv:1807.01613](https://arxiv.org/abs/1807.01613), [arXiv:1901.05761](https://arxiv.org/abs/1901.05761) |
| 5.6 | **LwF, iCaRL and DER++** | Distil the old model's outputs on new data (LwF); exemplar herding plus nearest-mean classification (iCaRL); replay with stored logits (DER++). | `experimental.train.DistillationReplay` | A class-incremental split of ContinualLearningExperiment: average accuracy and forgetting, compared with EWC, SI, `ReservoirReplay` and `GROW_FROZEN`. | [arXiv:1606.09282](https://arxiv.org/abs/1606.09282), [arXiv:1611.07725](https://arxiv.org/abs/1611.07725), [arXiv:2004.07211](https://arxiv.org/abs/2004.07211) |
| 5.7 | **GEM / A-GEM** | Project the new-task gradient so it does not increase loss on episodic memories (a QP for GEM; one averaged constraint for A-GEM). | `experimental.train.GradientEpisodicMemory` | ContinualLearningExperiment: accuracy, backward transfer and time per step, compared with replay and EWC at the same memory size. | [arXiv:1706.08840](https://arxiv.org/abs/1706.08840), [arXiv:1812.00420](https://arxiv.org/abs/1812.00420) |
| 5.8 | **PackNet / HAT** | Iterative prune-and-freeze of a per-task weight subset (PackNet); learned hard attention masks per task (HAT). | `experimental.train.ParameterIsolation` (reuses `WeightPruning`) | ContinualLearningExperiment: accuracy against capacity used per task, compared with `GROW_FROZEN` (progressive networks) and EWC. | [arXiv:1711.05769](https://arxiv.org/abs/1711.05769), [arXiv:1801.01423](https://arxiv.org/abs/1801.01423) |
| 5.9 | **Self-distillation fine-tuning (SDFT)** | The model, conditioned on a demonstration in context, acts as its own teacher; on-policy distillation into the unconditioned model learns new skills with little forgetting. | `experimental.llm.SelfDistillationFineTuner` (reuses `OnPolicyDistiller`) | SparseMemoryFacts and ForgettingVsKl: new-fact accuracy against background retention, compared with SFT, `SparseMemoryFinetuning` and on-policy RL (RL's Razor). | [arXiv:2601.19897](https://arxiv.org/abs/2601.19897) |
| 5.10 | **Population-based training** | Train a population in parallel; periodically copy the weights of the best (exploit) and perturb their hyperparameters (explore), giving schedules. | core `HyperparameterSearch` (new `pbt`); core-worthy | The hyperparameter test objectives and a learning-rate-sensitive `TrainingTricksExperiment` task: final score against compute, compared with Hyperband, TPE and 1.4 BO. | [arXiv:1711.09846](https://arxiv.org/abs/1711.09846) |
| 5.11 | **ENAS + zero-cost NAS proxies** | A weight-sharing supernet with an RL controller; architectures scored at initialisation by synflow, grad-norm or NASWOT. | `experimental.train.WeightSharingNas`, `experimental.train.ZeroCostProxies` | `TransformerBlockSearch` / DartsSearch space: rank correlation with fully trained accuracy and search cost, compared with DARTS and random search. | [arXiv:1802.03268](https://arxiv.org/abs/1802.03268), [arXiv:2101.08134](https://arxiv.org/abs/2101.08134) |
| 5.12 | **SWAG, Bayes by backprop and SGLD** | A Gaussian over SGD iterates (diagonal + low-rank); a mean-field variational posterior over weights; Langevin-noise posterior sampling. | `experimental.train.WeightPosterior` (extends `StochasticWeightAveraging`) | The 1.2 regression gap and ShapeImages OOD: NLL, ECE and OOD AUROC, compared with deep ensembles, MC dropout, `LastLayerLaplace` and the GP. | [arXiv:1902.02476](https://arxiv.org/abs/1902.02476), [arXiv:1505.05424](https://arxiv.org/abs/1505.05424), [Welling & Teh 2011](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf) |
| 5.13 | **Evidential deep learning + SNGP / DUQ** | Dirichlet evidence outputs (one forward pass); distance-aware single models (spectral norm + a random-feature GP head, or RBF centroids). | `experimental.nn.EvidentialHead`, `experimental.nn.DistanceAwareHead` | Same as 5.12: OOD AUROC and ECE per forward pass, compared with a deep ensemble at 5× cost. | [arXiv:1806.01768](https://arxiv.org/abs/1806.01768), [arXiv:2006.10108](https://arxiv.org/abs/2006.10108), [arXiv:2003.02037](https://arxiv.org/abs/2003.02037) |
| 5.14 | **KernelSHAP, LIME and counterfactual explanations** | Model-agnostic attribution from Shapley-weighted regression on coalitions; local surrogate linear models; minimal input changes that flip a prediction. | `experimental.nn.Attribution` (new methods) or `experimental.train.ModelAgnosticExplainer` | ExplainabilityBenchmark on tabular data with a planted rule: ground-truth feature recovery and deletion AUC, compared with integrated gradients and SmoothGrad. Also run on `GradientBoostedTrees`. | [arXiv:1705.07874](https://arxiv.org/abs/1705.07874), [arXiv:1602.04938](https://arxiv.org/abs/1602.04938), [arXiv:1711.00399](https://arxiv.org/abs/1711.00399) |
| 5.15 | **Grad-CAM and TCAV** | Class-activation maps from gradient-weighted conv features; testing with concept activation vectors (a linear probe direction plus a directional-derivative score). | `experimental.nn.GradCam`, `experimental.nn.ConceptActivationVectors` (reuses `LinearProbe`) | ExplainabilityBenchmark on ShapeImages with a spurious corner marker: pointing-game accuracy and the TCAV score for the "marker" concept, compared with saliency and integrated gradients. | [arXiv:1610.02391](https://arxiv.org/abs/1610.02391), [arXiv:1711.11279](https://arxiv.org/abs/1711.11279) |
| 5.16 | **Influence functions and TracIn** | Estimate each training example's effect on a test loss with inverse-Hessian-vector products, or by summing gradient dot products over checkpoints. | `experimental.train.TrainingDataAttribution` | DataPruningExperiment / ShapleyPruning: flipped-label detection and leave-k-out correlation, compared with EL2N and `InRunShapley`. | [arXiv:1703.04730](https://arxiv.org/abs/1703.04730), [arXiv:2002.08484](https://arxiv.org/abs/2002.08484) |
| 5.17 | **Causal effect estimation + NOTEARS** | IPW and doubly robust (AIPW) ATE; TARNet/DragonNet CATE heads; continuous DAG structure learning with an acyclicity constraint. | `experimental.ml.CausalEstimators`, `experimental.nn.TreatmentEffectNet`, `experimental.ml.Notears` | CausalAndFairness on a synthetic SCM with confounding: ATE bias, PEHE and SHD of the recovered graph, compared with a naive difference in means, an S-learner MLP and the PC-skeleton baseline. | [arXiv:1606.03976](https://arxiv.org/abs/1606.03976), [arXiv:1906.02120](https://arxiv.org/abs/1906.02120), [arXiv:1803.01422](https://arxiv.org/abs/1803.01422) |
| 5.18 | **Fairness and robustness to spurious correlation** | Reweighing, adversarial debiasing and equalised-odds post-processing; IRM and GroupDRO for worst-group accuracy. | `experimental.train.FairnessInterventions`, `experimental.train.GroupRobustTraining` | CausalAndFairness with a protected attribute and a colored-shapes spurious split: demographic-parity gap, equalised-odds gap and worst-group accuracy, compared with ERM. | [arXiv:1801.07593](https://arxiv.org/abs/1801.07593), [arXiv:1610.02413](https://arxiv.org/abs/1610.02413), [arXiv:1907.02893](https://arxiv.org/abs/1907.02893), [arXiv:1911.08731](https://arxiv.org/abs/1911.08731) |

## WP6. Unconventional computing on a CPU, and the 2025–26 LLM frontier that wave 5 missed (18)

Target:

- Items 6.1–6.8 (speculative computing) go in `experimental.nn` and `experiments.exotic`, next to `DiffractiveShapeClassifier`, `AnalogNoiseAwareTraining`, `ProbabilisticBits`, `SpikingClassifier` and `HyperdimensionalDemo`. Their baselines are a same-size MLP and the nearest existing exotic block.
- Items 6.9–6.18 (frontier) go in `experimental.llm`, `experimental.nn`, `experiments.reasoning`, `experiments.context` and `experiments.safety`. Their baselines are `GrpoTrainer`, `TestTimeCompute`, `KvCache`/`SnapKv` and `SafetyToyLm`, as wave 5 used them.

| # | Item | One line | Lives in | Experiment vs baseline | Source |
|---|---|---|---|---|---|
| 6.1 | **AKOrN (artificial Kuramoto oscillatory neurons)** | Units are unit vectors that synchronise through generalised Kuramoto dynamics; binding emerges as phase clusters. | `experimental.nn.KuramotoLayer` | Multi-shape ShapeImages object discovery: ARI, compared with slot attention (2.13). Sudoku-lite / `LatinSquarePuzzles`: compared with TRM. | [arXiv:2410.13821](https://arxiv.org/abs/2410.13821) |
| 6.2 | **Tensor networks: MPS classifier + Born machine** | Matrix-product-state weights contracted with a local feature map (DMRG-style sweeps or autograd); \|ψ\|² as a generative model with exact sampling. | `experimental.nn.MatrixProductState` | ToyTasks digits: accuracy against bond dimension, compared with a same-parameter MLP. Binarised ToyImages: NLL and exact samples, compared with `PBitIsingMachine` and PixelCNN. | [arXiv:1605.05775](https://arxiv.org/abs/1605.05775), [arXiv:1709.01662](https://arxiv.org/abs/1709.01662) |
| 6.3 | **Simulated variational quantum circuit classifier** | A ≤10-qubit state-vector simulator: angle/amplitude encoding, hardware-efficient ansatz, and parameter-shift gradients with shot noise. | `experimental.nn.QuantumCircuitLayer` | Two moons / circles: accuracy against qubits, layers and shots, compared with RFF (1.1) at equal parameters. Includes a barren-plateau gradient-variance plot against depth. | [arXiv:1804.00633](https://arxiv.org/abs/1804.00633), [arXiv:1803.00745](https://arxiv.org/abs/1803.00745) |
| 6.4 | **Stochastic-computing neural network** | Values as random bitstreams: AND multiplies, MUX scales-and-adds, and stream length trades accuracy for "energy". | `experimental.nn.StochasticBitstreamDense` | ToyTasks digits inference with a trained MLP converted to bitstreams: accuracy against stream length, compared with `BitLinear` / `TernaryMatrix` and 8-bit quantisation. | [Alaghi & Hayes 2013](https://dl.acm.org/doi/10.1145/2465787.2465794) |
| 6.5 | **Active dendrites** | Each neuron has dendritic segments matched against a context vector; the best segment gates the unit (k-WTA), giving sparse task-specific subnetworks. | `experimental.nn.ActiveDendriteLayer` | ContinualLearningExperiment permuted digits with a task context: forgetting, compared with EWC, SI and `GROW_FROZEN`. | [arXiv:2201.00042](https://arxiv.org/abs/2201.00042) |
| 6.6 | **STDP unsupervised spiking network (Diehl & Cook)** | LIF excitatory neurons with lateral inhibition and adaptive thresholds learn digit prototypes by spike-timing-dependent plasticity, with no gradients. | `experiments.exotic.StdpDigits` (reuses `LifLayer`, `SpikeEncoding`) | ToyTasks digits: label-assignment accuracy against neurons, compared with `SoftHebbClassifier` and the surrogate-gradient `SpikingClassifier`. | [Diehl & Cook 2015](https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full) |
| 6.7 | **Physics-aware training (PAT)** | Forward pass through a non-differentiable "physical" system (a simulated noisy mechanical or optical medium); backward pass through a learned digital twin. | `experimental.train.PhysicsAwareTraining` | `DiffractiveShapeClassifier` with fabrication noise and an unknown nonlinearity: accuracy, compared with in-silico training then transfer, and with `AnalogNoiseAwareTraining`. | [arXiv:2104.13386](https://arxiv.org/abs/2104.13386) |
| 6.8 | **GFlowNets (trajectory balance)** | Learn a sampler whose terminal probability is proportional to a reward, by matching flows along constructive trajectories. | `experimental.train.GFlowNet` | Hypergrid and small-molecule-like string generation: mode discovery and distribution L1 to the true reward distribution, compared with PPO (reward-maximising), MCMC and `MapElites`. | [arXiv:2201.13259](https://arxiv.org/abs/2201.13259) |
| 6.9 | **MesaNet + large-chunk test-time training (LaCT)** | MesaNet solves the in-context regression exactly at every step (conjugate gradient on a chunk-wise state); LaCT updates fast weights once per large chunk for hardware efficiency. | `experimental.nn.MesaLayer`, `experimental.nn.LargeChunkTttLayer` | DeltaFamilyShootout / MixerShootout recall: accuracy and loss, compared with `GatedDeltaNet`, `KimiDeltaAttention`, `TestTimeTrainingLayer` and `TitansLayer`. | [arXiv:2506.05233](https://arxiv.org/abs/2506.05233), [arXiv:2505.23884](https://arxiv.org/abs/2505.23884) |
| 6.10 | **Power-distribution sampling (reasoning with sampling)** | Training-free MCMC over the base model's sharpened distribution p^α, by resampling suffixes with Metropolis–Hastings acceptance. | `experimental.llm.PowerSampler` | ReasoningCoverage / TestTimeScaling on ArithmeticTask with the base model: pass@1 and pass@k diversity at equal tokens, compared with low-temperature sampling and with the GRPO-trained model. | [arXiv:2510.14901](https://arxiv.org/abs/2510.14901) |
| 6.11 | **Soft thinking + native parallel thinking (ParaThinker)** | Feed probability-weighted mixtures of token embeddings as "concept tokens" instead of sampled tokens; train to generate several reasoning paths in parallel, then merge. | `experimental.llm.SoftThinkingDecoder`, `experimental.llm.ParallelThinker` | ChainTask / ArithmeticTask: accuracy and tokens, compared with CoT, `CoconutCurriculum` latent reasoning, self-consistency and `DeepConf`. | [arXiv:2505.15778](https://arxiv.org/abs/2505.15778), [arXiv:2509.04475](https://arxiv.org/abs/2509.04475) |
| 6.12 | **ProRL (prolonged RL with reference reset)** | Long GRPO runs kept stable by a KL penalty and by periodically resetting the reference model and optimizer, over a diverse task mix. | `experimental.llm.GrpoTrainer` (reference-reset schedule) | ReasoningCoverage: pass@k at large k after 5× the usual steps, compared with plain GRPO (to test the "RL only sharpens" result) and with the RL's Razor setup. | [arXiv:2505.24864](https://arxiv.org/abs/2505.24864) |
| 6.13 | **SPIRAL (self-play on zero-sum games)** | One LM plays both sides of multi-turn zero-sum text games, trained by role-conditioned advantage estimation; the reasoning transfers to math. | `experiments.reasoning.SelfPlayGames` (reuses `TicTacToe`, `EloLeague`) | Text TicTacToe / Kuhn poker self-play: Elo, then transfer to ArithmeticTask, compared with `AbsoluteZero` and a fixed-opponent RL baseline. | [arXiv:2506.24119](https://arxiv.org/abs/2506.24119) |
| 6.14 | **Rubrics as rewards** | A checklist of weighted, instance-specific rubric items (judged by a verifier LM) is the reward for non-verifiable tasks. | `experimental.llm.RubricReward` | ClarifyingQuestions / Constitution tasks: rubric-satisfaction and held-out preference, compared with a scalar reward model (`RewardModel`) and with `GenerativeVerifier` Likert scores. | [arXiv:2507.17746](https://arxiv.org/abs/2507.17746) |
| 6.15 | **MemAgent + recursive language models (RLM)** | An RL-trained agent reads a long input chunk by chunk and overwrites a fixed-size memory; an RLM recursively calls itself on slices of the context held in an environment. | `experimental.llm.MemoryOverwriteAgent`, `experimental.llm.RecursiveLm` | A needle / multi-hop task at 4–32× the context length: accuracy and calls, compared with `SelfSummarizingContext`, `VirtualContextPager` and `RecurrentMemoryTransformer`. | [arXiv:2507.02259](https://arxiv.org/abs/2507.02259), [arXiv:2512.24601](https://arxiv.org/abs/2512.24601) |
| 6.16 | **Cartridges (self-study KV caches)** | Train a small KV cache offline for one corpus by distilling the in-context model's behaviour on synthetic self-quiz conversations; load it in place of the long prompt. | `experimental.llm.CartridgeTrainer` (on `KvCache` / `PrefixCache`) | HeadBudgetComparison corpus QA: accuracy against KV bytes, compared with the full context, `SnapKv`, `KVzip`-style eviction and prefix tuning (`SoftPrompts`). | [arXiv:2506.06266](https://arxiv.org/abs/2506.06266) |
| 6.17 | **Subliminal learning** | A teacher with a trait, fine-tuned from the same init, emits unrelated data (number sequences); a student trained on it acquires the trait. Transfer fails across different inits. | `experiments.safety.SubliminalTransfer` | SafetyToyLm teacher with a planted preference: student trait rate when sharing the init vs not, and after filtering the data, compared with a neutral teacher. | [arXiv:2507.14805](https://arxiv.org/abs/2507.14805) |
| 6.18 | **Emergent misalignment + inoculation prompting** | Narrow fine-tuning on a bad behaviour generalises to broad misbehaviour; adding an instruction that elicits the trait during training suppresses it at test time. | `experiments.safety.EmergentMisalignment` (reuses `PersonaVector`) | SafetyToyLm with a narrow "insecure" task: off-task harmful rate and persona-vector projection, compared with a neutral fine-tune and with the inoculated fine-tune. | [arXiv:2502.17424](https://arxiv.org/abs/2502.17424), [arXiv:2510.04340](https://arxiv.org/abs/2510.04340) |

---

## Notes on dependencies between items

**Classic ML (WP1)**
- First create the `experimental.ml` package and the `experiments.classic.ClassicMlShootout` harness, with shared datasets for spirals, circles, moons and blobs (from `ToyTasks` / `ToyPoints`) and the metrics ARI, trustworthiness and NLL. All WP1 items plug into it.
- 1.2 → 1.3 → 1.4: build the GP kernels and marginal likelihood once. SVGP and GP-BO reuse them, and 5.5 and 5.12 use the GP as their gold standard.
- 1.1 RFF is reused by 5.13 (the SNGP head) and by 6.3 as the classical baseline.
- 1.6 GMM is a baseline for 4.4. 1.6 k-means is the baseline for 1.13 and 1.14.
- 1.7 must reuse core `ThinSvd`, and 1.8 uses 1.7's PCA for initialisation.
- 1.9 HMM and 1.10 CRF share forward/Viterbi code; one agent should do both. 1.10's tagging task is reused by 2.14 and 2.15.
- 1.11 should share Gibbs/p-bit sampling with `PBitIsingMachine`.

**Classic DL (WP2)**
- 2.6 U-Net is needed by 4.1 pix2pix/CycleGAN, 4.10 ControlNet and optionally 4.7 EDM; schedule it early.
- 2.1–2.3 extend `TrainingTricksExperiment` / `NormalizationShootout`. 2.3 needs a CNN arm added to `NormalizationShootout`, which is currently transformer-only.
- 2.2, 2.4 and 2.5 are VisionShootout rows; one agent should take them.
- 2.7 → 2.8: the pointer network reuses the seq2seq encoder and attention.
- 2.9 shares the bAbI-like toy with the `experiments.memory` NTM/DNC tests.
- 2.11 Set Transformer's PMA and 2.12 Perceiver share a latent cross-attention block, and 2.13 slot attention reuses it too.
- 2.14 → 2.15 → (2.16, 2.17) form a language chain on one synthetic-grammar corpus. 2.16 and 2.17 extend `ElectraVsMaskedLm`, next to `FimTransform`.
- 2.18 reuses `ParamGroups` for per-layer learning rates.

**RL (WP3)**
- 3.1 TRPO needs Fisher-vector products. Write a conjugate-gradient helper in `experimental.rl` and reuse it in 3.14 CPO.
- 3.2 A3C and 3.3 V-trace share an actor-thread harness with configurable policy lag.
- 3.4 R2D2 and 3.16 RL² both need recurrent policies with stored state; build the sequence-replay buffer once.
- 3.5 IQN extends `RainbowDqn`'s `Distribution` enum.
- 3.6 NGU reuses `RandomNetworkDistillation`, `KnnIndex` and `Bandit`.
- 3.7 PlaNet's RSSM is the base for 3.8 TD-MPC2 (latent consistency) and is compared with 3.9 Dreamer 4. 3.9 also reuses `MeanFlow` shortcut targets and the DIAMOND harness.
- 3.10 extends core `Sac`, and 3.11 MPO shares its Q-ensemble critic.
- 3.17 reuses `DecisionTransformer`. 3.18 changes `Mcts` root selection and must not break `AlphaZero` / `MuZeroLite`.

**Generative (WP4)**
- 4.1 needs 2.6.
- 4.2 and 4.3 reuse core `Adversarial` (R1, hinge). 4.3 also needs a perceptual-feature network: use a frozen `SelfSupervisedBenchmark` encoder.
- 4.4 → 4.5: both use the same density-evaluation harness. 4.5 builds on core `NeuralOde` / `OdeSolver`.
- 4.6 and 4.7 extend the `TinyDdpm` / `DiffusionTargets` harness. Build 4.7's EDM preconditioning first, because 4.11 DMD and 4.16 sCM distil from an EDM-style teacher.
- 4.8 JEM is evaluated with `Calibration` and shares its OOD protocol with 5.12 and 5.13.
- 4.9 needs a VAE from `ImageAutoencoders`, plus a frozen SSL encoder for RAE.
- 4.13 and 4.14 sit beside `MaskedDiffusion`; coordinate with whoever owns `TextDiffusionVsAutoregressive`.
- 4.11, 4.16, 4.17 and 4.18 all report in `OneStepGenerators`; give them to one agent.

**Meta, continual, AutoML, uncertainty, explainability, causal (WP5)**
- 5.1–5.5 extend `MetaLearningExperiment`. 5.4 changes `Maml`, so do it before 5.2 LEO, which compares against ANIL.
- 5.5 needs 1.2's GP as a reference.
- 5.6–5.8 need a class-incremental split added to `ContinualLearningExperiment` (today it is domain-incremental permuted digits). Add the split first.
- 5.9 SDFT reuses `OnPolicyDistiller` and the `SparseMemoryFacts` stream.
- 5.10 PBT and 1.4 both extend core `HyperparameterSearch`; one agent should do both to avoid conflicts.
- 5.11 needs the `TransformerBlockSearch` space and `DartsSearch` for comparison.
- 5.12 and 5.13 share one uncertainty benchmark (regression gap plus image OOD) with `PopulationEnsemble`, `LastLayerLaplace` and 1.2.
- 5.14–5.16 share the new `ExplainabilityBenchmark` with planted ground truth; build it first. 5.15 TCAV reuses `LinearProbe`. 5.16 reuses the flipped-label setup of `DataPruningExperiment`.
- 5.17 and 5.18 share one synthetic SCM / protected-attribute generator in `experiments.safety.CausalAndFairness`.

**Unconventional and frontier (WP6)**
- 6.1 is compared with 2.13, so do it after slot attention.
- 6.3 uses 1.1 RFF as its classical baseline.
- 6.5 needs 5.6's continual harness only if the class-incremental split is used; otherwise it is independent.
- 6.6 reuses `LifLayer` and `SpikeEncoding`.
- 6.7 extends the `DiffractiveLayer` toy.
- 6.8 is independent.
- 6.9 extends the `GatedDeltaNet` chunked state code (the WP1 owner of wave 5).
- 6.10–6.14 all sit on `GrpoTrainer` / `TestTimeCompute` / `ArithmeticTask`. 6.12 changes `GrpoTrainer`, so land it before 6.13 and 6.14.
- 6.15 and 6.16 share the long-context QA corpus from `HeadBudgetComparison`.
- 6.17 and 6.18 share `SafetyToyLm` and `PersonaVector`; one agent should do both.

**Cross-package**
- The new `experimental.ml` package (WP1, plus 5.17's estimators) has no dependencies on the other packages and can start immediately.
- WP2 2.6 (U-Net) unblocks WP4 4.1 and 4.10. WP1 1.2 (GP) unblocks WP5 5.5 and 5.12.
- WP3 touches only `experimental.rl`, `Mcts` and core `Sac`, so it is independent of the rest.
- WP6 6.12 and any remaining WP3-of-wave-5 work both touch `GrpoTrainer`; schedule 6.12 after them.
- Core-worthy items (1.4, 5.10, and possibly 2.6 and 4.7) change core modules (`brainiac-train`, `brainiac-nn`). Land them in a separate PR from the experiments.
