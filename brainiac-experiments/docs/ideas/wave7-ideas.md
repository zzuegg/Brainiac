# Brainiac: wave 7 ideas

*Compiled 2026-11-18 against `claude/project-thread-8zvcl0` (main tree), plus every worktree under `.claude/worktrees/*` (30 worktrees). Waves 1–6 implemented about 700 techniques; see `experiment-results.md`, `unimplemented-ideas.md`, `wave5-ideas.md` and `wave6-ideas.md`. Wave 7 targets the areas that are still thin after six waves:*

- *(a) graph learning beyond the five GNN layers and the graph autoencoder;*
- *(b) recommender systems, information retrieval and tabular deep learning beyond `RecsysTabularBenchmarks`;*
- *(c) numerical methods for training (second-order and natural-gradient variants) and compression of training itself (gradients, activations, optimizer states);*
- *(d) probabilistic machine learning: probabilistic programming, MCMC and variational inference, Gaussian processes, kernels, optimal transport and topological data analysis;*
- *(e) multi-agent RL, game theory, robotics-style control and evolutionary methods;*
- *(f) speech and audio toys, program synthesis, neuro-symbolic learning, plus deeper follow-ups that could turn eight earlier negative results (waves 1–6) into fair tests. Follow-ups are marked "— follow-up" in their titles (7.2.18, 7.3.18, 7.5.5 and 7.6.14–7.6.18).*

*It also adds the late-2025/2026 items found by a web sweep (Nash policy gradient, evolution strategies at scale, TabPFN v2/2.5, semantic-ID scaling, large Sinkhorn couplings, Samba-ASR, AdaFisher, IVON, COAT).*

*Every item was grepped over all `*.java` sources under `src/` in all modules, in the main tree and in all 30 worktrees: 35,278 files, 2,618 unique by content after de-duplication (2,030 distinct file names). The search was case-insensitive and used class names, paper names, author-coined terms and javadoc keywords. An item is listed only when nothing matched, or when the only matches were unrelated. Examples of unrelated matches:*

- *"K-FAC" matching `RlNetworkFactory` ("networ**kFac**tory").*
- *"GraphCL" matching `graphClassification`, and "DIN" matching the local `dIn` in `MlMetrics`.*
- *"RotatE" and "ComplEx" matching `rotate` and the autograd `Complex` class.*
- *"EASE" matching "ease training" in `DenseBlock2d`, "LinUCB/Thompson" matching Thompson's construction in `RegexAutomaton`, and "SEAL" (link prediction) matching the SEAL self-editing LM.*
- *"Sinkhorn" matching only the doubly-stochastic projection inside `ManifoldHyperConnection` (mHC), not entropic optimal transport.*
- *"MaxSim" matching a local variable in `ContrastiveSearch`, "PSRO/double oracle" matching `double oracle` fields, "MAPPO" matching only a javadoc mention in `MultiAgentEnvironment`, and "NODE" matching every graph/brain node.*
- *"Signum" matching `Math.signum`, "Transducer" matching an interface name in `Seq2SeqShootout`, "HuBERT" matching Hubert (of the adjusted Rand index) in `MlMetrics`, "MMD" matching the MMD loss of `MomentMatchingGenerator` and the DTM metric (not a two-sample test), "Nash" matching `personaShift`, and "pitch/YIN" matching hundreds of unrelated identifiers.*

## How this was checked

- **Already implemented, so left out.** These turned up in the systematic check but already have code:
  - Graphs: GCN, GAT, GraphSAGE (whose javadoc mentions heterophily), GIN (whose javadoc mentions the Weisfeiler–Lehman test), `GraphReadout`, the graph autoencoder and `EquivariantGraphLayer` (EGNN).
  - Recsys and retrieval: factorization machines, DCN-V2 (`CrossNetwork`), TabNet-lite, two-tower retrieval, BPR matrix factorization, `VectorIndex.hnsw` (HNSW), `KnnIndex`, and a cross-encoder reranker in core `brainiac-llm` `Encoder` (inference only).
  - Optimisers: Shampoo, SOAP, PSGD Kron, Sophia (Gauss–Newton–Bartlett), TRPO (natural gradient for RL), `Adam8bit`, GaLore, APOLLO, LARS, LAMB, `Recompute` (gradient checkpointing), `NeuralOde` with the adjoint method, DeMo (`DecoupledMomentum`: sign-compressed momentum with a residual, so a sign/EF baseline exists), DiLoCo and Streaming DiLoCo.
  - Probabilistic: GP regression/classification, SVGP, deep kernel learning, GP-BO and BOHB, SWAG, Bayes by backprop, SGLD (`WeightPosterior`), last-layer Laplace, evidential, SNGP and DUQ (`DistanceAwareHead`), IWAE, minibatch exact OT by the Hungarian algorithm (`MinibatchOptimalTransport`), bridge matching (entropic OT limit), Metropolis–Hastings walks (`GFlowNetSampling`, `PowerSampler`).
  - Control and games: CEM planning, MPPI inside `TdMpc`, differentiable physics, QMIX/VDN, Elo leagues, AlphaZero, Gumbel MCTS, emergent communication (`ReferentialGame`).
  - Evolution: OpenAI-ES, CMA-ES, NEAT, HyperNEAT, MAP-Elites, novelty search, POET, AutoML-Zero, `ProgramEvolution`, merging as crossover (`ModelMerging`).
  - Audio: core `Signal` already has STFT/iSTFT, mel filterbanks, log-mel spectrograms and SpecAugment; CTC (`CtcLoss`), RVQ codec, WaveNet.
  - Program synthesis: enumerative synthesis with observational-equivalence pruning (inside `SelfImprovingCoders`), DreamCoder-style `LibraryLearner`, the AlphaGeometry-style `NeuroSymbolicProver`.
  - TTUR for GANs (already in `GanExperiment`).
- **Considered but cut** (to stay near 6 × 18, or too close to an existing block):
  - Hypergraph networks and temporal graph networks (TGN): no existing toy data or baseline to compare with fairly; revisit after 7.1.
  - GATv2: a one-line change to `GatLayer`, too small for an item.
  - Cluster-GCN / GraphSAINT sampling: their gain is memory at scale, invisible on `SyntheticGraphs`.
  - Speaker verification (x-vector, GE2E, AAM-softmax): needs a multi-speaker corpus the formant synthesiser does not model well.
  - CodeT-style test generation: needs a code LM stronger than the toy ones.
  - Mapper (TDA): a visualisation without a clear metric; persistent homology (7.4.16–17) covers TDA.
  - Hamiltonian/Lagrangian neural networks, PINNs, DeepONet, SINDy: still unimplemented, but they belong in a dedicated scientific-ML wave.
  - Expectation propagation, function-space VI, Wasserstein gradient flows (JKO): no toy where they beat the items kept.
  - Grokking and WGAN-GP negatives: these are test-budget artefacts (the tests stop early), not technique gaps. 7.5.5 gives the GAN case a proper min-max optimiser instead.
- **CPU-toy filter.** Every item is pure Java with no external libraries, runs in seconds to minutes on a CPU, and names a task that exists in `brainiac-experiments` or a tiny new generator next to it, plus a baseline already in the repo. The tasks used are:
  - `SyntheticGraphs` (stochastic block model, trees vs cycles), `SyntheticRecsys`, `ToyTasks` spirals/digits, `ToyPoints`, `ShapeImages`, `ToyImages`, `SeasonalSeries`, `ChaoticSeries`.
  - `ArithmeticTask`, `ChainTask`, `ListPrograms`, `LatinSquarePuzzles`, `CountingTasks`, `SyntheticGrammar`.
  - The `brainiac-rl` environments (CartPole, Pendulum, MountainCar, Lander, GridWorld, Bandit) and the `experimental.rl` games (`TicTacToe`, `ConnectThree`, `TwoStepGame`, `MultiAgentEnvironment`).
  - `TalkingDigits`' formant synthesiser and `ToyAudioCodec`.

**Format.** Wave 7 keeps wave 6's structure (checks, summary, one section per work package, dependency notes). Because each item now carries a one-paragraph mechanism, items are written as short blocks instead of table rows. Each block gives:

- **Block**: the reusable class, in `io.github.zzuegg.brainiac.experimental.<pkg>`.
- **Experiment**: the runnable measurement, in `io.github.zzuegg.brainiac.experiments.<theme>`.
- **Toy task**, **Baselines** (all already in the repo) and **Expected metric** (the number to report, with the hypothesis to test).

**Where an item should live:**

- `experimental.<pkg>`: a new block in `brainiac-experiments`. Two new packages are proposed: `experimental.graph` (the GNN layers have outgrown `experimental.nn`; the existing `GcnLayer`, `GatLayer`, `SageLayer`, `GinLayer` and `GraphOps` stay where they are, and new graph blocks import them) and `experimental.recsys` (recommender and retrieval blocks; `TwoTowerRetrieval`, `BprMatrixFactorization` and `FactorizationMachine` stay in `experimental.nn`). Everything else uses existing packages.
- `experiments.<theme>`: a new or extended runnable experiment. New themes: `experiments.retrieval` (WP2), `experiments.probabilistic` (WP4), `experiments.games` (WP5), `experiments.audio` (WP6) and `experiments.synthesis` (WP6).
- **core-worthy**: belongs in a core module once measured.

## Summary

| # | Work package | Items |
|---|---|---|
| WP1 | Graph learning: positional encodings, deep and heterophilous GNNs, expressivity, pooling, graph SSL, embeddings and algorithmic reasoning | 18 |
| WP2 | Recommender systems, information retrieval and tabular deep learning | 18 |
| WP3 | Numerical methods for training (second-order, natural gradient, implicit layers) and compression of training | 18 |
| WP4 | Probabilistic ML: probabilistic programming, MCMC and VI, Gaussian processes, kernels, optimal transport and topology | 18 |
| WP5 | Multi-agent RL, game theory, robotics-style control and evolutionary methods | 18 |
| WP6 | Speech and audio toys, program synthesis, neuro-symbolic learning and fair-test follow-ups of earlier negatives | 18 |
| | **Total** | **108** |

---

## WP1. Graph learning: positional encodings, deep and heterophilous GNNs, expressivity, pooling, graph SSL, embeddings and algorithmic reasoning (18)

Target: a new `experimental.graph` package, measured in a new `experiments.graphs.GraphLearningShootout` (node and graph tasks) and `experiments.graphs.GraphReasoning` (algorithmic and SAT tasks), next to the existing `GnnBenchmarks`. Baselines: `GcnLayer`, `GatLayer`, `SageLayer`, `GinLayer` and the feature-only MLP from `GnnBenchmarks`, at the same parameter count and step budget, over at least 3 seeds. `SyntheticGraphs` needs three new generators: a heterophilous SBM (pOut > pIn), regular/CSL graphs that 1-WL cannot tell apart, and a "tree neighbours-match" task for over-squashing.

### 7.1.1 GraphGPS with Laplacian and random-walk positional encodings (Rampášek et al., 2022; Dwivedi et al., 2020/2022)
Each GraphGPS layer adds a local message-passing update (GIN or GCN) and a global multi-head self-attention update over all nodes, then an MLP; the two paths are summed, so the model sees both the graph and all-pairs interactions. Because attention is permutation-invariant, nodes are given structural positional encodings: the k smallest non-trivial eigenvectors of the normalised Laplacian (LapPE, with random sign flips during training) or random-walk return probabilities diag(Pᵏ) for k = 1..K (RWSE).
- **Block:** `experimental.graph.GraphGpsLayer`, `experimental.graph.GraphPositionalEncodings` (reuses the Laplacian of `experimental.ml.SpectralClustering`).
- **Experiment:** `experiments.graphs.GraphLearningShootout`.
- **Toy task:** `SyntheticGraphs.communities` node classification; trees-vs-cycles; and a "count the triangles" graph regression.
- **Baselines:** GCN, GIN, GAT; GIN + RWSE without attention (ablation).
- **Expected metric:** accuracy / MAE. Hypothesis: PEs alone lift GIN on triangle counting; attention adds little on the SBM where GCN is already 0.83.
- **Source:** [arXiv:2205.12454](https://arxiv.org/abs/2205.12454), [arXiv:2003.00982](https://arxiv.org/abs/2003.00982), [arXiv:2110.07875](https://arxiv.org/abs/2110.07875)

### 7.1.2 APPNP and SGC: decoupled propagation (Klicpera et al., 2019; Wu et al., 2019)
Both separate feature transformation from propagation. SGC removes all non-linearities: it precomputes Ŝᵏ·X with the normalised adjacency Ŝ and trains logistic regression on it. APPNP predicts H = MLP(X), then runs K steps of personalised-PageRank propagation Z ← (1−α)ŜZ + αH, so it can propagate far (K = 10–20) without over-smoothing, because the teleport term keeps each node anchored to its own prediction.
- **Block:** `experimental.graph.DecoupledPropagation` (`sgc`, `appnp` factories).
- **Experiment:** `GraphLearningShootout`.
- **Toy task:** SBM node classification at feature noise 2.0 and a label rate of 5%.
- **Baselines:** GCN (2 layers), GAT, MLP.
- **Expected metric:** accuracy and training time. Hypothesis: SGC matches GCN at a fraction of the time; APPNP beats GCN at low label rates.
- **Source:** [arXiv:1810.05997](https://arxiv.org/abs/1810.05997), [arXiv:1902.07153](https://arxiv.org/abs/1902.07153)

### 7.1.3 Deep GNNs without over-smoothing: GCNII, PairNorm and DropEdge (Chen et al., 2020; Zhao & Akoglu, 2020; Rong et al., 2020)
Stacking many GCN layers drives every node embedding towards the same vector. GCNII adds an initial residual (mix in α·H⁽⁰⁾ at each layer) and identity mapping (W ← (1−β)I + βW, with β shrinking with depth). PairNorm centres embeddings and rescales them so the total pairwise distance stays constant. DropEdge randomly removes edges at every training step, which slows smoothing and acts as augmentation.
- **Block:** `experimental.graph.Gcn2Layer`, `experimental.graph.PairNorm`, `experimental.graph.DropEdge`.
- **Experiment:** `GraphLearningShootout`, depth sweep 2/8/32/64.
- **Toy task:** SBM node classification.
- **Baselines:** a plain GCN stack, GCN + residual connections.
- **Expected metric:** accuracy against depth, plus mean pairwise embedding distance (the smoothing measure). Hypothesis: plain GCN collapses towards chance by depth 32; GCNII holds within 0.02 of its depth-2 accuracy.
- **Source:** [arXiv:2007.02133](https://arxiv.org/abs/2007.02133), [arXiv:1909.12223](https://arxiv.org/abs/1909.12223), [arXiv:1907.10903](https://arxiv.org/abs/1907.10903)

### 7.1.4 Heterophily: H2GCN and GPR-GNN (Zhu et al., 2020; Chien et al., 2021)
On heterophilous graphs neighbours tend to have different labels, so averaging them hurts. H2GCN separates ego and neighbour embeddings (concatenation, not a sum), aggregates 1-hop and 2-hop neighbourhoods separately, and concatenates every layer's output. GPR-GNN propagates an MLP prediction with learned generalised-PageRank weights γ_k over hops, so it can learn high-pass (negative γ) as well as low-pass filters.
- **Block:** `experimental.graph.H2GcnLayer`, `experimental.graph.GeneralizedPageRank`.
- **Experiment:** `GraphLearningShootout`, homophily sweep h = 0.1 → 0.9.
- **Toy task:** a new heterophilous SBM in `SyntheticGraphs` (pOut > pIn).
- **Baselines:** GCN, GraphSAGE (whose javadoc claims it helps under heterophily: this item tests the claim), MLP.
- **Expected metric:** accuracy against the homophily ratio. Hypothesis: GCN falls below the MLP at h < 0.3, and H2GCN/GPR-GNN stay above it.
- **Source:** [arXiv:2006.11468](https://arxiv.org/abs/2006.11468), [arXiv:2006.07988](https://arxiv.org/abs/2006.07988)

### 7.1.5 Neural sheaf diffusion (Bodnar et al., 2022)
A cellular sheaf attaches a d-dimensional vector space to every node and edge, with learned linear restriction maps F_{v◁e} between them. Diffusion uses the sheaf Laplacian, X ← X − σ(Δ_F (I ⊗ W₁) X W₂), so signals are transported between neighbours through the restriction maps instead of being averaged. With orthogonal or general maps the diffusion can separate classes on heterophilous graphs, where plain graph diffusion merges them. The maps are predicted from the features of the two endpoints.
- **Block:** `experimental.graph.SheafDiffusionLayer`.
- **Experiment:** `GraphLearningShootout` (heterophily sweep shared with 7.1.4).
- **Toy task:** heterophilous SBM.
- **Baselines:** GCN, 7.1.4 H2GCN and GPR-GNN.
- **Expected metric:** accuracy at h = 0.1–0.3 for diagonal, orthogonal and general maps.
- **Source:** [arXiv:2202.04579](https://arxiv.org/abs/2202.04579)

### 7.1.6 GRAND: graph neural diffusion (Chamberlain et al., 2021)
Message passing as a continuous diffusion PDE on the graph, dX/dt = (A(X) − I)X, where A(X) is a learned attention matrix over edges, integrated by an ODE solver. Depth becomes integration time, so the number of "layers" is set by the solver; with adaptive or implicit steps the model can diffuse far without over-smoothing weights.
- **Block:** `experimental.graph.GraphDiffusionOde` (on core `NeuralOde` / `OdeSolver`).
- **Experiment:** `GraphLearningShootout`.
- **Toy task:** SBM node classification.
- **Baselines:** GCN and GAT at matched depth; 7.1.3 GCNII.
- **Expected metric:** accuracy against integration time T and solver steps.
- **Source:** [arXiv:2106.10934](https://arxiv.org/abs/2106.10934)

### 7.1.7 Over-squashing: a virtual node and FoSR rewiring (Alon & Yahav, 2021; Gilmer et al., 2017; Karhadkar et al., 2023)
When information from exponentially many nodes must pass through a few edges, it is squashed into fixed-size vectors. Two fixes are tested: a virtual node connected to every node (a global relay), and FoSR, which adds edges one by one to maximise the first-order increase of the spectral gap, keeping the original graph intact. The tree "neighbours-match" task needs information from every leaf at the root.
- **Block:** `experimental.graph.GraphRewiring` (`virtualNode`, `fosr`).
- **Experiment:** `GraphLearningShootout`.
- **Toy task:** a new `SyntheticGraphs.neighboursMatch(depth)` generator (Alon & Yahav).
- **Baselines:** GCN and GIN on the raw graph; 7.1.1 GraphGPS (all-pairs attention as the ceiling).
- **Expected metric:** accuracy against tree depth 2–6. Hypothesis: GIN fails from depth 4 onwards; the virtual node and FoSR delay the failure.
- **Source:** [arXiv:2006.05205](https://arxiv.org/abs/2006.05205), [arXiv:1704.01212](https://arxiv.org/abs/1704.01212), [arXiv:2210.11790](https://arxiv.org/abs/2210.11790)

### 7.1.8 Beyond 1-WL: random node features and subgraph GNNs (ESAN) (Sato et al., 2021; Bevilacqua et al., 2022)
Message passing cannot distinguish graphs that the 1-WL test cannot (for example two non-isomorphic 3-regular graphs). Random node features break the symmetry: each node gets a fresh random vector per forward pass, which makes GNNs universal in probability. ESAN represents a graph as a bag of subgraphs (node-deleted or edge-deleted), runs a shared GNN on each and pools them with a DeepSets layer, which is strictly more expressive than 1-WL.
- **Block:** `experimental.graph.RandomNodeFeatures`, `experimental.graph.SubgraphEnsembleGnn` (reuses `SetEncoder`).
- **Experiment:** `GraphLearningShootout`, expressivity split.
- **Toy task:** a new `SyntheticGraphs.circularSkipLinks` (CSL) generator: 10 isomorphism classes of 4-regular graphs; plus 3-regular pairs.
- **Baselines:** GIN (the 1-WL bound, whose javadoc cites WL), GIN + RWSE (7.1.1).
- **Expected metric:** CSL accuracy. Hypothesis: GIN stays at chance (0.10); RNF and ESAN go well above.
- **Source:** [arXiv:2002.03155](https://arxiv.org/abs/2002.03155), [arXiv:2110.02910](https://arxiv.org/abs/2110.02910), [arXiv:1903.02541](https://arxiv.org/abs/1903.02541)

### 7.1.9 PNA: principal neighbourhood aggregation (Corso et al., 2020)
A single aggregator loses information about a neighbour multiset. PNA combines several aggregators (mean, max, min, standard deviation) with degree scalers (identity, amplification log(d+1)/δ, attenuation δ/log(d+1)), and mixes the 12 resulting vectors with a linear layer. This lets it count and compare neighbourhoods of varying degree.
- **Block:** `experimental.graph.PnaLayer`.
- **Experiment:** `GraphLearningShootout`.
- **Toy task:** trees vs cycles; a degree-varied "max-neighbour-feature" regression; graph property regression (diameter, eccentricity).
- **Baselines:** GIN (sum), GCN (mean), GraphSAGE.
- **Expected metric:** accuracy / log-MSE per property.
- **Source:** [arXiv:2004.05718](https://arxiv.org/abs/2004.05718)

### 7.1.10 Hierarchical pooling: DiffPool, SAGPool and top-k pooling (Ying et al., 2018; Lee et al., 2019; Gao & Ji, 2019)
Flat readouts sum all nodes at once. DiffPool learns a soft assignment S = softmax(GNN(X, A)) of nodes to clusters and coarsens with X' = SᵀZ, A' = SᵀAS, with link-prediction and entropy auxiliary losses. Top-k pooling (gPool) keeps the k nodes with the largest learned projection score; SAGPool computes that score with a GNN, so it uses structure.
- **Block:** `experimental.graph.DiffPool`, `experimental.graph.TopKPooling` (score by projection or by GNN).
- **Experiment:** `GraphLearningShootout`, graph-classification split.
- **Toy task:** trees vs cycles; a new "two communities vs three communities" graph-classification set from the SBM generator.
- **Baselines:** GIN with sum readout (0.85 in `GnnBenchmarks`), GCN with mean readout.
- **Expected metric:** accuracy and parameters.
- **Source:** [arXiv:1806.08804](https://arxiv.org/abs/1806.08804), [arXiv:1904.08082](https://arxiv.org/abs/1904.08082), [arXiv:1905.05178](https://arxiv.org/abs/1905.05178)

### 7.1.11 Graph self-supervision: DGI, GraphCL and GraphMAE (Veličković et al., 2019; You et al., 2020; Hou et al., 2022)
Three ways to pretrain a GNN encoder without labels. Deep Graph Infomax maximises mutual information between node embeddings and a graph summary, contrasting against a feature-shuffled corrupted graph. GraphCL applies InfoNCE between two augmented views (node dropping, edge perturbation, feature masking, subgraph sampling). GraphMAE masks node features and reconstructs them through a re-masked GNN decoder with a scaled-cosine loss.
- **Block:** `experimental.graph.GraphSelfSupervision` (`dgi`, `graphCl`, `graphMae`; reuses `SelfSupervisedLosses` InfoNCE).
- **Experiment:** a new `experiments.graphs.GraphPretraining`.
- **Toy task:** SBM: pretrain on the whole graph, then a linear probe with 1% / 5% labels.
- **Baselines:** a randomly initialised GCN encoder with a linear probe; the supervised GCN; the graph autoencoder of `GnnBenchmarks`.
- **Expected metric:** linear-probe accuracy at each label rate.
- **Source:** [arXiv:1809.10341](https://arxiv.org/abs/1809.10341), [arXiv:2010.13902](https://arxiv.org/abs/2010.13902), [arXiv:2205.10803](https://arxiv.org/abs/2205.10803)

### 7.1.12 DeepWalk and node2vec (Perozzi et al., 2014; Grover & Leskovec, 2016)
Shallow node embeddings from random walks: generate walks, treat them as sentences, and train skip-gram with negative sampling. DeepWalk uses uniform walks; node2vec biases them with a return parameter p and an in-out parameter q, interpolating between BFS-like (structural roles) and DFS-like (communities) exploration.
- **Block:** `experimental.graph.RandomWalkEmbedding` (reuses the SGNS trainer of `StaticWordEmbeddings`).
- **Experiment:** `GraphLearningShootout`, embedding split.
- **Toy task:** SBM node classification by logistic regression on the embedding; link prediction on held-out edges.
- **Baselines:** the graph autoencoder (link AUC 0.75), the common-neighbours heuristic (0.67), spectral embedding from `SpectralClustering`.
- **Expected metric:** node accuracy and link AUC over a (p, q) grid.
- **Source:** [arXiv:1403.6652](https://arxiv.org/abs/1403.6652), [arXiv:1607.00653](https://arxiv.org/abs/1607.00653)

### 7.1.13 Knowledge-graph embeddings: TransE, DistMult, ComplEx and RotatE (Bordes et al., 2013; Yang et al., 2015; Trouillon et al., 2016; Sun et al., 2019)
Link prediction over (head, relation, tail) triples with a scoring function per model: TransE uses −‖h + r − t‖; DistMult uses the trilinear ⟨h, r, t⟩, which is symmetric; ComplEx uses Re⟨h, r, t̄⟩ in complex space, which handles antisymmetry; RotatE rotates h by a unit-modulus complex r, which models symmetry, inversion and composition. All are trained with self-adversarial negative sampling.
- **Block:** `experimental.graph.KnowledgeGraphEmbedding` (one class, four scorers; reuses autograd `Complex`).
- **Experiment:** a new `experiments.graphs.KnowledgeGraphCompletion`.
- **Toy task:** a synthetic family-tree KG with planted symmetric (sibling), antisymmetric (parent), inverse (child) and composed (grandparent) relations.
- **Baselines:** a rule-free frequency baseline; the two-tower dot-product model (`TwoTowerRetrieval`) over entity pairs.
- **Expected metric:** filtered MRR and Hits@10, per relation pattern. Hypothesis: DistMult fails on antisymmetric relations, TransE on symmetric ones, and RotatE handles all four.
- **Source:** [TransE, NeurIPS 2013](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data), [arXiv:1412.6575](https://arxiv.org/abs/1412.6575), [arXiv:1606.06357](https://arxiv.org/abs/1606.06357), [arXiv:1902.10197](https://arxiv.org/abs/1902.10197)

### 7.1.14 Neural algorithmic reasoning, CLRS-lite (Veličković et al., 2020, 2022)
A GNN "processor" is trained to imitate each step of a classical algorithm (BFS, Bellman–Ford shortest paths, Prim's MST) with hint supervision on intermediate states, using max aggregation to align with the algorithm's structure. The test is generalisation to graphs 4× larger than in training, where pattern-matching models fail.
- **Block:** `experimental.graph.AlgorithmicProcessor` (an MPNN with max aggregation and a gated update), plus `experimental.data.AlgorithmTraces` for the step-wise hints.
- **Experiment:** `experiments.graphs.GraphReasoning`.
- **Toy task:** random graphs with 8–16 nodes for training and 64 for testing; BFS reachability, Bellman–Ford distances, MST edges.
- **Baselines:** GIN and GAT without hints; the processor with sum aggregation; 7.1.1 GraphGPS.
- **Expected metric:** per-node accuracy at 4× size. Hypothesis: max aggregation plus hints extrapolates, and sum aggregation does not.
- **Source:** [arXiv:1910.10593](https://arxiv.org/abs/1910.10593), [arXiv:2205.15659](https://arxiv.org/abs/2205.15659)

### 7.1.15 NeuroSAT (Selsam et al., 2019)
A message-passing network on the literal–clause bipartite graph of a CNF formula, trained only on the single-bit label "satisfiable". Literal and clause states are updated by LSTMs (or GRUs) for T rounds, with each literal also messaging its negation. A satisfying assignment can often be decoded by clustering the literal embeddings, and running more rounds at test time solves harder formulas.
- **Block:** `experimental.graph.NeuroSat` (reuses `GruCell`).
- **Experiment:** `GraphReasoning`.
- **Toy task:** random SR(n) pairs (one SAT, one UNSAT, differing in one literal) with n = 10 for training and n = 20–40 for testing.
- **Baselines:** a GIN graph classifier on the same bipartite graph; a WalkSAT solver with a flip budget (written in the experiment) as the non-learned reference.
- **Expected metric:** classification accuracy and decoded-assignment success rate against message rounds.
- **Source:** [arXiv:1802.03685](https://arxiv.org/abs/1802.03685)

### 7.1.16 DiGress: discrete denoising diffusion for graphs (Vignac et al., 2023)
Graph generation by discrete diffusion over node and edge categories. The noise process uses marginal-preserving transition matrices, so noisy graphs keep realistic edge density. A graph transformer predicts the clean graph from the noisy one, and extra structural features (cycle counts, spectral features) are added to its input.
- **Block:** `experimental.graph.GraphDiffusion` (reuses the discrete-noise maths of `MaskedDiffusion` and 7.1.1's attention).
- **Experiment:** a new `experiments.graphs.GraphGeneration`.
- **Toy task:** generate SBM graphs with 2–3 communities, and planar-ish grid graphs.
- **Baselines:** an Erdős–Rényi model with matched density; a GRU that writes the adjacency row by row (GraphRNN-lite, written in the experiment); the graph autoencoder decoder with sampled latents.
- **Expected metric:** MMD of the degree, clustering and spectral distributions to held-out graphs; validity (community structure recovered).
- **Source:** [arXiv:2209.14734](https://arxiv.org/abs/2209.14734)

### 7.1.17 NeuralWalker: random walks as sequences (Chen et al., 2024/2025)
Sample random walks over the graph, embed each walk with its node features and walk positional encodings, and process it with a sequence model (a Mamba or S4 layer, or a 1-D convolution). Walk outputs are scattered back to the nodes they visited and combined with local message passing. Long walks capture long-range dependencies that over-squash in plain MPNNs, at cost linear in walk length.
- **Block:** `experimental.graph.RandomWalkSequenceLayer` (reuses `MambaLayer` / `S4dLayer`).
- **Experiment:** `GraphLearningShootout` (neighbours-match split shared with 7.1.7).
- **Toy task:** neighbours-match at depth 4–6; triangle counting.
- **Baselines:** GIN, 7.1.1 GraphGPS, 7.1.7 virtual node.
- **Expected metric:** accuracy against walk length and the number of walks.
- **Source:** [arXiv:2406.03386](https://arxiv.org/abs/2406.03386)

### 7.1.18 Label propagation and Correct & Smooth (Zhu & Ghahramani, 2002; Huang et al., 2021)
Classic transductive label propagation spreads known labels over the normalised adjacency until convergence. Correct & Smooth starts from a cheap base predictor (an MLP on features), propagates its residual errors on the training nodes ("correct"), then propagates the corrected predictions ("smooth"). It often matches or beats deep GNNs at a fraction of the cost.
- **Block:** `experimental.graph.CorrectAndSmooth` (`labelPropagation`, `correctAndSmooth`).
- **Experiment:** `GraphLearningShootout`.
- **Toy task:** SBM at feature noise 0.5, 2.0 and 4.0.
- **Baselines:** GCN, GAT, MLP, 7.1.2 APPNP.
- **Expected metric:** accuracy and wall time. Hypothesis: C&S on an MLP reaches GCN's 0.83 at noise 2.0 with no message passing during training.
- **Source:** [Zhu & Ghahramani 2002](http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf), [arXiv:2010.13993](https://arxiv.org/abs/2010.13993)

## WP2. Recommender systems, information retrieval and tabular deep learning (18)

Target: a new `experimental.recsys` package, plus `experimental.nn` for the tabular blocks. Measurements go in `experiments.graphs.RecsysTabularBenchmarks` (extended), a new `experiments.retrieval.RetrievalShootout` and a new `experiments.graphs.TabularShootout`. Baselines: BPR matrix factorization (recall@10 0.38), two-tower retrieval (0.50), the factorization machine (0.73), DCN-V2 (0.933), TabNet-lite, the MLP, logistic regression, core `GradientBoostedTrees` and `RandomForest`, all at the same budget. `SyntheticRecsys` needs two additions: per-user interaction sequences with a planted transition structure, and logged bandit feedback with known propensities. Retrieval needs a small synthetic corpus generator (`experimental.data.SyntheticCorpus`), with planted topics, paraphrases and lexical-overlap distractors.

### 7.2.1 LightGCN (He et al., 2020)
Collaborative filtering on the user–item bipartite graph with all transforms removed: embeddings are only propagated, E⁽ᵏ⁺¹⁾ = ÂE⁽ᵏ⁾ with symmetric normalisation, and the final embedding is the mean over layers 0..K. It is trained with the BPR loss. The paper shows that feature transforms and non-linearities hurt CF.
- **Block:** `experimental.recsys.LightGcn`.
- **Experiment:** `RecsysTabularBenchmarks`, a new `graphCf` split.
- **Toy task:** `SyntheticRecsys` implicit feedback.
- **Baselines:** BPR-MF, two-tower retrieval, and a GCN with transforms (NGCF-lite, as an ablation).
- **Expected metric:** recall@10 and NDCG@10 against layers K = 0–4. Hypothesis: K = 2–3 beats BPR-MF (0.38).
- **Source:** [arXiv:2002.02126](https://arxiv.org/abs/2002.02126)

### 7.2.2 Sequential recommendation: SASRec, BERT4Rec and DIN (Kang & McAuley, 2018; Sun et al., 2019; Zhou et al., 2018)
SASRec is a causal transformer over a user's item history that predicts the next item. BERT4Rec is a bidirectional masked-item model over the same sequences. DIN (deep interest network) scores a candidate item with target attention: the history is weighted by its relevance to that candidate before pooling, instead of a fixed user vector.
- **Block:** `experimental.recsys.SequentialRecommender` (`sasRec`, `bert4Rec`; reuses core `TransformerBlock`), `experimental.recsys.TargetAttentionPooling` (DIN).
- **Experiment:** `RecsysTabularBenchmarks`, a new `sequential` split.
- **Toy task:** the new `SyntheticRecsys` sequences (users follow item-transition chains with drifting interests).
- **Baselines:** BPR-MF (order-free), a most-popular baseline, a first-order Markov chain (item-to-item counts), a GRU (GRU4Rec-lite).
- **Expected metric:** HR@10 and NDCG@10 for next-item prediction; AUC for DIN click prediction against the FM's 0.73.
- **Source:** [arXiv:1808.09781](https://arxiv.org/abs/1808.09781), [arXiv:1904.06690](https://arxiv.org/abs/1904.06690), [arXiv:1706.06978](https://arxiv.org/abs/1706.06978)

### 7.2.3 Mult-VAE and EASE (Liang et al., 2018; Steck, 2019)
Two autoencoder views of implicit feedback. Mult-VAE encodes a user's whole interaction vector, decodes a multinomial over items, and anneals β on the KL term. EASE is a shallow linear autoencoder with a closed form: B = I − P·diagMat(1/diag(P)), where P = (XᵀX + λI)⁻¹, with a zero-diagonal constraint. It has no training loop and is often state of the art.
- **Block:** `experimental.recsys.MultVae`, `experimental.recsys.EaseRecommender` (host-side, double precision, reuses `MlMatrices`).
- **Experiment:** `RecsysTabularBenchmarks`.
- **Toy task:** `SyntheticRecsys` implicit feedback with strong item–item co-occurrence.
- **Baselines:** BPR-MF, 7.2.1 LightGCN, item-kNN cosine.
- **Expected metric:** recall@10 / NDCG@10 and fit time. Hypothesis: EASE matches or beats BPR-MF with a single matrix inverse.
- **Source:** [arXiv:1802.05814](https://arxiv.org/abs/1802.05814), [arXiv:1905.03375](https://arxiv.org/abs/1905.03375)

### 7.2.4 NeuMF, Wide & Deep and DeepFM (He et al., 2017; Cheng et al., 2016; Guo et al., 2017)
Three classic deep CTR/CF architectures. NeuMF fuses a GMF branch (an elementwise product of embeddings) with an MLP over concatenated embeddings. Wide & Deep jointly trains a linear model over crossed features (memorisation) and a deep MLP over embeddings (generalisation). DeepFM replaces the wide part with an FM that shares embeddings with the deep part, so no manual crosses are needed.
- **Block:** `experimental.recsys.NeuralCollaborativeFiltering`, `experimental.recsys.WideAndDeep` (`wideDeep`, `deepFm`; reuses `FactorizationMachine`).
- **Experiment:** `RecsysTabularBenchmarks` click split.
- **Toy task:** `SyntheticRecsys` click prediction.
- **Baselines:** logistic regression (0.59), FM (0.73), DCN-V2 (`CrossNetwork`), BPR-MF for ranking.
- **Expected metric:** accuracy / AUC / log-loss; HR@10 for NeuMF.
- **Source:** [arXiv:1708.05031](https://arxiv.org/abs/1708.05031), [arXiv:1606.07792](https://arxiv.org/abs/1606.07792), [arXiv:1703.04247](https://arxiv.org/abs/1703.04247)

### 7.2.5 Generative retrieval with semantic IDs (TIGER) (Rajput et al., 2023; scaling study 2025)
Each item gets a short tuple of codewords (a semantic ID) from an RQ-VAE over its content embedding, so similar items share prefixes. A seq2seq transformer then generates the next item's semantic ID token by token from the user's history, with beam search constrained to valid IDs. Retrieval becomes generation, and new items can be served through their content codes. The 2025 scaling study finds that semantic-ID models saturate early, while ID-based LLM recommenders keep scaling.
- **Block:** `experimental.recsys.SemanticIdTokenizer` (reuses `ResidualVectorQuantizer`), `experimental.recsys.GenerativeRetriever` (reuses `TransformerSeq2Seq`, prefix-trie constrained beam).
- **Experiment:** `RecsysTabularBenchmarks` sequential split (shared with 7.2.2).
- **Toy task:** `SyntheticRecsys` sequences with item content features, plus a cold-start split of unseen items.
- **Baselines:** SASRec (7.2.2), two-tower + HNSW (`VectorIndex.hnsw`).
- **Expected metric:** recall@10 on all items and on cold-start items; ID collision rate; recall against codebook depth.
- **Source:** [arXiv:2305.05065](https://arxiv.org/abs/2305.05065), [arXiv:2509.25522](https://arxiv.org/abs/2509.25522)

### 7.2.6 HSTU: generative recommenders (Zhai et al., 2024)
Hierarchical Sequential Transduction Units replace softmax attention with pointwise SiLU-normalised attention plus an elementwise gating U ⊙ Attn(Q, K)V and relative attention biases over position and time. Recommendation is reformulated as sequential transduction over a unified stream of actions and items, which trains one model for both retrieval and ranking. The paper reports a power-law scaling of quality with compute that DLRMs lack.
- **Block:** `experimental.recsys.HstuBlock`.
- **Experiment:** `RecsysTabularBenchmarks` sequential split.
- **Toy task:** `SyntheticRecsys` sequences with timestamps and action types (click/skip).
- **Baselines:** SASRec (7.2.2), DIN, DCN-V2 for ranking.
- **Expected metric:** HR@10 / NDCG@10 and click AUC at three model sizes (a small scaling plot).
- **Source:** [arXiv:2402.17152](https://arxiv.org/abs/2402.17152)

### 7.2.7 Off-policy evaluation and learning from logged feedback: IPS, SNIPS, DR and Switch (Dudík et al., 2011; Swaminathan & Joachims, 2015; Wang et al., 2017)
Estimate the reward of a new recommendation policy from logs collected by an old one. IPS reweights each logged reward by π_new(a|x)/π_log(a|x); SNIPS self-normalises the weights to reduce variance; doubly robust (DR) adds a reward model as a control variate; Switch uses the model where weights are large. Counterfactual risk minimisation (POEM-style) then learns a policy by maximising the clipped estimate.
- **Block:** `experimental.recsys.OffPolicyEstimators` (the existing `CausalEstimators.ipw/aipw` compute ATEs for binary treatment, not policy values over many actions, so this is a separate block).
- **Experiment:** a new `experiments.graphs.LoggedBanditFeedback`.
- **Toy task:** the new `SyntheticRecsys` logged-bandit generator with a known logging policy and ground-truth rewards.
- **Baselines:** the direct method (a reward-model regression); the naive average of logged rewards.
- **Expected metric:** RMSE of the estimated policy value against the true value, against log size and policy divergence; the true value of the learned policy.
- **Source:** [arXiv:1103.4601](https://arxiv.org/abs/1103.4601), [SNIPS, NeurIPS 2015](https://papers.nips.cc/paper/5748-the-self-normalized-estimator-for-counterfactual-learning), [arXiv:1612.01205](https://arxiv.org/abs/1612.01205)

### 7.2.8 Contextual bandits: LinUCB, linear Thompson sampling and neural-linear (Li et al., 2010; Agrawal & Goyal, 2013; Riquelme et al., 2018)
Online recommendation with exploration. LinUCB keeps a ridge-regression posterior per arm and picks argmax θ̂ᵀx + α√(xᵀA⁻¹x). Linear TS samples θ from the Gaussian posterior. Neural-linear puts Bayesian linear regression on top of a neural network's last layer, which is retrained periodically. Core `Bandit` is context-free, and the `Agent57Learner` UCB is a sliding-window UCB over a few arms.
- **Block:** `experimental.recsys.ContextualBandit` (`linUcb`, `linearThompson`, `neuralLinear`).
- **Experiment:** `LoggedBanditFeedback` (online split).
- **Toy task:** a contextual simulator over `SyntheticRecsys` users and items, with linear and non-linear reward variants.
- **Baselines:** ε-greedy on the same reward model; core `Bandit`-style UCB1 ignoring context; a two-tower model retrained on greedy logs.
- **Expected metric:** cumulative regret over 10k rounds.
- **Source:** [arXiv:1003.0146](https://arxiv.org/abs/1003.0146), [arXiv:1209.3352](https://arxiv.org/abs/1209.3352), [arXiv:1802.09127](https://arxiv.org/abs/1802.09127)

### 7.2.9 BM25 and dense passage retrieval (Robertson & Zaragoza, 2009; Karpukhin et al., 2020)
BM25 scores documents by IDF-weighted term frequency, with saturation k₁ and length normalisation b, over an inverted index. DPR trains a dual encoder with InfoNCE over in-batch negatives plus BM25-mined hard negatives. The comparison shows where lexical matching wins (rare exact terms) and where dense matching wins (paraphrases); a hybrid uses reciprocal rank fusion. The repo has TF-IDF only as a slot-ranking heuristic inside `SparseMemoryFinetuning`.
- **Block:** `experimental.recsys.Bm25Index`, `experimental.recsys.DenseRetriever` (dual encoder; reuses core `TransformerBlock` and `VectorIndex`).
- **Experiment:** `experiments.retrieval.RetrievalShootout`.
- **Toy task:** `SyntheticCorpus` queries, split into a lexical-overlap set and a paraphrase set.
- **Baselines:** TF-IDF cosine; the untrained encoder; `TwoTowerRetrieval` as a feature-tower reference.
- **Expected metric:** recall@1/10 and MRR@10 per split, and for the hybrid.
- **Source:** [Robertson & Zaragoza 2009](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf), [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)

### 7.2.10 ColBERT late interaction (Khattab & Zaharia, 2020; Santhanam et al., 2022)
Keep one embedding per token for both query and document, and score with MaxSim: Σ_i max_j q_i·d_j. This is more expressive than one vector per document, but documents can still be encoded offline. ColBERTv2 compresses the token vectors with centroids plus residual quantisation, and adds denoised supervision.
- **Block:** `experimental.recsys.LateInteractionRetriever` (reuses `KnnIndex` for candidate generation and `ResidualVectorQuantizer` for compression).
- **Experiment:** `RetrievalShootout`.
- **Toy task:** `SyntheticCorpus`, including multi-aspect queries that need two separate terms matched.
- **Baselines:** 7.2.9 DPR and BM25; the core `Encoder` cross-encoder reranker as the quality ceiling.
- **Expected metric:** MRR@10 against bytes per document (fp32 vs 2-bit residual).
- **Source:** [arXiv:2004.12832](https://arxiv.org/abs/2004.12832), [arXiv:2112.01488](https://arxiv.org/abs/2112.01488)

### 7.2.11 SPLADE: learned sparse retrieval (Formal et al., 2021)
An MLM head maps each token to vocabulary logits; the document representation is w_v = max_t log(1 + ReLU(logit_{t,v})), a sparse bag of expansion terms. It is trained contrastively with a FLOPS regulariser that pushes towards sparsity. The result can be served from an inverted index (7.2.9) while capturing synonyms.
- **Block:** `experimental.recsys.SparseLexicalEncoder` (reuses 7.2.9's inverted index).
- **Experiment:** `RetrievalShootout`.
- **Toy task:** `SyntheticCorpus` paraphrase split.
- **Baselines:** BM25, DPR.
- **Expected metric:** MRR@10 against average non-zeros per document (the sparsity–quality curve).
- **Source:** [arXiv:2107.05720](https://arxiv.org/abs/2107.05720)

### 7.2.12 IVF-PQ and anisotropic quantisation (Jégou et al., 2011; Guo et al., 2020)
An approximate nearest-neighbour index built from a coarse k-means inverted file (probe n_probe lists) and product quantisation: split each vector into m sub-vectors, quantise each with its own 256-word codebook, and compute distances by table lookup. ScaNN's anisotropic loss weights the quantisation error parallel to the vector more heavily, which preserves inner products.
- **Block:** `experimental.recsys.IvfPqIndex` (reuses `experimental.ml.KMeans`).
- **Experiment:** `RetrievalShootout` ANN split.
- **Toy task:** 50k embeddings from the two-tower model, plus random Gaussian clusters.
- **Baselines:** `VectorIndex.flat` (exact) and `VectorIndex.hnsw`.
- **Expected metric:** recall@10 against queries per second and bytes per vector.
- **Source:** [Jégou et al. 2011](https://ieeexplore.ieee.org/document/5432202), [arXiv:1908.10396](https://arxiv.org/abs/1908.10396)

### 7.2.13 Learning to rank: RankNet, LambdaRank/LambdaMART and ListNet (Burges et al., 2005, 2010; Cao et al., 2007)
Train a scorer from graded relevance lists. RankNet uses a pairwise logistic loss on score differences. LambdaRank scales each pair's gradient by the |ΔNDCG| of swapping the pair, and LambdaMART boosts trees with those lambdas. ListNet uses the cross-entropy between top-one softmax distributions of labels and scores.
- **Block:** `experimental.recsys.RankingLosses` (`rankNet`, `lambdaRank`, `listNet`); `lambdaMart` as a new loss mode of core `GradientBoostedTrees` (core-worthy).
- **Experiment:** `RetrievalShootout` reranking split.
- **Toy task:** query–document feature vectors (BM25, dense score, length, overlap) with graded labels from `SyntheticCorpus`.
- **Baselines:** pointwise regression (MLP and GBT on the labels).
- **Expected metric:** NDCG@5/10.
- **Source:** [RankNet, ICML 2005](https://icml.cc/Conferences/2005/proceedings/papers/012_LearningToRank_BurgesEtAl.pdf), [Burges 2010, MSR-TR-2010-82](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/), [ListNet, ICML 2007](https://dl.acm.org/doi/10.1145/1273496.1273513)

### 7.2.14 Matryoshka representation learning (Kusupati et al., 2022)
Train one embedding so that every prefix of sizes d/16, d/8, …, d is itself a good embedding, by summing the loss over nested prefixes. At serving time the system picks a size per budget, or uses short prefixes for a first shortlist and full vectors for reranking (adaptive retrieval). Matryoshka exists in the repo only as a nested SAE dictionary (`SparseAutoencoder`), not for embeddings.
- **Block:** `experimental.recsys.MatryoshkaEmbedding` (a loss wrapper for any encoder).
- **Experiment:** `RetrievalShootout`.
- **Toy task:** DPR (7.2.9) and the two-tower model on `SyntheticRecsys`.
- **Baselines:** separately trained models at each size; truncating a plain embedding; PCA truncation.
- **Expected metric:** recall@10 against dimension; adaptive-retrieval recall at fixed FLOPs.
- **Source:** [arXiv:2205.13147](https://arxiv.org/abs/2205.13147)

### 7.2.15 FT-Transformer with numerical embeddings (Gorishniy et al., 2021, 2022)
Tokenise every tabular feature: each numeric column x_j becomes x_j·w_j + b_j (or a periodic PLR embedding, Linear(ReLU(sin/cos(2πcx)))), each categorical value gets an embedding, and a [CLS] token is added. Then a plain transformer runs over the feature tokens. The 2022 paper shows that the numeric embeddings alone also lift plain MLPs.
- **Block:** `experimental.nn.FeatureTokenizer` (linear and PLR embeddings), `experimental.nn.FtTransformer`.
- **Experiment:** a new `experiments.graphs.TabularShootout` (with the existing `tabular` data of `RecsysTabularBenchmarks`, plus 7.2.18's benchmark).
- **Toy task:** the existing tabular task; a high-frequency target (sin of a feature), where PLR should help.
- **Baselines:** MLP, TabNet-lite, DCN-V2, `GradientBoostedTrees`.
- **Expected metric:** accuracy / RMSE, and the MLP + PLR ablation.
- **Source:** [arXiv:2106.11959](https://arxiv.org/abs/2106.11959), [arXiv:2203.05556](https://arxiv.org/abs/2203.05556)

### 7.2.16 TabM and RealMLP (Gorishniy et al., ICLR 2025; Holzmüller et al., 2024)
TabM is an MLP that efficiently imitates an ensemble of k MLPs: BatchEnsemble-style rank-1 multiplicative adapters (r_i, s_i) give k members with shared weights, which are trained jointly and averaged. RealMLP is a bag of tuned defaults for tabular MLPs: robust scaling plus smooth clipping, PBLD numeric embeddings, a diagonal scaling layer, parametric activations, and specific learning-rate and weight-decay schedules.
- **Block:** `experimental.nn.BatchEnsembleDense`, `experimental.nn.TabM`, `experimental.nn.RealMlp`.
- **Experiment:** `TabularShootout`.
- **Toy task:** as 7.2.15.
- **Baselines:** MLP, `PopulationEnsemble` (a true deep ensemble at k× cost), GBT, 7.2.15.
- **Expected metric:** accuracy and train time. Hypothesis: TabM matches a 5-member deep ensemble at about 1.3× the cost of one MLP.
- **Source:** [arXiv:2410.24210](https://arxiv.org/abs/2410.24210), [arXiv:2407.04491](https://arxiv.org/abs/2407.04491), [arXiv:2002.06715](https://arxiv.org/abs/2002.06715)

### 7.2.17 TabPFN-style prior-fitted networks (Hollmann et al., 2023; Nature 2025; TabPFN-2.5, 2025)
Train a transformer once, on millions of small synthetic datasets drawn from a prior over structural causal models and random MLPs. Each training example is a whole dataset: the training rows (x, y) and the test rows x are tokens, and the network predicts the test labels in context, approximating Bayesian posterior prediction. At use time there is no gradient step: classification is one forward pass. v2 adds two-way attention (across rows and across features).
- **Block:** `experimental.nn.PriorFittedNetwork`, `experimental.data.TabularPrior` (an SCM/random-MLP dataset sampler).
- **Experiment:** `TabularShootout`, small-data split (n ≤ 200 rows, ≤ 10 features).
- **Toy task:** held-out tasks drawn from a different prior family, and the repo's tabular task subsampled to 50–200 rows.
- **Baselines:** logistic regression, `RandomForest`, `GradientBoostedTrees`, MLP, `GaussianProcess` classification.
- **Expected metric:** accuracy / ROC-AUC against training rows, and pretraining cost. Hypothesis: a CPU-sized PFN wins at n ≤ 100.
- **Source:** [arXiv:2207.01848](https://arxiv.org/abs/2207.01848), [Nature 2025](https://www.nature.com/articles/s41586-024-08328-6), [arXiv:2511.08667](https://arxiv.org/abs/2511.08667)

### 7.2.18 Differentiable trees and neural additive models on a trees-favouring benchmark (Popov et al., 2019; Agarwal et al., 2021; Grinsztajn et al., 2022) — follow-up
Wave 6 found that random forests lose to the MLP on the repo's tabular data (0.830 vs 0.894). Grinsztajn et al. explain when trees win: uninformative features, irregular (non-smooth) targets and non-rotation-invariant data. This item builds that benchmark, so the tree-vs-NN comparison is fair, and adds two hybrids. NODE uses differentiable oblivious decision trees with entmax feature selection. A NAM is a sum of one small MLP per feature, which keeps interpretability.
- **Block:** `experimental.nn.ObliviousTreeEnsemble` (NODE), `experimental.nn.NeuralAdditiveModel`, `experimental.data.TabularStressSuite` (adds uninformative columns, irregular step targets and random rotations).
- **Experiment:** `TabularShootout`.
- **Toy task:** the stress suite at 0, 50% and 90% uninformative features, with and without rotation.
- **Baselines:** `RandomForest`, `GradientBoostedTrees`, MLP, FT-Transformer (7.2.15).
- **Expected metric:** accuracy against the share of uninformative features. Hypothesis: trees overtake the MLP as useless features grow, and lose the advantage after rotation; NAM shape functions recover planted per-feature effects (correlation).
- **Source:** [arXiv:1909.06312](https://arxiv.org/abs/1909.06312), [arXiv:2004.13912](https://arxiv.org/abs/2004.13912), [arXiv:2207.08815](https://arxiv.org/abs/2207.08815)

## WP3. Numerical methods for training (second-order, natural gradient, implicit layers) and compression of training (18)

Target: `experimental.train` (optimisers and compressors, as `GradientTransform` or `Optimizer` implementations) and `experimental.nn` (implicit and reversible layers). Measurements go in:

- `experiments.optim.ModernOptimizerExperiment` and `TrainingDynamicsExperiment` (optimiser rows);
- `experiments.optim.LowCommunicationTraining` (gradient compression, next to DiLoCo, Streaming DiLoCo and DeMo);
- a new `experiments.optim.SecondOrderShootout`;
- a new `experiments.optim.MemoryEfficientTraining`.

Baselines: AdamW, Shampoo, SOAP, PSGD Kron, Sophia, Muon, `Adam8bit`, GaLore, core `Recompute` and DeMo, at equal steps and, for second-order methods, at equal wall-clock as well. Every row reports at least 3 seeds.

### 7.3.1 K-FAC and EKFAC (Martens & Grosse, 2015; George et al., 2018)
Approximate each dense layer's Fisher block as a Kronecker product A ⊗ G of the input-activation covariance A = E[aaᵀ] and the pre-activation gradient covariance G = E[ggᵀ], with gradients g sampled from the model's own predictive distribution. The natural-gradient step is then G⁻¹∇W A⁻¹, with damping and running averages. EKFAC refits the diagonal in the Kronecker eigenbasis, which is closer to the true Fisher (SOAP does a similar refit, but for Adam's second moment).
- **Block:** `experimental.train.Kfac` (`kfac`, `ekfac`), with `LayerHooks` to capture a and g.
- **Experiment:** `experiments.optim.SecondOrderShootout`.
- **Toy task:** a deep autoencoder on `ToyTasks` digits (the classic K-FAC benchmark); spirals with an MLP.
- **Baselines:** AdamW, Shampoo, SOAP, PSGD Kron.
- **Expected metric:** loss against steps and against wall time.
- **Source:** [arXiv:1503.05671](https://arxiv.org/abs/1503.05671), [arXiv:1806.03884](https://arxiv.org/abs/1806.03884)

### 7.3.2 AdaFisher (Gomes et al., ICLR 2025)
An Adam-style optimiser that replaces the second moment with a diagonal-block Kronecker approximation of the Fisher. It keeps only the diagonals of the K-FAC factors (A, G), which costs about as much as Adam, and uses them as the preconditioner with EMA and damping. The paper, and its 2025 Fisher-analysis follow-up, show that the Fisher's energy is concentrated on the diagonals of the Kronecker factors.
- **Block:** `experimental.train.AdaFisher` (shares factor capture with 7.3.1).
- **Experiment:** `SecondOrderShootout`; `ModernOptimizerExperiment` row.
- **Toy task:** the ModernOptimizerExperiment task; the CNN of `VisionShootout`.
- **Baselines:** AdamW, Adam-mini, Sophia, 7.3.1 K-FAC.
- **Expected metric:** steps to the target loss and time per step.
- **Source:** [arXiv:2405.16397](https://arxiv.org/abs/2405.16397), [arXiv:2504.20096](https://arxiv.org/abs/2504.20096)

### 7.3.3 L-BFGS: full-batch and stochastic (Liu & Nocedal, 1989; Schraudolph et al., 2007)
A quasi-Newton method that builds an implicit inverse-Hessian approximation from the last m pairs (s_k = Δθ, y_k = Δ∇) with the two-loop recursion, and uses a strong-Wolfe line search. It is the standard optimiser for small, smooth, full-batch problems such as PINN-like fitting, neural fields and style transfer. The online variant (oLBFGS) computes y on the same minibatch, which keeps the curvature pairs consistent.
- **Block:** `experimental.train.Lbfgs` (`fullBatch`, `online`).
- **Experiment:** `SecondOrderShootout`.
- **Toy task:** `NeuralFields` Fourier-feature image fit (full batch); 1-D GP-gap regression; minibatch spirals.
- **Baselines:** Adam, SGD with momentum, 7.3.4 Hessian-free.
- **Expected metric:** final loss and wall time to the target; divergence rate in the minibatch setting.
- **Source:** [Liu & Nocedal 1989](https://link.springer.com/article/10.1007/BF01589116), [Schraudolph et al. 2007](https://proceedings.mlr.press/v2/schraudolph07a.html)

### 7.3.4 Hessian-free optimisation (Martens, 2010)
Truncated Newton: at each step, minimise the local quadratic model with conjugate gradient, using Gauss–Newton-vector products Gv computed exactly by an R-operator (forward-mode) pass plus a backward pass, never forming G. Damping is adapted Levenberg–Marquardt style, and the CG solution is warm-started from the previous step. The repo uses Gauss–Newton only for estimates (Sophia's diagonal, the last-layer Laplace covariance), never as an optimiser.
- **Block:** `experimental.train.HessianFree` (a GN-vector product helper shared with 7.3.6 and 7.3.7).
- **Experiment:** `SecondOrderShootout`.
- **Toy task:** the deep-autoencoder pathological-curvature task; a vanilla RNN on long-lag `CountingTasks` copy (Martens & Sutskever's case).
- **Baselines:** Adam, 7.3.1 K-FAC, 7.3.3 L-BFGS.
- **Expected metric:** loss against wall time; the longest copy lag solved.
- **Source:** [Martens 2010, ICML](https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf)

### 7.3.5 AdaHessian (Yao et al., 2021)
An Adam-like optimiser whose second moment is the square of a Hutchinson estimate of the Hessian diagonal, diag(H) ≈ E[z ⊙ Hz] with Rademacher z. One Hessian-vector product per step is computed by double backprop. It adds spatial averaging over conv kernels and a Hessian power k ∈ (0, 1]. Sophia uses a clipped Gauss–Newton or Hutchinson diagonal for a different update rule.
- **Block:** `experimental.train.AdaHessian`.
- **Experiment:** `ModernOptimizerExperiment`, `SecondOrderShootout`.
- **Toy task:** as 7.3.2.
- **Baselines:** AdamW, Sophia (Hutchinson variant), 7.3.2 AdaFisher.
- **Expected metric:** loss against steps and against time (an HVP costs about 2× a gradient).
- **Source:** [arXiv:2006.00719](https://arxiv.org/abs/2006.00719)

### 7.3.6 Levenberg–Marquardt / damped Gauss–Newton for small networks (Levenberg, 1944; Marquardt, 1963)
For least-squares losses with few parameters (≤ 2k), solve (JᵀJ + λ diag(JᵀJ))δ = −Jᵀr exactly, with the full Jacobian J, raising or lowering λ by whether the step reduced the loss. It converges quadratically near a minimum and is the classic trainer for small regression networks and system identification.
- **Block:** `experimental.train.LevenbergMarquardt` (dense Cholesky through `MlMatrices`).
- **Experiment:** `SecondOrderShootout`, small-network split.
- **Toy task:** a 1-hidden-layer network on `ChaoticSeries` one-step prediction; KAN-fit targets from `KolmogorovArnoldFit`.
- **Baselines:** Adam, L-BFGS (7.3.3), `KanLayer` with Adam.
- **Expected metric:** final MSE (orders of magnitude) and wall time.
- **Source:** [Marquardt 1963](https://epubs.siam.org/doi/10.1137/0111030)

### 7.3.7 True vs empirical Fisher, and exact natural gradient on tiny networks (Kunstner et al., 2019)
A diagnostic item. On networks with ≤ 500 parameters, compute the exact Fisher (sampling labels from the model), the empirical Fisher (using the true labels) and the Hessian, and take natural-gradient steps with each. Kunstner et al. show that the empirical Fisher can distort the step badly (it rescales by the squared gradients, like Adam), which questions "Adam is approximate natural gradient". The results calibrate how good K-FAC, AdaFisher and Shampoo are as approximations.
- **Block:** `experimental.train.ExactNaturalGradient`.
- **Experiment:** `SecondOrderShootout`, diagnostics split.
- **Toy task:** logistic regression and a tiny MLP on spirals; misspecified linear regression.
- **Baselines:** SGD, Adam, 7.3.1 K-FAC.
- **Expected metric:** loss against steps; cosine similarity of each approximate preconditioned step (K-FAC, AdaFisher, Shampoo, Adam) to the exact natural-gradient step.
- **Source:** [arXiv:1905.12558](https://arxiv.org/abs/1905.12558)

### 7.3.8 Deep equilibrium models with Anderson acceleration and implicit differentiation (Bai et al., 2019; Walker & Ni, 2011)
An "infinitely deep" weight-tied layer: the output is the fixed point z* = f_θ(z*, x), found by Anderson acceleration (a quasi-Newton mixing of the last m iterates). Gradients come from the implicit function theorem: one linear solve (I − ∂f/∂z)ᵀu = ∂L/∂z, itself solved by fixed-point iteration, with memory constant in depth. Jacobian regularisation keeps the fixed point stable.
- **Block:** `experimental.nn.DeepEquilibriumLayer`, `experimental.train.AndersonSolver`.
- **Experiment:** a new `experiments.optim.ImplicitDepth`.
- **Toy task:** ShapeImages classification; `ChainTask` multi-hop.
- **Baselines:** `UniversalTransformer` / `LoopedTransformer` (weight tying with explicit unrolling), `RecurrentDepthModel` (Huginn), a plain stack of equal compute.
- **Expected metric:** accuracy, solver iterations, and training memory against the unrolled equivalent.
- **Source:** [arXiv:1909.01377](https://arxiv.org/abs/1909.01377), [Walker & Ni 2011](https://epubs.siam.org/doi/10.1137/10078356X)

### 7.3.9 PowerSGD gradient compression (Vogels et al., 2019)
Compress each gradient matrix to rank r with one step of power iteration (P = MQ, orthogonalise P, Q = MᵀP), warm-started from the previous Q. Each worker sends P and Q (all-reducible), and the residual M − PQᵀ is kept as error feedback and added to the next gradient.
- **Block:** `experimental.train.PowerSgd` (a `GradientTransform` around a simulated all-reduce).
- **Experiment:** `LowCommunicationTraining`.
- **Toy task:** the existing LowCommunicationTraining task with simulated workers.
- **Baselines:** synchronous data parallel, DeMo, DiLoCo, Streaming DiLoCo.
- **Expected metric:** final loss against bytes communicated (the existing axis: 12,009 kB sync vs 45 kB Streaming DiLoCo).
- **Source:** [arXiv:1905.13727](https://arxiv.org/abs/1905.13727)

### 7.3.10 Top-k sparsification with error feedback: DGC and EF21 (Lin et al., 2018; Richtárik et al., 2021)
Send only the k largest-magnitude gradient entries (0.1–1%) and accumulate the rest locally. Deep gradient compression adds momentum correction, local clipping and warm-up of the sparsity. EF21 replaces classic error feedback with a Markov compressor of gradient differences, g_i ← g_i + C(∇f_i − g_i), which has clean convergence guarantees. DeMo already sends a sign-compressed DCT of momentum; this item covers the unstructured sparse family.
- **Block:** `experimental.train.SparseGradientCompression` (`dgc`, `ef21`).
- **Experiment:** `LowCommunicationTraining`.
- **Toy task:** as 7.3.9.
- **Baselines:** sync data parallel, DeMo, 7.3.9 PowerSGD, top-k without error feedback (the ablation).
- **Expected metric:** loss against bytes communicated, for k from 10% down to 0.1%.
- **Source:** [arXiv:1712.01887](https://arxiv.org/abs/1712.01887), [arXiv:2106.05203](https://arxiv.org/abs/2106.05203), [arXiv:1901.09847](https://arxiv.org/abs/1901.09847)

### 7.3.11 QSGD and TernGrad: stochastic gradient quantisation (Alistarh et al., 2017; Wen et al., 2017)
Unbiased stochastic quantisation of gradients to s levels per bucket, scaled by the bucket norm (QSGD), or to {−1, 0, +1} times a scalar (TernGrad), with Elias coding of the integer levels. Unbiasedness means no error feedback is needed, at the cost of variance that grows as s shrinks.
- **Block:** `experimental.train.QuantizedGradients` (reuses the stochastic-rounding code of `StochasticRounding`).
- **Experiment:** `LowCommunicationTraining`.
- **Toy task:** as 7.3.9.
- **Baselines:** fp32 sync, DeMo, 7.3.10.
- **Expected metric:** loss against bits per coordinate (1–8), and the measured gradient variance.
- **Source:** [arXiv:1610.02132](https://arxiv.org/abs/1610.02132), [arXiv:1705.07878](https://arxiv.org/abs/1705.07878)

### 7.3.12 ActNN: compressed activations for backprop (Chen et al., 2021)
Store the activations needed for the backward pass at 2 bits, with per-group min/max scaling and stochastic rounding, and decompress them only when computing gradients. The forward pass stays exact; the gradient is unbiased with bounded extra variance. Bits can be allocated per layer by a sensitivity estimate. This complements gradient checkpointing (`Recompute`), which trades compute for memory instead.
- **Block:** `experimental.train.CompressedActivations` (a custom autograd `Op` wrapper for dense and conv layers).
- **Experiment:** a new `experiments.optim.MemoryEfficientTraining`.
- **Toy task:** the `VisionShootout` CNN and a small transformer LM on `CountingTasks`.
- **Baselines:** full-precision training; `Recompute` checkpointing; bf16 storage.
- **Expected metric:** final accuracy / loss against peak activation bytes and time per step.
- **Source:** [arXiv:2104.14129](https://arxiv.org/abs/2104.14129)

### 7.3.13 Reversible residual networks (Gomez et al., 2017)
Split the features into (x₁, x₂) and use additive coupling, y₁ = x₁ + F(x₂), y₂ = x₂ + G(y₁). Each layer's input can then be recomputed exactly from its output during the backward pass, so activation memory is constant in depth. The same trick gives reversible transformers (Reformer-style).
- **Block:** `experimental.nn.ReversibleBlock` (generic over F and G, with a custom backward).
- **Experiment:** `MemoryEfficientTraining`.
- **Toy task:** a 32-block ResNet on ShapeImages; a 16-layer transformer LM.
- **Baselines:** the non-reversible network with and without `Recompute`; 7.3.12 ActNN.
- **Expected metric:** accuracy, peak memory, time per step, and numerical drift of the reconstructed activations (in fp32 and bf16).
- **Source:** [arXiv:1707.04585](https://arxiv.org/abs/1707.04585)

### 7.3.14 4-bit optimizer states and COAT FP8 states (Li et al., 2023; Xi et al., ICLR 2025)
Shrink the Adam moments beyond core `Adam8bit`. The 4-bit variant uses smaller blocks, rank-1 normalisation of the second moment (row and column scales) and a linear quantiser that excludes zero, which fixes the zero-point failure of 4-bit second moments. COAT stores FP8 states with "dynamic range expansion" (a power transform that matches the state's range to FP8's) and quantises activations in mixed granularity.
- **Block:** `experimental.train.LowBitOptimizerStates` (`fourBit`, `coatFp8`; reuses `MicroscalingFormat` / `Fp8Format`).
- **Experiment:** `MemoryEfficientTraining`; `TrainingStabilityExperiment` row.
- **Toy task:** the transformer LM and the CNN.
- **Baselines:** fp32 Adam, `Adam8bit`, Adam-mini, APOLLO.
- **Expected metric:** final loss against optimizer-state bytes per parameter; divergence count across seeds.
- **Source:** [arXiv:2309.01507](https://arxiv.org/abs/2309.01507), [arXiv:2410.19313](https://arxiv.org/abs/2410.19313)

### 7.3.15 ReLoRA: high-rank training through low-rank restarts (Lialin et al., 2023)
Pretrain with LoRA adapters that are merged into the frozen weights every T steps and then re-initialised, with a partial optimizer-state reset and a jagged learning-rate schedule (a short warm-up after each restart). The sum of many rank-r updates gives a high-rank total update, while memory stays at LoRA levels.
- **Block:** `experimental.train.ReLora` (reuses `LoraAdapterDense`).
- **Experiment:** `MemoryEfficientTraining` pretraining split.
- **Toy task:** a transformer LM on `SyntheticGrammar` from scratch.
- **Baselines:** full-rank AdamW, GaLore, APOLLO, plain LoRA from scratch (no restarts).
- **Expected metric:** validation loss against trainable parameters and optimizer bytes; the effective rank of the total update.
- **Source:** [arXiv:2307.05695](https://arxiv.org/abs/2307.05695)

### 7.3.16 Unit scaling and u-μP (Blake et al., 2023, 2024)
Choose initialisations and per-op scaling factors so that every activation, weight and gradient has unit variance at initialisation, in both the forward and the backward pass. Low-precision formats (FP8, bf16) then need no loss scaling or per-tensor scaling. u-μP combines this with μP, so hyperparameters transfer across width and the defaults are close to optimal.
- **Block:** `experimental.nn.UnitScaledOps` (scaled matmul, softmax and residual add), `experimental.train.UnitMup`.
- **Experiment:** `DepthTransferExperiment` / `ScionTransferExperiment` (learning-rate transfer across width); `Fp4Training` (low-precision row).
- **Toy task:** the transformer LM at widths 32–256.
- **Baselines:** `MaximalUpdate` (μP), `CompleteP`, standard parameterisation with `DynamicLossScaler`.
- **Expected metric:** shift of the optimal learning rate across width; loss in FP8 without loss scaling against fp32.
- **Source:** [arXiv:2303.11257](https://arxiv.org/abs/2303.11257), [arXiv:2407.17465](https://arxiv.org/abs/2407.17465)

### 7.3.17 Shrink and perturb (Ash & Adams, 2020)
Warm-starting from a model trained on part of the data generalises worse than training from scratch, even after convergence. Shrink-and-perturb fixes this: before each new data chunk, multiply all weights by λ < 1 and add small fresh-initialisation noise. This restores gradient diversity and plasticity at little cost.
- **Block:** `experimental.train.ShrinkPerturb` (a step hook).
- **Experiment:** `ContinualLearningExperiment` / `StreamingRl`-style incremental-data split, as a new `experiments.optim.WarmStartGap`.
- **Toy task:** ShapeImages, with data arriving in 10 chunks; a stationary CartPole value fit with growing replay.
- **Baselines:** warm start, training from scratch on all data so far, `ContinualBackprop` (ReDo-style resets).
- **Expected metric:** final test accuracy and the gap to scratch training.
- **Source:** [arXiv:1910.08475](https://arxiv.org/abs/1910.08475)

### 7.3.18 Forward gradients with local losses and activity perturbation (Ren et al., ICLR 2023) — follow-up
The first learning-rule run (`LearnerComparison`) measured plain weight-perturbed forward gradients at 0.79 against 0.987 for backprop on spirals, a large gap that is well explained by gradient-estimator variance, which grows with the number of parameters perturbed. Ren et al. cut the variance in two ways: perturb activations instead of weights (far fewer dimensions), and attach local losses to many small blocks, so each forward gradient covers only a few units. This item is the fair test of whether forward-mode learning can close the gap.
- **Block:** `experimental.train.LocalForwardGradient` (beside `ForwardGradientLearner`; a `LocalLearner` implementation).
- **Experiment:** `experiments.learning.LearnerComparison`.
- **Toy task:** spirals and `ToyTasks` digits (the existing rows).
- **Baselines:** `ForwardGradientLearner` (0.79), backprop (0.987), `FeedbackAlignment`, `MonoForward`.
- **Expected metric:** test accuracy and the measured cosine of the gradient estimate to the true gradient.
- **Source:** [arXiv:2210.03310](https://arxiv.org/abs/2210.03310)

## WP4. Probabilistic ML: probabilistic programming, MCMC and VI, Gaussian processes, kernels, optimal transport and topology (18)

Target: `experimental.ml` (host-side, double precision) for samplers, kernels, OT and TDA; `experimental.train` for weight-space inference; `experimental.nn` for neural OT. Measurements go in a new `experiments.probabilistic` theme, plus the existing `experiments.techniques.UncertaintyBenchmark` (regression gap and image OOD), `experiments.classic.GaussianProcessUncertainty`, `DensityBenchmark` and `ProjectionExperiments`. Baselines: `WeightPosterior` (SWAG, Bayes by backprop, SGLD), `PopulationEnsemble`, `LastLayerLaplace`, `GaussianProcess`, `SparseVariationalGp`, `GaussianMixture`, `MinibatchOptimalTransport` (exact Hungarian OT) and `BridgeMatching`.

### 7.4.1 A tiny probabilistic programming language (Wingate et al., 2011; van de Meent et al., 2018)
A lightweight, trace-based PPL embedded in Java. A model is a lambda that calls `sample(name, dist)` and `observe(dist, value)`. Inference engines re-run the program and handle those calls: likelihood weighting, single-site Metropolis–Hastings on traces (lightweight MH), sequential Monte Carlo with resampling at observe points, and an ADVI back end that uses autograd for the guide. This gives one place where all later samplers plug in.
- **Block:** `experimental.ml.ProbabilisticProgram` (`Trace`, `Distribution`, the `ImportanceSampler`, `TraceMh`, `Smc` engines).
- **Experiment:** a new `experiments.probabilistic.ProbabilisticPrograms`.
- **Toy task:** Bayesian linear regression (exact posterior known); a coin-bias change-point model; an HMM written as a program (compared with `HiddenMarkovModel` forward–backward); the "sprinkler" network (compared with `FactorGraph` variable elimination).
- **Baselines:** the closed-form or exact answers from `experimental.ml` (`HiddenMarkovModel`, `FactorGraph`, `GaussianProcess` for a GP prior written as a program).
- **Expected metric:** posterior-mean error and effective sample size per second, per engine.
- **Source:** [Wingate et al. 2011, AISTATS](https://proceedings.mlr.press/v15/wingate11a.html), [arXiv:1809.10756](https://arxiv.org/abs/1809.10756)

### 7.4.2 Hamiltonian Monte Carlo and NUTS for Bayesian neural networks, with a cold-posterior sweep (Neal, 2011; Hoffman & Gelman, 2014; Wenzel et al., 2020)
HMC proposes distant moves by simulating Hamiltonian dynamics with leapfrog steps, using the gradient of the log posterior, then applies an MH correction. NUTS removes the trajectory-length tuning by doubling the trajectory until it makes a U-turn, and adapts the step size by dual averaging. Full-batch HMC on a small BNN is the "gold standard" posterior (Izmailov et al., 2021). The posterior-temperature sweep (T = 1 vs T < 1) tests the cold-posterior effect.
- **Block:** `experimental.train.HamiltonianMonteCarlo` (`hmc`, `nuts`; the same code is a PPL engine for 7.4.1).
- **Experiment:** `UncertaintyBenchmark` and a new `experiments.probabilistic.PosteriorQuality`.
- **Toy task:** the 1-D regression gap and a 2-D spiral BNN (≤ 1k parameters); the funnel and banana targets for sampler diagnostics.
- **Baselines:** SGLD and SWAG (`WeightPosterior`), deep ensembles, `LastLayerLaplace`, `GaussianProcess`.
- **Expected metric:** NLL, ECE and in-gap/out-of-gap coverage; agreement of each method's predictive distribution with HMC (total variation); R̂ and ESS; NLL against temperature.
- **Source:** [arXiv:1206.1901](https://arxiv.org/abs/1206.1901), [arXiv:1111.4246](https://arxiv.org/abs/1111.4246), [arXiv:2002.02405](https://arxiv.org/abs/2002.02405), [arXiv:2104.14421](https://arxiv.org/abs/2104.14421)

### 7.4.3 Stein variational gradient descent (Liu & Wang, 2016)
A deterministic particle method. n particles are moved along φ(x) = (1/n) Σ_j [k(x_j, x)∇log p(x_j) + ∇_{x_j}k(x_j, x)]: the first term drives particles towards high density, and the kernel-gradient term repels them from each other, so the set approximates the posterior. On BNNs each particle is a full network, which makes it a "repulsive deep ensemble".
- **Block:** `experimental.train.SteinVariationalGradientDescent` (RBF kernel with the median heuristic).
- **Experiment:** `PosteriorQuality`, `UncertaintyBenchmark`.
- **Toy task:** a 2-D mixture of Gaussians (mode coverage); BNN regression gap; ShapeImages OOD.
- **Baselines:** deep ensembles of the same size (`PopulationEnsemble`), SGLD, 7.4.2 HMC.
- **Expected metric:** MMD to HMC samples; NLL, ECE and OOD AUROC; functional diversity of the members.
- **Source:** [arXiv:1608.04471](https://arxiv.org/abs/1608.04471)

### 7.4.4 IVON: improved variational online Newton (Shen et al., ICML 2024)
Mean-field Gaussian variational learning that costs the same as Adam. The weights are sampled from N(m, 1/(N(h + δ))), and the Hessian estimate h is updated by a reparameterisation-trick estimator (no extra backward pass), with a Riemannian-gradient correction that keeps h positive. The update looks like Adam, but its "second moment" is a posterior precision. The paper trains GPT-2 and ImageNet models with better calibration than AdamW.
- **Block:** `experimental.train.Ivon`.
- **Experiment:** `UncertaintyBenchmark`; `ModernOptimizerExperiment` row.
- **Toy task:** the regression gap, ShapeImages OOD, and the transformer LM (for the "does it train as well as AdamW" question).
- **Baselines:** AdamW, Bayes by backprop (`WeightPosterior`), SWAG, MC dropout.
- **Expected metric:** accuracy, NLL, ECE and OOD AUROC at equal steps and cost.
- **Source:** [arXiv:2402.17641](https://arxiv.org/abs/2402.17641)

### 7.4.5 Flow-based variational inference: ADVI, planar/radial flows and IAF (Kucukelbir et al., 2017; Rezende & Mohamed, 2015; Kingma et al., 2016)
Fit q(z) to a posterior by maximising the ELBO with reparameterised gradients. ADVI transforms constrained variables to ℝ and fits a mean-field or full-rank Gaussian. Normalising-flow VI pushes the Gaussian through K planar or radial flows, or an inverse autoregressive flow, so q can be skewed and multimodal; the log-determinant terms are cheap by construction.
- **Block:** `experimental.ml.FlowVariationalInference` (`advi`, `planar`, `radial`, `iaf`; the IAF reuses `MaskedAutoregressiveFlow`).
- **Experiment:** `PosteriorQuality`.
- **Toy task:** the banana, funnel and two-moons posteriors; Bayesian logistic regression on `ToyTasks` spirals features.
- **Baselines:** mean-field Gaussian VI, 7.4.2 NUTS (reference samples).
- **Expected metric:** ELBO, KL to the reference (estimated with NUTS samples), and wall time.
- **Source:** [arXiv:1603.00788](https://arxiv.org/abs/1603.00788), [arXiv:1505.05770](https://arxiv.org/abs/1505.05770), [arXiv:1606.04934](https://arxiv.org/abs/1606.04934)

### 7.4.6 Annealed importance sampling and SMC samplers for marginal likelihoods (Neal, 2001; Del Moral et al., 2006)
Estimate log Z (the evidence) by moving samples through a sequence of tempered distributions p_β ∝ p₀^(1−β) p₁^β, with MCMC transitions at each β and importance weights accumulated along the way. SMC samplers add resampling when the ESS drops. In deep generative models, AIS is the standard way to measure a VAE's test log-likelihood (Wu et al., 2017), which is tighter than the IWAE bound.
- **Block:** `experimental.ml.AnnealedImportanceSampler` (uses the HMC transitions of 7.4.2).
- **Experiment:** `PosteriorQuality`; `HierarchicalVaeExperiment` (evaluation row).
- **Toy task:** Gaussian mixtures with a known log Z; the VAEs of `ImageAutoencoders` / `HierarchicalVae`.
- **Baselines:** naive importance sampling from the prior; the IWAE-k bound already used by `HierarchicalVae`.
- **Expected metric:** log Z bias and variance against compute; the VAE log-likelihood gap between AIS and IWAE-5000.
- **Source:** [arXiv:physics/9803008](https://arxiv.org/abs/physics/9803008), [Del Moral et al. 2006](https://academic.oup.com/jrsssb/article/68/3/411/7110641), [arXiv:1611.04273](https://arxiv.org/abs/1611.04273)

### 7.4.7 Simulation-based inference: SMC-ABC and neural posterior estimation (Sisson et al., 2007; Greenberg et al., 2019; Cranmer et al., 2020)
Inference when the likelihood is intractable but simulation is cheap. ABC keeps parameters whose simulated summary statistics land within ε of the observed ones, with ε shrinking over SMC rounds. Neural posterior estimation trains a conditional density estimator q(θ | x) (a conditional spline flow) on (θ, simulate(θ)) pairs. The amortised posterior then works for any new observation, and the sequential variant (APT) focuses simulations near the observed data.
- **Block:** `experimental.ml.SimulationBasedInference` (`smcAbc`, `neuralPosterior`; reuses `SplineFlow` / `MaskedAutoregressiveFlow` with a conditioner).
- **Experiment:** a new `experiments.probabilistic.SimulatorInference`.
- **Toy task:** Lotka–Volterra predator–prey (4 parameters); a two-moons simulator; the Gaussian-linear model (analytic posterior).
- **Baselines:** rejection ABC; 7.4.2 NUTS on the Gaussian-linear model, where the likelihood is tractable.
- **Expected metric:** C2ST accuracy (a classifier distinguishing reference from estimated posterior samples; 0.5 is perfect) against the number of simulations.
- **Source:** [Sisson et al. 2007, PNAS](https://www.pnas.org/doi/10.1073/pnas.0607208104), [arXiv:1905.07488](https://arxiv.org/abs/1905.07488), [arXiv:1911.01429](https://arxiv.org/abs/1911.01429)

### 7.4.8 Deep Gaussian processes and the spectral mixture kernel (Salimbeni & Deisenroth, 2017; Wilson & Adams, 2013)
A deep GP composes GP layers, f = f_L ∘ … ∘ f_1, each a sparse variational GP with its own inducing points. It is trained by doubly stochastic VI, sampling through the layers with the reparameterisation trick, which gives non-stationary, non-Gaussian marginals. The spectral mixture kernel models the spectral density as a Gaussian mixture, k(τ) = Σ_q w_q exp(−2π²τ²v_q) cos(2πτμ_q), so a single GP can discover and extrapolate quasi-periodic structure.
- **Block:** `experimental.ml.DeepGaussianProcess` (stacks `SparseVariationalGp`), `experimental.ml.SpectralMixtureKernel` (a `StationaryKernel`).
- **Experiment:** `GaussianProcessUncertainty`; `ForecastingBenchmark` row.
- **Toy task:** a step function and a heteroscedastic 1-D series (deep GP); `SeasonalSeries` long-horizon extrapolation (spectral mixture).
- **Baselines:** SVGP with RBF, deep kernel learning (`DeepKernelLearning`, a wave-6 negative), DLinear and seasonal naive for forecasting.
- **Expected metric:** test NLL and CRPS; forecast MSE at horizons 1×–4× the period.
- **Source:** [arXiv:1705.08933](https://arxiv.org/abs/1705.08933), [arXiv:1302.4245](https://arxiv.org/abs/1302.4245)

### 7.4.9 Neural tangent kernel and NNGP kernel regression (Jacot et al., 2018; Lee et al., 2018, 2019)
As width goes to infinity, a randomly initialised network is a GP whose kernel (the NNGP) has a layer-wise closed-form recursion (the arc-cosine kernel for ReLU), and gradient-descent training follows kernel regression with the NTK. Compute both kernels analytically for MLPs and the empirical NTK of finite networks, then compare kernel predictions with trained networks of growing width.
- **Block:** `experimental.ml.InfiniteWidthKernels` (`nngp`, `ntk`, `empiricalNtk`; NNGP and NTK plug into `GaussianProcess` as kernels).
- **Experiment:** a new `experiments.probabilistic.InfiniteWidth`.
- **Toy task:** spirals and the 1-D regression gap, with 200–2,000 training points.
- **Baselines:** trained MLPs at widths 16–2048 (`SimpleMlp`), the RBF GP, `RandomFourierFeatures`.
- **Expected metric:** test accuracy / NLL; distance between the network's and the NTK predictor's outputs against width; how much the empirical NTK moves during training (feature learning).
- **Source:** [arXiv:1806.07572](https://arxiv.org/abs/1806.07572), [arXiv:1711.00165](https://arxiv.org/abs/1711.00165), [arXiv:1902.06720](https://arxiv.org/abs/1902.06720)

### 7.4.10 Kernel two-sample and independence tests: MMD and HSIC (Gretton et al., 2012, 2008)
The unbiased MMD² statistic between two samples, with a permutation test (or the asymptotic null) for p-values and the median-heuristic bandwidth or a learned deep kernel. HSIC is the squared Hilbert–Schmidt norm of the cross-covariance operator; it measures dependence of any form, and its gamma-approximated null gives a test. The repo uses MMD only as a training loss (IMM) and a metric (DTM).
- **Block:** `experimental.ml.KernelTests` (`mmdTest`, `hsicTest`, a deep-kernel MMD).
- **Experiment:** `experiments.probabilistic.KernelHypothesisTests`; also as a sample-quality test inside `GenerativeScores`.
- **Toy task:** Gaussian vs Laplace, mean and variance shifts; non-linear dependence (a ring, sinusoidal); generated vs real `ToyPoints`.
- **Baselines:** the t-test and Kolmogorov–Smirnov per coordinate; Pearson and distance correlation; the sliced-Wasserstein distance from `GenerativeScores`.
- **Expected metric:** test power against sample size at α = 0.05, and the type-I error rate.
- **Source:** [Gretton et al. 2012, JMLR](https://jmlr.org/papers/v13/gretton12a.html), [Gretton et al. 2008, NeurIPS](https://papers.nips.cc/paper/3201-a-kernel-statistical-test-of-independence)

### 7.4.11 Nyström kernel ridge regression and FALKON (Williams & Seeger, 2001; Rudi et al., 2017)
Kernel ridge regression costs O(n³). Nyström picks m ≪ n landmark points and approximates K ≈ K_nm K_mm⁻¹ K_mn. FALKON then solves the Nyström system by conjugate gradient with a Cholesky preconditioner built from the landmarks, which gives optimal statistical rates in O(n·m·t) time.
- **Block:** `experimental.ml.NystromKernelRidge` (`nystrom`, `falkon`; reuses `KernelFunction`).
- **Experiment:** `ClassicMlShootout`, scaling split.
- **Toy task:** spirals and circles up to 50k points; the GP-gap regression.
- **Baselines:** exact KRR / `GaussianProcess` mean at small n, `RandomFourierFeatures` + ridge at an equal feature count, `KernelSvm`.
- **Expected metric:** test error against m and fit time.
- **Source:** [Williams & Seeger 2001](https://papers.nips.cc/paper/1866-using-the-nystrom-method-to-speed-up-kernel-machines), [arXiv:1705.10958](https://arxiv.org/abs/1705.10958)

### 7.4.12 Epinet and DDU: cheap epistemic uncertainty (Osband et al., 2023; Mukhoti et al., 2023)
Two single-model routes to epistemic uncertainty. An epinet is a small network added to a frozen or jointly trained base. It takes the base's features and a random index z, and adds a learned output σ(features, z) plus a fixed random prior, so sampling z gives an "ensemble" whose joint predictions are well calibrated. DDU (deep deterministic uncertainty) fits a class-conditional Gaussian density to the features of a spectrally normalised, residual network, and uses the feature log-density as the epistemic score.
- **Block:** `experimental.nn.Epinet`, `experimental.nn.FeatureDensityUncertainty` (DDU; reuses `GaussianMixture` and `SpectralNormDense`).
- **Experiment:** `UncertaintyBenchmark`.
- **Toy task:** the regression gap, ShapeImages OOD, and Osband's joint log-loss on a "dyadic" sampling of test points.
- **Baselines:** deep ensembles (5×), `DistanceAwareHead` (SNGP/DUQ), `EvidentialHead`, MC dropout.
- **Expected metric:** OOD AUROC, ECE and joint NLL (τ = 10) against forward-pass cost.
- **Source:** [arXiv:2107.08924](https://arxiv.org/abs/2107.08924), [arXiv:2102.11582](https://arxiv.org/abs/2102.11582)

### 7.4.13 Entropic OT: Sinkhorn, Sinkhorn divergences and large Sinkhorn couplings for flow matching (Cuturi, 2013; Feydy et al., 2019; Klein et al., 2025)
Entropic regularisation, OT_ε = min ⟨P, C⟩ − εH(P), is solved by alternating row and column scalings (Sinkhorn iterations, in log space for stability) in O(n²) per iteration. The Sinkhorn divergence S_ε = OT_ε(α, β) − ½OT_ε(α, α) − ½OT_ε(β, β) is a debiased, differentiable loss that interpolates between MMD (ε → ∞) and OT (ε → 0). The 2025 study shows that OT-CFM trained with large Sinkhorn couplings (n in the thousands, small ε) straightens flows well beyond small-minibatch exact OT. The repo's only Sinkhorn is the mHC doubly-stochastic projection, and its OT block is the exact Hungarian solver.
- **Block:** `experimental.ml.EntropicTransport` (`sinkhorn`, `sinkhornDivergence`, `coupling`), used as a pairing in `MinibatchOptimalTransport`.
- **Experiment:** `RectifiedFlowExperiment` / `OneStepGenerators` (pairing choice), `ToyPoints` loss comparisons.
- **Toy task:** ToyPoints moons → rings; ToyPoints generator training with S_ε as the loss.
- **Baselines:** independent pairing; Hungarian minibatch OT at n = 256; reflow; the sliced-Wasserstein loss.
- **Expected metric:** one-step and few-step sample distance and path straightness, against ε and coupling size; Sinkhorn time per iteration.
- **Source:** [arXiv:1306.0895](https://arxiv.org/abs/1306.0895), [arXiv:1810.08278](https://arxiv.org/abs/1810.08278), [arXiv:2506.05526](https://arxiv.org/abs/2506.05526), [arXiv:2302.00482](https://arxiv.org/abs/2302.00482)

### 7.4.14 Neural OT with input-convex neural networks (Amos et al., 2017; Makkuva et al., 2020)
By Brenier's theorem, the quadratic-cost OT map is the gradient of a convex potential, T = ∇φ. An ICNN keeps φ convex in its input: its hidden-to-hidden weights are non-negative and its activations convex and non-decreasing. A min-max dual objective with two ICNNs (φ and its conjugate) learns the Monge map from samples. Once trained, the map is continuous and can transport new points.
- **Block:** `experimental.nn.InputConvexNetwork`, `experimental.nn.NeuralMongeMap`.
- **Experiment:** `BridgeExperiment` (transport rows).
- **Toy task:** Gaussian → Gaussian (a closed-form map, to check), ToyPoints moons ↔ rings.
- **Baselines:** `BridgeMatching`, rectified flow with reflow, 7.4.13 Sinkhorn barycentric projection, the exact discrete OT map on samples.
- **Expected metric:** L²-UVP error against the true map (Gaussian case); transport cost and target fit (sliced W) otherwise.
- **Source:** [arXiv:1609.07152](https://arxiv.org/abs/1609.07152), [arXiv:1908.10962](https://arxiv.org/abs/1908.10962)

### 7.4.15 Gromov–Wasserstein alignment and Wasserstein barycenters (Mémoli, 2011; Peyré et al., 2016; Cuturi & Doucet, 2014)
Gromov–Wasserstein matches two metric spaces that live in different domains by comparing intra-domain distance matrices: min_P Σ |C₁(i, k) − C₂(j, l)|² P_ij P_kl. It is solved by entropic projected gradient (a Sinkhorn step per iteration) and works for graph matching and cross-domain embedding alignment. Wasserstein barycenters average distributions in the OT geometry (fixed-support iterative Bregman projections), which gives "shape averages".
- **Block:** `experimental.ml.GromovWasserstein`, `experimental.ml.WassersteinBarycenter` (on 7.4.13).
- **Experiment:** a new `experiments.probabilistic.TransportGeometry`.
- **Toy task:** matching a `SyntheticGraphs` graph to a permuted, noisy copy; aligning two word-vector spaces from `WordVectors` (different seeds); a barycenter of ShapeImages digit glyphs.
- **Baselines:** spectral matching (Laplacian eigenvectors), Procrustes alignment with known anchors; the Euclidean mean of images.
- **Expected metric:** node-matching accuracy against noise; word-alignment precision@1; visual and sliced-W quality of the barycenter.
- **Source:** [Mémoli 2011](https://link.springer.com/article/10.1007/s10208-011-9093-5), [Peyré et al. 2016, ICML](https://proceedings.mlr.press/v48/peyre16.html), [arXiv:1310.4375](https://arxiv.org/abs/1310.4375)

### 7.4.16 Persistent homology features: Vietoris–Rips barcodes and persistence images (Edelsbrunner & Harer, 2010; Adams et al., 2017)
Build the Vietoris–Rips filtration of a point cloud: add edges and triangles as the radius grows. Track when connected components (H₀) and loops (H₁) are born and die, by boundary-matrix reduction over 𝔽₂, which gives a persistence barcode. Vectorise the barcode as a persistence image (a Gaussian-smoothed, persistence-weighted grid) or as Betti curves, so a standard classifier can use it.
- **Block:** `experimental.ml.PersistentHomology` (`ripsBarcode` for dimensions 0 and 1, `persistenceImage`, `bettiCurve`, `bottleneckDistance`).
- **Experiment:** a new `experiments.probabilistic.TopologicalFeatures`.
- **Toy task:** classify `ToyPoints` clouds by topology: blob, one circle, two circles, figure-eight, torus sample, under noise and random rotations; and classify `ChaoticSeries` regimes by delay embedding.
- **Baselines:** an MLP and `SetEncoder` (Deep Sets) on raw points; `EquivariantGraphLayer` (EGNN) on a k-NN graph.
- **Expected metric:** accuracy, especially under rotation and point-count changes; barcode compute time against n.
- **Source:** [Edelsbrunner & Harer 2010](https://bookstore.ams.org/mbk-69), [arXiv:1507.06217](https://arxiv.org/abs/1507.06217)

### 7.4.17 Topological autoencoders and topology-layer losses (Moor et al., 2020; Gabrielsson et al., 2020)
Make persistence differentiable: each barcode birth or death value equals a specific pairwise distance, so gradients flow to the two points that created it. A topological autoencoder adds a loss that matches the 0-dimensional persistence pairings of the input space and the latent space, so the latent keeps clusters and connectivity. A topology layer can also penalise or encourage a number of loops or components in a generated output.
- **Block:** `experimental.nn.TopologicalLoss` (on 7.4.16's pairings).
- **Experiment:** `ProjectionExperiments` (new rows) and `TopologicalFeatures`.
- **Toy task:** nested spheres / linked rings in 10-D projected to 2-D; ShapeImages features.
- **Baselines:** a plain autoencoder, PCA, t-SNE and UMAP (`NeighbourEmbedding`).
- **Expected metric:** trustworthiness, KL of density, and bottleneck distance between input and latent barcodes.
- **Source:** [arXiv:1906.00722](https://arxiv.org/abs/1906.00722), [arXiv:1905.12200](https://arxiv.org/abs/1905.12200)

### 7.4.18 Dirichlet-process mixture models (Neal, 2000; Kulis & Jordan, 2012)
Bayesian non-parametric clustering where the number of components is inferred. A collapsed Gibbs sampler draws each point's assignment from the Chinese-restaurant-process prior times the conjugate (Normal–inverse-Wishart) predictive, opening a new cluster with probability ∝ α. A truncated stick-breaking variational version, and DP-means (its small-variance limit: k-means with a penalty λ per cluster), are fast alternatives.
- **Block:** `experimental.ml.DirichletProcessMixture` (`gibbs`, `variational`, `dpMeans`; reuses `GaussianMixture` sufficient statistics).
- **Experiment:** `ClassicMlShootout`, model-selection split.
- **Toy task:** `ToyPoints` blobs with an unknown K from 2 to 12, unequal sizes, and a stream where new clusters appear over time.
- **Baselines:** `GaussianMixture` with BIC selection, `KMeans` with the elbow rule, `Dbscan`.
- **Expected metric:** adjusted Rand index and error in K; held-out log-likelihood.
- **Source:** [Neal 2000](https://www.tandfonline.com/doi/abs/10.1080/10618600.2000.10474879), [arXiv:1111.0352](https://arxiv.org/abs/1111.0352)

## WP5. Multi-agent RL, game theory, robotics-style control and evolutionary methods (18)

Target: `experimental.rl` (agents, games and controllers) and `experimental.evolve` (optimisers). Measurements go in:

- a new `experiments.games` theme: `ImperfectInformationGames` and `MatrixGameDynamics`;
- `experiments.control.GoalsImitationAndTeams` (cooperative MARL) and a new `experiments.control.ClassicalControl`;
- `experiments.evolution` (new `BlackBoxOptimizers` and `MultiObjectiveSearch`).

Baselines: independent learners (PPO / `Dqn` per agent), `ValueDecomposition` (QMIX/VDN), `EloLeague` self-play, `AlphaZero`, `CemPlanner`, `TdMpc` (MPPI), `DifferentiableSim`, SAC/PPO, and `CmaEs`, `EvolutionStrategy`, `MapElites` and `Neat`, over at least 3 seeds. Two small game environments are new: Kuhn poker and Leduc poker (`experimental.rl.KuhnPoker`, `experimental.rl.LeducPoker`), which have exact best responses for exploitability.

### 7.5.1 Counterfactual regret minimisation: CFR, CFR+ and MCCFR (Zinkevich et al., 2007; Tammelin, 2014; Lanctot et al., 2009)
Solve two-player zero-sum imperfect-information games by regret minimisation at every information set. Each iteration traverses the game tree, computes counterfactual values, accumulates regrets, and sets the next strategy by regret matching; the average strategy converges to a Nash equilibrium. CFR+ clips negative cumulative regrets and weights the average linearly, which converges much faster. External-sampling MCCFR samples the opponent's and chance's actions, so iterations are cheap.
- **Block:** `experimental.rl.CounterfactualRegret` (`cfr`, `cfrPlus`, `mccfr`), `experimental.rl.KuhnPoker`, `experimental.rl.LeducPoker`, `experimental.rl.Exploitability` (an exact best-response calculator).
- **Experiment:** `experiments.games.ImperfectInformationGames`.
- **Toy task:** Kuhn poker (the known equilibrium value is −1/18) and Leduc poker.
- **Baselines:** independent PPO self-play, `EloLeague` self-play with Dqn, uniform random.
- **Expected metric:** exploitability (mbb/hand) against iterations and against wall time.
- **Source:** [Zinkevich et al. 2007](https://papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information), [arXiv:1407.5042](https://arxiv.org/abs/1407.5042), [Lanctot et al. 2009](https://papers.nips.cc/paper/3713-monte-carlo-sampling-for-regret-minimization-in-extensive-games)

### 7.5.2 Fictitious play and neural fictitious self-play (Brown, 1951; Heinrich & Silver, 2016)
In fictitious play, each player best-responds to the empirical average of the opponent's past strategies. NFSP makes this neural: each agent has a best-response Q-network trained by DQN on reservoir-free replay, and an average-policy network trained by supervised learning (with a reservoir buffer) on its own best-response actions. It acts with a mixture of the two (anticipatory parameter η).
- **Block:** `experimental.rl.NeuralFictitiousSelfPlay` (reuses `Dqn`, `ReservoirReplay`, `BehaviorCloning`).
- **Experiment:** `ImperfectInformationGames`.
- **Toy task:** Kuhn and Leduc poker.
- **Baselines:** 7.5.1 CFR+ (the convergence reference), DQN self-play (which cycles), `EloLeague` self-play.
- **Expected metric:** exploitability of the average policy against environment steps.
- **Source:** [arXiv:1603.01121](https://arxiv.org/abs/1603.01121)

### 7.5.3 PSRO and the double oracle (Lanctot et al., 2017; McMahan et al., 2003)
Policy-Space Response Oracles grows a population of policies. It plays every pair to build an empirical payoff matrix (a meta-game), solves the meta-game for a meta-strategy (Nash or α-rank), and trains a new best response (by RL) against that mixture. With a Nash meta-solver this is the double-oracle algorithm, which avoids the strategy cycling of naive self-play.
- **Block:** `experimental.rl.PolicySpaceResponseOracles` (an LP meta-solver in `experimental.ml`, PPO best responses).
- **Experiment:** `ImperfectInformationGames`; `SelfPlayArena` row.
- **Toy task:** Kuhn/Leduc poker; a non-transitive "rock–paper–scissors with skill" game; `ConnectThree`.
- **Baselines:** naive self-play, `EloLeague` (fictitious-play-like league), 7.5.2 NFSP.
- **Expected metric:** exploitability (poker); population Nash-averaged Elo (ConnectThree).
- **Source:** [arXiv:1711.00832](https://arxiv.org/abs/1711.00832), [McMahan et al. 2003, ICML](https://cdn.aaai.org/ICML/2003/ICML03-067.pdf)

### 7.5.4 Regularised last-iterate equilibrium learning: magnetic mirror descent, R-NaD and Nash policy gradient (Sokota et al., 2023; Perolat et al., 2022; 2025)
Plain policy gradient and CFR's current iterate cycle in zero-sum games; only their averages converge. MMD adds a KL "magnet" towards a reference policy to each mirror-descent step, so the last iterate converges to a regularised equilibrium. R-NaD (DeepNash's Stratego algorithm) periodically moves the magnet to the current policy, which converges to the Nash equilibrium itself. Nash policy gradient (2025) uses an iteratively refined regulariser inside ordinary policy-gradient training, with no average-policy network.
- **Block:** `experimental.rl.RegularizedEquilibriumLearner` (`mmd`, `rnad`, `nashPg`).
- **Experiment:** `ImperfectInformationGames`; `MatrixGameDynamics`.
- **Toy task:** matching pennies and rock–paper–scissors (last-iterate trajectories); Kuhn and Leduc poker.
- **Baselines:** PPO self-play (cycles), 7.5.1 CFR+ average policy, 7.5.2 NFSP.
- **Expected metric:** last-iterate exploitability against iterations.
- **Source:** [arXiv:2206.05825](https://arxiv.org/abs/2206.05825), [arXiv:2206.15378](https://arxiv.org/abs/2206.15378), [arXiv:2510.18183](https://arxiv.org/abs/2510.18183)

### 7.5.5 Optimistic and extragradient methods for min-max games (Korpelevich, 1976; Daskalakis et al., 2018; Gidel et al., 2019; Mescheder et al., 2017) — follow-up
Simultaneous gradient descent–ascent cycles or diverges on the bilinear game min_x max_y xy, and GAN training inherits this. Extragradient takes a look-ahead half step and then updates with the gradient at the look-ahead point. Optimistic GDA uses 2∇_t − ∇_{t−1}, the same correction with no extra gradient call. Consensus optimisation adds a penalty on the gradient-field norm. This item also gives the early WGAN-GP negative ("needs far more generator steps") a fairer optimiser.
- **Block:** `experimental.train.MinMaxOptimizers` (`extragradient`, `optimisticAdam`, `consensus`; wraps core `Adversarial`).
- **Experiment:** `MatrixGameDynamics` (bilinear and quadratic games); `GanExperiment` (new rows).
- **Toy task:** bilinear and Dirac-GAN games; the two-blob WGAN-GP of `GanExperiment`.
- **Baselines:** simultaneous and alternating GDA with Adam; TTUR (already in `GanExperiment`); spectral norm (sliced W 0.393).
- **Expected metric:** distance to equilibrium against iterations; WGAN-GP sliced W against generator steps.
- **Source:** [arXiv:1711.00141](https://arxiv.org/abs/1711.00141), [arXiv:1802.10551](https://arxiv.org/abs/1802.10551), [arXiv:1705.10461](https://arxiv.org/abs/1705.10461)

### 7.5.6 Centralised critics: MADDPG and COMA (Lowe et al., 2017; Foerster et al., 2018)
Centralised training with decentralised execution for actor-critic methods. MADDPG gives each agent a deterministic actor on its own observation and a critic Q_i(s, a₁…a_n) that sees everyone's actions, which makes the critic's target stationary. COMA, for discrete actions, uses one centralised critic and a counterfactual advantage: Q(s, a) − Σ_{a'_i} π_i(a'_i) Q(s, (a_{−i}, a'_i)), which marginalises out only agent i's action, so credit is assigned per agent.
- **Block:** `experimental.rl.Maddpg` (reuses `Td3`'s deterministic actor), `experimental.rl.CounterfactualMultiAgent` (COMA).
- **Experiment:** `GoalsImitationAndTeams` (MARL rows).
- **Toy task:** `MultiAgentEnvironment` cooperative tasks; a continuous "cooperative navigation" particle world (3 agents, 3 landmarks; a small new env).
- **Baselines:** independent learners, QMIX/VDN (`ValueDecomposition`).
- **Expected metric:** team return and coverage against steps.
- **Source:** [arXiv:1706.02275](https://arxiv.org/abs/1706.02275), [arXiv:1705.08926](https://arxiv.org/abs/1705.08926)

### 7.5.7 MAPPO and IPPO (Yu et al., 2022; de Witt et al., 2020)
The surprising finding: PPO is a strong multi-agent learner when tuned properly. IPPO runs independent PPO per agent with parameter sharing and agent IDs. MAPPO adds a centralised value function on the global state. Both use value normalisation, death masking and a small number of epochs per batch. `MultiAgentEnvironment` cites MAPPO in its javadoc, but there is no implementation.
- **Block:** `experimental.rl.MultiAgentPpo` (`ippo`, `mappo`; reuses core `Ppo`).
- **Experiment:** `GoalsImitationAndTeams`.
- **Toy task:** as 7.5.6.
- **Baselines:** QMIX/VDN, 7.5.6 MADDPG and COMA, independent DQN.
- **Expected metric:** team return against environment steps.
- **Source:** [arXiv:2103.01955](https://arxiv.org/abs/2103.01955), [arXiv:2011.09533](https://arxiv.org/abs/2011.09533)

### 7.5.8 Opponent shaping: LOLA and M-FOS (Foerster et al., 2018; Lu et al., 2022)
Naive learners in social dilemmas converge to mutual defection. LOLA differentiates through the opponent's anticipated learning step, ∇_θ₁ V₁(θ₁, θ₂ + Δθ₂(θ₁)), which leads to reciprocity (tit-for-tat) in the iterated prisoner's dilemma. Model-free opponent shaping (M-FOS) learns a meta-policy over whole learning episodes that shapes any learner, with no differentiable opponent model.
- **Block:** `experimental.rl.OpponentShaping` (`lola`, `mfos`), `experimental.rl.IteratedMatrixGame` (IPD, matching pennies, stag hunt).
- **Experiment:** `MatrixGameDynamics`.
- **Toy task:** the iterated prisoner's dilemma with one-step memory policies; the coin game (a small grid).
- **Baselines:** naive policy-gradient learners on both sides.
- **Expected metric:** average reward per step and the probability of cooperation after defection; exploitation of a naive learner.
- **Source:** [arXiv:1709.04326](https://arxiv.org/abs/1709.04326), [arXiv:2205.01447](https://arxiv.org/abs/2205.01447)

### 7.5.9 Learned communication: CommNet and TarMAC (Sukhbaatar et al., 2016; Das et al., 2019)
Agents exchange continuous messages that are learned end to end. In CommNet, each agent's hidden state is updated with the mean of the other agents' hidden states, over several communication rounds. TarMAC adds targeted messages: each sender emits a key and a value, each receiver attends with its own query, so the network learns whom to listen to.
- **Block:** `experimental.rl.CommunicatingAgents` (`commNet`, `tarMac`).
- **Experiment:** `GoalsImitationAndTeams`; `ReferentialGame` (comparison row).
- **Toy task:** a partially observed "predator–prey" or "traffic junction" in `MultiAgentEnvironment`, where only one agent sees the goal.
- **Baselines:** no-communication MAPPO (7.5.7), a discrete Gumbel channel (as in `ReferentialGame`), a full-observation oracle.
- **Expected metric:** success rate; attention entropy (does TarMAC attend to the informed agent?).
- **Source:** [arXiv:1605.07736](https://arxiv.org/abs/1605.07736), [arXiv:1810.11187](https://arxiv.org/abs/1810.11187)

### 7.5.10 LQR, iLQR and DDP (Kalman, 1960; Li & Todorov, 2004; Tassa et al., 2012)
Classical optimal control. LQR solves the discrete Riccati recursion for a linear model with quadratic cost. iLQR linearises the dynamics (and quadratises the cost) around the current trajectory, solves the time-varying LQR backward pass, and rolls out forward with a line search, with Levenberg–Marquardt regularisation. DDP adds second-order dynamics terms. With a known or learned model it beats sampling planners in samples per decision.
- **Block:** `experimental.rl.LinearQuadraticControl` (`lqr`, `ilqr`, `ddp`; uses autograd Jacobians of `DifferentiableSim` or of a learned `MlpWorldModel`).
- **Experiment:** a new `experiments.control.ClassicalControl`.
- **Toy task:** Pendulum swing-up, CartPole balance (and swing-up), and the mass-spring reach of `DifferentiablePhysicsControl`.
- **Baselines:** `CemPlanner`, MPPI (`TdMpc`'s planner), backprop through the simulator (`DifferentiablePhysicsControl`, 0.024 from goal), SAC.
- **Expected metric:** cost and success rate against planner time per step; the same with a learned model.
- **Source:** [Li & Todorov 2004](https://homes.cs.washington.edu/~todorov/papers/LiICINCO04.pdf), [Tassa et al. 2012, IROS](https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf)

### 7.5.11 Model-based diffusion for trajectory optimisation (Pan et al., NeurIPS 2024)
Use the reverse diffusion process as a trajectory optimiser, with no training: the target distribution is p(τ) ∝ exp(−J(τ)/λ), and its score at each noise level is estimated by Monte Carlo from rollouts of the known model (as in MPPI, but annealed over noise levels). This escapes local minima where MPPI and CEM get stuck, and demonstrations can be mixed in as a data prior.
- **Block:** `experimental.rl.ModelBasedDiffusionPlanner`.
- **Experiment:** `ClassicalControl`.
- **Toy task:** Pendulum swing-up, a 2-D point mass navigating around obstacles (a non-convex cost), Lander.
- **Baselines:** `CemPlanner`, MPPI, 7.5.10 iLQR, `DiffusionPolicy` (trained, as an amortised reference).
- **Expected metric:** final cost and success rate at an equal number of rollouts per decision.
- **Source:** [arXiv:2407.01573](https://arxiv.org/abs/2407.01573)

### 7.5.12 Domain randomisation and rapid motor adaptation (Tobin et al., 2017; Kumar et al., 2021)
Make a policy robust to model mismatch by training over randomised physics (masses, lengths, friction, actuation gains). RMA adds an adaptation module: during training, a base policy gets the true physics parameters through an encoder; an adaptation network is then trained to predict that latent from the recent state–action history, so the deployed policy identifies the system online.
- **Block:** `experimental.rl.DomainRandomization` (an `Environment` wrapper over Pendulum and CartPole parameters), `experimental.rl.RapidMotorAdaptation`.
- **Experiment:** `ClassicalControl`, sim-to-real split ("real" = held-out parameter ranges).
- **Toy task:** Pendulum and CartPole with mass and length outside the training range.
- **Baselines:** PPO/SAC trained on nominal physics; a domain-randomised policy without adaptation; `MetaRlAgent` (RL²).
- **Expected metric:** return on held-out physics, and return in the first 50 steps after a mid-episode parameter change.
- **Source:** [arXiv:1703.06907](https://arxiv.org/abs/1703.06907), [arXiv:2107.04034](https://arxiv.org/abs/2107.04034)

### 7.5.13 PILCO: probabilistic inference for learning control (Deisenroth & Rasmussen, 2011)
A very data-efficient model-based method. It learns a GP dynamics model, propagates a Gaussian state distribution through it by analytic moment matching, computes the expected long-horizon cost, and improves an RBF or linear policy with analytic gradients through the whole rollout. CartPole swing-up is classically solved in a handful of episodes.
- **Block:** `experimental.rl.Pilco` (uses `experimental.ml.GaussianProcess` with SE-kernel moment matching).
- **Experiment:** `ClassicalControl`, data-efficiency split.
- **Toy task:** Pendulum swing-up and CartPole swing-up.
- **Baselines:** SAC, PPO, `TdMpc`, `LatentPlanner` (PlaNet), `DreamerAgent`.
- **Expected metric:** return against real environment steps (log scale).
- **Source:** [Deisenroth & Rasmussen 2011, ICML](https://www.doc.ic.ac.uk/~mpd37/publications/pilco-icml2011.pdf)

### 7.5.14 Dynamic movement primitives (Ijspeert et al., 2013)
Encode a demonstrated trajectory as a stable second-order attractor, τÿ = α(β(g − y) − ẏ) + f(x), where the forcing term f is a weighted sum of Gaussian basis functions in a decaying phase variable x. A DMP learns from one demonstration by linear regression, generalises to new goals and time scales, and can be refined by RL in the low-dimensional weight space (PI², or reward-weighted regression).
- **Block:** `experimental.rl.MovementPrimitive` (fit, rollout, and policy improvement by reward-weighted averaging).
- **Experiment:** `ChunkedRobotPolicies` (comparison rows) and `ClassicalControl`.
- **Toy task:** the 2-D reach of `MultimodalReach` / `WindyReach`, with goals not seen in the demonstrations.
- **Baselines:** behaviour cloning (MLP), `DiffusionPolicy`, `FlowActionPolicy`.
- **Expected metric:** goal error on new goals from 1, 5 and 20 demonstrations; smoothness (jerk).
- **Source:** [Ijspeert et al. 2013, Neural Computation](https://direct.mit.edu/neco/article/25/2/328/7853)

### 7.5.15 Differential evolution and particle swarm optimisation (Storn & Price, 1997; Kennedy & Eberhart, 1995)
Two classic population optimisers. DE/rand/1/bin forms a mutant a + F(b − c) from three random members, applies binomial crossover with the target, and keeps the better of the two. PSO moves particles with inertia plus random pulls towards their personal best and the swarm best. Both are strong on multimodal, non-separable black-box functions and need no covariance matrix.
- **Block:** `experimental.evolve.DifferentialEvolution`, `experimental.evolve.ParticleSwarm`.
- **Experiment:** a new `experiments.evolution.BlackBoxOptimizers`.
- **Toy task:** Rastrigin, Rosenbrock, Ackley and Schwefel in 10–50 dimensions; neuroevolution of CartPole and Pendulum policies.
- **Baselines:** `CmaEs`, `EvolutionStrategy` (OpenAI-ES), random search.
- **Expected metric:** best value against function evaluations; success rate within a budget.
- **Source:** [Storn & Price 1997](https://link.springer.com/article/10.1023/A:1008202821328), [Kennedy & Eberhart 1995](https://ieeexplore.ieee.org/document/488968)

### 7.5.16 NSGA-II: multi-objective evolution (Deb et al., 2002)
Rank the population into Pareto fronts by fast non-dominated sorting, and break ties within a front by crowding distance, so the population spreads along the front. Elitism comes from merging parents and offspring before selection. This item uses it for joint accuracy, size and latency search over network architectures.
- **Block:** `experimental.evolve.Nsga2` (plus a hypervolume indicator).
- **Experiment:** a new `experiments.evolution.MultiObjectiveSearch`.
- **Toy task:** ZDT1–3 test functions; MLP/CNN architecture search on ShapeImages for accuracy vs parameters vs MACs (the `ArchitectureSearchShootout` space).
- **Baselines:** weighted-sum scalarisation with `CmaEs`, `MapElites` (with a size descriptor), random search.
- **Expected metric:** hypervolume against evaluations; spread of the front.
- **Source:** [Deb et al. 2002](https://ieeexplore.ieee.org/document/996017)

### 7.5.17 Evolutionary model merging (Akiba et al., Nature Machine Intelligence 2025)
Search, with CMA-ES, over how to merge several fine-tuned models: per-layer mixing coefficients in parameter space (task arithmetic, TIES or DARE weights), and, in data-flow space, which layers of which model to stack in what order. Fitness is the score on a small validation set, so merged models can combine skills (for example maths and another language) without gradient training. The repo has merging methods and uses merging as a crossover, but nothing searches the merge recipe.
- **Block:** `experimental.evolve.EvolutionaryMerge` (on `ModelMerging` and `CmaEs`).
- **Experiment:** `experiments.evolution.PopulationEnsembles` (new rows), or a new `experiments.evolution.MergeSearch`.
- **Toy task:** two `ArithmeticTask` specialists (addition, multiplication) and a `SyntheticGrammar` LM fine-tuned from one base; the target is a mixed task.
- **Baselines:** uniform task arithmetic, TIES, DARE, SLERP and Fisher merging with hand-set coefficients; multi-task fine-tuning (the upper bound).
- **Expected metric:** mixed-task accuracy against fitness evaluations.
- **Source:** [arXiv:2403.13187](https://arxiv.org/abs/2403.13187)

### 7.5.18 Evolution strategies at scale for LLM fine-tuning (Qiu et al., 2025)
Full-parameter fine-tuning of language models with plain ES: perturb all weights with seeded Gaussian noise, evaluate each perturbed model's sequence-level reward, and step along the reward-weighted noise. Only seeds and scalars need to be communicated. The paper reports better sample efficiency than PPO/GRPO on reasoning tasks, more stable runs, and less reward hacking, because ES optimises a smoothed reward over parameters rather than over token probabilities.
- **Block:** `experimental.evolve.EvolutionStrategyFineTuner` (reuses `ParameterVector` seeded noise and `EvolutionStrategy`).
- **Experiment:** `experiments.reasoning.ArithmeticRl` (new rows).
- **Toy task:** `ArithmeticTask` and `ChainTask` with the existing RLVR rewards; the "conciseness" reward-hacking probe from `LengthControlledThinking`.
- **Baselines:** `GrpoTrainer` (GRPO, DAPO), `PpoLmTrainer`, `MeZo` (zeroth-order SGD).
- **Expected metric:** accuracy against reward evaluations; variance across 3 seeds; KL to the base model at equal gain.
- **Source:** [arXiv:2509.24372](https://arxiv.org/abs/2509.24372)

## WP6. Speech and audio toys, program synthesis, neuro-symbolic learning and fair-test follow-ups of earlier negatives (18)

Target:

- Items 7.6.1–7.6.6 (audio) go in `experimental.nn` and a new `experiments.audio` theme, next to `TalkingDigits` (formant digits + CTC) and `ToyAudioCodec`. They use core `Signal` (STFT, log-mel, SpecAugment). The baseline is `TalkingDigits`' frame-stacked MLP + CTC recogniser, at equal parameters.
- Items 7.6.7–7.6.10 (program synthesis) go in `experimental.evolve` / `experimental.llm` and a new `experiments.synthesis` theme, on the `ListPrograms` DSL. Baselines: the enumerative synthesiser in `SelfImprovingCoders`, `LibraryLearner` and `ProgramEvolution`.
- Items 7.6.11–7.6.14 (neuro-symbolic) go in `experimental.nn` / `experimental.train` and `experiments.reasoning`, with `LatinSquarePuzzles` (TRM, HRM, AKOrN) and `NeuroSymbolicProver` as baselines.
- Items 7.6.15–7.6.18 are follow-ups of earlier negative results, in `experimental.train` / `experimental.ml`, measured in the experiments that produced the negative.

### 7.6.1 MFCC features and dynamic time warping templates (Davis & Mermelstein, 1980; Sakoe & Chiba, 1978)
The classic recogniser. Compute MFCCs (log-mel energies from core `Signal`, then a DCT-II, keeping 13 coefficients plus deltas). Store one or a few template utterances per word, and classify by the minimum dynamic-time-warping distance, with a Sakoe–Chiba band. It needs no training, which gives the floor that learned recognisers must beat in the few-data regime.
- **Block:** `experimental.nn.Mfcc` (on `Signal.logMel`), `experimental.ml.DynamicTimeWarping`.
- **Experiment:** a new `experiments.audio.SpokenDigitRecognition` (isolated-word split built from `TalkingDigits`' synthesiser).
- **Toy task:** isolated spoken digits with speaker (formant) variation and noise, using 1, 5 and 50 examples per class.
- **Baselines:** the `TalkingDigits` MLP + CTC on log band energies; the same MLP on MFCCs.
- **Expected metric:** word accuracy against examples per class and SNR.
- **Source:** [Davis & Mermelstein 1980](https://ieeexplore.ieee.org/document/1163420), [Sakoe & Chiba 1978](https://ieeexplore.ieee.org/document/1163055)

### 7.6.2 Conformer and Mamba (Samba-ASR) encoders with log-mel + SpecAugment input (Gulati et al., 2020; Shakhadri et al., 2025)
The Conformer sandwiches multi-head self-attention and a depthwise-convolution module between two half-step feed-forward layers (macaron style), so local acoustic patterns and global context are both modelled. Samba-ASR replaces the attention encoder with Mamba state-space blocks, giving linear-time encoding. Both read log-mel features with SpecAugment. These already exist in core `Signal`, but `TalkingDigits` uses neither.
- **Block:** `experimental.nn.ConformerBlock` (reuses core `MultiHeadAttention` and depthwise `Conv`); the Samba encoder is a stack of `MambaLayer`.
- **Experiment:** `experiments.audio.SpokenDigitRecognition` (continuous-string split with CTC).
- **Toy task:** `TalkingDigits` digit strings, plus a harder variant with overlapping formants, fast speech and noise.
- **Baselines:** the `TalkingDigits` MLP + CTC; a transformer encoder; a BiGRU; each with and without SpecAugment.
- **Expected metric:** character (digit) error rate and parameters.
- **Source:** [arXiv:2005.08100](https://arxiv.org/abs/2005.08100), [arXiv:2501.02832](https://arxiv.org/abs/2501.02832), [arXiv:1904.08779](https://arxiv.org/abs/1904.08779)

### 7.6.3 RNN-Transducer loss (Graves, 2012)
CTC assumes the output labels are conditionally independent given the audio. RNN-T adds a prediction network over the previous labels and a joint network, z(t, u) = J(enc_t, pred_u). The loss sums over all alignments on the T × U lattice by a forward–backward recursion with blank and label transitions, which makes the model streaming-capable and label-context-aware.
- **Block:** `experimental.nn.TransducerLoss` (forward–backward in log space, with greedy and beam decoding).
- **Experiment:** `SpokenDigitRecognition`.
- **Toy task:** `TalkingDigits` strings with a planted label dependency (for example, digits follow a Markov grammar), where label context helps.
- **Baselines:** CTC (`CtcLoss`) on the same encoder (7.6.2); CTC + an n-gram LM rescorer (`KneserNeyLm`).
- **Expected metric:** digit error rate; latency in frames before each emission.
- **Source:** [arXiv:1211.3711](https://arxiv.org/abs/1211.3711)

### 7.6.4 wav2vec 2.0 / HuBERT-style self-supervised speech pretraining (Baevski et al., 2020; Hsu et al., 2021)
Pretrain on unlabelled audio and fine-tune with few labels. wav2vec 2.0 masks spans of convolutional features, and a transformer must pick the true quantised latent for each masked frame among distractors (contrastive loss, Gumbel product quantiser, diversity loss). HuBERT instead predicts k-means cluster IDs of MFCCs (then of its own features) for the masked frames. Wave 6 cut these as "covered by TalkingDigits", but no audio SSL exists in the repo.
- **Block:** `experimental.nn.MaskedSpeechPretraining` (`wav2vec2`, `hubert`; reuses `GumbelChannel`, `KMeans` and `SelfSupervisedLosses`).
- **Experiment:** `SpokenDigitRecognition`, low-label split.
- **Toy task:** 10k unlabelled synthetic utterances, then CTC fine-tuning with 10, 50 and 200 labelled strings.
- **Baselines:** the same encoder trained from scratch; 7.6.1 MFCC + DTW.
- **Expected metric:** digit error rate against labelled utterances.
- **Source:** [arXiv:2006.11477](https://arxiv.org/abs/2006.11477), [arXiv:2106.07447](https://arxiv.org/abs/2106.07447)

### 7.6.5 DDSP: a harmonic-plus-noise differentiable synthesiser (Engel et al., 2020)
The network predicts interpretable synthesiser controls, not samples: a fundamental frequency, the amplitudes of each harmonic, and a filtered-noise spectrum per frame. A differentiable additive synthesiser and a noise filter render the audio, and training uses a multi-scale spectral loss (core `Signal.stft`). The strong inductive bias gives high-quality audio from tiny models and little data.
- **Block:** `experimental.nn.HarmonicNoiseSynth`, `experimental.nn.MultiScaleSpectralLoss`.
- **Experiment:** `experiments.audio.AudioResynthesis` (with `ToyAudioCodec`).
- **Toy task:** resynthesise `TalkingDigits` utterances and simple melodies from extracted f0 and loudness; timbre transfer between two synthetic "instruments".
- **Baselines:** `WaveNet` and `ToyAudioCodec` (RVQ) at an equal parameter count.
- **Expected metric:** multi-scale spectral distance, f0 error and parameters; recognisability by the `TalkingDigits` recogniser.
- **Source:** [arXiv:2001.04643](https://arxiv.org/abs/2001.04643)

### 7.6.6 Conv-TasNet with permutation-invariant training (Luo & Mesgarani, 2019; Kolbæk et al., 2017)
Time-domain source separation: a learned 1-D convolutional encoder replaces the STFT, a temporal convolutional network predicts one mask per source, and a transposed-convolution decoder reconstructs each source. Utterance-level permutation-invariant training (uPIT) computes the loss (negative SI-SNR) under the best assignment of outputs to sources, which removes the label-permutation problem.
- **Block:** `experimental.nn.ConvTasNet` (reuses `TemporalConvNet`), `experimental.nn.PermutationInvariantLoss`.
- **Experiment:** `experiments.audio.SourceSeparation`.
- **Toy task:** mixtures of two `TalkingDigits` speakers (different formant sets) at 0 and 5 dB.
- **Baselines:** FastICA (`ComponentAnalysis`, which needs multi-channel input, so use 2-microphone mixtures for it), an STFT-mask MLP with PIT, and the ideal ratio mask (the oracle).
- **Expected metric:** SI-SNR improvement (dB); digit error rate of the `TalkingDigits` recogniser on separated outputs.
- **Source:** [arXiv:1809.07454](https://arxiv.org/abs/1809.07454), [arXiv:1703.06284](https://arxiv.org/abs/1703.06284)

### 7.6.7 DeepCoder: neural-guided enumerative synthesis (Balog et al., 2017)
A network reads the input/output examples of a task and predicts which DSL primitives the solution uses. The predictions then order or prune the enumerative search (sort-and-add, or DFS ordered by probability). Search stays exact and the network only guides it, which gives large speed-ups at equal solve rate.
- **Block:** `experimental.evolve.GuidedEnumerator` (an attribute-predictor MLP over encoded IO examples, plus the observational-equivalence enumerator factored out of `SelfImprovingCoders`).
- **Experiment:** a new `experiments.synthesis.ProgramSynthesisShootout`.
- **Toy task:** `ListPrograms` tasks of length 2–5, with random programs as training data.
- **Baselines:** the unguided enumerator with observational-equivalence pruning; `LibraryLearner` (DreamCoder); `ProgramEvolution`.
- **Expected metric:** solve rate against programs enumerated or wall time (the classic "speed-up at equal solve rate" curve).
- **Source:** [arXiv:1611.01989](https://arxiv.org/abs/1611.01989)

### 7.6.8 RobustFill: neural program generation from examples (Devlin et al., 2017)
An attention seq2seq model encodes each input/output example pair separately, decodes the program token by token while attending over all pairs (late max-pooling across examples), and uses beam search with execution filtering. Only beam candidates consistent with all examples are kept. It is compared with "neural program induction", where the network outputs the answer directly.
- **Block:** `experimental.llm.ProgramGenerator` (reuses `Seq2SeqAttention` / `TransformerSeq2Seq`).
- **Experiment:** `ProgramSynthesisShootout`.
- **Toy task:** `ListPrograms` tasks; generalisation to programs longer than in training.
- **Baselines:** 7.6.7 DeepCoder, the plain enumerator, an induction model (a transformer that maps a new input to its output).
- **Expected metric:** generalisation accuracy on held-out inputs, against beam width; time per task.
- **Source:** [arXiv:1703.07469](https://arxiv.org/abs/1703.07469)

### 7.6.9 Genetic programming for symbolic regression with lexicase selection (Koza, 1992; Helmuth et al., 2015; Cranmer, 2023)
Evolve expression trees (+, −, ×, ÷, sin, exp, constants) by subtree crossover and mutation, with constant optimisation and a parsimony pressure. Lexicase selection filters candidates on test cases in a random order rather than by aggregate error, which preserves specialists and diversity. PySR-style multi-population search with simplification is the modern reference.
- **Block:** `experimental.evolve.SymbolicRegression` (an expression tree, GP operators, `lexicase` and `tournament` selection).
- **Experiment:** `experiments.synthesis.SymbolicRegressionBenchmark`.
- **Toy task:** Feynman-style equations with 1–3 variables (for example x·y/(1+z), sin(x)·eˣ); the extrapolation split of `ArithmeticExtrapolation`.
- **Baselines:** `NeuralArithmeticUnit` (NALU), `KanLayer` with symbolic snapping, an MLP; tournament selection (the ablation).
- **Expected metric:** exact-recovery rate; extrapolation error outside the training range.
- **Source:** [Koza 1992](https://mitpress.mit.edu/9780262527910/genetic-programming/), [Helmuth et al. 2015](https://ieeexplore.ieee.org/document/6920034), [arXiv:2305.01582](https://arxiv.org/abs/2305.01582)

### 7.6.10 The Neural GPU (Kaiser & Sutskever, 2016)
A convolutional GRU applied to a 2-D state of width equal to the input length, for as many steps as the input is long. It is a weight-tied, parallel cellular computer that can learn binary addition and multiplication from short examples and run them on much longer ones, provided training uses curriculum, parameter sharing relaxation and noise.
- **Block:** `experimental.nn.NeuralGpu`.
- **Experiment:** `ArithmeticExtrapolation` (length-generalisation rows).
- **Toy task:** binary addition and multiplication, trained on 1–20 bits and tested on 100–200 bits.
- **Baselines:** a transformer with RoPE / NoPE (`PositionShootout` setups), `LoopedTransformer`, `NeuralArithmeticUnit` (operand-level, not digit-level), `NeuralCellularAutomaton`-style updates.
- **Expected metric:** exact sequence accuracy against test length.
- **Source:** [arXiv:1511.08228](https://arxiv.org/abs/1511.08228)

### 7.6.11 DeepProbLog / Scallop-style neural predicates (Manhaeve et al., 2018; Li et al., 2023)
A logic program calls neural networks as probabilistic facts, for example `digit(Img, D)` with probabilities from a classifier. The query's probability is computed by exact or top-k provenance (weighted model counting over the proofs), and gradients flow back through it into the classifier. Images can thus be trained from labels on the result alone ("the two digits sum to 7").
- **Block:** `experimental.nn.NeuralPredicateProgram` (a small Datalog with a top-k proofs semiring and differentiable provenance).
- **Experiment:** a new `experiments.reasoning.NeuroSymbolicArithmetic`.
- **Toy task:** "digit addition" from `ToyTasks` digit images, with only the sum supervised (1-digit, then multi-digit numbers); sorting digit images by order relations.
- **Baselines:** an end-to-end CNN + MLP regressing the sum from both images; a CNN trained with the true digit labels (the oracle).
- **Expected metric:** sum accuracy and implied single-digit accuracy against training examples; generalisation to 3- and 4-digit numbers.
- **Source:** [arXiv:1805.10872](https://arxiv.org/abs/1805.10872), [arXiv:2304.04812](https://arxiv.org/abs/2304.04812)

### 7.6.12 Semantic loss (Xu et al., 2018)
A loss that measures how far a network's output distribution is from satisfying a logical constraint: −log WMC(α, p), the weighted model count of the constraint under the predicted Bernoulli probabilities. It is computed exactly by compiling the constraint to a circuit, or by enumeration for small constraints (exactly-one, path, permutation). It improves semi-supervised learning and the validity of structured outputs.
- **Block:** `experimental.train.SemanticLoss` (constraint compiler for exactly-one, at-most-k and Latin-square row/column constraints).
- **Experiment:** `LatinSquarePuzzles` (new rows); semi-supervised ShapeImages.
- **Toy task:** Latin squares (all-different rows and columns); ShapeImages with 1% labels plus the exactly-one constraint on unlabelled data.
- **Baselines:** the plain MLP (0.87 on Latin squares), TRM and HRM, entropy minimisation for semi-supervised learning.
- **Expected metric:** fully valid solution rate; semi-supervised accuracy.
- **Source:** [arXiv:1711.11157](https://arxiv.org/abs/1711.11157)

### 7.6.13 Differentiable inductive logic programming and neural logic machines (Evans & Grefenstette, 2018; Dong et al., 2019)
Learn first-order rules from examples. ∂ILP generates candidate clauses from templates, assigns each a softmax weight, and runs differentiable forward chaining (a fuzzy conjunction and disjunction over valuations) for T steps. The learned weights read off as a logic program. Neural Logic Machines instead use tensors of arity 0–3 with MLPs and quantifier-style reductions (exists, forall) between arities, which scales to larger relational tasks.
- **Block:** `experimental.nn.DifferentiableIlp`, `experimental.nn.NeuralLogicMachine`.
- **Experiment:** a new `experiments.reasoning.RuleLearning`.
- **Toy task:** family-tree relations (grandparent, uncle), graph reachability, "even", and the blocks-world sort from NLM; train on 10-node worlds and test on 50-node worlds.
- **Baselines:** a GNN (GIN) and a transformer over relation triples; 7.1.13 KG embeddings for the family tree.
- **Expected metric:** accuracy on larger worlds; exactness of the extracted rules (∂ILP).
- **Source:** [arXiv:1711.04574](https://arxiv.org/abs/1711.04574), [arXiv:1904.11694](https://arxiv.org/abs/1904.11694)

### 7.6.14 SATNet: a differentiable MAXSAT layer (Wang et al., 2019) — follow-up
A layer that solves a semidefinite relaxation of MAXSAT by fast coordinate descent (the mixing method) in the forward pass, and differentiates the solution analytically in the backward pass. The clause matrix S is learned, so the layer learns the rules of a logic puzzle from input–output examples alone. This targets the early negative where TRM and HRM (about 0.72) lost to a plain MLP (about 0.87) on `LatinSquarePuzzles`, with an architecture whose inductive bias is constraint satisfaction.
- **Block:** `experimental.nn.SatNetLayer`.
- **Experiment:** `LatinSquarePuzzles`.
- **Toy task:** Latin squares at the existing sizes; parity of 20 bits (SATNet's other benchmark).
- **Baselines:** the MLP (about 0.87), TRM, HRM, `KuramotoLayer` (AKOrN, 0.62 solved), 7.6.12 semantic loss.
- **Expected metric:** cell accuracy and fully solved rate; training time.
- **Source:** [arXiv:1905.12149](https://arxiv.org/abs/1905.12149)

### 7.6.15 V-REx and Just Train Twice for spurious correlations (Krueger et al., 2021; Liu et al., 2021) — follow-up
Wave 6 found IRM worse than ERM at every penalty weight (worst group 0.268 vs 0.379). Two alternatives that often work where IRM fails. V-REx penalises the variance of the training risk across environments, which is simpler and more stable than IRM's gradient penalty. JTT needs no group labels: train ERM briefly, upweight the examples it misclassifies (which are mostly minority groups) by λ_up, and retrain.
- **Block:** `experimental.train.GroupRobustTraining` (new `vrex`, `justTrainTwice` modes; beside the existing GroupDRO and IRM).
- **Experiment:** `experiments.safety.CausalAndFairness`.
- **Toy task:** the existing coloured-shapes spurious split.
- **Baselines:** ERM (0.379 worst group), IRM (0.268), GroupDRO (which needs group labels).
- **Expected metric:** worst-group and average accuracy, and whether group labels were needed.
- **Source:** [arXiv:2003.00688](https://arxiv.org/abs/2003.00688), [arXiv:2107.09044](https://arxiv.org/abs/2107.09044)

### 7.6.16 DAGMA and GES for causal structure learning (Bello et al., 2022; Chickering, 2002) — follow-up
Wave 6 found NOTEARS worse than the PC algorithm (SHD 5 vs 3). DAGMA replaces NOTEARS' trace-exponential acyclicity function with a log-determinant one, h(W) = −log det(sI − W∘W) + d log s. Its gradients do not vanish and the central path is better conditioned, which gives more accurate graphs at a fraction of the time. GES is the classic score-based alternative: greedy forward then backward search over equivalence classes with the BIC score.
- **Block:** `experimental.ml.Dagma` (beside `Notears`), `experimental.ml.GreedyEquivalenceSearch`.
- **Experiment:** `CausalAndFairness` (structure-learning rows).
- **Toy task:** the existing synthetic SCM, plus Erdős–Rényi DAGs with 10–20 nodes under linear-Gaussian and non-linear mechanisms.
- **Baselines:** NOTEARS (SHD 5), the PC skeleton (SHD 3).
- **Expected metric:** SHD, true and false positive rates, and wall time against graph size.
- **Source:** [arXiv:2209.08037](https://arxiv.org/abs/2209.08037), [Chickering 2002, JMLR](https://www.jmlr.org/papers/v3/chickering02b.html)

### 7.6.17 Incremental predictive coding (Salvatori et al., ICLR 2024) — follow-up
Standard predictive coding runs inference to equilibrium before each weight update, and on looped graphs this becomes unstable or slow (`LearnerComparison`: about 0.53 on a looped Brain graph vs 0.96 on a feed-forward one). Incremental PC updates the value nodes and the weights simultaneously at every inference step (an incremental EM). It is faster, needs no inference-length tuning, and is reported to be stable on arbitrary (including cyclic) graph topologies.
- **Block:** `experimental.train.PredictiveCodingNetwork` (a new `incremental` mode).
- **Experiment:** `experiments.learning.LearnerComparison`.
- **Toy task:** digits on the looped Brain graph (the negative row) and on the sparse feed-forward Brain graph; spirals.
- **Baselines:** standard PC (0.53 looped, 0.96 feed-forward), backprop.
- **Expected metric:** test accuracy and wall time per epoch.
- **Source:** [arXiv:2212.00720](https://arxiv.org/abs/2212.00720)

### 7.6.18 I-JEPA with the full recipe and a learning-curve budget (Assran et al., 2023) — follow-up
An early run found I-JEPA-lite and BYOL at or near the random-encoder baseline (0.39 and 0.42 vs 0.34–0.40) "at this budget". This item re-runs them as a fair test, with the paper's recipe pieces that the lite version skipped: multi-block target masks with a large context block, a cosine EMA momentum schedule from 0.996 to 1.0, a predictor narrower than the encoder, and 10× the steps. The budget-matched comparison is reported at both the old and the new step count. The 7.1.11 graph SSL methods share the same "beat the random encoder" protocol.
- **Block:** `experimental.nn.SelfSupervisedLosses` (I-JEPA multi-block masking and the EMA schedule, via `MomentumTeacher`).
- **Experiment:** `experiments.selfsupervised.SelfSupervisedBenchmark`.
- **Toy task:** ShapeImages linear probe (the existing protocol).
- **Baselines:** the random encoder (0.34–0.40), MAE (0.59), SimCLR (0.56).
- **Expected metric:** linear-probe accuracy against steps (a learning curve rather than one point), and representation rank (an effective-rank collapse check).
- **Source:** [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)

---

## Notes on dependencies between items

**Graphs (WP1)**
- First create the `experimental.graph` package and the `GraphLearningShootout` harness. Add the new `SyntheticGraphs` generators: the heterophilous SBM, CSL / regular graphs, neighbours-match trees, random SAT pairs and algorithm traces.
- 7.1.1's Laplacian and RWSE encodings are reused by 7.1.8 (as a baseline), 7.1.15 and 7.1.16 (structural features). Build them first, reusing `SpectralClustering`'s Laplacian.
- 7.1.2, 7.1.3, 7.1.6 and 7.1.18 share one propagation helper (normalised adjacency powers). 7.1.4 and 7.1.5 share the heterophily sweep; one agent should do both.
- 7.1.7 and 7.1.17 share the neighbours-match task.
- 7.1.12 reuses the SGNS code of `StaticWordEmbeddings`. 7.1.13 reuses autograd `Complex`, and its family-tree KG is shared with 7.6.13.
- 7.1.14 and 7.1.15 go in `GraphReasoning`; 7.1.16 needs 7.1.1's attention layer.

**Recsys, retrieval and tabular (WP2)**
- Extend `SyntheticRecsys` with sequences, timestamps and content features (7.2.2, 7.2.5, 7.2.6) and with a logged-bandit generator (7.2.7, 7.2.8). Build `experimental.data.SyntheticCorpus` for 7.2.9–7.2.14.
- 7.2.9's inverted index is reused by 7.2.11; 7.2.9's dense retriever is the base for 7.2.10 and 7.2.14.
- 7.2.13's LambdaMART changes core `GradientBoostedTrees`; land it in a separate core PR.
- 7.2.15 → 7.2.16 → 7.2.18 share `TabularShootout`. 7.2.17 needs a dataset prior; it is the slowest item of the package (pretraining), so start it early.

**Numerics and compression (WP3)**
- Build one curvature toolkit first: layer hooks for (a, g), Gauss–Newton- and Hessian-vector products, and a CG solver. 7.3.1, 7.3.2, 7.3.4, 7.3.5 and 7.3.7 all use it; 7.3.8's implicit backward and 7.3.6's damping logic reuse the solver. TRPO's existing CG helper (`TrustRegion`) can be generalised.
- 7.3.9–7.3.11 plug into `LowCommunicationTraining`'s simulated all-reduce; one agent should do all three.
- 7.3.12–7.3.15 share the new `MemoryEfficientTraining` harness with a peak-memory meter. 7.3.14 reuses `MicroscalingFormat` / `Fp8Format`.
- 7.3.16 touches `MaximalUpdate`-style parameterisation; coordinate with whoever owns `DepthTransferExperiment`.
- 7.3.18 only adds a `LocalLearner`.

**Probabilistic (WP4)**
- 7.4.1 PPL first: 7.4.2 HMC/NUTS, 7.4.5 VI and 7.4.6 AIS become its engines, and 7.4.7 uses its simulators.
- 7.4.2's NUTS samples are the reference for 7.4.3, 7.4.4, 7.4.5 and 7.4.12 in `PosteriorQuality` and `UncertaintyBenchmark`.
- 7.4.8 and 7.4.9 extend `GaussianProcess` kernels; 7.4.11 reuses `KernelFunction`.
- 7.4.13 is the base of 7.4.14 (a comparison row) and 7.4.15 (its inner solver). 7.4.13's pairing change touches `MinibatchOptimalTransport`; keep the Hungarian path as the default.
- 7.4.16 → 7.4.17: build the barcode code once, with persistence pairs exposed for gradients.

**Games, control and evolution (WP5)**
- Build `KuhnPoker`, `LeducPoker` and the exact `Exploitability` calculator first; 7.5.1–7.5.4 all report exploitability. 7.5.3 needs an LP solver for meta-Nash (a small simplex in `experimental.ml`), which 7.5.4's matrix games also use.
- 7.5.5 changes only optimiser wrappers, but adds rows to `GanExperiment`.
- 7.5.6, 7.5.7 and 7.5.9 share a new continuous cooperative-navigation env and `MultiAgentEnvironment`; one agent should do all three. 7.5.8 is independent.
- 7.5.10 → 7.5.11 → 7.5.13 share `ClassicalControl` and a "known model" interface over `DifferentiableSim` and the core Pendulum and CartPole physics. 7.5.13 needs `GaussianProcess` from `experimental.ml`. 7.5.12 wraps core environments only.
- 7.5.15–7.5.17 share a black-box `Fitness` interface with `CmaEs`. 7.5.18 touches `ArithmeticRl` and the `GrpoTrainer` comparisons; schedule it after any open `GrpoTrainer` work.

**Audio, synthesis, neuro-symbolic, follow-ups (WP6)**
- Build `experiments.audio.SpokenDigitRecognition` first, with an isolated-word split and a harder string split. 7.6.1 → 7.6.2 → 7.6.3 → 7.6.4 then build on one encoder interface. 7.6.5 and 7.6.6 are independent of the recognisers but use the recogniser as a metric.
- 7.6.7 factors the enumerator out of `SelfImprovingCoders`; do it before 7.6.8. 7.6.9 is independent. 7.6.10 extends `ArithmeticExtrapolation`.
- 7.6.11–7.6.14 share a small Datalog / constraint representation: 7.6.12's constraint compiler is used by 7.6.14 as a baseline, and 7.6.13 shares the family-tree KG with 7.1.13.
- 7.6.15 and 7.6.16 both touch `CausalAndFairness`; one agent should do both. 7.6.17 changes `PredictiveCodingNetwork`; 7.6.18 changes `SelfSupervisedLosses` and must keep the existing rows reproducible.

**Cross-package**
- `experimental.graph`, `experimental.recsys` and the WP4 `experimental.ml` additions have no dependencies on each other and can start immediately.
- WP5 7.5.13 (PILCO) needs only the existing `experimental.ml.GaussianProcess`; WP4 7.4.8 extends its kernels, so land 7.4.8 after 7.5.13 starts or coordinate the kernel interface.
- WP4 7.4.13 (Sinkhorn) is the inner solver of 7.4.15 and an optional pairing for WP1 7.1.16's diffusion training; WP3 7.3.8 and WP3 7.3.4 share the CG / fixed-point solver code.
- WP1 7.1.13 and WP6 7.6.13 share the family-tree knowledge graph; whichever lands first owns the generator in `experimental.data`.
- Core-worthy items: 7.2.13 LambdaMART (core `GradientBoostedTrees`), 7.3.1 K-FAC, 7.3.3 L-BFGS and possibly 7.3.16 unit scaling (`brainiac-train` / `brainiac-nn`). Land them in a separate PR from the experiments.
