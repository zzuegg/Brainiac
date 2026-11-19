# Brainiac: ideas not yet implemented

*Compiled 2026-11-17 against `claude/project-thread-8zvcl0` (current main). It checks `experimental-ai-ideas.md` and a sweep of well-known current training and inference techniques against the core modules (`brainiac-nn`, `-train`, `-llm`, `-rl`, `-evolve`, `-autograd`, `-data`, `-tensor`) and against `brainiac-experiments` (packages `experimental.*` and `experiments.*`, plus `RESULTS.md`). Before listing an item I grepped `src/main` for its class names, paper names and javadoc keywords, excluding `build/` and `.claude/worktrees/`. An item is listed only when nothing matched, or when the only matches were unrelated (for example a "clip" match for CLIP).*

## How this was checked

- **X1–X31 review list.** No review document with that list exists under `/mnt/project-files/research/` or in the repo, and the catalogue does not reference one. The numbers survive only as tags in `experiment-results.md`, `RESULTS.md`, javadoc and commit messages: X2, X4, X6, X7, X10, X13–X31. X1, X3, X5, X8, X9, X11 and X12 appear nowhere, and X20 appears only in a commit message (switching-neuron MoE). Every tagged X item has a class and a measured experiment, so none of them is listed below. If the review doc turns up, check its X1/X3/X5/X8/X9/X11/X12 entries against this list.
- **Catalogue coverage.** About 150 of the roughly 170 catalogue entries already have an `experimental.*` class, and most of those also have an `experiments.*` measurement. The entries still missing are tagged **[cat]** below. Four entries are only partly done and are tagged **[cat, partial]**: 3D Gaussian Splatting (only `GaussianSplats2d` exists), thermodynamic DTMs (only `PBitIsingMachine` exists, with no denoising chain), SpikeGPT (only the `LifLayer` classifier exists, with no spiking LM) and modular duality (only `MaximalUpdate` and `Muon` exist).
- **Techniques found to exist already, and so left out.** μP (`MaximalUpdate`), Schedule-Free, WSD/SGDR/1cycle (`TrainingSchedules`), MTP, differential attention, NSA, TTT, Titans, BLT patching (`EntropyPatcher`, `BytePatchPooling`), mHC, DPO/IPO/KTO/ORPO/SimPO, GRPO/Dr. GRPO/DAPO, SPIN, self-rewarding, RFT, Medusa, Jacobi/prompt-lookup decoding, core speculative decoding, early exit, model soups/TIES/DARE/task arithmetic, logit distillation, GQA, QK-norm, soft-capping, z-loss, AGC, aux-loss-free MoE balancing, DyT, LayerScale, DeepNorm, Peri-LN, PiSSA/VeRA/IA3/DoRA/LoRA+, QLoRA, GPTQ/AWQ/SmoothQuant/LSQ/FP8, min-p/typical/eta/DRY, YaRN/NTK, StreamingLLM sinks, H2O/SnapKV, FlashAttention-style tiling (CPU kernels and Vulkan flash decoding), mixed precision with loss scaling, SuperBPE/BPE-dropout, MinHash dedup, DoReMi-style mixing, document-masked packing, sequence-length warm-up.

**Where an item should live:**
- `experimental.<pkg>`: a new block in `brainiac-experiments`.
- `experiments.<theme>`: a new or extended runnable experiment.
- **core-worthy**: belongs in a core module once measured. Each item still needs an experimental block first.

## Summary

| # | Work package | Items |
|---|---|---|
| WP1 | Attention, residual and MoE architecture refinements | 16 |
| WP2 | Optimizers and training dynamics | 14 |
| WP3 | Post-training: RL for LLMs, preference variants, distillation, PEFT | 17 |
| WP4 | Inference, serving and quantization | 13 |
| WP5 | Agents, reasoning search and open-ended self-improvement | 10 |
| WP6 | Generative, multimodal, physical and exotic | 15 |
| | **Total** | **85** |

---

## WP1. Attention, residual and MoE architecture refinements (16)

Target: `experimental.nn`, measured in `experiments.attention.AttentionShootout`, `experiments.sequence.MixerShootout` and `experiments.techniques.PositionShootout`. The shared baseline is the core `TransformerBlock` (and GQA `Llama` where KV size matters), at the same parameter count and step budget.

| # | Item | One line | Lives in | Experiment vs baseline |
|---|---|---|---|---|
| 1.1 | **MLA (multi-head latent attention)** | DeepSeek-V2: K and V are compressed into one low-rank latent per token, with decoupled RoPE keys; the cache stores only the latent. | `experimental.nn.LatentAttention`; core-worthy for `brainiac-llm` | AttentionShootout row: validation loss and KV bytes per token, compared with MHA and GQA. |
| 1.2 | **Value residual learning (ResFormer)** | Every layer's V is mixed with the first layer's V, which eases over-smoothing in deep stacks. | `experimental.nn.ValueResidualBlock` | AttentionShootout: loss after N steps at depth 8 and 16, compared with `TransformerBlock`. |
| 1.3 | **Cross-layer KV sharing (CLA / YOCO)** | Adjacent layers share one KV projection, or the upper half reuses one global cache. | `experimental.nn.CrossLayerKvBlock` | LongContextTraining: loss compared with KV memory, against an unshared stack. |
| 1.4 | **Sigmoid attention** | Elementwise sigmoid with a `-log n` bias in place of softmax; no row normalisation. | `experimental.nn.SigmoidAttention` | AttentionShootout plus length extrapolation in PositionShootout, compared with softmax. |
| 1.5 | **Stick-breaking attention** | Each key claims a stick-breaking share of the attention, so recency is built in and no position encoding is needed. | `experimental.nn.StickBreakingAttention` | PositionShootout: exact-lag task and extrapolation, compared with RoPE and NoPE. |
| 1.6 | **Scalable-Softmax (SSMax) / log-n logit scaling** | Multiply the logits by `s·log n` so attention does not flatten as context grows. | `experimental.nn` (option on `PositionalAttention`); core-worthy | PositionShootout: accuracy at 4x and 8x the training length, compared with plain softmax. |
| 1.7 | **Softpick** | A rectified, sum-not-to-one softmax that removes attention sinks and massive activations. | `experimental.nn.Softpick` | AttentionShootout: loss, the sink share (via `AttentionCacheViewer`) and 4-bit quantization error, compared with softmax and `GatedAttention`. |
| 1.8 | **Learnable sink logits / register tokens** | A learned per-head "attend to nothing" logit (as in gpt-oss), or extra register tokens (as in ViT registers). | `experimental.nn` (flag on attention blocks) | AttentionShootout and VisionShootout: loss and attention-map entropy, compared with no sink. |
| 1.9 | **Talking-heads attention** | Learned mixing across heads before and after the softmax. | `experimental.nn.TalkingHeadsAttention` | AttentionShootout: loss at the same parameter count, compared with MHA. |
| 1.10 | **Selective attention** | A parameter-free rule in which a token can mask later attention to earlier tokens, shrinking the effective context. | `experimental.nn.SelectiveAttention` | EvictionComparison: loss and kept context size, compared with H2O and full attention. |
| 1.11 | **Canon layers** | Short causal depthwise convolutions inserted around attention and the MLP, to share information horizontally. | `experimental.nn.CanonLayer` | AttentionShootout and MixerShootout recall tasks, compared with a block without canon layers. |
| 1.12 | **iRoPE (interleaved NoPE/RoPE layers)** | Every k-th layer drops the position encoding (and may use temperature-scaled attention) for length generalisation. | `experimental.nn` (layer schedule in `TunableTransformerBlock`) | PositionShootout: extrapolation, compared with all-RoPE and all-NoPE stacks. |
| 1.13 | **Hymba-style parallel attention + SSM heads** | Attention heads and Mamba heads run in parallel inside one layer, with learnable meta tokens. | `experimental.nn.ParallelHybridBlock` | MixerShootout: recall and loss, compared with `HybridLinearStack` (Kimi 3:1, sequential). |
| 1.14 | **Activation sparsity (TEAL / CATS)** | Training-free magnitude thresholding of hidden activations, which skips weight columns at decode time. | `experimental.nn.SparseActivationDense`; core-worthy for CPU decode | DecodingShowdown: tokens/s and perplexity at 25/40/50% sparsity, compared with dense. |
| 1.15 | **Sparse upcycling / drop-upcycling** | Initialise a MoE from a trained dense FFN by copying it, optionally re-initialising part of each expert. | `experimental.nn` (factory on `FineGrainedMoe`) | AttentionShootout MoE row: loss after continued training, compared with a MoE from scratch and the dense model at equal compute. |
| 1.16 | **Expert-choice routing with capacity factor and router z-loss** | Experts pick their top tokens (perfect balance); the token-choice variant gets a capacity factor and the ST-MoE router z-loss. | `experimental.nn.ExpertChoiceMoe` | AttentionShootout MoE row: loss and expert load variance, compared with `FineGrainedMoe` top-k. |

## WP2. Optimizers and training dynamics (14)

Target: `experimental.train`. Each item gets a row in `experiments.techniques.TrainingTricksExperiment` or `experiments.optim` (extending the OptimizerZoo-style comparisons), with AdamW, core `Muon` and `Lion` as baselines at a tuned learning rate.

| # | Item | One line | Lives in | Experiment vs baseline |
|---|---|---|---|---|
| 2.1 | **Grokfast** | Amplify the slow (EMA or moving-average) component of the gradient to speed up grokking. | `experimental.train.Grokfast` (a `GradientTransform`) | GrokkingExperiment: steps until validation accuracy exceeds 0.99, compared with AdamW plus weight decay. |
| 2.2 | **Cautious optimizers (C-AdamW / C-Lion)** | Mask update components whose sign disagrees with the current gradient. | `experimental.train.Cautious` (wrapper) | TrainingTricksExperiment: loss at a fixed step budget, compared with plain AdamW and Lion. |
| 2.3 | **MARS** | Variance-reduced preconditioned updates (STORM-style gradient correction) on top of AdamW or Lion. | `experimental.train.Mars` | TrainingTricksExperiment on a small LM: steps to target loss, compared with AdamW. |
| 2.4 | **ADOPT** | An Adam variant that converges for any β₂ by normalising with the previous second moment. | `experimental.train.Adopt` | TrainingTricksExperiment: loss and stability over a β₂ sweep, compared with Adam. |
| 2.5 | **Adam-mini** | One second-moment scalar per parameter block (head or neuron), which saves about 50% of optimizer memory. | `experimental.train.AdamMini` | TrainingTricksExperiment: loss and optimizer-state bytes, compared with AdamW and core `Adam8bit`. |
| 2.6 | **StableAdamW / Adam-atan2** | AdamW with per-tensor update clipping, or `atan2(m, sqrt v)` with no ε. | `experimental.train.StableAdamW` | A high-learning-rate stress row in TrainingTricksExperiment: divergence rate across seeds, compared with AdamW. |
| 2.7 | **SPAM and loss-spike skipping** | Reset momentum and clip spiky gradients when a gradient or loss spike is detected, then skip the bad batch. | `experimental.train.SpikeGuard` | TrainingTricksExperiment with injected outlier batches: final loss and number of recoveries, compared with AdamW plus norm clipping. |
| 2.8 | **PSGD Kron** | A Kronecker-factored preconditioner fitted online via Lie-group updates. | `experimental.train.PsgdKron` | TrainingTricksExperiment: loss per step and per second, compared with `Shampoo`, `Soap` and Muon. |
| 2.9 | **Scion / modular-norm optimizers** [cat, partial] | Per-layer dual-norm (spectral, ℓ∞, sign) updates from the modular-duality theory; Muon is one special case. | `experimental.train.Scion` | Learning-rate transfer across widths 64→512, compared with Muon plus `MaximalUpdate`. |
| 2.10 | **MuonClip (QK-clip)** | Muon with a post-step rescale of W_q and W_k whenever an attention logit exceeds a threshold. | `experimental.train.MuonClip`; core-worthy (option on `Muon`) | High-learning-rate transformer run: maximum attention logit and loss, compared with Muon. |
| 2.11 | **LAWA / checkpoint averaging** | A running average of the last k checkpoints, taken during training rather than at the end. | `experimental.train.CheckpointAveraging` | TrainingTricksExperiment: validation loss at each checkpoint, compared with core `Ema` and `StochasticWeightAveraging`. |
| 2.12 | **Batch-size ramp** | Grow the batch size during training (a critical-batch-size schedule) in place of decaying the learning rate. | `experimental.train.BatchSizeSchedule` | TrainingTricksExperiment: loss at an equal token budget, compared with a fixed batch plus cosine schedule. |
| 2.13 | **Stochastic rounding** | Round low-precision (bf16 or 8-bit) weights and updates stochastically so small updates are not lost. | `experimental.train` (an option for mixed-precision `Trainer`); core-worthy | Pure-bf16 weight training: final loss, compared with round-to-nearest bf16 and fp32 master weights. |
| 2.14 | **Data pruning by difficulty (EL2N / forgetting scores)** | Score examples early in training and drop the easiest or noisiest fraction. | `experimental.data.DataPruner` | VisionShootout on digits: accuracy compared with the fraction kept, against random pruning. |

## WP3. Post-training: RL for LLMs, preference variants, distillation, PEFT (17)

Target: `experimental.llm` and `experimental.train`. Measurements go in `experiments.reasoning.ArithmeticRl`, `experiments.language.PreferenceTuning`, `experiments.reasoning.SelfImprovement`, `experiments.language.PeftShootout` and a new `experiments.language.DistillationShootout`.

| # | Item | One line | Lives in | Experiment vs baseline |
|---|---|---|---|---|
| 3.1 | **PPO for LLMs (RLHF PPO with value head)** | The classic token-level PPO with a learned critic, GAE and a KL-to-reference penalty. | `experimental.llm.PpoLmTrainer` | ArithmeticRl: greedy accuracy and wall time at equal samples, compared with `GrpoTrainer` (GRPO, DAPO). |
| 3.2 | **RLOO** | REINFORCE with a leave-one-out group baseline and no clipping or critic. | `experimental.llm` (a `GrpoTrainer.Variant`, or `RlooTrainer`) | ArithmeticRl: accuracy compared with GRPO and Dr. GRPO. |
| 3.3 | **ReMax / REINFORCE++** | REINFORCE with a greedy-rollout baseline (ReMax), or with global batch advantage normalisation and a token KL (REINFORCE++). | `experimental.llm` (more variants) | ArithmeticRl: accuracy and variance across seeds, compared with GRPO. |
| 3.4 | **GSPO** | A sequence-level importance ratio and clipping in place of token-level ones (Qwen3). | `experimental.llm` (a `GrpoTrainer.Variant`) | ArithmeticRl: accuracy and training stability, compared with GRPO and DAPO. |
| 3.5 | **cDPO / robust DPO** | DPO with label smoothing (conservative) or an unbiased noisy-label correction (rDPO). | `experimental.train.PreferenceLosses` (new methods) | PreferenceTuning with 20% flipped preference labels: win rate compared with DPO. |
| 3.6 | **DPOP and β-DPO** | DPO-Positive adds a penalty that keeps the chosen likelihood from falling; β-DPO adapts β per batch. | `experimental.train.PreferenceLosses` | PreferenceTuning: chosen log-likelihood drift and win rate, compared with DPO. |
| 3.7 | **STaR (self-taught reasoner)** | Fine-tune on self-generated rationales that reach the right answer, with "rationalisation" hints on failures. | `experimental.llm.StarTrainer` | SelfImprovement and ArithmeticRl: accuracy after k rounds, compared with `RejectionSamplingFineTuning`, which has no rationalisation. |
| 3.8 | **Quiet-STaR** [cat] | Learn hidden thought tokens at every position via REINFORCE on the improvement in next-token likelihood. | `experimental.llm.QuietStar` | CoconutCurriculum and ArithmeticTask: held-out accuracy, compared with no thoughts and with `LatentThoughtLm` (Coconut). |
| 3.9 | **SEAL (self-adapting LM)** [cat] | The model writes its own fine-tuning "self-edits"; RL rewards edits that improve downstream accuracy after a LoRA update. | `experimental.llm.SelfEditTrainer` | A new `experiments.reasoning.SelfEdits` on a toy fact-injection task: recall after an update, compared with fine-tuning on the raw passage. |
| 3.10 | **TTRL (test-time RL)** | GRPO on unlabelled test prompts, using the majority-vote answer as a pseudo-reward. | `experimental.llm` (a reward source for `GrpoTrainer`) | TestTimeScaling: accuracy compared with self-consistency and greedy decoding. |
| 3.11 | **Constitutional AI / RLAIF** | The model critiques and revises its own outputs against written rules, and the revisions train a preference model. | `experimental.llm.ConstitutionalLoop` | SelfImprovement: rule-violation rate and task accuracy, compared with `SelfRewardingTrainer`. |
| 3.12 | **On-policy distillation (GKD)** | The student samples, and the teacher scores those samples token by token with a generalised JSD. | `experimental.llm.OnPolicyDistiller` | DistillationShootout: student accuracy and perplexity, compared with offline logit KD (`DistillationLoss`). |
| 3.13 | **MiniLLM (reverse-KL distillation)** | The student minimises reverse KL, which is mode-seeking, via a policy-gradient estimate. | `experimental.llm` (loss in the distiller) | DistillationShootout: generation quality and exposure-bias error, compared with forward-KL KD. |
| 3.14 | **Sequence-level knowledge distillation** | Train the student on the teacher's beam or greedy outputs, not on its soft labels. | `experimental.llm` (data builder) | DistillationShootout: compared with token-level KD at an equal number of teacher calls. |
| 3.15 | **Feature/hidden-state distillation (FitNets / TinyBERT)** | Match intermediate hidden states and attention maps through learned projections. | `experimental.nn.FeatureDistillationLoss` | DistillationShootout and VisionShootout: student accuracy, compared with logit-only KD. |
| 3.16 | **SLERP / Fisher / RegMean merging** | Spherical interpolation, Fisher-weighted averaging and closed-form regression merging of fine-tuned models. | `experimental.train.ModelMerging` (new methods) | The merging test task (two skills): accuracy on both skills, compared with uniform soup, TIES and DARE. |
| 3.17 | **rsLoRA and LoftQ** | Rank-stabilised LoRA scaling (α/√r), and quantization-aware LoRA initialisation for quantized bases. | `experimental.nn.LoraAdapterDense` (new factories) | PeftShootout: loss compared with rank (rsLoRA), and accuracy on a 4-bit base (LoftQ) compared with QLoRA. |

## WP4. Inference, serving and quantization (13)

Target: mostly core-worthy for `brainiac-llm` and `brainiac-tensor`. Each item is prototyped as an `experimental.llm` / `experimental.nn` block first and measured in `experiments.language.DecodingShowdown`, `experiments.language.CompressionComparison`, `experiments.techniques.QuantizationShootout` and a new `experiments.language.ServingThroughput`. Baseline: the core `Llama` + `Generator` path.

| # | Item | One line | Lives in | Experiment vs baseline |
|---|---|---|---|---|
| 4.1 | **Prefix caching (radix tree)** | Reuse the KV of shared prompt prefixes across requests. | core-worthy (`brainiac-llm`); prototype `experimental.llm.PrefixCache` | ServingThroughput with a shared system prompt: time to first token, compared with no reuse. |
| 4.2 | **Paged KV cache** | A fixed-size block pool with per-sequence block tables in place of contiguous growable buffers. | core-worthy (`brainiac-llm`); prototype `experimental.llm.PagedKvCache` | ServingThroughput: peak memory and maximum concurrent sequences, compared with the current growable cache. |
| 4.3 | **Continuous batching** | Iteration-level scheduling: sequences join and leave the decode batch at every step. | core-worthy (`brainiac-llm`) | ServingThroughput: tokens/s and p95 latency under a Poisson arrival trace, compared with static batching. |
| 4.4 | **Chunked prefill** | Split long prompts into chunks interleaved with decode steps. | core-worthy (`brainiac-llm`) | ServingThroughput: decode-latency jitter with long prompts arriving, compared with monolithic prefill. |
| 4.5 | **KIVI 2-bit KV cache** [cat] | Per-channel key and per-token value 2-bit quantization, with a full-precision residual window. | `experimental.nn.QuantizedKvCache`; core-worthy | LongContextTraining and EvictionComparison: output error and KV bytes, compared with fp32 KV and with H2O at equal memory. |
| 4.6 | **EAGLE / tree speculative decoding** | A feature-level draft head proposes a token tree, which is verified in one pass with tree attention masks. | `experimental.llm.EagleDraft`; core-worthy (tree verify in `Generator`) | DecodingShowdown: accepted tokens per step and speed-up, compared with core speculative decoding, `MedusaHeads` and Jacobi. |
| 4.7 | **Hadamard-rotation quantization (QuaRot / SpinQuant)** | Rotate weights and activations by a random Hadamard matrix to remove outliers before 4-bit quantization. | `experimental.nn.RotatedQuant`; core-worthy | QuantizationShootout and CompressionComparison: W4A4 perplexity, compared with `SmoothQuant` and `Gptq`. |
| 4.8 | **HQQ (half-quadratic quantization)** | Calibration-free weight quantization that optimises zero-point and scale with a sparse-error proxy. | `experimental.nn.Hqq` | CompressionComparison: 3- and 4-bit perplexity and quantize time, compared with GPTQ and AWQ. |
| 4.9 | **MXFP4 / NVFP4 microscaling formats** | 4-bit floats with a shared 8-bit scale per 16 or 32 elements. | `experimental.nn` (next to `Fp8Format`); core-worthy for `brainiac-tensor` storage | QuantizationShootout: accuracy compared with int4 group quantization and FP8. |
| 4.10 | **Top-nσ and XTC samplers** | Top-nσ keeps logits within n standard deviations of the max; XTC sometimes excludes the top choices for creativity. | `experimental.llm.LogitsProcessors`; core-worthy for `Sampling` | DecodingShowdown: accuracy at high temperature and distinct-n, compared with min-p and top-p. |
| 4.11 | **Entropy-adaptive sampling (entropix-style)** | Switch between greedy, sampling, branching or "think" according to logit entropy and varentropy. | `experimental.llm.EntropyAdaptiveSampler` | DecodingShowdown and TestTimeScaling: accuracy at equal tokens, compared with greedy and fixed temperature. |
| 4.12 | **Budget forcing (s1)** | Control thinking length at decode time: suppress end-of-thinking to extend, or force it to cut short. | `experimental.llm` (a `LogitsProcessor` plus driver) | TestTimeScaling: accuracy compared with thinking-token budget, against unconstrained generation. |
| 4.13 | **Activation-aware KV offload to host** | Keep only recent and hot KV blocks on the device and page the rest to host RAM. | `experimental.llm` (on top of 4.2); core-worthy for the Vulkan backend | ServingThroughput on Vulkan: maximum context and tokens/s, compared with device-only KV. |

## WP5. Agents, reasoning search and open-ended self-improvement (10)

Target: `experimental.llm`, `experimental.evolve` and `experimental.rl`, measured in `experiments.agents`, `experiments.reasoning` and `experiments.evolution`. All are toy-scale and use the existing tiny LMs and DSLs (`AbsoluteZero`, `ToolUse`).

| # | Item | One line | Lives in | Experiment vs baseline |
|---|---|---|---|---|
| 5.1 | **Generative Agents memory stream** [cat] | Memory retrieval scored by recency × importance × relevance, plus periodic reflection summaries. | `experimental.llm.MemoryStream` | `experiments.agents` multi-day toy-world QA: recall of past events, compared with a plain sliding-window context and `VirtualContextPager`. |
| 5.2 | **Voyager skill library** [cat] | An auto-curriculum plus a growing library of verified code skills retrieved by embedding. | `experimental.llm.SkillLibrary` | ToolUse/AbsoluteZero DSL: tasks solved over time, compared with no library and with `ReflexionLoop`. |
| 5.3 | **Graph of Thoughts** [cat] | Thought nodes that can be merged, refined and looped as a dataflow graph. | `experimental.llm.TestTimeCompute` (new strategy) | TestTimeScaling on sorting/set tasks: accuracy at equal LM calls, compared with ToT beam search. |
| 5.4 | **MCTS over reasoning steps (rStar / LLM-MCTS)** | UCT search over partial solutions with a PRM or rollout value. | `experimental.llm` (reuses `experimental.rl.Mcts` and `ProcessRewardModel`) | TestTimeScaling: accuracy at equal LM calls, compared with ToT DFS and best-of-N. |
| 5.5 | **AlphaEvolve-style program evolution** [cat] | An LM mutates program text and an evaluator scores it, with a MAP-Elites archive. Target: Brainiac's own matmul and softmax kernels. | `experimental.evolve.ProgramEvolution` | A new `experiments.evolution.EvolvedKernels`: best evaluated kernel speed or correctness, compared with random mutation and `AutoMlZero`. |
| 5.6 | **Darwin Gödel Machine (toy)** [cat] | Agents that edit their own scaffold code, kept in an open-ended archive and validated on a benchmark. | `experimental.evolve.SelfModifyingArchive` | DSL coding benchmark: score over generations, compared with greedy hill-climbing (no archive). |
| 5.7 | **AI Scientist pipeline (toy)** [cat] | An automated idea → config → run → write-up loop over Brainiac experiments, with an LM reviewer. | `experiments.agents.AutoResearcher` | The share of generated configs that beat the baseline experiment, compared with random config search. |
| 5.8 | **DreamCoder library learning** [cat] | Wake-sleep program synthesis that compresses solved programs into new DSL primitives. | `experimental.evolve.LibraryLearner` | DSL list tasks: solve rate and search time per round, compared with a fixed DSL. |
| 5.9 | **AlphaGeometry-style neuro-symbolic prover** [cat] | A neural proposer of auxiliary constructions plus a symbolic deduction engine that checks them. | `experiments.reasoning.NeuroSymbolicProver` | Toy algebra or geometry goals: proof rate, compared with the symbolic engine alone. |
| 5.10 | **Learned-memory agent loop (MemGPT + Reflexion fusion)** | A single agent loop where reflections are written into a paged long-term store and retrieved by relevance. | `experimental.llm` (composes `VirtualContextPager`, `ReflexionLoop`, 5.1) | ReflectiveSolver: solve rate over long task sequences, compared with Reflexion alone. |

## WP6. Generative, multimodal, physical and exotic (15)

Target: `experimental.nn`, `experimental.nn.diffusion` and `experimental.rl`, measured in `experiments.generative`, `experiments.exotic` and a new `experiments.multimodal`.

| # | Item | One line | Lives in | Experiment vs baseline |
|---|---|---|---|---|
| 6.1 | **Genie (latent-action world model)** [cat] | Learn discrete latent actions plus a dynamics model from action-free frames, giving a playable model. | `experimental.rl.LatentActionWorldModel` | DreamingAgent grid frames: next-frame error and action recovery, compared with an action-conditioned `MlpWorldModel`. |
| 6.2 | **MusicGen-style codec LM** [cat] | A transformer over the `ResidualVectorQuantizer` codec tokens, using the delay interleaving pattern. | `experiments.generative.CodecLanguageModel` | ToyAudioCodec: token NLL and reconstruction quality, compared with a flattened-RVQ LM. |
| 6.3 | **Chameleon early-fusion multimodal** [cat] | VQ image tokens and text tokens in one sequence for one transformer. | `experiments.multimodal.EarlyFusion` | Digit image ↔ label captioning: accuracy both ways, compared with separate models. |
| 6.4 | **Large Concept Model** [cat] | Autoregression (MSE or diffusion) over sentence embeddings in place of tokens. | `experimental.nn.ConceptLm` | ChainTask summaries: next-sentence embedding error and decoded accuracy, compared with a token LM at equal FLOPs. |
| 6.5 | **SpikeGPT (spiking LM)** [cat, partial] | An RWKV-style language model with LIF spiking activations and surrogate gradients. | `experimental.nn` (combines `LifLayer` and `Rwkv7TimeMix`) | CountingTasks: perplexity and spike rate, compared with the non-spiking RWKV. |
| 6.6 | **Differentiable physics (DiffTaichi-style)** [cat] | Autograd through a mass-spring or rigid-body step to optimise controllers directly. | `experimental.rl.DifferentiableSim` | Reach or throw task: controller quality at equal samples, compared with PPO and CEM. |
| 6.7 | **3D Gaussian Splatting** [cat, partial] | 3-D anisotropic Gaussians with projection, depth sorting and alpha blending. | `experimental.nn.GaussianSplats3d` (extends the 2-D version) | NeuralFields on a synthetic multi-view scene: PSNR and train time, compared with a NeRF-style `FourierFeatures` MLP. |
| 6.8 | **Denoising thermodynamic models (DTM)** [cat, partial] | Chained p-bit Gibbs samplers trained as diffusion-like denoising stages. | `experimental.nn` (on top of `PBitIsingMachine`) | ProbabilisticBits and ToyPoints: sample quality (MMD), compared with a single Ising machine and `TinyDdpm`. |
| 6.9 | **MaskGIT** | Parallel iterative decoding of VQ image tokens with a cosine masking schedule. | `experiments.generative.MaskGit` | ImageAutoencoders VQ tokens: FID-proxy and decode steps, compared with an autoregressive token model. |
| 6.10 | **VAR (next-scale prediction)** | Autoregression over coarse-to-fine multi-scale VQ token maps. | `experimental.nn.diffusion` (or `generative`) | ToyImages: sample quality and steps, compared with MaskGIT and DDPM. |
| 6.11 | **MeanFlow / shortcut models** | One- or few-step generation by learning average velocities or step-size-conditioned flows. | `experimental.nn.diffusion.MeanFlow` | RectifiedFlowExperiment: 1-step quality, compared with reflow and `ConsistencyModel`. |
| 6.12 | **v-prediction and min-SNR weighting** | Alternative diffusion targets and loss weighting for faster convergence. | `experimental.nn.diffusion` (options on denoisers) | TinyDdpm: sample quality at fixed steps, compared with ε-prediction and uniform weighting. |
| 6.13 | **CLIP / SigLIP contrastive image-text** | A two-tower image and text encoder trained with InfoNCE or a pairwise sigmoid loss. | `experimental.nn.ContrastivePairLoss` (extends `SelfSupervisedLosses`) | `experiments.multimodal`: zero-shot digit classification from label text, compared with a linear probe on SimCLR features. |
| 6.14 | **ELECTRA (replaced-token detection)** | A small generator corrupts tokens and the discriminator is pre-trained to detect them. | `experimental.llm` (pre-training objective) | CountingTasks probe accuracy at an equal compute budget, compared with masked-LM pre-training. |
| 6.15 | **Fill-in-the-middle (FIM) training** | The PSM/SPM document transform, so a causal LM learns infilling. | `experimental.data` (transform next to `SequencePacker`) | Infilling accuracy and left-to-right loss, compared with a plain causal LM (the "FIM is free" check). |

---

## Notes for the implementing agents

- WP1 and WP2 are independent of each other and of everything else.
- In WP4, items 4.1–4.4 and 4.13 share one KV/serving refactor; give them to a single agent. 4.5 depends on 4.2 only if it goes into core.
- WP3 needs a small `DistillationShootout` harness first (3.12–3.15).
- WP5 items 5.5–5.9 share a toy DSL and evaluator; reuse the ones in `experiments.reasoning.AbsoluteZero`.
- WP6 6.2, 6.3, 6.9 and 6.10 all need VQ tokenizers that already exist (`VectorQuantizer`, `ResidualVectorQuantizer`).
