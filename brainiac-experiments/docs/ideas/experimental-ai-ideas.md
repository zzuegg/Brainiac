# Brainiac: catalogue of experimental AI ideas

*Compiled 2026-11-17 from three parallel web-research sweeps. About 170 ideas, 20 themes. Every link appeared in search results during the research; where an agent could only find a project or journal page, that is linked instead of arXiv.*

## How to read this

Each row says what the idea is, why it might be interesting for Brainiac (a from-scratch Java 27 library with CPU/OpenCL/Vulkan backends, dense/CNN/transformer layers, RL, quantization and neuroevolution), a key link, and a rough difficulty:

| Rating | Meaning |
|---|---|
| **L** | Low. Days. Reuses what Brainiac already has. |
| **M** | Medium. Needs one new primitive or a non-trivial training loop. |
| **H** | High. Several new primitives, custom kernels or a fragile training recipe. |
| **R** | Research. Ideas still forming, needs big compute, or hardware Brainiac can't reach. Expect to experiment rather than implement. |

---

## Start here

### Shared primitives that unlock many ideas at once
Building one of these makes a whole cluster of entries cheap.

| Primitive | Unlocks |
|---|---|
| **FFT kernel** (complex tensors) | Hyena, Fourier Neural Operator, optical nets, S4 |
| **Parallel scan / chunked linear attention** | Mamba-2, Gated DeltaNet, RWKV-7, xLSTM, Kimi Linear, Infini-attention |
| **ODE solver library** (Euler, RK4, Dopri5) | Neural ODE/CDE, flow matching, CfC/liquid nets, consistency models |
| **Gradients inside the forward pass** (nested autodiff) | TTT layers, Titans, Nested Learning, Energy-Based Transformers, MAML |
| **Sparse gather/scatter + top-k** | MoE, Mixture-of-Depths/Recursions, PEER, memory layers, Engram, GNNs, KV eviction |
| **Forward hooks on any layer** | Steering vectors, sparse autoencoders, ROME, activation patching, Coconut |
| **Vector index (kNN / LSH)** | kNN-LM, RETRO, Neural Episodic Control, novelty search, MemGPT-style memory |
| **Seed-reproducible perturbations** | ES at scale, MeZO, neuroevolution of large models |

### Quick wins (Low difficulty, high novelty)
Gated Attention, Differential Transformer, Forgetting Transformer, nGPT, Multi-Token Prediction, Flow Matching, YaRN, attention sinks, H2O, steering vectors, sparse autoencoders, Neural Cellular Automata, Lenia, Next-Gen Reservoir Computing, hyperdimensional computing, Direct Feedback Alignment, Mono-Forward, Schedule-Free, Lion, Muon, MeZO, MAP-Elites, novelty search, Random Network Distillation, DPO/GRPO, Reflexion, self-consistency.

### Showcase demos that fit a laptop
- **Tiny Recursive Model** (about 7M parameters) on Sudoku or ARC puzzles.
- **BitNet ternary kernel** using the Java Vector API.
- **Neural Cellular Automata** that grow and regenerate on the Vulkan backend.
- **Differentiable logic gate network** compiled to pure bit operations.
- **Grokking** on modular arithmetic.

### The most speculative, "future" bets
Nested Learning / HOPE, Darwin Gödel Machine, self-referential weight matrices, Thousand Brains (Monty), Continuous Thought Machines, Energy-Based Transformers, Absolute Zero self-play, thermodynamic computing, active inference, AutoML-Zero, learned optimizers.

---

## 1. Sequence models beyond attention

| Idea | What / why for Brainiac | Link | Diff. |
|---|---|---|---|
| **S4** (structured state spaces) | A continuous linear state-space system computed as a long convolution (training) or a recurrence (inference). It is the root of the SSM family and shows one layer running two ways. | [2111.00396](https://arxiv.org/abs/2111.00396) | H |
| **Mamba** (selective SSM) | SSM parameters depend on the input, so the model chooses what to remember. Linear time, constant memory per token, which suits CPU inference. | [2312.00752](https://arxiv.org/abs/2312.00752) | M-H |
| **Mamba-2 / SSD** | Shows SSMs are a structured form of linear attention, computable as chunked matmuls. Uses the GEMMs Brainiac already runs fast. | [2405.21060](https://arxiv.org/abs/2405.21060) | M |
| **Mamba-3** (2026) | Adds complex/rotary state updates and a MIMO formulation for better hardware use at inference. Shows where SSMs are heading. | [2603.15569](https://arxiv.org/abs/2603.15569) | H |
| **RWKV-7 "Goose"** | An RNN that trains in parallel like a transformer, with a generalised delta-rule state. Open weights could be imported to check correctness. | [2503.14456](https://arxiv.org/abs/2503.14456) | M |
| **Gated DeltaNet** | Linear attention combining fast forgetting (gating) with precise overwriting (delta rule). Now a standard block in production hybrids. | [2412.06464](https://arxiv.org/abs/2412.06464) | M-H |
| **Kimi Linear** | A hybrid stack of delta-rule linear attention and periodic full attention (about 3:1), with about 75% less KV cache. A recipe for mixed stacks. | [2510.26692](https://arxiv.org/abs/2510.26692) | H |
| **xLSTM** (sLSTM + mLSTM) | Modernised LSTMs with exponential gating and a parallel matrix memory. A natural upgrade to any existing LSTM/GRU code. | [2405.04517](https://arxiv.org/abs/2405.04517) | M |
| **Hyena** | Attention replaced by FFT long convolutions with filters generated by a small network. Reuses CNN machinery at 100k+ lengths. | [2302.10866](https://arxiv.org/abs/2302.10866) | M |
| **Test-Time Training (TTT) layers** | The RNN hidden state is itself a small model updated by a gradient step per token. "A layer that trains during inference". | [2407.04620](https://arxiv.org/abs/2407.04620) | H |

## 2. Attention and transformer refinements

| Idea | What / why for Brainiac | Link | Diff. |
|---|---|---|---|
| **Differential Transformer** | Attention as the difference of two softmax maps, cancelling attention noise. Reported to reduce hallucination. | [2410.05258](https://arxiv.org/abs/2410.05258) | L |
| **Gated Attention** | A per-head sigmoid gate on the attention output removes attention sinks and stabilises training. Very high payoff per line of code. | [2505.06708](https://arxiv.org/abs/2505.06708) | L |
| **Forgetting Transformer (FoX)** | A data-dependent forget gate on attention logits. Needs no positional embedding; a simple alternative to RoPE. | [2503.02130](https://arxiv.org/abs/2503.02130) | L-M |
| **nGPT** (normalized transformer) | All vectors live on the unit hypersphere; residuals become interpolation. Reports 4-20x fewer steps, and bounded values help quantization. | [2410.01131](https://arxiv.org/abs/2410.01131) | L-M |
| **Native Sparse Attention** | Trainable sparse attention from compressed tokens, selected blocks and a sliding window, designed to be fast on real hardware. | [2502.11089](https://arxiv.org/abs/2502.11089) | H |
| **mHC** (manifold-constrained hyper-connections) | Several learnable residual streams mixed by doubly-stochastic matrices. A rare new idea about the residual stream itself. | [2512.24880](https://arxiv.org/abs/2512.24880) | M |
| **Multi-Token Prediction** | Extra heads predict the next n tokens. Better sample efficiency plus self-speculative decoding. | [2404.19737](https://arxiv.org/abs/2404.19737) | L-M |
| **Free Transformer** | The decoder conditions on unsupervised random latents (VAE-style), so it "decides" global properties before writing. Latent planning. | [2510.17558](https://arxiv.org/abs/2510.17558) | M |

## 3. Conditional compute and adaptive depth

| Idea | What / why for Brainiac | Link | Diff. |
|---|---|---|---|
| **Fine-grained MoE + shared experts** (DeepSeekMoE) | Many small experts plus always-on shared ones. Small experts suit cache-sized CPU matmuls. | [2401.06066](https://arxiv.org/abs/2401.06066) | M |
| **Soft MoE** | Experts process weighted averages of all tokens. Fully dense and differentiable, no load-balancing headaches. | [2308.00951](https://arxiv.org/abs/2308.00951) | L-M |
| **PEER** (a million experts) | Product-key retrieval picks from over a million single-neuron experts. Retrieval becomes the compute primitive. | [2407.04153](https://arxiv.org/abs/2407.04153) | H |
| **Mixture-of-Depths** | A router lets each token skip a block under a fixed budget. Dynamic compute with static shapes. | [2404.02258](https://arxiv.org/abs/2404.02258) | M |
| **Universal Transformer** | One shared block applied repeatedly with adaptive halting. The classic looped transformer. | [1807.03819](https://arxiv.org/abs/1807.03819) | L-M |
| **Mixture-of-Recursions** | Shared recursive layers with a router giving each token its own depth. Small models that think longer on hard tokens. | [2507.10524](https://arxiv.org/abs/2507.10524) | M-H |
| **PonderNet** | Learned per-step halting probability. A general "think longer" wrapper for any recurrent block. | [2107.05407](https://arxiv.org/abs/2107.05407v2) | L-M |

## 4. Memory as a layer

| Idea | What / why for Brainiac | Link | Diff. |
|---|---|---|---|
| **Neural Turing Machine** | Controller plus external memory with differentiable read/write heads. The classic "memory is a layer". | [1410.5401](https://arxiv.org/pdf/1410.5401) | M |
| **Differentiable Neural Computer** | NTM with dynamic allocation and temporal links for in-order recall. A stress test for custom autograd ops. | [Nature 2016 PDF](https://psych205.stanford.edu/papers/GravesWayne16DNC.pdf) | H |
| **Modern Hopfield layers** | Exponential-capacity associative memory whose update equals attention. Reuses attention kernels. | [2008.02217](https://arxiv.org/abs/2008.02217) | L |
| **Memory Layers at Scale** | Huge trainable key-value tables replace some FFNs: capacity without FLOPs. Trades compute for RAM. | [2412.09764](https://arxiv.org/abs/2412.09764) | M |
| **Engram** (2026) | Hashed N-gram lookup memory as a separate sparsity axis next to MoE. Tables can live in host RAM while compute runs on GPU. | [2601.07372](https://arxiv.org/html/2601.07372v1) | M |
| **Neural Episodic Control** | An RL agent with a kNN dictionary from embeddings to Q-values. Very sample-efficient; plugs into the RL stack. | [1703.01988](https://arxiv.org/abs/1703.01988v1) | M |
| **Titans** | A neural long-term memory updated at inference by a "surprise" gradient with momentum and forgetting, next to attention. Over 2M tokens of context. | [2501.00663](https://arxiv.org/abs/2501.00663) | H |
| **Nested Learning / HOPE** | Views a model and its optimiser as nested optimisation problems at different frequencies: a continuum of memories. Could organise a whole memory API. | [2512.24695](https://arxiv.org/abs/2512.24695) | R |
| **Self-referential weight matrix** | A weight matrix that learns to rewrite itself at runtime. The purest self-modifying net; evolve it with neuroevolution. | [2202.05780](https://arxiv.org/pdf/2202.05780.pdf) | M |
| **Differentiable plasticity** | Each connection has a learned Hebbian trace, so the net keeps learning during its lifetime. | [1804.02464](https://arxiv.org/abs/1804.02464) | M |

## 5. Context management, retrieval and long context

| Idea | What / why for Brainiac | Link | Diff. |
|---|---|---|---|
| **kNN-LM** | Mix the LM's output with a kNN lookup over a datastore. Add knowledge by adding rows, not retraining. | [1911.00172](https://arxiv.org/pdf/1911.00172) | L-M |
| **RETRO** | Chunked cross-attention over retrieved neighbours; matches much larger models. Small model plus big store. | [2112.04426](https://www.arxiv.org/pdf/2112.04426v3) | M-H |
| **MemGPT** | Context window as RAM, external storage as disk, paged by function calls. A context-manager utility around any model. | [2310.08560](https://arxiv.org/pdf/2310.08560) | L-M |
| **Agentic Context Engineering (ACE)** | Context as an evolving playbook curated by generator, reflector and curator roles. Learning with no weight updates. | [2510.04618](https://arxiv.org/abs/2510.04618) | L |
| **Sleep-time compute** | The model pre-thinks about context while idle, before queries arrive. A natural Java background-thread feature. | [2504.13171](https://arxiv.org/abs/2504.13171) | L-M |
| **YaRN / RoPE scaling** | Interpolate RoPE frequencies per band for 4-16x longer context after short fine-tuning. | [2309.00071](https://arxiv.org/abs/2309.00071) | L |
| **Attention sinks / StreamingLLM** | Keep the first few tokens plus a sliding window for unbounded streaming generation. | [2309.17453](https://arxiv.org/abs/2309.17453v4) | L |
| **H2O** | Evict all but the "heavy-hitter" tokens from the KV cache; 5-20x less KV memory. | [2306.14048](https://arxiv.org/abs/2306.14048v2) | L-M |
| **SnapKV** | One-shot KV compression voted by an observation window at the end of the prompt. | [2404.14469](https://arxiv.org/abs/2404.14469) | L-M |
| **KIVI** | 2-bit KV-cache quantization with no tuning. Reuses the quantization module on a new target. | [2402.02750](https://arxiv.org/abs/2402.02750) | M |
| **Landmark attention** | A landmark token per block lets attention fetch whole blocks: retrieval inside attention. | [2305.16300](https://arxiv.org/abs/2305.16300v2) | M |
| **Infini-attention** | A compressive linear memory in each attention layer for unbounded context at constant memory. | [2404.07143](https://arxiv.org/abs/2404.07143) | M |
| **Ring attention** | Sequence split across devices in a ring. Could ring across several OpenCL/Vulkan devices, or CPU plus GPU. | [2310.01889](https://arxiv.org/abs/2310.01889v3) | H |
| **Recurrent Memory Transformer** | Memory tokens passed between segments make a transformer a segment-level RNN; scaled past 1M tokens. | [2207.06881](https://arxiv.org/abs/2207.06881v2) | L-M |

## 6. Reasoning and test-time compute

| Idea | What / why for Brainiac | Link | Diff. |
|---|---|---|---|
| **Self-consistency** | Sample many reasoning chains, majority-vote the answer. The simplest test-time scaling knob. | [2203.11171](https://arxiv.org/abs/2203.11171) | L |
| **Tree of Thoughts** | Reasoning as a search tree with self-evaluation; reuses MCTS/search code from RL. | [2305.10601](https://arxiv.org/pdf/2305.10601) | L-M |
| **Graph of Thoughts** | Thoughts merged, refined and fed back in a graph: reasoning as a dataflow graph. | [2308.09687](https://arxiv.org/abs/2308.09687v1) | L-M |
| **Compute-optimal test-time scaling** | Spending inference compute adaptively can beat a 14x larger model. Small models plus smart loops. | [2408.03314](https://arxiv.org/abs/2408.03314) | M |
| **Process reward models** | Reward every reasoning step, not just the answer. A reusable step-level critic. | [2305.20050](https://arxiv.org/abs/2305.20050) | M |
| **RL for reasoning (DeepSeek-R1)** | Pure RL with verifiable rewards elicits long chains of thought and self-checking. | [overview](https://alphaxiv.org/overview/2501.12948v1) | M |
| **Quiet-STaR** | The model learns hidden "thoughts" at every token from plain text, via REINFORCE. | [2403.09629](https://arxiv.org/abs/2403.09629) | H |
| **Coconut** (continuous thought) | Feed the last hidden state back as the next input instead of a token: reasoning in latent space. A minimal feedback loop. | [2412.06769](https://arxiv.org/abs/2412.06769) | M |
| **Hierarchical Reasoning Model** | Slow planner plus fast worker recurrent modules; about 27M parameters solve Sudoku, mazes and ARC without chain-of-thought. | [2506.21734](https://arxiv.org/abs/2506.21734) | M |
| **Tiny Recursive Model** | One 2-layer, about 7M-parameter net recursively refines latent and answer; beats big LLMs on ARC-AGI. The most laptop-friendly frontier result. | [2510.04871](https://arxiv.org/abs/2510.04871) | L-M |
| **Continuous Thought Machines** | Neurons with private activation histories; synchronisation over internal ticks is the representation. A new neuron abstraction. | [2505.05522](https://arxiv.org/abs/2505.05522) | H |
| **Energy-Based Transformers** | Learn an energy over (context, prediction) and "think" by gradient descent on it. Inference as optimisation, with built-in verification. | [2507.02092](https://arxiv.org/abs/2507.02092) | H |
| **Speculative decoding** | A small draft model proposes, the big one verifies in parallel; 2-3x faster with identical outputs. Pairs with a quantized draft. | [2211.17192](https://arxiv.org/pdf/2211.17192) | L-M |

## 7. Self-improvement and feedback loops

| Idea | What / why for Brainiac | Link | Diff. |
|---|---|---|---|
| **Reflexion** | After failing, the agent writes a self-reflection into episodic memory for the next try. RL through text. | [2303.11366](https://arxiv.org/abs/2303.11366) | L |
| **Self-Rewarding LMs** | The model judges its own outputs, then iterative DPO improves both answering and judging. | [2401.10020](https://arxiv.org/abs/2401.10020) | M |
| **SPIN** (self-play fine-tuning) | Learn to tell human data from your own previous generations. Improvement without new labels. | [2401.01335](https://arxiv.org/abs/2401.01335v1) | L-M |
| **SEAL** (self-adapting LMs) | The model writes its own fine-tuning data; RL rewards edits that help after training. A continual self-update loop. | [2506.10943](https://arxiv.org/abs/2506.10943) | H |
| **Absolute Zero** | One model proposes and solves its own tasks with a code executor as verifier. Self-play with zero human data; try it with a toy DSL. | [2505.03335](https://arxiv.org/abs/2505.03335) | R |
| **Darwin Gödel Machine** | A coding agent rewrites its own code and keeps an open-ended archive of variants, validated on benchmarks. Brainiac's own code could be the genome. | [2505.22954](https://arxiv.org/abs/2505.22954) | R |
| **AlphaEvolve** | LLMs mutate programs, evaluators score them. Target: evolving Brainiac's own OpenCL/Vulkan kernels. | [2506.13131](https://arxiv.org/abs/2506.13131) | M-H |
| **The AI Scientist** | Automated idea, code, experiment, paper and review pipeline. Brainiac as a fast, deterministic experiment substrate. | [2408.06292](https://arxiv.org/pdf/2408.06292) | H |
| **AutoML-Zero** | Evolve whole learning algorithms from basic maths ops; it rediscovered backprop-like rules. | [2003.03384](https://arxiv.org/abs/2003.03384) | H |
| **Lion / symbolic optimiser search** | Program search over update rules found Lion (sign momentum, less memory than Adam). Lion is quick to add; the search is research. | [2302.06675](https://arxiv.org/abs/2302.06675) | L / R |
| **Learned optimizers (VeLO)** | A small network, meta-trained with ES, outputs the parameter updates. "The optimiser is a network". | [2211.09760](https://arxiv.org/abs/2211.09760) | R |

## 8. Agents, tools and multi-agent societies

| Idea | What / why for Brainiac | Link | Diff. |
|---|---|---|---|
| **Toolformer** | The model teaches itself where to call tools, keeping only calls that lower perplexity. | [2302.04761](https://arxiv.org/pdf/2302.04761) | M |
| **Voyager** | An agent with an auto-curriculum and a growing library of verified code skills: procedural memory. | [2305.16291](https://arxiv.org/abs/2305.16291) | M |
| **Generative Agents** | Memory stream with recency × importance × relevance retrieval, plus reflection and planning. A well-specified agent memory. | [2304.03442](https://arxiv.org/pdf/2304.03442) | L-M |
| **Multi-agent debate** | Several instances critique each other over rounds; better factuality with small models. | [2305.14325](https://arxiv.org/abs/2305.14325) | L |
| **Emergent communication** | Cooperating agents invent their own language. A scientific toy using the RL module. | [2407.03302](https://arxiv.org/pdf/2407.03302) | M |

## 9. World models, RL and intrinsic motivation

| Idea | What / why for Brainiac | Link | Diff. |
|---|---|---|---|
| **DreamerV3** | A latent world model; the actor-critic learns purely in imagination. One config for 150+ tasks. Flagship model-based RL. | [2301.04104](https://arxiv.org/abs/2301.04104) | H |
| **Genie** | A playable world model learned from unlabelled video: environments learned, not coded. | [2402.15391](https://arxiv.org/pdf/2402.15391) | R |
| **I-JEPA** | Predict masked regions in representation space, not pixels. Cheaper than generation; LeCun's world-model direction. | [2301.08243](https://arxiv.org/abs/2301.08243) | M |
| **V-JEPA 2** | JEPA scaled to video, with an action-conditioned variant for zero-shot robot planning. | [2506.09985](https://arxiv.org/abs/2506.09985) | M-H |
| **GRPO** | Critic-free PPO variant with group-normalised advantages. Simpler than PPO, good for any verifiable-reward task. | [2402.03300](https://arxiv.org/abs/2402.03300) | L-M |
| **DPO** | Preference alignment as a classification loss; no reward model, no PPO. | [2305.18290](https://arxiv.org/abs/2305.18290v1) | L |
| **Random Network Distillation** | Curiosity bonus = error predicting a fixed random network. Two MLPs. | [1810.12894](https://arxiv.org/abs/1810.12894) | L |
| **Prioritized Level Replay / UED** | Replay and generate levels where learning potential is highest. A gradient-based relative of POET. | [2110.02439](https://arxiv.org/abs/2110.02439) | M |
| **Streaming deep RL** | Learn one sample at a time, no replay buffer. Tiny agents perfect for the CPU backend. | [2410.14606](https://arxiv.org/abs/2410.14606) | L-M |
| **Homeostatic RL** | Reward = reducing deviation of internal variables from set points. Needs-driven artificial-life agents. | [eLife](https://elifesciences.org/articles/04811) | L |
| **Active inference** | Act to minimise expected free energy, unifying perception, action and curiosity. An alternative to reward. | [2207.06415](https://arxiv.org/abs/2207.06415v1) | R |

## 10. Learning without (full) backprop

| Idea | What / why for Brainiac | Link | Diff. |
|---|---|---|---|
| **Forward-Forward** | Two forward passes (positive and negative data), local "goodness" per layer. No backward graph. | [2212.13345](https://arxiv.org/abs/2212.13345) | L |
| **Mono-Forward** (2025) | Local cross-entropy per layer via a small projection; reportedly matches backprop with less memory. | [2501.09238](https://arxiv.org/html/2501.09238v1) | L |
| **NoProp** (2025) | Each block learns to denoise a noisy label embedding independently. Fully parallel block training. | [2503.24322](https://arxiv.org/abs/2503.24322) | M |
| **Predictive coding networks** | Layers predict the layer below; local error minimisation approximates backprop. A feedback loop in every layer. | [2407.04117](https://arxiv.org/abs/2407.04117) | M |
| **Equilibrium Propagation** | Settle free, then nudged; the state difference gives a local gradient. The main candidate for analog hardware. | [1602.05179](https://arxiv.org/abs/1602.05179) | M / R |
| **Direct Feedback Alignment** | Send output error to every layer through fixed random matrices. Removes the sequential backward pass. | [1609.01596](https://arxiv.org/abs/1609.01596) | L |
| **Difference Target Propagation** | Layer targets via learned inverses. Works for non-differentiable and quantized layers. | [1412.7525](https://arxiv.org/abs/1412.7525) | M |
| **Synthetic gradients** | Small models predict each layer's gradient, decoupling layers. Asynchronous training across devices. | [1608.05343](https://arxiv.org/abs/1608.05343v2) | M |
| **SoftHebb** | Soft winner-take-all Hebbian learning with no feedback at all, in deep CNNs. Plasticity parameters can be evolved. | [2209.11883](https://arxiv.org/abs/2209.11883v2) | L-M |
| **Forward gradients** | Unbiased gradient from a single forward-mode JVP. Dual numbers are easy to write from scratch. | [2202.08587](https://arxiv.org/abs/2202.08587v1) | M |
| **e-prop** | Online BPTT approximation with per-synapse eligibility traces. O(1) memory in time for RNN/SNN agents. | [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7367848/) | M-H |

## 11. Evolution, open-endedness and architecture search

| Idea | What / why for Brainiac | Link | Diff. |
|---|---|---|---|
| **ES at LLM scale** (2025) | Plain evolution strategies fine-tune billion-parameter LLMs, matching PPO/GRPO with less reward hacking. Workers exchange only seeds and fitnesses. | [2509.24372](https://arxiv.org/abs/2509.24372v2) | L-M |
| **MeZO** | Zeroth-order fine-tuning with seed-regenerated noise: training at inference memory cost. | [2305.17333](https://arxiv.org/abs/2305.17333v3) | L |
| **NEAT / HyperNEAT** | Evolve topology and weights; HyperNEAT evolves a CPPN that paints large weight patterns. | [UT Austin](https://nn.cs.utexas.edu/pub-view.php?PubID=114) | M |
| **Novelty Search** | Reward behavioural novelty, ignore the objective. Beats objective search on deceptive tasks. | [PDF](https://www.cs.swarthmore.edu/~meeden/cs81/f17/papers/lehman_ecj11.pdf) | L |
| **MAP-Elites** | A grid of the best solution per behaviour cell: a map of diverse high performers. | [1504.04909](https://arxiv.org/abs/1504.04909) | L |
| **POET** | Co-evolve environments and agents with transfer: an endless, self-growing curriculum. | [1901.01753](https://arxiv.org/abs/1901.01753v1) | H |
| **DARTS** | Differentiable architecture search by softmax over candidate ops. Gradient alternative to evolutionary NAS. | [1806.09055](https://arxiv.org/abs/1806.09055v1) | M-H |
| **HyperNetworks** | A small net generates a larger net's weights. Evolve the hypernet to compress the search space. | [1609.09106](https://arxiv.org/abs/1609.09106) | L-M |

## 12. Optimisers and training dynamics

| Idea | What / why for Brainiac | Link | Diff. |
|---|---|---|---|
| **Muon** | Momentum then Newton-Schulz orthogonalisation; about 2x AdamW efficiency at scale. A few matmuls. | [2502.16982](https://arxiv.org/abs/2502.16982v1) | L-M |
| **Modular duality** | Theory of per-layer dual norms; Muon falls out, and hyperparameters transfer across sizes. | [2410.21265](https://arxiv.org/abs/2410.21265v2) | R |
| **Shampoo / SOAP** | Kronecker-factored second-order preconditioning; SOAP runs Adam in its eigenbasis. | [2409.11321](https://arxiv.org/abs/2409.11321) | H |
| **Sophia** | Clipped optimiser with a cheap diagonal Hessian estimate. | [2305.14342](https://arxiv.org/abs/2305.14342v4) | M |
| **Schedule-Free** | No learning-rate schedule, no need to know total steps. Ideal for online/continual training. | [2405.15682](https://arxiv.org/abs/2405.15682) | L |
| **Prodigy / D-Adaptation** | Learning rate adapts itself from an online distance estimate. Good defaults without tuning. | [2306.06101](https://arxiv.org/abs/2306.06101v4) | L |
| **μP** | Width-aware init and per-layer LR so small-model hyperparameters transfer to big models. Tune on CPU, scale on GPU. | [2203.03466](https://arxiv.org/abs/2203.03466v2) | L-M |
| **SAM** | Step from an adversarially perturbed point to seek flat minima. Wraps any optimiser. | [2010.01412](https://arxiv.org/abs/2010.01412) | L |
| **GaLore** | Low-rank gradient projection: full-parameter training in far less memory. | [2403.03507](https://arxiv.org/abs/2403.03507v1) | M |
| **Grokking** | Sudden generalisation long after overfitting. A cheap, reproducible stress test of training dynamics. | [2201.02177](https://arxiv.org/abs/2201.02177) | L |
| **Early dropout** | Dropout only early helps big models underfit less. A scheduler change. | [2303.01500](https://arxiv.org/abs/2303.01500v2) | L |
| **Curriculum / self-paced learning** | Train easy to hard, possibly model-chosen. Cheap to add to data loaders. | [2101.10382](https://arxiv.org/abs/2101.10382v3) | L |
| **Lottery Ticket Hypothesis** | Sparse subnetworks rewound to their init train to full accuracy. Pairs with compression tools. | [1803.03635](https://arxiv.org/abs/1803.03635v4) | L |
| **RigL** | Train sparse from scratch, dropping and regrowing connections. Close to neuroevolution's mutations. | [1911.11134](https://arxiv.org/abs/1911.11134v3) | M |

## 13. Continual and lifelong learning

| Idea | What / why for Brainiac | Link | Diff. |
|---|---|---|---|
| **EWC** | Fisher-weighted penalty anchors weights important to earlier tasks. Classic anti-forgetting. | [1612.00796](https://arxiv.org/abs/1612.00796) | L |
| **Continual Backprop** (loss of plasticity) | Nets slowly lose the ability to learn; reinitialising low-utility units fixes it. Matters for long-running agents. | [2306.13812](https://arxiv.org/abs/2306.13812v1) | L-M |
| **Sleep-like replay** | An offline "sleep" phase with Hebbian updates reduces forgetting. Could become a `sleep()` call between tasks. | [Nat. Comms 2022](https://link.springer.com/content/pdf/10.1038/s41467-022-34938-7.pdf) | M |

## 14. Model composition, adapters and distillation

| Idea | What / why for Brainiac | Link | Diff. |
|---|---|---|---|
| **LoRA** | Low-rank update on frozen weights. Baseline cheap fine-tuning, combinable with quantized bases. | [2106.09685](https://arxiv.org/abs/2106.09685) | L |
| **DoRA** | Magnitude/direction split with LoRA on direction; consistently beats LoRA. | [2402.09353](https://arxiv.org/abs/2402.09353v4) | L |
| **Task arithmetic** | Add, subtract or combine "task vectors" (fine-tuned minus base). Weight algebra, no training. | [2212.04089](https://arxiv.org/abs/2212.04089) | L |
| **TIES-Merging** | Trim, sign-elect and merge fine-tuned models. Merging as a crossover operator for neuroevolution. | [2306.01708](https://arxiv.org/abs/2306.01708) | L |
| **Model soups / EMA** | Average fine-tuned weights for free accuracy. | [2203.05482](https://arxiv.org/abs/2203.05482) | L |
| **Knowledge distillation** | Train a small student on a teacher's soft outputs. Key for CPU deployment. | [1503.02531](https://arxiv.org/abs/1503.02531) | L |

## 15. Self-supervised learning

| Idea | What / why for Brainiac | Link | Diff. |
|---|---|---|---|
| **SimCLR** | Contrastive learning across augmentations (InfoNCE). Reference SSL for CNNs. | [2002.05709](https://huggingface.co/papers/2002.05709) | L-M |
| **DINOv2** | Self-distillation with an EMA teacher, no labels or negatives. Very strong visual features. | [2304.07193](https://arxiv.org/abs/2304.07193) | M |

## 16. Generative models (images, text, 3D, audio)

| Idea | What / why for Brainiac | Link | Diff. |
|---|---|---|---|
| **Flow matching** | Regress a velocity field from noise to data; generation is an ODE solve. The simplest modern generative loss. | [2210.02747](https://arxiv.org/abs/2210.02747) | L |
| **Diffusion Transformer (DiT)** | Transformer over latent image patches with adaLN conditioning. Reuses the transformer stack for images. | [2212.09748](https://arxiv.org/abs/2212.09748) | M |
| **Consistency models** | Map any noisy point straight to the clean sample: one-step generation, practical on CPU. | [2303.01469](https://arxiv.org/abs/2303.01469) | M |
| **LLaDA** (masked diffusion text) | An 8B LLM generating by iterative parallel unmasking; competitive with autoregressive models. | [2502.09992](https://arxiv.org/abs/2502.09992) | L-M |
| **Large Concept Models** | Autoregression over sentence embeddings ("concepts") instead of tokens. | [2412.08821](https://arxiv.org/abs/2412.08821) | H |
| **Byte Latent Transformer** | Tokenizer-free; bytes grouped into entropy-based patches. Removes the tokenizer dependency. | [2412.09871](https://arxiv.org/abs/2412.09871) | H |
| **Chameleon** (early-fusion any-to-any) | Images and text tokenised into one sequence for one transformer. A clean multimodal path. | [2405.09818](https://arxiv.org/abs/2405.09818v1) | M-H |
| **NeRF** | An MLP as a 3D scene, rendered by ray marching. Classic demo. | [2003.08934](https://arxiv.org/pdf/2003.08934) | M |
| **3D Gaussian Splatting** | Scenes as millions of Gaussians with a differentiable rasteriser. A showcase for the Vulkan backend. | [2308.04079](https://arxiv.org/pdf/2308.04079) | H |
| **SoundStream** (neural audio codec) | Conv encoder + residual VQ + decoder. RVQ links to quantization; tokens feed audio LMs. | [2107.03312](https://arxiv.org/abs/2107.03312) | M-H |
| **MusicGen** | A transformer over codec tokens with text/melody conditioning. "Codec tokens + LM" recipe. | [2306.05284](https://arxiv.org/abs/2306.05284) | M |

## 17. Continuous-time and function-learning networks

| Idea | What / why for Brainiac | Link | Diff. |
|---|---|---|---|
| **Neural ODEs** | Depth as continuous time solved by an ODE solver with adjoint backprop. Constant memory, adaptive compute. | [1806.07366](https://arxiv.org/abs/1806.07366) | M |
| **Neural CDEs** | Continuous-time RNN driven by an interpolated input path. Handles irregular, messy time series. | [2005.08926](https://arxiv.org/abs/2005.08926) | M-H |
| **Liquid / CfC networks** | Closed-form liquid time-constant neurons; tiny, interpretable controllers for RL and robotics. | [2106.13898](https://arxiv.org/abs/2106.13898) | L-M |
| **Fourier Neural Operator** | Learn mappings between function spaces in Fourier space: resolution-invariant PDE surrogates. | [2010.08895](https://arxiv.org/abs/2010.08895) | M |
| **KAN** (Kolmogorov-Arnold) | Learnable splines on edges instead of fixed activations. Interpretable, snaps to symbolic formulas. | [2404.19756](https://arxiv.org/abs/2404.19756) | M |

## 18. Structured, self-organising and brain-inspired

| Idea | What / why for Brainiac | Link | Diff. |
|---|---|---|---|
| **Capsule networks** | Vector capsules with routing-by-agreement capture part-whole hierarchies. | [1710.09829](https://arxiv.org/abs/1710.09829) | M |
| **EGNN** (equivariant GNN) | Rotation/translation-equivariant message passing. Opens graphs, molecules and physics. | [2102.09844](https://arxiv.org/abs/2102.09844) | M |
| **Neural Cellular Automata** | A tiny learned per-cell rule grows and regenerates patterns. Local and massively parallel: ideal for Vulkan. | [Google Research](https://research.google/pubs/growing-neural-cellular-automata/) | L |
| **Lenia** | Continuous cellular automata with hundreds of lifelike "species". Uses conv kernels; a neuroevolution playground. | [1812.05433](https://arxiv.org/pdf/1812.05433) | L |
| **FractalNet** | Ultra-deep self-similar nets from a recursive rule with drop-path. Easy to evolve. | [1605.07648](https://arxiv.org/abs/1605.07648) | L |
| **Hyperdimensional computing / VSA** | Compute with 10k-bit random vectors via bind/bundle/permute. XOR and popcount on Java `long`s; one-shot learning. | [2111.06077](https://arxiv.org/abs/2111.06077) | L |
| **Shared global workspace** | Specialist modules compete to write to a small shared workspace that broadcasts back. Consciousness-inspired bottleneck. | [2103.01197](https://arxiv.org/abs/2103.01197v2) | M |
| **Thousand Brains (Monty)** | Many cortical-column modules learn object models in reference frames and vote. A radically non-backprop paradigm. | [2412.18354](https://arxiv.org/pdf/2412.18354) | R |

## 19. Hardware-oriented and exotic computing

| Idea | What / why for Brainiac | Link | Diff. |
|---|---|---|---|
| **BitNet b1.58** | Ternary {-1,0,+1} LLMs matching FP16 at scale; matmuls become adds. Fast with the Java Vector API. | [2402.17764](https://arxiv.org/abs/2402.17764) | M |
| **Differentiable logic gate networks** | Each node learns which of 16 logic gates it is, then compiles to pure boolean circuits. The end-point of quantization. | [2411.04732](https://arxiv.org/abs/2411.04732) | M |
| **Spiking nets (SpikeGPT)** | Binary spikes with surrogate gradients; an RWKV-style spiking LM. Targets neuromorphic chips. | [2302.13939](https://arxiv.org/abs/2302.13939) | M |
| **Next-Gen Reservoir Computing** | Polynomial features of delayed inputs plus one ridge regression. Forecasts chaotic systems without backprop. | [2106.07688](https://arxiv.org/abs/2106.07688) | L |
| **Optical / diffractive nets** | Physical diffractive layers trained in simulation as complex phase masks. Shares FFT and complex tensors with FNO. | [1804.08711](https://arxiv.org/abs/1804.08711) | M |
| **Analog in-memory attention** | Attention in analog gain-cell arrays with hardware-aware training. Add noise simulation to the quantization stack. | [2409.19315](https://arxiv.org/abs/2409.19315) | R |
| **Thermodynamic computing** (DTMs) | Probabilistic-bit Gibbs samplers chained into diffusion-like denoisers; claims about 10,000x energy savings. A software simulator is easy. | [2510.23972](https://arxiv.org/abs/2510.23972) | R |

## 20. Uncertainty, distributed training, interpretability and beyond

| Idea | What / why for Brainiac | Link | Diff. |
|---|---|---|---|
| **Laplace approximation** | Gaussian around trained weights gives calibrated uncertainty almost for free. | [2106.14806](https://arxiv.org/abs/2106.14806) | M |
| **Deep ensembles** | N independently trained nets; strong uncertainty baseline and RL curiosity via disagreement. | [1612.01474](https://arxiv.org/pdf/1612.01474v3.pdf) | L |
| **Federated averaging** | Clients train locally, the server averages; data never leaves clients. | [1602.05629](https://arxiv.org/abs/1602.05629) | L-M |
| **DiLoCo** | Many local steps, an outer Nesterov step on pseudo-gradients; about 500x less communication. A home cluster of mixed GPUs. | [2311.08105](https://arxiv.org/abs/2311.08105) | M |
| **Differentiable physics (DiffTaichi)** | Gradients through simulators to optimise controllers. Extends autograd beyond networks. | [1910.00935](https://arxiv.org/abs/1910.00935v3) | H |
| **Sparse autoencoders** | Break layer activations into interpretable features. A built-in `SAE.train(layer)` plus feature browser. | [Transformer Circuits](https://transformer-circuits.pub/2023/monosemantic-features/) | L-M |
| **Steering vectors** | Add an activation-difference vector to the residual stream to steer behaviour. A forward hook. | [2308.10248](https://arxiv.org/abs/2308.10248v4) | L |
| **ROME** | Locate facts by causal tracing, rewrite with a rank-one update. A "debugger for weights". | [2202.05262](https://arxiv.org/abs/2202.05262) | M |
| **Machine unlearning** | Remove the influence of specific data after training: the "forget" side of a memory API. | [2503.01854](https://arxiv.org/html/2503.01854v2) | M |
| **DreamCoder** | Wake-sleep program learning that grows its own DSL library and dreams training programs. | [2006.08381](https://arxiv.org/pdf/2006.08381) | R |
| **AlphaGeometry** | Neural intuition proposes, a symbolic engine proves. Template for any domain with a checker. | [Nature](https://www.nature.com/articles/s41586-023-06747-5.pdf) | H |

---

*Not covered for lack of verified links: "Linear Transformers are Secretly Fast Weight Programmers" and Genie 3. Both are worth a look.*
