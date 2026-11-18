package io.github.zzuegg.brainiac.experimental.combopt;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The attention model of Kool et al. for routing: a transformer encoder over the nodes (no
 * positional encoding, so it is permutation-equivariant; core {@link TransformerBlock}s) and
 * an autoregressive decoder that, at each step, forms a context query from the graph
 * embedding, the first and the last node (and the remaining capacity for CVRP), refines it
 * with one multi-head glimpse over the unvisited nodes, and scores every node with a
 * single-head compatibility clipped by {@code C tanh(.)}, masking illegal nodes.
 *
 * <p>Training is REINFORCE with a greedy-rollout baseline: {@link RolloutBaseline} keeps a
 * frozen copy of the policy whose greedy tour length is the baseline of each instance, and
 * replaces the copy after an epoch when the current policy is better on a validation set by
 * a one-sided paired t-test at the 5% level.
 *
 * <p>The decoder takes an {@link Adapter}, the hook of efficient active search
 * ({@link ActiveSearch}): it can shift the node embeddings, transform the query, or add
 * per-(current, next) logit offsets. {@link #rollout} can also force the first node (POMO's
 * multiple starts) or whole solutions (teacher forcing, for the likelihood of a given
 * tour).
 *
 * <p>Paper: Kool, van Hoof and Welling, "Attention, Learn to Solve Routing Problems!" (ICLR
 * 2019), <a href="https://arxiv.org/abs/1803.08475">arXiv:1803.08475</a>.
 */
public final class AttentionModel {

    /** How the decoder picks nodes. */
    public enum Decode {
        /** the most likely node */
        GREEDY,
        /** a sample from the policy */
        SAMPLE
    }

    /** Decoder hooks for test-time adaptation; every method defaults to "no change". */
    public interface Adapter {
        /**
         * @param embeddings node embeddings {@code [batch, nodes, dim]}
         * @return the embeddings to decode from
         */
        default Var embeddings(Var embeddings) {
            return embeddings;
        }

        /**
         * @param query the glimpse output {@code [batch, dim]}
         * @return the query to score nodes with
         */
        default Var query(Var query) {
            return query;
        }

        /**
         * @param current the current node of each rollout (-1 before the first)
         * @return logit offsets {@code [batch, nodes]}, or {@code null}
         */
        default Var logitBias(int[] current) {
            return null;
        }

        /** @return the adapter's trainable values */
        default List<Var> parameters() {
            return List.of();
        }
    }

    /**
     * A batch of rollouts.
     *
     * @param solutions each rollout's solution
     * @param lengths their lengths
     * @param logp the summed log-likelihood of each solution {@code [batch, 1]} (forced
     *     first nodes excluded)
     * @param graph the mean node embedding of each instance {@code [batch, dim]}
     */
    public record Rollout(int[][] solutions, double[] lengths, Var logp, Var graph) {
        /** @return the mean length */
        public double meanLength() {
            double s = 0;
            for (double l : lengths) {
                s += l;
            }
            return s / lengths.length;
        }
    }

    private static final float CLIP = 10f;

    private final RoutingEnv.Kind kind;
    private final int dim;
    private final int heads;
    private final int layers;
    private final long seed;
    private final Var embed;
    private final Var embedBias;
    private final List<TransformerBlock> blocks = new ArrayList<>();
    private final Var wGraph;
    private final Var wStep;
    private final Var placeholder;
    private final Var wKey;
    private final Var wValue;
    private final Var wOut;
    private final Var wLogit;

    private AttentionModel(RoutingEnv.Kind kind, int dim, int heads, int layers, long seed) {
        if (dim % heads != 0) {
            throw new IllegalArgumentException("dim must be a multiple of heads");
        }
        this.kind = kind;
        this.dim = dim;
        this.heads = heads;
        this.layers = layers;
        this.seed = seed;
        SplittableRandom r = new SplittableRandom(seed);
        int f = RoutingEnv.features(kind);
        this.embed = matrix(r, f, dim);
        this.embedBias = Var.param(Tensor.zeros(Shape.of(dim)));
        for (int l = 0; l < layers; l++) {
            blocks.add((TransformerBlock) TransformerBlock.of(heads).ffnSize(2 * dim).create(Shape.of(1, dim), r));
        }
        this.wGraph = matrix(r, dim, dim);
        this.wStep = matrix(r, 2 * dim + 1, dim);
        this.placeholder = Var.param(Tensor.uniform(Shape.of(1, 2 * dim), -1, 1, r));
        this.wKey = matrix(r, dim, dim);
        this.wValue = matrix(r, dim, dim);
        this.wOut = matrix(r, dim, dim);
        this.wLogit = matrix(r, dim, dim);
    }

    static Var matrix(SplittableRandom r, int in, int out) {
        return Var.param(Tensor.normal(Shape.of(in, out), 0, (float) (1 / Math.sqrt(in)), r));
    }

    /**
     * Creates a model.
     *
     * @param kind the routing problem
     * @param dim the embedding width
     * @param heads attention heads
     * @param layers encoder blocks
     * @param seed the seed
     * @return the model
     */
    public static AttentionModel of(RoutingEnv.Kind kind, int dim, int heads, int layers, long seed) {
        return new AttentionModel(kind, dim, heads, layers, seed);
    }

    /** @return a deep copy with the same weights */
    public AttentionModel copy() {
        AttentionModel m = new AttentionModel(kind, dim, heads, layers, seed);
        m.load(this);
        return m;
    }

    /**
     * Copies another model's weights into this one.
     *
     * @param other a model of the same shape
     */
    public void load(AttentionModel other) {
        List<Var> to = parameters();
        List<Var> from = other.parameters();
        for (int i = 0; i < to.size(); i++) {
            Tensor v = from.get(i).value();
            to.get(i).assign(Tensor.of(v.shape(), v.toArray()));
        }
    }

    /** @return the problem kind */
    public RoutingEnv.Kind kind() {
        return kind;
    }

    /** @return the embedding width */
    public int dim() {
        return dim;
    }

    /**
     * Encodes a batch of same-size instances.
     *
     * @param envs the instances
     * @return node embeddings {@code [batch, nodes, dim]}
     */
    public Var encode(List<RoutingEnv> envs) {
        int b = envs.size();
        int n = envs.get(0).nodes();
        int f = RoutingEnv.features(kind);
        float[] x = new float[b * n * f];
        for (int i = 0; i < b; i++) {
            System.arraycopy(envs.get(i).nodeFeatures(), 0, x, i * n * f, n * f);
        }
        Var h = Var.of(Tensor.of(Shape.of(b, n, f), x)).matmul(embed).add(embedBias);
        for (TransformerBlock block : blocks) {
            h = block.forward(List.of(h), false);
        }
        return h;
    }

    /**
     * Rolls out the policy on a batch.
     *
     * @param envs the instances (same size)
     * @param decode greedy or sampled
     * @param first per rollout the forced first node (TSP) or first customer (CVRP), or
     *     {@code null}
     * @param random the sampling source
     * @return the rollouts
     */
    public Rollout rollout(List<RoutingEnv> envs, Decode decode, int[] first, SplittableRandom random) {
        return rollout(envs, null, decode, first, null, null, random);
    }

    /**
     * Rolls out the policy with every option.
     *
     * @param envs the instances (same size)
     * @param embeddings precomputed node embeddings {@code [batch, nodes, dim]}, or
     *     {@code null} to encode {@code envs}
     * @param decode greedy or sampled
     * @param first forced first nodes, or {@code null}
     * @param forced whole forced solutions (teacher forcing; shorter ones are padded with
     *     no-ops), or {@code null}; with {@code first} as well, the first step is still left
     *     out of the likelihood
     * @param adapter decoder hooks, or {@code null}
     * @param random the sampling source
     * @return the rollouts
     */
    public Rollout rollout(List<RoutingEnv> envs, Var embeddings, Decode decode, int[] first, int[][] forced,
            Adapter adapter, SplittableRandom random) {
        Adapter ad = adapter == null ? new Adapter() { } : adapter;
        int b = envs.size();
        int n = envs.get(0).nodes();
        int dh = dim / heads;
        Var h = ad.embeddings(embeddings != null ? embeddings : encode(envs));
        Var graph = h.mean(1, false);
        Var graphCtx = graph.matmul(wGraph);
        Var keys = h.matmul(wKey).reshape(b, n, heads, dh).permute(0, 2, 3, 1);       // [b, M, dh, n]
        Var values = h.matmul(wValue).reshape(b, n, heads, dh).permute(0, 2, 1, 3);   // [b, M, n, dh]
        Var logitKeys = h.matmul(wLogit).permute(0, 2, 1);                           // [b, d, n]
        Var flat = h.reshape(b * n, dim);
        RoutingEnv.State[] states = new RoutingEnv.State[b];
        for (int i = 0; i < b; i++) {
            states[i] = envs.get(i).start();
        }
        Var logp = null;
        int t = 0;
        while (true) {
            boolean all = true;
            for (RoutingEnv.State s : states) {
                all &= s.done();
            }
            if (all) {
                break;
            }
            int[] current = new int[b];
            float[] mask = new float[b * n];
            float[] active = new float[b];
            float[] rem = new float[b];
            boolean firstStep = kind == RoutingEnv.Kind.TSP && t == 0;
            for (int i = 0; i < b; i++) {
                current[i] = states[i].current();
                boolean[] m = states[i].mask();
                for (int j = 0; j < n; j++) {
                    mask[i * n + j] = m[j] ? 0 : -1e9f;
                }
                active[i] = states[i].done() ? 0 : 1;
                rem[i] = (float) states[i].remaining();
            }
            Var stepIn;
            if (firstStep) {
                stepIn = Var.concat(1, placeholder.expand(Shape.of(b, 2 * dim)), Var.of(Tensor.of(Shape.of(b, 1), rem)));
            } else {
                float[] fi = new float[b];
                float[] li = new float[b];
                for (int i = 0; i < b; i++) {
                    fi[i] = i * n + Math.max(0, states[i].first());
                    li[i] = i * n + Math.max(0, current[i]);
                }
                stepIn = Var.concat(1, flat.gather(0, Tensor.of(Shape.of(b), fi)),
                        flat.gather(0, Tensor.of(Shape.of(b), li)), Var.of(Tensor.of(Shape.of(b, 1), rem)));
            }
            Var q = graphCtx.add(stepIn.matmul(wStep));
            Var scores = q.reshape(b, heads, 1, dh).matmul(keys).mul((float) (1 / Math.sqrt(dh)))
                    .add(Tensor.of(Shape.of(b, 1, 1, n), mask));
            Var glimpse = scores.softmax(3).matmul(values).reshape(b, dim).matmul(wOut);
            glimpse = ad.query(glimpse);
            Var logits = glimpse.reshape(b, 1, dim).matmul(logitKeys).reshape(b, n).mul((float) (1 / Math.sqrt(dim)))
                    .tanh().mul(CLIP);
            Var bias = firstStep ? null : ad.logitBias(current);
            if (bias != null) {
                logits = logits.add(bias);
            }
            Var lp = logits.add(Tensor.of(Shape.of(b, n), mask)).logSoftmax(1);
            float[] probs = lp.value().toArray();
            int[] pick = new int[b];
            boolean forcedFirst = first != null && t == 0;
            for (int i = 0; i < b; i++) {
                if (states[i].done()) {
                    pick[i] = argmaxLegal(states[i].mask());
                } else if (forced != null) {
                    pick[i] = t < forced[i].length ? forced[i][t] : argmaxLegal(states[i].mask());
                } else if (forcedFirst) {
                    pick[i] = first[i];
                } else if (decode == Decode.GREEDY) {
                    int best = -1;
                    for (int j = 0; j < n; j++) {
                        if (mask[i * n + j] == 0 && (best < 0 || probs[i * n + j] > probs[i * n + best])) {
                            best = j;
                        }
                    }
                    pick[i] = best;
                } else {
                    double u = random.nextDouble();
                    int last = -1;
                    for (int j = 0; j < n; j++) {
                        if (mask[i * n + j] == 0) {
                            last = j;
                            u -= Math.exp(probs[i * n + j]);
                            if (u <= 0) {
                                break;
                            }
                        }
                    }
                    pick[i] = last;
                }
            }
            if (!forcedFirst) {
                float[] idx = new float[b];
                for (int i = 0; i < b; i++) {
                    idx[i] = pick[i];
                }
                Var step = lp.takeAlong(Tensor.of(Shape.of(b, 1), idx)).mul(Tensor.of(Shape.of(b, 1), active));
                logp = logp == null ? step : logp.add(step);
            }
            for (int i = 0; i < b; i++) {
                states[i].step(pick[i]);
            }
            t++;
        }
        int[][] sol = new int[b][];
        double[] len = new double[b];
        for (int i = 0; i < b; i++) {
            sol[i] = states[i].solution();
            len[i] = envs.get(i).length(sol[i]);
        }
        if (logp == null) {
            logp = Var.of(Tensor.zeros(Shape.of(b, 1)));
        }
        return new Rollout(sol, len, logp, graph);
    }

    private static int argmaxLegal(boolean[] m) {
        for (int j = 0; j < m.length; j++) {
            if (m[j]) {
                return j;
            }
        }
        throw new IllegalStateException("no legal node");
    }

    /**
     * The REINFORCE loss {@code mean((L - b) log p)}.
     *
     * @param rollout sampled rollouts
     * @param baseline one baseline length per rollout
     * @return the loss
     */
    public static Var reinforceLoss(Rollout rollout, double[] baseline) {
        int b = rollout.lengths().length;
        float[] adv = new float[b];
        for (int i = 0; i < b; i++) {
            adv[i] = (float) (rollout.lengths()[i] - baseline[i]);
        }
        return rollout.logp().mul(Tensor.of(Shape.of(b, 1), adv)).mean();
    }

    /**
     * Greedy lengths without gradients.
     *
     * @param envs the instances
     * @return greedy solution lengths
     */
    public double[] greedyLengths(List<RoutingEnv> envs) {
        return Var.noGrad(() -> rollout(envs, Decode.GREEDY, null, null)).lengths();
    }

    /**
     * The best of {@code samples} sampled solutions per instance, without gradients.
     *
     * @param env the instance
     * @param samples the sample count
     * @param random the source
     * @return the shortest length
     */
    public double bestOfSamples(RoutingEnv env, int samples, SplittableRandom random) {
        List<RoutingEnv> rep = new ArrayList<>();
        for (int i = 0; i < samples; i++) {
            rep.add(env);
        }
        double[] l = Var.noGrad(() -> rollout(rep, Decode.SAMPLE, null, random)).lengths();
        double best = Double.POSITIVE_INFINITY;
        for (double v : l) {
            best = Math.min(best, v);
        }
        return best;
    }

    /**
     * One optimizer step with gradient-norm clipping.
     *
     * @param loss the loss
     * @param parameters the parameters
     * @param optimizer the optimizer
     * @param maxNorm the global gradient norm cap
     */
    public static void step(Var loss, List<Var> parameters, Optimizer optimizer, double maxNorm) {
        loss.backward();
        double sq = 0;
        for (Var p : parameters) {
            if (p.grad() != null) {
                for (float g : p.grad().toArray()) {
                    sq += g * g;
                }
            }
        }
        double norm = Math.sqrt(sq);
        if (norm > maxNorm) {
            float s = (float) (maxNorm / norm);
            for (Var p : parameters) {
                if (p.grad() != null) {
                    p.setGrad(p.grad().mul(s));
                }
            }
        }
        optimizer.step(parameters);
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embed, embedBias));
        for (TransformerBlock block : blocks) {
            p.addAll(block.parameters());
        }
        p.addAll(List.of(wGraph, wStep, placeholder, wKey, wValue, wOut, wLogit));
        return p;
    }

    /**
     * Kool et al.'s greedy-rollout baseline with paired t-test replacement.
     */
    public static final class RolloutBaseline {
        private final AttentionModel baseline;
        private final List<RoutingEnv> validation;
        private double[] baselineValidation;
        private int replacements;

        /**
         * Creates the baseline from the current policy.
         *
         * @param policy the policy (copied)
         * @param validation instances for the replacement test
         */
        public RolloutBaseline(AttentionModel policy, List<RoutingEnv> validation) {
            this.baseline = policy.copy();
            this.validation = List.copyOf(validation);
            this.baselineValidation = baseline.greedyLengths(validation);
        }

        /**
         * @param envs instances
         * @return the frozen policy's greedy lengths
         */
        public double[] lengths(List<RoutingEnv> envs) {
            return baseline.greedyLengths(envs);
        }

        /**
         * The end-of-epoch test: replaces the baseline by the policy when the policy's mean
         * greedy length is lower and a one-sided paired t-test rejects equality at 5%.
         *
         * @param policy the current policy
         * @return whether it was replaced
         */
        public boolean update(AttentionModel policy) {
            double[] now = policy.greedyLengths(validation);
            int n = now.length;
            double mean = 0;
            for (int i = 0; i < n; i++) {
                mean += baselineValidation[i] - now[i];
            }
            mean /= n;
            double var = 0;
            for (int i = 0; i < n; i++) {
                double d = baselineValidation[i] - now[i] - mean;
                var += d * d;
            }
            double se = Math.sqrt(var / Math.max(1, n - 1) / n);
            double tStat = se > 0 ? mean / se : (mean > 0 ? Double.POSITIVE_INFINITY : 0);
            // one-sided 5% critical value of Student's t (normal approximation above 30 df)
            double critical = n > 30 ? 1.645 : 1.70;
            if (mean > 0 && tStat > critical) {
                baseline.load(policy);
                baselineValidation = now;
                replacements++;
                return true;
            }
            return false;
        }

        /** @return how often the baseline was replaced */
        public int replacements() {
            return replacements;
        }
    }
}
