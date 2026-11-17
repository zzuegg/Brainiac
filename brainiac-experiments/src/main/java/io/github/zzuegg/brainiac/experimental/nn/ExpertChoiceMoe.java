package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A mixture of experts with capacity-limited routing, in two flavours:
 *
 * <ul>
 *   <li>{@link Routing#EXPERT_CHOICE expert choice}: every expert picks its
 *       {@code capacity = capacityFactor * rows / experts} best-scoring tokens from the
 *       batch. Loads are perfectly even by construction (no balance loss needed), and a token
 *       can get any number of experts, more for hard tokens and none for easy ones (the
 *       residual stream carries those).</li>
 *   <li>{@link Routing#TOKEN_CHOICE token choice} with a capacity factor: every token picks its
 *       top-{@code k} experts, but an expert takes at most
 *       {@code capacityFactor * rows * k / experts} assignments; overflow assignments are
 *       dropped, in row order (Switch Transformer / GShard). An optional Switch balance loss
 *       {@code E sum_i f_i P_i} evens out the loads.</li>
 * </ul>
 *
 * <p>Both can add the ST-MoE <b>router z-loss</b> {@code mean(logsumexp(router logits)^2)},
 * which keeps router logits small and so the routing numerically stable. The weighted losses
 * are this layer's {@link #auxiliaryLoss()}, which the core trainer adds to the loss. Gates
 * are the router's softmax probabilities. Works on any input {@code [..., features]}.
 *
 * <p>Note that expert choice selects among all tokens of the batch at once, so in a causal
 * language model a token's routing can depend on later tokens; it suits encoders, or
 * decoders with per-step capacity at inference.
 *
 * <p>Papers: Zhou et al., "Mixture-of-Experts with Expert Choice Routing",
 * <a href="https://arxiv.org/abs/2202.09368">arXiv:2202.09368</a>; Zoph et al., "ST-MoE:
 * Designing Stable and Transferable Sparse Expert Models",
 * <a href="https://arxiv.org/abs/2202.08906">arXiv:2202.08906</a>; Fedus et al., "Switch
 * Transformers", <a href="https://arxiv.org/abs/2101.03961">arXiv:2101.03961</a>.
 */
public final class ExpertChoiceMoe implements Layer {

    /** Who chooses. */
    public enum Routing {
        /** Experts pick their top tokens. */
        EXPERT_CHOICE,
        /** Tokens pick their top-k experts, up to each expert's capacity. */
        TOKEN_CHOICE
    }

    private final int features;
    private final int outputs;
    private final List<Layer> experts;
    private final Routing routing;
    private final int topK;
    private final float capacityFactor;
    private final Var router;
    private float zLossWeight;
    private float balanceWeight;
    private Var lastZLoss;
    private Var lastBalance;
    private int[] lastLoad;
    private int[] lastExpertsPerToken;
    private double lastDropped;

    /**
     * Creates the layer.
     *
     * @param features the input width
     * @param outputs the output width of every expert
     * @param experts the experts, {@code [rows, features] -> [rows, outputs]}
     * @param routing who chooses
     * @param topK token choice: experts per token; expert choice: the average experts per
     *     token, used as the default capacity factor's meaning
     * @param capacityFactor expert choice: tokens per expert as a multiple of
     *     {@code rows / experts}; token choice: slack over an even share of assignments
     * @param random the random source for the router
     */
    public ExpertChoiceMoe(int features, int outputs, List<? extends Layer> experts, Routing routing, int topK,
            float capacityFactor, RandomGenerator random) {
        if (topK < 1 || topK > experts.size() || capacityFactor <= 0) {
            throw new IllegalArgumentException("need 1 <= topK <= experts and a positive capacity factor");
        }
        this.features = features;
        this.outputs = outputs;
        this.experts = List.copyOf(experts);
        this.routing = routing;
        this.topK = topK;
        this.capacityFactor = capacityFactor;
        this.router = MixerOps.weight(random, features, experts.size());
    }

    /**
     * Creates a layer of SiLU MLP experts keeping the width.
     *
     * @param features the width
     * @param experts the number of experts
     * @param hidden each expert's hidden width
     * @param routing who chooses
     * @param topK experts per token (token choice)
     * @param capacityFactor the capacity factor
     * @param random the random source
     * @return the layer
     */
    public static ExpertChoiceMoe of(int features, int experts, int hidden, Routing routing, int topK,
            float capacityFactor, RandomGenerator random) {
        List<Layer> list = new ArrayList<>();
        for (int i = 0; i < experts; i++) {
            list.add(new SiluMlpExpert(features, hidden, features, random));
        }
        return new ExpertChoiceMoe(features, features, list, routing, topK, capacityFactor, random);
    }

    /**
     * Describes a width-keeping layer for {@code Model.sequential().add(...)}, a drop-in for a
     * feed-forward network.
     *
     * @param experts the number of experts
     * @param hidden each expert's hidden width
     * @param routing who chooses
     * @param topK experts per token (token choice)
     * @param capacityFactor the capacity factor
     * @param zLossWeight the router z-loss weight, for example {@code 1e-3}
     * @return the spec
     */
    public static LayerSpec spec(int experts, int hidden, Routing routing, int topK, float capacityFactor,
            float zLossWeight) {
        return MixerOps.sameShape((in, r) -> of(in.dim(-1), experts, hidden, routing, topK, capacityFactor, r)
                .zLossWeight(zLossWeight));
    }

    /**
     * Sets the weight of the router z-loss in {@link #auxiliaryLoss()}.
     *
     * @param weight the weight, for example {@code 1e-3}; 0 for none
     * @return this layer
     */
    public ExpertChoiceMoe zLossWeight(float weight) {
        this.zLossWeight = weight;
        return this;
    }

    /**
     * Sets the weight of the Switch balance loss in {@link #auxiliaryLoss()} (token choice).
     *
     * @param weight the weight, for example {@code 0.01}; 0 for none
     * @return this layer
     */
    public ExpertChoiceMoe balanceWeight(float weight) {
        this.balanceWeight = weight;
        return this;
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        if (s.dim(-1) != features) {
            throw new IllegalArgumentException("ExpertChoiceMoe expects [..., " + features + "], got " + s);
        }
        int rows = Math.toIntExact(s.size() / features);
        int e = experts.size();
        Var x = input.reshape(rows, features);
        Var logits = x.matmul(router);
        Var probs = MixerOps.softmax(logits);                                  // [rows, E]
        float[] p = probs.value().toArray();
        int[][] members = routing == Routing.EXPERT_CHOICE ? expertChoice(p, rows, e) : tokenChoice(p, rows, e);
        Var y = null;
        int[] load = new int[e];
        int[] perToken = new int[rows];
        for (int i = 0; i < e; i++) {
            int[] idx = members[i];
            load[i] = idx.length;
            if (idx.length == 0) {
                continue;
            }
            int[] flat = new int[idx.length];
            for (int j = 0; j < idx.length; j++) {
                flat[j] = idx[j] * e + i;
                perToken[idx[j]]++;
            }
            Tensor ids = MixerOps.ids(idx);
            Var gate = probs.reshape(rows * e).gather(0, MixerOps.ids(flat)).reshape(idx.length, 1);
            Var part = experts.get(i).forward(x.gather(0, ids)).mul(gate).scatterAdd(0, ids, rows);
            y = y == null ? part : y.add(part);
        }
        if (y == null) {
            y = Var.of(Tensor.zeros(Shape.of(rows, outputs)));
        }
        Var max = logits.max(-1, true).detach();
        Var lse = logits.sub(max).exp().sum(-1, true).log().add(max);
        lastZLoss = lse.square().mean();
        float[] f = new float[e];
        for (int i = 0; i < e; i++) {
            f[i] = load[i] / (float) Math.max(1, rows * topK);
        }
        lastBalance = probs.mean(0, false).mul(Tensor.of(Shape.of(e), f)).sum().mul(e);
        lastLoad = load;
        lastExpertsPerToken = perToken;
        int[] outDims = s.toArray();
        outDims[outDims.length - 1] = outputs;
        return y.reshape(outDims);
    }

    private int[][] expertChoice(float[] p, int rows, int e) {
        int capacity = Math.max(1, Math.min(rows, Math.round(capacityFactor * rows / e)));
        int[][] members = new int[e][];
        float[] column = new float[rows];
        for (int i = 0; i < e; i++) {
            for (int n = 0; n < rows; n++) {
                column[n] = p[n * e + i];
            }
            members[i] = MixerOps.topK(column, 0, rows, capacity);
            java.util.Arrays.sort(members[i]);
        }
        lastDropped = 0;
        return members;
    }

    private int[][] tokenChoice(float[] p, int rows, int e) {
        int capacity = Math.max(1, (int) Math.ceil(capacityFactor * rows * topK / (double) e));
        List<List<Integer>> lists = new ArrayList<>();
        for (int i = 0; i < e; i++) {
            lists.add(new ArrayList<>());
        }
        int dropped = 0;
        for (int n = 0; n < rows; n++) {
            for (int choice : MixerOps.topK(p, n * e, e, topK)) {
                if (lists.get(choice).size() < capacity) {
                    lists.get(choice).add(n);
                } else {
                    dropped++;
                }
            }
        }
        lastDropped = dropped / (double) (rows * topK);
        int[][] members = new int[e][];
        for (int i = 0; i < e; i++) {
            members[i] = lists.get(i).stream().mapToInt(Integer::intValue).toArray();
        }
        return members;
    }

    @Override
    public Var auxiliaryLoss() {
        Var a = zLossWeight == 0 || lastZLoss == null ? null : lastZLoss.mul(zLossWeight);
        Var b = balanceWeight == 0 || lastBalance == null ? null : lastBalance.mul(balanceWeight);
        return a == null ? b : b == null ? a : a.add(b);
    }

    /** @return the router z-loss of the last forward pass, or {@code null} */
    public Var zLoss() {
        return lastZLoss;
    }

    /** @return the Switch balance loss {@code E sum f_i P_i} of the last forward pass (1 when even) */
    public Var balanceLoss() {
        return lastBalance;
    }

    /** @return the tokens each expert processed in the last forward pass */
    public int[] expertLoad() {
        return lastLoad == null ? new int[experts.size()] : lastLoad.clone();
    }

    /** @return how many experts each token got in the last forward pass */
    public int[] expertsPerToken() {
        return lastExpertsPerToken == null ? new int[0] : lastExpertsPerToken.clone();
    }

    /** @return the share of token-choice assignments dropped for capacity in the last pass */
    public double droppedFraction() {
        return lastDropped;
    }

    /**
     * The coefficient of variation (standard deviation over mean) of a load vector; 0 for
     * perfectly even loads.
     *
     * @param load tokens per expert
     * @return the coefficient of variation
     */
    public static double loadImbalance(int[] load) {
        double mean = java.util.Arrays.stream(load).average().orElse(0);
        if (mean == 0) {
            return 0;
        }
        double var = 0;
        for (int l : load) {
            var += (l - mean) * (l - mean);
        }
        return Math.sqrt(var / load.length) / mean;
    }

    /** @return the router weight {@code [features, experts]} */
    public Var router() {
        return router;
    }

    /** @return the experts */
    public List<Layer> experts() {
        return experts;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(router));
        experts.forEach(l -> p.addAll(l.parameters()));
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return outputs;
    }

    @Override
    public String toString() {
        return "ExpertChoiceMoe(" + experts.size() + " experts, " + routing + ", capacity " + capacityFactor + ")";
    }
}
