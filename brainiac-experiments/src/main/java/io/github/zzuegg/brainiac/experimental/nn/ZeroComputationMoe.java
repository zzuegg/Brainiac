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
 * MoE++: a top-k mixture of experts whose pool mixes ordinary FFN experts with
 * <b>zero-computation experts</b>, so a token that the router sends to them uses less FFN
 * compute:
 *
 * <ul>
 *   <li><b>zero</b> expert: outputs 0 (the token's slot is simply dropped);</li>
 *   <li><b>copy</b> expert: outputs its input (skip);</li>
 *   <li><b>constant</b> expert: {@code a_1 x + a_2 v} with a learned vector {@code v} and
 *       {@code (a_1, a_2) = softmax(x W_c)} (replace the token by a learned constant, in
 *       part).</li>
 * </ul>
 *
 * <p>The router is softmax over all experts, top-k by score; the output is the gate-weighted
 * sum of the chosen experts. <b>Gating residuals</b> pass routing information between layers:
 * a layer {@link #following following} another adds {@code r_prev W_g} to its router logits,
 * with {@code r_prev} the previous layer's logits. Only FFN experts cost real compute, and
 * they only run on the rows routed to them; {@link #ffnExpertsPerToken()} measures it.
 *
 * <p>Paper: Jin et al., "MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation
 * Experts", <a href="https://arxiv.org/abs/2410.07348">arXiv:2410.07348</a>.
 */
public final class ZeroComputationMoe implements Layer {

    /** The kind of an expert slot. */
    public enum Kind {
        /** A SiLU MLP. */
        FFN,
        /** Outputs zero. */
        ZERO,
        /** Outputs the input. */
        COPY,
        /** Mixes the input with a learned constant. */
        CONSTANT
    }

    private final int features;
    private final int topK;
    private final List<Kind> kinds;
    private final List<Layer> ffns;
    private final List<Var> constants;
    private final List<Var> constantMix;
    private final Var router;
    private final Var residualRouter;
    private ZeroComputationMoe previous;
    private Var lastLogits;
    private double lastFfnPerToken;
    private int[] lastLoad;

    /**
     * Creates the layer with SiLU MLP experts.
     *
     * @param features the width
     * @param ffnExperts the FFN experts
     * @param zero the zero experts
     * @param copy the copy experts
     * @param constant the constant experts
     * @param hidden each FFN expert's hidden width
     * @param topK experts per token
     * @param random the random source
     */
    public ZeroComputationMoe(int features, int ffnExperts, int zero, int copy, int constant, int hidden, int topK,
            RandomGenerator random) {
        int total = ffnExperts + zero + copy + constant;
        if (topK < 1 || topK > total || ffnExperts < 1) {
            throw new IllegalArgumentException("need at least one FFN expert and 1 <= topK <= experts");
        }
        this.features = features;
        this.topK = topK;
        List<Kind> k = new ArrayList<>();
        List<Layer> f = new ArrayList<>();
        List<Var> c = new ArrayList<>();
        List<Var> cm = new ArrayList<>();
        for (int i = 0; i < ffnExperts; i++) {
            k.add(Kind.FFN);
            f.add(new SiluMlpExpert(features, hidden, features, random));
        }
        for (int i = 0; i < zero; i++) {
            k.add(Kind.ZERO);
        }
        for (int i = 0; i < copy; i++) {
            k.add(Kind.COPY);
        }
        for (int i = 0; i < constant; i++) {
            k.add(Kind.CONSTANT);
            c.add(MixerOps.uniform(random, 0.1f, features));
            cm.add(MixerOps.weight(random, features, 2));
        }
        this.kinds = List.copyOf(k);
        this.ffns = List.copyOf(f);
        this.constants = List.copyOf(c);
        this.constantMix = List.copyOf(cm);
        this.router = MixerOps.weight(random, features, total);
        this.residualRouter = MixerOps.filled(0f, total, total);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param ffnExperts the FFN experts
     * @param zero the zero experts
     * @param copy the copy experts
     * @param constant the constant experts
     * @param hidden each FFN expert's hidden width
     * @param topK experts per token
     * @return the spec
     */
    public static LayerSpec spec(int ffnExperts, int zero, int copy, int constant, int hidden, int topK) {
        return MixerOps.sameShape((in, r) -> new ZeroComputationMoe(in.dim(-1), ffnExperts, zero, copy, constant,
                hidden, topK, r));
    }

    /**
     * Adds gating residuals from an earlier layer with the same number of experts: this
     * layer's router logits get {@code previous logits x W_g} ({@code W_g} starts at zero).
     *
     * @param earlier the previous MoE++ layer, run before this one on the same tokens
     * @return this layer
     */
    public ZeroComputationMoe following(ZeroComputationMoe earlier) {
        if (earlier.kinds.size() != kinds.size()) {
            throw new IllegalArgumentException("gating residuals need the same number of experts");
        }
        this.previous = earlier;
        return this;
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Shape s = input.shape();
        if (s.dim(-1) != features) {
            throw new IllegalArgumentException("ZeroComputationMoe expects [..., " + features + "], got " + s);
        }
        int rows = Math.toIntExact(s.size() / features);
        int e = kinds.size();
        Var x = input.reshape(rows, features);
        Var logits = x.matmul(router);
        if (previous != null && previous.lastLogits != null && previous.lastLogits.shape().dim(0) == rows) {
            logits = logits.add(previous.lastLogits.matmul(residualRouter));
        }
        lastLogits = logits;
        Var prob = MixerOps.softmax(logits);
        float[] p = prob.value().toArray();
        int[][] chosen = new int[rows][];
        for (int n = 0; n < rows; n++) {
            chosen[n] = MixerOps.topK(p, n * e, e, topK);
        }
        Var flatProb = prob.reshape(rows * e);
        Var y = Var.of(Tensor.zeros(Shape.of(rows, features)));
        int[] load = new int[e];
        long ffnSlots = 0;
        int ffnIndex = 0;
        int constIndex = 0;
        for (int i = 0; i < e; i++) {
            List<Integer> members = new ArrayList<>();
            for (int n = 0; n < rows; n++) {
                for (int c : chosen[n]) {
                    if (c == i) {
                        members.add(n);
                    }
                }
            }
            load[i] = members.size();
            Kind kind = kinds.get(i);
            if (kind == Kind.FFN) {
                ffnSlots += members.size();
            }
            if (members.isEmpty() || kind == Kind.ZERO) {
                ffnIndex += kind == Kind.FFN ? 1 : 0;
                constIndex += kind == Kind.CONSTANT ? 1 : 0;
                continue;
            }
            int[] idx = members.stream().mapToInt(Integer::intValue).toArray();
            int[] flat = new int[idx.length];
            for (int j = 0; j < idx.length; j++) {
                flat[j] = idx[j] * e + i;
            }
            Var w = flatProb.gather(0, MixerOps.ids(flat)).reshape(idx.length, 1);
            Var xi = x.gather(0, MixerOps.ids(idx));
            Var out = switch (kind) {
                case FFN -> ffns.get(ffnIndex++).forward(xi, training);
                case COPY -> xi;
                case CONSTANT -> {
                    Var a = MixerOps.softmax(xi.matmul(constantMix.get(constIndex)));
                    Var v = constants.get(constIndex++);
                    yield xi.mul(a.slice(1, 0, 1)).add(a.slice(1, 1, 2).mul(v));
                }
                case ZERO -> throw new IllegalStateException();
            };
            y = y.add(out.mul(w).scatterAdd(0, MixerOps.ids(idx), rows));
        }
        lastLoad = load;
        lastFfnPerToken = ffnSlots / (double) rows;
        return y.reshape(s.toArray());
    }

    /** @return the mean number of FFN experts a token used in the last pass (its FFN compute in expert units) */
    public double ffnExpertsPerToken() {
        return lastFfnPerToken;
    }

    /** @return the rows each expert slot received in the last pass, FFN experts first */
    public int[] expertLoad() {
        return lastLoad == null ? new int[kinds.size()] : lastLoad.clone();
    }

    /** @return the kind of every expert slot */
    public List<Kind> kinds() {
        return kinds;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(router));
        ffns.forEach(l -> p.addAll(l.parameters()));
        p.addAll(constants);
        p.addAll(constantMix);
        if (previous != null) {
            p.add(residualRouter);
        }
        return p;
    }

    @Override
    public String toString() {
        return "ZeroComputationMoe(" + kinds + ", top-" + topK + ")";
    }
}
