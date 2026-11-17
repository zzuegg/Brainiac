package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.random.RandomGenerator;

/**
 * A fine-grained mixture of experts with shared experts (DeepSeekMoE). Every token goes
 * through the always-on shared experts, plus its {@code topK} best routed experts out of
 * many small ones, weighted by the router's softmax affinities:
 *
 * <pre>
 * y = sum_shared e(x) + sum_{i in topK(s + bias)} s_i e_i(x),   s = softmax(x W_r)
 * </pre>
 *
 * <p>Tokens are really dispatched: each routed expert runs only on the rows sent to it
 * (gathered), and the results are gathered back per token, so the compute grows with
 * {@code topK}, not with the number of experts. Many small experts give finer
 * specialisation; shared experts hold common knowledge so routed ones need not repeat it.
 *
 * <p>Load balancing, two ways: {@link #balanceLoss()} is the expert-level auxiliary loss
 * {@code alpha * sum_i f_i P_i} (with {@code f_i} the share of assignments scaled by
 * {@code E / K}, {@code P_i} the mean affinity) to add to the training loss; and
 * {@link #updateBalanceBias(float)} is DeepSeek-V3's auxiliary-loss-free method, a per-expert
 * bias used only for choosing experts, nudged after each step towards even loads.
 *
 * <p>Works on any input {@code [..., features]}; experts are any {@link Layer}s mapping
 * {@code [rows, features]} to {@code [rows, outputs]}.
 *
 * <p>Papers: Dai et al., "DeepSeekMoE", <a href="https://arxiv.org/abs/2401.06066">arXiv:2401.06066</a>;
 * DeepSeek-AI, "DeepSeek-V3 Technical Report", <a href="https://arxiv.org/abs/2412.19437">arXiv:2412.19437</a>.
 */
public final class FineGrainedMoe implements Layer {

    /** A two-layer MLP expert. */
    private static final class MlpExpert implements Layer {
        private final Var w1;
        private final Var b1;
        private final Var w2;

        MlpExpert(int in, int hidden, int out, RandomGenerator random) {
            w1 = MixerOps.weight(random, in, hidden);
            b1 = MixerOps.filled(0f, hidden);
            w2 = MixerOps.weight(random, hidden, out);
        }

        @Override
        public Var forward(Var input) {
            return Activation.SILU.forward(input.matmul(w1).add(b1)).matmul(w2);
        }

        @Override
        public List<Var> parameters() {
            return List.of(w1, b1, w2);
        }
    }

    private final int features;
    private final int outputs;
    private final int topK;
    private final List<Layer> routed;
    private final List<Layer> shared;
    private final Var router;
    private final float[] bias;
    private Var lastBalanceLoss;
    private int[] lastLoad;

    /**
     * Creates a layer with given experts.
     *
     * @param features the input width
     * @param outputs the output width of every expert
     * @param topK how many routed experts each token uses
     * @param routed the routed experts, {@code [rows, features] -> [rows, outputs]}
     * @param shared the shared experts, used by every token; may be empty
     * @param random the random source for the router
     */
    public FineGrainedMoe(int features, int outputs, int topK, List<? extends Layer> routed,
            List<? extends Layer> shared, RandomGenerator random) {
        if (topK < 1 || topK > routed.size()) {
            throw new IllegalArgumentException("topK must be in [1, " + routed.size() + "], got " + topK);
        }
        this.features = features;
        this.outputs = outputs;
        this.topK = topK;
        this.routed = List.copyOf(routed);
        this.shared = List.copyOf(shared);
        this.router = MixerOps.weight(random, features, routed.size());
        this.bias = new float[routed.size()];
    }

    /**
     * Creates a layer of SiLU MLP experts that maps {@code features} to {@code features}.
     *
     * @param features the width
     * @param experts the number of routed experts
     * @param sharedExperts the number of shared experts
     * @param hidden each expert's hidden width
     * @param topK routed experts per token
     * @param random the random source
     * @return the layer
     */
    public static FineGrainedMoe of(int features, int experts, int sharedExperts, int hidden, int topK,
            RandomGenerator random) {
        List<Layer> r = new ArrayList<>();
        for (int i = 0; i < experts; i++) {
            r.add(new MlpExpert(features, hidden, features, random));
        }
        List<Layer> s = new ArrayList<>();
        for (int i = 0; i < sharedExperts; i++) {
            s.add(new MlpExpert(features, hidden, features, random));
        }
        return new FineGrainedMoe(features, features, topK, r, s, random);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}: SiLU MLP experts keeping
     * the input width; a drop-in for a feed-forward {@code Dense} pair.
     *
     * @param experts the number of routed experts
     * @param sharedExperts the number of shared experts
     * @param hidden each expert's hidden width
     * @param topK routed experts per token
     * @return the spec
     */
    public static LayerSpec spec(int experts, int sharedExperts, int hidden, int topK) {
        return MixerOps.sameShape((in, r) -> of(in.dim(-1), experts, sharedExperts, hidden, topK, r));
    }

    /** Expert indices per row, chosen by the biased affinities. */
    private int[][] route(float[] scores, int rows) {
        int e = routed.size();
        float[] biased = new float[e];
        int[][] chosen = new int[rows][];
        for (int n = 0; n < rows; n++) {
            for (int i = 0; i < e; i++) {
                biased[i] = scores[n * e + i] + bias[i];
            }
            chosen[n] = MixerOps.topK(biased, 0, e, topK);
        }
        return chosen;
    }

    /**
     * Returns which routed experts each row of the input uses.
     *
     * @param input {@code [..., features]}
     * @return {@code [rows][topK]} expert indices
     */
    public int[][] assignments(Var input) {
        int rows = Math.toIntExact(input.shape().size() / features);
        Var x = input.reshape(rows, features);
        return route(MixerOps.softmax(x.matmul(router)).value().toArray(), rows);
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        if (s.dim(-1) != features) {
            throw new IllegalArgumentException("FineGrainedMoe expects [..., " + features + "], got " + s);
        }
        int rows = Math.toIntExact(s.size() / features);
        int e = routed.size();
        Var x = input.reshape(rows, features);
        Var affinity = MixerOps.softmax(x.matmul(router));   // [rows, E]
        int[][] chosen = route(affinity.value().toArray(), rows);
        // dispatch: rows per expert, run each expert once, remember where each (row, slot) landed
        List<List<Integer>> members = new ArrayList<>();
        for (int i = 0; i < e; i++) {
            members.add(new ArrayList<>());
        }
        for (int n = 0; n < rows; n++) {
            for (int k = 0; k < topK; k++) {
                members.get(chosen[n][k]).add(n);
            }
        }
        int[] position = new int[rows * topK];
        int[] load = new int[e];
        List<Var> outs = new ArrayList<>();
        int offset = 0;
        for (int i = 0; i < e; i++) {
            List<Integer> m = members.get(i);
            load[i] = m.size();
            if (m.isEmpty()) {
                continue;
            }
            int[] idx = m.stream().mapToInt(Integer::intValue).toArray();
            outs.add(routed.get(i).forward(x.gather(0, MixerOps.ids(idx))));
            for (int j = 0; j < idx.length; j++) {
                int n = idx[j];
                // the slot of expert i in row n
                int slot = 0;
                while (chosen[n][slot] != i) {
                    slot++;
                }
                position[n * topK + slot] = offset + j;
            }
            offset += idx.length;
        }
        Var all = outs.size() == 1 ? outs.getFirst() : Var.concat(0, outs.toArray(Var[]::new));
        int[] flatChoice = new int[rows * topK];
        for (int n = 0; n < rows; n++) {
            for (int k = 0; k < topK; k++) {
                flatChoice[n * topK + k] = n * e + chosen[n][k];
            }
        }
        Var gates = affinity.reshape(rows * e).gather(0, MixerOps.ids(flatChoice)).reshape(rows, topK, 1);
        Var y = all.gather(0, MixerOps.ids(position)).reshape(rows, topK, outputs).mul(gates).sum(1, false);
        for (Layer sh : shared) {
            y = y.add(sh.forward(x));
        }
        // auxiliary balance loss: f_i = E / (K rows) * load_i, P_i = mean affinity
        float[] f = new float[e];
        for (int i = 0; i < e; i++) {
            f[i] = e * load[i] / (float) (topK * rows);
        }
        lastBalanceLoss = affinity.mean(0, false).mul(Tensor.of(Shape.of(e), f)).sum();
        lastLoad = load;
        int[] outDims = s.toArray();
        outDims[outDims.length - 1] = outputs;
        return y.reshape(outDims);
    }

    /**
     * The expert-level balance loss {@code sum_i f_i P_i} of the last forward pass; it is 1
     * for perfectly even loads and grows with imbalance. Multiply by a small factor and add
     * it to the training loss.
     *
     * @return a scalar, or {@code null} before the first forward pass
     */
    public Var balanceLoss() {
        return lastBalanceLoss;
    }

    /** @return how many (row, slot) assignments each routed expert got in the last forward pass */
    public int[] expertLoad() {
        return lastLoad == null ? new int[routed.size()] : lastLoad.clone();
    }

    /**
     * The auxiliary-loss-free balancing step: raises the selection bias of under-loaded
     * experts and lowers it for over-loaded ones, by {@code rate} each, using the loads of the
     * last forward pass. The bias affects only which experts are chosen, not the gate values.
     *
     * @param rate the bias update speed, for example {@code 0.001}
     */
    public void updateBalanceBias(float rate) {
        if (lastLoad == null) {
            return;
        }
        double mean = Arrays.stream(lastLoad).average().orElse(0);
        for (int i = 0; i < bias.length; i++) {
            bias[i] += lastLoad[i] < mean ? rate : lastLoad[i] > mean ? -rate : 0;
        }
    }

    /** @return the routed experts */
    public List<Layer> routedExperts() {
        return routed;
    }

    /** @return the router weight {@code [features, experts]} */
    public Var router() {
        return router;
    }

    /**
     * Computes the same output densely, running every expert on every row; for tests.
     *
     * @param input {@code [rows, features]}
     * @param weight the gate value of each row and expert, zero where unused
     * @return the output
     */
    Var dense(Var input, Function<Integer, Var> weight) {
        Var y = null;
        for (int i = 0; i < routed.size(); i++) {
            Var part = routed.get(i).forward(input).mul(weight.apply(i));
            y = y == null ? part : y.add(part);
        }
        for (Layer sh : shared) {
            y = y.add(sh.forward(input));
        }
        return y;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(router));
        routed.forEach(l -> p.addAll(l.parameters()));
        shared.forEach(l -> p.addAll(l.parameters()));
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return outputs;
    }

    @Override
    public String toString() {
        return "FineGrainedMoe(" + routed.size() + " routed, " + shared.size() + " shared, top-" + topK + ")";
    }
}
