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
 * ReMoE, a fully differentiable mixture of experts: the TopK-plus-softmax router is
 * replaced by a ReLU, {@code g = ReLU(x W_r)}, and {@code y = sum_i g_i E_i(x)} over the
 * experts with a positive gate. ReLU routing is continuous (an expert fades in and out
 * instead of jumping at the top-k boundary), and the number of active experts can differ per
 * token, so hard tokens can use more experts than easy ones.
 *
 * <p>Sparsity is controlled by an L1 penalty on the gates, weighted per expert by its load
 * (which also balances it), {@code L = lambda sum_i f_i mean(g_i)} with
 * {@code f_i = E / (k N) * count_i}. The coefficient adapts after every training pass,
 * {@code lambda <- lambda * alpha^sign(target - sparsity)}, towards the sparsity
 * {@code 1 - k / E} of a top-{@code k} router, so the average compute matches top-k without a
 * hand-tuned penalty. The weighted penalty is the {@link #auxiliaryLoss()}.
 *
 * <p>The router has a bias: for inputs that are symmetric around zero, the sign of a bias-free
 * {@code x W} is positive for half of them whatever {@code W} is, so without it no expert
 * could be active for fewer than half of the tokens.
 *
 * <p>Experts run only on the rows routed to them. Works on {@code [..., features]}.
 *
 * <p>Paper: Wang et al., "ReMoE: Fully Differentiable Mixture-of-Experts with ReLU Routing",
 * <a href="https://arxiv.org/abs/2412.14711">arXiv:2412.14711</a>.
 */
public final class ReluRoutedMoe implements Layer {

    private final int features;
    private final int targetActive;
    private final List<Layer> experts;
    private final Var router;
    private final Var routerBias;
    private final float alpha;
    private float lambda;
    private Var lastPenalty;
    private int[] lastLoad;
    private int[] lastPerToken;

    /**
     * Creates the layer.
     *
     * @param features the input and output width
     * @param experts the experts, {@code [rows, features] -> [rows, features]}
     * @param targetActive the average experts per token to aim for ({@code k})
     * @param initialLambda the starting L1 coefficient
     * @param alpha the adaptation factor of the coefficient, above 1 (the paper uses 1.2)
     * @param random the random source for the router
     */
    public ReluRoutedMoe(int features, List<? extends Layer> experts, int targetActive, float initialLambda,
            float alpha, RandomGenerator random) {
        if (targetActive < 1 || targetActive > experts.size() || alpha <= 1) {
            throw new IllegalArgumentException("need 1 <= targetActive <= experts and alpha > 1");
        }
        this.features = features;
        this.targetActive = targetActive;
        this.experts = List.copyOf(experts);
        this.router = MixerOps.weight(random, features, experts.size());
        this.routerBias = MixerOps.filled(0f, experts.size());
        this.lambda = initialLambda;
        this.alpha = alpha;
    }

    /**
     * Creates a layer of SiLU MLP experts with lambda starting at {@code 1e-3} and
     * {@code alpha = 1.05} (gentler than the paper's 1.2, for runs of hundreds of steps).
     *
     * @param features the width
     * @param experts the number of experts
     * @param hidden each expert's hidden width
     * @param targetActive the average experts per token
     * @param random the random source
     * @return the layer
     */
    public static ReluRoutedMoe of(int features, int experts, int hidden, int targetActive, RandomGenerator random) {
        List<Layer> list = new ArrayList<>();
        for (int i = 0; i < experts; i++) {
            list.add(new SiluMlpExpert(features, hidden, features, random));
        }
        return new ReluRoutedMoe(features, list, targetActive, 1e-3f, 1.05f, random);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param experts the number of experts
     * @param hidden each expert's hidden width
     * @param targetActive the average experts per token
     * @return the spec
     */
    public static LayerSpec spec(int experts, int hidden, int targetActive) {
        return MixerOps.sameShape((in, r) -> of(in.dim(-1), experts, hidden, targetActive, r));
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Shape s = input.shape();
        if (s.dim(-1) != features) {
            throw new IllegalArgumentException("ReluRoutedMoe expects [..., " + features + "], got " + s);
        }
        int rows = Math.toIntExact(s.size() / features);
        int e = experts.size();
        Var x = input.reshape(rows, features);
        Var gates = x.matmul(router).add(routerBias).relu();                                    // [rows, E]
        float[] g = gates.value().toArray();
        int[] load = new int[e];
        int[] perToken = new int[rows];
        Var y = Var.of(Tensor.zeros(Shape.of(rows, features)));
        Var flatGates = gates.reshape(rows * e);
        for (int i = 0; i < e; i++) {
            List<Integer> members = new ArrayList<>();
            for (int n = 0; n < rows; n++) {
                if (g[n * e + i] > 0) {
                    members.add(n);
                    perToken[n]++;
                }
            }
            load[i] = members.size();
            if (members.isEmpty()) {
                continue;
            }
            int[] idx = members.stream().mapToInt(Integer::intValue).toArray();
            int[] flat = new int[idx.length];
            for (int j = 0; j < idx.length; j++) {
                flat[j] = idx[j] * e + i;
            }
            Var w = flatGates.gather(0, MixerOps.ids(flat)).reshape(idx.length, 1);
            Var out = experts.get(i).forward(x.gather(0, MixerOps.ids(idx)), training).mul(w);
            y = y.add(out.scatterAdd(0, MixerOps.ids(idx), rows));
        }
        float[] f = new float[e];
        for (int i = 0; i < e; i++) {
            f[i] = e * load[i] / (float) (targetActive * rows);
        }
        lastLoad = load;
        lastPerToken = perToken;
        if (training) {
            lastPenalty = gates.mean(0, false).mul(Tensor.of(Shape.of(e), f)).sum().mul(lambda);
            double sparsity = 1 - sum(load) / (double) (rows * e);
            double target = 1 - targetActive / (double) e;
            lambda *= (float) Math.pow(alpha, Math.signum(target - sparsity));
        } else {
            lastPenalty = null;
        }
        int[] outDims = s.toArray();
        return y.reshape(outDims);
    }

    private static long sum(int[] a) {
        long s = 0;
        for (int v : a) {
            s += v;
        }
        return s;
    }

    /** @return the load-weighted L1 penalty of the last training pass (already times lambda) */
    @Override
    public Var auxiliaryLoss() {
        return lastPenalty;
    }

    /** @return the current L1 coefficient */
    public float lambda() {
        return lambda;
    }

    /** @return how many rows each expert processed in the last pass */
    public int[] expertLoad() {
        return lastLoad == null ? new int[experts.size()] : lastLoad.clone();
    }

    /** @return how many experts each row used in the last pass */
    public int[] expertsPerToken() {
        return lastPerToken == null ? new int[0] : lastPerToken.clone();
    }

    /** @return the experts */
    public List<Layer> experts() {
        return experts;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(router, routerBias));
        experts.forEach(l -> p.addAll(l.parameters()));
        return p;
    }

    @Override
    public String toString() {
        return "ReluRoutedMoe(" + experts.size() + " experts, target " + targetActive + " active)";
    }
}
