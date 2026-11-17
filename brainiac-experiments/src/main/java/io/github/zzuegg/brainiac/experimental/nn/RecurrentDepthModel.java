package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Recurrent-depth latent reasoning (Huginn): a prelude embeds the input, a recurrent core
 * block is iterated {@code r} times on a latent state, and a coda reads the result out:
 *
 * <pre>
 * e = P(x);   s_0 ~ N(0, sigma^2);   s_i = R(adapter([s_{i-1}, e]));   y = C(s_r)
 * </pre>
 *
 * <p>The input embedding is injected at every iteration, so the core is a fixed-point style
 * refinement that stays anchored to the input. During training {@code r} is drawn per batch
 * from a log-normal Poisson distribution with mean {@code meanIterations} (heavy tail, so the
 * model sees both few and many iterations), and gradients flow through the last
 * {@code backpropIterations} iterations only (truncated backpropagation; earlier states are
 * detached), which keeps memory constant. At test time {@link #iterations(int)} sets
 * {@code r}: more iterations means more compute and, on tasks that need sequential reasoning,
 * better answers, without any halting unit (unlike ACT or PonderNet).
 *
 * <p>Input and output are {@code [batch, steps, features]}; prelude, core and coda keep that
 * shape. {@link #lastIterations()} reports the iterations of the last pass.
 *
 * <p>Paper: Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent
 * Depth Approach", <a href="https://arxiv.org/abs/2502.05171">arXiv:2502.05171</a>.
 */
public final class RecurrentDepthModel implements Layer {

    private final int features;
    private final Layer prelude;
    private final Layer core;
    private final Layer coda;
    private final Var adapter;
    private final double meanIterations;
    private final int backpropIterations;
    private final float stateStd;
    private final SplittableRandom sampler;
    private int testIterations;
    private int lastIterations;

    /**
     * Creates the model.
     *
     * @param features the width
     * @param prelude the prelude
     * @param core the recurrent core
     * @param coda the coda
     * @param meanIterations the mean of the training iteration distribution
     * @param backpropIterations iterations that receive gradients
     * @param seed the seed of the iteration sampler and the initial state
     * @param random the random source of the adapter
     */
    public RecurrentDepthModel(int features, Layer prelude, Layer core, Layer coda, double meanIterations,
            int backpropIterations, long seed, SplittableRandom random) {
        if (meanIterations < 1 || backpropIterations < 1) {
            throw new IllegalArgumentException("need meanIterations >= 1 and backpropIterations >= 1");
        }
        this.features = features;
        this.prelude = prelude;
        this.core = core;
        this.coda = coda;
        float[] a = Tensor.uniform(Shape.of(2 * features, features), -0.02f, 0.02f, random).toArray();
        for (int i = 0; i < features; i++) {
            a[i * features + i] += 1;                     // the state passes through
            a[(features + i) * features + i] += 1;        // and the input embedding is added
        }
        this.adapter = Var.param(Tensor.of(Shape.of(2 * features, features), a));
        this.meanIterations = meanIterations;
        this.backpropIterations = backpropIterations;
        this.stateStd = (float) Math.sqrt(0.4);
        this.sampler = new SplittableRandom(seed);
        this.testIterations = (int) Math.round(meanIterations);
    }

    /**
     * Describes the model for {@code Model.sequential().add(...)}.
     *
     * @param prelude the prelude spec
     * @param core the core spec (iterated)
     * @param coda the coda spec
     * @param meanIterations the mean training iterations
     * @param backpropIterations iterations that receive gradients
     * @return the spec
     */
    public static LayerSpec of(LayerSpec prelude, LayerSpec core, LayerSpec coda, double meanIterations,
            int backpropIterations) {
        return MixerOps.sameShape((in, r) -> new RecurrentDepthModel(in.dim(-1), prelude.create(in, r),
                core.create(in, r), coda.create(in, r), meanIterations, backpropIterations, r.nextLong(), r));
    }

    /**
     * Draws a training iteration count: {@code tau ~ N(log(mean - 1) - 1/2, 1)},
     * {@code r ~ Poisson(e^tau) + 1}, so {@code E[r] = mean} (the paper's log-normal Poisson).
     *
     * @param random the random source
     * @param mean the mean iterations
     * @return the count, at least 1
     */
    public static int sampleIterations(SplittableRandom random, double mean) {
        double tau = Math.log(Math.max(mean - 1, 0.25)) - 0.5 + random.nextGaussian();
        double lambda = Math.min(Math.exp(tau), 4 * mean);
        // Poisson by inversion
        double l = Math.exp(-lambda);
        int k = 0;
        double p = 1;
        do {
            k++;
            p *= random.nextDouble();
        } while (p > l);
        return k;   // Poisson sample k - 1, plus one
    }

    /**
     * Sets the iterations used outside training (test-time compute).
     *
     * @param r the iterations, at least 1
     * @return this model
     */
    public RecurrentDepthModel iterations(int r) {
        if (r < 1) {
            throw new IllegalArgumentException("at least one iteration");
        }
        this.testIterations = r;
        return this;
    }

    /** @return the iterations of the last forward pass */
    public int lastIterations() {
        return lastIterations;
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        MixerOps.requireSequence(input, features, "RecurrentDepthModel");
        int r = training ? sampleIterations(sampler, meanIterations) : testIterations;
        lastIterations = r;
        Var e = prelude.forward(input, training);
        SplittableRandom noise = training ? sampler : new SplittableRandom(1234);
        Var s = Var.of(Tensor.normal(e.shape(), 0, stateStd, noise));
        for (int i = 0; i < r; i++) {
            Var state = i < r - backpropIterations ? s.detach() : s;
            s = core.forward(Var.concat(-1, state, i < r - backpropIterations ? e.detach() : e).matmul(adapter),
                    training);
        }
        return coda.forward(s, training);
    }

    @Override
    public Var auxiliaryLoss() {
        Var total = null;
        for (Layer l : List.of(prelude, core, coda)) {
            Var a = l.auxiliaryLoss();
            if (a != null) {
                total = total == null ? a : total.add(a);
            }
        }
        return total;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(prelude.parameters());
        p.addAll(core.parameters());
        p.add(adapter);
        p.addAll(coda.parameters());
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "RecurrentDepthModel(mean r=" + meanIterations + ", backprop " + backpropIterations + ", core " + core
                + ")";
    }
}
