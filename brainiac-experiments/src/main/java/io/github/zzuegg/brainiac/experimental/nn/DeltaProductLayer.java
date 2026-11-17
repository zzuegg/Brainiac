package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Gated DeltaProduct: {@code steps} delta-rule updates per token, each with its own key,
 * value and step size, so the state transition of one token is a product of
 * {@code steps} generalized Householder matrices:
 *
 * <pre>
 * S_t = alpha_t (prod_{j=1..n} (I - beta_{t,j} k_{t,j} k_{t,j}^T)) S_{t-1} + ...
 * </pre>
 *
 * <p>One Householder factor ({@code n = 1}, Gated DeltaNet) is a reflection at best; a product
 * of {@code n} of them can represent any orthogonal map that needs {@code n} reflections, so
 * the recurrence can track states of non-abelian groups: with {@code n = 2} and
 * {@code beta in (0, 2)} (negative eigenvalues allowed) it can compose rotations such as the
 * 3-cycles of the permutation group S3, which a single reflection cannot. Raising {@code n}
 * trades compute for expressivity smoothly between DeltaNet and a full RNN.
 *
 * <p>The {@code n} updates of a token are run as {@code n} virtual steps of the gated delta
 * rule ({@code DeltaRuleOps.chunkwise}); the decay applies at the first one and the output is
 * read at the last one.
 *
 * <p>Paper: Siems et al., "DeltaProduct: Improving State-Tracking in Linear RNNs via
 * Householder Products", <a href="https://arxiv.org/abs/2502.10297">arXiv:2502.10297</a>;
 * Grazzi et al., "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues",
 * <a href="https://arxiv.org/abs/2411.12537">arXiv:2411.12537</a>.
 */
public final class DeltaProductLayer implements Layer {

    private final int features;
    private final int heads;
    private final int headSize;
    private final int steps;
    private final boolean negativeEigenvalues;
    private final int chunk;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wBeta;
    private final Var wAlpha;
    private final Var alphaBias;
    private final Var wGate;
    private final Var wo;

    /**
     * Creates the layer.
     *
     * @param features the width, a multiple of {@code heads}
     * @param heads the heads
     * @param steps Householder updates per token ({@code n_h})
     * @param negativeEigenvalues whether {@code beta} ranges over {@code (0, 2)} instead of
     *     {@code (0, 1)}
     * @param chunk the chunk length in tokens
     * @param random the random source
     */
    public DeltaProductLayer(int features, int heads, int steps, boolean negativeEigenvalues, int chunk,
            RandomGenerator random) {
        if (features % heads != 0 || steps < 1 || chunk < 1) {
            throw new IllegalArgumentException("need heads dividing features, steps >= 1 and chunk >= 1");
        }
        this.features = features;
        this.heads = heads;
        this.headSize = features / heads;
        this.steps = steps;
        this.negativeEigenvalues = negativeEigenvalues;
        this.chunk = chunk;
        this.wq = MixerOps.weight(random, features, features);
        this.wk = MixerOps.weight(random, features, steps * features);
        this.wv = MixerOps.weight(random, features, steps * features);
        this.wBeta = MixerOps.weight(random, features, steps * heads);
        this.wAlpha = MixerOps.uniform(random, 0.01f, features, heads);
        this.alphaBias = MixerOps.filled(-4f, heads);
        this.wGate = MixerOps.weight(random, features, features);
        this.wo = MixerOps.weight(random, features, features);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}, with negative eigenvalues
     * and chunks of 4 tokens.
     *
     * @param heads the heads
     * @param steps Householder updates per token
     * @return the spec
     */
    public static LayerSpec of(int heads, int steps) {
        return of(heads, steps, true);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}, with chunks of 4 tokens.
     *
     * @param heads the heads
     * @param steps Householder updates per token
     * @param negativeEigenvalues whether {@code beta} ranges over {@code (0, 2)}
     * @return the spec
     */
    public static LayerSpec of(int heads, int steps, boolean negativeEigenvalues) {
        return MixerOps.sameShape((in, r) -> new DeltaProductLayer(in.dim(-1), heads, steps, negativeEigenvalues, 4,
                r));
    }

    /** {@code [b, t, n * h * d] -> [b, h, t * n, d]}: the n sub-steps of a token are consecutive. */
    private Var interleave(Var x, int b, int t, int d) {
        return x.reshape(b, t, steps, heads, d).permute(0, 3, 1, 2, 4).reshape(b, heads, t * steps, d);
    }

    /** The expanded q, k, v, log alpha [b, h, t n, d] and beta [b, h, t n]. */
    private Var[] project(Var x) {
        int b = x.shape().dim(0);
        int t = x.shape().dim(1);
        int d = headSize;
        Var k = MixerOps.unit(interleave(x.matmul(wk), b, t, d));
        Var v = interleave(x.matmul(wv), b, t, d);
        Var beta = x.matmul(wBeta).sigmoid();
        if (negativeEigenvalues) {
            beta = beta.mul(2f);
        }
        beta = interleave(beta, b, t, 1).reshape(b, heads, t * steps);
        // decay only at the first sub-step of every token; read only at the last
        Var logAlpha = x.matmul(wAlpha).add(alphaBias).softplus().neg().reshape(b, t, 1, heads, 1);
        float[] first = new float[steps];
        first[0] = 1;
        float[] last = new float[steps];
        last[steps - 1] = 1;
        Tensor firstMask = Tensor.of(Shape.of(1, 1, steps, 1, 1), first);
        Tensor lastMask = Tensor.of(Shape.of(1, 1, steps, 1, 1), last);
        logAlpha = logAlpha.mul(firstMask).expand(Shape.of(b, t, steps, heads, d)).permute(0, 3, 1, 2, 4)
                .reshape(b, heads, t * steps, d);
        Var q = MixerOps.unit(x.matmul(wq).reshape(b, t, 1, heads, d)).mul(lastMask)
                .permute(0, 3, 1, 2, 4).reshape(b, heads, t * steps, d);
        return new Var[] {q, k, v, logAlpha, beta};
    }

    private Var finish(Var x, Var expanded) {
        int b = x.shape().dim(0);
        int t = x.shape().dim(1);
        Var o = expanded.reshape(b, heads, t, steps, headSize).slice(3, steps - 1, steps)
                .reshape(b, heads, t, headSize);
        Var gate = x.matmul(wGate).activation(io.github.zzuegg.brainiac.tensor.spi.ActivationOp.SILU, 0);
        return MixerOps.merge(MixerOps.rmsNorm(o)).mul(gate).matmul(wo);
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "DeltaProductLayer");
        Var[] p = project(input);
        return finish(input, DeltaRuleOps.chunkwise(p[0], p[1], p[2], p[3], p[4], chunk * steps));
    }

    /**
     * The same output computed sub-step by sub-step with the recurrent state.
     *
     * @param input {@code [batch, steps, features]}
     * @return {@code [batch, steps, features]}
     */
    public Var recurrent(Var input) {
        MixerOps.requireSequence(input, features, "DeltaProductLayer");
        Var[] p = project(input);
        return finish(input, DeltaRuleOps.recurrent(p[0], p[1], p[2], p[3], p[4]));
    }

    /** @return the Householder updates per token */
    public int householderSteps() {
        return steps;
    }

    @Override
    public List<Var> parameters() {
        return List.of(wq, wk, wv, wBeta, wAlpha, alphaBias, wGate, wo);
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "DeltaProductLayer(" + features + ", heads=" + heads + ", n_h=" + steps
                + (negativeEigenvalues ? ", beta in (0,2)" : "") + ")";
    }
}
