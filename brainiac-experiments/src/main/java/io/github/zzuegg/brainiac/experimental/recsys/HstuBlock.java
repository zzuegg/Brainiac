package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A Hierarchical Sequential Transduction Unit, the block of the generative recommenders of Zhai et
 * al., 2024 ("Actions Speak Louder than Words", <a href="https://arxiv.org/abs/2402.17152">
 * arXiv:2402.17152</a>). It replaces softmax attention and the feed-forward network by one
 * gated, pointwise-normalised attention:
 * <pre>
 *   U, V, Q, K = split(SiLU(f1(norm(x))))
 *   A = SiLU(Q K^T + rab_pos + rab_time) / n        (causal, no softmax)
 *   y = x + f2(norm(A V) * U)
 * </pre>
 * Without the softmax the attention weights do not compete, so the block can express how much
 * history matters (intensity), not only which part; {@code U} gates the result elementwise. The
 * relative attention biases are learned per relative position and per log-scaled time gap
 * between two events. The block is causal.
 *
 * <p>As a {@link MultiInputLayer} it takes {@code [batch, steps, features]} and optionally a
 * second input {@code [batch, steps]} of event timestamps (without it there is no time bias),
 * and returns {@code [batch, steps, features]}.
 */
public final class HstuBlock implements MultiInputLayer {

    /** The number of log2 time-gap buckets. */
    public static final int TIME_BUCKETS = 16;

    private final int features;
    private final int heads;
    private final int maxSteps;
    private final LayerNorm norm1;
    private final LayerNorm norm2;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;
    private final Var positionBias;
    private final Var timeBias;

    /**
     * Creates a block.
     *
     * @param features the features per step, divisible by {@code heads}
     * @param heads the attention heads
     * @param maxSteps the longest sequence (relative position biases)
     * @param random the initialization source
     */
    public HstuBlock(int features, int heads, int maxSteps, SplittableRandom random) {
        if (features % heads != 0) {
            throw new IllegalArgumentException(heads + " heads do not divide " + features + " features");
        }
        this.features = features;
        this.heads = heads;
        this.maxSteps = maxSteps;
        norm1 = LayerNorm.of(features);
        norm2 = LayerNorm.of(features);
        w1 = RecsysOps.matrix(random, features, 4 * features);
        b1 = RecsysOps.zeros(4 * features);
        w2 = RecsysOps.matrix(random, features, features);
        b2 = RecsysOps.zeros(features);
        positionBias = RecsysOps.zeros(maxSteps);
        timeBias = RecsysOps.zeros(TIME_BUCKETS);
    }

    /**
     * @param heads the attention heads
     * @return a spec for {@code [steps, features]} inputs, as {@code TransformerBlock.of} gives
     */
    public static io.github.zzuegg.brainiac.nn.LayerSpec spec(int heads) {
        return new io.github.zzuegg.brainiac.nn.LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public io.github.zzuegg.brainiac.nn.Layer create(Shape input, SplittableRandom random) {
                return new HstuBlock(input.dim(-1), heads, input.dim(0), random);
            }
        };
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Var x = inputs.get(0);
        int b = x.shape().dim(0);
        int n = x.shape().dim(1);
        if (n > maxSteps) {
            throw new IllegalArgumentException(n + " steps, at most " + maxSteps);
        }
        int dh = features / heads;
        Var proj = Activation.SILU.forward(norm1.forward(x).matmul(w1).add(b1));
        Var u = proj.slice(2, 0, features);
        Var v = split(proj.slice(2, features, 2 * features), b, n, dh);
        Var q = split(proj.slice(2, 2 * features, 3 * features), b, n, dh);
        Var k = split(proj.slice(2, 3 * features, 4 * features), b, n, dh);
        Var scores = q.matmul(k.permute(0, 1, 3, 2)).div((float) Math.sqrt(dh));
        float[] rel = new float[n * n];
        float[] causal = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                rel[i * n + j] = i - j;
                causal[i * n + j] = 1;
            }
        }
        scores = scores.add(positionBias.gather(0, Tensor.of(Shape.of(n * n), rel)).reshape(1, 1, n, n));
        if (inputs.size() > 1) {
            float[] t = inputs.get(1).value().toArray();
            float[] bucket = new float[b * n * n];
            for (int s = 0; s < b; s++) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double gap = Math.abs(t[s * n + i] - t[s * n + j]);
                        bucket[(s * n + i) * n + j] = Math.min(TIME_BUCKETS - 1,
                                (int) (Math.log(1 + gap) / Math.log(2)));
                    }
                }
            }
            scores = scores.add(timeBias.gather(0, Tensor.of(Shape.of(b * n * n), bucket)).reshape(b, 1, n, n));
        }
        Var attn = Activation.SILU.forward(scores).mul(Tensor.of(Shape.of(1, 1, n, n), causal)).div(n);
        Var mixed = attn.matmul(v).permute(0, 2, 1, 3).reshape(b, n, features);
        return x.add(norm2.forward(mixed).mul(u).matmul(w2).add(b2));
    }

    private Var split(Var x, int b, int n, int dh) {
        return x.reshape(b, n, heads, dh).permute(0, 2, 1, 3);
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return inputs.get(0);
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new java.util.ArrayList<>(norm1.parameters());
        p.addAll(norm2.parameters());
        p.addAll(List.of(w1, b1, w2, b2, positionBias, timeBias));
        return p;
    }

    @Override
    public String toString() {
        return "HstuBlock(" + features + ", heads=" + heads + ")";
    }
}
