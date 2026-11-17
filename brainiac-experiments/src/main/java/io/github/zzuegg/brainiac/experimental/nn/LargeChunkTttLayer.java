package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Large-chunk test-time training, LaCT (Zhang et al., "Test-time training done right",
 * <a href="https://arxiv.org/abs/2505.23884">arXiv:2505.23884</a>): a fast-weight matrix per
 * head is a key-to-value memory trained <em>while reading</em>, but it is updated once per large
 * chunk of tokens instead of once per token (or tiny mini-batch, as in {@link TestTimeTrainingLayer}),
 * which turns the inner loop into a few big matrix products (high hardware utilization) and lets
 * the state be large. Within a chunk, causal softmax attention covers the tokens the fast weights
 * have not absorbed yet, the paper's hybrid design.
 *
 * <pre>
 * for each chunk c:
 *   o_t   = q_t W_c + localAttention(q_t, chunk keys and values up to t)
 *   W_(c+1) = normalize( W_c - eta * sum_(t in c) beta_t k_t^T (k_t W_c - v_t) )
 * </pre>
 *
 * <p>The update is one gradient step on the chunk's regression loss with learned per-token rates
 * {@code beta}, followed by row normalization of the fast weights (the paper's weight
 * normalization, which keeps the large-step update stable). The outer loop learns the
 * projections, the initial fast weights and the rates by backpropagating through the updates.
 * Input and output {@code [batch, steps, features]}.
 */
public final class LargeChunkTttLayer implements Layer {
    private final int features;
    private final int heads;
    private final int headSize;
    private final int chunk;
    private final boolean localAttention;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wBeta;
    private final Var fast0;
    private final Var etaRaw;
    private final Var wo;

    /**
     * Creates the layer.
     *
     * @param features the width, a multiple of {@code heads}
     * @param heads heads
     * @param chunk tokens per fast-weight update
     * @param localAttention whether causal attention within each chunk is added
     * @param random the random source
     */
    public LargeChunkTttLayer(int features, int heads, int chunk, boolean localAttention, RandomGenerator random) {
        if (features % heads != 0 || chunk < 1) {
            throw new IllegalArgumentException("need features divisible by heads and chunk >= 1");
        }
        this.features = features;
        this.heads = heads;
        this.headSize = features / heads;
        this.chunk = chunk;
        this.localAttention = localAttention;
        this.wq = MixerOps.weight(random, features, features);
        this.wk = MixerOps.weight(random, features, features);
        this.wv = MixerOps.weight(random, features, features);
        this.wBeta = MixerOps.weight(random, features, heads);
        float[] eye = new float[heads * headSize * headSize];
        for (int h = 0; h < heads; h++) {
            for (int i = 0; i < headSize; i++) {
                eye[(h * headSize + i) * headSize + i] = 1;
            }
        }
        this.fast0 = Var.param(Tensor.of(Shape.of(1, heads, headSize, headSize), eye)
                .add(Tensor.uniform(Shape.of(1, heads, headSize, headSize), -0.1f, 0.1f, random)));
        this.etaRaw = MixerOps.filled(0f, heads);
        this.wo = MixerOps.weight(random, features, features);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param heads heads
     * @param chunk tokens per update
     * @param localAttention with causal attention inside each chunk
     * @return the spec
     */
    public static LayerSpec of(int heads, int chunk, boolean localAttention) {
        return MixerOps.sameShape((in, r) -> new LargeChunkTttLayer(in.dim(-1), heads, chunk, localAttention, r));
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "LargeChunkTttLayer");
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        Var q = MixerOps.unit(MixerOps.split(input.matmul(wq), heads));                 // [b, h, t, d]
        Var k = MixerOps.unit(MixerOps.split(input.matmul(wk), heads));
        Var v = MixerOps.split(input.matmul(wv).activation(ActivationOp.SILU, 0), heads);
        Var beta = input.matmul(wBeta).sigmoid().permute(0, 2, 1).reshape(b, heads, t, 1);
        Var eta = etaRaw.softplus().reshape(1, heads, 1, 1);
        Var w = fast0.expand(Shape.of(b, heads, headSize, headSize));
        List<Var> out = new ArrayList<>();
        for (int start = 0; start < t; start += chunk) {
            int end = Math.min(t, start + chunk);
            Var qc = q.slice(2, start, end);
            Var kc = k.slice(2, start, end);
            Var vc = v.slice(2, start, end);
            Var o = qc.matmul(w);
            if (localAttention) {
                o = o.add(Functional.attention(qc, kc, vc, true, (float) (1 / Math.sqrt(headSize))));
            }
            out.add(o);
            if (end < t) {
                Var residual = kc.matmul(w).sub(vc).mul(beta.slice(2, start, end));      // [b, h, c, d]
                w = w.sub(kc.transpose().matmul(residual).mul(eta));
                w = w.div(w.square().sum(-1, true).add(1e-6f).sqrt());
            }
        }
        Var o = Var.concat(2, out.toArray(Var[]::new));
        return MixerOps.merge(MixerOps.rmsNorm(o)).matmul(wo);
    }

    @Override
    public List<Var> parameters() {
        return List.of(wq, wk, wv, wBeta, fast0, etaRaw, wo);
    }

    @Override
    public String toString() {
        return "LargeChunkTttLayer(" + features + ", heads=" + heads + ", chunk=" + chunk + ")";
    }
}
