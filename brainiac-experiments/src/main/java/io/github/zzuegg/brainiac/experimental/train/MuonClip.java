package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.MultiHeadAttention;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * MuonClip (Kimi K2, Moonshot AI 2025, <a href="https://arxiv.org/abs/2507.20534">arXiv:2507.20534</a>):
 * Muon plus <em>QK-clip</em>. At high learning rates Muon lets attention logits
 * {@code q·k / sqrt(d)} grow without bound, which saturates the softmax and eventually
 * spikes the loss. QK-clip watches the largest logit of every head after each step and,
 * whenever it exceeds a threshold {@code tau}, rescales that head's query and key
 * projections by {@code sqrt(tau / max)} each, so the logit falls back to {@code tau}
 * while the attention pattern's ranking is kept. Unlike QK-norm it changes nothing in the
 * forward pass and costs nothing when logits are small.
 *
 * <p>Wraps any optimizer (Muon is the intended one, through {@code Optimizer.groups}).
 * Register each attention layer with {@link #watch(MultiHeadAttention)} (the core layer) or
 * {@link #watch(Var, Var, int, int, int, float, boolean)}, and give it the layer's input of
 * the current batch with {@link #record} before {@link #step}. With grouped keys
 * ({@code kvHeads < heads}) a key head is shared, so only the query head is scaled, by the
 * whole {@code tau / max}. Logits are measured without rotary positions (rotations keep
 * the norms that bound them).
 */
public final class MuonClip implements Optimizer {

    private final Optimizer inner;
    private final float threshold;
    private final List<Watched> watched = new ArrayList<>();
    private float lastMax;
    private int clips;

    private static final class Watched {
        final Var wq;
        final Var wk;
        final int heads;
        final int kvHeads;
        final int headSize;
        final float scale;
        final boolean causal;
        Tensor input;

        Watched(Var wq, Var wk, int heads, int kvHeads, int headSize, float scale, boolean causal) {
            this.wq = wq;
            this.wk = wk;
            this.heads = heads;
            this.kvHeads = kvHeads;
            this.headSize = headSize;
            this.scale = scale;
            this.causal = causal;
        }
    }

    /**
     * @param inner the optimizer, usually Muon for matrices and AdamW for the rest
     * @param threshold the largest allowed attention logit {@code tau}, 100 in Kimi K2
     */
    public MuonClip(Optimizer inner, float threshold) {
        if (!(threshold > 0)) {
            throw new IllegalArgumentException("threshold must be positive");
        }
        this.inner = inner;
        this.threshold = threshold;
    }

    /**
     * Watches a core attention layer (self-attention: queries and keys from one input).
     *
     * @param layer the layer
     * @return this optimizer
     */
    public MuonClip watch(MultiHeadAttention layer) {
        MultiHeadAttention.Settings s = layer.settings();
        float scale = s.scale() != 0 ? s.scale() : (float) (1 / Math.sqrt(s.headSize()));
        List<Var> p = layer.parameters();
        return watch(p.get(0), p.get(1), s.heads(), s.kvHeads(), s.headSize(), scale, s.causal());
    }

    /**
     * Watches a pair of projections.
     *
     * @param wq the query projection {@code [features, heads * headSize]}
     * @param wk the key projection {@code [features, kvHeads * headSize]}
     * @param heads the query heads
     * @param kvHeads the key heads
     * @param headSize the size per head
     * @param scale the logit scale, usually {@code 1 / sqrt(headSize)}
     * @param causal whether future positions are masked
     * @return this optimizer
     */
    public MuonClip watch(Var wq, Var wk, int heads, int kvHeads, int headSize, float scale, boolean causal) {
        watched.add(new Watched(wq, wk, heads, kvHeads, headSize, scale, causal));
        return this;
    }

    /**
     * Gives the input of a watched layer for this batch.
     *
     * @param layer the layer, watched
     * @param input its input {@code [batch, steps, features]}
     */
    public void record(MultiHeadAttention layer, Tensor input) {
        record(layer.parameters().get(0), input);
    }

    /**
     * Gives the input of the watched projections with this query weight.
     *
     * @param wq the query projection
     * @param input the layer's input {@code [batch, steps, features]}
     */
    public void record(Var wq, Tensor input) {
        for (Watched w : watched) {
            if (w.wq == wq) {
                w.input = input;
                return;
            }
        }
        throw new IllegalArgumentException("not watched");
    }

    /**
     * The largest attention logit of each query head for an input.
     *
     * @param wq the query projection
     * @param wk the key projection
     * @param input {@code [batch, steps, features]}
     * @param heads query heads
     * @param kvHeads key heads
     * @param headSize size per head
     * @param scale the logit scale
     * @param causal whether to ignore future positions
     * @return {@code [heads]} maxima
     */
    public static float[] maxLogits(Tensor wq, Tensor wk, Tensor input, int heads, int kvHeads, int headSize,
            float scale, boolean causal) {
        int n = input.shape().dim(0);
        int t = input.shape().dim(1);
        Tensor q = input.matmul(wq).reshape(n, t, heads, headSize).permute(0, 2, 1, 3);
        Tensor k = input.matmul(wk).reshape(n, t, kvHeads, headSize).permute(0, 2, 1, 3);
        int group = heads / kvHeads;
        float[] out = new float[heads];
        for (int h = 0; h < heads; h++) {
            Tensor qh = q.slice(1, h, h + 1);
            Tensor kh = k.slice(1, h / group, h / group + 1);
            Tensor logits = qh.matmul(kh.transpose()).mul(scale);
            if (causal) {
                Tensor future = Tensor.upperTriangle(t, logits.dtype(), logits.device()).sub(
                        Linalg.identity(t, logits.dtype(), logits.device()));
                logits = logits.sub(future.mul(1e30f)).maximum(Tensor.full(Shape.of(), -1e30f));
            }
            out[h] = logits.max().item();
        }
        return out;
    }

    @Override
    public void step(List<Var> parameters) {
        inner.step(parameters);
        float worst = Float.NEGATIVE_INFINITY;
        for (Watched w : watched) {
            if (w.input == null) {
                continue;
            }
            float[] max = maxLogits(w.wq.value(), w.wk.value(), w.input, w.heads, w.kvHeads, w.headSize, w.scale,
                    w.causal);
            w.input = null;
            float[] qScale = new float[w.heads * w.headSize];
            float[] kScale = new float[w.kvHeads * w.headSize];
            Arrays.fill(kScale, 1f);
            boolean any = false;
            int group = w.heads / w.kvHeads;
            for (int h = 0; h < w.heads; h++) {
                worst = Math.max(worst, max[h]);
                float gamma = max[h] > threshold ? threshold / max[h] : 1f;
                any |= gamma < 1;
                float sq = (float) Math.sqrt(gamma);
                for (int i = 0; i < w.headSize; i++) {
                    qScale[h * w.headSize + i] = group == 1 ? sq : gamma;
                    if (group == 1) {
                        kScale[h * w.headSize + i] = sq;
                    }
                }
            }
            if (any) {
                clips++;
                w.wq.assign(w.wq.value().mul(Tensor.of(Shape.of(1, qScale.length), qScale)));
                w.wk.assign(w.wk.value().mul(Tensor.of(Shape.of(1, kScale.length), kScale)));
            }
        }
        lastMax = worst;
    }

    /** @return the largest attention logit seen after the last step (before clipping) */
    public float lastMaxLogit() {
        return lastMax;
    }

    /** @return the steps at which some head was clipped */
    public int clips() {
        return clips;
    }

    @Override
    public void resetState(Var parameter, Tensor keep) {
        inner.resetState(parameter, keep);
    }

    @Override
    public void forget(Var parameter) {
        inner.forget(parameter);
    }

    @Override
    public float learningRate() {
        return inner.learningRate();
    }

    @Override
    public void learningRate(float learningRate) {
        inner.learningRate(learningRate);
    }

    @Override
    public Map<String, Tensor> state(List<Var> parameters) {
        return inner.state(parameters);
    }

    @Override
    public void loadState(List<Var> parameters, Map<String, Tensor> state) {
        inner.loadState(parameters, state);
    }

    @Override
    public String toString() {
        return "MuonClip(" + inner + ", tau=" + threshold + ")";
    }
}
