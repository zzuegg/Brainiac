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
 * A test-time training (TTT) layer: an RNN whose hidden state is the weights {@code W} of a
 * small inner model, updated by a gradient step on a self-supervised loss for every token,
 * also at inference. For each token, a "training view" {@code k = x theta_K} must be mapped to
 * a "label view" {@code v = x theta_V}:
 *
 * <pre>
 * l(W; x_t) = || f(k_t; W) - v_t ||^2,     W_t = W_{t-1} - eta_t grad l(W_{t-1}; x_t)
 * z_t = f(q_t; W_t),   q_t = x_t theta_Q
 * </pre>
 *
 * <p>{@link Variant#LINEAR} uses {@code f(k) = k W} (TTT-Linear) and {@link Variant#MLP} a
 * two-layer {@code f(k) = tanh(k W1) W2} (TTT-MLP). The per-token learning rate
 * {@code eta_t = eta sigmoid(x_t w)} is learned; keys and queries are L2-normalized so the
 * step size does not depend on their scale. Tokens are processed in mini-batches of
 * {@code batch} that all take their gradient at the weights from the start of the
 * mini-batch, computed in the paper's "dual form": no per-token weight matrices are built,
 * e.g. for TTT-Linear {@code Z = Q W0 - 2 (Q K^T . mask . eta) (K W0 - V)}. The inner
 * gradients are written with ordinary {@link Var} operations, so the outer loss trains
 * {@code theta}, the initial weights and the learning rates through the inner updates
 * (gradients of gradients). A mini-batch of 1 is the fully online update.
 *
 * <p>Input and output are {@code [batch, steps, features]}; causal.
 *
 * <p>Paper: Sun et al., "Learning to (Learn at Test Time): RNNs with Expressive Hidden
 * States", <a href="https://arxiv.org/abs/2407.04620">arXiv:2407.04620</a>.
 */
public final class TestTimeTrainingLayer implements Layer {

    /** The inner model. */
    public enum Variant {
        /** {@code f(k) = k W}. */
        LINEAR,
        /** {@code f(k) = tanh(k W1) W2}. */
        MLP
    }

    private final int features;
    private final int heads;
    private final int headSize;
    private final int hidden;
    private final int miniBatch;
    private final float baseRate;
    private final Variant variant;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wo;
    private final Var wEta;
    private final Var init1;
    private final Var init2;
    private float[] lastInnerLosses = new float[0];

    /**
     * Creates the layer.
     *
     * @param features the width, a multiple of {@code heads}
     * @param heads the number of heads, each with its own inner model
     * @param variant linear or MLP inner model
     * @param miniBatch tokens per inner gradient step
     * @param baseRate the inner learning rate scale {@code eta}
     * @param random the random source
     */
    public TestTimeTrainingLayer(int features, int heads, Variant variant, int miniBatch, float baseRate,
            RandomGenerator random) {
        if (features % heads != 0) {
            throw new IllegalArgumentException("features " + features + " not divisible by heads " + heads);
        }
        this.features = features;
        this.heads = heads;
        this.headSize = features / heads;
        this.hidden = 2 * headSize;
        this.miniBatch = miniBatch;
        this.baseRate = baseRate;
        this.variant = variant;
        this.wq = MixerOps.weight(random, features, features);
        this.wk = MixerOps.weight(random, features, features);
        this.wv = MixerOps.weight(random, features, features);
        this.wo = MixerOps.weight(random, features, features);
        this.wEta = MixerOps.uniform(random, 0.01f, features, heads);
        int out1 = variant == Variant.LINEAR ? headSize : hidden;
        this.init1 = MixerOps.uniform(random, (float) (0.1 / Math.sqrt(headSize)), heads, headSize, out1);
        this.init2 = MixerOps.uniform(random, (float) (0.1 / Math.sqrt(hidden)), heads, hidden, headSize);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param heads the number of heads
     * @param variant the inner model
     * @param miniBatch tokens per inner step
     * @param baseRate the inner learning rate scale
     * @return the spec
     */
    public static LayerSpec of(int heads, Variant variant, int miniBatch, float baseRate) {
        return MixerOps.sameShape((in, r) -> new TestTimeTrainingLayer(in.dim(-1), heads, variant, miniBatch,
                baseRate, r));
    }

    /**
     * The inner reconstruction loss of each mini-batch in the last forward pass, measured
     * with the weights before that mini-batch's update (averaged over batch and heads). It
     * falls along a sequence as the inner model learns the sequence's structure.
     *
     * @return one loss per mini-batch
     */
    public float[] innerLosses() {
        return lastInnerLosses.clone();
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "TestTimeTrainingLayer");
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        Var q = MixerOps.unit(MixerOps.split(input.matmul(wq), heads));
        Var k = MixerOps.unit(MixerOps.split(input.matmul(wk), heads));
        Var v = MixerOps.split(input.matmul(wv), heads);
        Var eta = input.matmul(wEta).sigmoid().mul(baseRate).permute(0, 2, 1);   // [b, h, t]
        Var w1 = init1.expand(Shape.of(b, heads, headSize, init1.shape().dim(2)));
        Var w2 = init2.expand(Shape.of(b, heads, hidden, headSize));
        List<Var> outs = new ArrayList<>();
        List<Float> losses = new ArrayList<>();
        for (int start = 0; start < t; start += miniBatch) {
            int end = Math.min(t, start + miniBatch);
            int c = end - start;
            Var qc = q.slice(2, start, end);
            Var kc = k.slice(2, start, end);
            Var vc = v.slice(2, start, end);
            Var etaRow = eta.slice(2, start, end).reshape(b, heads, 1, c);   // eta_j along columns
            Var etaCol = eta.slice(2, start, end).reshape(b, heads, c, 1);
            Var mask = qc.matmul(kc.transpose()).mul(MixerOps.lowerOnes(c)).mul(etaRow);   // [b, h, c, c]
            if (variant == Variant.LINEAR) {
                Var err = kc.matmul(w1).sub(vc);                                // dl/dy / 2
                losses.add(err.square().sum(-1, false).mean().value().item());
                outs.add(qc.matmul(w1).sub(mask.matmul(err).mul(2f)));
                w1 = w1.sub(kc.mul(etaCol).transpose().matmul(err).mul(2f));
            } else {
                Var hk = kc.matmul(w1).tanh();                                  // [b, h, c, hidden]
                Var err = hk.matmul(w2).sub(vc);
                losses.add(err.square().sum(-1, false).mean().value().item());
                Var dy = err.mul(2f);
                Var da = dy.matmul(w2.transpose()).mul(hk.square().neg().add(1f));
                // z_i = tanh(q_i W1_i) W2_i in the dual form
                Var hq = qc.matmul(w1).sub(mask.matmul(da)).tanh();
                Var hMask = hq.matmul(hk.transpose()).mul(MixerOps.lowerOnes(c)).mul(etaRow);
                outs.add(hq.matmul(w2).sub(hMask.matmul(dy)));
                w1 = w1.sub(kc.mul(etaCol).transpose().matmul(da));
                w2 = w2.sub(hk.mul(etaCol).transpose().matmul(dy));
            }
        }
        float[] l = new float[losses.size()];
        for (int i = 0; i < l.length; i++) {
            l[i] = losses.get(i);
        }
        lastInnerLosses = l;
        Var z = outs.size() == 1 ? outs.getFirst() : Var.concat(2, outs.toArray(Var[]::new));
        return MixerOps.rmsNorm(MixerOps.merge(z)).matmul(wo);
    }

    @Override
    public List<Var> parameters() {
        return variant == Variant.LINEAR ? List.of(wq, wk, wv, wo, wEta, init1)
                : List.of(wq, wk, wv, wo, wEta, init1, init2);
    }

    @Override
    public String toString() {
        return "TestTimeTrainingLayer(" + features + ", " + variant + ", miniBatch=" + miniBatch + ")";
    }
}
