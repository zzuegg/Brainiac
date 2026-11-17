package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Titans' neural memory as a trainable sequence layer (memory as a layer, "MAL", Behrouz et
 * al. 2025, <a href="https://arxiv.org/abs/2501.00663">arXiv:2501.00663</a>), with a linear
 * memory per head so the test-time gradient step has a closed form and the whole layer is
 * differentiable end to end. For each step, with L2-normalized key {@code k}, value
 * {@code v}, query {@code q} and data-dependent gates {@code theta, eta, alpha} in (0, 1):
 * <pre>
 * grad = k^T (k M - v)                      gradient of 1/2 ||k M - v||^2
 * S    = eta S - theta grad                 surprise with momentum
 * M    = (1 - alpha) M + S                  forgetting, then memorizing
 * y    = q M
 * </pre>
 * Without momentum this is the gated delta rule; momentum lets a surprising event keep
 * writing for several steps. The outer projections and gate projections train by
 * backpropagation through the sequence; see {@link TitansMemory} for a deeper,
 * inference-only memory. Input {@code [batch, steps, features]}, output the same shape.
 */
public final class TitansLayer implements Layer {

    private final int features;
    private final int heads;
    private final int headSize;
    private final Var wk;
    private final Var wv;
    private final Var wq;
    private final Var wo;
    private final Var wGates;
    private final Var bGates;
    private boolean momentum = true;

    /**
     * Creates the layer.
     *
     * @param features the input and output features
     * @param heads the heads
     * @param headSize the key and value size per head
     * @param random the random source for the weights
     */
    public TitansLayer(int features, int heads, int headSize, SplittableRandom random) {
        this.features = features;
        this.heads = heads;
        this.headSize = headSize;
        int hd = heads * headSize;
        wk = Var.param(Init.XAVIER.create(Shape.of(features, hd), features, hd, random));
        wv = Var.param(Init.XAVIER.create(Shape.of(features, hd), features, hd, random));
        wq = Var.param(Init.XAVIER.create(Shape.of(features, hd), features, hd, random));
        wo = Var.param(Init.XAVIER.create(Shape.of(hd, features), hd, features, random));
        wGates = Var.param(Tensor.normal(Shape.of(features, 3 * heads), 0, 0.01f, random));
        // start with theta ~ 0.5, eta ~ 0.27, alpha ~ 0.05
        float[] b = new float[3 * heads];
        for (int h = 0; h < heads; h++) {
            b[h] = 0;
            b[heads + h] = -1;
            b[2 * heads + h] = -3;
        }
        bGates = Var.param(Tensor.of(Shape.of(3 * heads), b));
    }

    /**
     * Describes the layer for model builders, on inputs {@code [steps, features]}.
     *
     * @param heads the heads
     * @param headSize the key and value size per head
     * @return the spec
     */
    public static LayerSpec spec(int heads, int headSize) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new TitansLayer(input.dim(-1), heads, headSize, random);
            }
        };
    }

    /**
     * @param on whether surprise carries momentum; off gives the gated delta rule
     * @return this layer
     */
    public TitansLayer momentum(boolean on) {
        this.momentum = on;
        return this;
    }

    private Var split(Var x, Var w, int n, int t, boolean normalize) {
        Var y = x.matmul(w).reshape(n, t, heads, headSize).permute(0, 2, 1, 3);
        return normalize ? y.div(y.square().sum(-1, true).add(1e-6f).sqrt()) : y;
    }

    @Override
    public Var forward(Var input) {
        int n = input.shape().dim(0);
        int t = input.shape().dim(1);
        if (input.shape().rank() != 3 || input.shape().dim(2) != features) {
            throw new IllegalArgumentException(this + " expects [batch, steps, " + features + "], got " + input.shape());
        }
        Var k = split(input, wk, n, t, true);
        Var v = split(input, wv, n, t, false);
        Var q = split(input, wq, n, t, true);
        Var g = input.matmul(wGates).add(bGates).sigmoid();   // [n, t, 3h]
        Var theta = g.slice(2, 0, heads).permute(0, 2, 1).reshape(n, heads, t, 1);
        Var eta = g.slice(2, heads, 2 * heads).permute(0, 2, 1).reshape(n, heads, t, 1);
        Var alpha = g.slice(2, 2 * heads, 3 * heads).permute(0, 2, 1).reshape(n, heads, t, 1);
        Var memory = Var.of(Tensor.zeros(Shape.of(n, heads, headSize, headSize)));
        Var surprise = memory;
        Var[] outs = new Var[t];
        for (int s = 0; s < t; s++) {
            Var ks = k.slice(2, s, s + 1);
            Var grad = ks.transpose().matmul(ks.matmul(memory).sub(v.slice(2, s, s + 1)));
            Var th = theta.slice(2, s, s + 1);
            Var update = grad.mul(th).neg();
            surprise = momentum ? surprise.mul(eta.slice(2, s, s + 1)).add(update) : update;
            memory = memory.mul(alpha.slice(2, s, s + 1).neg().add(1f)).add(surprise);
            outs[s] = q.slice(2, s, s + 1).matmul(memory);
        }
        Var y = Var.concat(2, outs);
        return y.permute(0, 2, 1, 3).reshape(n, t, heads * headSize).matmul(wo);
    }

    @Override
    public List<Var> parameters() {
        return List.of(wk, wv, wq, wo, wGates, bGates);
    }

    @Override
    public String toString() {
        return "TitansLayer(" + heads + " heads" + (momentum ? "" : ", no momentum") + ")";
    }
}
