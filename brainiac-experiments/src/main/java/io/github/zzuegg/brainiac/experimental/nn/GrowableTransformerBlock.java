package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A causal transformer block whose design is fully configurable and whose size can grow
 * while it trains without changing what it computes.
 *
 * <p><b>Design genes</b> ({@link Config}): query heads and key/value heads (grouped-query
 * attention), an optional sigmoid output gate per attention channel (gated attention),
 * the placement of normalization (pre-norm, post-norm, "sandwich"/peri-norm around each
 * sub-layer, or none), RMS or layer norm, and the FFN width with an activation per hidden
 * neuron ({@link Activation#perUnit}).
 *
 * <p><b>Function-preserving growth</b> (Net2Net, Chen et al. 2016,
 * <a href="https://arxiv.org/abs/1511.05641">arXiv:1511.05641</a>):
 * <ul>
 *   <li>{@link #widenFfn}: new hidden neurons copy existing ones, and each copied neuron's
 *       outgoing weights are split between it and its copies in unequal random shares that
 *       add up to the original, so the output is unchanged while the copies receive
 *       different gradients and diverge;</li>
 *   <li>{@link #addHeadGroup}: a new key/value head with its query heads; their output
 *       projection rows start at zero, so they contribute nothing until trained.</li>
 *   <li>{@link #identity}: a block whose output projections are zero is the identity on
 *       the residual stream, so depth can grow too (see {@link GrowableTransformer}).</li>
 * </ul>
 *
 * <p>Input and output are {@code [batch, time, model]}.
 */
public final class GrowableTransformerBlock implements Layer {

    /** Where normalization sits. */
    public enum NormPlacement {
        /** {@code x + f(norm(x))}: stable, the modern default */
        PRE,
        /** {@code norm(x + f(x))}: the original transformer */
        POST,
        /** {@code x + norm(f(norm(x)))}: peri-LN, normalizes both ends of each sub-layer */
        SANDWICH,
        /** {@code x + f(x)}: no normalization */
        NONE
    }

    /**
     * The design of a block.
     *
     * @param model the residual width
     * @param heads query heads
     * @param kvHeads key/value heads, dividing {@code heads}
     * @param headDim the width of a head
     * @param gated whether attention outputs pass a sigmoid gate computed from the input
     * @param norm where normalization sits
     * @param rms RMS norm instead of layer norm
     * @param ffnActivations one activation per FFN hidden neuron (its length is the FFN width)
     */
    public record Config(int model, int heads, int kvHeads, int headDim, boolean gated, NormPlacement norm,
            boolean rms, List<Activation> ffnActivations) {
        /** Checks the shape constraints. */
        public Config {
            ffnActivations = List.copyOf(ffnActivations);
            if (model < 1 || heads < 1 || kvHeads < 1 || headDim < 1 || ffnActivations.isEmpty()) {
                throw new IllegalArgumentException("sizes must be positive");
            }
            if (heads % kvHeads != 0) {
                throw new IllegalArgumentException("heads " + heads + " must be a multiple of kvHeads " + kvHeads);
            }
            Objects.requireNonNull(norm);
        }

        /**
         * A plain pre-norm block with one activation for the whole FFN.
         *
         * @param model the residual width
         * @param heads attention heads
         * @param headDim the head width
         * @param ffn the FFN width
         * @param activation the FFN activation
         * @return the config
         */
        public static Config of(int model, int heads, int headDim, int ffn, Activation activation) {
            return new Config(model, heads, heads, headDim, false, NormPlacement.PRE, true,
                    java.util.Collections.nCopies(ffn, activation));
        }

        /** @return the FFN width */
        public int ffn() {
            return ffnActivations.size();
        }
    }

    private Config config;
    private Var wq;
    private Var wk;
    private Var wv;
    private Var wo;
    private Var wg;
    private Var w1;
    private Var b1;
    private Var w2;
    private Var b2;
    private final Var[] gammas = new Var[4];
    private final Var[] betas = new Var[4];
    private Activation activation;

    private GrowableTransformerBlock(Config config) {
        this.config = config;
    }

    /**
     * Creates a block with random weights.
     *
     * @param config the design
     * @param random the random source
     * @return the block
     */
    public static GrowableTransformerBlock of(Config config, SplittableRandom random) {
        GrowableTransformerBlock b = new GrowableTransformerBlock(config);
        int d = config.model();
        int qd = config.heads() * config.headDim();
        int kd = config.kvHeads() * config.headDim();
        int f = config.ffn();
        b.wq = param(random(d, qd, random, 1.0 / Math.sqrt(d)), d, qd);
        b.wk = param(random(d, kd, random, 1.0 / Math.sqrt(d)), d, kd);
        b.wv = param(random(d, kd, random, 1.0 / Math.sqrt(d)), d, kd);
        b.wo = param(random(qd, d, random, 1.0 / Math.sqrt(qd)), qd, d);
        if (config.gated()) {
            b.wg = param(random(d, qd, random, 1.0 / Math.sqrt(d)), d, qd);
        }
        b.w1 = param(random(d, f, random, 1.0 / Math.sqrt(d)), d, f);
        b.b1 = Var.param(Tensor.zeros(Shape.of(f)));
        b.w2 = param(random(f, d, random, 1.0 / Math.sqrt(f)), f, d);
        b.b2 = Var.param(Tensor.zeros(Shape.of(d)));
        b.initNorms();
        b.activation = activationOf(config);
        return b;
    }

    /**
     * A block that adds nothing to the residual stream (zero output projections), for
     * function-preserving depth growth. Needs a placement with a residual around each
     * sub-layer ({@code PRE}, {@code SANDWICH} or {@code NONE}).
     *
     * @param config the design
     * @param random the random source for the other weights
     * @return the block
     */
    public static GrowableTransformerBlock identity(Config config, SplittableRandom random) {
        if (config.norm() == NormPlacement.POST) {
            throw new IllegalArgumentException("a post-norm block is never the identity");
        }
        GrowableTransformerBlock b = of(config, random);
        b.wo = Var.param(Tensor.zeros(b.wo.shape()));
        b.w2 = Var.param(Tensor.zeros(b.w2.shape()));
        return b;
    }

    /**
     * A spec for {@code Model.sequential().add(...)}.
     *
     * @param config the design; its model width must match the input's last dimension
     * @return the spec
     */
    public static LayerSpec spec(Config config) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                if (input.dim(-1) != config.model()) {
                    throw new IllegalArgumentException("input width " + input.dim(-1) + " does not match " + config);
                }
                return of(config, random);
            }
        };
    }

    private void initNorms() {
        int d = config.model();
        Arrays.fill(gammas, null);
        Arrays.fill(betas, null);
        int count = switch (config.norm()) {
            case PRE, POST -> 2;
            case SANDWICH -> 4;
            case NONE -> 0;
        };
        for (int i = 0; i < count; i++) {
            gammas[i] = Var.param(Tensor.ones(Shape.of(d)));
            if (!config.rms()) {
                betas[i] = Var.param(Tensor.zeros(Shape.of(d)));
            }
        }
    }

    private static Activation activationOf(Config c) {
        Activation a = Activation.perUnit(c.ffnActivations().toArray(Activation[]::new));
        return a.boundWidth() == c.ffn() || a.boundWidth() < 0 ? a : a.bind(c.ffn());
    }

    private static float[] random(int rows, int cols, SplittableRandom r, double scale) {
        float[] w = new float[rows * cols];
        for (int i = 0; i < w.length; i++) {
            w[i] = (float) (r.nextGaussian() * scale);
        }
        return w;
    }

    private static Var param(float[] values, int rows, int cols) {
        return Var.param(Tensor.of(Shape.of(rows, cols), values));
    }

    /** @return the current design */
    public Config config() {
        return config;
    }

    private Var norm(Var x, int i) {
        if (gammas[i] == null) {
            return x;
        }
        Shape s = x.shape();
        int d = s.dim(-1);
        Var r = x.reshape(Math.toIntExact(s.size() / d), d);
        if (!config.rms()) {
            r = r.sub(r.mean(1, true));
        }
        r = r.div(r.square().mean(1, true).add(1e-5f).sqrt()).mul(gammas[i]);
        if (betas[i] != null) {
            r = r.add(betas[i]);
        }
        return r.reshape(s.toArray());
    }

    private Var attention(Var x) {
        Shape s = x.shape();
        int b = s.dim(0);
        int t = s.dim(1);
        int h = config.heads();
        int kv = config.kvHeads();
        int hd = config.headDim();
        Var q = x.matmul(wq).reshape(b, t, h, hd).permute(0, 2, 1, 3);
        Var k = x.matmul(wk).reshape(b, t, kv, hd).permute(0, 2, 1, 3);
        Var v = x.matmul(wv).reshape(b, t, kv, hd).permute(0, 2, 1, 3);
        Var a = Functional.attention(q, k, v, true, (float) (1 / Math.sqrt(hd)));
        Var merged = a.permute(0, 2, 1, 3).reshape(b, t, h * hd);
        if (wg != null) {
            merged = merged.mul(x.matmul(wg).sigmoid());
        }
        return merged.matmul(wo);
    }

    private Var ffn(Var x) {
        return activation.forward(x.matmul(w1).add(b1)).matmul(w2).add(b2);
    }

    @Override
    public Var forward(Var x) {
        return switch (config.norm()) {
            case PRE -> {
                Var h = x.add(attention(norm(x, 0)));
                yield h.add(ffn(norm(h, 1)));
            }
            case POST -> {
                Var h = norm(x.add(attention(x)), 0);
                yield norm(h.add(ffn(h)), 1);
            }
            case SANDWICH -> {
                Var h = x.add(norm(attention(norm(x, 0)), 2));
                yield h.add(norm(ffn(norm(h, 1)), 3));
            }
            case NONE -> {
                Var h = x.add(attention(x));
                yield h.add(ffn(h));
            }
        };
    }

    @Override
    public List<Var> parameters() {
        List<Var> out = new ArrayList<>(List.of(wq, wk, wv, wo));
        if (wg != null) {
            out.add(wg);
        }
        out.addAll(List.of(w1, b1, w2, b2));
        for (int i = 0; i < 4; i++) {
            if (gammas[i] != null) {
                out.add(gammas[i]);
            }
            if (betas[i] != null) {
                out.add(betas[i]);
            }
        }
        out.addAll(activation.parameters());
        return out;
    }

    @Override
    public List<Var> buffers() {
        return activation.buffers();
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    // ------------------------------------------------------------------ growth

    private static float[] values(Var v) {
        return v.value().to(DType.FLOAT32).toArray();
    }

    /**
     * Grows the FFN to a new width, keeping the function: each new neuron copies a random
     * existing one (same incoming weights, bias and activation), and the copied neuron's
     * outgoing weights are shared between the copies in random proportions summing to 1.
     *
     * @param width the new width, at least the current one
     * @param random the random source
     */
    public void widenFfn(int width, SplittableRandom random) {
        int f = config.ffn();
        int d = config.model();
        if (width < f) {
            throw new IllegalArgumentException("can only grow: " + width + " < " + f);
        }
        if (width == f) {
            return;
        }
        int[] source = new int[width];
        for (int j = 0; j < width; j++) {
            source[j] = j < f ? j : random.nextInt(f);
        }
        // shares of each original neuron's outgoing weights among its copies
        double[] share = new double[width];
        double[] total = new double[f];
        for (int j = 0; j < width; j++) {
            share[j] = 0.5 + random.nextDouble();
            total[source[j]] += share[j];
        }
        float[] w1v = values(w1);
        float[] b1v = values(b1);
        float[] w2v = values(w2);
        float[] nw1 = new float[d * width];
        float[] nb1 = new float[width];
        float[] nw2 = new float[width * d];
        List<Activation> acts = new ArrayList<>();
        for (int j = 0; j < width; j++) {
            int s = source[j];
            for (int i = 0; i < d; i++) {
                nw1[i * width + j] = w1v[i * f + s];
            }
            nb1[j] = b1v[s];
            double frac = share[j] / total[s];
            for (int i = 0; i < d; i++) {
                nw2[j * d + i] = (float) (w2v[s * d + i] * frac);
            }
            acts.add(config.ffnActivations().get(s));
        }
        w1 = param(nw1, d, width);
        b1 = Var.param(Tensor.of(Shape.of(width), nb1));
        w2 = param(nw2, width, d);
        config = new Config(d, config.heads(), config.kvHeads(), config.headDim(), config.gated(), config.norm(),
                config.rms(), acts);
        activation = activationOf(config);
    }

    /**
     * Adds one key/value head with its {@code heads / kvHeads} query heads, keeping the
     * function: the new heads' rows of the output projection start at zero.
     *
     * @param random the random source for the new heads' input projections
     */
    public void addHeadGroup(SplittableRandom random) {
        int d = config.model();
        int hd = config.headDim();
        int group = config.heads() / config.kvHeads();
        int newHeads = config.heads() + group;
        int newKv = config.kvHeads() + 1;
        double scale = 1.0 / Math.sqrt(d);
        wq = appendColumns(wq, d, group * hd, random, scale);
        wk = appendColumns(wk, d, hd, random, scale);
        wv = appendColumns(wv, d, hd, random, scale);
        if (wg != null) {
            wg = appendColumns(wg, d, group * hd, random, scale);
        }
        float[] wo0 = values(wo);
        float[] nwo = Arrays.copyOf(wo0, newHeads * hd * d);   // new rows are zero
        wo = param(nwo, newHeads * hd, d);
        config = new Config(d, newHeads, newKv, hd, config.gated(), config.norm(), config.rms(),
                config.ffnActivations());
    }

    private static Var appendColumns(Var w, int rows, int extra, SplittableRandom r, double scale) {
        float[] old = values(w);
        int cols = w.shape().dim(1);
        int n = cols + extra;
        float[] out = new float[rows * n];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(old, i * cols, out, i * n, cols);
            for (int j = cols; j < n; j++) {
                out[i * n + j] = (float) (r.nextGaussian() * scale);
            }
        }
        return param(out, rows, n);
    }

    /**
     * A deep copy of this block, for progressive stacking.
     *
     * @return the copy
     */
    public GrowableTransformerBlock copy() {
        GrowableTransformerBlock b = new GrowableTransformerBlock(config);
        b.wq = Var.param(wq.value());
        b.wk = Var.param(wk.value());
        b.wv = Var.param(wv.value());
        b.wo = Var.param(wo.value());
        b.wg = wg == null ? null : Var.param(wg.value());
        b.w1 = Var.param(w1.value());
        b.b1 = Var.param(b1.value());
        b.w2 = Var.param(w2.value());
        b.b2 = Var.param(b2.value());
        for (int i = 0; i < 4; i++) {
            b.gammas[i] = gammas[i] == null ? null : Var.param(gammas[i].value());
            b.betas[i] = betas[i] == null ? null : Var.param(betas[i].value());
        }
        b.activation = activationOf(config);
        return b;
    }

    /**
     * How much the loss wants each part to grow: the root-mean-square gradient of the
     * attention output projection and of the FFN output projection, read after a backward
     * pass (0 when there is no gradient).
     *
     * @return {@code [attention, ffn]}
     */
    public double[] growthPressure() {
        return new double[] {rms(wo.grad()), rms(w2.grad())};
    }

    private static double rms(Tensor g) {
        if (g == null) {
            return 0;
        }
        return Math.sqrt(g.square().mean().item());
    }

    @Override
    public String toString() {
        return "GrowableTransformerBlock(" + config.heads() + "/" + config.kvHeads() + " heads x " + config.headDim()
                + ", ffn " + config.ffn() + ", " + config.norm() + (config.rms() ? " rms" : " layer")
                + (config.gated() ? ", gated" : "") + ")";
    }
}
