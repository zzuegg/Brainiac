package io.github.zzuegg.brainiac.experimental.anomaly;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Dinomaly: reconstruction of frozen encoder features for multi-class anomaly detection. The
 * concatenated features of several encoder layers pass a bottleneck MLP whose dropout is the
 * noise that stops identity mapping; a decoder of attention blocks reconstructs every encoder
 * layer; the anomaly map is the mean cosine distance between encoder and decoder features.
 *
 * <p>The {@link Config} switches the three ingredients: linear attention (which "cannot focus"
 * and so does not copy local anomalies) or softmax attention; bottleneck dropout or none; and
 * the loose loss (one cosine distance per token between the concatenation of all encoder
 * layers and all decoder layers, with hard-point mining: tokens below the batch's
 * {@code 1 - hardShare} quantile of distance have their loss scaled by 0.1) or the layer-wise
 * point-wise loss.
 *
 * <p>Input features are {@code [tokens][dims]} per layer, per image.
 *
 * <p>Paper: Guo et al., "Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised
 * Anomaly Detection", CVPR 2025, <a href="https://arxiv.org/abs/2405.14325">arXiv:2405.14325</a>.
 */
public final class Dinomaly {

    /**
     * The ingredients.
     *
     * @param linearAttention linear attention instead of softmax attention
     * @param dropout the bottleneck dropout (0 for none)
     * @param looseLoss grouped loss with hard-point mining instead of layer-wise loss
     * @param hardShare the share of hardest tokens kept at full weight
     */
    public record Config(boolean linearAttention, float dropout, boolean looseLoss, double hardShare) {

        /** @return the full method */
        public static Config dinomaly() {
            return new Config(true, 0.2f, true, 0.1);
        }

        /** @return the ablation: softmax attention, no dropout, layer-wise loss */
        public static Config plain() {
            return new Config(false, 0f, false, 1.0);
        }
    }

    private final Config config;
    private final int[] layerDims;
    private final int width;
    private final int blocks;
    private final int heads;
    private final List<Var> params = new ArrayList<>();
    private final Var in1;
    private final Var in2;
    private final Var[] qkv;
    private final Var[] proj;
    private final Var[] mlp1;
    private final Var[] mlp2;
    private final Var[] heads2;
    private final SplittableRandom noise;

    /**
     * Creates a model.
     *
     * @param layerDims the encoder layers' widths
     * @param width the bottleneck and decoder width
     * @param blocks decoder blocks
     * @param heads attention heads
     * @param config the ingredients
     * @param seed the seed
     */
    public Dinomaly(int[] layerDims, int width, int blocks, int heads, Config config, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.config = config;
        this.layerDims = layerDims.clone();
        this.width = width;
        this.blocks = blocks;
        this.heads = heads;
        this.noise = r.split();
        int d = Arrays.stream(layerDims).sum();
        in1 = param(r, d, width);
        in2 = param(r, width, width);
        qkv = new Var[blocks];
        proj = new Var[blocks];
        mlp1 = new Var[blocks];
        mlp2 = new Var[blocks];
        for (int b = 0; b < blocks; b++) {
            qkv[b] = param(r, width, 3 * width);
            proj[b] = param(r, width, width);
            mlp1[b] = param(r, width, 2 * width);
            mlp2[b] = param(r, 2 * width, width);
        }
        heads2 = new Var[layerDims.length];
        for (int l = 0; l < layerDims.length; l++) {
            heads2[l] = param(r, width, layerDims[l]);
        }
    }

    private Var param(SplittableRandom r, int in, int out) {
        Var v = Var.param(Tensor.normal(Shape.of(in, out), 0, (float) Math.sqrt(1.0 / in), r));
        params.add(v);
        return v;
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        return List.copyOf(params);
    }

    private Var dropout(Var x, boolean train) {
        if (!train || config.dropout() <= 0) {
            return x;
        }
        int n = (int) x.shape().size();
        float[] m = new float[n];
        float keep = 1 - config.dropout();
        for (int i = 0; i < n; i++) {
            m[i] = noise.nextDouble() < config.dropout() ? 0 : 1 / keep;
        }
        return x.mul(Tensor.of(x.shape(), m));
    }

    private static Var input(List<List<float[][]>> layers, int[] batch) {
        int tokens = layers.get(0).get(0).length;
        int d = 0;
        for (List<float[][]> l : layers) {
            d += l.get(0)[0].length;
        }
        float[] v = new float[batch.length * tokens * d];
        int k = 0;
        for (int i : batch) {
            for (int t = 0; t < tokens; t++) {
                for (List<float[][]> l : layers) {
                    float[] f = l.get(i)[t];
                    System.arraycopy(f, 0, v, k, f.length);
                    k += f.length;
                }
            }
        }
        return Var.of(Tensor.of(Shape.of(batch.length, tokens, d), v));
    }

    /** Decoder outputs per layer, {@code [b, tokens, dims_l]}. */
    private Var[] decode(Var x, boolean train) {
        int b = x.shape().dim(0);
        int t = x.shape().dim(1);
        Var h = dropout(x.matmul(in1).relu(), train);
        h = dropout(h.matmul(in2), train);
        int dk = width / heads;
        for (int k = 0; k < blocks; k++) {
            Var n = h.normalize(2, 1e-5f);
            Var all = n.matmul(qkv[k]);
            Var q = all.slice(2, 0, width).reshape(b, t, heads, dk).permute(0, 2, 1, 3);
            Var kk = all.slice(2, width, 2 * width).reshape(b, t, heads, dk).permute(0, 2, 1, 3);
            Var v = all.slice(2, 2 * width, 3 * width).reshape(b, t, heads, dk).permute(0, 2, 1, 3);
            Var a = config.linearAttention() ? Functional.linearAttention(q, kk, v, false)
                    : Functional.attention(q, kk, v, null, (float) (1 / Math.sqrt(dk)), 0f, null);
            h = h.add(a.permute(0, 2, 1, 3).reshape(b, t, width).matmul(proj[k]));
            h = h.add(h.normalize(2, 1e-5f).matmul(mlp1[k]).relu().matmul(mlp2[k]));
        }
        Var[] out = new Var[layerDims.length];
        for (int l = 0; l < layerDims.length; l++) {
            out[l] = h.matmul(heads2[l]);
        }
        return out;
    }

    private static Var cosineDistance(Var a, Var b) {
        Var na = a.div(a.square().sum(2, true).add(1e-8f).sqrt());
        Var nb = b.div(b.square().sum(2, true).add(1e-8f).sqrt());
        return na.mul(nb).sum(2, false).mul(-1f).add(1f);
    }

    private Var[] targets(Var x) {
        Var[] t = new Var[layerDims.length];
        int off = 0;
        for (int l = 0; l < layerDims.length; l++) {
            t[l] = x.slice(2, off, off + layerDims[l]);
            off += layerDims[l];
        }
        return t;
    }

    /**
     * Trains on normal images.
     *
     * @param layers per encoder layer, per image its features {@code [tokens][dims]}
     * @param steps steps
     * @param batch batch size
     * @param lr the learning rate
     * @param seed the seed
     */
    public void train(List<List<float[][]>> layers, int steps, int batch, float lr, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(lr);
        int n = layers.get(0).size();
        for (int s = 0; s < steps; s++) {
            int[] idx = new int[batch];
            for (int i = 0; i < batch; i++) {
                idx[i] = r.nextInt(n);
            }
            Var x = input(layers, idx);
            Var[] y = decode(x, true);
            Var[] t = targets(x);
            Var loss;
            if (config.looseLoss()) {
                Var d = cosineDistance(Var.concat(2, t), Var.concat(2, y));     // [b, tokens]
                float[] dv = d.value().toArray();
                float[] sorted = dv.clone();
                Arrays.sort(sorted);
                float thr = sorted[(int) Math.min(sorted.length - 1, Math.floor((1 - config.hardShare()) * sorted.length))];
                float[] w = new float[dv.length];
                for (int i = 0; i < dv.length; i++) {
                    w[i] = dv[i] >= thr ? 1f : 0.1f;
                }
                loss = d.mul(Tensor.of(d.shape(), w)).mean();
            } else {
                loss = null;
                for (int l = 0; l < t.length; l++) {
                    Var dl = cosineDistance(t[l], y[l]).mean();
                    loss = loss == null ? dl : loss.add(dl);
                }
                loss = loss.div(t.length);
            }
            params.forEach(Var::zeroGrad);
            loss.backward();
            opt.step(params);
        }
        params.forEach(Var::zeroGrad);
    }

    /**
     * Anomaly maps: per token the mean over layers of the cosine distance.
     *
     * @param layers per encoder layer, per image its features
     * @param image the image index
     * @return the token scores
     */
    public double[] map(List<List<float[][]>> layers, int image) {
        return Var.noGrad(() -> {
            Var x = input(layers, new int[] {image});
            Var[] y = decode(x, false);
            Var[] t = targets(x);
            Var sum = null;
            for (int l = 0; l < t.length; l++) {
                Var d = cosineDistance(t[l], y[l]);
                sum = sum == null ? d : sum.add(d);
            }
            float[] v = sum.div(t.length).value().toArray();
            double[] out = new double[v.length];
            for (int i = 0; i < v.length; i++) {
                out[i] = v[i];
            }
            return out;
        });
    }
}
