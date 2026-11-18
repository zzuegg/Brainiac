package io.github.zzuegg.brainiac.experimental.anomaly;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Deep SVDD (Ruff et al., ICML 2018, <a href="https://proceedings.mlr.press/v80/ruff18a.html">PMLR 80</a>)
 * and its semi-supervised extension Deep SAD (Ruff et al., ICLR 2020,
 * <a href="https://arxiv.org/abs/1906.02694">arXiv:1906.02694</a>). A network {@code phi}
 * without bias terms (biases would let it map everything onto the centre, the trivial
 * "hypersphere collapse" solution) is trained to pull normal data towards a fixed centre
 * {@code c}, the mean of the initial embeddings: the loss is the mean of
 * {@code |phi(x) - c|^2}. Deep SAD adds the few labelled anomalies with the inverse term
 * {@code eta (|phi(x) - c|^2 + eps)^-1}, which pushes them away. The anomaly score is the
 * squared distance to {@code c}.
 */
public final class DeepSvdd {

    private final List<Var> weights = new ArrayList<>();
    private Tensor centre;

    /**
     * Creates the network: bias-free leaky-ReLU layers of the given widths, the last one linear.
     *
     * @param inputs the input features
     * @param widths the layer widths, the last being the embedding size
     * @param seed the initialisation seed
     */
    public DeepSvdd(int inputs, int[] widths, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int in = inputs;
        for (int w : widths) {
            weights.add(Var.param(Tensor.normal(Shape.of(in, w), 0, (float) Math.sqrt(2.0 / in), r)));
            in = w;
        }
    }

    /**
     * The embedding.
     *
     * @param x {@code [batch, inputs]}
     * @return {@code [batch, embedding]}
     */
    public Var embed(Var x) {
        Var h = x;
        for (int i = 0; i < weights.size(); i++) {
            h = h.matmul(weights.get(i));
            if (i + 1 < weights.size()) {
                h = h.maximum(h.mul(0.1f));
            }
        }
        return h;
    }

    /**
     * Pretrains the network as the encoder of a bias-free autoencoder (a mirrored decoder is
     * trained alongside and discarded), as the paper does before fixing the centre, so the
     * embedding starts informative rather than random.
     *
     * @param x training points {@code [n][d]}
     * @param steps Adam steps on minibatches of 64
     * @param learningRate the learning rate
     * @param seed the batching and decoder seed
     */
    public void pretrain(double[][] x, int steps, float learningRate, long seed) {
        int n = x.length;
        int d = x[0].length;
        SplittableRandom r = new SplittableRandom(seed);
        List<Var> decoder = new ArrayList<>();
        for (int i = weights.size() - 1; i >= 0; i--) {
            Shape s = weights.get(i).shape();
            decoder.add(Var.param(Tensor.normal(Shape.of(s.dim(1), s.dim(0)), 0,
                    (float) Math.sqrt(2.0 / s.dim(1)), r)));
        }
        List<Var> all = new ArrayList<>(weights);
        all.addAll(decoder);
        Optimizer opt = Optimizer.adam(learningRate);
        int b = Math.min(64, n);
        for (int s = 0; s < steps; s++) {
            float[] bx = new float[b * d];
            for (int i = 0; i < b; i++) {
                int k = r.nextInt(n);
                for (int j = 0; j < d; j++) {
                    bx[i * d + j] = (float) x[k][j];
                }
            }
            Tensor t = Tensor.of(Shape.of(b, d), bx);
            Var h = embed(Var.of(t));
            for (int i = 0; i < decoder.size(); i++) {
                h = h.matmul(decoder.get(i));
                if (i + 1 < decoder.size()) {
                    h = h.maximum(h.mul(0.1f));
                }
            }
            h.sub(t).square().mean().backward();
            opt.step(all);
        }
    }

    private Var squaredDistance(Var x) {
        return embed(x).sub(centre).square().sum(1, false);
    }

    /**
     * Trains on (mostly normal) data, optionally with a few labelled anomalies.
     *
     * @param x training points {@code [n][d]}
     * @param labelledAnomaly true for the labelled anomalies (Deep SAD), or {@code null} for
     *     plain one-class Deep SVDD
     * @param eta the weight of the labelled anomalies' inverse-distance term
     * @param steps Adam steps on minibatches of 64
     * @param learningRate the learning rate
     * @param seed the batching seed
     */
    public void fit(double[][] x, boolean[] labelledAnomaly, float eta, int steps, float learningRate, long seed) {
        int n = x.length;
        int d = x[0].length;
        float[] all = new float[n * d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                all[i * d + j] = (float) x[i][j];
            }
        }
        Tensor tx = Tensor.of(Shape.of(n, d), all);
        // the centre: the mean initial embedding, with near-zero coordinates moved off zero
        float[] c = Var.noGrad(() -> embed(Var.of(tx)).mean(0, false)).value().toArray();
        for (int j = 0; j < c.length; j++) {
            if (Math.abs(c[j]) < 0.1f) {
                c[j] = c[j] < 0 ? -0.1f : 0.1f;
            }
        }
        centre = Tensor.of(Shape.of(1, c.length), c);
        Optimizer opt = Optimizer.adamW(learningRate, 1e-4f);
        SplittableRandom r = new SplittableRandom(seed);
        int b = Math.min(64, n);
        for (int s = 0; s < steps; s++) {
            float[] bx = new float[b * d];
            float[] sign = new float[b];
            float[] normal = new float[b];
            for (int i = 0; i < b; i++) {
                int k = r.nextInt(n);
                System.arraycopy(all, k * d, bx, i * d, d);
                boolean anomalous = labelledAnomaly != null && labelledAnomaly[k];
                sign[i] = anomalous ? eta : 0;
                normal[i] = anomalous ? 0 : 1;
            }
            Var dist = squaredDistance(Var.of(Tensor.of(Shape.of(b, d), bx)));
            Var loss = dist.mul(Tensor.of(Shape.of(b), normal))
                    .add(dist.add(1e-3f).pow(-1f).mul(Tensor.of(Shape.of(b), sign))).mean();
            loss.backward();
            opt.step(weights);
        }
    }

    /**
     * Anomaly scores.
     *
     * @param x points {@code [m][d]}
     * @return squared distances to the centre, higher is more anomalous
     */
    public double[] score(double[][] x) {
        int m = x.length;
        int d = x[0].length;
        float[] all = new float[m * d];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < d; j++) {
                all[i * d + j] = (float) x[i][j];
            }
        }
        float[] v = Var.noGrad(() -> squaredDistance(Var.of(Tensor.of(Shape.of(m, d), all)))).value().toArray();
        double[] out = new double[m];
        for (int i = 0; i < m; i++) {
            out[i] = v[i];
        }
        return out;
    }

    /** @return the trainable weights */
    public List<Var> parameters() {
        return List.copyOf(weights);
    }
}
