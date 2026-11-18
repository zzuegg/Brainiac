package io.github.zzuegg.brainiac.experimental.nlp;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * SimCSE sentence embeddings over a small convolutional encoder (token embeddings, one width-3
 * convolution with {@code tanh}, masked mean pooling).
 *
 * <ul>
 *   <li>{@link #trainUnsupervised}: each sentence is encoded twice with independent dropout
 *       masks and the two views are positives of an in-batch InfoNCE loss;
 *   <li>{@link #trainSupervised}: entailment-like positives, with a contradiction per anchor
 *       added as a hard negative to the softmax denominator;
 *   <li>{@link #trainMlm}: the masked-token baseline (predict 15% masked tokens from the
 *       convolution states), embeddings then mean-pooled.
 * </ul>
 *
 * <p>{@link #alignment} and {@link #uniformity} are Wang and Isola's diagnostics. Paper: Gao,
 * Yao, Chen, "SimCSE: Simple Contrastive Learning of Sentence Embeddings", EMNLP 2021,
 * <a href="https://arxiv.org/abs/2104.08821">arXiv:2104.08821</a>.
 */
public final class SimCse {

    private final int vocab;
    private final int dim;
    private final Var embedding;
    private final Var conv;
    private final Var convBias;
    private final Var mlmOut;
    private final float dropout;

    /**
     * Creates an encoder.
     *
     * @param vocab the vocabulary; id {@code vocab} pads and {@code vocab + 1} masks
     * @param dim the embedding width
     * @param dropout the dropout rate on token embeddings while training
     * @param seed the seed
     */
    public SimCse(int vocab, int dim, float dropout, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.vocab = vocab;
        this.dim = dim;
        this.dropout = dropout;
        this.embedding = Var.param(Tensor.normal(Shape.of(vocab + 2, dim), 0, 0.3f, r));
        this.conv = Var.param(Tensor.normal(Shape.of(dim, dim, 3), 0, (float) Math.sqrt(1.0 / (3 * dim)), r));
        this.convBias = Var.param(Tensor.zeros(Shape.of(dim)));
        this.mlmOut = Var.param(Tensor.normal(Shape.of(dim, vocab), 0, (float) Math.sqrt(1.0 / dim), r));
    }

    private static int width(List<int[]> s) {
        int t = 1;
        for (int[] x : s) {
            t = Math.max(t, x.length);
        }
        return t;
    }

    private Tensor ids(List<int[]> s, int t) {
        float[] v = new float[s.size() * t];
        for (int i = 0; i < s.size(); i++) {
            for (int k = 0; k < t; k++) {
                v[i * t + k] = k < s.get(i).length ? s.get(i)[k] : vocab;
            }
        }
        return Tensor.of(Shape.of(s.size() * t), v);
    }

    private static Tensor mask(List<int[]> s, int t) {
        float[] v = new float[s.size() * t];
        for (int i = 0; i < s.size(); i++) {
            for (int k = 0; k < Math.min(t, s.get(i).length); k++) {
                v[i * t + k] = 1f / s.get(i).length;
            }
        }
        return Tensor.of(Shape.of(s.size(), t, 1), v);
    }

    /** Token states {@code [b, t, dim]}. */
    private Var states(List<int[]> s, int t, Tensor tokens, SplittableRandom drop) {
        int b = s.size();
        Var x = embedding.gather(0, tokens).reshape(b, t, dim);
        if (drop != null && dropout > 0) {
            float[] m = new float[b * t * dim];
            for (int i = 0; i < m.length; i++) {
                m[i] = drop.nextDouble() < dropout ? 0f : 1f / (1f - dropout);
            }
            x = x.mul(Tensor.of(Shape.of(b, t, dim), m));
        }
        Var h = Functional.conv(x.permute(0, 2, 1), conv, convBias, new int[] {1}, new int[] {1}, new int[] {1},
                new int[] {1}, 1).tanh();
        return h.permute(0, 2, 1);
    }

    private Var pooled(List<int[]> s, SplittableRandom drop) {
        int t = width(s);
        return states(s, t, ids(s, t), drop).mul(mask(s, t)).sum(1, false);
    }

    private static Var unit(Var x) {
        return x.div(x.square().sum(1, true).add(1e-8f).sqrt());
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        return List.of(embedding, conv, convBias, mlmOut);
    }

    /**
     * Encodes sentences without dropout.
     *
     * @param sentences token ids
     * @return unit-length embeddings
     */
    public float[][] embed(List<int[]> sentences) {
        return rows(Var.noGrad(() -> unit(pooled(sentences, null))).value(), sentences.size());
    }

    /**
     * Mean of the (static) token embeddings, without the encoder.
     *
     * @param sentences token ids
     * @return unit-length embeddings
     */
    public float[][] staticEmbed(List<int[]> sentences) {
        int t = width(sentences);
        Var v = Var.noGrad(() -> unit(embedding.gather(0, ids(sentences, t)).reshape(sentences.size(), t, dim)
                .mul(mask(sentences, t)).sum(1, false)));
        return rows(v.value(), sentences.size());
    }

    private float[][] rows(Tensor t, int n) {
        float[] a = t.toArray();
        float[][] out = new float[n][dim];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a, i * dim, out[i], 0, dim);
        }
        return out;
    }

    private static List<int[]> pick(List<int[]> s, int[] idx) {
        List<int[]> out = new ArrayList<>();
        for (int i : idx) {
            out.add(s.get(i));
        }
        return out;
    }

    private static int[] batch(int n, int b, SplittableRandom r) {
        int[] idx = new int[b];
        for (int i = 0; i < b; i++) {
            idx[i] = r.nextInt(n);
        }
        return idx;
    }

    private static Tensor diagonal(int b) {
        float[] v = new float[b];
        for (int i = 0; i < b; i++) {
            v[i] = i;
        }
        return Tensor.of(Shape.of(b), v);
    }

    private void fit(java.util.function.IntFunction<Var> loss, int steps, float lr) {
        Optimizer opt = Optimizer.adam(lr);
        List<Var> p = parameters();
        for (int s = 0; s < steps; s++) {
            p.forEach(Var::zeroGrad);
            loss.apply(s).backward();
            opt.step(p);
        }
        p.forEach(Var::zeroGrad);
    }

    /**
     * Unsupervised SimCSE: two dropout views per sentence.
     *
     * @param sentences unlabelled sentences
     * @param steps steps
     * @param batch batch size
     * @param lr learning rate
     * @param temperature the softmax temperature
     * @param seed the seed
     */
    public void trainUnsupervised(List<int[]> sentences, int steps, int batch, float lr, float temperature,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        fit(s -> {
            List<int[]> b = pick(sentences, batch(sentences.size(), batch, r));
            Var a = unit(pooled(b, r));
            Var c = unit(pooled(b, r));
            return Functional.crossEntropy(a.matmul(c.transpose()).div(temperature), diagonal(batch));
        }, steps, lr);
    }

    /**
     * Supervised SimCSE with hard negatives.
     *
     * @param anchors anchors
     * @param positives a positive per anchor
     * @param negatives a hard negative per anchor
     * @param steps steps
     * @param batch batch size
     * @param lr learning rate
     * @param temperature the softmax temperature
     * @param seed the seed
     */
    public void trainSupervised(List<int[]> anchors, List<int[]> positives, List<int[]> negatives, int steps,
            int batch, float lr, float temperature, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        fit(s -> {
            int[] idx = batch(anchors.size(), batch, r);
            Var a = unit(pooled(pick(anchors, idx), r));
            Var p = unit(pooled(pick(positives, idx), r));
            Var n = unit(pooled(pick(negatives, idx), r));
            Var logits = a.matmul(Var.concat(0, p, n).transpose()).div(temperature);
            return Functional.crossEntropy(logits, diagonal(batch));
        }, steps, lr);
    }

    /**
     * Masked-token pre-training.
     *
     * @param sentences unlabelled sentences
     * @param steps steps
     * @param batch batch size
     * @param lr learning rate
     * @param seed the seed
     */
    public void trainMlm(List<int[]> sentences, int steps, int batch, float lr, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        fit(s -> {
            List<int[]> b = pick(sentences, batch(sentences.size(), batch, r));
            int t = width(b);
            float[] in = ids(b, t).toArray();
            float[] target = new float[in.length];
            boolean any = false;
            for (int i = 0; i < in.length; i++) {
                target[i] = -1;
                if (in[i] < vocab && r.nextDouble() < 0.15) {
                    target[i] = in[i];
                    in[i] = vocab + 1;
                    any = true;
                }
            }
            if (!any) {
                target[0] = in[0] < vocab ? in[0] : -1;
            }
            Var h = states(b, t, Tensor.of(Shape.of(in.length), in), r).reshape(b.size() * t, dim);
            return Functional.crossEntropy(h.matmul(mlmOut), Tensor.of(Shape.of(in.length), target));
        }, steps, lr);
    }

    /**
     * @param a embeddings
     * @param b their positives
     * @return the mean squared distance of positive pairs (lower is better)
     */
    public static double alignment(float[][] a, float[][] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += squared(a[i], b[i]);
        }
        return s / a.length;
    }

    /**
     * @param a unit embeddings
     * @return {@code log mean exp(-2 |x - y|^2)} over distinct pairs (lower is more uniform)
     */
    public static double uniformity(float[][] a) {
        double s = 0;
        int n = 0;
        for (int i = 0; i < a.length; i++) {
            for (int j = i + 1; j < a.length; j++) {
                s += Math.exp(-2 * squared(a[i], a[j]));
                n++;
            }
        }
        return Math.log(s / Math.max(1, n));
    }

    /**
     * @param a a vector
     * @param b another
     * @return the dot product
     */
    public static double dot(float[] a, float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static double squared(float[] a, float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return s;
    }
}
