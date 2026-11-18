package io.github.zzuegg.brainiac.experimental.align;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.PreferenceLosses;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.List;
import java.util.Objects;

/**
 * General preference models with preference embeddings (Zhang et al. 2024,
 * <a href="https://arxiv.org/abs/2410.02197">arXiv:2410.02197</a>). A scalar reward (the
 * Bradley-Terry model) can represent only transitive preferences: if {@code a > b} and
 * {@code b > c} then {@code r(a) > r(c)}. The general preference model embeds each response
 * once as a vector {@code v(y)} in {@code 2k} dimensions and scores
 * {@code P(a > b) = sigmoid(<R v(a), v(b)>)} with {@code R} the fixed block-diagonal
 * skew-symmetric matrix of {@code k} 2x2 rotations {@code [[0, -1], [1, 0]]}. Skew symmetry
 * guarantees {@code P(a > b) + P(b > a) = 1} by construction, cycles such as rock, paper,
 * scissors become representable (three embeddings at 120 degrees in one rotation plane), and
 * scoring {@code n} candidates costs {@code n} embeddings rather than the {@code n^2} forward
 * passes of a pairwise classifier.
 *
 * <p>The same class provides the two baselines with the same feature network:
 * {@link Kind#BRADLEY_TERRY} (a scalar head, {@code sigmoid(r(a) - r(b))}) and
 * {@link Kind#PAIRWISE} (a classifier on the concatenated features, antisymmetrized as
 * {@code f(a, b) - f(b, a)}, the quadratic-cost upper bound). Inputs are feature vectors of
 * the responses (and their prompt), so any encoder fits in front.
 */
public final class PreferenceEmbeddingModel {

    /** The model family. */
    public enum Kind {
        /** Preference embeddings with a skew-symmetric bilinear score. */
        EMBEDDING,
        /** A scalar reward per response (Bradley-Terry). */
        BRADLEY_TERRY,
        /** A classifier over both responses at once. */
        PAIRWISE
    }

    private final Kind kind;
    private final int features;
    private final int dims;
    private final Model net;
    private final Tensor rotation;
    private Optimizer optimizer = Optimizer.adam(1e-2f);

    private PreferenceEmbeddingModel(Kind kind, int features, int hidden, int k, long seed) {
        this.kind = Objects.requireNonNull(kind);
        this.features = features;
        this.dims = kind == Kind.EMBEDDING ? 2 * k : 1;
        int in = kind == Kind.PAIRWISE ? 2 * features : features;
        this.net = Model.sequential().input(in).dense(hidden, Activation.TANH).dense(dims).seed(seed).build();
        float[] r = new float[dims * dims];
        if (kind == Kind.EMBEDDING) {
            // (v R^T)_j = (R v)_j: R has R[2i+1][2i] = 1 and R[2i][2i+1] = -1
            for (int i = 0; i < k; i++) {
                r[(2 * i) * dims + 2 * i + 1] = 1;
                r[(2 * i + 1) * dims + 2 * i] = -1;
            }
        }
        this.rotation = Tensor.of(Shape.of(dims, dims), r);
    }

    /**
     * @param features the length of a response's feature vector
     * @param hidden the hidden width of the embedding network
     * @param k rotation planes (the embedding has {@code 2k} dimensions)
     * @param seed the initialization seed
     * @return a general preference model
     */
    public static PreferenceEmbeddingModel embedding(int features, int hidden, int k, long seed) {
        return new PreferenceEmbeddingModel(Kind.EMBEDDING, features, hidden, k, seed);
    }

    /**
     * @param features the length of a response's feature vector
     * @param hidden the hidden width
     * @param seed the initialization seed
     * @return a Bradley-Terry (scalar reward) model with the same network
     */
    public static PreferenceEmbeddingModel bradleyTerry(int features, int hidden, long seed) {
        return new PreferenceEmbeddingModel(Kind.BRADLEY_TERRY, features, hidden, 1, seed);
    }

    /**
     * @param features the length of a response's feature vector
     * @param hidden the hidden width
     * @param seed the initialization seed
     * @return a pairwise classifier on concatenated features
     */
    public static PreferenceEmbeddingModel pairwise(int features, int hidden, long seed) {
        return new PreferenceEmbeddingModel(Kind.PAIRWISE, features, hidden, 1, seed);
    }

    /** @return the model family */
    public Kind kind() {
        return kind;
    }

    /**
     * @param o the optimizer
     * @return this model
     */
    public PreferenceEmbeddingModel optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        return net.parameters();
    }

    private static Tensor rows(float[][] x) {
        int d = x[0].length;
        float[] v = new float[x.length * d];
        for (int i = 0; i < x.length; i++) {
            System.arraycopy(x[i], 0, v, i * d, d);
        }
        return Tensor.of(Shape.of(x.length, d), v);
    }

    /**
     * The embeddings {@code v(y)} (or the scalar rewards of the Bradley-Terry model).
     *
     * @param x feature vectors {@code [n][features]}
     * @return {@code [n, 2k]} (or {@code [n, 1]})
     */
    public Var embed(float[][] x) {
        if (kind == Kind.PAIRWISE) {
            throw new IllegalStateException("a pairwise classifier has no per-response embedding");
        }
        return net.forward(Var.of(rows(x)), true);
    }

    /**
     * The preference logits {@code log P(a > b) / P(b > a)}.
     *
     * @param a features of the first responses
     * @param b features of the second responses
     * @return {@code [n]}, antisymmetric in {@code a} and {@code b}
     */
    public Var logits(float[][] a, float[][] b) {
        int n = a.length;
        return switch (kind) {
            case EMBEDDING -> embed(a).matmul(rotation).mul(embed(b)).sum(1, false);
            case BRADLEY_TERRY -> embed(a).sub(embed(b)).reshape(n);
            case PAIRWISE -> {
                float[][] ab = new float[n][];
                float[][] ba = new float[n][];
                for (int i = 0; i < n; i++) {
                    ab[i] = concat(a[i], b[i]);
                    ba[i] = concat(b[i], a[i]);
                }
                yield net.forward(Var.of(rows(ab)), true).sub(net.forward(Var.of(rows(ba)), true)).reshape(n);
            }
        };
    }

    private static float[] concat(float[] a, float[] b) {
        float[] o = new float[a.length + b.length];
        System.arraycopy(a, 0, o, 0, a.length);
        System.arraycopy(b, 0, o, a.length, b.length);
        return o;
    }

    /**
     * @param a features of one response
     * @param b features of another
     * @return {@code P(a > b)}
     */
    public double prob(float[] a, float[] b) {
        float z = Var.noGrad(() -> logits(new float[][] {a}, new float[][] {b})).value().toArray()[0];
        return 1 / (1 + Math.exp(-z));
    }

    /**
     * All pairwise preference probabilities among candidates: one embedding per candidate for
     * the embedding and Bradley-Terry models, {@code n^2} classifier calls for the pairwise one.
     *
     * @param x the candidates' features
     * @return {@code P[i][j] = P(x_i > x_j)}
     */
    public double[][] matrix(float[][] x) {
        int n = x.length;
        double[][] p = new double[n][n];
        if (kind == Kind.PAIRWISE) {
            float[][] a = new float[n * n][];
            float[][] b = new float[n * n][];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    a[i * n + j] = x[i];
                    b[i * n + j] = x[j];
                }
            }
            float[] z = Var.noGrad(() -> logits(a, b)).value().toArray();
            for (int i = 0; i < n * n; i++) {
                p[i / n][i % n] = 1 / (1 + Math.exp(-z[i]));
            }
            return p;
        }
        float[] e = Var.noGrad(() -> embed(x)).value().toArray();
        float[] rot = rotation.toArray();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double z = 0;
                if (kind == Kind.BRADLEY_TERRY) {
                    z = e[i] - e[j];
                } else {
                    for (int u = 0; u < dims; u++) {
                        double rv = 0;
                        for (int w = 0; w < dims; w++) {
                            rv += e[i * dims + w] * rot[w * dims + u];
                        }
                        z += rv * e[j * dims + u];
                    }
                }
                p[i][j] = 1 / (1 + Math.exp(-z));
            }
        }
        return p;
    }

    /**
     * One step of the binary cross-entropy (Bradley-Terry likelihood) on preferences.
     *
     * @param winners features of the preferred responses
     * @param losers features of the others
     * @return the loss before the step
     */
    public float step(float[][] winners, float[][] losers) {
        Var z = logits(winners, losers);
        Var loss = PreferenceLosses.logSigmoid(z).mean().neg();
        float v = loss.value().item();
        loss.backward();
        optimizer.step(parameters());
        return v;
    }

    /**
     * @param winners features of the preferred responses
     * @param losers features of the others
     * @return the share of pairs whose preferred response the model prefers
     */
    public double accuracy(float[][] winners, float[][] losers) {
        float[] z = Var.noGrad(() -> logits(winners, losers)).value().toArray();
        int ok = 0;
        for (float v : z) {
            ok += v > 0 ? 1 : 0;
        }
        return (double) ok / z.length;
    }

    /** @return the feature length */
    public int features() {
        return features;
    }
}
