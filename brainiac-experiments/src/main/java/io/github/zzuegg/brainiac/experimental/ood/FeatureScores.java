package io.github.zzuegg.brainiac.experimental.ood;

import io.github.zzuegg.brainiac.experimental.nn.KnnIndex;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Post-hoc OOD scores in the penultimate feature space of a trained classifier, fitted once on
 * in-distribution training features; higher means "more in-distribution":
 *
 * <ul>
 *   <li>{@link #mahalanobis}: class-conditional Gaussians with a shared covariance, scored by
 *       the negative minimum Mahalanobis distance (Lee et al., NeurIPS 2018,
 *       <a href="https://arxiv.org/abs/1807.03888">arXiv:1807.03888</a>).</li>
 *   <li>{@link #knn}: the negative distance to the k-th nearest L2-normalised training feature,
 *       through {@link KnnIndex}, with no distribution assumption (Sun et al., ICML 2022,
 *       <a href="https://arxiv.org/abs/2204.06507">arXiv:2204.06507</a>).</li>
 *   <li>{@link #vim}: virtual-logit matching: the norm of the feature's residual outside the
 *       principal subspace of the training features (centred at {@code o = -pinv(W^T) b}),
 *       scaled by {@code alpha = mean max logit / mean residual norm} into a virtual logit,
 *       scored as {@code logsumexp(logits) - alpha |residual|} (Wang et al., CVPR 2022,
 *       <a href="https://arxiv.org/abs/2203.10807">arXiv:2203.10807</a>).</li>
 * </ul>
 */
public final class FeatureScores {

    private final int dim;
    private final TiedGaussian gaussian;
    private final KnnIndex index;
    private final int k;
    private final double[] origin;
    private final double[][] residualBasis;
    private final double alpha;
    private final Dense head;

    private FeatureScores(int dim, TiedGaussian gaussian, KnnIndex index, int k, double[] origin,
            double[][] residualBasis, double alpha, Dense head) {
        this.dim = dim;
        this.gaussian = gaussian;
        this.index = index;
        this.k = k;
        this.origin = origin;
        this.residualBasis = residualBasis;
        this.alpha = alpha;
        this.head = head;
    }

    /**
     * Fits all three scores.
     *
     * @param features training features {@code [n, d]}
     * @param labels their classes
     * @param head the classifier's linear head (for ViM's origin and logits)
     * @param k the neighbour rank of the k-NN score, such as 10
     * @param principal the dimension of ViM's principal subspace, below {@code d}
     * @return the fitted scores
     */
    public static FeatureScores fit(Tensor features, int[] labels, Dense head, int k, int principal) {
        int n = features.shape().dim(0);
        int d = features.shape().dim(1);
        int classes = head.outputSize(0);
        double[] f = features.toDoubleArray();
        TiedGaussian gaussian = TiedGaussian.fit(features, labels, classes);
        // deep k-NN on normalised features
        KnnIndex index = KnnIndex.create(d, 1, KnnIndex.Metric.L2);
        for (int i = 0; i < n; i++) {
            index.add(normalised(f, i, d), new float[1]);
        }
        // ViM: origin, principal subspace and alpha
        double[] w = head.weight().value().toDoubleArray();   // [d, classes]
        double[] bias = head.bias().value().toDoubleArray();
        double[] gram = new double[classes * classes];
        for (int a = 0; a < classes; a++) {
            for (int b = 0; b < classes; b++) {
                for (int j = 0; j < d; j++) {
                    gram[a * classes + b] += w[j * classes + a] * w[j * classes + b];
                }
            }
        }
        for (int a = 0; a < classes; a++) {
            gram[a * classes + a] += 1e-6;
        }
        double[] negB = new double[classes];
        for (int a = 0; a < classes; a++) {
            negB[a] = -bias[a];
        }
        double[] coef = Linalg.solve(tensor(gram, classes, classes), tensor(negB, classes, 1)).toDoubleArray();
        double[] origin = new double[d];
        for (int j = 0; j < d; j++) {
            for (int a = 0; a < classes; a++) {
                origin[j] += w[j * classes + a] * coef[a];
            }
        }
        double[] centred = new double[d * d];
        for (int i = 0; i < n; i++) {
            for (int a = 0; a < d; a++) {
                double xa = f[i * d + a] - origin[a];
                for (int b = a; b < d; b++) {
                    centred[a * d + b] += xa * (f[i * d + b] - origin[b]) / n;
                }
            }
        }
        for (int a = 0; a < d; a++) {
            for (int b = 0; b < a; b++) {
                centred[a * d + b] = centred[b * d + a];
            }
        }
        Tensor[] eig = Linalg.eigh(tensor(centred, d, d));
        double[] vec = eig[1].toDoubleArray();
        int residualDims = d - Math.min(principal, d - 1);
        double[][] basis = new double[residualDims][d];   // the eigenvectors of the smallest eigenvalues
        for (int r = 0; r < residualDims; r++) {
            for (int j = 0; j < d; j++) {
                basis[r][j] = vec[j * d + r];
            }
        }
        FeatureScores partial = new FeatureScores(d, gaussian, index, k, origin, basis, 1, head);
        double sumLogit = 0;
        double sumRes = 0;
        double[] maxLogit = LogitScores.maxLogit(head.forward(io.github.zzuegg.brainiac.autograd.Var.of(features))
                .value());
        double[] res = partial.residualNorms(f, n);
        for (int i = 0; i < n; i++) {
            sumLogit += maxLogit[i];
            sumRes += res[i];
        }
        double alpha = sumLogit / Math.max(1e-12, sumRes);
        return new FeatureScores(d, gaussian, index, k, origin, basis, alpha, head);
    }

    /**
     * @param features {@code [n, d]}
     * @return the negative minimum Mahalanobis distance to a class mean
     */
    public double[] mahalanobis(Tensor features) {
        return gaussian.score(features);
    }

    /**
     * @param features {@code [n, d]}
     * @return the negative distance to the k-th nearest normalised training feature
     */
    public double[] knn(Tensor features) {
        double[] f = features.toDoubleArray();
        int n = f.length / dim;
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            KnnIndex.Hit[] hits = index.search(normalised(f, i, dim), k);
            out[i] = -Math.sqrt(Math.max(0, -hits[hits.length - 1].score()));
        }
        return out;
    }

    /**
     * @param features {@code [n, d]}
     * @return the ViM score {@code logsumexp(logits) - alpha |residual|}
     */
    public double[] vim(Tensor features) {
        double[] f = features.toDoubleArray();
        int n = f.length / dim;
        double[] energy = LogitScores.energy(
                head.forward(io.github.zzuegg.brainiac.autograd.Var.of(features)).value(), 1f);
        double[] res = residualNorms(f, n);
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = energy[i] - alpha * res[i];
        }
        return out;
    }

    /** @return ViM's scale of the virtual logit */
    public double vimAlpha() {
        return alpha;
    }

    private double[] residualNorms(double[] f, int n) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (double[] b : residualBasis) {
                double p = 0;
                for (int j = 0; j < dim; j++) {
                    p += b[j] * (f[i * dim + j] - origin[j]);
                }
                s += p * p;
            }
            out[i] = Math.sqrt(s);
        }
        return out;
    }

    private static float[] normalised(double[] f, int row, int d) {
        double s = 0;
        for (int j = 0; j < d; j++) {
            s += f[row * d + j] * f[row * d + j];
        }
        double inv = 1 / Math.sqrt(s + 1e-12);
        float[] out = new float[d];
        for (int j = 0; j < d; j++) {
            out[j] = (float) (f[row * d + j] * inv);
        }
        return out;
    }

    private static Tensor tensor(double[] v, int r, int c) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = (float) v[i];
        }
        return Tensor.of(Shape.of(r, c), f);
    }
}
