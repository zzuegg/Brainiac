package io.github.zzuegg.brainiac.experimental.ssl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SelfSupervisedLosses;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * The KoLeo regulariser of DINOv2 (Oquab et al., 2024), from the Kozachenko-Leonenko
 * differential-entropy estimator (Sablayrolles et al., ICLR 2019): for each L2-normalised
 * embedding in the batch, {@code -log} of the distance to its nearest neighbour, averaged.
 * Minimising it pushes every point away from its closest neighbour, spreading the batch
 * uniformly over the sphere, which helps nearest-neighbour retrieval without changing the main
 * loss. {@link #uniformity} is the matching diagnostic of Wang and Isola (ICML 2020).
 *
 * <p>Papers: <a href="https://arxiv.org/abs/1806.03198">arXiv:1806.03198</a>,
 * <a href="https://arxiv.org/abs/2304.07193">arXiv:2304.07193</a>.
 */
public final class KoLeo {

    private KoLeo() {
    }

    /**
     * The KoLeo loss {@code -mean_i log(min_{j != i} ||z_i - z_j|| + eps)} on normalised rows.
     *
     * @param z the embeddings {@code [n, d]}, {@code n >= 2}
     * @return a scalar to add (with a small weight, 0.1 in DINOv2) to the main loss
     */
    public static Var loss(Var z) {
        int n = z.shape().dim(0);
        Var u = SelfSupervisedLosses.normalize(z);
        float[] self = new float[n * n];
        for (int i = 0; i < n; i++) {
            self[i * n + i] = -4;
        }
        Var dots = u.matmul(u.transpose()).add(Tensor.of(Shape.of(n, n), self, z.value().device()));
        Var nearest = dots.max(1, false);
        Var distance = nearest.mul(-2f).add(2f).maximum(0f).add(1e-8f).sqrt();
        return distance.add(1e-8f).log().mean().neg();
    }

    /**
     * Wang and Isola's uniformity: {@code log mean_{i<j} exp(-t ||u_i - u_j||^2)} of the
     * normalised rows; lower is more uniform.
     *
     * @param z the embeddings {@code [n, d]}
     * @param t the kernel scale, 2 in the paper
     * @return the uniformity
     */
    public static double uniformity(Tensor z, double t) {
        int n = z.shape().dim(0);
        int d = z.shape().dim(1);
        float[] v = z.toArray();
        double[][] u = new double[n][d];
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < d; j++) {
                s += v[i * d + j] * v[i * d + j];
            }
            s = Math.sqrt(s) + 1e-12;
            for (int j = 0; j < d; j++) {
                u[i][j] = v[i * d + j] / s;
            }
        }
        double sum = 0;
        int pairs = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double dist = 0;
                for (int k = 0; k < d; k++) {
                    double e = u[i][k] - u[j][k];
                    dist += e * e;
                }
                sum += Math.exp(-t * dist);
                pairs++;
            }
        }
        return Math.log(sum / pairs);
    }
}
