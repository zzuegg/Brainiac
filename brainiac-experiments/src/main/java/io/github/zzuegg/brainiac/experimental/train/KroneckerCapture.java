package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Layer hooks for Kronecker-factored curvature ({@link Kfac}, {@link AdaFisher}): routes
 * each dense layer's {@code z = a W + b} through {@link #dense} during one forward pass, which
 * records the layer input {@code a} and the pre-activation {@code z}. {@link #factors} then
 * takes one backward pass of a loss and returns, per layer, the inputs {@code a} (with a
 * column of ones for the bias) and the per-example pre-activation gradients
 * {@code delta_i = dloss_i / dz_i}: the two ingredients of the Kronecker factors
 * {@code A = E[a a^T]} and {@code G = E[delta delta^T]}.
 *
 * <p>For the true Fisher pass a loss on labels sampled from the model's own predictions
 * ({@link Sophia#sampleLabels}); passing the loss on the true labels gives the empirical
 * Fisher instead.
 */
public final class KroneckerCapture {

    /**
     * One recorded dense layer.
     *
     * @param weight its {@code [in, out]} weight
     * @param bias its {@code [1, out]} bias, or {@code null}
     * @param input the layer input {@code a}, {@code [batch, in]}
     * @param preActivation the recorded {@code z}
     */
    public record Site(Var weight, Var bias, Tensor input, Var preActivation) {
    }

    /**
     * A layer's statistics from one batch.
     *
     * @param site the layer
     * @param a {@code [batch][in (+1)]} inputs, with a trailing 1 when the layer has a bias
     * @param delta {@code [batch][out]} per-example pre-activation gradients
     */
    public record Factors(Site site, double[][] a, double[][] delta) {

        /**
         * @return {@code A = a^T a / batch}
         */
        public double[][] inputCovariance() {
            return covariance(a);
        }

        /**
         * @return {@code G = delta^T delta / batch}
         */
        public double[][] gradientCovariance() {
            return covariance(delta);
        }

        /**
         * The batch-mean gradient of the combined weight {@code [W; b]}: {@code a^T delta / batch}.
         *
         * @return {@code [in (+1)][out]}
         */
        public double[][] gradient() {
            int n = a.length;
            int p = a[0].length;
            int q = delta[0].length;
            double[][] g = new double[p][q];
            for (int i = 0; i < n; i++) {
                for (int r = 0; r < p; r++) {
                    double ar = a[i][r] / n;
                    for (int c = 0; c < q; c++) {
                        g[r][c] += ar * delta[i][c];
                    }
                }
            }
            return g;
        }

        private static double[][] covariance(double[][] x) {
            int n = x.length;
            int d = x[0].length;
            double[][] c = new double[d][d];
            for (double[] row : x) {
                for (int r = 0; r < d; r++) {
                    double v = row[r] / n;
                    for (int s = r; s < d; s++) {
                        c[r][s] += v * row[s];
                    }
                }
            }
            for (int r = 0; r < d; r++) {
                for (int s = 0; s < r; s++) {
                    c[r][s] = c[s][r];
                }
            }
            return c;
        }
    }

    private final List<Site> sites = new ArrayList<>();

    /** Creates an empty capture for one forward pass. */
    public KroneckerCapture() {
    }

    /**
     * Computes and records a dense layer's pre-activation.
     *
     * @param weight the {@code [in, out]} weight
     * @param bias the {@code [1, out]} bias, or {@code null}
     * @param input the layer input {@code [batch, in]}
     * @return {@code input W + b}
     */
    public Var dense(Var weight, Var bias, Var input) {
        Var z = input.matmul(weight);
        if (bias != null) {
            z = z.add(bias);
        }
        sites.add(new Site(weight, bias, input.value(), z));
        return z;
    }

    /** @return the recorded layers, in forward order */
    public List<Site> sites() {
        return List.copyOf(sites);
    }

    /**
     * Back-propagates a batch-mean loss to every recorded pre-activation (without touching
     * the parameters' gradients) and returns the per-layer statistics.
     *
     * @param loss a mean over the batch of per-example losses
     * @return one entry per recorded layer
     */
    public List<Factors> factors(Var loss) {
        List<Var> zs = new ArrayList<>(sites.size());
        for (Site s : sites) {
            zs.add(s.preActivation());
        }
        List<Var> dz = Var.gradients(loss, zs, false);
        List<Factors> out = new ArrayList<>(sites.size());
        for (int k = 0; k < sites.size(); k++) {
            Site s = sites.get(k);
            int n = s.input().shape().dim(0);
            int in = s.input().shape().dim(1);
            int extra = s.bias() == null ? 0 : 1;
            float[] av = s.input().toArray();
            double[][] a = new double[n][in + extra];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < in; j++) {
                    a[i][j] = av[i * in + j];
                }
                if (extra == 1) {
                    a[i][in] = 1;
                }
            }
            int outDim = dz.get(k).shape().dim(1);
            float[] dv = dz.get(k).value().toArray();
            double[][] d = new double[n][outDim];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < outDim; j++) {
                    d[i][j] = dv[i * outDim + j] * (double) n;
                }
            }
            out.add(new Factors(s, a, d));
        }
        return out;
    }
}
