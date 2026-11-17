package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * A last-layer Laplace approximation: after ordinary training, put a Gaussian over the
 * weights of the final softmax layer, centred on the trained weights with the inverse
 * Hessian of the loss (generalised Gauss-Newton, plus a prior precision) as covariance.
 * Predictions average over that Gaussian, which keeps in-distribution accuracy and makes
 * the model much less overconfident far from the data, almost for free: the Hessian of one
 * layer is small and is computed once.
 *
 * <p>The predictive uses the probit approximation per class,
 * {@code softmax(mu_c / sqrt(1 + pi/8 var_c))}. The prior precision can be chosen by
 * validation NLL ({@link #fitPriorPrecision}).
 *
 * <p>Daxberger et al. 2021, "Laplace Redux: Effortless Bayesian Deep Learning",
 * <a href="https://arxiv.org/abs/2106.14806">arXiv:2106.14806</a>; Kristiadi et al. 2020,
 * "Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks",
 * <a href="https://arxiv.org/abs/2002.10118">arXiv:2002.10118</a>.
 */
public final class LastLayerLaplace {

    private final Model model;
    private final int features;
    private final int classes;
    /** the MAP weights with the bias as the last row, {@code [features + 1][classes]} */
    private final double[][] w;
    /** the GGN of the data term, row/column index {@code i * classes + c} */
    private final double[][] ggn;
    private double prior;
    private double[][] covariance;

    private LastLayerLaplace(Model model, double[][] w, double[][] ggn, int features, int classes) {
        this.model = model;
        this.w = w;
        this.ggn = ggn;
        this.features = features;
        this.classes = classes;
    }

    /**
     * Fits the approximation to a trained classifier whose last layer is a {@link Dense}
     * layer with a softmax activation.
     *
     * @param model the trained model
     * @param inputs training inputs
     * @param priorPrecision the prior precision (the weight decay strength)
     * @return the approximation
     */
    public static LastLayerLaplace fit(Model model, Tensor inputs, double priorPrecision) {
        List<Layer> layers = model.layers();
        if (!(layers.getLast() instanceof Dense head)) {
            throw new IllegalArgumentException("the last layer must be a dense softmax layer");
        }
        float[] wt = head.weight().value().to(DType.FLOAT32).toArray();
        int f = head.weight().shape().dim(0);
        int c = head.weight().shape().dim(1);
        float[] bias = head.bias() == null ? new float[c] : head.bias().value().to(DType.FLOAT32).toArray();
        double[][] w = new double[f + 1][c];
        for (int i = 0; i < f; i++) {
            for (int k = 0; k < c; k++) {
                w[i][k] = wt[i * c + k];
            }
        }
        for (int k = 0; k < c; k++) {
            w[f][k] = bias[k];
        }
        double[][] phi = features(model, inputs, f);
        int n = phi.length;
        int dim = (f + 1) * c;
        double[][] ggn = new double[dim][dim];
        for (int s = 0; s < n; s++) {
            double[] p = softmax(logits(w, phi[s], f, c));
            for (int i = 0; i <= f; i++) {
                for (int j = 0; j <= f; j++) {
                    double pp = phi[s][i] * phi[s][j];
                    if (pp == 0) {
                        continue;
                    }
                    for (int a = 0; a < c; a++) {
                        for (int b = 0; b < c; b++) {
                            double h = (a == b ? p[a] : 0) - p[a] * p[b];
                            ggn[i * c + a][j * c + b] += pp * h;
                        }
                    }
                }
            }
        }
        LastLayerLaplace l = new LastLayerLaplace(model, w, ggn, f, c);
        l.priorPrecision(priorPrecision);
        return l;
    }

    /** The penultimate features with a constant 1 appended. */
    private static double[][] features(Model model, Tensor inputs, int f) {
        List<Var> trace = LayerHooks.trace(model, Var.of(inputs));
        float[] h = trace.get(trace.size() - 2).value().to(DType.FLOAT32).toArray();
        int n = h.length / f;
        double[][] out = new double[n][f + 1];
        for (int s = 0; s < n; s++) {
            for (int i = 0; i < f; i++) {
                out[s][i] = h[s * f + i];
            }
            out[s][f] = 1;
        }
        return out;
    }

    private static double[] logits(double[][] w, double[] phi, int f, int c) {
        double[] z = new double[c];
        for (int i = 0; i <= f; i++) {
            for (int k = 0; k < c; k++) {
                z[k] += phi[i] * w[i][k];
            }
        }
        return z;
    }

    private static double[] softmax(double[] z) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : z) {
            max = Math.max(max, v);
        }
        double s = 0;
        double[] p = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            p[i] = Math.exp(z[i] - max);
            s += p[i];
        }
        for (int i = 0; i < z.length; i++) {
            p[i] /= s;
        }
        return p;
    }

    /**
     * Sets the prior precision and recomputes the posterior covariance.
     *
     * @param precision the prior precision
     */
    public void priorPrecision(double precision) {
        this.prior = precision;
        int dim = ggn.length;
        double[][] h = new double[dim][];
        for (int i = 0; i < dim; i++) {
            h[i] = ggn[i].clone();
            h[i][i] += precision;
        }
        covariance = QuantMath.inverse(h);
    }

    /** @return the prior precision */
    public double priorPrecision() {
        return prior;
    }

    /**
     * Picks the prior precision with the lowest NLL on validation data from a log grid.
     *
     * @param inputs validation inputs
     * @param labels validation classes
     * @return the chosen precision
     */
    public double fitPriorPrecision(Tensor inputs, int[] labels) {
        double best = prior;
        double bestNll = Double.POSITIVE_INFINITY;
        for (double logp = -3; logp <= 3.01; logp += 0.5) {
            priorPrecision(Math.pow(10, logp));
            double[][] p = predict(inputs);
            double nll = 0;
            for (int i = 0; i < labels.length; i++) {
                nll -= Math.log(Math.max(p[i][labels[i]], 1e-12));
            }
            if (nll < bestNll) {
                bestNll = nll;
                best = prior;
            }
        }
        priorPrecision(best);
        return best;
    }

    /**
     * The MAP prediction (the trained model's own).
     *
     * @param inputs a batch
     * @return probabilities {@code [sample][class]}
     */
    public double[][] predictMap(Tensor inputs) {
        double[][] phi = features(model, inputs, features);
        double[][] out = new double[phi.length][];
        for (int s = 0; s < phi.length; s++) {
            out[s] = softmax(logits(w, phi[s], features, classes));
        }
        return out;
    }

    /**
     * The Laplace predictive with the probit approximation.
     *
     * @param inputs a batch
     * @return probabilities {@code [sample][class]}
     */
    public double[][] predict(Tensor inputs) {
        double[][] phi = features(model, inputs, features);
        double[][] out = new double[phi.length][];
        for (int s = 0; s < phi.length; s++) {
            double[] mu = logits(w, phi[s], features, classes);
            double[] z = new double[classes];
            for (int c = 0; c < classes; c++) {
                double var = 0;
                for (int i = 0; i <= features; i++) {
                    if (phi[s][i] == 0) {
                        continue;
                    }
                    for (int j = 0; j <= features; j++) {
                        var += phi[s][i] * phi[s][j] * covariance[i * classes + c][j * classes + c];
                    }
                }
                z[c] = mu[c] / Math.sqrt(1 + Math.PI / 8 * Math.max(var, 0));
            }
            out[s] = softmax(z);
        }
        return out;
    }
}
