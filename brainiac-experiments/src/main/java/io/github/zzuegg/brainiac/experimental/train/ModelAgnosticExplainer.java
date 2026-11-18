package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.function.Function;
import java.util.stream.IntStream;

/**
 * Explanations that only query a model: any {@code Function<double[][], double[]>} that
 * scores a batch of rows (one class's probability, a regression value, a boosted tree
 * ensemble's {@code predict}).
 *
 * <ul>
 *   <li>{@link #kernelShap} (Lundberg and Lee 2017, <a href="https://arxiv.org/abs/1705.07874">arXiv:1705.07874</a>):
 *       Shapley values as the solution of a weighted linear regression over feature
 *       coalitions, a coalition's value being the mean score with the absent features taken
 *       from background rows. Weighted by the Shapley kernel
 *       {@code (d - 1) / (C(d, |z|) |z| (d - |z|))} and constrained to add up to
 *       {@code f(x) - E f(background)}; all coalitions when there are few enough, sampled
 *       otherwise.</li>
 *   <li>{@link #lime} (Ribeiro et al. 2016, <a href="https://arxiv.org/abs/1602.04938">arXiv:1602.04938</a>):
 *       a ridge-regression surrogate fitted to Gaussian perturbations around the input,
 *       weighted by an exponential kernel on the standardized distance; its coefficients
 *       times the feature scales are the attributions.</li>
 *   <li>{@link #counterfactual} (Wachter et al. 2017, <a href="https://arxiv.org/abs/1711.00399">arXiv:1711.00399</a>):
 *       the nearest input, in the scale-normalized L1 distance, whose score crosses a
 *       threshold: proximal gradient descent on
 *       {@code lambda (f(x') - target)^2 + |x' - x|_1} (the target a margin past the
 *       threshold), with finite-difference gradients and soft-thresholding (which
 *       keeps the change sparse), doubling {@code lambda} until the prediction flips.</li>
 * </ul>
 */
public final class ModelAgnosticExplainer {

    private final Function<double[][], double[]> model;

    /**
     * Creates the explainer.
     *
     * @param model scores a batch of rows, one score per row
     */
    public ModelAgnosticExplainer(Function<double[][], double[]> model) {
        this.model = model;
    }

    /**
     * KernelSHAP attributions.
     *
     * @param x the row to explain
     * @param background rows that stand in for absent features
     * @param samples the most coalitions to evaluate; all {@code 2^d - 2} if that is fewer
     * @param random draws the coalitions when sampling
     * @return one Shapley value per feature, adding up to {@code f(x) - E f(background)}
     */
    public double[] kernelShap(double[] x, double[][] background, int samples, SplittableRandom random) {
        int d = x.length;
        double fx = model.apply(new double[][] {x})[0];
        double base = mean(model.apply(background));
        boolean[][] coalitions;
        double[] weights;
        if (d < 31 && (1L << d) - 2 <= samples) {
            int n = (1 << d) - 2;
            coalitions = new boolean[n][d];
            weights = new double[n];
            for (int mask = 1; mask <= n; mask++) {
                for (int j = 0; j < d; j++) {
                    coalitions[mask - 1][j] = (mask >> j & 1) == 1;
                }
                weights[mask - 1] = shapleyKernel(d, Integer.bitCount(mask));
            }
        } else {
            // sample sizes in proportion to the kernel's total weight per size, then members uniformly
            double[] sizeWeight = new double[d];
            double total = 0;
            for (int s = 1; s < d; s++) {
                sizeWeight[s] = (d - 1.0) / (s * (d - s));
                total += sizeWeight[s];
            }
            coalitions = new boolean[samples][d];
            weights = new double[samples];
            for (int i = 0; i < samples; i++) {
                double u = random.nextDouble() * total;
                int size = 1;
                while (size < d - 1 && u > sizeWeight[size]) {
                    u -= sizeWeight[size];
                    size++;
                }
                int[] order = permutation(d, random);
                for (int j = 0; j < size; j++) {
                    coalitions[i][order[j]] = true;
                }
                weights[i] = 1;
            }
        }
        int n = coalitions.length;
        int b = background.length;
        double[][] rows = new double[n * b][];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < b; k++) {
                double[] r = background[k].clone();
                for (int j = 0; j < d; j++) {
                    if (coalitions[i][j]) {
                        r[j] = x[j];
                    }
                }
                rows[i * b + k] = r;
            }
        }
        double[] scores = model.apply(rows);
        // eliminate the last feature with the efficiency constraint: phi_d = (fx - base) - sum others
        double[] a = new double[n * (d - 1)];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double v = 0;
            for (int k = 0; k < b; k++) {
                v += scores[i * b + k] / b;
            }
            double zd = coalitions[i][d - 1] ? 1 : 0;
            double sw = Math.sqrt(weights[i]);
            for (int j = 0; j < d - 1; j++) {
                a[i * (d - 1) + j] = sw * ((coalitions[i][j] ? 1 : 0) - zd);
            }
            y[i] = sw * (v - base - zd * (fx - base));
        }
        double[] phi = lstsq(a, y, n, d - 1, 1e-9);
        double[] out = Arrays.copyOf(phi, d);
        double rest = fx - base;
        for (int j = 0; j < d - 1; j++) {
            rest -= phi[j];
        }
        out[d - 1] = rest;
        return out;
    }

    private static double shapleyKernel(int d, int size) {
        return (d - 1.0) / (binomial(d, size) * size * (d - size));
    }

    private static double binomial(int n, int k) {
        double c = 1;
        for (int i = 1; i <= k; i++) {
            c = c * (n - k + i) / i;
        }
        return c;
    }

    /**
     * LIME attributions from a local linear surrogate.
     *
     * @param x the row to explain
     * @param scale each feature's standard deviation, which sets the perturbation and the distance
     * @param samples how many perturbed rows
     * @param kernelWidth the width of the exponential proximity kernel, in standardized units
     *        (LIME's default is {@code 0.75 sqrt(d)})
     * @param random draws the perturbations
     * @return the surrogate's coefficient times the feature scale, per feature
     */
    public double[] lime(double[] x, double[] scale, int samples, double kernelWidth, SplittableRandom random) {
        int d = x.length;
        double[][] rows = new double[samples][d];
        double[][] z = new double[samples][d];
        double[] w = new double[samples];
        for (int i = 0; i < samples; i++) {
            double dist = 0;
            for (int j = 0; j < d; j++) {
                z[i][j] = i == 0 ? 0 : random.nextGaussian();
                rows[i][j] = x[j] + z[i][j] * scale[j];
                dist += z[i][j] * z[i][j];
            }
            w[i] = Math.exp(-dist / (kernelWidth * kernelWidth));
        }
        double[] f = model.apply(rows);
        double[] a = new double[samples * (d + 1)];
        double[] y = new double[samples];
        for (int i = 0; i < samples; i++) {
            double sw = Math.sqrt(w[i]);
            for (int j = 0; j < d; j++) {
                a[i * (d + 1) + j] = sw * z[i][j];
            }
            a[i * (d + 1) + d] = sw;
            y[i] = sw * f[i];
        }
        return Arrays.copyOf(lstsq(a, y, samples, d + 1, 1e-3), d);
    }

    /**
     * A counterfactual explanation.
     *
     * @param input the original row
     * @param changed the counterfactual row
     * @param score the model's score for it
     * @param valid whether the score crossed the threshold
     * @param features the indices of the features that changed by more than 1% of their scale
     */
    public record Counterfactual(double[] input, double[] changed, double score, boolean valid, int[] features) {
    }

    /**
     * Searches for a sparse, close counterfactual whose score crosses {@code threshold} to
     * the other side.
     *
     * @param x the row
     * @param scale each feature's scale (the L1 distance divides by it)
     * @param threshold the decision threshold; the score is best a log-odds or another
     *        unbounded value, since a saturated probability has no gradient to follow
     * @param margin how far past the threshold the search aims
     * @param steps proximal-gradient steps per {@code lambda}
     * @param h the finite-difference step, in units of {@code scale}: small for smooth
     *        models, a sizeable share of a unit for piecewise-constant ones such as trees
     * @return the counterfactual; {@code valid} is false if no lambda up to 1024 flipped it
     */
    public Counterfactual counterfactual(double[] x, double[] scale, double threshold, double margin, int steps,
            double h) {
        int d = x.length;
        double f0 = model.apply(new double[][] {x})[0];
        boolean above = f0 > threshold;
        double target = threshold + (above ? -margin : margin);
        double[] best = null;
        double bestScore = f0;
        double lr = 0.05;
        double[] u = new double[d];   // the change in units of scale, warm-started across lambdas
        for (double lambda = 1.0 / 64; lambda <= 1024 && best == null; lambda *= 2) {
            double shrink = lr / lambda;   // the L1 weight 1/lambda relative to the fit term
            for (int s = 0; s < steps; s++) {
                double[][] probe = new double[2 * d + 1][];
                double[] cur = new double[d];
                for (int j = 0; j < d; j++) {
                    cur[j] = x[j] + u[j] * scale[j];
                }
                probe[2 * d] = cur;
                for (int j = 0; j < d; j++) {
                    probe[2 * j] = cur.clone();
                    probe[2 * j][j] += h * scale[j];
                    probe[2 * j + 1] = cur.clone();
                    probe[2 * j + 1][j] -= h * scale[j];
                }
                double[] f = model.apply(probe);
                double fc = f[2 * d];
                if (above ? fc < threshold : fc > threshold) {
                    best = cur;
                    bestScore = fc;
                    break;
                }
                double miss = fc - target;
                for (int j = 0; j < d; j++) {
                    double grad = 2 * miss * (f[2 * j] - f[2 * j + 1]) / (2 * h);
                    double v = u[j] - lr * Math.clamp(grad, -20, 20);
                    u[j] = Math.signum(v) * Math.max(0, Math.abs(v) - shrink);   // soft-threshold: the L1 prox
                }
            }
        }
        boolean valid = best != null;
        double[] cf = valid ? best : x.clone();
        int[] changed = IntStream.range(0, d).filter(j -> Math.abs(cf[j] - x[j]) > 0.01 * scale[j]).toArray();
        return new Counterfactual(x.clone(), cf, bestScore, valid, changed);
    }

    private static double[] lstsq(double[] a, double[] y, int m, int n, double ridge) {
        Tensor ta = Tensor.of(Shape.of(m, n), a, DType.FLOAT64, Device.cpu());
        Tensor tb = Tensor.of(Shape.of(m, 1), y, DType.FLOAT64, Device.cpu());
        return Linalg.lstsq(ta, tb, ridge).toDoubleArray();
    }

    private static double mean(double[] v) {
        double s = 0;
        for (double x : v) {
            s += x;
        }
        return s / v.length;
    }

    private static int[] permutation(int n, SplittableRandom random) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }
}
