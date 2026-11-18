package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.experimental.discovery.Distributions;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;

import java.util.SplittableRandom;

/**
 * Planted-truth generators for statistical learning: Gaussian designs with Toeplitz correlation
 * and an s-sparse coefficient vector of known support, heavy-tailed and contaminated noise,
 * grouped designs, sparse precision matrices with a known conditional-independence graph
 * (chain, hub, random), count outcomes from Poisson, negative binomial, zero-inflated Poisson
 * and Tweedie models with exposure, nonlinear selection tasks (XOR pairs, sine features,
 * redundant copies), and streams of p-values and e-values with a known null set. Every
 * generator is seeded and returns its ground truth so methods can be scored exactly.
 */
public final class SparseDesigns {

    private SparseDesigns() {
    }

    /** Noise distributions for the linear designs. */
    public enum Noise {
        /** Gaussian noise. */
        GAUSSIAN,
        /** Student t with 2 degrees of freedom (infinite variance), scaled to the same IQR. */
        STUDENT_T2
    }

    /** How the planted coefficients are drawn. */
    public enum Coefficients {
        /** Random signs, magnitudes uniform in [1, 2]. */
        UNIFORM,
        /** Random signs, half large (magnitude 3) and half small (magnitude 0.5). */
        MIXED
    }

    /**
     * A regression data set with its planted truth.
     *
     * @param x training design {@code [n][p]}
     * @param y training response
     * @param beta the true coefficients (no intercept)
     * @param xTest a fresh test design
     * @param yTest test responses
     * @param sigma the noise scale
     */
    public record Regression(double[][] x, double[] y, double[] beta, double[][] xTest, double[] yTest,
            double sigma) {

        /**
         * The true support.
         *
         * @return whether each coefficient is non-zero
         */
        public boolean[] support() {
            boolean[] s = new boolean[beta.length];
            for (int j = 0; j < beta.length; j++) {
                s[j] = beta[j] != 0;
            }
            return s;
        }
    }

    /**
     * A linear model {@code y = X beta + sigma eps} with AR(1) (Toeplitz) feature correlation
     * {@code corr(x_i, x_j) = rho^|i-j|}.
     *
     * @param n training rows (the test set has the same size)
     * @param p features
     * @param s non-zero coefficients, at random positions
     * @param rho the Toeplitz correlation
     * @param snr the signal-to-noise ratio {@code var(X beta) / sigma^2}
     * @param noise the noise law
     * @param coefficients how the non-zero coefficients are drawn
     * @param seed the seed
     * @return the data set
     */
    public static Regression linear(int n, int p, int s, double rho, double snr, Noise noise,
            Coefficients coefficients, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] beta = new double[p];
        int[] perm = permutation(p, r);
        for (int k = 0; k < s; k++) {
            double mag = coefficients == Coefficients.UNIFORM ? 1 + r.nextDouble() : (k % 2 == 0 ? 3 : 0.5);
            beta[perm[k]] = r.nextBoolean() ? mag : -mag;
        }
        double signal = 0;
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                if (beta[i] != 0 && beta[j] != 0) {
                    signal += beta[i] * beta[j] * Math.pow(rho, Math.abs(i - j));
                }
            }
        }
        double sigma = Math.sqrt(signal / snr);
        double[][] x = toeplitzDesign(n, p, rho, r);
        double[][] xt = toeplitzDesign(n, p, rho, r);
        return new Regression(x, respond(x, beta, sigma, noise, r), beta, xt, respond(xt, beta, sigma, noise, r),
                sigma);
    }

    /**
     * A grouped design: {@code groups} blocks of {@code size} features, correlated within a
     * block (exchangeable correlation {@code rho}), with every feature of the first
     * {@code active} blocks (after a random permutation of blocks) non-zero.
     *
     * @param n training rows
     * @param groups the number of groups
     * @param size features per group
     * @param active active groups
     * @param rho the within-group correlation
     * @param snr the signal-to-noise ratio
     * @param seed the seed
     * @return the data set; feature {@code j} belongs to group {@code j / size}
     */
    public static Regression grouped(int n, int groups, int size, int active, double rho, double snr, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int p = groups * size;
        double[] beta = new double[p];
        int[] perm = permutation(groups, r);
        for (int a = 0; a < active; a++) {
            int g = perm[a];
            for (int k = 0; k < size; k++) {
                beta[g * size + k] = (0.5 + r.nextDouble()) * (r.nextBoolean() ? 1 : -1);
            }
        }
        double signal = 0;
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                double c = i == j ? 1 : i / size == j / size ? rho : 0;
                signal += beta[i] * beta[j] * c;
            }
        }
        double sigma = Math.sqrt(signal / snr);
        double[][] x = groupDesign(n, groups, size, rho, r);
        double[][] xt = groupDesign(n, groups, size, rho, r);
        return new Regression(x, respond(x, beta, sigma, Noise.GAUSSIAN, r), beta, xt,
                respond(xt, beta, sigma, Noise.GAUSSIAN, r), sigma);
    }

    /**
     * Contaminates a training set: a share of rows gets a large vertical shift in {@code y}
     * and, optionally, its features moved far out (leverage points pulling the fit).
     *
     * @param data the clean data
     * @param share the contaminated share of training rows
     * @param leverage whether contaminated rows also become leverage points
     * @param seed the seed
     * @return a new data set and the indices of the contaminated rows as a mask
     */
    public static Contaminated contaminate(Regression data, double share, boolean leverage, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = data.y().length;
        double[][] x = new double[n][];
        double[] y = data.y().clone();
        boolean[] bad = new boolean[n];
        for (int i = 0; i < n; i++) {
            x[i] = data.x()[i].clone();
        }
        int[] perm = permutation(n, r);
        int k = (int) Math.round(share * n);
        double scale = 10 * Math.max(data.sigma(), 1);
        for (int a = 0; a < k; a++) {
            int i = perm[a];
            bad[i] = true;
            if (leverage) {
                for (int j = 0; j < x[i].length; j++) {
                    x[i][j] = 5 + r.nextGaussian();
                }
                // the outlying response follows the opposite slope, so the points pull the fit
                double f = 0;
                for (int j = 0; j < x[i].length; j++) {
                    f += x[i][j] * data.beta()[j];
                }
                y[i] = -f + data.sigma() * r.nextGaussian();
            } else {
                y[i] += scale + data.sigma() * r.nextGaussian();
            }
        }
        return new Contaminated(new Regression(x, y, data.beta(), data.xTest(), data.yTest(), data.sigma()), bad);
    }

    /**
     * A contaminated data set.
     *
     * @param data the data
     * @param outlier which training rows were contaminated
     */
    public record Contaminated(Regression data, boolean[] outlier) {
    }

    /** Shapes of planted conditional-independence graphs. */
    public enum GraphShape {
        /** A path {@code 0 - 1 - ... - (p-1)}. */
        CHAIN,
        /** Hubs, each linked to a block of about ten satellites. */
        HUB,
        /** An Erdos-Renyi graph with about {@code p} edges. */
        RANDOM
    }

    /**
     * A Gaussian graphical model with samples.
     *
     * @param precision the true precision matrix {@code Theta}
     * @param edges the true graph (the off-diagonal support of {@code Theta})
     * @param samples {@code n} rows drawn from {@code N(0, Theta^-1)}
     */
    public record GaussianGraph(double[][] precision, boolean[][] edges, double[][] samples) {
    }

    /**
     * Draws a sparse, diagonally dominant precision matrix of the given shape and samples from
     * the Gaussian it defines.
     *
     * @param shape the graph shape
     * @param p the dimension
     * @param n samples
     * @param seed the seed
     * @return the model and its samples
     */
    public static GaussianGraph gaussianGraph(GraphShape shape, int p, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        boolean[][] e = new boolean[p][p];
        switch (shape) {
            case CHAIN -> {
                for (int i = 0; i + 1 < p; i++) {
                    e[i][i + 1] = e[i + 1][i] = true;
                }
            }
            case HUB -> {
                for (int i = 0; i < p; i++) {
                    int hub = (i / 10) * 10;
                    if (i != hub) {
                        e[i][hub] = e[hub][i] = true;
                    }
                }
            }
            case RANDOM -> {
                double prob = 2.0 / (p - 1);
                for (int i = 0; i < p; i++) {
                    for (int j = i + 1; j < p; j++) {
                        if (r.nextDouble() < prob) {
                            e[i][j] = e[j][i] = true;
                        }
                    }
                }
            }
            default -> throw new IllegalArgumentException(String.valueOf(shape));
        }
        double[][] theta = new double[p][p];
        for (int i = 0; i < p; i++) {
            for (int j = i + 1; j < p; j++) {
                if (e[i][j]) {
                    double v = (0.3 + 0.2 * r.nextDouble()) * (r.nextBoolean() ? 1 : -1);
                    theta[i][j] = theta[j][i] = v;
                }
            }
        }
        for (int i = 0; i < p; i++) {
            double s = 0;
            for (int j = 0; j < p; j++) {
                s += Math.abs(theta[i][j]);
            }
            theta[i][i] = Math.max(1, s + 0.2);
        }
        double[][] sigma = MlMatrices.choleskyInverse(MlMatrices.cholesky(theta));
        double[][] l = MlMatrices.cholesky(sigma);
        double[][] x = new double[n][p];
        for (int t = 0; t < n; t++) {
            double[] z = new double[p];
            for (int j = 0; j < p; j++) {
                z[j] = r.nextGaussian();
            }
            for (int i = 0; i < p; i++) {
                double s = 0;
                for (int j = 0; j <= i; j++) {
                    s += l[i][j] * z[j];
                }
                x[t][i] = s;
            }
        }
        return new GaussianGraph(theta, e, x);
    }

    /** Count and claim-amount families. */
    public enum CountFamily {
        /** Poisson counts. */
        POISSON,
        /** Negative binomial (gamma-Poisson) counts; see {@link SparseDesigns#counts}. */
        NEGATIVE_BINOMIAL,
        /** A share of structural zeros mixed with Poisson counts. */
        ZERO_INFLATED_POISSON,
        /** Tweedie compound Poisson-gamma amounts with power 1.5. */
        TWEEDIE
    }

    /**
     * Count data with exposure.
     *
     * @param x training features {@code [n][p]}
     * @param exposure training exposures (the offset is their log)
     * @param y training outcomes
     * @param xTest test features
     * @param exposureTest test exposures
     * @param yTest test outcomes
     * @param beta the true log-mean coefficients, intercept first
     * @param muTest the true test means of the count part
     */
    public record Counts(double[][] x, double[] exposure, double[] y, double[][] xTest, double[] exposureTest,
            double[] yTest, double[] beta, double[] muTest) {
    }

    /**
     * Draws count outcomes {@code y ~ F(mu)} with {@code log mu = log(exposure) + b0 + x beta}.
     *
     * @param family the family
     * @param n rows per split
     * @param p features
     * @param dispersion the overdispersion: for the negative binomial the variance is
     *     {@code mu + mu^2 / k} with {@code k = 1 / (dispersion - 1)} for dispersion &gt; 1
     *     (dispersion 1 is Poisson); for the Tweedie it is {@code phi}
     * @param zeroShare the structural-zero probability (zero-inflated only)
     * @param seed the seed
     * @return the data
     */
    public static Counts counts(CountFamily family, int n, int p, double dispersion, double zeroShare, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] beta = new double[p + 1];
        beta[0] = 0.3;
        for (int j = 1; j <= p; j++) {
            beta[j] = j <= Math.max(1, p / 2) ? 0.4 * (r.nextBoolean() ? 1 : -1) * (0.5 + r.nextDouble()) : 0;
        }
        double[][] x = new double[n][p];
        double[][] xt = new double[n][p];
        double[] ex = new double[n];
        double[] ext = new double[n];
        double[] y = new double[n];
        double[] yt = new double[n];
        double[] mut = new double[n];
        for (int split = 0; split < 2; split++) {
            double[][] xs = split == 0 ? x : xt;
            double[] es = split == 0 ? ex : ext;
            double[] ys = split == 0 ? y : yt;
            for (int i = 0; i < n; i++) {
                double eta = beta[0];
                for (int j = 0; j < p; j++) {
                    xs[i][j] = r.nextGaussian() * 0.8;
                    eta += beta[j + 1] * xs[i][j];
                }
                es[i] = 0.5 + 1.5 * r.nextDouble();
                double mu = es[i] * Math.exp(eta);
                if (split == 1) {
                    mut[i] = mu;
                }
                ys[i] = switch (family) {
                    case POISSON -> poisson(mu, r);
                    case NEGATIVE_BINOMIAL -> dispersion <= 1 ? poisson(mu, r)
                            : poisson(mu * gamma(1 / (dispersion - 1), r) * (dispersion - 1), r);
                    case ZERO_INFLATED_POISSON -> r.nextDouble() < zeroShare ? 0 : poisson(mu, r);
                    case TWEEDIE -> tweedie(mu, dispersion, 1.5, r);
                };
            }
        }
        return new Counts(x, ex, y, xt, ext, yt, beta, mut);
    }

    /**
     * A Tweedie compound Poisson-gamma draw.
     *
     * @param mu the mean
     * @param phi the dispersion
     * @param power the power in {@code (1, 2)}
     * @param r the random source
     * @return the amount, exactly zero with probability {@code exp(-lambda)}
     */
    public static double tweedie(double mu, double phi, double power, SplittableRandom r) {
        double lambda = Math.pow(mu, 2 - power) / (phi * (2 - power));
        double shape = (2 - power) / (power - 1);
        double scale = phi * (power - 1) * Math.pow(mu, power - 1);
        int k = poisson(lambda, r);
        double s = 0;
        for (int i = 0; i < k; i++) {
            s += gamma(shape, r) * scale;
        }
        return s;
    }

    /**
     * A nonlinear feature-selection task.
     *
     * @param x features {@code [n][p]}
     * @param y binary labels
     * @param relevant the truly relevant features
     * @param redundant noisy copies of relevant features (relevant to y, but redundant)
     */
    public record Selection(double[][] x, int[] y, int[] relevant, int[] redundant) {
    }

    /**
     * A binary task whose label depends on an XOR pair {@code (x0, x1)}, a sine feature
     * {@code x2}, and linear features {@code x3, x4}: {@code y = 1[2 sign(x0 x1) + 1.5 sin(2 x2)
     * + x3 - x4 + 0.3 eps > 0]}. Features {@code 5 .. 4 + copies} are noisy copies of
     * {@code x3}; the rest are noise.
     *
     * @param n rows
     * @param p features (at least {@code 5 + copies})
     * @param copies redundant copies of {@code x3}
     * @param seed the seed
     * @return the task
     */
    public static Selection nonlinear(int n, int p, int copies, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][p];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                x[i][j] = r.nextGaussian();
            }
            for (int c = 0; c < copies; c++) {
                x[i][5 + c] = x[i][3] + 0.1 * r.nextGaussian();
            }
            double f = 2 * Math.signum(x[i][0] * x[i][1]) + 1.5 * Math.sin(2 * x[i][2]) + x[i][3] - x[i][4]
                    + 0.3 * r.nextGaussian();
            y[i] = f > 0 ? 1 : 0;
        }
        int[] red = new int[copies];
        for (int c = 0; c < copies; c++) {
            red[c] = 5 + c;
        }
        return new Selection(x, y, new int[] {0, 1, 2, 3, 4}, red);
    }

    /** Where the non-null hypotheses sit in a p-value stream. */
    public enum Placement {
        /** Uniformly at random. */
        UNIFORM,
        /** Concentrated in the first fifth of the stream. */
        EARLY,
        /** Concentrated in the last fifth of the stream. */
        LATE
    }

    /**
     * A stream of one-sided z-tests.
     *
     * @param p p-values {@code 1 - Phi(z)}
     * @param e likelihood-ratio e-values {@code exp(mu z - mu^2 / 2)}, valid under the null
     * @param z the z-statistics
     * @param isNull whether each hypothesis is null
     */
    public record PValues(double[] p, double[] e, double[] z, boolean[] isNull) {
    }

    /**
     * Draws {@code m} z-statistics, {@code N(0, 1)} under the null and {@code N(mu, 1)}
     * otherwise, equicorrelated with {@code rho} inside consecutive blocks of {@code block}.
     *
     * @param m hypotheses
     * @param pi0 the null proportion
     * @param mu the alternative mean
     * @param rho the within-block correlation (0 for independence)
     * @param block the block size
     * @param placement where the non-nulls go
     * @param seed the seed
     * @return the stream
     */
    public static PValues pValues(int m, double pi0, double mu, double rho, int block, Placement placement,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        boolean[] nul = new boolean[m];
        int k = (int) Math.round((1 - pi0) * m);
        java.util.Arrays.fill(nul, true);
        int lo = placement == Placement.LATE ? m - Math.max(k, m / 5) : 0;
        int hi = placement == Placement.EARLY ? Math.max(k, m / 5) : m;
        if (placement == Placement.UNIFORM) {
            lo = 0;
            hi = m;
        }
        int[] perm = permutation(hi - lo, r);
        for (int a = 0; a < k; a++) {
            nul[lo + perm[a]] = false;
        }
        double[] z = new double[m];
        double shared = 0;
        for (int i = 0; i < m; i++) {
            if (i % Math.max(1, block) == 0) {
                shared = r.nextGaussian();
            }
            z[i] = Math.sqrt(rho) * shared + Math.sqrt(1 - rho) * r.nextGaussian() + (nul[i] ? 0 : mu);
        }
        double[] p = new double[m];
        double[] e = new double[m];
        for (int i = 0; i < m; i++) {
            p[i] = upperNormal(z[i]);
            e[i] = Math.exp(mu * z[i] - mu * mu / 2);
        }
        return new PValues(p, e, z, nul);
    }

    /**
     * The upper normal tail {@code 1 - Phi(z)}.
     *
     * @param z the statistic
     * @return the one-sided p-value
     */
    public static double upperNormal(double z) {
        double two = Distributions.normalTwoSided(z);
        return z >= 0 ? two / 2 : 1 - two / 2;
    }

    // ---- helpers

    private static double[][] toeplitzDesign(int n, int p, double rho, SplittableRandom r) {
        double[][] x = new double[n][p];
        double c = Math.sqrt(1 - rho * rho);
        for (int i = 0; i < n; i++) {
            x[i][0] = r.nextGaussian();
            for (int j = 1; j < p; j++) {
                x[i][j] = rho * x[i][j - 1] + c * r.nextGaussian();
            }
        }
        return x;
    }

    private static double[][] groupDesign(int n, int groups, int size, double rho, SplittableRandom r) {
        double[][] x = new double[n][groups * size];
        for (int i = 0; i < n; i++) {
            for (int g = 0; g < groups; g++) {
                double shared = r.nextGaussian();
                for (int k = 0; k < size; k++) {
                    x[i][g * size + k] = Math.sqrt(rho) * shared + Math.sqrt(1 - rho) * r.nextGaussian();
                }
            }
        }
        return x;
    }

    private static double[] respond(double[][] x, double[] beta, double sigma, Noise noise, SplittableRandom r) {
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double s = 0;
            for (int j = 0; j < beta.length; j++) {
                s += x[i][j] * beta[j];
            }
            double eps = noise == Noise.GAUSSIAN ? r.nextGaussian()
                    : 0.6 * r.nextGaussian() / Math.sqrt(gamma(1, r)); // t2 = Z / sqrt(chi2_2 / 2)
            y[i] = s + sigma * eps;
        }
        return y;
    }

    /**
     * A uniformly random permutation.
     *
     * @param n the size
     * @param r the random source
     * @return a permutation of {@code 0 .. n-1}
     */
    public static int[] permutation(int n, SplittableRandom r) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
        return a;
    }

    /**
     * A Poisson draw (Knuth's method below 30, a rounded normal above).
     *
     * @param lambda the mean
     * @param r the random source
     * @return the count
     */
    public static int poisson(double lambda, SplittableRandom r) {
        if (lambda <= 0) {
            return 0;
        }
        if (lambda > 30) {
            return (int) Math.max(0, Math.round(lambda + Math.sqrt(lambda) * r.nextGaussian()));
        }
        double l = Math.exp(-lambda);
        int k = 0;
        double prod = r.nextDouble();
        while (prod > l) {
            k++;
            prod *= r.nextDouble();
        }
        return k;
    }

    /**
     * A unit-scale gamma draw by Marsaglia and Tsang, boosted for shapes below 1.
     *
     * @param shape the shape
     * @param r the random source
     * @return the draw
     */
    public static double gamma(double shape, SplittableRandom r) {
        if (shape < 1) {
            return gamma(shape + 1, r) * Math.pow(r.nextDouble(), 1 / shape);
        }
        double d = shape - 1.0 / 3;
        double c = 1 / Math.sqrt(9 * d);
        while (true) {
            double z = r.nextGaussian();
            double v = 1 + c * z;
            if (v <= 0) {
                continue;
            }
            v = v * v * v;
            if (Math.log(r.nextDouble()) < 0.5 * z * z + d - d * v + d * Math.log(v)) {
                return d * v;
            }
        }
    }
}
