package io.github.zzuegg.brainiac.experimental.stats;

import io.github.zzuegg.brainiac.experimental.discovery.LeastSquares;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Penalised linear and logistic regression by cyclic coordinate descent, the glmnet recipe
 * (Friedman, Hastie and Tibshirani, J. Stat. Softw. 2010): the objective
 * {@code (1/2n) ||y - b0 - X b||^2 + lambda P(b)} (or the mean logistic deviance / 2 through an
 * IRLS outer loop) is minimised one coordinate (or one group) at a time on standardised
 * columns, along a decreasing {@code lambda} grid with warm starts, sequential strong-rule
 * screening and active-set cycling with a KKT check.
 *
 * <p>Penalties: the lasso (Tibshirani 1996), the elastic net {@code alpha |b| + (1-alpha) b^2/2}
 * (Zou and Hastie 2005), the folded-concave SCAD (Fan and Li 2001) and MCP (Zhang 2010) with the
 * closed-form thresholding of Breheny and Huang (2011), the adaptive lasso {@code |b_j| /
 * |b_init_j|^gamma} with a ridge pilot (Zou 2006), and the group and sparse-group lasso
 * {@code sqrt(p_g) ||b_g||_2} (Yuan and Lin 2006), solved by majorised block updates. Lambda is
 * picked by K-fold cross-validation ({@code lambda_min} or the one-standard-error rule) or by
 * BIC / extended BIC (Chen and Chen 2008). Coefficients are reported on the original scale.
 *
 * <p>Source: <a href="https://arxiv.org/abs/math/0406456">arXiv:math/0406456</a> (LARS, the exact
 * path this solver is checked against; see {@link Lars}).
 */
public final class PenalizedRegression {

    /** The penalty. */
    public enum Penalty {
        /** {@code lambda |b|}. */
        LASSO,
        /** {@code lambda (alpha |b| + (1 - alpha) b^2 / 2)}. */
        ELASTIC_NET,
        /** SCAD with concavity {@code gamma} (default 3.7). */
        SCAD,
        /** MCP with concavity {@code gamma} (default 3). */
        MCP,
        /** Lasso weighted by {@code 1 / |b_ridge|^gamma} (default gamma 1). */
        ADAPTIVE_LASSO,
        /** {@code lambda sqrt(p_g) ||b_g||_2} over the groups. */
        GROUP_LASSO,
        /** {@code lambda (alpha |b| + (1 - alpha) sqrt(p_g) ||b_g||_2)}. */
        SPARSE_GROUP_LASSO
    }

    /** The response family. */
    public enum Family {
        /** Squared error. */
        GAUSSIAN,
        /** Binary logistic regression (labels 0/1). */
        LOGISTIC
    }

    /**
     * A fit at one lambda, coefficients on the original scale.
     *
     * @param lambda the penalty level
     * @param intercept the intercept
     * @param beta the coefficients
     */
    public record Fit(double lambda, double intercept, double[] beta) {

        /**
         * The linear predictor.
         *
         * @param x rows
         * @return {@code b0 + x b} per row
         */
        public double[] predict(double[][] x) {
            double[] out = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                double s = intercept;
                for (int j = 0; j < beta.length; j++) {
                    s += x[i][j] * beta[j];
                }
                out[i] = s;
            }
            return out;
        }

        /**
         * The number of non-zero coefficients.
         *
         * @return the degrees of freedom of a lasso fit
         */
        public int nonZero() {
            int k = 0;
            for (double b : beta) {
                if (b != 0) {
                    k++;
                }
            }
            return k;
        }

        /**
         * The selected support.
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
     * A regularisation path.
     *
     * @param lambdas the decreasing grid
     * @param fits one fit per lambda
     * @param updates coordinate (or group) updates spent on the whole path
     */
    public record Path(double[] lambdas, List<Fit> fits, long updates) {
    }

    /**
     * A cross-validated choice.
     *
     * @param path the path on all data
     * @param cvMean the mean validation loss per lambda
     * @param cvSe its standard error
     * @param best the index of {@code lambda_min}
     * @param oneSe the index of the one-standard-error choice
     */
    public record CrossValidation(Path path, double[] cvMean, double[] cvSe, int best, int oneSe) {

        /**
         * The fit at {@code lambda_min}.
         *
         * @return the fit
         */
        public Fit min() {
            return path.fits().get(best);
        }

        /**
         * The fit with the largest lambda within one standard error of the minimum.
         *
         * @return the fit
         */
        public Fit oneStandardError() {
            return path.fits().get(oneSe);
        }
    }

    private final Penalty penalty;
    private Family family = Family.GAUSSIAN;
    private double alpha = 0.5;
    private double gamma = Double.NaN;
    private int[] groups;
    private double[] weights;
    private boolean screening = true;
    private double tolerance = 1e-8;
    private int maxPasses = 10_000;

    private PenalizedRegression(Penalty penalty) {
        this.penalty = Objects.requireNonNull(penalty);
    }

    /**
     * Starts a model with a penalty.
     *
     * @param penalty the penalty
     * @return the builder-style model
     */
    public static PenalizedRegression of(Penalty penalty) {
        return new PenalizedRegression(penalty);
    }

    /**
     * Sets the response family.
     *
     * @param f the family
     * @return this
     */
    public PenalizedRegression family(Family f) {
        this.family = Objects.requireNonNull(f);
        return this;
    }

    /**
     * Sets the mixing weight of the elastic net or sparse-group lasso (the l1 share).
     *
     * @param a the mix in {@code (0, 1]}
     * @return this
     */
    public PenalizedRegression alpha(double a) {
        this.alpha = a;
        return this;
    }

    /**
     * Sets the concavity of SCAD/MCP, or the exponent of the adaptive-lasso weights.
     *
     * @param g the parameter
     * @return this
     */
    public PenalizedRegression gamma(double g) {
        this.gamma = g;
        return this;
    }

    /**
     * Sets the group of every feature (group penalties).
     *
     * @param g the group index of each column
     * @return this
     */
    public PenalizedRegression groups(int[] g) {
        this.groups = g.clone();
        return this;
    }

    /**
     * Sets per-feature penalty factors (multiplying lambda); the adaptive lasso computes its own
     * when none are given.
     *
     * @param w the factors
     * @return this
     */
    public PenalizedRegression penaltyFactors(double[] w) {
        this.weights = w.clone();
        return this;
    }

    /**
     * Turns strong-rule screening and active-set cycling on (default) or off (plain cyclic
     * coordinate descent over every feature in every pass, the reference for speed-ups).
     *
     * @param on whether to screen
     * @return this
     */
    public PenalizedRegression screening(boolean on) {
        this.screening = on;
        return this;
    }

    /**
     * Sets the convergence tolerance on the largest coefficient change (standardised scale).
     *
     * @param tol the tolerance
     * @return this
     */
    public PenalizedRegression tolerance(double tol) {
        this.tolerance = tol;
        return this;
    }

    // ------------------------------------------------------------------ public entry points

    /**
     * Fits at one lambda (via a short warm-started path from {@code lambda_max}).
     *
     * @param x the design {@code [n][p]}
     * @param y the response (0/1 for logistic)
     * @param lambda the penalty level
     * @return the fit
     */
    public Fit fit(double[][] x, double[] y, double lambda) {
        Problem pr = new Problem(x, y);
        double lmax = pr.lambdaMax();
        int steps = 20;
        double[] grid = new double[steps + 1];
        for (int k = 0; k < steps; k++) {
            grid[k] = Math.max(lambda, lmax * Math.pow(Math.max(lambda / lmax, 1e-4), k / (double) steps));
        }
        grid[steps] = lambda;
        Path p = pr.path(grid);
        return p.fits().get(steps);
    }

    /**
     * The largest useful lambda, where every coefficient is zero.
     *
     * @param x the design
     * @param y the response
     * @return {@code lambda_max}
     */
    public double lambdaMax(double[][] x, double[] y) {
        return new Problem(x, y).lambdaMax();
    }

    /**
     * Solves a log-spaced path from {@code lambda_max} down to {@code ratio * lambda_max}.
     *
     * @param x the design
     * @param y the response
     * @param count grid points
     * @param ratio the smallest lambda as a share of the largest
     * @return the path
     */
    public Path path(double[][] x, double[] y, int count, double ratio) {
        Problem pr = new Problem(x, y);
        return pr.path(grid(pr.lambdaMax(), count, ratio));
    }

    /**
     * Solves the path over a given decreasing grid.
     *
     * @param x the design
     * @param y the response
     * @param lambdas the grid
     * @return the path
     */
    public Path path(double[][] x, double[] y, double[] lambdas) {
        return new Problem(x, y).path(lambdas.clone());
    }

    /**
     * K-fold cross-validation over a path (mean squared error, or mean deviance for logistic).
     *
     * @param x the design
     * @param y the response
     * @param count grid points
     * @param ratio the smallest lambda share
     * @param folds the number of folds
     * @param seed the fold assignment seed
     * @return the choice
     */
    public CrossValidation crossValidate(double[][] x, double[] y, int count, double ratio, int folds, long seed) {
        Problem all = new Problem(x, y);
        double[] lambdas = grid(all.lambdaMax(), count, ratio);
        Path full = all.path(lambdas);
        int n = y.length;
        int[] fold = new int[n];
        int[] perm = permutation(n, new SplittableRandom(seed));
        for (int i = 0; i < n; i++) {
            fold[perm[i]] = i % folds;
        }
        double[][] loss = new double[folds][lambdas.length];
        for (int f = 0; f < folds; f++) {
            List<Integer> tr = new ArrayList<>();
            List<Integer> va = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                (fold[i] == f ? va : tr).add(i);
            }
            Path p = new Problem(rows(x, tr), values(y, tr)).path(lambdas);
            double[][] xv = rows(x, va);
            double[] yv = values(y, va);
            for (int k = 0; k < lambdas.length; k++) {
                loss[f][k] = loss(p.fits().get(k), xv, yv);
            }
        }
        double[] mean = new double[lambdas.length];
        double[] se = new double[lambdas.length];
        int best = 0;
        for (int k = 0; k < lambdas.length; k++) {
            double s = 0;
            double s2 = 0;
            for (int f = 0; f < folds; f++) {
                s += loss[f][k];
                s2 += loss[f][k] * loss[f][k];
            }
            mean[k] = s / folds;
            se[k] = Math.sqrt(Math.max(0, s2 / folds - mean[k] * mean[k]) / Math.max(1, folds - 1));
            if (mean[k] < mean[best]) {
                best = k;
            }
        }
        int oneSe = best;
        for (int k = 0; k < best; k++) {
            if (mean[k] <= mean[best] + se[best]) {
                oneSe = k;
                break;
            }
        }
        return new CrossValidation(full, mean, se, best, oneSe);
    }

    /**
     * Picks the path fit with the smallest (extended) BIC:
     * {@code n log(RSS / n) + df log n + 2 g df log p} (deviance in place of the first term for
     * logistic). {@code g = 0} gives BIC.
     *
     * @param path the path
     * @param x the design it was fitted on
     * @param y the response
     * @param ebicGamma the EBIC parameter in {@code [0, 1]}
     * @return the chosen fit
     */
    public Fit selectByInformation(Path path, double[][] x, double[] y, double ebicGamma) {
        int n = y.length;
        int p = x[0].length;
        Fit best = null;
        double bestScore = Double.POSITIVE_INFINITY;
        for (Fit f : path.fits()) {
            int df = f.nonZero();
            double fitTerm = family == Family.GAUSSIAN ? n * Math.log(Math.max(1e-300, loss(f, x, y)))
                    : n * loss(f, x, y);
            double score = fitTerm + df * Math.log(n) + 2 * ebicGamma * df * Math.log(p);
            if (score < bestScore) {
                bestScore = score;
                best = f;
            }
        }
        return best;
    }

    /**
     * The mean validation loss of a fit: squared error, or logistic deviance per row.
     *
     * @param f the fit
     * @param x rows
     * @param y responses
     * @return the loss
     */
    public double loss(Fit f, double[][] x, double[] y) {
        double[] eta = f.predict(x);
        double s = 0;
        for (int i = 0; i < y.length; i++) {
            if (family == Family.GAUSSIAN) {
                s += (y[i] - eta[i]) * (y[i] - eta[i]);
            } else {
                double pr = 1 / (1 + Math.exp(-eta[i]));
                pr = Math.min(1 - 1e-12, Math.max(1e-12, pr));
                s += -2 * (y[i] * Math.log(pr) + (1 - y[i]) * Math.log(1 - pr));
            }
        }
        return s / y.length;
    }

    /**
     * A log-spaced decreasing grid.
     *
     * @param max the largest lambda
     * @param count grid points
     * @param ratio the smallest as a share of the largest
     * @return the grid
     */
    public static double[] grid(double max, int count, double ratio) {
        double[] g = new double[count];
        for (int k = 0; k < count; k++) {
            g[k] = max * Math.pow(ratio, k / (double) Math.max(1, count - 1));
        }
        return g;
    }

    /**
     * The soft-thresholding operator {@code sign(z) max(|z| - t, 0)}.
     *
     * @param z the value
     * @param t the threshold
     * @return the shrunk value
     */
    public static double soft(double z, double t) {
        return z > t ? z - t : z < -t ? z + t : 0;
    }

    // ------------------------------------------------------------------ the solver

    /** One standardised problem. */
    private final class Problem {
        final int n;
        final int p;
        final double[][] xs; // column-major standardised design [p][n]
        final double[] mean;
        final double[] sd;
        final double[] y;
        final double ymean;
        final double[] w; // penalty factors
        final int[][] groupMembers;
        final int[] groupOf;
        final double g;

        Problem(double[][] x, double[] yRaw) {
            n = yRaw.length;
            p = x[0].length;
            xs = new double[p][n];
            mean = new double[p];
            sd = new double[p];
            for (int j = 0; j < p; j++) {
                double m = 0;
                for (int i = 0; i < n; i++) {
                    m += x[i][j];
                }
                m /= n;
                double v = 0;
                for (int i = 0; i < n; i++) {
                    double d = x[i][j] - m;
                    v += d * d;
                }
                double s = Math.sqrt(v / n);
                if (s < 1e-12) {
                    s = 1;
                }
                mean[j] = m;
                sd[j] = s;
                for (int i = 0; i < n; i++) {
                    xs[j][i] = (x[i][j] - m) / s;
                }
            }
            double ym = 0;
            for (double v : yRaw) {
                ym += v;
            }
            ymean = ym / n;
            y = yRaw.clone();
            g = Double.isNaN(gamma) ? switch (penalty) {
                case SCAD -> 3.7;
                case MCP -> 3.0;
                default -> 1.0;
            } : gamma;
            if (weights != null) {
                w = weights.clone();
            } else if (penalty == Penalty.ADAPTIVE_LASSO) {
                w = adaptiveWeights();
            } else {
                w = new double[p];
                Arrays.fill(w, 1);
            }
            if (penalty == Penalty.GROUP_LASSO || penalty == Penalty.SPARSE_GROUP_LASSO) {
                if (groups == null || groups.length != p) {
                    throw new IllegalStateException("group penalties need groups(...) for every column");
                }
                int ng = Arrays.stream(groups).max().orElse(-1) + 1;
                List<List<Integer>> mem = new ArrayList<>();
                for (int k = 0; k < ng; k++) {
                    mem.add(new ArrayList<>());
                }
                for (int j = 0; j < p; j++) {
                    mem.get(groups[j]).add(j);
                }
                groupMembers = new int[ng][];
                for (int k = 0; k < ng; k++) {
                    groupMembers[k] = mem.get(k).stream().mapToInt(Integer::intValue).toArray();
                }
                groupOf = groups.clone();
            } else {
                groupMembers = null;
                groupOf = null;
            }
        }

        private double[] adaptiveWeights() {
            double[][] rowsStd = new double[n][p];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < p; j++) {
                    rowsStd[i][j] = xs[j][i];
                }
            }
            double[] yc = new double[n];
            for (int i = 0; i < n; i++) {
                yc[i] = y[i] - ymean;
            }
            // a ridge pilot (n > p: nearly OLS); weights 1 / |b|^gamma, capped
            LeastSquares.Fit pilot = LeastSquares.fit(rowsStd, yc, false, Math.max(1e-3 * n, n > p ? 1e-6 : n * 0.1));
            double[] out = new double[p];
            for (int j = 0; j < p; j++) {
                out[j] = Math.min(1e4, 1 / Math.pow(Math.abs(pilot.beta()[j]) + 1e-6, g));
            }
            double s = 0;
            for (double v : out) {
                s += v;
            }
            for (int j = 0; j < p; j++) {
                out[j] *= p / s;
            }
            return out;
        }

        double l1Share() {
            return switch (penalty) {
                case ELASTIC_NET, SPARSE_GROUP_LASSO -> alpha;
                case GROUP_LASSO -> 0;
                default -> 1;
            };
        }

        double lambdaMax() {
            double[] r = new double[n];
            double centre = family == Family.GAUSSIAN ? ymean : ymean;
            for (int i = 0; i < n; i++) {
                r[i] = y[i] - centre;
            }
            if (penalty == Penalty.GROUP_LASSO) {
                double best = 0;
                for (int[] grp : groupMembers) {
                    double s = 0;
                    for (int j : grp) {
                        double c = dot(xs[j], r) / n;
                        s += c * c;
                    }
                    best = Math.max(best, Math.sqrt(s) / Math.sqrt(grp.length));
                }
                return best;
            }
            double best = 0;
            double a = l1Share();
            for (int j = 0; j < p; j++) {
                if (w[j] > 0) {
                    best = Math.max(best, Math.abs(dot(xs[j], r)) / n / (a * w[j]));
                }
            }
            return best;
        }

        Path path(double[] lambdas) {
            double[] b = new double[p];
            double b0 = family == Family.GAUSSIAN ? ymean : Math.log(ymean / (1 - ymean));
            List<Fit> fits = new ArrayList<>();
            long[] updates = {0};
            double prev = lambdaMax();
            for (double lam : lambdas) {
                b0 = family == Family.GAUSSIAN ? solveGaussian(b, lam, prev, updates)
                        : solveLogistic(b, b0, lam, prev, updates);
                fits.add(toOriginal(lam, b0, b));
                prev = lam;
            }
            return new Path(lambdas, fits, updates[0]);
        }

        private Fit toOriginal(double lam, double b0, double[] b) {
            double[] beta = new double[p];
            double icpt = b0;
            for (int j = 0; j < p; j++) {
                beta[j] = b[j] / sd[j];
                icpt -= beta[j] * mean[j];
            }
            return new Fit(lam, icpt, beta);
        }

        // Gaussian: weights all 1, curvature v_j = 1 on standardised columns
        private double solveGaussian(double[] b, double lam, double prevLam, long[] updates) {
            double[] r = new double[n];
            for (int i = 0; i < n; i++) {
                r[i] = y[i] - ymean;
            }
            for (int j = 0; j < p; j++) {
                if (b[j] != 0) {
                    axpy(-b[j], xs[j], r);
                }
            }
            double[] wt = null;
            double[] v = new double[p];
            Arrays.fill(v, 1);
            coordinateDescent(b, r, wt, v, lam, prevLam, updates);
            return ymean;
        }

        private double solveLogistic(double[] b, double b0Start, double lam, double prevLam, long[] updates) {
            double b0 = b0Start;
            double[] eta = new double[n];
            for (int outer = 0; outer < 50; outer++) {
                for (int i = 0; i < n; i++) {
                    double s = b0;
                    for (int j = 0; j < p; j++) {
                        if (b[j] != 0) {
                            s += xs[j][i] * b[j];
                        }
                    }
                    eta[i] = s;
                }
                double[] wt = new double[n];
                double[] r = new double[n]; // working residual z - eta
                for (int i = 0; i < n; i++) {
                    double pr = 1 / (1 + Math.exp(-eta[i]));
                    double wi = Math.max(pr * (1 - pr), 1e-5);
                    wt[i] = wi;
                    r[i] = (y[i] - pr) / wi;
                }
                double[] v = new double[p];
                for (int j = 0; j < p; j++) {
                    double s = 0;
                    for (int i = 0; i < n; i++) {
                        s += wt[i] * xs[j][i] * xs[j][i];
                    }
                    v[j] = s / n;
                }
                double[] old = b.clone();
                double oldB0 = b0;
                // intercept step, then coordinates, then intercept again
                b0 += interceptStep(r, wt);
                coordinateDescent(b, r, wt, v, lam, prevLam, updates);
                b0 += interceptStep(r, wt);
                double change = Math.abs(b0 - oldB0);
                for (int j = 0; j < p; j++) {
                    change = Math.max(change, Math.abs(b[j] - old[j]));
                }
                if (change < 1e-6) {
                    break;
                }
            }
            return b0;
        }

        private double interceptStep(double[] r, double[] wt) {
            double num = 0;
            double den = 0;
            for (int i = 0; i < n; i++) {
                num += wt[i] * r[i];
                den += wt[i];
            }
            double d = num / den;
            for (int i = 0; i < n; i++) {
                r[i] -= d;
            }
            return d;
        }

        /** Coordinate (or block) descent on a weighted quadratic, updating b and r in place. */
        private void coordinateDescent(double[] b, double[] r, double[] wt, double[] v, double lam, double prevLam,
                long[] updates) {
            boolean grouped = groupMembers != null;
            int units = grouped ? groupMembers.length : p;
            boolean[] eligible = new boolean[units];
            if (screening) {
                double a = l1Share();
                for (int u = 0; u < units; u++) {
                    if (grouped) {
                        int[] grp = groupMembers[u];
                        boolean nz = false;
                        double s = 0;
                        for (int j : grp) {
                            nz |= b[j] != 0;
                            double c = grad(j, r, wt);
                            s += c * c;
                        }
                        double thr = Math.sqrt(grp.length) * (2 * lam - prevLam);
                        if (penalty == Penalty.SPARSE_GROUP_LASSO) {
                            thr = Math.min(thr * (1 - alpha) + alpha * (2 * lam - prevLam), thr);
                        }
                        eligible[u] = nz || Math.sqrt(s) >= thr;
                    } else {
                        eligible[u] = b[u] != 0 || Math.abs(grad(u, r, wt)) >= a * w[u] * (2 * lam - prevLam);
                    }
                }
            } else {
                Arrays.fill(eligible, true);
            }
            while (true) {
                // converge on the eligible set, cycling on the active set after one full pass
                for (int pass = 0; pass < maxPasses; pass++) {
                    double change = sweep(b, r, wt, v, lam, eligible, false, updates);
                    if (change < tolerance) {
                        break;
                    }
                    if (screening) {
                        for (int inner = 0; inner < maxPasses; inner++) {
                            if (sweep(b, r, wt, v, lam, eligible, true, updates) < tolerance) {
                                break;
                            }
                        }
                    }
                }
                if (!screening) {
                    return;
                }
                // KKT check on the screened-out units
                boolean violated = false;
                for (int u = 0; u < units; u++) {
                    if (eligible[u]) {
                        continue;
                    }
                    if (grouped) {
                        double s = 0;
                        double sl1 = 0;
                        for (int j : groupMembers[u]) {
                            double c = grad(j, r, wt);
                            s += c * c;
                            double st = soft(c, lam * (penalty == Penalty.SPARSE_GROUP_LASSO ? alpha : 0));
                            sl1 += st * st;
                        }
                        double norm = penalty == Penalty.SPARSE_GROUP_LASSO ? Math.sqrt(sl1) : Math.sqrt(s);
                        double thr = lam * Math.sqrt(groupMembers[u].length)
                                * (penalty == Penalty.SPARSE_GROUP_LASSO ? 1 - alpha : 1);
                        if (norm > thr + 1e-12) {
                            eligible[u] = true;
                            violated = true;
                        }
                    } else if (Math.abs(grad(u, r, wt)) > l1Share() * lam * w[u] + 1e-12) {
                        eligible[u] = true;
                        violated = true;
                    }
                }
                if (!violated) {
                    return;
                }
            }
        }

        /** {@code (1/n) sum_i w_i x_ij r_i}. */
        private double grad(int j, double[] r, double[] wt) {
            double s = 0;
            double[] col = xs[j];
            if (wt == null) {
                for (int i = 0; i < n; i++) {
                    s += col[i] * r[i];
                }
            } else {
                for (int i = 0; i < n; i++) {
                    s += wt[i] * col[i] * r[i];
                }
            }
            return s / n;
        }

        private double sweep(double[] b, double[] r, double[] wt, double[] v, double lam, boolean[] eligible,
                boolean activeOnly, long[] updates) {
            double maxChange = 0;
            if (groupMembers != null) {
                for (int u = 0; u < groupMembers.length; u++) {
                    if (!eligible[u]) {
                        continue;
                    }
                    int[] grp = groupMembers[u];
                    if (activeOnly) {
                        boolean nz = false;
                        for (int j : grp) {
                            nz |= b[j] != 0;
                        }
                        if (!nz) {
                            continue;
                        }
                    }
                    updates[0]++;
                    maxChange = Math.max(maxChange, groupUpdate(grp, b, r, wt, v, lam));
                }
                return maxChange;
            }
            for (int j = 0; j < p; j++) {
                if (!eligible[j] || (activeOnly && b[j] == 0)) {
                    continue;
                }
                updates[0]++;
                double z = grad(j, r, wt) + v[j] * b[j];
                double nb = threshold(z, v[j], lam * w[j]);
                double d = nb - b[j];
                if (d != 0) {
                    if (wt == null) {
                        axpy(-d, xs[j], r);
                    } else {
                        // r is the working residual z - eta: eta moves by d x_j
                        axpy(-d, xs[j], r);
                    }
                    b[j] = nb;
                    maxChange = Math.max(maxChange, Math.abs(d) * Math.sqrt(v[j]));
                }
            }
            return maxChange;
        }

        /** Minimises {@code (v/2)(b - z/v)^2 + pen(b)} in closed form. */
        private double threshold(double z, double v, double lam) {
            double az = Math.abs(z);
            return switch (penalty) {
                case LASSO, ADAPTIVE_LASSO -> soft(z, lam) / v;
                case ELASTIC_NET -> soft(z, lam * alpha) / (v + lam * (1 - alpha));
                // the curvature guards keep the update convex when IRLS weights are small (logistic)
                case MCP -> az <= g * lam * v ? soft(z, lam) / Math.max(v - 1 / g, 0.1 * v) : z / v;
                case SCAD -> az <= lam * (v + 1) ? soft(z, lam) / v
                        : az <= v * g * lam ? soft(z, g * lam / (g - 1)) / Math.max(v - 1 / (g - 1), 0.1 * v)
                        : z / v;
                default -> throw new IllegalStateException();
            };
        }

        /** A majorised block update: one proximal step with the block's Lipschitz constant. */
        private double groupUpdate(int[] grp, double[] b, double[] r, double[] wt, double[] v, double lam) {
            double lip = blockLipschitz(grp, wt);
            int k = grp.length;
            double[] z = new double[k];
            for (int a = 0; a < k; a++) {
                z[a] = b[grp[a]] + grad(grp[a], r, wt) / lip;
            }
            double l1 = penalty == Penalty.SPARSE_GROUP_LASSO ? lam * alpha / lip : 0;
            double l2 = lam * Math.sqrt(k) * (penalty == Penalty.SPARSE_GROUP_LASSO ? 1 - alpha : 1) / lip;
            double norm = 0;
            for (int a = 0; a < k; a++) {
                z[a] = soft(z[a], l1);
                norm += z[a] * z[a];
            }
            norm = Math.sqrt(norm);
            double scale = norm <= l2 ? 0 : 1 - l2 / norm;
            double change = 0;
            for (int a = 0; a < k; a++) {
                int j = grp[a];
                double nb = z[a] * scale;
                double d = nb - b[j];
                if (d != 0) {
                    axpy(-d, xs[j], r);
                    b[j] = nb;
                    change = Math.max(change, Math.abs(d));
                }
            }
            return change;
        }

        private final java.util.Map<Integer, Double> lipCache = new java.util.HashMap<>();

        private double blockLipschitz(int[] grp, double[] wt) {
            if (wt == null && lipCache.containsKey(grp[0])) {
                return lipCache.get(grp[0]);
            }
            int k = grp.length;
            double[][] gram = new double[k][k];
            for (int a = 0; a < k; a++) {
                for (int c = a; c < k; c++) {
                    double s = 0;
                    for (int i = 0; i < n; i++) {
                        s += (wt == null ? 1 : wt[i]) * xs[grp[a]][i] * xs[grp[c]][i];
                    }
                    gram[a][c] = gram[c][a] = s / n;
                }
            }
            double[] q = new double[k];
            Arrays.fill(q, 1 / Math.sqrt(k));
            double lambda = 0;
            for (int it = 0; it < 50; it++) {
                double[] nq = new double[k];
                for (int a = 0; a < k; a++) {
                    for (int c = 0; c < k; c++) {
                        nq[a] += gram[a][c] * q[c];
                    }
                }
                double norm = Math.sqrt(dot(nq, nq));
                if (norm == 0) {
                    break;
                }
                lambda = norm;
                for (int a = 0; a < k; a++) {
                    q[a] = nq[a] / norm;
                }
            }
            double lip = Math.max(lambda * 1.0001, 1e-8);
            if (wt == null) {
                lipCache.put(grp[0], lip);
            }
            return lip;
        }
    }

    // ------------------------------------------------------------------ small helpers

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static void axpy(double a, double[] x, double[] y) {
        for (int i = 0; i < y.length; i++) {
            y[i] += a * x[i];
        }
    }

    private static int[] permutation(int n, SplittableRandom r) {
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

    private static double[][] rows(double[][] x, List<Integer> idx) {
        double[][] out = new double[idx.size()][];
        for (int k = 0; k < out.length; k++) {
            out[k] = x[idx.get(k)];
        }
        return out;
    }

    private static double[] values(double[] y, List<Integer> idx) {
        double[] out = new double[idx.size()];
        for (int k = 0; k < out.length; k++) {
            out[k] = y[idx.get(k)];
        }
        return out;
    }
}
