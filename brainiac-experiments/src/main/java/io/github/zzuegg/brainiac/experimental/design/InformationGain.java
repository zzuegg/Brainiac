package io.github.zzuegg.brainiac.experimental.design;

import io.github.zzuegg.brainiac.experimental.data.SimulatorSuite.DesignProblem;
import io.github.zzuegg.brainiac.experimental.estimators.MutualInformation;
import io.github.zzuegg.brainiac.tensor.Matrices;

import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Estimators of the expected information gain {@code EIG(xi) = E[log p(y | theta, xi) - log
 * p(y | xi)]} of a design (Lindley 1956; Foster et al., NeurIPS 2019,
 * <a href="https://arxiv.org/abs/1903.05480">arXiv:1903.05480</a>):
 * <ul>
 *   <li>{@link #nmc}: nested Monte Carlo, consistent but biased upwards at finite inner
 *       samples, with {@code N M} likelihood evaluations;</li>
 *   <li>{@link #pce}: the prior contrastive estimation lower bound, which includes the
 *       generating {@code theta_0} among {@code L + 1} contrasts (at most
 *       {@code log(L + 1)});</li>
 *   <li>{@link #nce}: the nested contrastive upper bound, the same without {@code theta_0};</li>
 *   <li>{@link #posterior}: the Barber-Agakov lower bound {@code E[log q(theta | y) - log
 *       p(theta)]} with an amortised Gaussian {@code q} whose mean is polynomial in {@code y},
 *       fitted on one sample and evaluated on another;</li>
 *   <li>{@link #marginal}: the upper bound {@code E[log p(y | theta) - log q(y)]} with a
 *       Gaussian mixture {@code q(y)} fitted by EM (for continuous outcomes);</li>
 *   <li>{@link #mine}: the neural Donsker-Varadhan bound of MINEBED (Kleinegesse and Gutmann,
 *       ICML 2020) through {@link MutualInformation#variational}, which needs only
 *       simulations.</li>
 * </ul>
 * {@link #sequentialPce} and {@link #sequentialNce} bound the total information of a whole
 * adaptive history (Foster et al., ICML 2021).
 */
public final class InformationGain {

    private InformationGain() {
    }

    private static double logMeanExp(double[] v, int from, int to) {
        double m = Double.NEGATIVE_INFINITY;
        for (int i = from; i < to; i++) {
            m = Math.max(m, v[i]);
        }
        double s = 0;
        for (int i = from; i < to; i++) {
            s += Math.exp(v[i] - m);
        }
        return m + Math.log(s / (to - from));
    }

    /**
     * Nested Monte Carlo.
     *
     * @param problem the problem
     * @param design the design
     * @param outer outer samples {@code N}
     * @param inner inner samples {@code M} per outer sample
     * @param random the random source
     * @return the estimate in nats
     */
    public static double nmc(DesignProblem problem, double[] design, int outer, int inner, SplittableRandom random) {
        double s = 0;
        double[] l = new double[inner];
        for (int n = 0; n < outer; n++) {
            double[] t = problem.samplePrior(random);
            double y = problem.simulate(t, design, random);
            for (int m = 0; m < inner; m++) {
                l[m] = problem.logLikelihood(y, problem.samplePrior(random), design);
            }
            s += problem.logLikelihood(y, t, design) - logMeanExp(l, 0, inner);
        }
        return s / outer;
    }

    private static double contrastive(DesignProblem problem, double[] design, int outer, int contrasts,
            boolean includeTruth, SplittableRandom random) {
        double s = 0;
        double[] l = new double[contrasts + 1];
        for (int n = 0; n < outer; n++) {
            double[] t = problem.samplePrior(random);
            double y = problem.simulate(t, design, random);
            l[0] = problem.logLikelihood(y, t, design);
            for (int m = 1; m <= contrasts; m++) {
                l[m] = problem.logLikelihood(y, problem.samplePrior(random), design);
            }
            s += l[0] - logMeanExp(l, includeTruth ? 0 : 1, contrasts + 1);
        }
        return s / outer;
    }

    /**
     * The PCE lower bound.
     *
     * @param problem the problem
     * @param design the design
     * @param outer outer samples
     * @param contrasts contrastive samples {@code L}
     * @param random the random source
     * @return the bound in nats
     */
    public static double pce(DesignProblem problem, double[] design, int outer, int contrasts,
            SplittableRandom random) {
        return contrastive(problem, design, outer, contrasts, true, random);
    }

    /**
     * The NCE upper bound (in expectation).
     *
     * @param problem the problem
     * @param design the design
     * @param outer outer samples
     * @param contrasts contrastive samples {@code L}
     * @param random the random source
     * @return the bound in nats
     */
    public static double nce(DesignProblem problem, double[] design, int outer, int contrasts,
            SplittableRandom random) {
        return contrastive(problem, design, outer, contrasts, false, random);
    }

    private static double[] features(double y, double mean, double sd, int degree) {
        double z = (y - mean) / sd;
        double[] f = new double[degree + 1];
        f[0] = 1;
        for (int k = 1; k <= degree; k++) {
            f[k] = f[k - 1] * z;
        }
        return f;
    }

    /**
     * The Barber-Agakov posterior lower bound.
     *
     * @param problem the problem
     * @param priorLogDensity the prior's log density
     * @param design the design
     * @param samples samples for fitting and as many for evaluation
     * @param degree the polynomial degree of the posterior mean in {@code y}
     * @param random the random source
     * @return the bound in nats
     */
    public static double posterior(DesignProblem problem, ToDoubleFunction<double[]> priorLogDensity, double[] design,
            int samples, int degree, SplittableRandom random) {
        int d = problem.parameterDim();
        int f = degree + 1;
        double[][] th = new double[samples][];
        double[] y = new double[samples];
        double ym = 0;
        for (int i = 0; i < samples; i++) {
            th[i] = problem.samplePrior(random);
            y[i] = problem.simulate(th[i], design, random);
            ym += y[i] / samples;
        }
        double yv = 0;
        for (double v : y) {
            yv += (v - ym) * (v - ym) / samples;
        }
        double ys = Math.sqrt(Math.max(yv, 1e-12));
        double[][] ftf = new double[f][f];
        double[][] fth = new double[f][d];
        for (int i = 0; i < samples; i++) {
            double[] phi = features(y[i], ym, ys, degree);
            for (int a = 0; a < f; a++) {
                for (int b = 0; b < f; b++) {
                    ftf[a][b] += phi[a] * phi[b];
                }
                for (int c = 0; c < d; c++) {
                    fth[a][c] += phi[a] * th[i][c];
                }
            }
        }
        for (int a = 0; a < f; a++) {
            ftf[a][a] += 1e-6 * samples;
        }
        double[][] inv = Matrices.inverse(ftf);
        double[][] coef = Matrices.multiply(inv, fth);
        double[][] cov = new double[d][d];
        for (int i = 0; i < samples; i++) {
            double[] res = residual(th[i], features(y[i], ym, ys, degree), coef);
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    cov[a][b] += res[a] * res[b] / samples;
                }
            }
        }
        for (int a = 0; a < d; a++) {
            cov[a][a] += 1e-9;
        }
        double[][] prec = Matrices.inverse(cov);
        double logNorm = -0.5 * Math.log(Matrices.determinant(cov)) - 0.5 * d * Math.log(2 * Math.PI);
        double s = 0;
        for (int i = 0; i < samples; i++) {
            double[] t = problem.samplePrior(random);
            double yy = problem.simulate(t, design, random);
            double[] res = residual(t, features(yy, ym, ys, degree), coef);
            double q = 0;
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    q += res[a] * prec[a][b] * res[b];
                }
            }
            s += logNorm - 0.5 * q - priorLogDensity.applyAsDouble(t);
        }
        return s / samples;
    }

    private static double[] residual(double[] theta, double[] phi, double[][] coef) {
        double[] r = theta.clone();
        for (int c = 0; c < r.length; c++) {
            for (int a = 0; a < phi.length; a++) {
                r[c] -= phi[a] * coef[a][c];
            }
        }
        return r;
    }

    /**
     * The marginal upper bound with a Gaussian-mixture {@code q(y)}; valid for continuous
     * outcomes.
     *
     * @param problem the problem
     * @param design the design
     * @param samples samples for fitting and as many for evaluation
     * @param components mixture components
     * @param random the random source
     * @return the bound in nats
     */
    public static double marginal(DesignProblem problem, double[] design, int samples, int components,
            SplittableRandom random) {
        double[] y = new double[samples];
        for (int i = 0; i < samples; i++) {
            y[i] = problem.simulate(problem.samplePrior(random), design, random);
        }
        double[] sorted = y.clone();
        java.util.Arrays.sort(sorted);
        int k = components;
        double[] w = new double[k];
        double[] mu = new double[k];
        double[] var = new double[k];
        double tot = 0;
        double m0 = 0;
        for (double v : y) {
            m0 += v / samples;
        }
        for (double v : y) {
            tot += (v - m0) * (v - m0) / samples;
        }
        for (int c = 0; c < k; c++) {
            w[c] = 1.0 / k;
            mu[c] = sorted[(int) ((c + 0.5) / k * samples)];
            var[c] = Math.max(tot / (k * k), 1e-6);
        }
        double[][] resp = new double[samples][k];
        for (int it = 0; it < 100; it++) {
            for (int i = 0; i < samples; i++) {
                double[] l = new double[k];
                for (int c = 0; c < k; c++) {
                    l[c] = Math.log(w[c]) + logNormal(y[i], mu[c], var[c]);
                }
                double z = logMeanExp(l, 0, k) + Math.log(k);
                for (int c = 0; c < k; c++) {
                    resp[i][c] = Math.exp(l[c] - z);
                }
            }
            for (int c = 0; c < k; c++) {
                double nk = 1e-9;
                double sm = 0;
                for (int i = 0; i < samples; i++) {
                    nk += resp[i][c];
                    sm += resp[i][c] * y[i];
                }
                mu[c] = sm / nk;
                double sv = 0;
                for (int i = 0; i < samples; i++) {
                    sv += resp[i][c] * (y[i] - mu[c]) * (y[i] - mu[c]);
                }
                var[c] = Math.max(sv / nk, 1e-4 * tot + 1e-9);
                w[c] = nk / samples;
            }
        }
        double s = 0;
        double[] l = new double[k];
        for (int i = 0; i < samples; i++) {
            double[] t = problem.samplePrior(random);
            double yy = problem.simulate(t, design, random);
            for (int c = 0; c < k; c++) {
                l[c] = Math.log(Math.max(w[c], 1e-300)) + logNormal(yy, mu[c], var[c]);
            }
            s += problem.logLikelihood(yy, t, design) - (logMeanExp(l, 0, k) + Math.log(k));
        }
        return s / samples;
    }

    private static double logNormal(double y, double mu, double var) {
        return -0.5 * (y - mu) * (y - mu) / var - 0.5 * Math.log(2 * Math.PI * var);
    }

    /**
     * The MINE (Donsker-Varadhan) neural lower bound from simulations only.
     *
     * @param problem the problem
     * @param design the design
     * @param samples simulations
     * @param steps Adam steps
     * @param seed the seed
     * @return the estimate in nats
     */
    public static double mine(DesignProblem problem, double[] design, int samples, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] th = new double[samples][];
        double[][] y = new double[samples][1];
        for (int i = 0; i < samples; i++) {
            th[i] = problem.samplePrior(r);
            y[i][0] = problem.simulate(th[i], design, r);
        }
        return MutualInformation.variational(MutualInformation.Bound.MINE, th, y, 32, 16, 128, steps, 2e-3, 5,
                seed);
    }

    private static double[] historyLogLik(DesignProblem problem, double[][] thetas, double[][] designs, double[] ys) {
        double[] l = new double[thetas.length];
        for (int m = 0; m < thetas.length; m++) {
            for (int t = 0; t < ys.length; t++) {
                l[m] += problem.logLikelihood(ys[t], thetas[m], designs[t]);
            }
        }
        return l;
    }

    /**
     * The sequential PCE lower bound for one realised history, with {@code theta_0} the
     * parameter that generated it.
     *
     * @param problem the problem
     * @param theta0 the true parameter
     * @param designs the designs used
     * @param ys the outcomes
     * @param contrasts fresh prior draws {@code L}
     * @param random the random source
     * @return {@code log p(h | theta_0) - log mean_{l=0..L} p(h | theta_l)}
     */
    public static double sequentialPce(DesignProblem problem, double[] theta0, double[][] designs, double[] ys,
            int contrasts, SplittableRandom random) {
        double[][] th = new double[contrasts + 1][];
        th[0] = theta0;
        for (int m = 1; m <= contrasts; m++) {
            th[m] = problem.samplePrior(random);
        }
        double[] l = historyLogLik(problem, th, designs, ys);
        return l[0] - logMeanExp(l, 0, contrasts + 1);
    }

    /**
     * The sequential NCE upper bound for one realised history.
     *
     * @param problem the problem
     * @param theta0 the true parameter
     * @param designs the designs used
     * @param ys the outcomes
     * @param contrasts fresh prior draws {@code L}
     * @param random the random source
     * @return {@code log p(h | theta_0) - log mean_{l=1..L} p(h | theta_l)}
     */
    public static double sequentialNce(DesignProblem problem, double[] theta0, double[][] designs, double[] ys,
            int contrasts, SplittableRandom random) {
        double[][] th = new double[contrasts + 1][];
        th[0] = theta0;
        for (int m = 1; m <= contrasts; m++) {
            th[m] = problem.samplePrior(random);
        }
        double[] l = historyLogLik(problem, th, designs, ys);
        return l[0] - logMeanExp(l, 1, contrasts + 1);
    }
}
