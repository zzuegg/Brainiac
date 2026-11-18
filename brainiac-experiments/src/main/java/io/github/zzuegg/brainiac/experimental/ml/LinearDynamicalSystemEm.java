package io.github.zzuegg.brainiac.experimental.ml;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Maximum-likelihood linear dynamical systems by EM with missing observations (Shumway and
 * Stoffer 1982, <a href="https://doi.org/10.1111/j.1467-9892.1982.tb00349.x">doi</a>; Ghahramani
 * and Hinton 1996, <a href="https://mlg.eng.cam.ac.uk/zoubin/papers/tr-96-2.pdf">tech. report
 * CRG-TR-96-2</a>): {@code x_t = F x_t-1 + w}, {@code y_t = H x_t + offset + v}, {@code w ~ N(0, Q)},
 * {@code v ~ N(0, R)} with diagonal {@code R}.
 *
 * <p>The E-step runs {@link StateSpaceFilter#kalman} with {@code NaN}-masked observations, then
 * the RTS smoother, and the lag-one cross-covariances {@code V_{t,t-1|T} = V_{t|T} J_{t-1}^T}
 * with the smoother gain {@code J_{t-1} = P_{t-1|t-1} F^T P_{t|t-1}^-1}. The M-step is in closed
 * form: {@code F = (sum E[x_t x_t-1^T]) (sum E[x_t-1 x_t-1^T])^-1},
 * {@code Q = mean(E[x_t x_t^T] - F E[x_t-1 x_t^T])}, each row of {@code H} regressed on the
 * smoothed moments of the steps where that channel is observed, and {@code R_ii} from the same
 * residual moments; the prior is the smoothed first state. Several sequences share one set of
 * parameters. The observed-data log-likelihood is recorded at every iteration and never
 * decreases (up to the variance floors).
 */
public final class LinearDynamicalSystemEm {

    private LinearDynamicalSystemEm() {
    }

    /**
     * A fitted system.
     *
     * @param model the parameters (in the centred coordinates)
     * @param offset the per-channel mean added to {@code H x}
     * @param logLikelihoods the observed-data log-likelihood before each M-step
     */
    public record Fit(StateSpaceFilter.LinearModel model, double[] offset, double[] logLikelihoods) {

        /**
         * Smoothed reconstructions of one sequence.
         *
         * @param ys observations {@code [T][m]}, {@code NaN} where missing
         * @return {@code {means [T][m], variances [T][m]}} of {@code y_t} given all observations
         */
        public double[][][] reconstruct(double[][] ys) {
            StateSpaceFilter.Estimate e = StateSpaceFilter.kalman(model, centre(ys, offset), true);
            int steps = ys.length;
            int m = offset.length;
            double[][] mu = new double[steps][m];
            double[][] var = new double[steps][m];
            double[][] h = model.h();
            for (int t = 0; t < steps; t++) {
                double[][] hp = MlMatrices.multiply(h, e.covariances()[t]);
                for (int i = 0; i < m; i++) {
                    mu[t][i] = MlMatrices.dot(h[i], e.means()[t]) + offset[i];
                    var[t][i] = MlMatrices.dot(hp[i], h[i]) + model.r()[i][i];
                }
            }
            return new double[][][] {mu, var};
        }

        /**
         * One-step-ahead predictions from the filter: {@code E[y_t | y_<t]}.
         *
         * @param ys observations, {@code NaN} where missing
         * @return {@code [T][m]}; row 0 is the prior prediction
         */
        public double[][] oneStepAhead(double[][] ys) {
            StateSpaceFilter.Estimate e = StateSpaceFilter.kalman(model, centre(ys, offset), false);
            int steps = ys.length;
            int m = offset.length;
            double[][] out = new double[steps][m];
            for (int t = 0; t < steps; t++) {
                double[] x = t == 0 ? model.x0() : mv(model.f(), e.means()[t - 1]);
                for (int i = 0; i < m; i++) {
                    out[t][i] = MlMatrices.dot(model.h()[i], x) + offset[i];
                }
            }
            return out;
        }
    }

    private static double[] mv(double[][] a, double[] x) {
        double[] y = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            y[i] = MlMatrices.dot(a[i], x);
        }
        return y;
    }

    static double[][] centre(double[][] ys, double[] offset) {
        double[][] out = new double[ys.length][offset.length];
        for (int t = 0; t < ys.length; t++) {
            for (int i = 0; i < offset.length; i++) {
                out[t][i] = ys[t][i] - offset[i];
            }
        }
        return out;
    }

    private static double[][] eye(int n, double v) {
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            a[i][i] = v;
        }
        return a;
    }

    private static double[][] outer(double[] a, double[] b) {
        double[][] o = new double[a.length][b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                o[i][j] = a[i] * b[j];
            }
        }
        return o;
    }

    private static void addTo(double[][] acc, double[][] a, double scale) {
        for (int i = 0; i < acc.length; i++) {
            for (int j = 0; j < acc[0].length; j++) {
                acc[i][j] += scale * a[i][j];
            }
        }
    }

    private static double[][] symmetrise(double[][] a, double floor) {
        int n = a.length;
        double[][] s = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                s[i][j] = 0.5 * (a[i][j] + a[j][i]);
            }
            s[i][i] = Math.max(s[i][i], floor);
        }
        return s;
    }

    /** {@code a b^-1} for symmetric positive definite {@code b}. */
    private static double[][] rightSolve(double[][] a, double[][] b) {
        double[][] l = MlMatrices.cholesky(b);
        return MlMatrices.transpose(MlMatrices.choleskySolve(l, MlMatrices.transpose(a)));
    }

    /**
     * Fits one system to several sequences.
     *
     * @param sequences observations {@code [S][T][m]}, {@code NaN} where missing
     * @param stateDim the latent dimension
     * @param iterations EM iterations
     * @param seed the initialisation seed
     * @return the fit
     */
    public static Fit fit(double[][][] sequences, int stateDim, int iterations, long seed) {
        int m = sequences[0][0].length;
        int d = stateDim;
        double[] offset = new double[m];
        double[] var = new double[m];
        int[] cnt = new int[m];
        for (double[][] seq : sequences) {
            for (double[] y : seq) {
                for (int i = 0; i < m; i++) {
                    if (!Double.isNaN(y[i])) {
                        offset[i] += y[i];
                        var[i] += y[i] * y[i];
                        cnt[i]++;
                    }
                }
            }
        }
        for (int i = 0; i < m; i++) {
            offset[i] = cnt[i] == 0 ? 0 : offset[i] / cnt[i];
            var[i] = cnt[i] < 2 ? 1 : Math.max(1e-6, var[i] / cnt[i] - offset[i] * offset[i]);
        }
        List<double[][]> centred = new ArrayList<>();
        for (double[][] seq : sequences) {
            centred.add(centre(seq, offset));
        }
        SplittableRandom r = new SplittableRandom(seed);
        double[][] f = eye(d, 0);
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                f[i][j] = (i == j ? 0.9 : 0) + 0.05 * r.nextGaussian();
            }
        }
        double[][] h = new double[m][d];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < d; j++) {
                h[i][j] = Math.sqrt(var[i] / d) * r.nextGaussian();
            }
        }
        double[][] q = eye(d, 0.1);
        double[][] rr = new double[m][m];
        for (int i = 0; i < m; i++) {
            rr[i][i] = 0.5 * var[i];
        }
        double[] x0 = new double[d];
        double[][] p0 = eye(d, 1);
        double[] lls = new double[iterations];
        for (int it = 0; it < iterations; it++) {
            StateSpaceFilter.LinearModel model = new StateSpaceFilter.LinearModel(f, q, h, rr, x0, p0);
            double[][] s11 = new double[d][d];     // sum_{t>=2} E[x_t x_t^T]
            double[][] s10 = new double[d][d];     // sum_{t>=2} E[x_t x_t-1^T]
            double[][] s00 = new double[d][d];     // sum_{t>=2} E[x_t-1 x_t-1^T]
            double[][][] hNum = new double[m][1][d];
            double[][][] hDen = new double[m][d][d];
            double[] yy = new double[m];
            int[] obs = new int[m];
            double[] x0Sum = new double[d];
            double[][] p0Sum = new double[d][d];
            int transitions = 0;
            double ll = 0;
            List<double[]> firsts = new ArrayList<>();
            for (double[][] ys : centred) {
                StateSpaceFilter.Estimate filt = StateSpaceFilter.kalman(model, ys, false);
                StateSpaceFilter.Estimate sm = StateSpaceFilter.kalman(model, ys, true);
                ll += filt.logLikelihood();
                int steps = ys.length;
                double[][][] second = new double[steps][][];
                for (int t = 0; t < steps; t++) {
                    second[t] = outer(sm.means()[t], sm.means()[t]);
                    addTo(second[t], sm.covariances()[t], 1);
                    for (int i = 0; i < m; i++) {
                        if (!Double.isNaN(ys[t][i])) {
                            addTo(hNum[i], new double[][] {sm.means()[t]}, ys[t][i]);
                            addTo(hDen[i], second[t], 1);
                            yy[i] += ys[t][i] * ys[t][i];
                            obs[i]++;
                        }
                    }
                }
                for (int t = 1; t < steps; t++) {
                    double[][] pred = MlMatrices.multiply(MlMatrices.multiply(f, filt.covariances()[t - 1]),
                            MlMatrices.transpose(f));
                    addTo(pred, q, 1);
                    // J_{t-1} = P_{t-1|t-1} F^T P_{t|t-1}^-1
                    double[][] j = rightSolve(MlMatrices.multiply(filt.covariances()[t - 1], MlMatrices.transpose(f)),
                            symmetrise(pred, 1e-12));
                    double[][] cross = MlMatrices.multiply(sm.covariances()[t], MlMatrices.transpose(j));
                    addTo(cross, outer(sm.means()[t], sm.means()[t - 1]), 1);
                    addTo(s10, cross, 1);
                    addTo(s11, second[t], 1);
                    addTo(s00, second[t - 1], 1);
                    transitions++;
                }
                firsts.add(sm.means()[0]);
                for (int a = 0; a < d; a++) {
                    x0Sum[a] += sm.means()[0][a] / centred.size();
                }
                addTo(p0Sum, sm.covariances()[0], 1.0 / centred.size());
            }
            lls[it] = ll;
            double[][] s00r = symmetrise(s00, 1e-9);
            f = rightSolve(s10, s00r);
            double[][] qn = new double[d][d];
            addTo(qn, s11, 1.0 / transitions);
            addTo(qn, MlMatrices.multiply(f, MlMatrices.transpose(s10)), -1.0 / transitions);
            q = symmetrise(qn, 1e-6);
            double[][] hn = new double[m][];
            double[][] rn = new double[m][m];
            for (int i = 0; i < m; i++) {
                if (obs[i] == 0) {
                    hn[i] = h[i];
                    rn[i][i] = rr[i][i];
                    continue;
                }
                hn[i] = rightSolve(hNum[i], symmetrise(hDen[i], 1e-9))[0];
                double res = yy[i] - MlMatrices.dot(hn[i], hNum[i][0]);
                rn[i][i] = Math.max(1e-6, res / obs[i]);
            }
            h = hn;
            rr = rn;
            x0 = x0Sum;
            double[][] pz = new double[d][d];
            addTo(pz, p0Sum, 1);
            for (double[] first : firsts) {
                double[] diff = new double[d];
                for (int a = 0; a < d; a++) {
                    diff[a] = first[a] - x0[a];
                }
                addTo(pz, outer(diff, diff), 1.0 / centred.size());
            }
            p0 = symmetrise(pz, 1e-6);
        }
        return new Fit(new StateSpaceFilter.LinearModel(f, q, h, rr, x0, p0), offset, lls);
    }

    /**
     * The eigenvalues of a real square matrix by the unshifted QR iteration on its Hessenberg
     * form, as {@code {re, im}} pairs sorted by modulus (for comparing transition matrices up to a
     * change of basis).
     *
     * @param a the matrix
     * @return {@code [n][2]}
     */
    public static double[][] eigenvalues(double[][] a) {
        int n = a.length;
        if (n == 1) {
            return new double[][] {{a[0][0], 0}};
        }
        if (n == 2) {
            double tr = a[0][0] + a[1][1];
            double det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
            double disc = tr * tr / 4 - det;
            double[][] out = disc >= 0
                    ? new double[][] {{tr / 2 + Math.sqrt(disc), 0}, {tr / 2 - Math.sqrt(disc), 0}}
                    : new double[][] {{tr / 2, Math.sqrt(-disc)}, {tr / 2, -Math.sqrt(-disc)}};
            java.util.Arrays.sort(out, (u, v) -> Double.compare(Math.hypot(v[0], v[1]), Math.hypot(u[0], u[1])));
            return out;
        }
        double[][] h = new double[n][];
        for (int i = 0; i < n; i++) {
            h[i] = a[i].clone();
        }
        for (int it = 0; it < 500 * n; it++) {
            // QR by Gram-Schmidt, then RQ
            double[][] qm = new double[n][n];
            double[][] rm = new double[n][n];
            for (int j = 0; j < n; j++) {
                double[] v = new double[n];
                for (int i = 0; i < n; i++) {
                    v[i] = h[i][j];
                }
                for (int k = 0; k < j; k++) {
                    double dot = 0;
                    for (int i = 0; i < n; i++) {
                        dot += qm[i][k] * h[i][j];
                    }
                    rm[k][j] = dot;
                    for (int i = 0; i < n; i++) {
                        v[i] -= dot * qm[i][k];
                    }
                }
                double norm = Math.sqrt(MlMatrices.dot(v, v));
                rm[j][j] = norm;
                for (int i = 0; i < n; i++) {
                    qm[i][j] = norm < 1e-300 ? 0 : v[i] / norm;
                }
            }
            h = MlMatrices.multiply(rm, qm);
        }
        List<double[]> out = new ArrayList<>();
        int i = 0;
        while (i < n) {
            if (i + 1 < n && Math.abs(h[i + 1][i]) > 1e-6 * (Math.abs(h[i][i]) + Math.abs(h[i + 1][i + 1]) + 1e-12)) {
                double[][] block = {{h[i][i], h[i][i + 1]}, {h[i + 1][i], h[i + 1][i + 1]}};
                double[][] ev = eigenvalues(block);
                out.add(ev[0]);
                out.add(ev[1]);
                i += 2;
            } else {
                out.add(new double[] {h[i][i], 0});
                i++;
            }
        }
        double[][] res = out.toArray(new double[0][]);
        java.util.Arrays.sort(res, (u, v) -> Double.compare(Math.hypot(v[0], v[1]), Math.hypot(u[0], u[1])));
        return res;
    }
}
