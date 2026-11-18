package io.github.zzuegg.brainiac.experimental.science;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ScienceSims;
import io.github.zzuegg.brainiac.tensor.Matrices;
import io.github.zzuegg.brainiac.tensor.Optimize;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Lbfgs;

import java.util.List;
import java.util.function.ToDoubleFunction;

/**
 * Single-season site-occupancy models (MacKenzie et al., "Estimating site occupancy rates when
 * detection probabilities are less than one", Ecology 2002): repeated visits separate the
 * probability that a site is occupied, {@code logit psi = b0 + b1 x}, from the probability of
 * detecting the species when present, {@code logit p = a0 + a1 w}. A site's likelihood is the
 * zero-inflated binomial {@code psi prod p^y (1 - p)^(1 - y) + (1 - psi) [no detections]}; it is
 * written with autograd {@link Var}s and maximised by core {@link Lbfgs}, and Wald intervals
 * come from the numerical Hessian.
 *
 * <p>{@link #royleNichols} is the abundance-induced heterogeneity model (Royle and Nichols,
 * Ecology 2003): {@code N ~ Poisson(lambda)}, {@code p = 1 - (1 - r)^N}, {@code psi = 1 - e^-lambda}.
 */
public final class OccupancyModel {

    /**
     * A fit.
     *
     * @param coefficients {b0, b1, a0, a1}
     * @param standardErrors their standard errors
     * @param meanPsi the mean fitted occupancy over the sites
     * @param nll the negative log-likelihood
     */
    public record Fit(double[] coefficients, double[] standardErrors, double meanPsi, double nll) {

        /**
         * @param k a coefficient index
         * @param truth its true value
         * @return whether the 95% Wald interval covers the truth
         */
        public boolean covers(int k, double truth) {
            return Math.abs(coefficients[k] - truth) <= 1.96 * standardErrors[k];
        }
    }

    private OccupancyModel() {
    }

    /**
     * The negative log-likelihood of the standard model.
     *
     * @param s the survey
     * @param c {b0, b1, a0, a1}
     * @return the NLL
     */
    public static double nll(ScienceSims.OccupancySurvey s, double[] c) {
        double total = 0;
        for (int i = 0; i < s.x().length; i++) {
            double eta = c[0] + c[1] * s.x()[i];
            double a = -softplus(-eta);
            boolean any = false;
            for (int j = 0; j < s.y()[i].length; j++) {
                double z = c[2] + c[3] * s.w()[i][j];
                a += s.y()[i][j] == 1 ? -softplus(-z) : -softplus(z);
                any |= s.y()[i][j] == 1;
            }
            double l = any ? a : logAddExp(a, -softplus(eta));
            total -= l;
        }
        return total;
    }

    private static double softplus(double v) {
        return v > 30 ? v : Math.log1p(Math.exp(v));
    }

    private static double logAddExp(double a, double b) {
        double m = Math.max(a, b);
        return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
    }

    /**
     * Fits the standard occupancy model by core L-BFGS on an autograd likelihood.
     *
     * @param s the survey
     * @return the fit
     */
    public static Fit fit(ScienceSims.OccupancySurvey s) {
        int sites = s.x().length;
        int visits = s.y()[0].length;
        float[] x = new float[sites];
        float[] w = new float[sites * visits];
        float[] y = new float[sites * visits];
        float[] any = new float[sites];
        for (int i = 0; i < sites; i++) {
            x[i] = (float) s.x()[i];
            for (int j = 0; j < visits; j++) {
                w[i * visits + j] = (float) s.w()[i][j];
                y[i * visits + j] = s.y()[i][j];
                if (s.y()[i][j] == 1) {
                    any[i] = 1;
                }
            }
        }
        Tensor xt = Tensor.of(Shape.of(sites, 1), x);
        Tensor wt = Tensor.of(Shape.of(sites, visits), w);
        Tensor yt = Tensor.of(Shape.of(sites, visits), y);
        Tensor anyT = Tensor.of(Shape.of(sites, 1), any);
        Var b0 = Var.param(Tensor.zeros(Shape.of(1)));
        Var b1 = Var.param(Tensor.zeros(Shape.of(1)));
        Var a0 = Var.param(Tensor.zeros(Shape.of(1)));
        Var a1 = Var.param(Tensor.zeros(Shape.of(1)));
        List<Var> params = List.of(b0, b1, a0, a1);
        new Lbfgs(10, 1e-8).minimize(params, () -> {
            Var eta = Var.of(xt).mul(b1).add(b0);
            Var z = Var.of(wt).mul(a1).add(a0);
            Var yv = Var.of(yt);
            // y log p + (1 - y) log(1 - p) = -softplus(-z) y - softplus(z) (1 - y)
            Var det = z.neg().softplus().mul(yv).add(z.softplus().mul(yv.neg().add(1f))).neg().sum(1, true);
            Var occ = eta.neg().softplus().neg().add(det);
            Var empty = eta.softplus().neg();
            Var ll = Var.where(anyT, occ, occ.logaddexp(empty));
            return ll.sum().neg();
        }, 200);
        double[] c = {b0.value().item(), b1.value().item(), a0.value().item(), a1.value().item()};
        // polish in double precision
        ToDoubleFunction<double[]> f = v -> nll(s, v);
        c = Optimize.nelderMead(f, c, 0.01, 400, 1e-12).x();
        return finish(f, c, s);
    }

    private static Fit finish(ToDoubleFunction<double[]> f, double[] c, ScienceSims.OccupancySurvey s) {
        double[][] cov = Matrices.inverse(CaptureRecapture.hessian(f, c, 1e-4));
        double[] se = new double[c.length];
        for (int k = 0; k < c.length; k++) {
            se[k] = Math.sqrt(Math.max(cov[k][k], 0));
        }
        double mean = 0;
        for (double v : s.x()) {
            mean += CaptureRecapture.sig(c[0] + c[1] * v) / s.x().length;
        }
        return new Fit(c, se, mean, f.applyAsDouble(c));
    }

    /**
     * The Royle–Nichols NLL, summing abundance up to {@code maxN}.
     *
     * @param s the survey
     * @param c {log-lambda intercept, slope, logit-r intercept, slope}
     * @param maxN the largest abundance summed
     * @return the NLL
     */
    public static double royleNicholsNll(ScienceSims.OccupancySurvey s, double[] c, int maxN) {
        double total = 0;
        for (int i = 0; i < s.x().length; i++) {
            double lambda = Math.exp(c[0] + c[1] * s.x()[i]);
            double sum = 0;
            double pois = Math.exp(-lambda);
            for (int n = 0; n <= maxN; n++) {
                if (n > 0) {
                    pois *= lambda / n;
                }
                double l = pois;
                for (int j = 0; j < s.y()[i].length; j++) {
                    double r = CaptureRecapture.sig(c[2] + c[3] * s.w()[i][j]);
                    double p = 1 - Math.pow(1 - r, n);
                    l *= s.y()[i][j] == 1 ? p : 1 - p;
                }
                sum += l;
            }
            total -= Math.log(Math.max(sum, 1e-300));
        }
        return total;
    }

    /**
     * Fits the Royle–Nichols model by Nelder–Mead; the returned {@code meanPsi} is
     * {@code mean(1 - exp(-lambda))}.
     *
     * @param s the survey
     * @param maxN the abundance truncation
     * @return the fit (coefficients on the log-lambda and logit-r scales)
     */
    public static Fit royleNichols(ScienceSims.OccupancySurvey s, int maxN) {
        ToDoubleFunction<double[]> f = v -> royleNicholsNll(s, v, maxN);
        double[] c = Optimize.nelderMead(f, new double[] {0, 0.5, -0.5, 0.5}, 0.3, 3000, 1e-10).x();
        double[][] cov = Matrices.inverse(CaptureRecapture.hessian(f, c, 1e-4));
        double[] se = new double[4];
        for (int k = 0; k < 4; k++) {
            se[k] = Math.sqrt(Math.max(cov[k][k], 0));
        }
        double mean = 0;
        for (double v : s.x()) {
            mean += (1 - Math.exp(-Math.exp(c[0] + c[1] * v))) / s.x().length;
        }
        return new Fit(c, se, mean, f.applyAsDouble(c));
    }
}
