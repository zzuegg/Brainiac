package io.github.zzuegg.brainiac.experiments.science;

import io.github.zzuegg.brainiac.experimental.data.ScienceSims;
import io.github.zzuegg.brainiac.experimental.science.SirSurrogate;
import io.github.zzuegg.brainiac.experimental.timeseries.Arima;
import io.github.zzuegg.brainiac.experimental.timeseries.ExponentialSmoothing;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Epidemic inference and forecasting on simulated outbreaks with known truth (wave 13, items
 * 13.3.7 and 13.3.8).
 *
 * <ul>
 *   <li>{@link #reproductionNumber}: {@code R_t} RMSE and 90% interval coverage of EpiEstim,
 *       the random-walk Kalman smoother and the particle smoother on renewal-equation outbreaks
 *       with a step and a smooth change in {@code R_t}, and the nowcast error of the last week
 *       with and without the chain-ladder correction of reporting delays.</li>
 *   <li>{@link #forecasts}: 7- and 14-day MAE of a fitted SEIR model against ARIMA and
 *       exponential smoothing on a stochastic (Gillespie) SEIR outbreak, at forecast origins
 *       before, at and after the peak, plus the fitted {@code R0} with its profile interval.</li>
 *   <li>{@link #surrogate}: rolling five-week-window forecasts of weekly counts four weeks
 *       ahead with the physics-informed SIR surrogate, a data-only MLP surrogate and the exact
 *       ODE (same likelihood and sampler), against ARIMA; MAE, 50%/90% coverage and seconds
 *       per window, under SEIR truth (model mismatch); and surrogate extrapolation error
 *       outside the training box.</li>
 * </ul>
 */
public final class Epidemics {

    private Epidemics() {
    }

    /**
     * {@code R_t} and nowcast table.
     *
     * @param days outbreak length
     * @param seed the seed
     * @return metric to values
     */
    public static Map<String, double[]> reproductionNumber(int days, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        double[][] paths = {ScienceSims.stepRt(days, days * 2 / 5, 1.8, 0.8), ScienceSims.smoothRt(days, 1.8, 0.7)};
        String[] names = {"step", "smooth"};
        for (int k = 0; k < 2; k++) {
            ScienceSims.Outbreak o = ScienceSims.renewalOutbreak(paths[k], 20, 0.6, 4, 14, seed + k);
            for (io.github.zzuegg.brainiac.experimental.science.Epidemics.RtMethod m
                    : io.github.zzuegg.brainiac.experimental.science.Epidemics.RtMethod.values()) {
                io.github.zzuegg.brainiac.experimental.science.Epidemics.RtEstimate e =
                        io.github.zzuegg.brainiac.experimental.science.Epidemics.rt(m, o.onsets(), o.serialInterval(),
                                m == io.github.zzuegg.brainiac.experimental.science.Epidemics.RtMethod.EPIESTIM ? 7 : 5,
                                seed);
                double se = 0;
                int cov = 0;
                int n = 0;
                for (int t = 20; t < days; t++) {
                    if (Double.isNaN(e.mean()[t])) {
                        continue;
                    }
                    se += (e.mean()[t] - o.rt()[t]) * (e.mean()[t] - o.rt()[t]);
                    cov += o.rt()[t] >= e.lower()[t] && o.rt()[t] <= e.upper()[t] ? 1 : 0;
                    n++;
                }
                out.put(names[k] + " " + m + " {RMSE, 90% coverage}", new double[] {Math.sqrt(se / n), cov / (double) n});
            }
            // nowcast of the last 7 days
            int now = days - 1;
            int[] known = o.reportedBy(now);
            double[] nc = io.github.zzuegg.brainiac.experimental.science.Epidemics.nowcast(o.triangle(), now, 14);
            double raw = 0;
            double fixed = 0;
            for (int t = now - 6; t <= now; t++) {
                raw += Math.abs(known[t] - o.onsets()[t]) / Math.max(1.0, o.onsets()[t]);
                fixed += Math.abs(nc[t] - o.onsets()[t]) / Math.max(1.0, o.onsets()[t]);
            }
            out.put(names[k] + " last-7-day relative error {reported so far, nowcast}", new double[] {raw / 7, fixed / 7});
        }
        return out;
    }

    /**
     * Forecast table on a Gillespie SEIR outbreak.
     *
     * @param population the population
     * @param seed the seed
     * @return metric to values
     */
    public static Map<String, double[]> forecasts(int population, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        int days = 160;
        double[] r0 = new double[days];
        Arrays.fill(r0, 2.0);
        ScienceSims.SeirPath path = ScienceSims.seirGillespie(population, r0, 1 / 3.0, 1 / 4.0, 5, seed);
        int[] cases = new int[days];
        java.util.SplittableRandom r = new java.util.SplittableRandom(seed);
        int peak = 0;
        for (int t = 0; t < days; t++) {
            cases[t] = ScienceSims.binomial(path.newCases()[t], 0.5, r);
            if (path.newCases()[t] > path.newCases()[peak]) {
                peak = t;
            }
        }
        int[] origins = {peak - 10, peak, peak + 20};
        String[] labels = {"pre-peak", "peak", "tail"};
        for (int k = 0; k < origins.length; k++) {
            int t0 = origins[k];
            int[] hist = Arrays.copyOf(cases, t0);
            io.github.zzuegg.brainiac.experimental.science.Epidemics.Fit fit =
                    io.github.zzuegg.brainiac.experimental.science.Epidemics.fitSeir(hist, population, 1 / 3.0, 1 / 4.0, 0, k == 1);
            double[] mech = io.github.zzuegg.brainiac.experimental.science.Epidemics.seirIncidence(population,
                    fit.beta(), 1 / 3.0, fit.gamma(), fit.initial(), t0 + 14);
            double[] y = new double[t0];
            for (int t = 0; t < t0; t++) {
                y[t] = Math.log1p(hist[t]);
            }
            double[] ar = Arima.select(y, 2, 1, 0, 0, 200).forecast(y, 14).mean();
            double[] ets = ExponentialSmoothing.select(y, 1, 200).forecast(y, 14).mean();
            double[] mae = new double[6];
            for (int h = 0; h < 14; h++) {
                double truth = cases[t0 + h];
                double em = Math.abs(mech[t0 + h] * fit.ascertainment() - truth);
                double ea = Math.abs(Math.expm1(ar[h]) - truth);
                double ee = Math.abs(Math.expm1(ets[h]) - truth);
                if (h < 7) {
                    mae[0] += em / 7;
                    mae[2] += ea / 7;
                    mae[4] += ee / 7;
                }
                mae[1] += em / 14;
                mae[3] += ea / 14;
                mae[5] += ee / 14;
            }
            out.put(labels[k] + " MAE {SEIR 7d, SEIR 14d, ARIMA 7d, ARIMA 14d, ETS 7d, ETS 14d}", mae);
            if (k == 1) {
                out.put("peak fit R0 {estimate, profile lo, profile hi, truth}", new double[] {fit.r0(),
                    fit.r0Interval()[0], fit.r0Interval()[1], 2.0});
            }
        }
        return out;
    }

    /**
     * Surrogate forecasting table.
     *
     * @param trainSteps surrogate training steps
     * @param windowEnds the weeks at which forecasts are made
     * @param samples MCMC samples per window
     * @param seed the seed
     * @return metric to values
     */
    public static Map<String, double[]> surrogate(int trainSteps, int[] windowEnds, int samples, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        // SEIR truth on a weekly scale: infectious period 1 week, latent 2 days, R0 1.8
        int weeks = 30;
        double[] r0 = new double[weeks * 7];
        Arrays.fill(r0, 1.8);
        ScienceSims.SeirPath path = ScienceSims.seirGillespie(200_000, r0, 1 / 2.0, 1 / 7.0, 40, seed);
        int[] daily = path.newCases();
        int[] shifted = new int[daily.length];
        System.arraycopy(daily, 0, shifted, 21, daily.length - 21);   // the season starts in week 3
        int[] weekly = ScienceSims.weeklyReports(shifted, 0.3, 50, seed);
        SirSurrogate pinn = new SirSurrogate(32, 2, 1.0, seed);
        long t0 = System.nanoTime();
        pinn.train(100, trainSteps, 128, seed);
        double trainSeconds = (System.nanoTime() - t0) / 1e9;
        SirSurrogate mlp = new SirSurrogate(32, 2, 0.0, seed);
        mlp.train(100, trainSteps, 128, seed);
        SirSurrogate.Curve[] curves = {pinn::curve, mlp::curve, SirSurrogate::exactCurve};
        String[] names = {"PINN surrogate", "MLP surrogate (no physics)", "exact ODE"};
        int horizon = 4;
        for (int m = 0; m < curves.length; m++) {
            double mae4 = 0;
            double mae = 0;
            double c50 = 0;
            double c90 = 0;
            double secs = 0;
            for (int end : windowEnds) {
                SirSurrogate.Forecast f = SirSurrogate.forecast(curves[m], weekly, end, 5, horizon, samples, 20, seed + end);
                secs += f.seconds() / windowEnds.length;
                for (int h = 0; h < horizon; h++) {
                    double truth = weekly[end + h];
                    mae += Math.abs(f.mean()[h] - truth) / (horizon * windowEnds.length);
                    c50 += truth >= f.lower50()[h] && truth <= f.upper50()[h] ? 1.0 / (horizon * windowEnds.length) : 0;
                    c90 += truth >= f.lower90()[h] && truth <= f.upper90()[h] ? 1.0 / (horizon * windowEnds.length) : 0;
                }
                mae4 += Math.abs(f.mean()[horizon - 1] - weekly[end + horizon - 1]) / windowEnds.length;
            }
            out.put(names[m] + " {4-week MAE, mean MAE 1-4, cover50, cover90, s/window}",
                    new double[] {mae4, mae, c50, c90, secs});
        }
        double mae4 = 0;
        for (int end : windowEnds) {
            double[] y = new double[end];
            for (int k = 0; k < end; k++) {
                y[k] = Math.log1p(weekly[k]);
            }
            double[] fc = Arima.select(y, 2, 1, 0, 0, 200).forecast(y, horizon).mean();
            mae4 += Math.abs(Math.expm1(fc[horizon - 1]) - weekly[end + horizon - 1]) / windowEnds.length;
        }
        out.put("ARIMA 4-week MAE", new double[] {mae4});
        // curve error inside and outside the training box
        double[] u = new double[30];
        for (int k = 0; k < 30; k++) {
            u[k] = k;
        }
        double[][] params = {{2.0, 1.0}, {3.6, 0.5}};
        for (int p = 0; p < 2; p++) {
            double[] ex = SirSurrogate.exactCurve(params[p][0], params[p][1], u);
            double[] a = pinn.curve(params[p][0], params[p][1], u);
            double[] b = mlp.curve(params[p][0], params[p][1], u);
            double ea = 0;
            double eb = 0;
            for (int k = 0; k < 30; k++) {
                ea += Math.abs(a[k] - ex[k]) / 30;
                eb += Math.abs(b[k] - ex[k]) / 30;
            }
            out.put((p == 0 ? "inside" : "outside") + " box s(u) MAE {PINN, MLP}", new double[] {ea, eb});
        }
        out.put("surrogate training seconds", new double[] {trainSeconds});
        return out;
    }
}
