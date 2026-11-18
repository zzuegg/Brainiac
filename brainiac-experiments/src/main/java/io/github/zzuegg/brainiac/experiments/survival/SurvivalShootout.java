package io.github.zzuegg.brainiac.experiments.survival;

import io.github.zzuegg.brainiac.experimental.causal.Regressor;
import io.github.zzuegg.brainiac.experimental.data.SurvivalData;
import io.github.zzuegg.brainiac.experimental.survival.AftModels;
import io.github.zzuegg.brainiac.experimental.survival.DeepHit;
import io.github.zzuegg.brainiac.experimental.survival.DiscreteHazardNet;
import io.github.zzuegg.brainiac.experimental.survival.SurvivalForest;
import io.github.zzuegg.brainiac.experimental.survival.SurvivalMetrics;
import io.github.zzuegg.brainiac.experimental.survival.SurvivalModels;
import io.github.zzuegg.brainiac.experimental.survival.Weibull;
import io.github.zzuegg.brainiac.train.GradientBoostedTrees;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Survival analysis on {@link SurvivalData} (W9 1.16 and 1.17), every model trained on the same
 * censored training split and scored on a held-out split:
 * <ul>
 *   <li>{@link #proportional}: Kaplan-Meier, linear Cox and DeepSurv against censoring-naive
 *       baselines (an MLP regressing log time with censored rows treated as events or dropped,
 *       logistic regression on "event before the median time", gradient-boosted trees on log time)
 *       on Weibull data with a linear or nonlinear risk, by Harrell's C-index and, for models that
 *       give survival curves, the IPCW integrated Brier score;</li>
 *   <li>{@link #crossing}: discrete-time hazards, DeepHit and a random survival forest against Cox,
 *       DeepSurv and Kaplan-Meier per subgroup when the hazards cross, by time-dependent C-index
 *       and integrated Brier score;</li>
 *   <li>{@link #aft} (wave 11): accelerated failure time models and Deep Survival Machines
 *       against the references above, with median-time calibration;</li>
 *   <li>{@link #competing}: DeepHit with two cause heads against cause-specific Cox models, by the
 *       mean absolute cumulative-incidence error per cause and the per-cause time-dependent
 *       C-index.</li>
 * </ul>
 */
public final class SurvivalShootout {

    private SurvivalShootout() {
    }

    /**
     * Test scores of one model.
     *
     * @param cIndex Harrell's C-index (or the time-dependent one for {@link #crossing})
     * @param brier the integrated Brier score, NaN for models without survival curves
     */
    public record Score(double cIndex, double brier) {
    }

    private record Split(SurvivalData.Survival train, SurvivalData.Survival test, double[] grid) {
    }

    private static Split split(SurvivalData.Survival all, int n) {
        SurvivalData.Survival train = all.rows(0, n);
        SurvivalData.Survival test = all.rows(n, all.size());
        return new Split(train, test, SurvivalMetrics.grid(train.time(), 0.1, 0.85, 20));
    }

    private static double brier(double[][] curves, Split s) {
        return SurvivalMetrics.integratedBrier(curves, s.grid(), s.test().time(), s.test().event(), s.train().time(),
                s.train().event());
    }

    private static Score curveScore(double[][] curves, Split s, boolean timeDependent, double[] risk) {
        double c = timeDependent
                ? SurvivalMetrics.timeDependentC(curves, s.grid(), s.test().time(), s.test().event(), 0, false)
                : SurvivalMetrics.harrellC(risk, s.test().time(), s.test().event(), 0);
        return new Score(c, brier(curves, s));
    }

    private static double[][] coxCurves(SurvivalModels.Cox cox, double[][] x, double[] grid) {
        double[][] out = new double[x.length][grid.length];
        for (int i = 0; i < x.length; i++) {
            for (int g = 0; g < grid.length; g++) {
                out[i][g] = cox.survival(x[i], grid[g]);
            }
        }
        return out;
    }

    /**
     * Proportional hazards with a linear or nonlinear risk.
     *
     * @param risk the risk function
     * @param censoring the censoring share
     * @param n training units (500 more are held out)
     * @param steps Adam steps for the neural models
     * @param seed the seed
     * @return model name to test score
     */
    public static Map<String, Score> proportional(SurvivalData.Risk risk, double censoring, int n, int steps,
            long seed) {
        Split s = split(SurvivalData.weibull(n + 500, 8, risk, censoring, seed), n);
        SurvivalData.Survival tr = s.train();
        SurvivalData.Survival te = s.test();
        Map<String, Score> out = new LinkedHashMap<>();
        SurvivalModels.KaplanMeier km = new SurvivalModels.KaplanMeier(tr.time(), tr.event());
        double[][] kmCurves = new double[te.size()][s.grid().length];
        for (double[] row : kmCurves) {
            for (int g = 0; g < row.length; g++) {
                row[g] = km.survival(s.grid()[g]);
            }
        }
        out.put("Kaplan-Meier", curveScore(kmCurves, s, false, new double[te.size()]));
        SurvivalModels.Cox cox = new SurvivalModels.Cox(tr.x(), tr.time(), tr.event(), 1e-3);
        double[] coxRisk = new double[te.size()];
        for (int i = 0; i < coxRisk.length; i++) {
            coxRisk[i] = cox.logRisk(te.x()[i]);
        }
        out.put("Cox", curveScore(coxCurves(cox, te.x(), s.grid()), s, false, coxRisk));
        SurvivalModels.DeepSurv deep = new SurvivalModels.DeepSurv(tr.x()[0].length, 32, seed)
                .fit(tr.x(), tr.time(), tr.event(), steps, 256, 3e-3f, 1e-3f, seed);
        out.put("DeepSurv", curveScore(deep.survival(te.x(), s.grid()), s, false, deep.logRisk(te.x())));
        out.put("MLP log-time (censored as events)", naiveRegression(tr, te, false, steps, seed));
        out.put("MLP log-time (censored dropped)", naiveRegression(tr, te, true, steps, seed));
        out.put("logistic event-before-median", logistic(tr, te));
        double[] logTime = new double[tr.size()];
        for (int i = 0; i < logTime.length; i++) {
            logTime[i] = Math.log(tr.time()[i]);
        }
        GradientBoostedTrees gbt = GradientBoostedTrees.fit(tr.x(), logTime, GradientBoostedTrees.Objective.REGRESSION,
                GradientBoostedTrees.Settings.defaults().withTrees(80).withDepth(3));
        double[] gbtRisk = gbt.predict(te.x());
        for (int i = 0; i < gbtRisk.length; i++) {
            gbtRisk[i] = -gbtRisk[i];
        }
        out.put("GBT log-time (censored as events)",
                new Score(SurvivalMetrics.harrellC(gbtRisk, te.time(), te.event(), 0), Double.NaN));
        return out;
    }

    private static Score naiveRegression(SurvivalData.Survival tr, SurvivalData.Survival te, boolean drop, int steps,
            long seed) {
        int keep = 0;
        for (int e : tr.event()) {
            keep += !drop || e != 0 ? 1 : 0;
        }
        double[][] x = new double[keep][];
        double[] y = new double[keep];
        int k = 0;
        for (int i = 0; i < tr.size(); i++) {
            if (!drop || tr.event()[i] != 0) {
                x[k] = tr.x()[i];
                y[k++] = Math.log(tr.time()[i]);
            }
        }
        double[] pred = Regressor.mlp(32, steps, 3e-3f, seed).fit(x, y).predict(te.x());
        for (int i = 0; i < pred.length; i++) {
            pred[i] = -pred[i];
        }
        return new Score(SurvivalMetrics.harrellC(pred, te.time(), te.event(), 0), Double.NaN);
    }

    private static Score logistic(SurvivalData.Survival tr, SurvivalData.Survival te) {
        double[] sorted = tr.time().clone();
        java.util.Arrays.sort(sorted);
        double horizon = sorted[sorted.length / 2];
        int d = tr.x()[0].length;
        java.util.List<double[]> xs = new java.util.ArrayList<>();
        java.util.List<Integer> ys = new java.util.ArrayList<>();
        for (int i = 0; i < tr.size(); i++) {
            boolean failed = tr.event()[i] != 0 && tr.time()[i] <= horizon;
            boolean known = failed || tr.time()[i] > horizon;
            if (known) {
                xs.add(tr.x()[i]);
                ys.add(failed ? 1 : 0);
            }
        }
        double[] w = new double[d + 1];
        for (int it = 0; it < 500; it++) {
            double[] g = new double[d + 1];
            for (int i = 0; i < xs.size(); i++) {
                double z = w[0];
                for (int j = 0; j < d; j++) {
                    z += w[j + 1] * xs.get(i)[j];
                }
                double err = 1 / (1 + Math.exp(-z)) - ys.get(i);
                g[0] += err;
                for (int j = 0; j < d; j++) {
                    g[j + 1] += err * xs.get(i)[j];
                }
            }
            for (int j = 0; j <= d; j++) {
                w[j] -= 0.5 * g[j] / xs.size();
            }
        }
        double[] risk = new double[te.size()];
        for (int i = 0; i < risk.length; i++) {
            double z = w[0];
            for (int j = 0; j < d; j++) {
                z += w[j + 1] * te.x()[i][j];
            }
            risk[i] = z;
        }
        return new Score(SurvivalMetrics.harrellC(risk, te.time(), te.event(), 0), Double.NaN);
    }

    /**
     * Crossing hazards.
     *
     * @param censoring the censoring share
     * @param n training units (500 more are held out)
     * @param steps Adam steps for the neural models
     * @param trees survival-forest trees
     * @param seed the seed
     * @return model name to test score (time-dependent C-index, integrated Brier score)
     */
    public static Map<String, Score> crossing(double censoring, int n, int steps, int trees, long seed) {
        Split s = split(SurvivalData.crossingHazards(n + 500, 6, censoring, seed), n);
        SurvivalData.Survival tr = s.train();
        SurvivalData.Survival te = s.test();
        double[] grid = s.grid();
        Map<String, Score> out = new LinkedHashMap<>();
        SurvivalModels.Cox cox = new SurvivalModels.Cox(tr.x(), tr.time(), tr.event(), 1e-3);
        out.put("Cox", curveScore(coxCurves(cox, te.x(), grid), s, true, null));
        SurvivalModels.DeepSurv deep = new SurvivalModels.DeepSurv(tr.x()[0].length, 32, seed)
                .fit(tr.x(), tr.time(), tr.event(), steps, 256, 3e-3f, 1e-3f, seed);
        out.put("DeepSurv", curveScore(deep.survival(te.x(), grid), s, true, null));
        out.put("Kaplan-Meier per subgroup", curveScore(subgroupKm(tr, te, grid), s, true, null));
        double[] cuts = DiscreteHazardNet.quantileCuts(tr.time(), 12);
        DiscreteHazardNet hazard = new DiscreteHazardNet(tr.x()[0].length, 32, cuts, seed)
                .fit(tr.x(), tr.time(), tr.event(), steps, 128, 3e-3f, seed);
        out.put("discrete-time hazard net", curveScore(hazard.survival(te.x(), grid), s, true, null));
        DeepHit hit = new DeepHit(tr.x()[0].length, 32, 1, cuts, seed)
                .fit(tr.x(), tr.time(), tr.event(), steps, 128, 3e-3f, 0.2f, seed);
        out.put("DeepHit", curveScore(hit.survival(te.x(), grid), s, true, null));
        SurvivalForest rsf = SurvivalForest.fit(tr.x(), tr.time(), tr.event(), grid, trees, 10, 10, seed);
        out.put("random survival forest", curveScore(rsf.survival(te.x()), s, true, null));
        return out;
    }

    private static double[][] subgroupKm(SurvivalData.Survival tr, SurvivalData.Survival te, double[] grid) {
        SurvivalModels.KaplanMeier[] km = new SurvivalModels.KaplanMeier[2];
        for (int g = 0; g < 2; g++) {
            int c = 0;
            for (double[] row : tr.x()) {
                c += (row[0] > 0 ? 1 : 0) == g ? 1 : 0;
            }
            double[] t = new double[c];
            int[] e = new int[c];
            int k = 0;
            for (int i = 0; i < tr.size(); i++) {
                if ((tr.x()[i][0] > 0 ? 1 : 0) == g) {
                    t[k] = tr.time()[i];
                    e[k++] = tr.event()[i];
                }
            }
            km[g] = new SurvivalModels.KaplanMeier(t, e);
        }
        double[][] out = new double[te.size()][grid.length];
        for (int i = 0; i < te.size(); i++) {
            int g = te.x()[i][0] > 0 ? 1 : 0;
            for (int k = 0; k < grid.length; k++) {
                out[i][k] = km[g].survival(grid[k]);
            }
        }
        return out;
    }

    /**
     * Competing risks.
     *
     * @param n training units (500 more are held out)
     * @param steps Adam steps for DeepHit
     * @param seed the seed
     * @return model name to {@code {CIF error cause 1, CIF error cause 2, C-index cause 1, C-index cause 2}}
     */
    public static Map<String, double[]> competing(int n, int steps, long seed) {
        SurvivalData.Survival all = SurvivalData.competingRisks(n + 500, 6, 0.3, seed);
        SurvivalData.Survival tr = all.rows(0, n);
        SurvivalData.Survival te = all.rows(n, all.size());
        double[] grid = SurvivalMetrics.grid(tr.time(), 0.1, 0.85, 20);
        double[][][] truth = new double[2][te.size()][grid.length];
        for (int c = 0; c < 2; c++) {
            for (int i = 0; i < te.size(); i++) {
                for (int g = 0; g < grid.length; g++) {
                    truth[c][i][g] = te.incidence(i, c + 1, grid[g]);
                }
            }
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        double[] cuts = DiscreteHazardNet.quantileCuts(tr.time(), 12);
        DeepHit hit = new DeepHit(tr.x()[0].length, 32, 2, cuts, seed)
                .fit(tr.x(), tr.time(), tr.event(), steps, 128, 3e-3f, 0.2f, seed);
        out.put("DeepHit (2 cause heads)", incidenceScores(hit.incidence(te.x(), grid), truth, te, grid));
        out.put("cause-specific Cox", incidenceScores(causeSpecificCox(tr, te.x(), grid), truth, te, grid));
        return out;
    }

    private static double[] incidenceScores(double[][][] predicted, double[][][] truth, SurvivalData.Survival te,
            double[] grid) {
        double[] out = new double[4];
        for (int c = 0; c < 2; c++) {
            double err = 0;
            for (int i = 0; i < te.size(); i++) {
                for (int g = 0; g < grid.length; g++) {
                    err += Math.abs(predicted[c][i][g] - truth[c][i][g]) / (te.size() * grid.length);
                }
            }
            out[c] = err;
            out[2 + c] = SurvivalMetrics.timeDependentC(predicted[c], grid, te.time(), te.event(), c + 1, true);
        }
        return out;
    }

    /**
     * Cumulative incidence from one Cox model per cause (other causes treated as censoring):
     * {@code F_c(t) = sum over grid steps of S(s) dH_c(s)} with {@code S = exp(-H_1 - H_2)}.
     */
    private static double[][][] causeSpecificCox(SurvivalData.Survival tr, double[][] x, double[] grid) {
        SurvivalModels.Cox[] models = new SurvivalModels.Cox[2];
        for (int c = 0; c < 2; c++) {
            int[] e = new int[tr.size()];
            for (int i = 0; i < e.length; i++) {
                e[i] = tr.event()[i] == c + 1 ? 1 : 0;
            }
            models[c] = new SurvivalModels.Cox(tr.x(), tr.time(), e, 1e-3);
        }
        int fine = 200;
        double[] t = new double[fine + 1];
        for (int k = 0; k <= fine; k++) {
            t[k] = grid[grid.length - 1] * k / fine;
        }
        double[][][] out = new double[2][x.length][grid.length];
        for (int i = 0; i < x.length; i++) {
            double[][] h = new double[2][fine + 1];
            for (int c = 0; c < 2; c++) {
                for (int k = 0; k <= fine; k++) {
                    h[c][k] = -Math.log(Math.max(models[c].survival(x[i], t[k]), 1e-300));
                }
            }
            double[][] f = new double[2][fine + 1];
            for (int k = 1; k <= fine; k++) {
                double surv = Math.exp(-h[0][k - 1] - h[1][k - 1]);
                for (int c = 0; c < 2; c++) {
                    f[c][k] = f[c][k - 1] + surv * (h[c][k] - h[c][k - 1]);
                }
            }
            for (int c = 0; c < 2; c++) {
                for (int g = 0; g < grid.length; g++) {
                    int k = (int) Math.round(grid[g] / grid[grid.length - 1] * fine);
                    out[c][i][g] = f[c][Math.min(fine, k)];
                }
            }
        }
        return out;
    }

    // ------------------------------------------------------------ wave 11 (11.3.17): AFT models

    private static double[] medians(double[][] curves, double[] grid) {
        double[] out = new double[curves.length];
        for (int i = 0; i < curves.length; i++) {
            out[i] = grid[grid.length - 1];
            for (int g = 0; g < grid.length; g++) {
                if (curves[i][g] <= 0.5) {
                    if (g == 0) {
                        out[i] = grid[0];
                    } else {
                        double a = curves[i][g - 1];
                        double b = curves[i][g];
                        double w = a == b ? 0 : (a - 0.5) / (a - b);
                        out[i] = grid[g - 1] + w * (grid[g] - grid[g - 1]);
                    }
                    break;
                }
            }
        }
        return out;
    }

    private static double[] aftScore(double[][] curves, double[] median, Split s, boolean timeDependent,
            double[] risk) {
        SurvivalData.Survival te = s.test();
        double c = timeDependent
                ? SurvivalMetrics.timeDependentC(curves, s.grid(), te.time(), te.event(), 0, false)
                : SurvivalMetrics.harrellC(risk, te.time(), te.event(), 0);
        double err = Double.NaN;
        if (median != null) {
            err = 0;
            for (int i = 0; i < median.length; i++) {
                double truth = new Weibull(te.scale()[i][0], te.shape()[i][0]).median();
                err += Math.abs(Math.log(median[i]) - Math.log(truth)) / median.length;
            }
        }
        return new double[] {c, brier(curves, s), err};
    }

    private static double[] negLog(double[] median) {
        double[] r = new double[median.length];
        for (int i = 0; i < r.length; i++) {
            r[i] = -Math.log(median[i]);
        }
        return r;
    }

    /**
     * Accelerated failure time models (wave 11, item 11.3.17) against the wave 9 references on
     * the same split: the Weibull, log-normal and log-logistic AFT models and Deep Survival
     * Machines ({@link AftModels}) against Kaplan-Meier, Cox and DeepSurv, plus DeepHit and the
     * random survival forest on crossing hazards. Scores: Harrell's C-index (the time-dependent
     * one on crossing hazards), the IPCW integrated Brier score and the median-time calibration
     * error {@code mean |log median_hat - log median_true|} against the true Weibull medians
     * (medians of curve-based models read off a 60-point grid; NaN for the forest, whose grid is
     * fixed at fit time). The existing methods and their rows are unchanged.
     *
     * @param crossing whether to use the crossing-hazards data (else proportional Weibull data)
     * @param risk the risk of the proportional data (ignored on crossing hazards)
     * @param censoring the censoring share
     * @param n training units (500 more are held out)
     * @param steps Adam steps for the neural models
     * @param trees forest trees (crossing hazards only)
     * @param seed the seed
     * @return model name to {@code {C-index, IBS, median log error}}
     */
    public static Map<String, double[]> aft(boolean crossing, SurvivalData.Risk risk, double censoring, int n,
            int steps, int trees, long seed) {
        Split s = split(crossing ? SurvivalData.crossingHazards(n + 500, 6, censoring, seed)
                : SurvivalData.weibull(n + 500, 8, risk, censoring, seed), n);
        SurvivalData.Survival tr = s.train();
        SurvivalData.Survival te = s.test();
        double[] fine = SurvivalMetrics.grid(tr.time(), 0.01, 0.99, 60);
        Map<String, double[]> out = new LinkedHashMap<>();
        SurvivalModels.KaplanMeier km = new SurvivalModels.KaplanMeier(tr.time(), tr.event());
        double[][] kmCurves = new double[te.size()][s.grid().length];
        double[][] kmFine = new double[te.size()][fine.length];
        for (int i = 0; i < te.size(); i++) {
            for (int g = 0; g < s.grid().length; g++) {
                kmCurves[i][g] = km.survival(s.grid()[g]);
            }
            for (int g = 0; g < fine.length; g++) {
                kmFine[i][g] = km.survival(fine[g]);
            }
        }
        out.put("Kaplan-Meier", aftScore(kmCurves, medians(kmFine, fine), s, crossing, new double[te.size()]));
        SurvivalModels.Cox cox = new SurvivalModels.Cox(tr.x(), tr.time(), tr.event(), 1e-3);
        double[] coxRisk = new double[te.size()];
        for (int i = 0; i < coxRisk.length; i++) {
            coxRisk[i] = cox.logRisk(te.x()[i]);
        }
        out.put("Cox", aftScore(coxCurves(cox, te.x(), s.grid()), medians(coxCurves(cox, te.x(), fine), fine), s,
                crossing, coxRisk));
        SurvivalModels.DeepSurv deep = new SurvivalModels.DeepSurv(tr.x()[0].length, 32, seed)
                .fit(tr.x(), tr.time(), tr.event(), steps, 256, 3e-3f, 1e-3f, seed);
        out.put("DeepSurv", aftScore(deep.survival(te.x(), s.grid()), medians(deep.survival(te.x(), fine), fine), s,
                crossing, deep.logRisk(te.x())));
        if (crossing) {
            double[] cuts = DiscreteHazardNet.quantileCuts(tr.time(), 12);
            DeepHit hit = new DeepHit(tr.x()[0].length, 32, 1, cuts, seed)
                    .fit(tr.x(), tr.time(), tr.event(), steps, 128, 3e-3f, 0.2f, seed);
            out.put("DeepHit", aftScore(hit.survival(te.x(), s.grid()), medians(hit.survival(te.x(), fine), fine), s,
                    true, null));
            SurvivalForest rsf = SurvivalForest.fit(tr.x(), tr.time(), tr.event(), s.grid(), trees, 10, 10, seed);
            out.put("random survival forest", aftScore(rsf.survival(te.x()), null, s, true, null));
        }
        for (AftModels.Kind k : new AftModels.Kind[] {AftModels.Kind.WEIBULL, AftModels.Kind.LOG_NORMAL,
            AftModels.Kind.LOG_LOGISTIC}) {
            AftModels m = AftModels.fit(k, tr.x(), tr.time(), tr.event(), 1e-3);
            double[] med = m.median(te.x());
            out.put(k.name().toLowerCase().replace('_', '-') + " AFT", aftScore(m.survival(te.x(), s.grid()), med, s,
                    crossing, negLog(med)));
        }
        AftModels dsm = AftModels.deepSurvivalMachines(tr.x(), tr.time(), tr.event(), 4, 32, steps, 128, 5e-3f, 1,
                seed);
        double[] med = dsm.median(te.x());
        out.put("Deep Survival Machines (4 Weibulls)", aftScore(dsm.survival(te.x(), s.grid()), med, s, crossing,
                negLog(med)));
        return out;
    }
}
