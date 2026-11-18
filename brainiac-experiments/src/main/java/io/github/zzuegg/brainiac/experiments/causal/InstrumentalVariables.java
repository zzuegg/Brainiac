package io.github.zzuegg.brainiac.experiments.causal;

import io.github.zzuegg.brainiac.experimental.causal.Regressor;
import io.github.zzuegg.brainiac.experimental.data.CausalScenarios;
import io.github.zzuegg.brainiac.experimental.ml.RegressionForest;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Instrumental variables on Hartford et al.'s airline-demand simulation (W9 1.14,
 * {@link CausalScenarios#airline}): the price is confounded by an unobserved demand shock and fuel
 * cost is the instrument. Every method is scored twice on fresh data: the <em>observational</em>
 * MSE of predicting sales at the observed price, and the <em>counterfactual</em> MSE of
 * {@code h(p, x)} against the true demand when prices are set by intervention (shuffled across
 * rows, which breaks the confounding). Methods: DeepIV, linear 2SLS, an MLP regressing sales on
 * (price, covariates) directly, and a random forest adjusting for the observed covariates (what
 * covariate adjustment such as AIPW can see). Counterfactual errors are divided by the variance of
 * the true demand, so 1 is as bad as predicting its mean.
 */
public final class InstrumentalVariables {

    private InstrumentalVariables() {
    }

    /**
     * Runs the comparison.
     *
     * @param n training rows
     * @param rho the confounding strength
     * @param steps Adam steps per DeepIV stage (and for the direct MLP)
     * @param seed the seed
     * @return method name to {@code {observational MSE, relative counterfactual MSE}}
     */
    public static Map<String, double[]> run(int n, double rho, int steps, long seed) {
        CausalScenarios.IvData train = CausalScenarios.airline(n, rho, seed);
        CausalScenarios.IvData test = CausalScenarios.airline(1000, rho, seed + 1);
        double[] shuffled = test.price().clone();
        SplittableRandom r = new SplittableRandom(seed + 2);
        for (int i = shuffled.length - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            double t = shuffled[i];
            shuffled[i] = shuffled[j];
            shuffled[j] = t;
        }
        double[] truth = new double[shuffled.length];
        double mean = 0;
        for (int i = 0; i < truth.length; i++) {
            truth[i] = CausalScenarios.demand(shuffled[i], test.x()[i][0], test.x()[i][1]);
            mean += truth[i] / truth.length;
        }
        double var = 0;
        for (double v : truth) {
            var += (v - mean) * (v - mean) / truth.length;
        }
        double[][] z = column(train.z());
        Map<String, double[]> out = new LinkedHashMap<>();
        var deepIv = new io.github.zzuegg.brainiac.experimental.causal.InstrumentalVariables.DeepIv(2, 1, 5, 64, 8,
                seed).fit(train.x(), z, train.price(), train.sales(), steps, steps);
        out.put("DeepIV", score(deepIv::predict, test, shuffled, truth, var));
        var direct = new io.github.zzuegg.brainiac.experimental.causal.InstrumentalVariables.DeepIv(2, 1, 5, 64, 8,
                seed).fitDirect(train.x(), train.price(), train.sales(), steps);
        out.put("direct MLP (confounded)", score(direct::predict, test, shuffled, truth, var));
        var linear = io.github.zzuegg.brainiac.experimental.causal.InstrumentalVariables.twoStageLeastSquares(
                train.x(), z, train.price(), train.sales());
        out.put("linear 2SLS", score((p, x) -> {
            double[] o = new double[p.length];
            for (int i = 0; i < o.length; i++) {
                o[i] = linear.predict(p[i], x[i]);
            }
            return o;
        }, test, shuffled, truth, var));
        double[][] tx = new double[n][];
        for (int i = 0; i < n; i++) {
            tx[i] = new double[] {train.price()[i], train.x()[i][0], train.x()[i][1]};
        }
        Regressor.Fitted adjust = Regressor.forest(RegressionForest.Settings.defaults().withTrees(40).withSeed(seed))
                .fit(tx, train.sales());
        out.put("covariate adjustment (forest)", score((p, x) -> {
            double[] o = new double[p.length];
            for (int i = 0; i < o.length; i++) {
                o[i] = adjust.predict(new double[] {p[i], x[i][0], x[i][1]});
            }
            return o;
        }, test, shuffled, truth, var));
        return out;
    }

    private interface Structural {
        double[] predict(double[] price, double[][] x);
    }

    private static double[] score(Structural h, CausalScenarios.IvData test, double[] shuffled, double[] truth,
            double var) {
        double[] obs = h.predict(test.price(), test.x());
        double o = 0;
        for (int i = 0; i < obs.length; i++) {
            o += (obs[i] - test.sales()[i]) * (obs[i] - test.sales()[i]) / obs.length;
        }
        double[] cf = h.predict(shuffled, test.x());
        double c = 0;
        for (int i = 0; i < cf.length; i++) {
            c += (cf[i] - truth[i]) * (cf[i] - truth[i]) / cf.length;
        }
        return new double[] {o, c / var};
    }

    private static double[][] column(double[] v) {
        double[][] out = new double[v.length][];
        for (int i = 0; i < v.length; i++) {
            out[i] = new double[] {v[i]};
        }
        return out;
    }
}
