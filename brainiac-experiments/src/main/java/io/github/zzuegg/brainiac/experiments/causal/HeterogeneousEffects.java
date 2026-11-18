package io.github.zzuegg.brainiac.experiments.causal;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.causal.CausalForest;
import io.github.zzuegg.brainiac.experimental.causal.DoubleMl;
import io.github.zzuegg.brainiac.experimental.causal.MetaLearners;
import io.github.zzuegg.brainiac.experimental.causal.Regressor;
import io.github.zzuegg.brainiac.experimental.causal.UpliftMetrics;
import io.github.zzuegg.brainiac.experimental.causal.UpliftModels;
import io.github.zzuegg.brainiac.experimental.data.CausalScenarios;
import io.github.zzuegg.brainiac.experimental.ml.CausalEstimators;
import io.github.zzuegg.brainiac.experimental.ml.RegressionForest;
import io.github.zzuegg.brainiac.experimental.nn.TreatmentEffectNet;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.GradientBoostedTrees;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Heterogeneous treatment effects, debiased average effects and uplift targeting (W9 1.10 to
 * 1.13), all on {@link CausalScenarios} with a known per-unit effect:
 * <ul>
 *   <li>{@link #metaLearners}: S-, T-, X-, R- and DR-learners on random forests against the linear
 *       T-learner of {@link CausalEstimators}, TARNet and DragonNet ({@link TreatmentEffectNet}) and
 *       a causal forest, by PEHE on held-out units, in the Künzel et al. settings;</li>
 *   <li>{@link #averageEffect}: double machine learning (cross-fitted partially linear and AIPW
 *       scores) against plug-in AIPW with the same forest nuisances fitted on all data, the naive
 *       difference and logistic IPW: bias, RMSE and 95 % interval coverage over replications;</li>
 *   <li>{@link #causalForest}: the honest causal forest's PEHE and 90 % interval coverage on the
 *       step-shaped effect, against forest meta-learners;</li>
 *   <li>{@link #uplift}: uplift forests (Delta-Delta-P and KL), class transformation and
 *       T/X-learners against a response model and random targeting, by Qini coefficient and
 *       profit at the best targeting fraction.</li>
 * </ul>
 */
public final class HeterogeneousEffects {

    private HeterogeneousEffects() {
    }

    private static Regressor forest(int trees, long seed) {
        return Regressor.forest(RegressionForest.Settings.defaults().withTrees(trees).withMinLeaf(10).withSeed(seed));
    }

    /**
     * PEHE of every CATE estimator on held-out units.
     *
     * @param setting the Künzel setting
     * @param n training units (as many again are held out)
     * @param trees trees per forest
     * @param netSteps Adam steps for TARNet and DragonNet
     * @param seed the seed
     * @return method name to PEHE
     */
    public static Map<String, Double> metaLearners(CausalScenarios.Setting setting, int n, int trees, int netSteps,
            long seed) {
        CausalScenarios.Observational all = CausalScenarios.kunzel(setting, 2 * n, seed);
        CausalScenarios.Observational train = all.rows(0, n);
        CausalScenarios.Observational test = all.rows(n, 2 * n);
        Map<String, Double> out = new LinkedHashMap<>();
        MetaLearners learners = new MetaLearners(forest(trees, seed), forest(trees, seed + 1), forest(trees, seed + 2))
                .seed(seed);
        for (MetaLearners.Kind kind : MetaLearners.Kind.values()) {
            Regressor.Fitted tau = learners.fit(kind, train.x(), train.t(), train.y());
            out.put(kind + "-learner (forest)", MetaLearners.pehe(tau.predict(test.x()), test.tau()));
        }
        // residual learners with forest nuisances but a ridge final stage (Nie and Wager's point: the
        // effect can be simpler than the outcome)
        MetaLearners linearEffect = new MetaLearners(forest(trees, seed), Regressor.ridge(1.0),
                forest(trees, seed + 2)).seed(seed);
        for (MetaLearners.Kind kind : new MetaLearners.Kind[] {MetaLearners.Kind.R, MetaLearners.Kind.DR}) {
            Regressor.Fitted tau = linearEffect.fit(kind, train.x(), train.t(), train.y());
            out.put(kind + "-learner (forest nuisances, ridge effect)",
                    MetaLearners.pehe(tau.predict(test.x()), test.tau()));
        }
        // the ridge-per-arm T-learner of CausalEstimators.outcomes, refitted so it predicts held-out units
        Regressor.Fitted mu0 = Regressor.ridge(1e-3).fit(arm(train, 0)[0], arm(train, 0)[1][0]);
        Regressor.Fitted mu1 = Regressor.ridge(1e-3).fit(arm(train, 1)[0], arm(train, 1)[1][0]);
        double[] linTau = new double[test.size()];
        for (int i = 0; i < linTau.length; i++) {
            linTau[i] = mu1.predict(test.x()[i]) - mu0.predict(test.x()[i]);
        }
        out.put("T-learner (linear)", MetaLearners.pehe(linTau, test.tau()));
        for (TreatmentEffectNet.Kind kind : TreatmentEffectNet.Kind.values()) {
            out.put(kind == TreatmentEffectNet.Kind.TARNET ? "TARNet" : "DragonNet",
                    MetaLearners.pehe(net(kind, train, test.x(), netSteps, seed), test.tau()));
        }
        CausalForest cf = CausalForest.fit(train.x(), train.t(), train.y(),
                CausalForest.Settings.defaults(seed).withTrees(2 * trees));
        out.put("causal forest", MetaLearners.pehe(cf.predict(test.x()), test.tau()));
        return out;
    }

    private static double[][][] arm(CausalScenarios.Observational d, int a) {
        int c = 0;
        for (int v : d.t()) {
            c += v == a ? 1 : 0;
        }
        double[][] x = new double[c][];
        double[] y = new double[c];
        int k = 0;
        for (int i = 0; i < d.size(); i++) {
            if (d.t()[i] == a) {
                x[k] = d.x()[i];
                y[k++] = d.y()[i];
            }
        }
        return new double[][][] {x, {y}};
    }

    private static double[] net(TreatmentEffectNet.Kind kind, CausalScenarios.Observational train, double[][] test,
            int steps, long seed) {
        int d = train.x()[0].length;
        TreatmentEffectNet net = new TreatmentEffectNet(kind, d, 64, new SplittableRandom(seed));
        Tensor tx = matrix(train.x());
        Tensor tt = vector(train.t());
        Tensor ty = vector(train.y());
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        // early stopping on the factual MSE of the last 20% of the training rows, as in TARNet
        int n = train.size() * 4 / 5;
        int valid = train.size() - n;
        Tensor vx = tx.slice(0, n, n + valid);
        Tensor vt = tt.slice(0, n, n + valid);
        Tensor vy = ty.slice(0, n, n + valid);
        Tensor testX = matrix(test);
        int batch = 128;
        double best = Double.MAX_VALUE;
        double[] bestCate = Var.noGrad(() -> net.cate(testX)).toDoubleArray();
        for (int s = 0; s < steps; s++) {
            int from = r.nextInt(n - batch);
            net.loss(net.forward(Var.of(tx.slice(0, from, from + batch))), tt.slice(0, from, from + batch),
                    ty.slice(0, from, from + batch), 1f, 1f).backward();
            opt.step(net.parameters());
            if ((s + 1) % 25 == 0 || s == steps - 1) {
                Tensor out = Var.noGrad(() -> net.forward(Var.of(vx))).value();
                Tensor tc = vt.reshape(valid, 1);
                Tensor factual = out.slice(1, 1, 2).mul(tc).add(out.slice(1, 0, 1).mul(tc.neg().add(1f)));
                double mse = factual.sub(vy.reshape(valid, 1)).square().mean().item();
                if (mse < best) {
                    best = mse;
                    bestCate = Var.noGrad(() -> net.cate(testX)).toDoubleArray();
                }
            }
        }
        return bestCate;
    }

    private static Tensor matrix(double[][] x) {
        int d = x[0].length;
        float[] v = new float[x.length * d];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < d; j++) {
                v[i * d + j] = (float) x[i][j];
            }
        }
        return Tensor.of(Shape.of(x.length, d), v);
    }

    private static Tensor vector(double[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = (float) v[i];
        }
        return Tensor.of(Shape.of(v.length), f);
    }

    private static Tensor vector(int[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = v[i];
        }
        return Tensor.of(Shape.of(v.length), f);
    }

    /**
     * Summary of an ATE estimator over replications.
     *
     * @param bias mean estimate minus truth
     * @param rmse root mean squared error
     * @param coverage the share of 95 % intervals covering the truth (NaN without intervals)
     */
    public record AteSummary(double bias, double rmse, double coverage) {
    }

    /**
     * ATE estimators over replications of {@link CausalScenarios#nonlinearConfounding} (true ATE 1).
     *
     * @param n units per replication
     * @param replications replications
     * @param trees trees per nuisance forest
     * @param seed the base seed
     * @return estimator name to its summary
     */
    public static Map<String, AteSummary> averageEffect(int n, int replications, int trees, long seed) {
        String[] names = {"naive", "IPW (logistic)", "AIPW plug-in (forest, no cross-fit)",
            "DML-IRM (forest, cross-fit)", "DML-PLR (forest, cross-fit)", "DML-IRM (boosted, cross-fit)"};
        double[][] est = new double[names.length][replications];
        boolean[][] cover = new boolean[names.length][replications];
        for (int rep = 0; rep < replications; rep++) {
            long s = seed + 1000L * rep;
            CausalScenarios.Observational d = CausalScenarios.nonlinearConfounding(n, s);
            double truth = 1;
            Regressor rf = Regressor.forest(RegressionForest.Settings.defaults().withTrees(trees).withMinLeaf(2)
                    .withSeed(s));
            Regressor gbt = Regressor.boosted(GradientBoostedTrees.Settings.defaults().withTrees(60).withDepth(3));
            est[0][rep] = CausalEstimators.naive(d.t(), d.y());
            est[1][rep] = CausalEstimators.ipw(d.t(), d.y(), CausalEstimators.propensity(d.x(), d.t(), 1e-3), 0.01);
            DoubleMl.Estimate plug = DoubleMl.interactive(d.x(), d.t(), d.y(), rf, rf, 5, false, 0.02, s);
            DoubleMl.Estimate irm = DoubleMl.interactive(d.x(), d.t(), d.y(), rf, rf, 5, true, 0.02, s);
            DoubleMl.Estimate plr = DoubleMl.partiallyLinear(d.x(), asDouble(d.t()), d.y(), rf, rf, 5, true, s);
            DoubleMl.Estimate boosted = DoubleMl.interactive(d.x(), d.t(), d.y(), gbt, gbt, 5, true, 0.02, s);
            DoubleMl.Estimate[] withCi = {plug, irm, plr, boosted};
            for (int k = 0; k < withCi.length; k++) {
                est[2 + k][rep] = withCi[k].theta();
                cover[2 + k][rep] = withCi[k].covers(truth);
            }
        }
        Map<String, AteSummary> out = new LinkedHashMap<>();
        for (int k = 0; k < names.length; k++) {
            double bias = 0;
            double mse = 0;
            int covered = 0;
            for (int rep = 0; rep < replications; rep++) {
                bias += (est[k][rep] - 1) / replications;
                mse += (est[k][rep] - 1) * (est[k][rep] - 1) / replications;
                covered += cover[k][rep] ? 1 : 0;
            }
            out.put(names[k], new AteSummary(bias, Math.sqrt(mse), k < 2 ? Double.NaN
                    : covered / (double) replications));
        }
        return out;
    }

    private static double[] asDouble(int[] t) {
        double[] out = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            out[i] = t[i];
        }
        return out;
    }

    /**
     * The causal forest on Wager and Athey's step effect: PEHE and 90 % interval coverage on held-out
     * units, against forest R- and T-learners.
     *
     * @param n training units
     * @param trees causal-forest trees
     * @param seed the seed
     * @return metric name to value
     */
    public static Map<String, Double> causalForest(int n, int trees, long seed) {
        CausalScenarios.Observational all = CausalScenarios.stepEffect(n + 400, seed);
        CausalScenarios.Observational train = all.rows(0, n);
        CausalScenarios.Observational test = all.rows(n, n + 400);
        CausalForest cf = CausalForest.fit(train.x(), train.t(), train.y(),
                CausalForest.Settings.defaults(seed).withTrees(trees));
        double[] est = new double[test.size()];
        int covered = 0;
        double width = 0;
        for (int i = 0; i < est.length; i++) {
            CausalForest.Prediction p = cf.predict(test.x()[i]);
            est[i] = p.tau();
            double[] ci = p.interval(1.645);
            covered += ci[0] <= test.tau()[i] && test.tau()[i] <= ci[1] ? 1 : 0;
            width += (ci[1] - ci[0]) / est.length;
        }
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("causal forest PEHE", MetaLearners.pehe(est, test.tau()));
        out.put("causal forest 90% coverage", covered / (double) est.length);
        out.put("causal forest mean CI width", width);
        MetaLearners learners = new MetaLearners(forest(trees / 2, seed), forest(trees / 2, seed + 1),
                forest(trees / 2, seed + 2)).seed(seed);
        for (MetaLearners.Kind kind : new MetaLearners.Kind[] {MetaLearners.Kind.T, MetaLearners.Kind.R,
            MetaLearners.Kind.DR}) {
            out.put(kind + "-learner (forest) PEHE", MetaLearners.pehe(learners.fit(kind, train.x(), train.t(),
                    train.y()).predict(test.x()), test.tau()));
        }
        return out;
    }

    /**
     * Uplift targeting on a randomised campaign.
     *
     * @param n training customers (as many again are held out)
     * @param value the value of a conversion
     * @param cost the cost of a contact
     * @param seed the seed
     * @return model name to {@code {Qini coefficient, AUUC, best profit per customer, best fraction}}
     */
    public static Map<String, double[]> uplift(int n, double value, double cost, long seed) {
        CausalScenarios.Campaign all = CausalScenarios.campaign(2 * n, seed);
        double[][] trainX = java.util.Arrays.copyOfRange(all.x(), 0, n);
        int[] trainT = java.util.Arrays.copyOfRange(all.t(), 0, n);
        int[] trainY = java.util.Arrays.copyOfRange(all.y(), 0, n);
        double[][] testX = java.util.Arrays.copyOfRange(all.x(), n, 2 * n);
        int[] testT = java.util.Arrays.copyOfRange(all.t(), n, 2 * n);
        int[] testY = java.util.Arrays.copyOfRange(all.y(), n, 2 * n);
        double[] trueUplift = java.util.Arrays.copyOfRange(all.uplift(), n, 2 * n);
        Map<String, double[]> scores = new LinkedHashMap<>();
        scores.put("oracle (true uplift)", trueUplift);
        scores.put("random", new SplittableRandom(seed).doubles(n).toArray());
        Regressor rf = forest(60, seed);
        double[] yd = asDouble(trainY);
        scores.put("response model (forest)", rf.fit(trainX, yd).predict(testX));
        scores.put("class transformation (forest)",
                UpliftModels.classTransformation(trainX, trainT, trainY, rf).predict(testX));
        scores.put("two-model / T-learner (forest)", UpliftModels.twoModel(trainX, trainT, trainY, rf).predict(testX));
        scores.put("X-learner (forest)", new MetaLearners(rf, rf, Regressor.ridge(1e-3)).fit(MetaLearners.Kind.X,
                trainX, trainT, yd).predict(testX));
        scores.put("uplift forest (DDP)", UpliftModels.forest(trainX, trainT, trainY,
                UpliftModels.Criterion.DELTA_DELTA_P, 60, 6, 25, seed).predict(testX));
        scores.put("uplift forest (KL)", UpliftModels.forest(trainX, trainT, trainY, UpliftModels.Criterion.KL, 60, 6,
                25, seed).predict(testX));
        Map<String, double[]> out = new LinkedHashMap<>();
        for (Map.Entry<String, double[]> e : scores.entrySet()) {
            double[] profit = UpliftMetrics.bestProfit(e.getValue(), trueUplift, value, cost);
            out.put(e.getKey(), new double[] {UpliftMetrics.qiniCoefficient(e.getValue(), testT, testY),
                UpliftMetrics.auuc(e.getValue(), testT, testY), profit[0], profit[1]});
        }
        return out;
    }
}
