package io.github.zzuegg.brainiac.experiments.classic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.ml.IsolationForest;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.experimental.ml.RandomForest;
import io.github.zzuegg.brainiac.experiments.graphs.RecsysTabularBenchmarks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.GradientBoostedTrees;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Tree ensembles on tabular data and anomaly detection.
 * <ul>
 *   <li>{@link #tabular}: the {@code RecsysTabularBenchmarks} interaction table (label from
 *       products of features, plus distractor columns): a random forest (with its out-of-bag
 *       error) and extra trees against core {@link GradientBoostedTrees} and the MLP of
 *       {@code RecsysTabularBenchmarks};</li>
 *   <li>{@link #anomalies}: a correlated two-cluster cloud with 5% planted uniform outliers,
 *       scored unsupervised: the isolation forest against the k-nearest-neighbour distance and
 *       an autoencoder's reconstruction error, by AUROC.</li>
 * </ul>
 */
public final class ForestBenchmarks {

    private ForestBenchmarks() {
    }

    private static double[][] rows(Tensor t) {
        int n = t.shape().dim(0);
        int d = t.shape().dim(1);
        float[] v = t.toArray();
        double[][] x = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                x[i][j] = v[i * d + j];
            }
        }
        return x;
    }

    /**
     * Test accuracy on the tabular task.
     *
     * @param trees trees per forest
     * @param mlpSteps MLP training steps
     * @param seed the seed
     * @return model to test accuracy, plus "RF OOB error"
     */
    public static Map<String, Double> tabular(int trees, int mlpSteps, long seed) {
        SyntheticRecsys.Table train = SyntheticRecsys.tabular(2000, 10, seed);
        SyntheticRecsys.Table test = SyntheticRecsys.tabular(1000, 10, seed + 1);
        double[][] x = rows(train.features());
        double[][] xt = rows(test.features());
        float[] yf = train.labels().toArray();
        float[] ytf = test.labels().toArray();
        int[] y = new int[yf.length];
        double[] yd = new double[yf.length];
        for (int i = 0; i < y.length; i++) {
            y[i] = yf[i] > 0.5f ? 1 : 0;
            yd[i] = y[i];
        }
        int[] yt = new int[ytf.length];
        for (int i = 0; i < yt.length; i++) {
            yt[i] = ytf[i] > 0.5f ? 1 : 0;
        }
        Map<String, Double> out = new LinkedHashMap<>();
        RandomForest rf = RandomForest.fit(x, y, 2,
                RandomForest.Settings.defaults(RandomForest.Mode.RANDOM_FOREST).withTrees(trees));
        out.put("random forest", MlMetrics.accuracy(rf.predict(xt), yt));
        out.put("RF OOB error", rf.oobError());
        RandomForest et = RandomForest.fit(x, y, 2,
                RandomForest.Settings.defaults(RandomForest.Mode.EXTRA_TREES).withTrees(trees));
        out.put("extra trees", MlMetrics.accuracy(et.predict(xt), yt));
        GradientBoostedTrees gbt = GradientBoostedTrees.fit(x, yd, GradientBoostedTrees.Objective.BINARY,
                GradientBoostedTrees.Settings.defaults());
        double[] p = gbt.predict(xt);
        int[] gp = new int[p.length];
        for (int i = 0; i < p.length; i++) {
            gp[i] = p[i] > 0.5 ? 1 : 0;
        }
        out.put("GBT", MlMetrics.accuracy(gp, yt));
        Model mlp = Model.sequential().input(10).dense(32, Activation.RELU).dense(32, Activation.RELU).dense(1)
                .seed(seed).build();
        out.put("MLP", RecsysTabularBenchmarks.accuracy(v -> mlp.forward(v, true), mlp.parameters(), train, test,
                mlpSteps, 0.01f, seed));
        return out;
    }

    /**
     * Unsupervised anomaly detection AUROC.
     *
     * @param seed the seed
     * @return detector to AUROC of the planted outliers
     */
    public static Map<String, Double> anomalies(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int normal = 950;
        int outliers = 50;
        int d = 4;
        double[][] x = new double[normal + outliers][d];
        boolean[] anomalous = new boolean[x.length];
        for (int i = 0; i < normal; i++) {
            double shift = i % 2 == 0 ? -2 : 2;
            double z = r.nextGaussian();
            for (int j = 0; j < d; j++) {
                // a strongly correlated cloud: every coordinate follows z plus a little noise
                x[i][j] = shift + z + 0.3 * r.nextGaussian();
            }
        }
        for (int i = normal; i < x.length; i++) {
            for (int j = 0; j < d; j++) {
                x[i][j] = r.nextDouble(-5, 5);
            }
            anomalous[i] = true;
        }
        Map<String, Double> out = new LinkedHashMap<>();
        IsolationForest forest = IsolationForest.fit(x, 100, 256, seed);
        out.put("isolation forest", MlMetrics.auroc(forest.score(x), anomalous));
        double[][] d2 = MlMatrices.squaredDistances(x);
        double[] knn = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            int[] nn = MlMetrics.nearest(d2[i], i, 10);
            knn[i] = d2[i][nn[9]];
        }
        out.put("kNN distance", MlMetrics.auroc(knn, anomalous));
        Model ae = Model.sequential().input(d).dense(16, Activation.TANH).dense(1).dense(16, Activation.TANH).dense(d)
                .seed(seed).build();
        Tensor tx = Tensor.of(MlMatrices.toFloat(x));
        Optimizer opt = Optimizer.adam(0.01f);
        for (int s = 0; s < 400; s++) {
            ae.forward(Var.of(tx), true).sub(tx).square().mean().backward();
            opt.step(ae.parameters());
        }
        float[] rec = ae.predict(tx).toArray();
        double[] err = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < d; j++) {
                double e = rec[i * d + j] - x[i][j];
                err[i] += e * e;
            }
        }
        out.put("autoencoder", MlMetrics.auroc(err, anomalous));
        return out;
    }
}
