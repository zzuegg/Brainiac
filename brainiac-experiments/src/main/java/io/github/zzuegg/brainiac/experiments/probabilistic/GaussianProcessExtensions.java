package io.github.zzuegg.brainiac.experiments.probabilistic;

import io.github.zzuegg.brainiac.experimental.ml.ComponentAnalysis;
import io.github.zzuegg.brainiac.experimental.ml.GaussianProcess;
import io.github.zzuegg.brainiac.experimental.ml.GpLvm;
import io.github.zzuegg.brainiac.experimental.ml.HeteroscedasticGp;
import io.github.zzuegg.brainiac.experimental.ml.KernelFunction;
import io.github.zzuegg.brainiac.experimental.ml.KernelTests;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.experimental.ml.MultiOutputGp;
import io.github.zzuegg.brainiac.experimental.ml.StationaryKernel;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Gaussian-process extensions (wave 12, item 12.5.15).
 *
 * <ul>
 *   <li>{@link #multiOutput}: four outputs mixing two latent sinusoids, output 0 unobserved on
 *       {@code [0.35, 0.65]}; an independent GP on output 0 against ICM (one kernel, rank 2) and
 *       LMC (two kernels, rank 1), scored by RMSE and NLL in the gap.</li>
 *   <li>{@link #heteroscedastic}: 1-D regression with noise standard deviation growing from 0.05
 *       to 0.65; a homoscedastic GP against the most-likely-heteroscedastic GP, scored by test
 *       RMSE and NLL.</li>
 *   <li>{@link #latentVariables}: 2-D embeddings of a three-class manifold in 20 dimensions and of
 *       five-class digits, by PCA, kernel PCA and the GPLVM, scored by leave-one-out 5-NN accuracy
 *       in the latent space.</li>
 * </ul>
 */
public final class GaussianProcessExtensions {

    private static final double[] MIX_A = {1.0, 0.8, -0.7, 0.5};
    private static final double[] MIX_B = {0.3, -0.5, 0.4, 0.8};

    private GaussianProcessExtensions() {
    }

    private static double output(int p, double x) {
        return MIX_A[p] * Math.sin(2 * Math.PI * x) + MIX_B[p] * Math.sin(5 * Math.PI * x + 1);
    }

    /**
     * Coregionalised GPs against an independent GP on an output with a gap.
     *
     * @param perOutput observations per output before removing the gap
     * @param iterations Adam steps for every hyperparameter fit
     * @param seed the seed
     * @return model to {gap RMSE, gap NLL}
     */
    public static Map<String, double[]> multiOutput(int perOutput, int iterations, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int outputs = MIX_A.length;
        List<double[]> xs = new ArrayList<>();
        List<Integer> os = new ArrayList<>();
        List<Double> ys = new ArrayList<>();
        for (int p = 0; p < outputs; p++) {
            for (int i = 0; i < perOutput; i++) {
                double x = r.nextDouble();
                if (p == 0 && x > 0.35 && x < 0.65) {
                    continue;
                }
                xs.add(new double[] {x});
                os.add(p);
                ys.add(output(p, x) + 0.1 * r.nextGaussian());
            }
        }
        double[][] x = xs.toArray(new double[0][]);
        int[] o = os.stream().mapToInt(Integer::intValue).toArray();
        double[] y = ys.stream().mapToDouble(Double::doubleValue).toArray();
        int m = 60;
        double[][] gap = new double[m][1];
        double[] truth = new double[m];
        for (int i = 0; i < m; i++) {
            gap[i][0] = 0.35 + 0.3 * (i + 0.5) / m;
            truth[i] = output(0, gap[i][0]) + 0.1 * r.nextGaussian();
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        List<double[]> x0 = new ArrayList<>();
        List<Double> y0 = new ArrayList<>();
        for (int i = 0; i < x.length; i++) {
            if (o[i] == 0) {
                x0.add(x[i]);
                y0.add(y[i]);
            }
        }
        GaussianProcess single = GaussianProcess.optimize(x0.toArray(new double[0][]),
                y0.stream().mapToDouble(Double::doubleValue).toArray(), StationaryKernel.rbf(0.2, 1), 0.05,
                iterations);
        GaussianProcess.Prediction ps = single.predict(gap);
        out.put("independent GP", new double[] {rmse(ps.mean(), truth),
            MlMetrics.gaussianNll(ps.mean(), ps.variance(), truth)});
        MultiOutputGp icm = MultiOutputGp.fit(x, o, y, outputs, 1, 2, iterations, seed);
        double[][] pi = icm.predict(gap, 0);
        out.put("ICM (Q=1, R=2)", new double[] {rmse(pi[0], truth), MlMetrics.gaussianNll(pi[0], pi[1], truth)});
        MultiOutputGp lmc = MultiOutputGp.fit(x, o, y, outputs, 2, 1, iterations, seed);
        double[][] pl = lmc.predict(gap, 0);
        out.put("LMC (Q=2, R=1)", new double[] {rmse(pl[0], truth), MlMetrics.gaussianNll(pl[0], pl[1], truth)});
        return out;
    }

    /**
     * Heteroscedastic against homoscedastic regression.
     *
     * @param n training points
     * @param iterations Adam steps per GP fit
     * @param seed the seed
     * @return model to {test RMSE, test NLL}
     */
    public static Map<String, double[]> heteroscedastic(int n, int iterations, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][1];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i][0] = r.nextDouble();
            y[i] = Math.sin(2 * Math.PI * x[i][0]) + (0.05 + 0.6 * x[i][0]) * r.nextGaussian();
        }
        int m = 500;
        double[][] xt = new double[m][1];
        double[] yt = new double[m];
        for (int i = 0; i < m; i++) {
            xt[i][0] = r.nextDouble();
            yt[i] = Math.sin(2 * Math.PI * xt[i][0]) + (0.05 + 0.6 * xt[i][0]) * r.nextGaussian();
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        GaussianProcess home = GaussianProcess.optimize(x, y, StationaryKernel.rbf(0.2, 1), 0.1, iterations);
        GaussianProcess.Prediction ph = home.predict(xt);
        out.put("homoscedastic GP", new double[] {rmse(ph.mean(), yt), MlMetrics.gaussianNll(ph.mean(),
            ph.variance(), yt)});
        HeteroscedasticGp het = HeteroscedasticGp.fit(x, y, StationaryKernel.rbf(0.2, 1), 3, iterations);
        GaussianProcess.Prediction pe = het.predict(xt);
        out.put("heteroscedastic GP (ML-EM)", new double[] {rmse(pe.mean(), yt), MlMetrics.gaussianNll(pe.mean(),
            pe.variance(), yt)});
        return out;
    }

    /**
     * Two-dimensional latent spaces by PCA, kernel PCA and the GPLVM.
     *
     * @param perClass points per class
     * @param iterations GPLVM Adam steps
     * @param seed the seed
     * @return {@code "dataset / method"} to the leave-one-out 5-NN accuracy in the latent space
     */
    public static Map<String, Double> latentVariables(int perClass, int iterations, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        SplittableRandom r = new SplittableRandom(seed);
        int classes = 3;
        int n = classes * perClass;
        double[][] a = new double[20][2];
        double[] b = new double[20];
        for (int j = 0; j < 20; j++) {
            a[j][0] = 3 * r.nextGaussian();
            a[j][1] = 3 * r.nextGaussian();
            b[j] = 2 * Math.PI * r.nextDouble();
        }
        double[][] manifold = new double[n][20];
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) {
            int c = i % classes;
            labels[i] = c;
            double angle = 2 * Math.PI * c / classes;
            double z0 = Math.cos(angle) + 0.35 * r.nextGaussian();
            double z1 = Math.sin(angle) + 0.35 * r.nextGaussian();
            for (int j = 0; j < 20; j++) {
                manifold[i][j] = Math.sin(a[j][0] * z0 + a[j][1] * z1 + b[j]) + 0.1 * r.nextGaussian();
            }
        }
        embedAll(out, "manifold 20-D", manifold, labels, iterations);
        ToyTasks.Data digits = ToyTasks.digits(5, perClass, 8, 0.1, seed);
        double[][] dx = new double[digits.x().length][];
        for (int i = 0; i < dx.length; i++) {
            dx[i] = new double[digits.x()[i].length];
            for (int j = 0; j < dx[i].length; j++) {
                dx[i][j] = digits.x()[i][j];
            }
        }
        embedAll(out, "digits 64-D", dx, digits.y(), iterations);
        return out;
    }

    private static void embedAll(Map<String, Double> out, String name, double[][] x, int[] labels, int iterations) {
        out.put(name + " / PCA", MlMetrics.knnAccuracy(ComponentAnalysis.pca(x, 2).transform(x), labels, 5));
        double sigma = KernelTests.medianDistance(x);
        double[][] kp = ComponentAnalysis.kernelPca(x, KernelFunction.rbf(1 / (2 * sigma * sigma)), 2).transform(x);
        out.put(name + " / kernel PCA", MlMetrics.knnAccuracy(kp, labels, 5));
        out.put(name + " / GPLVM", MlMetrics.knnAccuracy(GpLvm.fit(x, 2, iterations, 0.05).latent(), labels, 5));
    }

    private static double rmse(double[] p, double[] t) {
        double s = 0;
        for (int i = 0; i < p.length; i++) {
            s += (p[i] - t[i]) * (p[i] - t[i]);
        }
        return Math.sqrt(s / p.length);
    }
}
