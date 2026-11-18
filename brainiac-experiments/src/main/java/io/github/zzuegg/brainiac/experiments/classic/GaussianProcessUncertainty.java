package io.github.zzuegg.brainiac.experiments.classic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SeasonalSeries;
import io.github.zzuegg.brainiac.experimental.ml.DeepKernelLearning;
import io.github.zzuegg.brainiac.experimental.ml.GaussianProcess;
import io.github.zzuegg.brainiac.experimental.ml.KMeans;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.experimental.ml.SparseVariationalGp;
import io.github.zzuegg.brainiac.experimental.ml.StationaryKernel;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Gaussian processes against the repo's neural uncertainty baselines.
 * <ul>
 *   <li>{@link #gapRegression}: 1-D regression with a gap in the training inputs. An exact GP
 *       with marginal-likelihood hyperparameters against a deep ensemble, MC dropout and a
 *       last-layer Laplace (for a Gaussian likelihood: Bayesian linear regression on the
 *       trained network's last hidden layer, the regression analogue of
 *       {@code LastLayerLaplace}). Reported: NLL, CRPS and 95% coverage on test points inside
 *       the training range and inside the gap.</li>
 *   <li>{@link #sparseScaling}: the same task with 5,000 points. A sparse variational GP
 *       (Hensman et al. 2013) with 4 to 64 inducing points against an exact GP on a
 *       1,000-point subset: test NLL against inducing-point count.</li>
 *   <li>{@link #deepKernel}: one-step forecasts of a {@link SeasonalSeries}: deep kernel
 *       learning against the same MLP alone (with a Gaussian residual) and a GP on the raw
 *       windows.</li>
 * </ul>
 */
public final class GaussianProcessUncertainty {

    private GaussianProcessUncertainty() {
    }

    /**
     * Scores of one method.
     *
     * @param nllIn NLL inside the training range
     * @param nllGap NLL inside the gap
     * @param crpsIn CRPS inside the training range
     * @param crpsGap CRPS inside the gap
     * @param coverageIn 95% interval coverage inside the training range
     * @param coverageGap 95% interval coverage inside the gap
     */
    public record Scores(double nllIn, double nllGap, double crpsIn, double crpsGap, double coverageIn,
            double coverageGap) {
    }

    /** The regression target. */
    static double target(double x) {
        return Math.sin(1.5 * x) + 0.3 * x;
    }

    /**
     * Draws inputs from {@code [-3, -1] u [1, 3]} (or the gap {@code (-1, 1)}) with noisy targets.
     *
     * @param n the number of points
     * @param gap whether to draw from the gap
     * @param noise the target noise standard deviation
     * @param random the source
     * @return {inputs {@code [n][1]}, targets {@code [n][1]}}
     */
    static double[][][] draw(int n, boolean gap, double noise, SplittableRandom random) {
        double[][] x = new double[n][1];
        double[][] y = new double[n][1];
        for (int i = 0; i < n; i++) {
            double v;
            if (gap) {
                v = random.nextDouble(-0.9, 0.9);
            } else {
                v = random.nextDouble(1, 3) * (random.nextBoolean() ? 1 : -1);
            }
            x[i][0] = v;
            y[i][0] = target(v) + noise * random.nextGaussian();
        }
        return new double[][][] {x, y};
    }

    private static double[] column(double[][] y) {
        double[] c = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            c[i] = y[i][0];
        }
        return c;
    }

    /** Full-batch Adam on the mean squared error. */
    private static void fit(Model model, double[][] x, double[][] y, int steps, float lr) {
        Tensor tx = Tensor.of(MlMatrices.toFloat(x));
        Tensor ty = Tensor.of(MlMatrices.toFloat(y));
        Optimizer opt = Optimizer.adam(lr);
        for (int s = 0; s < steps; s++) {
            model.forward(Var.of(tx), true).sub(ty).square().mean().backward();
            opt.step(model.parameters());
        }
    }

    private static double[] predict(Model model, double[][] x, boolean training) {
        float[] out = model.forward(Var.of(Tensor.of(MlMatrices.toFloat(x))), training).value().toArray();
        double[] d = new double[out.length];
        for (int i = 0; i < d.length; i++) {
            d[i] = out[i];
        }
        return d;
    }

    private static double residualVariance(double[] prediction, double[] y) {
        double s = 0;
        for (int i = 0; i < y.length; i++) {
            s += (prediction[i] - y[i]) * (prediction[i] - y[i]);
        }
        return Math.max(s / y.length, 1e-4);
    }

    private static Scores score(double[][] inMeanVar, double[] yIn, double[][] gapMeanVar, double[] yGap) {
        return new Scores(MlMetrics.gaussianNll(inMeanVar[0], inMeanVar[1], yIn),
                MlMetrics.gaussianNll(gapMeanVar[0], gapMeanVar[1], yGap),
                MlMetrics.gaussianCrps(inMeanVar[0], inMeanVar[1], yIn),
                MlMetrics.gaussianCrps(gapMeanVar[0], gapMeanVar[1], yGap),
                MlMetrics.coverage(inMeanVar[0], inMeanVar[1], yIn, 1.96),
                MlMetrics.coverage(gapMeanVar[0], gapMeanVar[1], yGap, 1.96));
    }

    /** Mean and variance over sampled predictions plus a noise variance. */
    private static double[][] moments(List<double[]> samples, double noise) {
        int n = samples.getFirst().length;
        double[] mean = new double[n];
        double[] var = new double[n];
        for (double[] s : samples) {
            for (int i = 0; i < n; i++) {
                mean[i] += s[i] / samples.size();
            }
        }
        for (double[] s : samples) {
            for (int i = 0; i < n; i++) {
                var[i] += (s[i] - mean[i]) * (s[i] - mean[i]) / samples.size();
            }
        }
        for (int i = 0; i < n; i++) {
            var[i] += noise;
        }
        return new double[][] {mean, var};
    }

    private static Model mlp(int inputs, int hidden, long seed) {
        return Model.sequential().input(inputs).dense(hidden, Activation.TANH).dense(hidden, Activation.TANH).dense(1)
                .seed(seed).build();
    }

    /**
     * The gap-regression comparison.
     *
     * @param seed the seed
     * @return method to scores
     */
    public static Map<String, Scores> gapRegression(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double noise = 0.1;
        double[][][] train = draw(60, false, noise, r);
        double[][][] testIn = draw(200, false, noise, r);
        double[][][] testGap = draw(100, true, noise, r);
        double[] yTrain = column(train[1]);
        double[] yIn = column(testIn[1]);
        double[] yGap = column(testGap[1]);
        Map<String, Scores> out = new LinkedHashMap<>();

        GaussianProcess gp = GaussianProcess.optimize(train[0], yTrain, StationaryKernel.rbf(1, 1), 0.05, 80);
        GaussianProcess.Prediction pIn = gp.predict(testIn[0]);
        GaussianProcess.Prediction pGap = gp.predict(testGap[0]);
        out.put("exact GP", score(new double[][] {pIn.mean(), pIn.variance()}, yIn,
                new double[][] {pGap.mean(), pGap.variance()}, yGap));

        List<double[]> ensIn = new java.util.ArrayList<>();
        List<double[]> ensGap = new java.util.ArrayList<>();
        double ensNoise = 0;
        Model first = null;
        for (int m = 0; m < 5; m++) {
            Model net = mlp(1, 32, seed * 10 + m);
            fit(net, train[0], train[1], 500, 0.01f);
            ensIn.add(predict(net, testIn[0], false));
            ensGap.add(predict(net, testGap[0], false));
            ensNoise += residualVariance(predict(net, train[0], false), yTrain) / 5;
            if (first == null) {
                first = net;
            }
        }
        out.put("deep ensemble", score(moments(ensIn, ensNoise), yIn, moments(ensGap, ensNoise), yGap));

        Model dropout = Model.sequential().input(1).dense(64, Activation.RELU).dropout(0.1f)
                .dense(64, Activation.RELU).dropout(0.1f).dense(1).seed(seed + 77).build();
        fit(dropout, train[0], train[1], 800, 0.01f);
        List<double[]> mcIn = new java.util.ArrayList<>();
        List<double[]> mcGap = new java.util.ArrayList<>();
        for (int s = 0; s < 40; s++) {
            mcIn.add(predict(dropout, testIn[0], true));
            mcGap.add(predict(dropout, testGap[0], true));
        }
        double mcNoise = residualVariance(predict(dropout, train[0], false), yTrain);
        out.put("MC dropout", score(moments(mcIn, mcNoise), yIn, moments(mcGap, mcNoise), yGap));

        out.put("last-layer Laplace", lastLayerLaplace(first, train[0], yTrain, testIn[0], yIn, testGap[0], yGap));
        return out;
    }

    /** Bayesian linear regression on the last hidden layer of a trained MLP (exact Laplace for MSE). */
    private static Scores lastLayerLaplace(Model net, double[][] x, double[] y, double[][] xIn, double[] yIn,
            double[][] xGap, double[] yGap) {
        double[][] phi = hidden(net, x);
        double noise = residualVariance(predict(net, x, false), y);
        int p = phi[0].length + 1;
        double prior = 1.0;
        double[][] precision = new double[p][p];
        double[] b = new double[p];
        for (int i = 0; i < phi.length; i++) {
            double[] f = withBias(phi[i]);
            for (int a = 0; a < p; a++) {
                for (int c = 0; c < p; c++) {
                    precision[a][c] += f[a] * f[c] / noise;
                }
                b[a] += f[a] * y[i] / noise;
            }
        }
        for (int a = 0; a < p; a++) {
            precision[a][a] += prior;
        }
        double[][] chol = MlMatrices.cholesky(precision);
        double[] w = MlMatrices.choleskySolve(chol, b);
        return score(blr(hidden(net, xIn), w, chol, noise), yIn, blr(hidden(net, xGap), w, chol, noise), yGap);
    }

    private static double[][] blr(double[][] phi, double[] w, double[][] chol, double noise) {
        double[] mean = new double[phi.length];
        double[] var = new double[phi.length];
        for (int i = 0; i < phi.length; i++) {
            double[] f = withBias(phi[i]);
            mean[i] = MlMatrices.dot(f, w);
            double[] v = MlMatrices.forwardSolve(chol, f);
            var[i] = MlMatrices.dot(v, v) + noise;
        }
        return new double[][] {mean, var};
    }

    private static double[] withBias(double[] f) {
        double[] g = java.util.Arrays.copyOf(f, f.length + 1);
        g[f.length] = 1;
        return g;
    }

    /** The activations entering the last layer of a sequential MLP. */
    private static double[][] hidden(Model net, double[][] x) {
        double[][][] captured = new double[1][][];
        int last = net.graph().size() - 1;
        net.forward(Var.of(Tensor.of(MlMatrices.toFloat(x))), false, (step, output) -> {
            if (step.index() == last - 1) {
                float[] v = output.value().toArray();
                int w = output.shape().dim(1);
                double[][] h = new double[x.length][w];
                for (int i = 0; i < x.length; i++) {
                    for (int t = 0; t < w; t++) {
                        h[i][t] = v[i * w + t];
                    }
                }
                captured[0] = h;
            }
            return output;
        });
        return captured[0];
    }

    /**
     * Sparse variational GPs with growing inducing sets on 5,000 points against an exact GP on a
     * subset.
     *
     * @param inducing the inducing-point counts to try
     * @param seed the seed
     * @return "exact GP (n=1000)" and "SVGP M=..." to test NLL (inside range and gap together)
     */
    public static Map<String, Double> sparseScaling(int[] inducing, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double noise = 0.1;
        double[][][] train = draw(5000, false, noise, r);
        double[][][] testIn = draw(300, false, noise, r);
        double[][][] testGap = draw(100, true, noise, r);
        double[][] xt = concat(testIn[0], testGap[0]);
        double[] yt = column(concat(testIn[1], testGap[1]));
        double[] y = column(train[1]);
        Map<String, Double> out = new LinkedHashMap<>();
        // hyperparameters from a small subset by marginal likelihood
        double[][] xs = java.util.Arrays.copyOf(train[0], 200);
        double[] ys = java.util.Arrays.copyOf(y, 200);
        GaussianProcess small = GaussianProcess.optimize(xs, ys, StationaryKernel.rbf(1, 1), 0.05, 60);
        GaussianProcess exact = GaussianProcess.fit(java.util.Arrays.copyOf(train[0], 1000),
                java.util.Arrays.copyOf(y, 1000), small.kernel(), small.noise());
        GaussianProcess.Prediction pe = exact.predict(xt);
        out.put("exact GP (n=1000)", MlMetrics.gaussianNll(pe.mean(), pe.variance(), yt));
        for (int m : inducing) {
            double[][] z = KMeans.fit(train[0], m, KMeans.Init.PLUS_PLUS, 1, 50, seed + m).centres();
            SparseVariationalGp sv = SparseVariationalGp.fit(train[0], y, z, small.kernel(), small.noise(), 256, 150,
                    1.0, seed);
            GaussianProcess.Prediction p = sv.predict(xt);
            out.put("SVGP M=" + m, MlMetrics.gaussianNll(p.mean(), p.variance(), yt));
        }
        return out;
    }

    private static double[][] concat(double[][] a, double[][] b) {
        double[][] c = java.util.Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, c, a.length, b.length);
        return c;
    }

    /**
     * Deep kernel learning on one-step forecasts of a seasonal series, against the same network
     * with a point output and a GP on the raw windows. Windows and targets are differenced
     * against the last observed value.
     *
     * @param seed the seed
     * @return method to {test NLL, test RMSE, 95% coverage}
     */
    public static Map<String, double[]> deepKernel(long seed) {
        int lookback = 24;
        float[] series = SeasonalSeries.generate(900, 24, 0.002, 1.0, 0.1, seed);
        double[][][] train = windows(series, 0, 600, lookback);
        double[][][] test = windows(series, 600, 900, lookback);
        double[] y = column(train[1]);
        double[] yt = column(test[1]);
        Map<String, double[]> out = new LinkedHashMap<>();

        Model alone = Model.sequential().input(lookback).dense(32, Activation.TANH).dense(4).dense(1).seed(seed)
                .build();
        fit(alone, train[0], train[1], 600, 0.01f);
        double[] pm = predict(alone, test[0], false);
        double var = residualVariance(predict(alone, train[0], false), y);
        double[] pv = new double[pm.length];
        java.util.Arrays.fill(pv, var);
        out.put("MLP alone", metrics(pm, pv, yt));

        GaussianProcess raw = GaussianProcess.optimize(java.util.Arrays.copyOf(train[0], 250),
                java.util.Arrays.copyOf(y, 250), StationaryKernel.rbf(3, 1), 0.05, 60);
        GaussianProcess.Prediction pr = raw.predict(test[0]);
        out.put("GP on windows", metrics(pr.mean(), pr.variance(), yt));

        Model features = Model.sequential().input(lookback).dense(32, Activation.TANH).dense(4).seed(seed).build();
        DeepKernelLearning dkl = DeepKernelLearning.train(features, train[0], y, StationaryKernel.rbf(1, 1), 0.05,
                Optimizer.adam(0.005f), 300, 64, 400, seed);
        GaussianProcess.Prediction pd = dkl.predict(test[0]);
        out.put("DKL", metrics(pd.mean(), pd.variance(), yt));
        return out;
    }

    private static double[] metrics(double[] mean, double[] var, double[] y) {
        double se = 0;
        for (int i = 0; i < y.length; i++) {
            se += (mean[i] - y[i]) * (mean[i] - y[i]);
        }
        return new double[] {MlMetrics.gaussianNll(mean, var, y), Math.sqrt(se / y.length),
            MlMetrics.coverage(mean, var, y, 1.96)};
    }

    private static double[][][] windows(float[] series, int from, int to, int lookback) {
        SeasonalSeries.Windows w = SeasonalSeries.windows(series, from, to, lookback, 1, 1);
        int n = w.size();
        float[] xs = w.inputs().toArray();
        float[] ys = w.targets().toArray();
        double[][] x = new double[n][lookback];
        double[][] y = new double[n][1];
        for (int i = 0; i < n; i++) {
            double last = xs[i * lookback + lookback - 1];
            for (int t = 0; t < lookback; t++) {
                x[i][t] = xs[i * lookback + t] - last;
            }
            y[i][0] = ys[i] - last;
        }
        return new double[][][] {x, y};
    }
}
