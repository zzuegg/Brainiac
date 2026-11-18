package io.github.zzuegg.brainiac.experiments.probabilistic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.InfiniteWidthKernels;
import io.github.zzuegg.brainiac.experimental.ml.KernelFunction;
import io.github.zzuegg.brainiac.experimental.ml.KernelGaussianProcess;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.experimental.ml.RandomFourierFeatures;
import io.github.zzuegg.brainiac.experimental.train.RidgeRegression;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Finite networks against their infinite-width kernels.
 *
 * <ul>
 *   <li>{@link #ntkAgreement}: one-hidden-layer ReLU networks in the NTK parameterisation,
 *       widths 16 to 2048, trained by full-batch gradient descent on the 1-D regression-gap
 *       data. The analytic NTK predicts the trained function exactly in the limit
 *       ({@code f_t = f_0 + Theta_*X Theta^-1 (I - (I - lr Theta)^t)(y - f_0)}, with the
 *       network's own {@code f_0}); the RMS gap between the two should shrink with width, and so
 *       should the relative change of the empirical NTK during training.</li>
 *   <li>{@link #spirals}: two-arm spirals with angle noise 0.35, 120 training points. Kernel regression with the
 *       NNGP and NTK (two hidden layers), the RBF kernel, random Fourier features with ridge
 *       regression (500 features), and trained networks of widths 16-1024: test accuracy.</li>
 *   <li>{@link #gapNll}: the GP posterior with the NNGP kernel against the RBF GP on the
 *       regression gap: Gaussian NLL in and outside the gap.</li>
 * </ul>
 */
public final class InfiniteWidth {

    private static final double SW = 2;
    private static final double SB = 0.1;

    private InfiniteWidth() {
    }

    /** A one-hidden-layer ReLU network in the NTK parameterisation. */
    private static final class NtkNet {
        final Var w;
        final Var b;
        final Var v;
        final Var b2;
        final int width;
        final int inputs;

        NtkNet(int inputs, int width, SplittableRandom r) {
            this.inputs = inputs;
            this.width = width;
            w = Var.param(Tensor.normal(Shape.of(inputs, width), 0, 1, r));
            b = Var.param(Tensor.normal(Shape.of(1, width), 0, 1, r));
            v = Var.param(Tensor.normal(Shape.of(width, 1), 0, 1, r));
            b2 = Var.param(Tensor.normal(Shape.of(1, 1), 0, 1, r));
        }

        Var forward(Var x) {
            Var h = x.matmul(w).mul((float) Math.sqrt(SW / inputs)).add(b.mul((float) Math.sqrt(SB))).relu();
            return h.matmul(v).mul((float) Math.sqrt(SW / width)).add(b2.mul((float) Math.sqrt(SB)));
        }

        List<Var> parameters() {
            return List.of(w, b, v, b2);
        }

        double[] predict(double[][] x) {
            return Var.noGrad(() -> forward(Var.of(tensor(x)))).value().toDoubleArray();
        }

        void train(double[][] x, double[] y, double lr, int steps) {
            Tensor tx = tensor(x);
            float[] yv = new float[y.length];
            for (int i = 0; i < y.length; i++) {
                yv[i] = (float) y[i];
            }
            Tensor ty = Tensor.of(Shape.of(y.length, 1), yv);
            for (int s = 0; s < steps; s++) {
                forward(Var.of(tx)).sub(ty).square().sum().mul(0.5f).backward();
                for (Var p : parameters()) {
                    p.assign(p.value().sub(p.grad().mul((float) lr)));
                    p.zeroGrad();
                }
            }
        }
    }

    private static Tensor tensor(double[][] x) {
        int d = x[0].length;
        float[] v = new float[x.length * d];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < d; j++) {
                v[i * d + j] = (float) x[i][j];
            }
        }
        return Tensor.of(Shape.of(x.length, d), v);
    }

    /** The regression-gap data with inputs embedded as {@code (x / 3, 1)}. */
    private static double[][][] gapData(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[40][];
        double[] y = new double[40];
        for (int i = 0; i < 40; i++) {
            double v = 1 + 2 * r.nextDouble();
            double xi = r.nextBoolean() ? v : -v;
            x[i] = new double[] {xi / 3, 1};
            y[i] = Math.sin(1.5 * xi) + 0.1 * r.nextGaussian();
        }
        double[][] grid = new double[61][];
        double[] truth = new double[61];
        for (int i = 0; i < 61; i++) {
            double g = -3 + 6.0 * i / 60;
            grid[i] = new double[] {g / 3, 1};
            truth[i] = Math.sin(1.5 * g);
        }
        return new double[][][] {x, {y}, grid, {truth}};
    }

    /**
     * Agreement of trained networks with the analytic NTK predictor, by width.
     *
     * @param widths the hidden widths
     * @param steps gradient-descent steps
     * @param seed the seed
     * @return {@code "width=..."} to {@code {RMS gap to the NTK predictor, relative empirical NTK change}}
     */
    public static Map<String, double[]> ntkAgreement(int[] widths, int steps, long seed) {
        double[][][] data = gapData(seed);
        double[][] x = data[0];
        double[] y = data[1][0];
        double[][] grid = data[2];
        KernelFunction ntk = InfiniteWidthKernels.ntk(1, SW, SB);
        double[][] theta = ntk.gram(x);
        MlMatrices.Eigen eig = MlMatrices.eigh(theta);
        double lmax = 0;
        for (double l : eig.values()) {
            lmax = Math.max(lmax, l);
        }
        double lr = 1 / lmax;
        int n = x.length;
        // M = Q diag((1 - (1 - lr l)^t) / l) Q^T
        double[][] m = new double[n][n];
        for (int k = 0; k < n; k++) {
            double l = Math.max(eig.values()[k], 1e-12);
            double f = (1 - Math.pow(1 - lr * l, steps)) / l;
            double[] q = eig.vector(k);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    m[i][j] += f * q[i] * q[j];
                }
            }
        }
        double[][] cross = ntk.matrix(grid, x);
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int width : widths) {
            NtkNet net = new NtkNet(2, width, new SplittableRandom(seed + width));
            double[] f0x = net.predict(x);
            double[] f0g = net.predict(grid);
            double[][] k0 = InfiniteWidthKernels.empiricalNtk(net.parameters(), net::forward, x);
            net.train(x, y, lr, steps);
            double[][] k1 = InfiniteWidthKernels.empiricalNtk(net.parameters(), net::forward, x);
            double[] fg = net.predict(grid);
            double[] resid = new double[n];
            for (int i = 0; i < n; i++) {
                resid[i] = y[i] - f0x[i];
            }
            double[] mr = new double[n];
            for (int i = 0; i < n; i++) {
                mr[i] = MlMatrices.dot(m[i], resid);
            }
            double gap = 0;
            for (int g = 0; g < grid.length; g++) {
                double lin = f0g[g] + MlMatrices.dot(cross[g], mr);
                gap += (lin - fg[g]) * (lin - fg[g]) / grid.length;
            }
            double dk = 0;
            double k0n = 0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    dk += (k1[i][j] - k0[i][j]) * (k1[i][j] - k0[i][j]);
                    k0n += k0[i][j] * k0[i][j];
                }
            }
            out.put("width=" + width, new double[] {Math.sqrt(gap), Math.sqrt(dk / k0n)});
        }
        return out;
    }

    /**
     * Spirals classification by kernels and by trained networks.
     *
     * @param seed the seed
     * @return method to {@code {test accuracy}}
     */
    public static Map<String, double[]> spirals(long seed) {
        ToyTasks.Data train = ToyTasks.spiral(60, 2, 0.35, seed);
        ToyTasks.Data test = ToyTasks.spiral(200, 2, 0.35, seed + 1);
        double[][] x = embed(train.x());
        double[][] xt = embed(test.x());
        double[] y = new double[x.length];
        for (int i = 0; i < y.length; i++) {
            y[i] = train.y()[i] == 1 ? 1 : -1;
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("NNGP (2 hidden)", new double[] {accuracy(KernelGaussianProcess.fitZeroMean(x, y,
                InfiniteWidthKernels.nngp(2, SW, SB), 1e-3).mean(xt), test.y())});
        out.put("NTK (2 hidden)", new double[] {accuracy(KernelGaussianProcess.fitZeroMean(x, y,
                InfiniteWidthKernels.ntk(2, SW, SB), 1e-3).mean(xt), test.y())});
        out.put("RBF kernel", new double[] {accuracy(KernelGaussianProcess.fitZeroMean(x, y, KernelFunction.rbf(20),
                1e-3).mean(xt), test.y())});
        RandomFourierFeatures rff = new RandomFourierFeatures(3, 500, 20, new SplittableRandom(seed));
        double[][] yy = new double[y.length][];
        for (int i = 0; i < y.length; i++) {
            yy[i] = new double[] {y[i]};
        }
        double[][] w = RidgeRegression.fit(rff.transform(x), yy, 1e-3);
        double[][] ft = rff.transform(xt);
        double[] pred = new double[ft.length];
        for (int i = 0; i < ft.length; i++) {
            pred[i] = RidgeRegression.predict(w, ft[i])[0];
        }
        out.put("RFF + ridge (500)", new double[] {accuracy(pred, test.y())});
        for (int width : new int[] {16, 128, 1024}) {
            NtkNet net = new NtkNet(3, width, new SplittableRandom(seed + width));
            net.train(x, y, 0.004, 800);
            out.put("trained MLP width=" + width, new double[] {accuracy(net.predict(xt), test.y())});
        }
        return out;
    }

    private static double[][] embed(float[][] x) {
        double[][] out = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            out[i] = new double[] {x[i][0], x[i][1], 1};
        }
        return out;
    }

    private static double accuracy(double[] scores, int[] labels) {
        int right = 0;
        for (int i = 0; i < scores.length; i++) {
            right += (scores[i] > 0 ? 1 : 0) == labels[i] ? 1 : 0;
        }
        return right / (double) scores.length;
    }

    /**
     * NNGP and RBF GP posteriors on the regression gap.
     *
     * @param seed the seed
     * @return kernel to {@code {NLL outside the gap, NLL in the gap}}
     */
    public static Map<String, double[]> gapNll(long seed) {
        double[][][] data = gapData(seed);
        Map<String, double[]> out = new LinkedHashMap<>();
        KernelFunction[] kernels = {InfiniteWidthKernels.nngp(2, SW, SB), InfiniteWidthKernels.ntk(2, SW, SB),
            KernelFunction.rbf(4)};
        String[] names = {"NNGP GP", "NTK GP", "RBF GP"};
        for (int k = 0; k < kernels.length; k++) {
            KernelGaussianProcess gp = KernelGaussianProcess.fit(data[0], data[1][0], kernels[k], 0.01);
            double[] mean = gp.mean(data[2]);
            double[] var = gp.latentVariance(data[2]);
            int nIn = 0;
            int nGap = 0;
            for (double[] g : data[2]) {
                if (Math.abs(g[0] * 3) < 1) {
                    nGap++;
                } else {
                    nIn++;
                }
            }
            double[][] split = new double[6][];
            split[0] = new double[nIn];
            split[1] = new double[nIn];
            split[2] = new double[nIn];
            split[3] = new double[nGap];
            split[4] = new double[nGap];
            split[5] = new double[nGap];
            int a = 0;
            int b = 0;
            for (int i = 0; i < mean.length; i++) {
                if (Math.abs(data[2][i][0] * 3) < 1) {
                    split[3][b] = mean[i];
                    split[4][b] = var[i] + 0.01;
                    split[5][b++] = data[3][0][i];
                } else {
                    split[0][a] = mean[i];
                    split[1][a] = var[i] + 0.01;
                    split[2][a++] = data[3][0][i];
                }
            }
            out.put(names[k], new double[] {MlMetrics.gaussianNll(split[0], split[1], split[2]),
                MlMetrics.gaussianNll(split[3], split[4], split[5])});
        }
        return out;
    }
}
