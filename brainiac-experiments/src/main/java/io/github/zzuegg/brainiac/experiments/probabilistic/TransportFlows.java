package io.github.zzuegg.brainiac.experiments.probabilistic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.EntropicTransport;
import io.github.zzuegg.brainiac.experimental.ml.McmcDiagnostics;
import io.github.zzuegg.brainiac.experimental.nn.MinibatchOptimalTransport;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Entropic optimal transport in generative modelling.
 *
 * <ul>
 *   <li>{@link #timing}: milliseconds per Sinkhorn iteration (log domain and kernel scaling)
 *       against one exact Hungarian solve, by problem size.</li>
 *   <li>{@link #flowPairing}: conditional flow matching from two moons to two concentric rings
 *       with a small velocity MLP, the noise-data pairing being independent, exact minibatch OT
 *       (Hungarian, n = 256) or a large Sinkhorn coupling (n = 1024, eps = 0.15, recomputed every six
 *       steps and sampled into pairs; Klein et al. 2025). Scores: sliced Wasserstein distance of one-step and four-step Euler
 *       samples to the target, and the straightness of the 20-step ODE path
 *       {@code E |(z1 - z0) - v(z_t, t)|^2}.</li>
 *   <li>{@link #generatorLoss}: a generator from Gaussian noise to the rings trained with the
 *       Sinkhorn divergence as its loss, against the sliced Wasserstein loss: final sliced
 *       Wasserstein distance.</li>
 * </ul>
 */
public final class TransportFlows {

    private TransportFlows() {
    }

    static double[][] moons(int n, SplittableRandom r) {
        double[][] x = new double[n][];
        for (int i = 0; i < n; i++) {
            double a = Math.PI * r.nextDouble();
            boolean upper = r.nextBoolean();
            double px = upper ? Math.cos(a) : 1 - Math.cos(a);
            double py = upper ? Math.sin(a) : 0.5 - Math.sin(a);
            x[i] = new double[] {1.5 * (px - 0.5) + 0.1 * r.nextGaussian(), 1.5 * (py - 0.25) + 0.1 * r.nextGaussian()};
        }
        return x;
    }

    static double[][] rings(int n, SplittableRandom r) {
        double[][] x = new double[n][];
        for (int i = 0; i < n; i++) {
            double a = 2 * Math.PI * r.nextDouble();
            double rad = (r.nextBoolean() ? 1 : 2) + 0.05 * r.nextGaussian();
            x[i] = new double[] {rad * Math.cos(a), rad * Math.sin(a)};
        }
        return x;
    }

    /**
     * Sinkhorn iteration cost against the Hungarian solver.
     *
     * @param sizes the problem sizes
     * @param seed the seed
     * @return {@code "n=..."} to {@code {log-domain ms/iteration, scaling ms/iteration, Hungarian ms}}
     */
    public static Map<String, double[]> timing(int[] sizes, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int n : sizes) {
            SplittableRandom r = new SplittableRandom(seed + n);
            double[][] x = moons(n, r);
            double[][] y = rings(n, r);
            double[][] c = EntropicTransport.squaredEuclidean(x, y);
            double[] w = EntropicTransport.uniform(n);
            long t = System.nanoTime();
            EntropicTransport.Plan p = EntropicTransport.sinkhorn(w, w, c, 0.1, 10, 0);
            double logMs = (System.nanoTime() - t) / 1e6 / p.iterations();
            t = System.nanoTime();
            EntropicTransport.sinkhornScaling(w, w, c, 0.5, 10);
            double scaleMs = (System.nanoTime() - t) / 1e6 / 10;
            double hungarian = Double.NaN;
            if (n <= 512) {
                t = System.nanoTime();
                MinibatchOptimalTransport.assign(toFloat(x), toFloat(y));
                hungarian = (System.nanoTime() - t) / 1e6;
            }
            out.put("n=" + n, new double[] {logMs, scaleMs, hungarian});
        }
        return out;
    }

    static float[][] toFloat(double[][] x) {
        float[][] f = new float[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                f[i][j] = (float) x[i][j];
            }
        }
        return f;
    }

    private static Tensor tensor(double[][] x) {
        float[] v = new float[x.length * 2];
        for (int i = 0; i < x.length; i++) {
            v[2 * i] = (float) x[i][0];
            v[2 * i + 1] = (float) x[i][1];
        }
        return Tensor.of(Shape.of(x.length, 2), v);
    }

    private static Tensor withTime(double[][] x, double[] t) {
        float[] v = new float[x.length * 3];
        for (int i = 0; i < x.length; i++) {
            v[3 * i] = (float) x[i][0];
            v[3 * i + 1] = (float) x[i][1];
            v[3 * i + 2] = (float) t[i];
        }
        return Tensor.of(Shape.of(x.length, 3), v);
    }

    private static double[][] velocity(Model net, double[][] x, double t) {
        double[] ts = new double[x.length];
        java.util.Arrays.fill(ts, t);
        double[] v = Var.noGrad(() -> net.forward(Var.of(withTime(x, ts)))).value().toDoubleArray();
        double[][] out = new double[x.length][2];
        for (int i = 0; i < x.length; i++) {
            out[i][0] = v[2 * i];
            out[i][1] = v[2 * i + 1];
        }
        return out;
    }

    private static double[][] euler(Model net, double[][] x0, int steps, double[] straightness) {
        double[][] z = new double[x0.length][];
        for (int i = 0; i < x0.length; i++) {
            z[i] = x0[i].clone();
        }
        double[][][] path = new double[steps][][];
        for (int s = 0; s < steps; s++) {
            double[][] v = velocity(net, z, (double) s / steps);
            path[s] = v;
            for (int i = 0; i < z.length; i++) {
                z[i][0] += v[i][0] / steps;
                z[i][1] += v[i][1] / steps;
            }
        }
        if (straightness != null) {
            double s = 0;
            for (int k = 0; k < steps; k++) {
                for (int i = 0; i < z.length; i++) {
                    double dx = z[i][0] - x0[i][0] - path[k][i][0];
                    double dy = z[i][1] - x0[i][1] - path[k][i][1];
                    s += (dx * dx + dy * dy) / (steps * z.length);
                }
            }
            straightness[0] = s;
        }
        return z;
    }

    /**
     * Flow matching with three pairings.
     *
     * @param steps training steps per pairing
     * @param seed the seed
     * @return pairing to {@code {1-step sliced W, 4-step sliced W, straightness, train seconds}}
     */
    public static Map<String, double[]> flowPairing(int steps, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        SplittableRandom er = new SplittableRandom(seed + 99);
        double[][] testSource = moons(1000, er);
        double[][] testTarget = rings(1000, er);
        for (String pairing : new String[] {"independent", "Hungarian n=256", "Sinkhorn n=1024"}) {
            SplittableRandom r = new SplittableRandom(seed);
            Model net = Model.sequential().input(3).dense(64, Activation.TANH).dense(64, Activation.TANH)
                    .dense(2, Activation.IDENTITY).seed(seed).build();
            Optimizer opt = Optimizer.adam(3e-3f);
            int batch = 256;
            double[][] poolSource = null;
            double[][] poolTarget = null;
            int[] partner = null;
            long start = System.nanoTime();
            for (int s = 0; s < steps; s++) {
                double[][] x0;
                double[][] x1;
                switch (pairing) {
                    case "Hungarian n=256" -> {
                        x0 = moons(batch, r);
                        float[][] paired = MinibatchOptimalTransport.pair(toFloat(x0), toFloat(rings(batch, r)));
                        x1 = new double[batch][];
                        for (int i = 0; i < batch; i++) {
                            x1[i] = new double[] {paired[i][0], paired[i][1]};
                        }
                    }
                    case "Sinkhorn n=1024" -> {
                        if (s % 6 == 0) {
                            poolSource = moons(1024, r);
                            poolTarget = rings(1024, r);
                            double[][] c = EntropicTransport.squaredEuclidean(poolSource, poolTarget);
                            double[] w = EntropicTransport.uniform(1024);
                            partner = EntropicTransport.pairing(EntropicTransport.sinkhornScaling(w, w, c, 0.15, 40),
                                    r);
                        }
                        x0 = new double[batch][];
                        x1 = new double[batch][];
                        for (int i = 0; i < batch; i++) {
                            int k = r.nextInt(1024);
                            x0[i] = poolSource[k];
                            x1[i] = poolTarget[partner[k]];
                        }
                    }
                    default -> {
                        x0 = moons(batch, r);
                        x1 = rings(batch, r);
                    }
                }
                double[] t = new double[batch];
                double[][] xt = new double[batch][2];
                float[] target = new float[batch * 2];
                for (int i = 0; i < batch; i++) {
                    t[i] = r.nextDouble();
                    for (int k = 0; k < 2; k++) {
                        xt[i][k] = (1 - t[i]) * x0[i][k] + t[i] * x1[i][k];
                        target[2 * i + k] = (float) (x1[i][k] - x0[i][k]);
                    }
                }
                net.forward(Var.of(withTime(xt, t)), true).sub(Tensor.of(Shape.of(batch, 2), target)).square().mean()
                        .backward();
                opt.step(net.parameters());
            }
            double secs = (System.nanoTime() - start) / 1e9;
            double[] straight = new double[1];
            double sw1 = McmcDiagnostics.slicedWasserstein(euler(net, testSource, 1, null), testTarget, 50, seed);
            double sw4 = McmcDiagnostics.slicedWasserstein(euler(net, testSource, 4, null), testTarget, 50, seed);
            euler(net, testSource, 20, straight);
            out.put(pairing, new double[] {sw1, sw4, straight[0], secs});
        }
        return out;
    }

    /**
     * A generator trained with the Sinkhorn divergence or the sliced Wasserstein loss.
     *
     * @param steps training steps
     * @param seed the seed
     * @return loss to {@code {final sliced W to the target}}
     */
    public static Map<String, double[]> generatorLoss(int steps, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        double[][] test = rings(1000, new SplittableRandom(seed + 5));
        for (String loss : new String[] {"Sinkhorn divergence", "sliced Wasserstein"}) {
            SplittableRandom r = new SplittableRandom(seed);
            Model gen = Model.sequential().input(2).dense(64, Activation.TANH).dense(64, Activation.TANH)
                    .dense(2, Activation.IDENTITY).seed(seed).build();
            Optimizer opt = Optimizer.adam(3e-3f);
            int batch = 256;
            for (int s = 0; s < steps; s++) {
                Var x = gen.forward(Var.of(Tensor.normal(Shape.of(batch, 2), 0, 1, r)), true);
                double[][] y = rings(batch, r);
                Var l;
                if (loss.equals("sliced Wasserstein")) {
                    Tensor dirs = Tensor.normal(Shape.of(2, 32), 0, 1, r);
                    Var px = x.matmul(Var.of(dirs)).sort(0, false);
                    Var py = Var.of(tensor(y)).matmul(Var.of(dirs)).sort(0, false);
                    l = px.sub(py).square().mean();
                } else {
                    l = EntropicTransport.divergenceLoss(x, y, 1.0, 30);
                }
                l.backward();
                opt.step(gen.parameters());
            }
            double[] v = Var.noGrad(() -> gen.forward(Var.of(Tensor.normal(Shape.of(1000, 2), 0, 1,
                    new SplittableRandom(seed + 7))))).value().toDoubleArray();
            double[][] g = new double[1000][];
            for (int i = 0; i < 1000; i++) {
                g[i] = new double[] {v[2 * i], v[2 * i + 1]};
            }
            out.put(loss, new double[] {McmcDiagnostics.slicedWasserstein(g, test, 50, seed)});
        }
        return out;
    }
}
