package io.github.zzuegg.brainiac.experiments.parallel;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.parallel.AsyncSgd;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.DataParallel;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * 9.6.7: asynchronous SGD with stale gradients. Hogwild! on sparse logistic regression over
 * {@link SyntheticRecsys#clicks} (real lock-free threads against a global lock and one thread),
 * and simulated staleness {@code tau = 0..32} on a dense MLP over the digits task, with plain
 * ASGD, delay-compensated DC-ASGD and staleness-aware learning rates, against synchronous core
 * {@link DataParallel} and DiLoCo-style local SGD with periodic averaging at the same number of
 * gradient evaluations.
 */
public final class AsyncTraining {

    private AsyncTraining() {
    }

    /**
     * Sparse logistic regression on one-hot click features plus the crossed pair of the first two
     * fields (the clicks' signal is mostly that interaction, which a purely linear model cannot
     * see), one weight per active feature plus a bias.
     */
    static final class SparseLogistic implements AsyncSgd.Problem {
        final int[][] active;
        final float[] label;
        final int[][] testActive;
        final float[] testLabel;
        final int dim;

        SparseLogistic(int rows, int fields, int categories, long seed) {
            SyntheticRecsys.Table t = SyntheticRecsys.clicks(rows + rows / 4, fields, categories, seed);
            float[] x = t.features().toArray();
            float[] y = t.labels().toArray();
            int width = fields * categories;
            this.dim = width + categories * categories + 1;
            int n = rows + rows / 4;
            int[][] act = new int[n][];
            for (int i = 0; i < n; i++) {
                int[] a = new int[fields + 1];
                int k = 0;
                for (int j = 0; j < width; j++) {
                    if (x[i * width + j] != 0) {
                        a[k++] = j;
                    }
                }
                a[k++] = width + a[0] * categories + (a[1] - categories);
                act[i] = java.util.Arrays.copyOf(a, k);
            }
            this.active = java.util.Arrays.copyOf(act, rows);
            this.label = java.util.Arrays.copyOf(y, rows);
            this.testActive = java.util.Arrays.copyOfRange(act, rows, n);
            this.testLabel = java.util.Arrays.copyOfRange(y, rows, n);
        }

        @Override
        public int dimension() {
            return dim;
        }

        @Override
        public int samples() {
            return label.length;
        }

        private double logit(float[] w, int[] a) {
            double z = w[dim - 1];
            for (int j : a) {
                z += w[j];
            }
            return z;
        }

        @Override
        public double gradient(float[] w, int[] batch, AsyncSgd.Sparse out) {
            double loss = 0;
            for (int i : batch) {
                double p = 1 / (1 + Math.exp(-logit(w, active[i])));
                float g = (float) ((p - label[i]) / batch.length);
                for (int j : active[i]) {
                    out.add(j, g);
                }
                out.add(dim - 1, g);
                loss += label[i] > 0 ? -Math.log(p + 1e-12) : -Math.log(1 - p + 1e-12);
            }
            return loss / batch.length;
        }

        @Override
        public double evaluate(float[] w) {
            double loss = 0;
            for (int i = 0; i < testLabel.length; i++) {
                double p = 1 / (1 + Math.exp(-logit(w, testActive[i])));
                loss += testLabel[i] > 0 ? -Math.log(p + 1e-12) : -Math.log(1 - p + 1e-12);
            }
            return loss / testLabel.length;
        }
    }

    /** A one-hidden-layer ReLU MLP with softmax cross-entropy over a flat parameter array. */
    static final class DenseMlp implements AsyncSgd.Problem {
        final ToyTasks.Data train;
        final ToyTasks.Data test;
        final int in;
        final int hidden;
        final int out;

        DenseMlp(ToyTasks.Data train, ToyTasks.Data test, int hidden) {
            this.train = train;
            this.test = test;
            this.in = train.features();
            this.hidden = hidden;
            this.out = train.classes();
        }

        @Override
        public int dimension() {
            return in * hidden + hidden + hidden * out + out;
        }

        @Override
        public int samples() {
            return train.size();
        }

        float[] init(long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            float[] w = new float[dimension()];
            for (int i = 0; i < in * hidden; i++) {
                w[i] = (float) (r.nextGaussian() * Math.sqrt(2.0 / in));
            }
            int o = in * hidden + hidden;
            for (int i = 0; i < hidden * out; i++) {
                w[o + i] = (float) (r.nextGaussian() * Math.sqrt(1.0 / hidden));
            }
            return w;
        }

        /** Forward one sample; fills h and returns the softmax probabilities. */
        private double[] forward(float[] w, float[] x, double[] h) {
            int b1 = in * hidden;
            int w2 = b1 + hidden;
            int b2 = w2 + hidden * out;
            for (int j = 0; j < hidden; j++) {
                double z = w[b1 + j];
                for (int i = 0; i < in; i++) {
                    z += x[i] * w[i * hidden + j];
                }
                h[j] = Math.max(0, z);
            }
            double[] p = new double[out];
            double max = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < out; k++) {
                double z = w[b2 + k];
                for (int j = 0; j < hidden; j++) {
                    z += h[j] * w[w2 + j * out + k];
                }
                p[k] = z;
                max = Math.max(max, z);
            }
            double sum = 0;
            for (int k = 0; k < out; k++) {
                p[k] = Math.exp(p[k] - max);
                sum += p[k];
            }
            for (int k = 0; k < out; k++) {
                p[k] /= sum;
            }
            return p;
        }

        @Override
        public double gradient(float[] w, int[] batch, AsyncSgd.Sparse g) {
            int b1 = in * hidden;
            int w2 = b1 + hidden;
            int b2 = w2 + hidden * out;
            double[] gw = new double[dimension()];
            double loss = 0;
            double[] h = new double[hidden];
            for (int s : batch) {
                float[] x = train.x()[s];
                double[] p = forward(w, x, h);
                int y = train.y()[s];
                loss -= Math.log(p[y] + 1e-12);
                double[] dz = p.clone();
                dz[y] -= 1;
                double[] dh = new double[hidden];
                for (int k = 0; k < out; k++) {
                    gw[b2 + k] += dz[k];
                    for (int j = 0; j < hidden; j++) {
                        gw[w2 + j * out + k] += h[j] * dz[k];
                        dh[j] += w[w2 + j * out + k] * dz[k];
                    }
                }
                for (int j = 0; j < hidden; j++) {
                    if (h[j] <= 0) {
                        continue;
                    }
                    gw[b1 + j] += dh[j];
                    for (int i = 0; i < in; i++) {
                        if (x[i] != 0) {
                            gw[i * hidden + j] += x[i] * dh[j];
                        }
                    }
                }
            }
            for (int i = 0; i < gw.length; i++) {
                if (gw[i] != 0) {
                    g.add(i, (float) (gw[i] / batch.length));
                }
            }
            return loss / batch.length;
        }

        @Override
        public double evaluate(float[] w) {
            double loss = 0;
            double[] h = new double[hidden];
            for (int i = 0; i < test.size(); i++) {
                loss -= Math.log(forward(w, test.x()[i], h)[test.y()[i]] + 1e-12);
            }
            return loss / test.size();
        }

        double accuracy(float[] w) {
            double[] h = new double[hidden];
            int ok = 0;
            for (int i = 0; i < test.size(); i++) {
                double[] p = forward(w, test.x()[i], h);
                int arg = 0;
                for (int k = 1; k < out; k++) {
                    if (p[k] > p[arg]) {
                        arg = k;
                    }
                }
                ok += arg == test.y()[i] ? 1 : 0;
            }
            return ok / (double) test.size();
        }
    }

    /**
     * A Hogwild! result.
     *
     * @param method the configuration
     * @param threads worker threads
     * @param loss held-out log loss after the same total number of updates
     * @param seconds wall time
     * @param speedUp wall time of one thread over this one
     */
    public record HogwildRow(String method, int threads, double loss, double seconds, double speedUp) {
    }

    /**
     * Hogwild! on sparse logistic regression: the same total updates split over 1 to 8 lock-free
     * threads, and over 8 threads holding a global lock.
     *
     * @param totalUpdates updates over all threads
     * @param seed the seed
     * @return rows
     */
    public static List<HogwildRow> hogwild(int totalUpdates, long seed) {
        SparseLogistic p = new SparseLogistic(20_000, 8, 32, seed);
        float[] init = new float[p.dimension()];
        List<HogwildRow> out = new ArrayList<>();
        double one = 0;
        for (int t : new int[] {1, 2, 4, 8}) {
            AsyncSgd.Curve c = AsyncSgd.hogwild(p, init, t, totalUpdates / t, 4, 0.1f, false, seed);
            if (t == 1) {
                one = c.wallSeconds();
            }
            out.add(new HogwildRow("Hogwild! (lock-free)", t, c.finalLoss(), c.wallSeconds(), one / c.wallSeconds()));
        }
        AsyncSgd.Curve locked = AsyncSgd.hogwild(p, init, 8, totalUpdates / 8, 4, 0.1f, true, seed);
        out.add(new HogwildRow("global lock", 8, locked.finalLoss(), locked.wallSeconds(),
                one / locked.wallSeconds()));
        return out;
    }

    /**
     * A staleness result.
     *
     * @param method the method
     * @param delay {@code tau}
     * @param loss held-out loss at the end
     * @param accuracy held-out accuracy at the end
     */
    public record StaleRow(String method, int delay, double loss, double accuracy) {
    }

    /**
     * Simulated asynchronous SGD on the dense digits MLP for each delay (each gradient computed
     * {@code tau} updates before it is applied), against synchronous data parallelism and local
     * SGD with the same number of gradient evaluations.
     *
     * @param delays the staleness values
     * @param updates updates per run
     * @param seed the seed
     * @return rows
     */
    public static List<StaleRow> staleness(int[] delays, int updates, long seed) {
        ToyTasks.Data train = ToyTasks.digits(10, 60, 8, 0.05, seed);
        ToyTasks.Data test = ToyTasks.digits(10, 20, 8, 0.05, seed, seed + 1000);
        DenseMlp mlp = new DenseMlp(train, test, 32);
        float[] init = mlp.init(seed);
        float lr = 0.3f;
        int batch = 8;
        List<StaleRow> out = new ArrayList<>();
        Map<String, float[]> settings = new LinkedHashMap<>();
        settings.put("ASGD", new float[] {0, 0});
        settings.put("DC-ASGD-a (lambda 2)", new float[] {2, 0});
        settings.put("staleness-aware lr", new float[] {0, 1});
        for (int tau : delays) {
            for (Map.Entry<String, float[]> e : settings.entrySet()) {
                float[] w = finalParameters(mlp, init, updates, batch, lr, tau, e.getValue()[0],
                        e.getValue()[1] > 0, seed);
                out.add(new StaleRow(e.getKey(), tau, mlp.evaluate(w), mlp.accuracy(w)));
            }
        }
        // synchronous data parallelism: 8 workers per step, so 8x fewer steps of an 8x batch
        out.add(synchronous(mlp, train, init, updates / 8, batch * 8, lr, seed));
        // local SGD (DiLoCo-style with plain averaging): 8 workers, 8 local steps between averages
        float[] local = localSgd(mlp, init, updates, batch, lr, 8, 8, seed);
        out.add(new StaleRow("local SGD (8 workers, average every 8)", 0, mlp.evaluate(local), mlp.accuracy(local)));
        return out;
    }

    /** Runs {@link AsyncSgd#simulate} and returns its final parameters (captured at its last evaluation). */
    private static float[] finalParameters(AsyncSgd.Problem p, float[] init, int updates, int batch, float lr,
            int tau, float lambda, boolean aware, long seed) {
        float[][] box = new float[1][];
        AsyncSgd.Problem capture = new AsyncSgd.Problem() {
            @Override
            public int dimension() {
                return p.dimension();
            }

            @Override
            public int samples() {
                return p.samples();
            }

            @Override
            public double gradient(float[] w, int[] b, AsyncSgd.Sparse out) {
                return p.gradient(w, b, out);
            }

            @Override
            public double evaluate(float[] w) {
                box[0] = w.clone();
                return 0;
            }
        };
        AsyncSgd.simulate(capture, init, updates, batch, lr, t -> tau, lambda, aware, 0, seed);
        return box[0];
    }

    private static StaleRow synchronous(DenseMlp mlp, ToyTasks.Data train, float[] init, int steps, int batch,
            float lr, long seed) {
        int in = mlp.in;
        int h = mlp.hidden;
        int o = mlp.out;
        float[] w = init.clone();
        List<Var> params = List.of(Var.param(Tensor.of(Shape.of(in, h), java.util.Arrays.copyOfRange(w, 0, in * h))),
                Var.param(Tensor.of(Shape.of(h), java.util.Arrays.copyOfRange(w, in * h, in * h + h))),
                Var.param(Tensor.of(Shape.of(h, o), java.util.Arrays.copyOfRange(w, in * h + h, in * h + h + h * o))),
                Var.param(Tensor.of(Shape.of(o), java.util.Arrays.copyOfRange(w, in * h + h + h * o, w.length))));
        List<Device> ds = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            ds.add(Device.cpu());
        }
        SplittableRandom r = new SplittableRandom(seed);
        try (DataParallel dp = new DataParallel(ds, params, (ps, x, y) -> Functional.crossEntropy(
                Var.of(x).matmul(ps.get(0)).add(ps.get(1)).relu().matmul(ps.get(2)).add(ps.get(3)), y),
                Optimizer.sgd(lr))) {
            for (int s = 0; s < steps; s++) {
                float[][] xs = new float[batch][];
                float[] ys = new float[batch];
                for (int i = 0; i < batch; i++) {
                    int k = r.nextInt(train.size());
                    xs[i] = train.x()[k];
                    ys[i] = train.y()[k];
                }
                dp.step(Tensor.of(xs), Tensor.of(Shape.of(batch), ys));
            }
        }
        int at = 0;
        for (Var p : params) {
            float[] a = p.value().toArray();
            System.arraycopy(a, 0, w, at, a.length);
            at += a.length;
        }
        return new StaleRow("synchronous DataParallel (8 workers)", 0, mlp.evaluate(w), mlp.accuracy(w));
    }

    private static float[] localSgd(DenseMlp mlp, float[] init, int updates, int batch, float lr, int workers,
            int every, long seed) {
        float[] global = init.clone();
        float[][] local = new float[workers][];
        SplittableRandom r = new SplittableRandom(seed);
        AsyncSgd.Sparse g = new AsyncSgd.Sparse(256);
        int[] b = new int[batch];
        int rounds = updates / (workers * every);
        for (int round = 0; round < rounds; round++) {
            for (int k = 0; k < workers; k++) {
                local[k] = global.clone();
                for (int s = 0; s < every; s++) {
                    for (int i = 0; i < batch; i++) {
                        b[i] = r.nextInt(mlp.samples());
                    }
                    g.clear();
                    mlp.gradient(local[k], b, g);
                    for (int e = 0; e < g.size(); e++) {
                        local[k][g.index(e)] -= lr * g.value(e);
                    }
                }
            }
            for (int i = 0; i < global.length; i++) {
                float s = 0;
                for (int k = 0; k < workers; k++) {
                    s += local[k][i];
                }
                global[i] = s / workers;
            }
        }
        return global;
    }
}
