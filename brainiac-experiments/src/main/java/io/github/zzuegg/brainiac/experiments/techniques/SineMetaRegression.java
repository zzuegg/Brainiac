package io.github.zzuegg.brainiac.experiments.techniques;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.NeuralProcess;
import io.github.zzuegg.brainiac.experimental.nn.SnailLearner;
import io.github.zzuegg.brainiac.experimental.train.Maml;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Few-shot sine regression, the MAML paper's benchmark, drawn exactly as in
 * {@link MetaLearningExperiment}: amplitude in {@code [0.5, 5]}, phase in {@code [0, pi]},
 * 10 support points and 50 query points in {@code [-5, 5]}, 20 test waves. It compares
 * amortized meta-learners with a gradient-based one and with the exact Bayesian answer:
 *
 * <ul>
 *   <li>{@link Learner#MAML}: second-order MAML, 5 SGD steps; its predictive spread is the
 *       support residual (homoscedastic), the best it can do without an uncertainty head.</li>
 *   <li>{@link Learner#SNAIL}: the support set as a sequence, one forward pass per query.</li>
 *   <li>{@link Learner#CNP} and {@link Learner#ANP}: neural processes with Gaussian outputs.</li>
 *   <li>{@link Learner#GP}: a Gaussian process with an RBF kernel whose amplitude and length
 *       scale are fitted per task by the marginal likelihood; the gold standard on smooth
 *       functions.</li>
 * </ul>
 */
public final class SineMetaRegression {

    /** The learners. */
    public enum Learner {
        /** Second-order MAML. */
        MAML,
        /** SNAIL over the support sequence. */
        SNAIL,
        /** Conditional neural process. */
        CNP,
        /** Attentive neural process. */
        ANP,
        /** Exact GP with per-task fitted hyperparameters. */
        GP
    }

    /**
     * Test-set scores.
     *
     * @param mse mean squared error of the predictive mean
     * @param nll mean Gaussian negative log-likelihood per point
     * @param coverage the share of query points inside the central 95% interval
     */
    public record Result(double mse, double nll, double coverage) {
    }

    private record Wave(Tensor xs, Tensor ys, Tensor xq, Tensor yq) {

        static Wave sample(SplittableRandom r, int support, int query) {
            double amp = 0.5 + 4.5 * r.nextDouble();
            double phase = Math.PI * r.nextDouble();
            Tensor[] s = points(r, support, amp, phase);
            Tensor[] q = points(r, query, amp, phase);
            return new Wave(s[0], s[1], q[0], q[1]);
        }

        static Tensor[] points(SplittableRandom r, int n, double amp, double phase) {
            float[] x = new float[n];
            float[] y = new float[n];
            for (int i = 0; i < n; i++) {
                x[i] = (float) (-5 + 10 * r.nextDouble());
                y[i] = (float) (amp * Math.sin(x[i] + phase));
            }
            return new Tensor[] {Tensor.of(Shape.of(n, 1), x), Tensor.of(Shape.of(n, 1), y)};
        }
    }

    private interface Predictor {
        /** {@code {mean[], sigma[]}} at the query points. */
        double[][] predict(Wave w);
    }

    private SineMetaRegression() {
    }

    /**
     * Meta-trains a learner and scores it on 20 new waves.
     *
     * @param learner the learner
     * @param iterations meta-training iterations (ignored by the GP)
     * @param seed the seed
     * @return the scores
     */
    public static Result run(Learner learner, int iterations, long seed) {
        Predictor p = switch (learner) {
            case MAML -> maml(iterations, seed);
            case SNAIL -> snail(iterations, seed);
            case CNP -> process(NeuralProcess.Kind.CONDITIONAL, iterations, seed);
            case ANP -> process(NeuralProcess.Kind.ATTENTIVE, iterations, seed);
            case GP -> SineMetaRegression::gaussianProcess;
        };
        SplittableRandom test = new SplittableRandom(seed + 1);
        double se = 0;
        double nll = 0;
        int inside = 0;
        int n = 0;
        for (int k = 0; k < 20; k++) {
            Wave w = Wave.sample(test, 10, 50);
            double[][] pred = p.predict(w);
            float[] y = w.yq().toArray();
            for (int i = 0; i < y.length; i++) {
                double d = y[i] - pred[0][i];
                double s = pred[1][i];
                se += d * d;
                nll += Math.log(s) + 0.5 * d * d / (s * s) + 0.5 * Math.log(2 * Math.PI);
                inside += Math.abs(d) <= 1.96 * s ? 1 : 0;
                n++;
            }
        }
        return new Result(se / n, nll / n, inside / (double) n);
    }

    private record SineTask(Model model, Wave w) implements Maml.Task {

        @Override
        public Var supportLoss() {
            return model.forward(Var.of(w.xs())).sub(w.ys()).square().mean();
        }

        @Override
        public Var queryLoss() {
            return model.forward(Var.of(w.xq())).sub(w.yq()).square().mean();
        }
    }

    private static Predictor maml(int iterations, long seed) {
        Model model = Model.sequential().input(1).dense(40, Activation.RELU, Init.XAVIER)
                .dense(40, Activation.RELU, Init.XAVIER).dense(1, Activation.IDENTITY).seed(seed).build();
        Maml maml = new Maml(model.parameters(), Optimizer.adam(3e-3f), 0.01f, 1, true);
        SplittableRandom r = new SplittableRandom(seed);
        for (int it = 0; it < iterations; it++) {
            List<SineTask> tasks = new ArrayList<>();
            for (int k = 0; k < 5; k++) {
                tasks.add(new SineTask(model, Wave.sample(r, 10, 10)));
            }
            maml.metaStep(tasks);
        }
        return w -> {
            List<Tensor> start = maml.snapshot();
            SineTask t = new SineTask(model, w);
            maml.adapt(t, 5);
            double sigma = Math.max(0.05, Math.sqrt(t.supportLoss().value().item()));
            float[] mean = model.forward(Var.of(w.xq())).value().toArray();
            maml.restore(start);
            return spread(mean, sigma);
        };
    }

    private static double[][] spread(float[] mean, double sigma) {
        double[][] out = new double[2][mean.length];
        for (int i = 0; i < mean.length; i++) {
            out[0][i] = mean[i];
            out[1][i] = sigma;
        }
        return out;
    }

    /** {@code [queries, support + 1, 3]}: support steps {@code (x/5, y/5, 1)}, then {@code (x/5, 0, 0)}. */
    private static Tensor sequences(Tensor xs, Tensor ys, Tensor xq) {
        float[] sx = xs.toArray();
        float[] sy = ys.toArray();
        float[] q = xq.toArray();
        int n = sx.length;
        float[] out = new float[q.length * (n + 1) * 3];
        for (int b = 0; b < q.length; b++) {
            for (int i = 0; i < n; i++) {
                int o = (b * (n + 1) + i) * 3;
                out[o] = sx[i] / 5;
                out[o + 1] = sy[i] / 5;
                out[o + 2] = 1;
            }
            out[(b * (n + 1) + n) * 3] = q[b] / 5;
        }
        return Tensor.of(Shape.of(q.length, n + 1, 3), out);
    }

    private static Predictor snail(int iterations, long seed) {
        SnailLearner snail = new SnailLearner(3, 1, 11, 16, 16, 16, 2, new SplittableRandom(seed));
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int it = 0; it < iterations; it++) {
            List<Tensor> seqs = new ArrayList<>();
            List<Tensor> ys = new ArrayList<>();
            for (int k = 0; k < 4; k++) {
                Wave w = Wave.sample(r, 10, 8);
                seqs.add(sequences(w.xs(), w.ys(), w.xq()));
                ys.add(w.yq());
            }
            snail.last(Var.of(Tensor.concat(0, seqs.toArray(new Tensor[0])))).mul(5f)
                    .sub(Tensor.concat(0, ys.toArray(new Tensor[0]))).square().mean().backward();
            opt.step(snail.parameters());
        }
        return w -> {
            float[] mean = snail.last(Var.of(sequences(w.xs(), w.ys(), w.xq()))).mul(5f).value().toArray();
            // SNAIL has no uncertainty head: use its leave-one-out support error as the spread
            double se = 0;
            float[] sx = w.xs().toArray();
            float[] sy = w.ys().toArray();
            for (int i = 0; i < sx.length; i++) {
                float[] rx = new float[sx.length - 1];
                float[] ry = new float[sx.length - 1];
                for (int j = 0, o = 0; j < sx.length; j++) {
                    if (j != i) {
                        rx[o] = sx[j];
                        ry[o++] = sy[j];
                    }
                }
                Tensor seq = sequences(Tensor.of(Shape.of(rx.length, 1), rx), Tensor.of(Shape.of(ry.length, 1), ry),
                        Tensor.of(Shape.of(1, 1), sx[i]));
                double d = sy[i] - 5 * snail.last(Var.of(seq)).value().item();
                se += d * d;
            }
            return spread(mean, Math.max(0.05, Math.sqrt(se / sx.length)));
        };
    }

    private static Predictor process(NeuralProcess.Kind kind, int iterations, long seed) {
        NeuralProcess np = new NeuralProcess(1, 1, 64, kind, new SplittableRandom(seed));
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        int tasks = 16;
        for (int it = 0; it < iterations; it++) {
            Tensor[] cx = new Tensor[tasks];
            Tensor[] cy = new Tensor[tasks];
            Tensor[] tx = new Tensor[tasks];
            Tensor[] ty = new Tensor[tasks];
            for (int k = 0; k < tasks; k++) {
                Wave w = Wave.sample(r, 10, 20);
                // targets include the context, as in the papers
                cx[k] = scale(w.xs()).reshape(1, 10, 1);
                cy[k] = w.ys().reshape(1, 10, 1);
                tx[k] = scale(Tensor.concat(0, w.xs(), w.xq())).reshape(1, 30, 1);
                ty[k] = Tensor.concat(0, w.ys(), w.yq()).reshape(1, 30, 1);
            }
            np.nll(Tensor.concat(0, cx), Tensor.concat(0, cy), Tensor.concat(0, tx), Tensor.concat(0, ty)).backward();
            opt.step(np.parameters());
        }
        return w -> {
            Var[] p = np.predict(scale(w.xs()), w.ys(), scale(w.xq()));
            float[] m = p[0].value().toArray();
            float[] s = p[1].value().toArray();
            double[][] out = new double[2][m.length];
            for (int i = 0; i < m.length; i++) {
                out[0][i] = m[i];
                out[1][i] = s[i];
            }
            return out;
        };
    }

    private static Tensor scale(Tensor x) {
        return x.div(5f);
    }

    /** An exact GP; amplitude and length scale by grid search on the marginal likelihood. */
    private static double[][] gaussianProcess(Wave w) {
        double[] xs = toDouble(w.xs());
        double[] ys = toDouble(w.ys());
        double[] xq = toDouble(w.xq());
        double noise = 1e-2;
        double bestLml = Double.NEGATIVE_INFINITY;
        double[] best = null;
        for (double amp : new double[] {0.5, 1, 2, 3, 4, 5}) {
            for (double len : new double[] {0.5, 0.75, 1, 1.25, 1.5, 2}) {
                Tensor l = Linalg.cholesky(kernel(xs, xs, amp, len, noise));
                Tensor alpha = solve(l, column(ys));
                double lml = 0;
                double[] a = alpha.toDoubleArray();
                double[] ld = l.toDoubleArray();
                for (int i = 0; i < ys.length; i++) {
                    lml += -0.5 * ys[i] * a[i] - Math.log(ld[i * ys.length + i]);
                }
                if (lml > bestLml) {
                    bestLml = lml;
                    best = new double[] {amp, len};
                }
            }
        }
        Tensor l = Linalg.cholesky(kernel(xs, xs, best[0], best[1], noise));
        Tensor kq = kernel(xs, xq, best[0], best[1], 0);
        double[] mean = kq.transpose().matmul(solve(l, column(ys))).toDoubleArray();
        double[] v = Linalg.solveTriangular(l, kq, true).toDoubleArray();
        double[][] out = new double[2][xq.length];
        for (int j = 0; j < xq.length; j++) {
            double var = best[0] * best[0] + noise;
            for (int i = 0; i < xs.length; i++) {
                var -= v[i * xq.length + j] * v[i * xq.length + j];
            }
            out[0][j] = mean[j];
            out[1][j] = Math.sqrt(Math.max(var, 1e-6));
        }
        return out;
    }

    private static Tensor solve(Tensor l, Tensor b) {
        return Linalg.solveTriangular(l.transpose(), Linalg.solveTriangular(l, b, true), false);
    }

    private static Tensor kernel(double[] a, double[] b, double amp, double len, double noise) {
        double[] k = new double[a.length * b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                double d = a[i] - b[j];
                k[i * b.length + j] = amp * amp * Math.exp(-d * d / (2 * len * len)) + (i == j && noise > 0 ? noise : 0);
            }
        }
        return Tensor.of(Shape.of(a.length, b.length), k, DType.FLOAT64, Device.cpu());
    }

    private static Tensor column(double[] v) {
        return Tensor.of(Shape.of(v.length, 1), v, DType.FLOAT64, Device.cpu());
    }

    private static double[] toDouble(Tensor t) {
        float[] f = t.toArray();
        double[] d = new double[f.length];
        for (int i = 0; i < f.length; i++) {
            d[i] = f[i];
        }
        return d;
    }
}
