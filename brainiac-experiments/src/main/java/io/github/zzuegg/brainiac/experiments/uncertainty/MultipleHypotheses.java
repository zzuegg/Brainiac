package io.github.zzuegg.brainiac.experiments.uncertainty;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.boost.NaturalGradientBoosting;
import io.github.zzuegg.brainiac.experimental.ensemble.MultipleChoiceLearning;
import io.github.zzuegg.brainiac.experimental.ood.FeatureClassifier;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Multiple-choice learning on ambiguous regression targets (wave 11 item 11.4.15): WTA,
 * relaxed WTA, evolving WTA and resilient (scored) WTA heads against a plain MSE regressor, an
 * ensemble of independent regressors, quantile-regression heads, a mixture density network and
 * NGBoost Gaussians, on three tasks with several valid answers per input:
 *
 * <ul>
 *   <li>{@link Task#ARM}: inverse kinematics of a two-link arm (link lengths 1 and 0.7): from the
 *       end-effector position, the two joint angles; elbow-up and elbow-down both reach it;</li>
 *   <li>{@link Task#SINE}: the inverse of {@code x = sin y} for {@code y ∈ [-π, 2π]}, three
 *       branches per {@code x};</li>
 *   <li>{@link Task#TURN}: the next position of an object at a position with a unit heading that
 *       turns left or right by 90 degrees (a feature-level stand-in for the moving shapes of
 *       {@code ShapeScenes}).</li>
 * </ul>
 * Metrics: the oracle (best-of-K) squared error against the sampled target, the coverage of the
 * true solution set (the share of valid answers within 0.25 of some hypothesis), and the NLL of
 * the methods that give a density (MDN, NGBoost; the WTA heads and ensembles with a Gaussian
 * kernel of a width fitted on validation data, weighted by the scoring heads for resilient MCL).
 */
public final class MultipleHypotheses {

    /** A task. */
    public enum Task {
        /** two-link arm inverse kinematics */
        ARM,
        /** the inverse of the sine */
        SINE,
        /** left or right turn */
        TURN
    }

    /**
     * Rows of a task, with every valid answer of each input.
     *
     * @param x inputs {@code [n][din]}
     * @param y the sampled answer {@code [n][dout]}
     * @param solutions every valid answer per input
     */
    public record Data(double[][] x, double[][] y, double[][][] solutions) {
    }

    /**
     * One method.
     *
     * @param method the method
     * @param oracle the best-of-K squared error
     * @param coverage the share of valid answers covered within 0.25
     * @param nll the test NLL (NaN if the method gives no density)
     */
    public record Row(String method, double oracle, double coverage, double nll) {
    }

    private static final double L1 = 1;
    private static final double L2 = 0.7;
    private static final int[] HIDDEN = {64, 64};

    private MultipleHypotheses() {
    }

    /**
     * Draws rows of a task.
     *
     * @param task the task
     * @param n rows
     * @param seed the seed
     * @return the rows
     */
    public static Data data(Task task, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][];
        double[][] y = new double[n][];
        double[][][] sol = new double[n][][];
        for (int i = 0; i < n; i++) {
            switch (task) {
                case ARM -> {
                    double t2 = (0.4 + 2.0 * r.nextDouble()) * (r.nextBoolean() ? 1 : -1);
                    double t1 = -1 + 2 * r.nextDouble();
                    x[i] = new double[] {L1 * Math.cos(t1) + L2 * Math.cos(t1 + t2),
                        L1 * Math.sin(t1) + L2 * Math.sin(t1 + t2)};
                    y[i] = new double[] {t1, t2};
                    double other1 = t1 + 2 * Math.atan2(L2 * Math.sin(t2), L1 + L2 * Math.cos(t2));
                    sol[i] = new double[][] {{t1, t2}, {other1, -t2}};
                }
                case SINE -> {
                    double v = -Math.PI + 3 * Math.PI * r.nextDouble();
                    double s = Math.sin(v);
                    x[i] = new double[] {s};
                    y[i] = new double[] {v};
                    List<double[]> all = new ArrayList<>();
                    double a = Math.asin(s);
                    for (int k = -1; k <= 1; k++) {
                        for (double c : new double[] {a + 2 * Math.PI * k, Math.PI - a + 2 * Math.PI * k}) {
                            if (c >= -Math.PI - 1e-9 && c <= 2 * Math.PI + 1e-9) {
                                all.add(new double[] {c});
                            }
                        }
                    }
                    sol[i] = all.toArray(double[][]::new);
                }
                case TURN -> {
                    double px = -1 + 2 * r.nextDouble();
                    double py = -1 + 2 * r.nextDouble();
                    double h = 2 * Math.PI * r.nextDouble();
                    double vx = Math.cos(h);
                    double vy = Math.sin(h);
                    x[i] = new double[] {px, py, vx, vy};
                    double[] left = {px - vy, py + vx};
                    double[] right = {px + vy, py - vx};
                    y[i] = r.nextBoolean() ? left : right;
                    sol[i] = new double[][] {left, right};
                }
            }
        }
        return new Data(x, y, sol);
    }

    /**
     * Every method on one task.
     *
     * @param task the task
     * @param heads the hypotheses {@code K}
     * @param steps training steps per network
     * @param seed the seed
     * @return one row per method
     */
    public static List<Row> run(Task task, int heads, int steps, long seed) {
        Data train = data(task, 1000, seed);
        Data val = data(task, 400, seed + 1);
        Data test = data(task, 600, seed + 2);
        int din = train.x()[0].length;
        int dout = train.y()[0].length;
        List<Row> rows = new ArrayList<>();
        // plain MSE regressor
        FeatureClassifier mse = FeatureClassifier.mlp(din, HIDDEN, dout, seed);
        fit(mse.parameters(), train, steps, seed, (b) -> mse.logits(Var.of(b[0]), true).sub(b[1]).square().mean());
        rows.add(hypothesisRow("mse", single(mse, test.x(), dout), single(mse, val.x(), dout), null, null, val, test));
        // ensemble of independent regressors
        List<FeatureClassifier> ens = new ArrayList<>();
        for (int m = 0; m < heads; m++) {
            FeatureClassifier f = FeatureClassifier.mlp(din, HIDDEN, dout, seed + 10 + m);
            fit(f.parameters(), train, steps, seed + 10 + m, (b) -> f.logits(Var.of(b[0]), true).sub(b[1]).square()
                    .mean());
            ens.add(f);
        }
        rows.add(hypothesisRow("ensemble", stack(ens, test.x(), dout), stack(ens, val.x(), dout), null, null, val,
                test));
        // quantile heads, per output dimension
        FeatureClassifier qr = FeatureClassifier.mlp(din, HIDDEN, heads * dout, seed + 20);
        float[] tau = new float[heads * dout];
        for (int k = 0; k < heads; k++) {
            for (int j = 0; j < dout; j++) {
                tau[k * dout + j] = (k + 0.5f) / heads;
            }
        }
        Tensor tauT = Tensor.of(Shape.of(1, heads * dout), tau);
        fit(qr.parameters(), train, steps, seed + 20, (b) -> {
            int n = b[0].shape().dim(0);
            Var z = qr.logits(Var.of(b[0]), true);
            Tensor t = b[1].reshape(n, 1, dout).expand(Shape.of(n, heads, dout)).reshape(n, heads * dout);
            Var u = z.neg().add(t);
            return u.mul(tauT).maximum(u.mul(tauT.sub(1f))).mean();
        });
        rows.add(hypothesisRow("quantile", reshape(qr.logits(tensor(test.x())), heads, dout),
                reshape(qr.logits(tensor(val.x())), heads, dout), null, null, val, test));
        // mixture density network
        FeatureClassifier mdn = FeatureClassifier.mlp(din, HIDDEN, heads * (2 * dout + 1), seed + 30);
        fit(mdn.parameters(), train, steps, seed + 30, (b) -> mdnNll(mdn.logits(Var.of(b[0]), true), b[1], heads,
                dout));
        rows.add(mdnRow(mdn, heads, dout, test));
        // NGBoost Gaussians, one per output dimension
        rows.add(ngboostRow(train, test, seed));
        // multiple-choice learning
        for (MultipleChoiceLearning.Mode mode : MultipleChoiceLearning.Mode.values()) {
            for (boolean scored : mode == MultipleChoiceLearning.Mode.WTA ? new boolean[] {false, true}
                    : new boolean[] {false}) {
                MultipleChoiceLearning.Network net = new MultipleChoiceLearning.Network(din, HIDDEN, heads, dout,
                        seed + 40);
                int[] step = {0};
                fit(net.parameters(), train, steps, seed + 40, (b) -> {
                    int top = mode == MultipleChoiceLearning.Mode.EVOLVING_WTA
                            ? Math.max(1, heads - (heads * step[0]++) / Math.max(1, steps / 2)) : 1;
                    return net.loss(b[0], b[1], mode, 0.05f, top, scored ? 1f : 0f);
                });
                String name = (scored ? "resilient " : "") + mode.name().toLowerCase(java.util.Locale.ROOT)
                        .replace('_', ' ');
                rows.add(hypothesisRow(name, net.hypotheses(tensor(test.x())), net.hypotheses(tensor(val.x())),
                        scored ? net.scores(tensor(test.x())) : null, scored ? net.scores(tensor(val.x())) : null, val,
                        test));
            }
        }
        return rows;
    }

    private static void fit(List<Var> params, Data d, int steps, long seed, Function<Tensor[], Var> loss) {
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        int din = d.x()[0].length;
        int dout = d.y()[0].length;
        int b = 128;
        for (int s = 0; s < steps; s++) {
            float[] bx = new float[b * din];
            float[] by = new float[b * dout];
            for (int i = 0; i < b; i++) {
                int row = r.nextInt(d.x().length);
                for (int j = 0; j < din; j++) {
                    bx[i * din + j] = (float) d.x()[row][j];
                }
                for (int j = 0; j < dout; j++) {
                    by[i * dout + j] = (float) d.y()[row][j];
                }
            }
            loss.apply(new Tensor[] {Tensor.of(Shape.of(b, din), bx), Tensor.of(Shape.of(b, dout), by)}).backward();
            opt.step(params);
        }
    }

    private static Var mdnNll(Var out, Tensor y, int k, int d) {
        int n = out.shape().dim(0);
        Var mu = out.slice(1, 0, k * d).reshape(n, k, d);
        Var logSigma = out.slice(1, k * d, 2 * k * d).reshape(n, k, d).clamp(-5f, 3f);
        Var logits = out.slice(1, 2 * k * d, k * (2 * d + 1));
        Var z = Var.of(y.reshape(n, 1, d)).sub(mu).div(logSigma.exp());
        Var comp = z.square().mul(-0.5f).sub(logSigma).sub((float) (0.5 * Math.log(2 * Math.PI))).sum(2, false);
        return logits.logSoftmax(1).add(comp).logsumexp(1, false).neg().mean();
    }

    private static Row mdnRow(FeatureClassifier mdn, int k, int d, Data test) {
        Tensor out = mdn.logits(tensor(test.x()));
        Var nll = Var.noGrad(() -> mdnNll(Var.of(out), tensor(test.y()), k, d));
        double[][][] hyp = reshape(out.slice(1, 0, k * d), k, d);
        return new Row("mdn", MultipleChoiceLearning.oracleError(hyp, test.y()), coverage(hyp, test), nll.value()
                .item());
    }

    private static Row ngboostRow(Data train, Data test, long seed) {
        int dout = train.y()[0].length;
        int n = test.x().length;
        double[][][] hyp = new double[n][1][dout];
        double nll = 0;
        for (int j = 0; j < dout; j++) {
            double[] y = new double[train.y().length];
            for (int i = 0; i < y.length; i++) {
                y[i] = train.y()[i][j];
            }
            NaturalGradientBoosting g = NaturalGradientBoosting.fit(train.x(), y, NaturalGradientBoosting.Family.NORMAL,
                    NaturalGradientBoosting.Score.LOG, new NaturalGradientBoosting.Settings(80, 0.08, 3, 10, true, 1,
                    seed));
            for (int i = 0; i < n; i++) {
                NaturalGradientBoosting.Distribution p = g.predict(test.x()[i]);
                hyp[i][0][j] = p.loc();
                double s = Math.exp(p.logScale());
                double z = (test.y()[i][j] - p.loc()) / s;
                nll += (0.5 * z * z + Math.log(s) + 0.5 * Math.log(2 * Math.PI)) / n;
            }
        }
        return new Row("ngboost", MultipleChoiceLearning.oracleError(hyp, test.y()), coverage(hyp, test), nll);
    }

    private static Row hypothesisRow(String name, double[][][] hyp, double[][][] valHyp, double[][] w,
            double[][] valW, Data val, Data test) {
        double[][] vw = valW == null ? uniform(valHyp) : valW;
        double best = Double.POSITIVE_INFINITY;
        double sigma = 0.1;
        for (double s : new double[] {0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0}) {
            double v = MultipleChoiceLearning.kernelNll(valHyp, vw, val.y(), s);
            if (v < best) {
                best = v;
                sigma = s;
            }
        }
        double nll = MultipleChoiceLearning.kernelNll(hyp, w == null ? uniform(hyp) : w, test.y(), sigma);
        return new Row(name, MultipleChoiceLearning.oracleError(hyp, test.y()), coverage(hyp, test), nll);
    }

    private static double[][] uniform(double[][][] hyp) {
        double[][] w = new double[hyp.length][hyp[0].length];
        for (double[] row : w) {
            java.util.Arrays.fill(row, 1.0 / row.length);
        }
        return w;
    }

    /**
     * The share of valid answers within 0.25 of some hypothesis.
     *
     * @param hyp {@code [n][K][d]}
     * @param data the rows with their solution sets
     * @return the coverage
     */
    public static double coverage(double[][][] hyp, Data data) {
        int hit = 0;
        int total = 0;
        for (int i = 0; i < hyp.length; i++) {
            for (double[] s : data.solutions()[i]) {
                total++;
                for (double[] h : hyp[i]) {
                    double e = 0;
                    for (int j = 0; j < s.length; j++) {
                        e += (h[j] - s[j]) * (h[j] - s[j]);
                    }
                    if (e <= 0.25 * 0.25) {
                        hit++;
                        break;
                    }
                }
            }
        }
        return hit / (double) total;
    }

    private static double[][][] single(FeatureClassifier f, double[][] x, int d) {
        return reshape(f.logits(tensor(x)), 1, d);
    }

    private static double[][][] stack(List<FeatureClassifier> fs, double[][] x, int d) {
        int n = x.length;
        double[][][] out = new double[n][fs.size()][];
        for (int m = 0; m < fs.size(); m++) {
            double[][][] h = single(fs.get(m), x, d);
            for (int i = 0; i < n; i++) {
                out[i][m] = h[i][0];
            }
        }
        return out;
    }

    private static double[][][] reshape(Tensor t, int k, int d) {
        int n = t.shape().dim(0);
        float[] v = t.toArray();
        int stride = v.length / n;
        double[][][] out = new double[n][k][d];
        for (int i = 0; i < n; i++) {
            for (int a = 0; a < k; a++) {
                for (int j = 0; j < d; j++) {
                    out[i][a][j] = v[i * stride + a * d + j];
                }
            }
        }
        return out;
    }

    private static Tensor tensor(double[][] rows) {
        int d = rows[0].length;
        float[] v = new float[rows.length * d];
        for (int i = 0; i < rows.length; i++) {
            for (int j = 0; j < d; j++) {
                v[i * d + j] = (float) rows[i][j];
            }
        }
        return Tensor.of(Shape.of(rows.length, d), v);
    }

    /**
     * Prints the tables.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        for (Task t : Task.values()) {
            run(t, 5, 1500, 1).forEach(r -> System.out.println(t + " " + r));
        }
    }
}
