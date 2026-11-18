package io.github.zzuegg.brainiac.experiments.symbolic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.LeastSquares;
import io.github.zzuegg.brainiac.experimental.boost.ExplainableBoosting;
import io.github.zzuegg.brainiac.experimental.data.ChaoticSeries;
import io.github.zzuegg.brainiac.experimental.logic.Anfis;
import io.github.zzuegg.brainiac.experimental.nn.KanLayer;
import io.github.zzuegg.brainiac.rl.CartPole;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.GradientBoostedTrees;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Wave 13 neuro-fuzzy experiments (item 13.2.15): ANFIS and fuzzy inference systems
 * ({@link io.github.zzuegg.brainiac.experimental.logic.FuzzyInference}) against an MLP of equal
 * parameter count, a KAN, Gaussian-RBF ridge regression and (on the 2-D task) explainable
 * boosting, on Mackey–Glass one-step-ahead prediction (Jang's benchmark: four lags six steps apart
 * predict six steps ahead), the 2-D sinc surface, and a cart-pole control surface imitated from a
 * PD-style law and then flown in closed loop. A last table shows how grid partitions explode with
 * the input dimension where subtractive clustering keeps the rule count small.
 */
public final class FuzzyInference {

    private FuzzyInference() {
    }

    /**
     * One table row.
     *
     * @param task the task
     * @param method the model
     * @param rmse test RMSE (or another headline metric, named in the task)
     * @param parameters adjustable parameters
     * @param rules rules (fuzzy systems) or training epochs (networks)
     * @param millis training time
     */
    public record Row(String task, String method, double rmse, double parameters, double rules, double millis) {

        @Override
        public String toString() {
            return String.format("%s/%s %.4f params=%.0f rules|epochs=%.0f %.0fms", task, method, rmse, parameters,
                rules, millis);
        }
    }

    private static double ms(long t0) {
        return (System.nanoTime() - t0) / 1e6;
    }

    private static double rmse(ToDoubleFunction<double[]> f, double[][] x, double[] y) {
        double s = 0;
        for (int i = 0; i < x.length; i++) {
            double e = f.applyAsDouble(x[i]) - y[i];
            s += e * e;
        }
        return Math.sqrt(s / x.length);
    }

    private static double[] bounds(double[][] x, boolean max) {
        double[] b = new double[x[0].length];
        java.util.Arrays.fill(b, max ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY);
        for (double[] row : x) {
            for (int i = 0; i < b.length; i++) {
                b[i] = max ? Math.max(b[i], row[i]) : Math.min(b[i], row[i]);
            }
        }
        return b;
    }

    /** Scales inputs to {@code [-1, 1]} with the training ranges. */
    private static double[][] scale(double[][] x, double[] lo, double[] hi) {
        double[][] out = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            out[i] = new double[x[i].length];
            for (int j = 0; j < x[i].length; j++) {
                out[i][j] = 2 * (x[i][j] - lo[j]) / Math.max(1e-12, hi[j] - lo[j]) - 1;
            }
        }
        return out;
    }

    private static Tensor matrix(double[][] rows) {
        int n = rows.length;
        int d = rows[0].length;
        float[] v = new float[n * d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                v[i * d + j] = (float) rows[i][j];
            }
        }
        return Tensor.of(Shape.of(n, d), v);
    }

    private static Tensor vector(double[] y) {
        float[] v = new float[y.length];
        for (int i = 0; i < y.length; i++) {
            v[i] = (float) y[i];
        }
        return Tensor.of(Shape.of(y.length, 1), v);
    }

    /** A fitted baseline: its predictor and its parameter count. */
    private record Model(ToDoubleFunction<double[]> predict, int parameters) {
    }

    /** A one-hidden-layer tanh MLP trained full batch with Adam on scaled inputs. */
    private static Model mlp(double[][] x, double[] y, int hidden, int epochs, long seed) {
        double[] lo = bounds(x, false);
        double[] hi = bounds(x, true);
        int d = x[0].length;
        SplittableRandom r = new SplittableRandom(seed);
        Var w1 = Var.param(Tensor.normal(Shape.of(d, hidden), 0, (float) Math.sqrt(1.0 / d), r));
        Var b1 = Var.param(Tensor.zeros(Shape.of(hidden)));
        Var w2 = Var.param(Tensor.normal(Shape.of(hidden, 1), 0, (float) Math.sqrt(1.0 / hidden), r));
        Var b2 = Var.param(Tensor.zeros(Shape.of(1)));
        List<Var> ps = List.of(w1, b1, w2, b2);
        Var xs = Var.of(matrix(scale(x, lo, hi)));
        Tensor t = vector(y);
        Optimizer opt = Optimizer.adam(0.01f);
        for (int e = 0; e < epochs; e++) {
            xs.matmul(w1).add(b1).tanh().matmul(w2).add(b2).sub(t).square().mean().backward();
            opt.step(ps);
        }
        return new Model(v -> Var.noGrad(() -> Var.of(matrix(scale(new double[][] {v}, lo, hi))).matmul(w1).add(b1)
            .tanh().matmul(w2).add(b2)).value().toArray()[0], d * hidden + 2 * hidden + 1);
    }

    /** A two-layer KAN {@code [d, hidden, 1]} with cubic splines on scaled inputs. */
    private static Model kan(double[][] x, double[] y, int hidden, int intervals, int epochs, long seed) {
        double[] lo = bounds(x, false);
        double[] hi = bounds(x, true);
        int d = x[0].length;
        KanLayer l1 = KanLayer.of(d, hidden, intervals, -1.05f, 1.05f, seed);
        KanLayer l2 = KanLayer.of(hidden, 1, intervals, -1.05f, 1.05f, seed + 1);
        List<Var> ps = new ArrayList<>(l1.parameters());
        ps.addAll(l2.parameters());
        Var xs = Var.of(matrix(scale(x, lo, hi)));
        Tensor t = vector(y);
        Optimizer opt = Optimizer.adam(0.01f);
        for (int e = 0; e < epochs; e++) {
            l2.forward(l1.forward(xs).tanh()).sub(t).square().mean().backward();
            opt.step(ps);
        }
        int count = 0;
        for (Var p : ps) {
            count += (int) p.shape().size();
        }
        return new Model(v -> Var.noGrad(() -> l2.forward(l1.forward(Var.of(matrix(scale(new double[][] {v}, lo, hi))))
            .tanh())).value().toArray()[0], count);
    }

    /** Gaussian-RBF features at {@code k} training points, ridge regression via augmented least squares. */
    private static Model rbfRidge(double[][] x, double[] y, int k, double ridge, long seed) {
        double[] lo = bounds(x, false);
        double[] hi = bounds(x, true);
        double[][] xs = scale(x, lo, hi);
        SplittableRandom r = new SplittableRandom(seed);
        double[][] c = new double[k][];
        for (int j = 0; j < k; j++) {
            c[j] = xs[r.nextInt(xs.length)];
        }
        double width = 2.0 / Math.pow(k, 1.0 / x[0].length);
        int n = x.length;
        double[][] design = new double[n + k + 1][k + 1];
        double[] target = new double[n + k + 1];
        for (int i = 0; i < n; i++) {
            design[i] = rbf(xs[i], c, width);
            target[i] = y[i];
        }
        for (int j = 0; j < k; j++) {
            design[n + j][j] = Math.sqrt(ridge);
        }
        double[] coef = LeastSquares.ols(design, target, false).coefficients();
        return new Model(v -> {
            double[] f = rbf(scale(new double[][] {v}, lo, hi)[0], c, width);
            double s = 0;
            for (int j = 0; j < f.length; j++) {
                s += f[j] * coef[j];
            }
            return s;
        }, k + 1 + k * x[0].length);
    }

    private static double[] rbf(double[] v, double[][] c, double width) {
        double[] f = new double[c.length + 1];
        for (int j = 0; j < c.length; j++) {
            double d2 = 0;
            for (int i = 0; i < v.length; i++) {
                d2 += (v[i] - c[j][i]) * (v[i] - c[j][i]);
            }
            f[j] = Math.exp(-d2 / (2 * width * width));
        }
        f[c.length] = 1;
        return f;
    }

    private static io.github.zzuegg.brainiac.experimental.logic.FuzzyInference gridSystem(double[][] x, int perInput,
            io.github.zzuegg.brainiac.experimental.logic.FuzzyInference.Type type,
            io.github.zzuegg.brainiac.experimental.logic.FuzzyInference.Membership mf) {
        return io.github.zzuegg.brainiac.experimental.logic.FuzzyInference.grid(type, mf, bounds(x, false),
            bounds(x, true), perInput);
    }

    /** Mackey–Glass lags: inputs {@code x(t - 3*step) .. x(t)} predict {@code x(t + horizon)}. */
    private static double[][][] lagged(double[] s, int inputs, int step, int horizon) {
        int start = (inputs - 1) * step;
        int n = s.length - start - horizon;
        double[][] x = new double[n][inputs];
        double[][] y = new double[n][1];
        for (int t = 0; t < n; t++) {
            for (int i = 0; i < inputs; i++) {
                x[t][i] = s[start + t - (inputs - 1 - i) * step];
            }
            y[t][0] = s[start + t + horizon];
        }
        return new double[][][] {x, y};
    }

    private static double[] column(double[][] y) {
        double[] out = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            out[i] = y[i][0];
        }
        return out;
    }

    /**
     * Mackey–Glass ({@code tau = 17}): 500 training and 500 test points.
     *
     * @param anfisEpochs hybrid epochs
     * @param seed the seed
     * @return the rows
     */
    public static List<Row> mackeyGlass(int anfisEpochs, long seed) {
        double[] s = ChaoticSeries.mackeyGlass(1100, 17, 1, 300);
        double[][][] data = lagged(s, 4, 6, 6);
        double[][] xTr = java.util.Arrays.copyOfRange(data[0], 0, 500);
        double[] yTr = column(java.util.Arrays.copyOfRange(data[1], 0, 500));
        double[][] xTe = java.util.Arrays.copyOfRange(data[0], 500, 1000);
        double[] yTe = column(java.util.Arrays.copyOfRange(data[1], 500, 1000));
        return regressionTable("mackey-glass", xTr, yTr, xTe, yTe, 2, anfisEpochs, false, seed);
    }

    /**
     * The 2-D sinc surface {@code sin(x) sin(y) / (x y)} on {@code [-10, 10]^2}: Jang's 121 grid
     * points for training, 400 random points for testing.
     *
     * @param anfisEpochs hybrid epochs
     * @param seed the seed
     * @return the rows
     */
    public static List<Row> sinc(int anfisEpochs, long seed) {
        double[][] xTr = new double[121][];
        double[] yTr = new double[121];
        for (int i = 0; i < 11; i++) {
            for (int j = 0; j < 11; j++) {
                double a = -10 + 2 * i;
                double b = -10 + 2 * j;
                xTr[i * 11 + j] = new double[] {a, b};
                yTr[i * 11 + j] = sinc2(a, b);
            }
        }
        SplittableRandom r = new SplittableRandom(seed);
        double[][] xTe = new double[400][];
        double[] yTe = new double[400];
        for (int i = 0; i < 400; i++) {
            xTe[i] = new double[] {r.nextDouble(-10, 10), r.nextDouble(-10, 10)};
            yTe[i] = sinc2(xTe[i][0], xTe[i][1]);
        }
        return regressionTable("sinc-2d", xTr, yTr, xTe, yTe, 4, anfisEpochs, true, seed);
    }

    private static double sinc2(double a, double b) {
        return (a == 0 ? 1 : Math.sin(a) / a) * (b == 0 ? 1 : Math.sin(b) / b);
    }

    private static List<Row> regressionTable(String task, double[][] xTr, double[] yTr, double[][] xTe, double[] yTe,
            int perInput, int anfisEpochs, boolean withEbm, long seed) {
        List<Row> rows = new ArrayList<>();
        var gauss = io.github.zzuegg.brainiac.experimental.logic.FuzzyInference.Membership.GAUSSIAN;
        var ts = io.github.zzuegg.brainiac.experimental.logic.FuzzyInference.Type.TAKAGI_SUGENO;
        // ANFIS hybrid
        long t0 = System.nanoTime();
        var fis = gridSystem(xTr, perInput, ts, gauss);
        Anfis anfis = new Anfis(fis);
        anfis.train(xTr, yTr, anfisEpochs, 10, 0.01, 1e-6, true);
        rows.add(new Row(task, "ANFIS hybrid (" + anfisEpochs + " epochs)", rmse(fis::predict, xTe, yTe),
            fis.parameters(), fis.rules(), ms(t0)));
        int params = fis.parameters();
        // ANFIS by gradient descent only, 10x the epochs
        t0 = System.nanoTime();
        var gd = gridSystem(xTr, perInput, ts, gauss);
        new Anfis(gd).train(xTr, yTr, anfisEpochs, 100, 0.01, 1e-6, false);
        rows.add(new Row(task, "ANFIS gradient only (" + 100 * anfisEpochs + " steps)", rmse(gd::predict, xTe, yTe),
            gd.parameters(), gd.rules(), ms(t0)));
        // the untrained grid system with least-squares consequents only
        t0 = System.nanoTime();
        var lsOnly = gridSystem(xTr, perInput, ts, gauss);
        lsOnly.fitConsequents(xTr, yTr, 1e-6);
        rows.add(new Row(task, "TS grid, LS consequents only", rmse(lsOnly::predict, xTe, yTe), lsOnly.parameters(),
            lsOnly.rules(), ms(t0)));
        // Mamdani with the same grid
        t0 = System.nanoTime();
        var mamdani = gridSystem(xTr, perInput,
            io.github.zzuegg.brainiac.experimental.logic.FuzzyInference.Type.MAMDANI, gauss);
        mamdani.fitConsequents(xTr, yTr, 0);
        rows.add(new Row(task, "Mamdani grid (centroid)", rmse(mamdani::predict, xTe, yTe), mamdani.parameters(),
            mamdani.rules(), ms(t0)));
        // MLP with the same parameter count at two budgets
        int d = xTr[0].length;
        int hidden = Math.max(2, (params - 1) / (d + 2));
        for (int epochs : new int[] {10 * anfisEpochs, 100 * anfisEpochs}) {
            t0 = System.nanoTime();
            Model m = mlp(xTr, yTr, hidden, epochs, seed);
            rows.add(new Row(task, "MLP h=" + hidden, rmse(m.predict(), xTe, yTe), m.parameters(), epochs, ms(t0)));
        }
        t0 = System.nanoTime();
        Model k = kan(xTr, yTr, 4, 5, 30 * anfisEpochs, seed);
        rows.add(new Row(task, "KAN [d,4,1] G=5", rmse(k.predict(), xTe, yTe), k.parameters(), 30 * anfisEpochs,
            ms(t0)));
        t0 = System.nanoTime();
        Model rb = rbfRidge(xTr, yTr, fis.rules() * (d + 1), 1e-4, seed);
        rows.add(new Row(task, "Gaussian RBF ridge", rmse(rb.predict(), xTe, yTe), rb.parameters(), 0, ms(t0)));
        if (withEbm) {
            t0 = System.nanoTime();
            ExplainableBoosting ebm = ExplainableBoosting.fit(xTr, yTr, GradientBoostedTrees.Objective.REGRESSION.loss(),
                ExplainableBoosting.Settings.defaults());
            rows.add(new Row(task, "ExplainableBoosting", rmse(ebm::predict, xTe, yTe), 0, 0, ms(t0)));
        }
        return rows;
    }

    // ---- cart-pole control surface -----------------------------------------------------------------

    /** The PD-style law: push towards the side the pole leans, with cart terms to stay on track. */
    private static double law(float[] o) {
        return Math.tanh(0.5 * (o[0] + 1.5 * o[1] + 18 * o[2] + 3 * o[3]));
    }

    private static double episodes(ToDoubleFunction<double[]> controller, int episodes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double total = 0;
        for (int e = 0; e < episodes; e++) {
            CartPole env = new CartPole(500);
            float[] o = env.reset(r);
            int steps = 0;
            while (true) {
                double u = controller.applyAsDouble(new double[] {o[0], o[1], o[2], o[3]});
                Environment.Step st = env.step(new float[] {u > 0 ? 1 : 0});
                steps++;
                o = st.observation();
                if (st.terminated() || st.truncated()) {
                    break;
                }
            }
            total += steps;
        }
        return total / episodes;
    }

    /**
     * The cart-pole control surface: states visited by the law with 30% random actions, the law's
     * continuous output as target; test RMSE on fresh states and closed-loop steps (of 500) when
     * the fitted surface flies the cart-pole.
     *
     * @param samples training states
     * @param anfisEpochs hybrid epochs
     * @param seed the seed
     * @return the rows
     */
    public static List<Row> cartPole(int samples, int anfisEpochs, long seed) {
        double[][][] tr = cartPoleData(samples, seed);
        double[][][] te = cartPoleData(samples / 2, seed + 1);
        double[] yTr = column(tr[1]);
        double[] yTe = column(te[1]);
        List<Row> rows = new ArrayList<>();
        rows.add(new Row("cart-pole closed-loop steps", "PD law (teacher)",
            episodes(v -> law(new float[] {(float) v[0], (float) v[1], (float) v[2], (float) v[3]}), 10, seed), 0, 0,
            0));
        long t0 = System.nanoTime();
        var fis = gridSystem(tr[0], 2, io.github.zzuegg.brainiac.experimental.logic.FuzzyInference.Type.TAKAGI_SUGENO,
            io.github.zzuegg.brainiac.experimental.logic.FuzzyInference.Membership.GAUSSIAN);
        new Anfis(fis).train(tr[0], yTr, anfisEpochs, 10, 0.01, 1e-6, true);
        double train = ms(t0);
        rows.add(new Row("cart-pole surface RMSE", "ANFIS hybrid", rmse(fis::predict, te[0], yTe), fis.parameters(),
            fis.rules(), train));
        rows.add(new Row("cart-pole closed-loop steps", "ANFIS hybrid", episodes(fis::predict, 10, seed), 0, 0, 0));
        t0 = System.nanoTime();
        Model m = mlp(tr[0], yTr, Math.max(2, (fis.parameters() - 1) / 6), 30 * anfisEpochs, seed);
        train = ms(t0);
        rows.add(new Row("cart-pole surface RMSE", "MLP", rmse(m.predict(), te[0], yTe), m.parameters(),
            30 * anfisEpochs, train));
        rows.add(new Row("cart-pole closed-loop steps", "MLP", episodes(m.predict(), 10, seed), 0, 0, 0));
        return rows;
    }

    private static double[][][] cartPoleData(int samples, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[samples][];
        double[][] y = new double[samples][1];
        CartPole env = new CartPole(500);
        float[] o = env.reset(r);
        for (int i = 0; i < samples; i++) {
            x[i] = new double[] {o[0], o[1], o[2], o[3]};
            y[i][0] = law(o);
            int a = r.nextDouble() < 0.3 ? r.nextInt(2) : y[i][0] > 0 ? 1 : 0;
            Environment.Step st = env.step(new float[] {a});
            o = st.terminated() || st.truncated() ? env.reset(r) : st.observation();
        }
        return new double[][][] {x, y};
    }

    // ---- rule growth --------------------------------------------------------------------------------

    /**
     * Rule counts of grid partitions against subtractive clustering as the input dimension grows
     * (Mackey–Glass with 4 and 8 lags); systems with at most 100 rules are fitted (least-squares
     * consequents plus a few ANFIS epochs) and report their test RMSE, larger grids report NaN.
     *
     * @param seed the seed
     * @return the rows
     */
    public static List<Row> ruleGrowth(long seed) {
        double[] s = ChaoticSeries.mackeyGlass(1100, 17, 1, 300);
        List<Row> rows = new ArrayList<>();
        var ts = io.github.zzuegg.brainiac.experimental.logic.FuzzyInference.Type.TAKAGI_SUGENO;
        var gauss = io.github.zzuegg.brainiac.experimental.logic.FuzzyInference.Membership.GAUSSIAN;
        for (int d : new int[] {4, 8}) {
            double[][][] data = lagged(s, d, d == 4 ? 6 : 3, 6);
            double[][] xTr = java.util.Arrays.copyOfRange(data[0], 0, 500);
            double[] yTr = column(java.util.Arrays.copyOfRange(data[1], 0, 500));
            double[][] xTe = java.util.Arrays.copyOfRange(data[0], 500, 1000);
            double[] yTe = column(java.util.Arrays.copyOfRange(data[1], 500, 1000));
            for (int m : new int[] {2, 3}) {
                long t0 = System.nanoTime();
                int rules = (int) Math.pow(m, d);
                double err = Double.NaN;
                if (rules <= 100) {
                    var g = gridSystem(xTr, m, ts, gauss);
                    new Anfis(g).train(xTr, yTr, 3, 10, 0.01, 1e-6, true);
                    err = rmse(g::predict, xTe, yTe);
                }
                rows.add(new Row("mackey-glass d=" + d, "grid " + m + " MFs/input", err,
                    rules * (2.0 * d + d + 1), rules, ms(t0)));
            }
            long t0 = System.nanoTime();
            var c = io.github.zzuegg.brainiac.experimental.logic.FuzzyInference.clustering(ts, gauss, xTr, 0.5);
            new Anfis(c).train(xTr, yTr, 3, 10, 0.01, 1e-6, true);
            rows.add(new Row("mackey-glass d=" + d, "subtractive clustering r=0.5", rmse(c::predict, xTe, yTe),
                c.parameters(), c.rules(), ms(t0)));
        }
        return rows;
    }
}
