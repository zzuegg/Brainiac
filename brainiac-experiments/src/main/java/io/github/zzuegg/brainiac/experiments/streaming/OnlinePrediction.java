package io.github.zzuegg.brainiac.experiments.streaming;

import io.github.zzuegg.brainiac.experimental.data.ChaoticSeries;
import io.github.zzuegg.brainiac.experimental.data.DriftingStreams;
import io.github.zzuegg.brainiac.experimental.data.SeasonalSeries;
import io.github.zzuegg.brainiac.experimental.ml.KernelFunction;
import io.github.zzuegg.brainiac.experimental.ml.KernelSvm;
import io.github.zzuegg.brainiac.experimental.online.BudgetedKernels;
import io.github.zzuegg.brainiac.experimental.online.ConformalPid;
import io.github.zzuegg.brainiac.experimental.online.ExpertAdvice;
import io.github.zzuegg.brainiac.experimental.stream.FtrlProximal;
import io.github.zzuegg.brainiac.experimental.stream.OnlineClassifier;
import io.github.zzuegg.brainiac.experimental.stream.OnlineLogistic;
import io.github.zzuegg.brainiac.experimental.timeseries.AutoEts;
import io.github.zzuegg.brainiac.experimental.timeseries.ConformalForecaster;
import io.github.zzuegg.brainiac.experimental.timeseries.ExponentialSmoothing;
import io.github.zzuegg.brainiac.experimental.timeseries.ThetaMethod;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Wave 12 items 12.2.5, 12.2.6 and 12.2.8: online prediction on streams.
 *
 * <ul>
 *   <li>{@link #kernels}: budgeted online kernel classifiers ({@link BudgetedKernels}) against
 *       linear online learners ({@link FtrlProximal}, {@link OnlineLogistic}) on binned or binary
 *       features and an offline {@link KernelSvm}, on streamed spirals, digits and a drifting
 *       RBF stream; prequential mistake rate, held-out accuracy and time per point.</li>
 *   <li>{@link #experts}: prediction with expert advice ({@link ExpertAdvice}) combining nine
 *       one-step forecasters on a seasonal series with regime switches, plus a stochastic and a
 *       switching synthetic expert set; loss, regret against the best expert and against the
 *       best sequence with three switches.</li>
 *   <li>{@link #intervals}: conformal PID control ({@link ConformalPid}) against ACI, sliding
 *       residual (EnbPI-style) and fixed split conformal ({@link ConformalForecaster}) around a
 *       Holt-Winters base forecaster, on a seasonal series with a level shift, a variance burst
 *       and a trend, and on Mackey-Glass.</li>
 * </ul>
 */
public final class OnlinePrediction {

    private OnlinePrediction() {
    }

    // ------------------------------------------------------------------ 12.2.5 kernels

    /**
     * One online classifier on one stream.
     *
     * @param task the stream
     * @param method the learner
     * @param budget its budget (support vectors, or random features), {@code -1} if unbounded
     * @param mistakeRate the prequential mistake rate ({@code NaN} for the offline model)
     * @param accuracy the held-out accuracy at the end
     * @param supportVectors the support vectors kept at the end
     * @param microsPerPoint the learning and prediction time per point
     */
    public record KernelRow(String task, String method, int budget, double mistakeRate, double accuracy,
            int supportVectors, double microsPerPoint) {
    }

    private record Task(String name, double[][] x, int[] y, double[][] testX, int[] testY, int classes, double gamma,
            double lo, double hi, boolean binary) {
    }

    private static double[][] toDouble(float[][] x) {
        double[][] d = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            d[i] = new double[x[i].length];
            for (int j = 0; j < x[i].length; j++) {
                d[i][j] = x[i][j];
            }
        }
        return d;
    }

    private static List<Task> kernelTasks(int n, long seed) {
        List<Task> tasks = new ArrayList<>();
        ToyTasks.Data spiral = ToyTasks.spiral(n / 3, 3, 0.1, seed);
        ToyTasks.Data spiralTest = ToyTasks.spiral(200, 3, 0.1, seed + 100);
        tasks.add(new Task("spirals", toDouble(spiral.x()), spiral.y(), toDouble(spiralTest.x()), spiralTest.y(), 3,
                20, -1, 1, false));
        ToyTasks.Data digits = ToyTasks.digits(10, n / 10, 8, 0.1, seed + 1, seed + 2);
        ToyTasks.Data digitsTest = ToyTasks.digits(10, 60, 8, 0.1, seed + 1, seed + 3);
        tasks.add(new Task("digits", toDouble(digits.x()), digits.y(), toDouble(digitsTest.x()), digitsTest.y(), 10,
                0.05, 0, 1, true));
        // drifting RBF stream: 2-d, 12 clusters, 2 classes, abrupt drift at the middle; the
        // held-out set is drawn from the final concept
        DriftingStreams.Stream rbf = DriftingStreams.rbf(2, 12, 2, DriftingStreams.Drift.ABRUPT, new int[] {n / 2}, 0,
                seed + 4);
        DriftingStreams.Sample[] s = DriftingStreams.take(rbf, n + 600);
        double[][] x = new double[n][];
        int[] y = new int[n];
        double[][] tx = new double[600][];
        int[] ty = new int[600];
        for (int i = 0; i < n + 600; i++) {
            if (i < n) {
                x[i] = s[i].x();
                y[i] = s[i].label();
            } else {
                tx[i - n] = s[i].x();
                ty[i - n] = s[i].label();
            }
        }
        tasks.add(new Task("drifting RBF", x, y, tx, ty, 2, 30, -0.2, 1.2, false));
        return tasks;
    }

    /** The sparse binary features of the linear learners: pixels on, or one-hot bins per coordinate. */
    private static int[] active(Task task, double[] x) {
        if (task.binary()) {
            int c = 0;
            int[] a = new int[x.length];
            for (int j = 0; j < x.length; j++) {
                if (x[j] > 0.5) {
                    a[c++] = j;
                }
            }
            return Arrays.copyOf(a, c);
        }
        int bins = 16;
        int[] a = new int[x.length];
        for (int j = 0; j < x.length; j++) {
            int b = (int) Math.floor((x[j] - task.lo()) / (task.hi() - task.lo()) * bins);
            a[j] = j * bins + Math.max(0, Math.min(bins - 1, b));
        }
        return a;
    }

    /** One-vs-rest over binary sparse learners, so they take multi-class streams. */
    private static final class OneVsRest implements OnlineClassifier {
        private final OnlineClassifier[] heads;

        OneVsRest(int classes, Supplier<OnlineClassifier> factory) {
            heads = new OnlineClassifier[classes == 2 ? 1 : classes];
            for (int c = 0; c < heads.length; c++) {
                heads[c] = factory.get();
            }
        }

        @Override
        public double[] predict(DriftingStreams.Sample sample) {
            if (heads.length == 1) {
                return heads[0].predict(sample);
            }
            double[] p = new double[heads.length];
            for (int c = 0; c < heads.length; c++) {
                p[c] = heads[c].predict(sample)[1];
            }
            return p;
        }

        @Override
        public void learn(DriftingStreams.Sample sample) {
            if (heads.length == 1) {
                heads[0].learn(sample);
                return;
            }
            for (int c = 0; c < heads.length; c++) {
                heads[c].learn(new DriftingStreams.Sample(sample.x(), sample.active(), sample.label() == c ? 1 : 0,
                        sample.concept()));
            }
        }
    }

    private static KernelRow runOnline(Task task, String method, int budget, OnlineClassifier learner,
            boolean sparse) {
        long start = System.nanoTime();
        int mistakes = 0;
        for (int i = 0; i < task.x().length; i++) {
            double[] x = task.x()[i];
            DriftingStreams.Sample s = new DriftingStreams.Sample(x, sparse ? active(task, x) : new int[0],
                    task.y()[i], 0);
            if (OnlineClassifier.argmax(learner.predict(s)) != task.y()[i]) {
                mistakes++;
            }
            learner.learn(s);
        }
        double micros = (System.nanoTime() - start) / 1e3 / task.x().length;
        int correct = 0;
        for (int i = 0; i < task.testX().length; i++) {
            double[] x = task.testX()[i];
            DriftingStreams.Sample s = new DriftingStreams.Sample(x, sparse ? active(task, x) : new int[0], 0, 0);
            correct += OnlineClassifier.argmax(learner.predict(s)) == task.testY()[i] ? 1 : 0;
        }
        int sv = learner instanceof BudgetedKernels b ? b.supportVectors() : 0;
        return new KernelRow(task.name(), method, budget, (double) mistakes / task.x().length,
                (double) correct / task.testX().length, sv, micros);
    }

    /**
     * 12.2.5: online kernel learning on a budget.
     *
     * @param n stream length per task
     * @param budgets the support-vector budgets (online RFF uses {@code 2.5×} as many features)
     * @param seed the seed
     * @return one row per task, learner and budget
     */
    public static List<KernelRow> kernels(int n, int[] budgets, long seed) {
        List<KernelRow> rows = new ArrayList<>();
        for (Task task : kernelTasks(n, seed)) {
            int d = task.x()[0].length;
            int k = task.classes();
            double g = task.gamma();
            int space = task.binary() ? d : d * 16;
            rows.add(runOnline(task, "FTRL-proximal (linear, sparse features)", -1,
                    new OneVsRest(k, () -> new FtrlProximal(space, 0.1, 1, 0, 0, 0, false)), true));
            rows.add(runOnline(task, "online logistic (AdaGrad, sparse features)", -1,
                    new OneVsRest(k, () -> new OnlineLogistic(space, 0.1, true, 0)), true));
            rows.add(runOnline(task, "kernel perceptron (unbounded)", -1,
                    BudgetedKernels.of(BudgetedKernels.Kind.KERNEL_PERCEPTRON, d, k, 0, g, 0, 0, seed), false));
            for (int b : budgets) {
                rows.add(runOnline(task, "NORMA", b,
                        BudgetedKernels.of(BudgetedKernels.Kind.NORMA, d, k, b, g, 1e-3, 0.2, seed), false));
                rows.add(runOnline(task, "Forgetron", b,
                        BudgetedKernels.of(BudgetedKernels.Kind.FORGETRON, d, k, b, g, 0, 0, seed), false));
                rows.add(runOnline(task, "BSGD (remove)", b,
                        BudgetedKernels.of(BudgetedKernels.Kind.BSGD_REMOVE, d, k, b, g, 1e-3, 0, seed), false));
                rows.add(runOnline(task, "BSGD (merge)", b,
                        BudgetedKernels.of(BudgetedKernels.Kind.BSGD_MERGE, d, k, b, g, 1e-3, 0, seed), false));
                int features = (int) Math.round(2.5 * b);
                rows.add(runOnline(task, "online RFF", features,
                        BudgetedKernels.of(BudgetedKernels.Kind.ONLINE_RFF, d, k, features, g, 1e-4, 0.2, seed),
                        false));
            }
            // offline upper bound: an SVM trained on (up to 1500 of) the stream, last concept first
            long start = System.nanoTime();
            int m = Math.min(1500, task.x().length);
            double[][] fx = Arrays.copyOfRange(task.x(), task.x().length - m, task.x().length);
            int[] fy = Arrays.copyOfRange(task.y(), task.y().length - m, task.y().length);
            KernelSvm svm = KernelSvm.fit(fx, fy, k, KernelFunction.rbf(g), 10, 1e-3);
            double micros = (System.nanoTime() - start) / 1e3 / m;
            int correct = 0;
            for (int i = 0; i < task.testX().length; i++) {
                correct += svm.predict(task.testX()[i]) == task.testY()[i] ? 1 : 0;
            }
            rows.add(new KernelRow(task.name(), "kernel SVM (batch, last " + m + " points)", -1, Double.NaN,
                    (double) correct / task.testX().length, svm.supportVectors(), micros));
        }
        return rows;
    }

    // ------------------------------------------------------------------ 12.2.6 experts

    /**
     * One combination rule on one expert set.
     *
     * @param setting the expert set
     * @param method the rule
     * @param meanLoss the mean loss per round (of the combined forecast for the forecasters,
     *     the expected loss for the synthetic sets)
     * @param regret the mean regret per round against the best expert in hindsight
     * @param switchingRegret the mean regret against the best sequence with three switches
     */
    public record ExpertRow(String setting, String method, double meanLoss, double regret, double switchingRegret) {
    }

    private static final int SEASON = 12;

    /** A seasonal series with four regimes (season, trend, random walk, strong season). */
    static double[] regimeSeries(int length, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] y = new double[length];
        int seg = length / 4;
        double level = 0;
        double walk = 0;
        for (int t = 0; t < length; t++) {
            int regime = Math.min(3, t / seg);
            double w = 2 * Math.PI * t / SEASON;
            double v = switch (regime) {
                case 0 -> Math.sin(w) + 0.2 * r.nextGaussian();
                case 1 -> 0.2 * Math.sin(w) + 0.03 * (t - seg) + 0.2 * r.nextGaussian();
                case 2 -> {
                    walk += 0.4 * r.nextGaussian();
                    yield walk + 0.05 * r.nextGaussian();
                }
                default -> 1.5 * Math.sin(w + 1) + 0.1 * r.nextGaussian();
            };
            if (t > 0 && t % seg == 0 && t / seg <= 3) {
                level = y[t - 1] - v;   // keep the series continuous across a regime change
            }
            y[t] = v + level;
        }
        return y;
    }

    private static double meanOf(double[] y, int from, int to) {
        double s = 0;
        for (int i = from; i < to; i++) {
            s += y[i];
        }
        return s / Math.max(1, to - from);
    }

    /** One-step forecasts of nine experts for every {@code t ≥ start}. */
    static double[][] expertForecasts(double[] y, int start) {
        int n = y.length;
        int experts = 9;
        double[][] f = new double[n][experts];
        double ses = meanOf(y, 0, start);
        double holtL = ses;
        double holtB = 0;
        AutoEts ets = null;
        ThetaMethod theta = null;
        int fitAt = -1;
        int refit = 60;
        int fitWindow = 144;
        for (int t = 0; t < n; t++) {
            if (t >= start) {
                if (ets == null || t - fitAt >= refit) {
                    double[] hist = Arrays.copyOfRange(y, t - fitWindow, t);
                    ets = AutoEts.select(hist, SEASON);
                    theta = ThetaMethod.fit(hist, SEASON);
                    fitAt = t;
                }
                int h = t - fitAt + 1;
                double[] row = f[t];
                row[0] = y[t - 1];
                row[1] = y[t - SEASON];
                row[2] = ses;
                row[3] = holtL + holtB;
                row[4] = y[t - 1] + (y[t - 1] - y[t - 1 - SEASON]) / SEASON;
                row[5] = meanOf(y, t - SEASON, t);
                row[6] = ets.forecast(h)[h - 1];
                row[7] = theta.forecast(h)[h - 1];
                row[8] = y[t - SEASON] + meanOf(y, t - SEASON, t) - meanOf(y, t - 2 * SEASON, t - SEASON);
            }
            double prevL = holtL;
            ses += 0.3 * (y[t] - ses);
            holtL = 0.3 * y[t] + 0.7 * (holtL + holtB);
            holtB = 0.1 * (holtL - prevL) + 0.9 * holtB;
        }
        return f;
    }

    static final String[] EXPERT_NAMES = {"naive", "seasonal naive", "SES", "Holt", "drift", "moving average",
        "AutoETS (refit every 60)", "Theta (refit every 60)", "seasonal naive + level"};

    private static ExpertAdvice advice(String name, int experts, int horizon) {
        double eta = ExpertAdvice.tunedRate(experts, horizon);
        return switch (name) {
            case "follow the leader" -> ExpertAdvice.of(ExpertAdvice.Kind.FOLLOW_THE_LEADER, experts, 0, 0);
            case "Hedge (tuned rate)" -> ExpertAdvice.of(ExpertAdvice.Kind.HEDGE, experts, eta, 0);
            case "AdaHedge" -> ExpertAdvice.of(ExpertAdvice.Kind.ADAHEDGE, experts, 0, 0);
            case "Squint" -> ExpertAdvice.of(ExpertAdvice.Kind.SQUINT, experts, 0, 0);
            default -> ExpertAdvice.of(ExpertAdvice.Kind.FIXED_SHARE, experts, 2, 0.01);
        };
    }

    private static final String[] RULES = {"follow the leader", "Hedge (tuned rate)", "AdaHedge", "Squint",
        "fixed share (eta 2, alpha 0.01)"};

    /**
     * 12.2.6: prediction with expert advice.
     *
     * @param length the forecast series length
     * @param seed the seed
     * @return one row per expert set and rule, plus the best expert, best switching sequence
     *     and uniform average
     */
    public static List<ExpertRow> experts(int length, long seed) {
        List<ExpertRow> rows = new ArrayList<>();
        // forecast combination: absolute error / 2, clipped to [0, 1]
        double[] y = regimeSeries(length, seed);
        int start = 156;
        double[][] f = expertForecasts(y, start);
        int horizon = length - start;
        int experts = EXPERT_NAMES.length;
        double[][] losses = new double[horizon][experts];
        for (int t = start; t < length; t++) {
            for (int e = 0; e < experts; e++) {
                losses[t - start][e] = Math.min(1, Math.abs(y[t] - f[t][e]) / 2);
            }
        }
        double[] cum = columnSums(losses);
        int bestExpert = argmin(cum);
        double best = cum[bestExpert];
        double switching = ExpertAdvice.bestSwitching(losses, 3);
        String setting = "regime-switching forecasts";
        for (String rule : RULES) {
            ExpertAdvice a = advice(rule, experts, horizon);
            double loss = 0;
            for (int t = start; t < length; t++) {
                double p = a.predict(f[t]);
                loss += Math.min(1, Math.abs(y[t] - p) / 2);
                a.update(losses[t - start]);
            }
            rows.add(new ExpertRow(setting, rule, loss / horizon, (loss - best) / horizon,
                    (loss - switching) / horizon));
        }
        double uniform = 0;
        for (int t = start; t < length; t++) {
            uniform += Math.min(1, Math.abs(y[t] - meanOf(f[t], 0, experts)) / 2);
        }
        rows.add(new ExpertRow(setting, "uniform average", uniform / horizon, (uniform - best) / horizon,
                (uniform - switching) / horizon));
        rows.add(new ExpertRow(setting, "best expert: " + EXPERT_NAMES[bestExpert], best / horizon, 0,
                (best - switching) / horizon));
        // synthetic sets: Bernoulli losses
        SplittableRandom r = new SplittableRandom(seed + 9);
        int n = 10;
        int rounds = 2000;
        for (String kind : new String[] {"stochastic (one expert 0.4, rest 0.5)", "switching best expert (every 250)"}) {
            double[][] l = new double[rounds][n];
            for (int t = 0; t < rounds; t++) {
                for (int e = 0; e < n; e++) {
                    double mean = kind.startsWith("stochastic") ? (e == 3 ? 0.4 : 0.5)
                            : (e == (t / 250) % n ? 0.2 : 0.6);
                    l[t][e] = r.nextDouble() < mean ? 1 : 0;
                }
            }
            double[] c = columnSums(l);
            double b = c[argmin(c)];
            double sw = ExpertAdvice.bestSwitching(l, rounds / 250 - 1);
            for (String rule : RULES) {
                ExpertAdvice a = advice(rule, n, rounds);
                for (double[] row : l) {
                    a.update(row);
                }
                rows.add(new ExpertRow(kind, rule, a.loss() / rounds, (a.loss() - b) / rounds,
                        (a.loss() - sw) / rounds));
            }
            double u = Arrays.stream(c).sum() / n;
            rows.add(new ExpertRow(kind, "uniform average", u / rounds, (u - b) / rounds, (u - sw) / rounds));
        }
        return rows;
    }

    private static double[] columnSums(double[][] m) {
        double[] s = new double[m[0].length];
        for (double[] row : m) {
            for (int j = 0; j < s.length; j++) {
                s[j] += row[j];
            }
        }
        return s;
    }

    private static int argmin(double[] v) {
        int b = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] < v[b]) {
                b = i;
            }
        }
        return b;
    }

    // ------------------------------------------------------------------ 12.2.8 intervals

    /**
     * One interval method on one series.
     *
     * @param series the series
     * @param method the method
     * @param coverage the long-run coverage after calibration
     * @param localDeviation the mean absolute deviation of coverage over windows of 50 from the
     *     target
     * @param worstWindow the lowest coverage of a window of 50
     * @param width the mean interval width
     */
    public record IntervalRow(String series, String method, double coverage, double localDeviation,
            double worstWindow, double width) {
    }

    /** A seasonal series with a level shift, a variance burst and a trend. */
    static double[] shiftingSeries(int length, long seed) {
        float[] base = SeasonalSeries.generate(length, SEASON, 0, 1, 0.2, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        double[] y = new double[length];
        for (int t = 0; t < length; t++) {
            y[t] = base[t];
            if (t >= length * 7 / 20) {
                y[t] += 3;
            }
            if (t >= length * 11 / 20 && t < length * 13 / 20) {
                y[t] += 0.8 * r.nextGaussian();
            }
            if (t >= length * 15 / 20) {
                y[t] += 0.02 * (t - length * 15 / 20);
            }
        }
        return y;
    }

    /** One-step Holt-Winters forecasts (additive, damped) with fixed parameters. */
    static double[] holtWinters(double[] y, double[] p, int period) {
        double alpha = p[0];
        double beta = p[1];
        double gamma = p[2];
        double phi = p[3];
        int m = Math.max(1, period);
        double[] f = new double[y.length];
        double level = meanOf(y, 0, m);
        double slope = period > 1 ? (meanOf(y, m, 2 * m) - level) / m : 0;
        double[] s = new double[m];
        if (period > 1) {
            for (int i = 0; i < m; i++) {
                s[i] = y[i] - level;
            }
        }
        for (int t = 0; t < y.length; t++) {
            int k = t % m;
            f[t] = level + phi * slope + s[k];
            double e = y[t] - f[t];
            level = level + phi * slope + alpha * e;
            slope = phi * slope + beta * e;
            if (period > 1) {
                s[k] += gamma * e;
            }
        }
        return f;
    }

    /**
     * 12.2.8: conformal PID control for online intervals.
     *
     * @param length the series length
     * @param alpha the target miscoverage
     * @param seed the seed
     * @return one row per series and method
     */
    public static List<IntervalRow> intervals(int length, double alpha, long seed) {
        List<IntervalRow> rows = new ArrayList<>();
        Map<String, double[]> series = new LinkedHashMap<>();
        series.put("shift, burst, trend", shiftingSeries(length, seed));
        double[] mg = ChaoticSeries.mackeyGlass(length, 17, 1, 300);
        series.put("Mackey-Glass", mg);
        int fitEnd = length * 3 / 20;
        int calEnd = length / 4;
        for (Map.Entry<String, double[]> e : series.entrySet()) {
            double[] y = e.getValue();
            boolean seasonal = e.getKey().startsWith("shift");
            int period = seasonal ? SEASON : 1;
            ExponentialSmoothing es = ExponentialSmoothing.fit(Arrays.copyOf(y, fitEnd),
                    seasonal ? ExponentialSmoothing.Trend.DAMPED : ExponentialSmoothing.Trend.ADDITIVE, period,
                    ExponentialSmoothing.Method.GRID_WARM, 3);
            double[] f = holtWinters(y, es.parameters(), period);
            double[] calF = Arrays.copyOfRange(f, fitEnd, calEnd);
            double[] calY = Arrays.copyOfRange(y, fitEnd, calEnd);
            double[] res = new double[calY.length];
            for (int i = 0; i < res.length; i++) {
                res[i] = calY[i] - calF[i];
            }
            Map<String, ConformalForecaster> cf = new LinkedHashMap<>();
            cf.put("split conformal (fixed)", ConformalForecaster.split(res, alpha));
            cf.put("ACI (gamma 0.005, window 200)", ConformalForecaster.aci(calF, calF, calY, alpha, 0.005, 200));
            cf.put("sliding residuals (EnbPI-style, window 200)", ConformalForecaster.enbpi(res, alpha, 200));
            for (Map.Entry<String, ConformalForecaster> m : cf.entrySet()) {
                Scorer sc = new Scorer(alpha);
                for (int t = calEnd; t < y.length; t++) {
                    double[] iv = m.getValue().interval(f[t], f[t]);
                    sc.add(iv, y[t]);
                    m.getValue().update(f[t], f[t], y[t]);
                }
                rows.add(sc.row(e.getKey(), m.getKey()));
            }
            Map<String, ConformalPid> pids = new LinkedHashMap<>();
            pids.put("quantile tracking (P)", ConformalPid.of(ConformalPid.Kind.QUANTILE_TRACKING, alpha));
            pids.put("PI (tracking + saturated integrator)", ConformalPid.of(ConformalPid.Kind.PI, alpha));
            pids.put("PID (PI + AR(3) scorecaster)", ConformalPid.of(ConformalPid.Kind.PID, alpha));
            for (Map.Entry<String, ConformalPid> m : pids.entrySet()) {
                ConformalPid c = m.getValue();
                for (int t = fitEnd; t < calEnd; t++) {
                    c.update(f[t], y[t]);
                }
                Scorer sc = new Scorer(alpha);
                for (int t = calEnd; t < y.length; t++) {
                    sc.add(c.interval(f[t]), y[t]);
                    c.update(f[t], y[t]);
                }
                rows.add(sc.row(e.getKey(), m.getKey()));
            }
        }
        return rows;
    }

    private static final class Scorer {
        private final double target;
        private final List<Integer> hits = new ArrayList<>();
        private double width;

        Scorer(double alpha) {
            this.target = 1 - alpha;
        }

        void add(double[] iv, double y) {
            hits.add(y >= iv[0] && y <= iv[1] ? 1 : 0);
            width += iv[1] - iv[0];
        }

        IntervalRow row(String series, String method) {
            int n = hits.size();
            double total = 0;
            for (int h : hits) {
                total += h;
            }
            double dev = 0;
            double worst = 1;
            int windows = 0;
            for (int w = 0; w + 50 <= n; w += 50) {
                double c = 0;
                for (int i = w; i < w + 50; i++) {
                    c += hits.get(i);
                }
                c /= 50;
                dev += Math.abs(c - target);
                worst = Math.min(worst, c);
                windows++;
            }
            return new IntervalRow(series, method, total / n, dev / Math.max(1, windows), worst, width / n);
        }
    }
}
