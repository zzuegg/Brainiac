package io.github.zzuegg.brainiac.experiments.multitask;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ClickLogs;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.experimental.multitask.Famo;
import io.github.zzuegg.brainiac.experimental.multitask.GradientSurgery;
import io.github.zzuegg.brainiac.experimental.multitask.LossWeighting;
import io.github.zzuegg.brainiac.experimental.multitask.ParetoDescent;
import io.github.zzuegg.brainiac.experimental.multitask.TaskGradients;
import io.github.zzuegg.brainiac.experimental.recsys.MultiGateExperts;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 10 items 10.3.13-10.3.16: multi-task gradient balancing on a shared trunk.
 *
 * <p>A two-layer MLP trunk feeds one linear head per task. Every method sees the same data,
 * steps, batch and Adam settings; only the way the task losses or task gradients are combined
 * differs: equal weights, the loss weightings of {@link LossWeighting} (uncertainty, GradNorm,
 * DWA), the gradient surgery of {@link GradientSurgery} (PCGrad, GradVac, GradOPS), the
 * multi-objective directions of {@link ParetoDescent} (MGDA on the shared weights, MGDA-UB on
 * the shared representation, CAGrad, Nash-MTL) and {@link Famo}. Each is scored by the average
 * relative improvement over single-task models,
 * {@code Delta_m = mean_t s_t (M_t - B_t) / B_t * 100} ({@code s_t = +1} when higher is
 * better), plus the share of steps with conflicting gradients and the wall-clock per step.
 *
 * <p>Problems ({@link Problem}):
 * <ul>
 *   <li>{@code DIGITS}: {@link ToyTasks#digits} with four labels, class (10-way), parity
 *       (2-way), magnitude (the digit as a number, regression) and stroke count (the class's
 *       mean number of lit pixels, regression), so the regression losses start an order of
 *       magnitude above the classification losses;</li>
 *   <li>{@code CLICKS}: wave 9's {@link ClickLogs#multiTask} two-label table at task
 *       correlation -0.5, scored by AUC, with {@link MultiGateExperts} MMoE as the architectural
 *       baseline;</li>
 *   <li>{@code SCALES}: two regressions of the same inputs whose targets differ 10x in scale
 *       (100x in loss).</li>
 * </ul>
 *
 * <p>{@link #toySurface} reproduces the classic two-objective loss surface of the PCGrad and
 * CAGrad papers, where summing the gradients stalls in a flat valley while conflict-aware
 * methods reach the Pareto set.
 */
public final class GradientBalancing {

    /** How the tasks are combined. */
    public enum Method {
        /** The plain sum. */
        EQUAL,
        /** Uncertainty weighting. */
        UNCERTAINTY,
        /** GradNorm. */
        GRADNORM,
        /** Dynamic weight averaging. */
        DWA,
        /** PCGrad. */
        PCGRAD,
        /** GradVac. */
        GRADVAC,
        /** GradOPS (alpha = 0). */
        GRADOPS,
        /** MGDA on the shared weights. */
        MGDA,
        /** MGDA-UB on the shared representation. */
        MGDA_UB,
        /** CAGrad (c = 0.4). */
        CAGRAD,
        /** Nash-MTL. */
        NASH_MTL,
        /** FAMO. */
        FAMO
    }

    /** The task sets. */
    public enum Problem {
        /** Four labels of the toy digits. */
        DIGITS,
        /** Two click labels at correlation -0.5. */
        CLICKS,
        /** Two regressions with a 100x loss-scale mismatch. */
        SCALES
    }

    /**
     * The outcome of one method on one problem.
     *
     * @param problem the problem
     * @param method the method name
     * @param metrics per-task test metrics (accuracy or AUC: higher is better; MSE: lower)
     * @param deltaM the average relative improvement over the single-task models, in percent
     * @param conflictShare the share of task pairs with negative gradient cosine, averaged over
     *     steps ({@code NaN} for methods that do not compute per-task gradients)
     * @param millisPerStep the training wall-clock per step
     */
    public record Result(Problem problem, String method, double[] metrics, double deltaM, double conflictShare,
            double millisPerStep) {

        @Override
        public String toString() {
            return String.format("%s %s Δm %+.1f%% metrics %s conflicts %.2f %.2f ms/step", problem, method, deltaM,
                    Arrays.toString(Arrays.stream(metrics).map(v -> Math.round(v * 1000) / 1000.0).toArray()),
                    conflictShare, millisPerStep);
        }
    }

    private enum Kind {
        CLASSES, REGRESSION
    }

    private record Task(String name, Kind kind, int outputs, boolean auc) {
    }

    private record Split(Tensor x, float[][] y) {
    }

    private record Data(List<Task> tasks, Split train, Split test) {
    }

    private GradientBalancing() {
    }

    private static Data data(Problem problem, long seed) {
        return switch (problem) {
            case DIGITS -> digits(seed);
            case CLICKS -> clicks(seed);
            case SCALES -> scales(seed);
        };
    }

    private static Data digits(long seed) {
        ToyTasks.Data train = ToyTasks.digits(10, 60, 8, 0.1, seed, seed * 31 + 7);
        ToyTasks.Data test = ToyTasks.digits(10, 30, 8, 0.1, seed, seed * 31 + 8);
        double[] strokes = new double[10];
        int[] count = new int[10];
        for (int i = 0; i < train.size(); i++) {
            int lit = 0;
            for (float v : train.x()[i]) {
                lit += v > 0.5f ? 1 : 0;
            }
            strokes[train.y()[i]] += lit;
            count[train.y()[i]]++;
        }
        double mean = 0;
        for (int c = 0; c < 10; c++) {
            strokes[c] /= Math.max(1, count[c]);
            mean += strokes[c] / 10;
        }
        List<Task> tasks = List.of(new Task("class", Kind.CLASSES, 10, false), new Task("parity", Kind.CLASSES, 2, false),
                new Task("magnitude", Kind.REGRESSION, 1, false), new Task("strokes", Kind.REGRESSION, 1, false));
        return new Data(tasks, digitSplit(train, strokes, mean), digitSplit(test, strokes, mean));
    }

    private static Split digitSplit(ToyTasks.Data d, double[] strokes, double mean) {
        float[][] y = new float[4][d.size()];
        for (int i = 0; i < d.size(); i++) {
            int c = d.y()[i];
            y[0][i] = c;
            y[1][i] = c % 2;
            y[2][i] = c - 4.5f;
            y[3][i] = (float) (strokes[c] - mean);
        }
        return new Split(d.inputs(), y);
    }

    private static Data clicks(long seed) {
        ClickLogs.MultiTaskTable t = ClickLogs.multiTask(4000, 3, 8, -0.5, seed);
        List<Task> tasks = List.of(new Task("click", Kind.CLASSES, 2, true), new Task("other", Kind.CLASSES, 2, true));
        return new Data(tasks, clickSplit(t.rows(0, 3000)), clickSplit(t.rows(3000, 4000)));
    }

    private static Split clickSplit(ClickLogs.MultiTaskTable t) {
        float[] l = t.labels().toArray();
        int n = t.rows();
        float[][] y = new float[2][n];
        for (int i = 0; i < n; i++) {
            y[0][i] = l[2 * i];
            y[1][i] = l[2 * i + 1];
        }
        return new Split(t.features(), y);
    }

    private static Data scales(long seed) {
        List<Task> tasks = List.of(new Task("small", Kind.REGRESSION, 1, false),
                new Task("large", Kind.REGRESSION, 1, false));
        return new Data(tasks, scaleSplit(1500, seed), scaleSplit(500, seed + 1));
    }

    private static Split scaleSplit(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] x = new float[n * 8];
        float[][] y = new float[2][n];
        for (int i = 0; i < n; i++) {
            double[] v = new double[8];
            for (int j = 0; j < 8; j++) {
                v[j] = r.nextGaussian();
                x[i * 8 + j] = (float) v[j];
            }
            y[0][i] = (float) (Math.sin(v[0]) + 0.5 * v[1] * v[2] + 0.1 * r.nextGaussian());
            y[1][i] = (float) (10 * (Math.cos(v[3]) * v[4] - 0.5 * v[0] + 0.1 * r.nextGaussian()));
        }
        return new Split(Tensor.of(Shape.of(n, 8), x), y);
    }

    /** A trunk and one head per task. */
    private record Net(Model trunk, List<Model> heads) {

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(trunk.parameters());
            for (Model h : heads) {
                p.addAll(h.parameters());
            }
            return p;
        }
    }

    private static Net net(int in, List<Task> tasks, long seed) {
        Model trunk = Model.sequential().input(in).dense(64, Activation.RELU).dense(64, Activation.RELU).seed(seed)
                .build();
        List<Model> heads = new ArrayList<>();
        for (int t = 0; t < tasks.size(); t++) {
            heads.add(Model.sequential().input(64).dense(tasks.get(t).outputs(), Activation.IDENTITY)
                    .seed(seed + 10 + t).build());
        }
        return new Net(trunk, heads);
    }

    private static Var taskLoss(Task task, Var out, float[] target) {
        Tensor y = Tensor.of(Shape.of(target.length), target);
        if (task.kind() == Kind.CLASSES) {
            return Functional.crossEntropy(out, y);
        }
        return out.reshape(target.length).sub(y).square().mean();
    }

    private static int[] sample(int n, int batch, SplittableRandom r) {
        int[] idx = new int[batch];
        for (int i = 0; i < batch; i++) {
            idx[i] = r.nextInt(n);
        }
        return idx;
    }

    private static Tensor rows(Tensor x, int[] idx) {
        int d = x.shape().dim(1);
        float[] v = x.toArray();
        float[] out = new float[idx.length * d];
        for (int i = 0; i < idx.length; i++) {
            System.arraycopy(v, idx[i] * d, out, i * d, d);
        }
        return Tensor.of(Shape.of(idx.length, d), out);
    }

    private static float[] pick(float[] y, int[] idx) {
        float[] out = new float[idx.length];
        for (int i = 0; i < idx.length; i++) {
            out[i] = y[idx[i]];
        }
        return out;
    }

    private static double metric(Task task, Var out, float[] y) {
        float[] o = out.value().toArray();
        int n = y.length;
        if (task.kind() == Kind.REGRESSION) {
            double mse = 0;
            for (int i = 0; i < n; i++) {
                mse += (o[i] - y[i]) * (o[i] - y[i]) / n;
            }
            return mse;
        }
        int k = task.outputs();
        if (task.auc()) {
            double[] score = new double[n];
            boolean[] pos = new boolean[n];
            for (int i = 0; i < n; i++) {
                score[i] = o[i * k + 1] - o[i * k];
                pos[i] = y[i] > 0.5f;
            }
            return MlMetrics.auroc(score, pos);
        }
        int hit = 0;
        for (int i = 0; i < n; i++) {
            int best = 0;
            for (int c = 1; c < k; c++) {
                best = o[i * k + c] > o[i * k + best] ? c : best;
            }
            hit += best == Math.round(y[i]) ? 1 : 0;
        }
        return (double) hit / n;
    }

    private static boolean higherIsBetter(Task task) {
        return task.kind() == Kind.CLASSES;
    }

    /**
     * Trains one single-task model per task: the reference of {@code Delta_m}.
     *
     * @param problem the problem
     * @param steps steps per model
     * @param seed the seed
     * @return the per-task test metric
     */
    public static double[] singleTask(Problem problem, int steps, long seed) {
        Data data = data(problem, seed);
        int in = data.train().x().shape().dim(1);
        double[] out = new double[data.tasks().size()];
        for (int t = 0; t < data.tasks().size(); t++) {
            Task task = data.tasks().get(t);
            Net net = net(in, List.of(task), seed + 100 * t);
            Optimizer opt = Optimizer.adam(3e-3f);
            SplittableRandom r = new SplittableRandom(seed + 5);
            int n = data.train().x().shape().dim(0);
            for (int s = 0; s < steps; s++) {
                int[] idx = sample(n, 64, r);
                Var h = net.trunk().forward(Var.of(rows(data.train().x(), idx)));
                taskLoss(task, net.heads().get(0).forward(h), pick(data.train().y()[t], idx)).backward();
                opt.step(net.parameters());
            }
            Var o = Var.noGrad(() -> net.heads().get(0).forward(net.trunk().forward(Var.of(data.test().x()))));
            out[t] = metric(task, o, data.test().y()[t]);
        }
        return out;
    }

    /**
     * Trains the shared network with one balancing method.
     *
     * @param problem the problem
     * @param method the method
     * @param steps training steps of 64 rows
     * @param seed the seed
     * @param single the single-task metrics from {@link #singleTask}
     * @return the result
     */
    public static Result run(Problem problem, Method method, int steps, long seed, double[] single) {
        Data data = data(problem, seed);
        List<Task> tasks = data.tasks();
        int tc = tasks.size();
        int in = data.train().x().shape().dim(1);
        Net net = net(in, tasks, seed);
        List<Var> shared = net.trunk().parameters();
        List<Var> lastShared = shared.subList(shared.size() - 2, shared.size());
        LossWeighting weighting = LossWeighting.of(switch (method) {
            case UNCERTAINTY -> LossWeighting.Kind.UNCERTAINTY;
            case GRADNORM -> LossWeighting.Kind.GRADNORM;
            case DWA -> LossWeighting.Kind.DWA;
            default -> LossWeighting.Kind.EQUAL;
        }, tc);
        GradientSurgery.GradVac gradVac = new GradientSurgery.GradVac(tc, 0.01);
        Famo famo = new Famo(tc, 0.025, 1e-3);
        List<Var> params = net.parameters();
        params.addAll(weighting.parameters());
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 5);
        int n = data.train().x().shape().dim(0);
        double conflicts = 0;
        int conflictSteps = 0;
        long start = System.nanoTime();
        for (int s = 0; s < steps; s++) {
            int[] idx = sample(n, 64, r);
            Tensor x = rows(data.train().x(), idx);
            float[][] y = new float[tc][];
            for (int t = 0; t < tc; t++) {
                y[t] = pick(data.train().y()[t], idx);
            }
            Var h = net.trunk().forward(Var.of(x));
            List<Var> losses = new ArrayList<>();
            for (int t = 0; t < tc; t++) {
                losses.add(taskLoss(tasks.get(t), net.heads().get(t).forward(h), y[t]));
            }
            switch (method) {
                case EQUAL, UNCERTAINTY, GRADNORM, DWA -> {
                    weighting.combine(losses, lastShared).backward();
                    opt.step(params);
                }
                case FAMO -> {
                    famo.weightedLoss(losses).backward();
                    opt.step(params);
                    double[] after = new double[tc];
                    Var.noGrad(() -> {
                        Var h2 = net.trunk().forward(Var.of(x));
                        for (int t = 0; t < tc; t++) {
                            after[t] = taskLoss(tasks.get(t), net.heads().get(t).forward(h2), y[t]).value().item();
                        }
                        return h2;
                    });
                    famo.update(after);
                }
                case MGDA_UB -> {
                    double[][] g = new double[tc][];
                    for (int t = 0; t < tc; t++) {
                        g[t] = TaskGradients.flatten(Var.gradients(losses.get(t), List.of(h), false));
                    }
                    conflicts += TaskGradients.conflictShare(g);
                    conflictSteps++;
                    double[] w = ParetoDescent.mgdaWeights(g);
                    Var total = null;
                    for (int t = 0; t < tc; t++) {
                        Var term = losses.get(t).mul((float) (w[t] * tc));
                        total = total == null ? term : total.add(term);
                    }
                    total.backward();
                    opt.step(params);
                }
                default -> {
                    double[][] g = TaskGradients.of(losses, shared);
                    conflicts += TaskGradients.conflictShare(g);
                    conflictSteps++;
                    double[] direction = switch (method) {
                        case PCGRAD -> GradientSurgery.pcGrad(g, r);
                        case GRADVAC -> gradVac.combine(g, r);
                        case GRADOPS -> GradientSurgery.gradOps(g, 0);
                        case MGDA -> ParetoDescent.mgda(g);
                        case CAGRAD -> ParetoDescent.caGrad(g, 0.4);
                        case NASH_MTL -> ParetoDescent.nashMtl(g);
                        default -> throw new IllegalStateException(method.toString());
                    };
                    Var total = losses.get(0);
                    for (int t = 1; t < tc; t++) {
                        total = total.add(losses.get(t));
                    }
                    total.backward();
                    TaskGradients.assign(direction, shared);
                    opt.step(params);
                }
            }
        }
        double millis = (System.nanoTime() - start) / 1e6 / Math.max(1, steps);
        Var ht = Var.noGrad(() -> net.trunk().forward(Var.of(data.test().x())));
        double[] metrics = new double[tc];
        for (int t = 0; t < tc; t++) {
            int task = t;
            Var o = Var.noGrad(() -> net.heads().get(task).forward(ht));
            metrics[t] = metric(tasks.get(t), o, data.test().y()[t]);
        }
        return new Result(problem, method.name(), metrics, deltaM(tasks, metrics, single),
                conflictSteps == 0 ? Double.NaN : conflicts / conflictSteps, millis);
    }

    private static double deltaM(List<Task> tasks, double[] m, double[] base) {
        double d = 0;
        for (int t = 0; t < m.length; t++) {
            double rel = (m[t] - base[t]) / base[t];
            d += (higherIsBetter(tasks.get(t)) ? rel : -rel) / m.length;
        }
        return 100 * d;
    }

    /**
     * Wave 9's MMoE (the architectural answer to task conflict) on the click problem, with
     * equal or uncertainty weights.
     *
     * @param kind the loss weighting (EQUAL or UNCERTAINTY)
     * @param steps training steps of 64 rows
     * @param seed the seed
     * @param single the single-task metrics
     * @return the result
     */
    public static Result mmoe(LossWeighting.Kind kind, int steps, long seed, double[] single) {
        Data data = clicks(seed);
        int in = data.train().x().shape().dim(1);
        MultiGateExperts m = new MultiGateExperts(MultiGateExperts.Kind.MMOE, in, 2, 4, 32, 16, 1,
                new SplittableRandom(seed));
        LossWeighting weighting = LossWeighting.of(kind, 2);
        List<Var> params = new ArrayList<>(m.parameters());
        params.addAll(weighting.parameters());
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 5);
        int n = data.train().x().shape().dim(0);
        long start = System.nanoTime();
        for (int s = 0; s < steps; s++) {
            int[] idx = sample(n, 64, r);
            List<Var> reps = m.taskRepresentation(Var.of(rows(data.train().x(), idx)));
            List<Var> losses = new ArrayList<>();
            for (int t = 0; t < 2; t++) {
                Var logit = m.tower(t, reps.get(t)).reshape(idx.length);
                losses.add(bce(logit, pick(data.train().y()[t], idx)));
            }
            weighting.combine(losses, List.of()).backward();
            opt.step(params);
        }
        double millis = (System.nanoTime() - start) / 1e6 / Math.max(1, steps);
        List<Var> reps = Var.noGrad(() -> m.taskRepresentation(Var.of(data.test().x())));
        double[] metrics = new double[2];
        for (int t = 0; t < 2; t++) {
            int task = t;
            float[] o = Var.noGrad(() -> m.tower(task, reps.get(task))).value().toArray();
            double[] score = new double[o.length];
            boolean[] pos = new boolean[o.length];
            for (int i = 0; i < o.length; i++) {
                score[i] = o[i];
                pos[i] = data.test().y()[t][i] > 0.5f;
            }
            metrics[t] = MlMetrics.auroc(score, pos);
        }
        return new Result(Problem.CLICKS, "MMoE+" + kind, metrics, deltaM(data.tasks(), metrics, single), Double.NaN,
                millis);
    }

    private static Var bce(Var logit, float[] y) {
        Tensor t = Tensor.of(Shape.of(y.length), y);
        // log(1 + e^x) - y x
        return logit.softplus().sub(logit.mul(t)).mean();
    }

    // ------------------------------------------------------------------ toy surface

    /**
     * The final point on the two-objective toy surface.
     *
     * @param method the method
     * @param f1 the first objective at the end, averaged over the starts
     * @param f2 the second objective
     * @param reached how many of the five starts ended on (within 1 of) the Pareto set, the
     *     segment {@code x2 = -8}, {@code |x1| <= 7}
     */
    public record ToyResult(String method, double f1, double f2, int reached) {
    }

    /** The starting points of the CAGrad toy. */
    private static final double[][] STARTS = {{-8.5, 7.5}, {-8.5, -5}, {9, 9}, {-7.5, -0.5}, {9, -1}};

    private static Var[] objectives(Var x) {
        Var x1 = x.slice(0, 0, 1).sum();
        Var x2 = x.slice(0, 1, 2).sum();
        Var t2 = x2.neg().tanh();
        Var f1 = x1.neg().sub(7).mul(0.5f).sub(t2).abs().maximum(5e-6f).log().add(6);
        Var f2 = x1.neg().add(3).mul(0.5f).add(t2).add(2).abs().maximum(5e-6f).log().add(6);
        Var c1 = x2.mul(0.5f).tanh().relu();
        Var c2 = x2.mul(-0.5f).tanh().relu();
        Var tail = x2.neg().sub(8).square().mul(0.1f);
        Var f1sq = x1.neg().add(7).square().add(tail).div(10).sub(20);
        Var f2sq = x1.neg().sub(7).square().add(tail).div(10).sub(20);
        return new Var[] {f1.mul(c1).add(f1sq.mul(c2)), f2.mul(c1).add(f2sq.mul(c2))};
    }

    /**
     * Runs a gradient-combination method on the toy surface from five starts with Adam.
     *
     * @param method EQUAL, PCGRAD, GRADOPS, MGDA, CAGRAD or NASH_MTL
     * @param steps Adam steps per start
     * @param learningRate Adam's learning rate
     * @return the mean final objectives
     */
    public static ToyResult toySurface(Method method, int steps, double learningRate) {
        double f1 = 0;
        double f2 = 0;
        int reached = 0;
        SplittableRandom r = new SplittableRandom(1);
        for (double[] s : STARTS) {
            Var x = Var.param(Tensor.of(Shape.of(2), (float) s[0], (float) s[1]));
            double[] m = new double[2];
            double[] v = new double[2];
            for (int k = 1; k <= steps; k++) {
                Var[] f = objectives(x);
                double[][] g = {TaskGradients.flatten(Var.gradients(f[0], List.of(x), false)),
                        TaskGradients.flatten(Var.gradients(f[1], List.of(x), false))};
                double[] d = switch (method) {
                    case EQUAL -> TaskGradients.combine(g, new double[] {1, 1});
                    case PCGRAD -> GradientSurgery.pcGrad(g, r);
                    case GRADOPS -> GradientSurgery.gradOps(g, 0);
                    case MGDA -> ParetoDescent.mgda(g);
                    case CAGRAD -> ParetoDescent.caGrad(g, 0.5);
                    case NASH_MTL -> ParetoDescent.nashMtl(g);
                    default -> throw new IllegalArgumentException("not a gradient method on the toy: " + method);
                };
                float[] xv = x.value().toArray();
                for (int i = 0; i < 2; i++) {
                    m[i] = 0.9 * m[i] + 0.1 * d[i];
                    v[i] = 0.999 * v[i] + 0.001 * d[i] * d[i];
                    double mh = m[i] / (1 - Math.pow(0.9, k));
                    double vh = v[i] / (1 - Math.pow(0.999, k));
                    xv[i] -= (float) (learningRate * mh / (Math.sqrt(vh) + 1e-8));
                }
                x.assign(Tensor.of(Shape.of(2), xv));
            }
            Var[] f = objectives(x);
            double a = f[0].value().item();
            double b = f[1].value().item();
            f1 += a / STARTS.length;
            f2 += b / STARTS.length;
            float[] end = x.value().toArray();
            reached += Math.abs(end[1] + 8) < 1 && Math.abs(end[0]) <= 7.5 ? 1 : 0;
        }
        return new ToyResult(method.name(), f1, f2, reached);
    }

    /**
     * The long-budget comparison (a Run): every method on every problem with 1500 steps and
     * three seeds, plus MMoE and the toy surface at 20000 steps.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        for (Problem p : Problem.values()) {
            for (long seed = 1; seed <= 3; seed++) {
                double[] single = singleTask(p, 1500, seed);
                System.out.println(p + " seed " + seed + " single-task " + Arrays.toString(single));
                for (Method m : Method.values()) {
                    System.out.println(run(p, m, 1500, seed, single));
                }
                if (p == Problem.CLICKS) {
                    System.out.println(mmoe(LossWeighting.Kind.EQUAL, 1500, seed, single));
                    System.out.println(mmoe(LossWeighting.Kind.UNCERTAINTY, 1500, seed, single));
                }
            }
        }
        for (Method m : List.of(Method.EQUAL, Method.PCGRAD, Method.GRADOPS, Method.MGDA, Method.CAGRAD,
                Method.NASH_MTL)) {
            System.out.println(toySurface(m, 20000, 0.002));
        }
    }
}
