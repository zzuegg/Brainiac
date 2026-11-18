package io.github.zzuegg.brainiac.experiments.multitask;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ClickLogs;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.experimental.multitask.GradientSurgery;
import io.github.zzuegg.brainiac.experimental.multitask.SoftSharing;
import io.github.zzuegg.brainiac.experimental.multitask.TaskGradients;
import io.github.zzuegg.brainiac.experimental.recsys.MultiGateExperts;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Wave 13 item 13.6.16: soft parameter sharing ({@link SoftSharing}: cross-stitch units, sluice
 * networks, MTAN-style task attention) against hard sharing, separate single-task columns,
 * {@link MultiGateExperts} (MMoE), and hard sharing with PCGrad ({@link GradientSurgery}, the loss-side
 * fix), at roughly matched parameter counts, on related tasks (toy digits: class, parity and the
 * digit value as a regression) and conflicting ones (the two click labels of
 * {@link ClickLogs#multiTask} at interaction correlation {@code -0.5}, the problem of
 * {@code GradientBalancing}). The score is {@code Delta_MTL}, the average relative change of each
 * task's metric against the separate columns, in percent (sign-flipped for errors).
 */
public final class SharingArchitectures {

    private SharingArchitectures() {
    }

    /** A task: classes (2 or more outputs, scored by accuracy or AUC) or a regression (MSE). */
    private record Task(String name, int outputs, boolean auc) {
    }

    private record Split(Tensor x, float[][] y) {
    }

    private record Problem(List<Task> tasks, Split train, Split test) {
    }

    private static Problem problem(String name, long seed) {
        if (name.equals("related")) {
            ToyTasks.Data tr = ToyTasks.digits(10, 60, 8, 0.15, seed, seed * 31 + 1);
            ToyTasks.Data te = ToyTasks.digits(10, 30, 8, 0.15, seed, seed * 31 + 2);
            List<Task> tasks = List.of(new Task("class", 10, false), new Task("parity", 2, false),
                    new Task("value", 1, false));
            return new Problem(tasks, digitSplit(tr), digitSplit(te));
        }
        ClickLogs.MultiTaskTable t = ClickLogs.multiTask(4000, 3, 8, -0.5, seed);
        List<Task> tasks = List.of(new Task("click", 2, true), new Task("other", 2, true));
        return new Problem(tasks, clickSplit(t.rows(0, 3000)), clickSplit(t.rows(3000, 4000)));
    }

    private static Split digitSplit(ToyTasks.Data d) {
        float[][] y = new float[3][d.size()];
        for (int i = 0; i < d.size(); i++) {
            y[0][i] = d.y()[i];
            y[1][i] = d.y()[i] % 2;
            y[2][i] = (d.y()[i] - 4.5f) / 3;
        }
        return new Split(d.inputs(), y);
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

    private static Var loss(Task task, Var out, float[] y) {
        Tensor t = Tensor.of(Shape.of(y.length), y);
        if (task.outputs() == 1) {
            return out.reshape(y.length).sub(t).square().mean();
        }
        return Functional.crossEntropy(out, t);
    }

    private static double metric(Task task, Var out, float[] y) {
        float[] o = out.value().toArray();
        int n = y.length;
        int k = task.outputs();
        if (k == 1) {
            double mse = 0;
            for (int i = 0; i < n; i++) {
                mse += (o[i] - y[i]) * (o[i] - y[i]) / n;
            }
            return mse;
        }
        if (task.auc()) {
            double[] s = new double[n];
            boolean[] pos = new boolean[n];
            for (int i = 0; i < n; i++) {
                s[i] = o[i * k + 1] - o[i * k];
                pos[i] = y[i] > 0.5f;
            }
            return MlMetrics.auroc(s, pos);
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

    /** A network under test: task outputs, all parameters, the parameters gradient surgery acts on. */
    private record Net(Function<Var, List<Var>> outputs, List<Var> parameters, List<Var> shared, int size,
            SoftSharing soft) {
    }

    private static Net soft(SoftSharing.Kind kind, int in, List<Task> tasks, int hidden, long seed) {
        int[] outs = tasks.stream().mapToInt(Task::outputs).toArray();
        SoftSharing s = new SoftSharing(kind, in, outs, hidden, 2, new SplittableRandom(seed));
        return new Net(s::taskOutputs, s.parameters(), s.sharedParameters(), s.parameterCount(), s);
    }

    /** The width of a one-trunk architecture whose parameter count is closest to {@code target}. */
    private static int matchedWidth(SoftSharing.Kind kind, int in, List<Task> tasks, int target) {
        int[] outs = tasks.stream().mapToInt(Task::outputs).toArray();
        int best = 8;
        int gap = Integer.MAX_VALUE;
        for (int w = 8; w <= 256; w += 2) {
            int p = new SoftSharing(kind, in, outs, w, 2, new SplittableRandom(0)).parameterCount();
            if (Math.abs(p - target) < gap) {
                gap = Math.abs(p - target);
                best = w;
            }
        }
        return best;
    }

    private static Net mmoe(int in, List<Task> tasks, int hidden, int target, long seed) {
        // pick the expert count whose size is closest to the target
        MultiGateExperts best = null;
        int gap = Integer.MAX_VALUE;
        for (int e = 2; e <= 8; e++) {
            MultiGateExperts m = new MultiGateExperts(MultiGateExperts.Kind.MMOE, in, tasks.size(), e, hidden, hidden,
                    1, new SplittableRandom(seed));
            int size = 0;
            for (Var v : m.parameters()) {
                size += (int) v.value().shape().size();
            }
            if (Math.abs(size - target) < gap) {
                gap = Math.abs(size - target);
                best = m;
            }
        }
        MultiGateExperts m = best;
        SplittableRandom r = new SplittableRandom(seed + 1);
        List<Var> heads = new ArrayList<>();
        for (Task t : tasks) {
            float[] w = new float[hidden * t.outputs()];
            for (int i = 0; i < w.length; i++) {
                w[i] = (float) (Math.sqrt(2.0 / hidden) * r.nextGaussian());
            }
            heads.add(Var.param(Tensor.of(Shape.of(hidden, t.outputs()), w)));
            heads.add(Var.param(Tensor.zeros(Shape.of(t.outputs()))));
        }
        List<Var> params = new ArrayList<>(m.parameters());
        params.addAll(heads);
        int size = 0;
        for (Var v : params) {
            size += (int) v.value().shape().size();
        }
        Function<Var, List<Var>> f = x -> {
            List<Var> reps = m.taskRepresentation(x);
            List<Var> out = new ArrayList<>();
            for (int t = 0; t < tasks.size(); t++) {
                out.add(reps.get(t).matmul(heads.get(2 * t)).add(heads.get(2 * t + 1)));
            }
            return out;
        };
        return new Net(f, params, m.parameters(), size, null);
    }

    /**
     * Runs every architecture on one problem.
     *
     * @param name {@code "related"} (digits: class, parity, value) or {@code "conflicting"} (clicks at
     *     correlation -0.5)
     * @param hidden the column width of the column architectures (one-trunk ones are widened to a
     *     matched parameter count)
     * @param steps Adam steps of 64 rows
     * @param seed the seed
     * @return architecture to {Delta_MTL in percent, parameters, per-task metrics...}; for the
     *     column-mixing kinds the last two entries are the own-task share of the mixing units in the
     *     first and the last layer
     */
    public static Map<String, double[]> run(String name, int hidden, int steps, long seed) {
        Problem p = problem(name, seed);
        int in = p.train().x().shape().dim(1);
        List<Task> tasks = p.tasks();
        int target = soft(SoftSharing.Kind.CROSS_STITCH, in, tasks, hidden, seed).size();
        int hardWidth = matchedWidth(SoftSharing.Kind.HARD, in, tasks, target);
        int attWidth = matchedWidth(SoftSharing.Kind.TASK_ATTENTION, in, tasks, target);
        Map<String, Net> nets = new LinkedHashMap<>();
        nets.put("separate columns (single-task)", soft(SoftSharing.Kind.SEPARATE, in, tasks, hidden, seed));
        nets.put("hard sharing (width " + hardWidth + ")", soft(SoftSharing.Kind.HARD, in, tasks, hardWidth, seed));
        nets.put("hard sharing + PCGrad", soft(SoftSharing.Kind.HARD, in, tasks, hardWidth, seed));
        nets.put("cross-stitch", soft(SoftSharing.Kind.CROSS_STITCH, in, tasks, hidden, seed));
        nets.put("cross-stitch + PCGrad", soft(SoftSharing.Kind.CROSS_STITCH, in, tasks, hidden, seed));
        nets.put("sluice", soft(SoftSharing.Kind.SLUICE, in, tasks, hidden, seed));
        nets.put("MTAN task attention (width " + attWidth + ")",
                soft(SoftSharing.Kind.TASK_ATTENTION, in, tasks, attWidth, seed));
        nets.put("MMoE", mmoe(in, tasks, hidden, target, seed));
        Map<String, double[]> out = new LinkedHashMap<>();
        double[] single = null;
        for (Map.Entry<String, Net> e : nets.entrySet()) {
            Net net = e.getValue();
            boolean pcgrad = e.getKey().contains("PCGrad");
            double[] m = train(net, p, steps, pcgrad, seed);
            if (single == null) {
                single = m;
            }
            double delta = 0;
            for (int t = 0; t < m.length; t++) {
                double rel = (m[t] - single[t]) / single[t];
                delta += (tasks.get(t).outputs() == 1 ? -rel : rel) / m.length * 100;
            }
            boolean mixing = net.soft() != null && (net.soft().kind() == SoftSharing.Kind.CROSS_STITCH
                    || net.soft().kind() == SoftSharing.Kind.SLUICE);
            double[] row = new double[2 + m.length + (mixing ? 2 : 0)];
            row[0] = delta;
            row[1] = net.size();
            System.arraycopy(m, 0, row, 2, m.length);
            if (mixing) {
                double[] share = net.soft().selfShare();
                row[2 + m.length] = share[0];
                row[3 + m.length] = share[share.length - 1];
            }
            out.put(e.getKey(), row);
        }
        return out;
    }

    private static double[] train(Net net, Problem p, int steps, boolean pcgrad, long seed) {
        List<Task> tasks = p.tasks();
        int tc = tasks.size();
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 5);
        int n = p.train().x().shape().dim(0);
        for (int s = 0; s < steps; s++) {
            int[] idx = new int[64];
            for (int i = 0; i < idx.length; i++) {
                idx[i] = r.nextInt(n);
            }
            List<Var> outs = net.outputs().apply(Var.of(rows(p.train().x(), idx)));
            List<Var> losses = new ArrayList<>();
            for (int t = 0; t < tc; t++) {
                losses.add(loss(tasks.get(t), outs.get(t), pick(p.train().y()[t], idx)));
            }
            Var total = losses.get(0);
            for (int t = 1; t < tc; t++) {
                total = total.add(losses.get(t));
            }
            if (net.soft() != null && net.soft().auxiliaryLoss() != null) {
                total = total.add(net.soft().auxiliaryLoss().mul(1e-3f));
            }
            if (pcgrad) {
                double[][] g = TaskGradients.of(losses, net.shared());
                double[] direction = GradientSurgery.pcGrad(g, r);
                total.backward();
                TaskGradients.assign(direction, net.shared());
            } else {
                total.backward();
            }
            opt.step(net.parameters());
        }
        List<Var> outs = Var.noGrad(() -> net.outputs().apply(Var.of(p.test().x())));
        double[] m = new double[tc];
        for (int t = 0; t < tc; t++) {
            m[t] = metric(tasks.get(t), outs.get(t), p.test().y()[t]);
        }
        return m;
    }
}
