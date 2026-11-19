package io.github.zzuegg.brainiac.experimental.symbolic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GinLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.GradientBoostedTrees;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Generalised planners that execute without search, trained on optimal plans of small
 * instances and run on larger ones.
 *
 * <ul>
 *   <li>{@link Mode#TRANSITION_MODEL}: a goal-conditioned learned transition model (Gupta et
 *       al., ICAPS 2026, <a href="https://arxiv.org/abs/2602.23148">arXiv:2602.23148</a>). The
 *       state is embedded by {@link WlFeatures} of its instance learning graph (which carries the
 *       goal), a regressor predicts the residual {@code phi(s') - phi(s)} of the next state on an
 *       optimal plan ("delta mode", ridge or one core {@link GradientBoostedTrees} model per
 *       embedding dimension, the paper's XGBoost), and the predicted embedding is snapped to the
 *       nearest valid successor among the applicable actions' results, so every step is
 *       symbolically valid.</li>
 *   <li>{@link Mode#GNN_POLICY}: a learned general policy (in the spirit of Ståhlberg, Bonet &amp;
 *       Geffner, KR 2022): a {@link GinLayer} network scores every successor state and the policy
 *       follows the best one; it is trained by softmax cross-entropy towards the optimal plan's
 *       successor.</li>
 *   <li>{@link Mode#RANDOM}: random applicable actions (the floor).</li>
 * </ul>
 * Rollouts stop at the goal or after a step cap. Unlike the paper's single greedy rollout,
 * already visited states are skipped when another successor exists (simple loop avoidance).
 */
public final class GeneralizedPlanner {

    /** The planner. */
    public enum Mode {
        /** WL residual transition model with nearest-successor snapping. */
        TRANSITION_MODEL,
        /** GNN successor-scoring policy. */
        GNN_POLICY,
        /** Uniformly random applicable actions. */
        RANDOM
    }

    /** The transition model's regressor. */
    public enum Regressor {
        /** Multi-output ridge regression. */
        RIDGE,
        /** One gradient-boosted tree ensemble per embedding dimension. */
        BOOSTED_TREES
    }

    /**
     * A rollout.
     *
     * @param plan the actions taken
     * @param solved whether the goal was reached
     * @param millis wall time
     */
    public record Rollout(List<StripsTask.Action> plan, boolean solved, double millis) {
    }

    private final Mode mode;
    private WlFeatures wl;
    private double[][] ridge;
    private GradientBoostedTrees[] trees;
    private GinLayer gin1;
    private GinLayer gin2;
    private Var head;
    private final Map<Long, Integer> colours = new HashMap<>();
    private final SplittableRandom random;
    private double trainMillis;

    private GeneralizedPlanner(Mode mode, long seed) {
        this.mode = mode;
        this.random = new SplittableRandom(seed);
    }

    /** @return the training time in milliseconds */
    public double trainMillis() {
        return trainMillis;
    }

    /**
     * The random baseline.
     *
     * @param seed the seed
     * @return the planner
     */
    public static GeneralizedPlanner random(long seed) {
        return new GeneralizedPlanner(Mode.RANDOM, seed);
    }

    /**
     * Trains the transition model.
     *
     * @param samples states of optimal plans
     * @param iterations WL rounds
     * @param regressor ridge or boosted trees
     * @param seed the seed
     * @return the planner
     */
    public static GeneralizedPlanner transitionModel(List<LearnedPlanningHeuristic.Sample> samples, int iterations,
            Regressor regressor, long seed) {
        long t0 = System.nanoTime();
        GeneralizedPlanner p = new GeneralizedPlanner(Mode.TRANSITION_MODEL, seed);
        p.wl = new WlFeatures(iterations);
        List<double[]> xs = new ArrayList<>();
        List<double[]> ys = new ArrayList<>();
        for (LearnedPlanningHeuristic.Sample s : samples) {
            if (s.next() == null) {
                continue;
            }
            xs.add(p.wl.features(s.task(), s.state(), true));
            ys.add(p.wl.features(s.task(), s.next(), true));
            for (StripsTask.Action a : s.task().applicable(s.state())) {
                p.wl.features(s.task(), a.apply(s.state()), true); // grow the vocabulary over successors too
            }
        }
        p.wl.freeze();
        double[][] x = p.wl.pad(xs);
        double[][] y = p.wl.pad(ys);
        int d = p.wl.dimension();
        double[][] delta = new double[x.length][d];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < d; j++) {
                delta[i][j] = y[i][j] - x[i][j];
            }
        }
        if (regressor == Regressor.RIDGE) {
            // one factorisation for all outputs: W = (X'X + lambda I)^-1 X' Delta
            double[][] a = new double[d + 1][d + 1];
            double[][] xty = new double[d + 1][d];
            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j <= d; j++) {
                    double xj = j < d ? x[i][j] : 1;
                    if (xj == 0) {
                        continue;
                    }
                    for (int k = 0; k <= d; k++) {
                        a[j][k] += xj * (k < d ? x[i][k] : 1);
                    }
                    for (int k = 0; k < d; k++) {
                        xty[j][k] += xj * delta[i][k];
                    }
                }
            }
            for (int j = 0; j < d; j++) {
                a[j][j] += 1e-1;
            }
            a[d][d] += 1e-9;
            double[][] inv = io.github.zzuegg.brainiac.tensor.Matrices.inverse(a);
            p.ridge = new double[d][d + 1];
            for (int out = 0; out < d; out++) {
                for (int j = 0; j <= d; j++) {
                    double v = 0;
                    for (int k = 0; k <= d; k++) {
                        v += inv[j][k] * xty[k][out];
                    }
                    p.ridge[out][j] = v;
                }
            }
        } else {
            p.trees = new GradientBoostedTrees[d];
            GradientBoostedTrees.Settings st = GradientBoostedTrees.Settings.defaults().withTrees(15).withDepth(3);
            for (int j = 0; j < d; j++) {
                double[] col = new double[x.length];
                boolean constant = true;
                for (int i = 0; i < x.length; i++) {
                    col[i] = delta[i][j];
                    constant &= col[i] == col[0];
                }
                p.trees[j] = constant ? null : GradientBoostedTrees.fit(x, col, GradientBoostedTrees.Objective.REGRESSION,
                    st);
            }
        }
        p.trainMillis = (System.nanoTime() - t0) / 1e6;
        return p;
    }

    /**
     * Trains the GNN policy.
     *
     * @param samples states of optimal plans
     * @param epochs passes over the samples
     * @param seed the seed
     * @return the planner
     */
    public static GeneralizedPlanner gnnPolicy(List<LearnedPlanningHeuristic.Sample> samples, int epochs, long seed) {
        long t0 = System.nanoTime();
        GeneralizedPlanner p = new GeneralizedPlanner(Mode.GNN_POLICY, seed);
        List<List<WlFeatures.Graph>> groups = new ArrayList<>();
        List<Integer> targets = new ArrayList<>();
        for (LearnedPlanningHeuristic.Sample s : samples) {
            if (s.next() == null) {
                continue;
            }
            List<WlFeatures.Graph> g = new ArrayList<>();
            int target = -1;
            for (StripsTask.Action a : s.task().applicable(s.state())) {
                BitSet t = a.apply(s.state());
                if (t.equals(s.next())) {
                    target = g.size();
                }
                WlFeatures.Graph gr = WlFeatures.graph(s.task(), t, null);
                for (long c : gr.colours()) {
                    p.colours.putIfAbsent(c, p.colours.size());
                }
                g.add(gr);
            }
            if (target >= 0 && g.size() > 1) {
                groups.add(g);
                targets.add(target);
            }
        }
        int width = p.colours.size();
        int hidden = 16;
        SplittableRandom r = new SplittableRandom(seed);
        p.gin1 = new GinLayer(width, hidden, hidden, r);
        p.gin2 = new GinLayer(hidden, hidden, hidden, r);
        p.head = Var.param(Tensor.normal(Shape.of(hidden, 1), 0, 0.1f, r));
        List<Var> params = new ArrayList<>(p.gin1.parameters());
        params.addAll(p.gin2.parameters());
        params.add(p.head);
        Optimizer opt = Optimizer.adam(5e-3f);
        for (int e = 0; e < epochs; e++) {
            for (int i = 0; i < groups.size(); i++) {
                Var scores = p.scores(groups.get(i)).reshape(1, groups.get(i).size());
                Var logp = scores.neg().logSoftmax(1);
                float[] onehot = new float[groups.get(i).size()];
                onehot[targets.get(i)] = 1;
                Var loss = logp.mul(Var.of(Tensor.of(Shape.of(1, onehot.length), onehot))).sum().neg();
                loss.backward();
                opt.step(params);
            }
        }
        p.trainMillis = (System.nanoTime() - t0) / 1e6;
        return p;
    }

    /** Scores graphs (lower is better), one block-diagonal batch. */
    private Var scores(List<WlFeatures.Graph> graphs) {
        int width = colours.size();
        int n = 0;
        for (WlFeatures.Graph g : graphs) {
            n += g.nodes();
        }
        float[] x = new float[n * width];
        float[] a = new float[n * n];
        float[] m = new float[graphs.size() * n];
        int off = 0;
        for (int b = 0; b < graphs.size(); b++) {
            WlFeatures.Graph g = graphs.get(b);
            for (int v = 0; v < g.nodes(); v++) {
                Integer c = colours.get(g.colours()[v]);
                if (c != null) {
                    x[(off + v) * width + c] = 1;
                }
                m[b * n + off + v] = 1;
                int[] e = g.edges()[v];
                for (int k = 0; k < e.length; k += 2) {
                    a[(off + v) * n + off + e[k]] = 1;
                }
            }
            off += g.nodes();
        }
        Var adj = Var.of(Tensor.of(Shape.of(n, n), a));
        Var h = gin1.forward(List.of(Var.of(Tensor.of(Shape.of(n, width), x)), adj), true).relu();
        h = gin2.forward(List.of(h, adj), true).relu();
        return Var.of(Tensor.of(Shape.of(graphs.size(), n), m)).matmul(h).matmul(head);
    }

    private double[] predictNext(double[] x) {
        int d = wl.dimension();
        double[] out = new double[d];
        for (int j = 0; j < d; j++) {
            double delta;
            if (ridge != null) {
                double[] w = ridge[j];
                delta = w[d];
                for (int k = 0; k < d; k++) {
                    delta += w[k] * x[k];
                }
            } else {
                delta = trees[j] == null ? 0 : trees[j].predict(x);
            }
            out[j] = x[j] + delta;
        }
        return out;
    }

    /**
     * Executes the planner on a task.
     *
     * @param task the task
     * @param maxSteps the step cap
     * @return the rollout
     */
    public Rollout run(StripsTask task, int maxSteps) {
        long t0 = System.nanoTime();
        BitSet s = task.initialState();
        Set<BitSet> visited = new HashSet<>();
        visited.add(s);
        List<StripsTask.Action> plan = new ArrayList<>();
        for (int step = 0; step < maxSteps && !task.isGoal(s); step++) {
            List<StripsTask.Action> app = task.applicable(s);
            List<StripsTask.Action> fresh = new ArrayList<>();
            for (StripsTask.Action a : app) {
                if (!visited.contains(a.apply(s))) {
                    fresh.add(a);
                }
            }
            List<StripsTask.Action> cand = fresh.isEmpty() ? app : fresh;
            if (cand.isEmpty()) {
                break;
            }
            StripsTask.Action pick = switch (mode) {
                case RANDOM -> cand.get(random.nextInt(cand.size()));
                case TRANSITION_MODEL -> {
                    double[] target = predictNext(java.util.Arrays.copyOf(wl.features(task, s, false),
                        wl.dimension()));
                    StripsTask.Action best = null;
                    double bd = Double.POSITIVE_INFINITY;
                    for (StripsTask.Action a : cand) {
                        BitSet t = a.apply(s);
                        if (task.isGoal(t)) {
                            best = a;
                            break;
                        }
                        double[] e = wl.features(task, t, false);
                        double dist = 0;
                        for (int j = 0; j < target.length; j++) {
                            double v = (j < e.length ? e[j] : 0) - target[j];
                            dist += v * v;
                        }
                        if (dist < bd) {
                            bd = dist;
                            best = a;
                        }
                    }
                    yield best;
                }
                case GNN_POLICY -> {
                    List<WlFeatures.Graph> gs = new ArrayList<>();
                    StripsTask.Action goal = null;
                    for (StripsTask.Action a : cand) {
                        BitSet t = a.apply(s);
                        if (task.isGoal(t)) {
                            goal = a;
                        }
                        gs.add(WlFeatures.graph(task, t, null));
                    }
                    if (goal != null) {
                        yield goal;
                    }
                    float[] sc = Var.noGrad(() -> scores(gs)).value().toArray();
                    int best = 0;
                    for (int i = 1; i < sc.length; i++) {
                        if (sc[i] < sc[best]) {
                            best = i;
                        }
                    }
                    yield cand.get(best);
                }
            };
            plan.add(pick);
            s = pick.apply(s);
            visited.add(s);
        }
        return new Rollout(plan, task.isGoal(s), (System.nanoTime() - t0) / 1e6);
    }
}
