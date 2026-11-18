package io.github.zzuegg.brainiac.experimental.symbolic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.GaussianProcess;
import io.github.zzuegg.brainiac.experimental.ml.StationaryKernel;
import io.github.zzuegg.brainiac.experimental.nn.GinLayer;
import io.github.zzuegg.brainiac.tensor.Matrices;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Learned planning heuristics that generalise across instance sizes (WL-GOOSE, Chen, Trevizan
 * &amp; Thiébaux, ICAPS 2024, <a href="https://arxiv.org/abs/2403.16508">arXiv:2403.16508</a>).
 * Training targets are the optimal costs-to-go of the states on optimal plans of small
 * instances (found by A* with LM-cut, {@link #optimalSamples}).
 *
 * <ul>
 *   <li>{@link Mode#LINEAR_RIDGE}: ridge regression on {@link WlFeatures} (the paper's linear
 *       model; trained in closed form in milliseconds).</li>
 *   <li>{@link Mode#GP}: Gaussian-process regression on the same features (RBF kernel with
 *       fitted hyperparameters), which also gives a predictive variance.</li>
 *   <li>{@link Mode#GNN}: the GNN baseline: two {@link GinLayer}s on the instance learning graph
 *       (one-hot initial colours, edge labels dropped), sum pooling and a linear read-out,
 *       trained by Adam on the squared error.</li>
 * </ul>
 */
public final class LearnedPlanningHeuristic {

    /** The model. */
    public enum Mode {
        /** Ridge regression on WL features. */
        LINEAR_RIDGE,
        /** Gaussian-process regression on WL features. */
        GP,
        /** A GIN on the instance learning graph. */
        GNN
    }

    /**
     * A state on an optimal plan.
     *
     * @param task its task
     * @param state the state
     * @param costToGo its optimal cost-to-go
     * @param next the next state on the plan (null at the goal)
     * @param nextAction the plan's action from this state (null at the goal)
     */
    public record Sample(StripsTask task, BitSet state, double costToGo, BitSet next, StripsTask.Action nextAction) {
    }

    /**
     * States of optimal plans (A* with LM-cut).
     *
     * @param tasks small tasks
     * @param budget the expansion budget per task (unsolved tasks are skipped)
     * @return the samples
     */
    public static List<Sample> optimalSamples(List<StripsTask> tasks, int budget) {
        List<Sample> out = new ArrayList<>();
        for (StripsTask t : tasks) {
            StateSpaceSearch.Result r = StateSpaceSearch.astar(t, Landmarks.lmCut(t), budget);
            if (!r.found()) {
                continue;
            }
            BitSet s = t.initialState();
            double remaining = r.cost();
            for (StripsTask.Action a : r.plan()) {
                BitSet n = a.apply(s);
                out.add(new Sample(t, s, remaining, n, a));
                remaining -= a.cost();
                s = n;
            }
            out.add(new Sample(t, s, 0, null, null));
        }
        return out;
    }

    private final Mode mode;
    private final WlFeatures wl;
    private double[] weights;
    private double bias;
    private GaussianProcess gp;
    private double[] scale;
    private GinLayer gin1;
    private GinLayer gin2;
    private Var head;
    private final Map<Long, Integer> nodeColours = new HashMap<>();
    private double trainMillis;

    private LearnedPlanningHeuristic(Mode mode, WlFeatures wl) {
        this.mode = mode;
        this.wl = wl;
    }

    /**
     * Trains a heuristic.
     *
     * @param mode the model
     * @param samples training states with costs-to-go
     * @param iterations WL rounds
     * @param seed the seed (GNN initialisation and batching)
     * @return the heuristic
     */
    public static LearnedPlanningHeuristic train(Mode mode, List<Sample> samples, int iterations, long seed) {
        long t0 = System.nanoTime();
        LearnedPlanningHeuristic h = new LearnedPlanningHeuristic(mode, new WlFeatures(iterations));
        double[] y = samples.stream().mapToDouble(Sample::costToGo).toArray();
        if (mode == Mode.GNN) {
            h.trainGnn(samples, y, seed);
        } else {
            List<double[]> rows = new ArrayList<>();
            for (Sample s : samples) {
                rows.add(h.wl.features(s.task(), s.state(), true));
            }
            h.wl.freeze();
            double[][] x = h.wl.pad(rows);
            if (mode == Mode.LINEAR_RIDGE) {
                double[] wb = ridge(x, y, 1e-1);
                h.weights = java.util.Arrays.copyOf(wb, wb.length - 1);
                h.bias = wb[wb.length - 1];
            } else {
                int d = h.wl.dimension();
                h.scale = new double[d];
                for (double[] r : x) {
                    for (int j = 0; j < d; j++) {
                        h.scale[j] = Math.max(h.scale[j], Math.abs(r[j]));
                    }
                }
                double[][] xs = h.scaled(x);
                h.gp = GaussianProcess.optimize(xs, y, StationaryKernel.rbf(Math.sqrt(d) / 2, 25), 0.1, 40);
            }
        }
        h.trainMillis = (System.nanoTime() - t0) / 1e6;
        return h;
    }

    /**
     * Ridge regression with an unpenalised intercept.
     *
     * @param x rows
     * @param y targets
     * @param lambda the ridge penalty
     * @return the weights followed by the intercept
     */
    public static double[] ridge(double[][] x, double[] y, double lambda) {
        int d = x[0].length;
        double[][] a = new double[d + 1][d + 1];
        double[] b = new double[d + 1];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j <= d; j++) {
                double xj = j < d ? x[i][j] : 1;
                if (xj == 0) {
                    continue;
                }
                b[j] += xj * y[i];
                for (int k = 0; k <= d; k++) {
                    double xk = k < d ? x[i][k] : 1;
                    a[j][k] += xj * xk;
                }
            }
        }
        for (int j = 0; j < d; j++) {
            a[j][j] += lambda;
        }
        a[d][d] += 1e-9;
        return Matrices.solve(a, b);
    }

    private double[][] scaled(double[][] x) {
        double[][] out = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            out[i] = scaled(x[i]);
        }
        return out;
    }

    private double[] scaled(double[] x) {
        double[] out = new double[scale.length];
        for (int j = 0; j < scale.length; j++) {
            out[j] = scale[j] == 0 ? 0 : x[j] / scale[j];
        }
        return out;
    }

    // ---- GNN -------------------------------------------------------------------------------------

    private int colourIndex(long c, boolean grow) {
        Integer i = nodeColours.get(c);
        if (i == null) {
            if (!grow) {
                return -1;
            }
            i = nodeColours.size();
            nodeColours.put(c, i);
        }
        return i;
    }

    private Var[] batch(List<WlFeatures.Graph> graphs, int width) {
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
                int c = colourIndex(g.colours()[v], false);
                if (c >= 0 && c < width) {
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
        return new Var[] {Var.of(Tensor.of(Shape.of(n, width), x)), Var.of(Tensor.of(Shape.of(n, n), a)),
            Var.of(Tensor.of(Shape.of(graphs.size(), n), m))};
    }

    private Var gnnForward(Var[] in) {
        Var h = gin1.forward(List.of(in[0], in[1]), true).relu();
        h = gin2.forward(List.of(h, in[1]), true).relu();
        return in[2].matmul(h).matmul(head);
    }

    private void trainGnn(List<Sample> samples, double[] y, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<WlFeatures.Graph> graphs = new ArrayList<>();
        for (Sample s : samples) {
            WlFeatures.Graph g = WlFeatures.graph(s.task(), s.state(), null);
            for (long c : g.colours()) {
                colourIndex(c, true);
            }
            graphs.add(g);
        }
        int width = nodeColours.size();
        int hidden = 16;
        gin1 = new GinLayer(width, hidden, hidden, r);
        gin2 = new GinLayer(hidden, hidden, hidden, r);
        head = Var.param(Tensor.normal(Shape.of(hidden, 1), 0, 0.1f, r));
        List<Var> params = new ArrayList<>(gin1.parameters());
        params.addAll(gin2.parameters());
        params.add(head);
        Optimizer opt = Optimizer.adam(5e-3f);
        int bs = 16;
        int[] order = new int[graphs.size()];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        for (int epoch = 0; epoch < 25; epoch++) {
            for (int i = order.length - 1; i > 0; i--) {
                int j = r.nextInt(i + 1);
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
            for (int s = 0; s < order.length; s += bs) {
                List<WlFeatures.Graph> gb = new ArrayList<>();
                float[] yb = new float[Math.min(bs, order.length - s)];
                for (int k = 0; k < yb.length; k++) {
                    gb.add(graphs.get(order[s + k]));
                    yb[k] = (float) y[order[s + k]];
                }
                Var pred = gnnForward(batch(gb, width));
                Var loss = pred.sub(Var.of(Tensor.of(Shape.of(yb.length, 1), yb))).square().mean();
                loss.backward();
                opt.step(params);
            }
        }
    }

    // ---- use ---------------------------------------------------------------------------------------

    /** @return the model */
    public Mode mode() {
        return mode;
    }

    /** @return the training time in milliseconds */
    public double trainMillis() {
        return trainMillis;
    }

    /** @return the WL feature map (null for the GNN) */
    public WlFeatures features() {
        return mode == Mode.GNN ? null : wl;
    }

    /**
     * The estimate.
     *
     * @param task the task
     * @param state the state
     * @return the predicted cost-to-go, floored at 0 (0 at goals)
     */
    public double estimate(StripsTask task, BitSet state) {
        if (task.isGoal(state)) {
            return 0;
        }
        return switch (mode) {
            case LINEAR_RIDGE -> {
                double[] x = wl.features(task, state, false);
                double v = bias;
                for (int j = 0; j < x.length; j++) {
                    v += weights[j] * x[j];
                }
                yield Math.max(0, v);
            }
            case GP -> Math.max(0, gp.predict(new double[][] {scaled(wl.features(task, state, false))}).mean()[0]);
            case GNN -> {
                List<WlFeatures.Graph> one = List.of(WlFeatures.graph(task, state, null));
                yield Math.max(0, Var.noGrad(() -> gnnForward(batch(one, nodeColours.size()))).value().toArray()[0]);
            }
        };
    }

    /**
     * The GP's predictive variance (GP mode only).
     *
     * @param task the task
     * @param state the state
     * @return the variance of the latent cost-to-go
     */
    public double variance(StripsTask task, BitSet state) {
        if (mode != Mode.GP) {
            throw new IllegalStateException("variance needs the GP mode");
        }
        return gp.predict(new double[][] {scaled(wl.features(task, state, false))}).latentVariance()[0];
    }

    /**
     * @param task the task to search
     * @return the heuristic as a search evaluator
     */
    public StateSpaceSearch.Evaluator evaluator(StripsTask task) {
        return (s, d) -> StateSpaceSearch.Evaluation.of(estimate(task, s));
    }
}
