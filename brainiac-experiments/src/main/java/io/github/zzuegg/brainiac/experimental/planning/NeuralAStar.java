package io.github.zzuegg.brainiac.experimental.planning;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Conv;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.Padding;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Neural A* (Yonetani et al., ICML 2021, <a href="https://arxiv.org/abs/2009.07476">arXiv:2009.07476</a>):
 * a CNN encoder maps the map (obstacles, start and goal channels) to a guidance cost
 * {@code phi in (0, 1)} per cell, and a differentiable A* runs on it. Each step picks the
 * open node with the smallest {@code f = g + h} through a straight-through softmax (the
 * forward pass takes the hard argmin, the backward pass the softmax gradient), adds it to the
 * closed list, and opens its 4-neighbours with {@code g' = g(selected) + phi}; neighbour
 * expansion is a matrix product with the fixed grid adjacency. The loss is the mean absolute
 * difference between the closed-list history and the expert path, so the encoder learns
 * costs that make the search expand few nodes off the expert path.
 *
 * <p>At test time the same search runs in hard mode on plain arrays ({@link #search}). The
 * map is {@code side x side}, 4-connected, with Manhattan distance as {@code h}.
 */
public final class NeuralAStar {

    /**
     * A hard search result.
     *
     * @param path the cells from start to goal, empty if not found
     * @param expanded the nodes expanded
     */
    public record Search(List<Integer> path, int expanded) {
    }

    private final Model encoder;
    private final int side;
    private final float temperature;
    private final Tensor adjacency;

    /**
     * Creates the planner.
     *
     * @param side the map side
     * @param hidden the encoder's hidden channels
     * @param temperature the softmax temperature of the node selection
     * @param seed the seed
     */
    public NeuralAStar(int side, int hidden, float temperature, long seed) {
        this.side = side;
        this.temperature = temperature;
        this.encoder = Model.sequential().input(3, side, side).seed(seed)
            .add(Conv.of(hidden, 3, 3).padding(Padding.SAME).activation(Activation.RELU))
            .add(Conv.of(hidden, 3, 3).padding(Padding.SAME).activation(Activation.RELU))
            .add(Conv.of(1, 1, 1).padding(Padding.SAME).activation(Activation.SIGMOID)).build();
        int n = side * side;
        float[] a = new float[n * n];
        for (int c = 0; c < n; c++) {
            int x = c % side;
            int y = c / side;
            int[][] m = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int[] d : m) {
                int nx = x + d[0];
                int ny = y + d[1];
                if (nx >= 0 && ny >= 0 && nx < side && ny < side) {
                    a[c * n + ny * side + nx] = 1;
                }
            }
        }
        this.adjacency = Tensor.of(Shape.of(n, n), a);
    }

    /** @return the encoder's parameters */
    public List<Var> parameters() {
        return encoder.parameters();
    }

    private static float[] input(boolean[] blocked, int start, int goal) {
        int n = blocked.length;
        float[] x = new float[3 * n];
        for (int i = 0; i < n; i++) {
            x[i] = blocked[i] ? 1 : 0;
        }
        x[n + start] = 1;
        x[2 * n + goal] = 1;
        return x;
    }

    private float[] manhattan(int goal) {
        int n = side * side;
        float[] h = new float[n];
        for (int c = 0; c < n; c++) {
            h[c] = Math.abs(c % side - goal % side) + Math.abs(c / side - goal / side);
        }
        return h;
    }

    /**
     * The learned guidance cost of a map.
     *
     * @param blocked the obstacles
     * @param start the start cell
     * @param goal the goal cell
     * @return per cell {@code phi}
     */
    public double[] costs(boolean[] blocked, int start, int goal) {
        float[] phi = Var.noGrad(() -> encoder.forward(Var.of(Tensor.of(Shape.of(1, 3, side, side),
            input(blocked, start, goal))))).value().toArray();
        double[] out = new double[phi.length];
        for (int i = 0; i < phi.length; i++) {
            out[i] = phi[i];
        }
        return out;
    }

    /**
     * Trains on expert paths.
     *
     * @param maps the obstacle maps
     * @param starts the start cells
     * @param goals the goal cells
     * @param paths per map the expert path cells
     * @param steps gradient steps
     * @param batch maps per step
     * @param learningRate Adam's rate
     * @param seed the seed
     * @return the final loss
     */
    public double fit(List<boolean[]> maps, int[] starts, int[] goals, List<List<Integer>> paths, int steps, int batch,
            float learningRate, long seed) {
        Optimizer opt = Optimizer.adam(learningRate);
        SplittableRandom r = new SplittableRandom(seed);
        int n = side * side;
        double last = 0;
        for (int s = 0; s < steps; s++) {
            float[] x = new float[batch * 3 * n];
            float[] free = new float[batch * n];
            float[] h = new float[batch * n];
            float[] expert = new float[batch * n];
            int[] st = new int[batch];
            int[] gl = new int[batch];
            for (int i = 0; i < batch; i++) {
                int k = r.nextInt(maps.size());
                System.arraycopy(input(maps.get(k), starts[k], goals[k]), 0, x, i * 3 * n, 3 * n);
                for (int c = 0; c < n; c++) {
                    free[i * n + c] = maps.get(k)[c] ? 0 : 1;
                }
                System.arraycopy(manhattan(goals[k]), 0, h, i * n, n);
                for (int c : paths.get(k)) {
                    expert[i * n + c] = 1;
                }
                st[i] = starts[k];
                gl[i] = goals[k];
            }
            Var phi = encoder.forward(Var.of(Tensor.of(Shape.of(batch, 3, side, side), x))).reshape(batch, n);
            Var history = differentiableSearch(phi, free, h, st, gl, batch);
            Var loss = history.sub(Var.of(Tensor.of(Shape.of(batch, n), expert))).abs().mean();
            last = loss.value().item();
            loss.backward();
            opt.step(parameters());
        }
        return last;
    }

    private Var differentiableSearch(Var phi, float[] free, float[] h, int[] start, int[] goal, int b) {
        int n = side * side;
        float[] open = new float[b * n];
        float[] closed = new float[b * n];
        float[] alive = new float[b];
        Arrays.fill(alive, 1);
        float[] g0 = new float[b * n];
        for (int i = 0; i < b; i++) {
            open[i * n + start[i]] = 1;
        }
        Var g = Var.of(Tensor.of(Shape.of(b, n), g0));
        Var history = Var.of(Tensor.zeros(Shape.of(b, n)));
        Tensor hT = Tensor.of(Shape.of(b, n), h);
        for (int t = 0; t < n; t++) {
            boolean any = false;
            for (float a : alive) {
                any |= a > 0;
            }
            if (!any) {
                break;
            }
            float[] penalty = new float[b * n];
            for (int k = 0; k < b * n; k++) {
                penalty[k] = open[k] > 0 ? 0 : -1e4f;
            }
            Var f = g.add(hT);
            Var logits = f.mul(-1f / temperature).add(Tensor.of(Shape.of(b, n), penalty));
            Var p = logits.softmax(1);
            float[] fv = f.value().toArray();
            float[] hard = new float[b * n];
            int[] sel = new int[b];
            for (int i = 0; i < b; i++) {
                int best = -1;
                for (int c = 0; c < n; c++) {
                    if (open[i * n + c] > 0 && (best < 0 || fv[i * n + c] < fv[i * n + best])) {
                        best = c;
                    }
                }
                sel[i] = best;
                if (best < 0) {
                    alive[i] = 0;
                } else if (alive[i] > 0) {
                    hard[i * n + best] = 1;
                }
            }
            Tensor aliveT = Tensor.of(Shape.of(b, 1), alive.clone());
            Var v = Var.of(Tensor.of(Shape.of(b, n), hard)).add(p.sub(p.detach()).mul(aliveT));
            history = history.add(v);
            Var gSel = g.mul(v).sum(1, true);
            float[] nbr = Tensor.of(Shape.of(b, n), hard).matmul(adjacency).toArray();
            float[] fresh = new float[b * n];
            for (int k = 0; k < b * n; k++) {
                fresh[k] = nbr[k] > 0 && free[k] > 0 && open[k] == 0 && closed[k] == 0 ? 1 : 0;
            }
            Tensor freshT = Tensor.of(Shape.of(b, n), fresh);
            float[] keep = new float[b * n];
            for (int k = 0; k < b * n; k++) {
                keep[k] = 1 - fresh[k];
            }
            g = g.mul(Tensor.of(Shape.of(b, n), keep)).add(gSel.add(phi).mul(freshT));
            for (int i = 0; i < b; i++) {
                if (alive[i] == 0 || sel[i] < 0) {
                    continue;
                }
                open[i * n + sel[i]] = 0;
                closed[i * n + sel[i]] = 1;
                if (sel[i] == goal[i]) {
                    alive[i] = 0;
                }
            }
            for (int k = 0; k < b * n; k++) {
                if (fresh[k] > 0 && alive[k / n] > 0) {
                    open[k] = 1;
                }
            }
        }
        return history;
    }

    /**
     * Hard A* on a cost map: {@code f = g + h}, entering a cell costs {@code phi}.
     *
     * @param blocked the obstacles
     * @param phi per-cell entry cost, or {@code null} for the learned one
     * @param start the start cell
     * @param goal the goal cell
     * @return the path and the expansions
     */
    public Search search(boolean[] blocked, double[] phi, int start, int goal) {
        double[] cost = phi != null ? phi : costs(blocked, start, goal);
        int n = side * side;
        float[] h = manhattan(goal);
        double[] g = new double[n];
        Arrays.fill(g, Double.POSITIVE_INFINITY);
        int[] parent = new int[n];
        Arrays.fill(parent, -1);
        boolean[] open = new boolean[n];
        boolean[] closed = new boolean[n];
        g[start] = 0;
        open[start] = true;
        int expanded = 0;
        while (true) {
            int best = -1;
            for (int c = 0; c < n; c++) {
                if (open[c] && (best < 0 || g[c] + h[c] < g[best] + h[best])) {
                    best = c;
                }
            }
            if (best < 0) {
                return new Search(List.of(), expanded);
            }
            open[best] = false;
            closed[best] = true;
            expanded++;
            if (best == goal) {
                break;
            }
            int x = best % side;
            int y = best / side;
            int[][] m = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int[] d : m) {
                int nx = x + d[0];
                int ny = y + d[1];
                if (nx < 0 || ny < 0 || nx >= side || ny >= side) {
                    continue;
                }
                int c = ny * side + nx;
                if (blocked[c] || closed[c] || open[c]) {
                    continue;
                }
                g[c] = g[best] + cost[c];
                parent[c] = best;
                open[c] = true;
            }
        }
        List<Integer> path = new ArrayList<>();
        for (int c = goal; c != -1; c = parent[c]) {
            path.add(0, c);
        }
        return new Search(path, expanded);
    }
}
