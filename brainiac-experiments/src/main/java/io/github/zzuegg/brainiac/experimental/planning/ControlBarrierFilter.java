package io.github.zzuegg.brainiac.experimental.planning;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A control barrier function safety filter (Ames et al., ECC 2019,
 * <a href="https://arxiv.org/abs/1903.11199">arXiv:1903.11199</a>) that wraps any
 * {@link Agent}: each proposed action {@code u_ref} is replaced by the closest action (in
 * squared distance) that satisfies the barrier conditions {@code a_i . u >= b_i} and the
 * action bounds, found by an exact active-set enumeration of the small quadratic program.
 * Wrapping an RL learner keeps exploration safe (Cheng et al., AAAI 2019,
 * <a href="https://arxiv.org/abs/1903.08792">arXiv:1903.08792</a>).
 *
 * <p>For a point robot with double-integrator dynamics {@code p'' = u} the position barrier
 * {@code h(p) >= 0} has relative degree two, so {@link #doubleIntegrator} uses the
 * exponential (high-order) CBF condition
 * {@code h'' + (alpha1 + alpha2) h' + alpha1 alpha2 h >= 0}, with
 * {@code h' = grad h . v} and {@code h'' = v^T Hess(h) v + grad h . u}, linear in {@code u}.
 * The {@link Barrier#analyticDiscs analytic} barrier is {@code |p - c|^2 - r^2} per disc; the
 * {@link LearnedBarrier learned} one is an MLP {@code h(p)} fitted with a hinge loss on safe
 * and unsafe samples (Dawson, Gao &amp; Fan, T-RO 2023), its gradient and Hessian taken by
 * finite differences.
 */
public final class ControlBarrierFilter implements Agent {

    /** A position barrier {@code h(p)}, positive in the safe set. */
    public interface Barrier {

        /**
         * @param p a position
         * @return {@code h(p)} for each constraint
         */
        double[] values(double[] p);

        /**
         * @param p a position
         * @return per constraint the gradient {@code [k][2]}
         */
        double[][] gradients(double[] p);

        /**
         * @param p a position
         * @return per constraint the Hessian {@code [k][2][2]}
         */
        double[][][] hessians(double[] p);

        /**
         * One barrier per disc obstacle: {@code |p - c|^2 - (r + margin)^2}.
         *
         * @param discs {@code [k][3]} as {@code (x, y, r)}
         * @param margin a safety margin
         * @return the barrier
         */
        static Barrier analyticDiscs(double[][] discs, double margin) {
            return new Barrier() {
                @Override
                public double[] values(double[] p) {
                    double[] h = new double[discs.length];
                    for (int i = 0; i < discs.length; i++) {
                        double dx = p[0] - discs[i][0];
                        double dy = p[1] - discs[i][1];
                        double r = discs[i][2] + margin;
                        h[i] = dx * dx + dy * dy - r * r;
                    }
                    return h;
                }

                @Override
                public double[][] gradients(double[] p) {
                    double[][] g = new double[discs.length][];
                    for (int i = 0; i < discs.length; i++) {
                        g[i] = new double[] {2 * (p[0] - discs[i][0]), 2 * (p[1] - discs[i][1])};
                    }
                    return g;
                }

                @Override
                public double[][][] hessians(double[] p) {
                    double[][][] h = new double[discs.length][][];
                    for (int i = 0; i < discs.length; i++) {
                        h[i] = new double[][] {{2, 0}, {0, 2}};
                    }
                    return h;
                }
            };
        }
    }

    /** A learned scalar barrier {@code h(p)}: an MLP trained with hinge losses. */
    public static final class LearnedBarrier implements Barrier {

        private final SimpleMlp net;
        private float[][][] w;
        private final double eps = 1e-3;

        /**
         * Creates an untrained barrier.
         *
         * @param hidden the hidden width
         * @param seed the seed
         */
        public LearnedBarrier(int hidden, long seed) {
            this.net = SimpleMlp.tanh(seed, 2, hidden, hidden, 1);
            snapshot();
        }

        private void snapshot() {
            w = new float[net.depth()][][];
            for (int l = 0; l < net.depth(); l++) {
                w[l] = new float[][] {net.weight(l).value().toArray(), net.bias(l).value().toArray()};
            }
        }

        private double eval(double x, double y) {
            int[] sizes = net.sizes();
            double[] h = {x, y};
            for (int l = 0; l < w.length; l++) {
                int out = sizes[l + 1];
                double[] o = new double[out];
                for (int j = 0; j < out; j++) {
                    o[j] = w[l][1][j];
                }
                for (int i = 0; i < h.length; i++) {
                    for (int j = 0; j < out; j++) {
                        o[j] += h[i] * w[l][0][i * out + j];
                    }
                }
                if (l < w.length - 1) {
                    for (int j = 0; j < out; j++) {
                        o[j] = Math.tanh(o[j]);
                    }
                }
                h = o;
            }
            return h[0];
        }

        /**
         * Fits {@code h >= gamma} on safe and {@code h <= -gamma} on unsafe positions.
         *
         * @param safe safe positions
         * @param unsafe unsafe positions
         * @param gamma the margin
         * @param steps gradient steps
         * @param learningRate Adam's rate
         * @param seed the seed
         */
        public void fit(List<double[]> safe, List<double[]> unsafe, float gamma, int steps, float learningRate,
                long seed) {
            Optimizer opt = Optimizer.adam(learningRate);
            SplittableRandom r = new SplittableRandom(seed);
            int b = 128;
            for (int s = 0; s < steps; s++) {
                float[] x = new float[2 * b * 2];
                float[] sign = new float[2 * b];
                for (int i = 0; i < 2 * b; i++) {
                    double[] p = i < b ? safe.get(r.nextInt(safe.size())) : unsafe.get(r.nextInt(unsafe.size()));
                    x[2 * i] = (float) p[0];
                    x[2 * i + 1] = (float) p[1];
                    sign[i] = i < b ? 1 : -1;
                }
                Var h = net.forward(Var.of(Tensor.of(Shape.of(2 * b, 2), x)));
                Var loss = h.mul(Tensor.of(Shape.of(2 * b, 1), sign)).neg().add(gamma).relu().mean();
                loss.backward();
                opt.step(net.parameters());
            }
            snapshot();
        }

        @Override
        public double[] values(double[] p) {
            return new double[] {eval(p[0], p[1])};
        }

        @Override
        public double[][] gradients(double[] p) {
            double gx = (eval(p[0] + eps, p[1]) - eval(p[0] - eps, p[1])) / (2 * eps);
            double gy = (eval(p[0], p[1] + eps) - eval(p[0], p[1] - eps)) / (2 * eps);
            return new double[][] {{gx, gy}};
        }

        @Override
        public double[][][] hessians(double[] p) {
            double f = eval(p[0], p[1]);
            double e = 1e-2;
            double hxx = (eval(p[0] + e, p[1]) - 2 * f + eval(p[0] - e, p[1])) / (e * e);
            double hyy = (eval(p[0], p[1] + e) - 2 * f + eval(p[0], p[1] - e)) / (e * e);
            double hxy = (eval(p[0] + e, p[1] + e) - eval(p[0] + e, p[1] - e) - eval(p[0] - e, p[1] + e)
                + eval(p[0] - e, p[1] - e)) / (4 * e * e);
            return new double[][][] {{{hxx, hxy}, {hxy, hyy}}};
        }
    }

    private final Agent policy;
    private final Barrier barrier;
    private final double alpha1;
    private final double alpha2;
    private final double[] lower;
    private final double[] upper;
    private final int positionIndex;
    private final int velocityIndex;
    private int interventions;
    private int infeasible;

    private ControlBarrierFilter(Agent policy, Barrier barrier, double alpha1, double alpha2, double[] lower,
            double[] upper, int positionIndex, int velocityIndex) {
        this.policy = policy;
        this.barrier = barrier;
        this.alpha1 = alpha1;
        this.alpha2 = alpha2;
        this.lower = lower.clone();
        this.upper = upper.clone();
        this.positionIndex = positionIndex;
        this.velocityIndex = velocityIndex;
    }

    /**
     * A filter for a planar double integrator whose observation holds the position and the
     * velocity.
     *
     * @param policy the wrapped policy
     * @param barrier the position barrier
     * @param alpha1 the first class-K gain
     * @param alpha2 the second class-K gain
     * @param lower the action lower bounds
     * @param upper the action upper bounds
     * @param positionIndex where {@code (px, py)} starts in the observation
     * @param velocityIndex where {@code (vx, vy)} starts
     * @return the filter
     */
    public static ControlBarrierFilter doubleIntegrator(Agent policy, Barrier barrier, double alpha1, double alpha2,
            double[] lower, double[] upper, int positionIndex, int velocityIndex) {
        return new ControlBarrierFilter(policy, barrier, alpha1, alpha2, lower, upper, positionIndex, velocityIndex);
    }

    /** @return how often the filter changed the action */
    public int interventions() {
        return interventions;
    }

    /** @return how often the QP was infeasible (the least-violating action was used) */
    public int infeasible() {
        return infeasible;
    }

    @Override
    public void startEpisode() {
        policy.startEpisode();
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        float[] u = policy.act(observation, explore);
        return filter(observation, u);
    }

    /**
     * Filters one action.
     *
     * @param observation the observation
     * @param proposed the proposed action
     * @return the safe action
     */
    public float[] filter(float[] observation, float[] proposed) {
        double[] p = {observation[positionIndex], observation[positionIndex + 1]};
        double[] v = {observation[velocityIndex], observation[velocityIndex + 1]};
        double[] h = barrier.values(p);
        double[][] g = barrier.gradients(p);
        double[][][] hess = barrier.hessians(p);
        List<double[]> cons = new ArrayList<>();
        for (int i = 0; i < h.length; i++) {
            double hDot = g[i][0] * v[0] + g[i][1] * v[1];
            double vHv = v[0] * (hess[i][0][0] * v[0] + hess[i][0][1] * v[1])
                + v[1] * (hess[i][1][0] * v[0] + hess[i][1][1] * v[1]);
            double b = -vHv - (alpha1 + alpha2) * hDot - alpha1 * alpha2 * h[i];
            cons.add(new double[] {g[i][0], g[i][1], b});
        }
        double[] ref = {proposed[0], proposed[1]};
        double[] out = project(ref, cons, lower, upper);
        if (out == null) {
            infeasible++;
            out = leastViolating(ref, cons, lower, upper);
        }
        if (Math.abs(out[0] - ref[0]) > 1e-9 || Math.abs(out[1] - ref[1]) > 1e-9) {
            interventions++;
        }
        float[] a = proposed.clone();
        a[0] = (float) out[0];
        a[1] = (float) out[1];
        return a;
    }

    /**
     * The closest 2-D point to {@code ref} satisfying {@code a . u >= b} for every constraint
     * and the box, by enumerating active sets of size 0 to 2.
     *
     * @param ref the reference
     * @param constraints rows {@code (a0, a1, b)}
     * @param lower box lower bounds
     * @param upper box upper bounds
     * @return the solution, or {@code null} if infeasible
     */
    public static double[] project(double[] ref, List<double[]> constraints, double[] lower, double[] upper) {
        List<double[]> all = new ArrayList<>(constraints);
        all.add(new double[] {1, 0, lower[0]});
        all.add(new double[] {-1, 0, -upper[0]});
        all.add(new double[] {0, 1, lower[1]});
        all.add(new double[] {0, -1, -upper[1]});
        double[] best = null;
        double bestCost = Double.POSITIVE_INFINITY;
        List<double[]> candidates = new ArrayList<>();
        candidates.add(ref.clone());
        for (double[] c : all) {
            double nn = c[0] * c[0] + c[1] * c[1];
            if (nn < 1e-18) {
                continue;
            }
            double t = (c[2] - (c[0] * ref[0] + c[1] * ref[1])) / nn;
            candidates.add(new double[] {ref[0] + t * c[0], ref[1] + t * c[1]});
        }
        for (int i = 0; i < all.size(); i++) {
            for (int j = i + 1; j < all.size(); j++) {
                double[] a = all.get(i);
                double[] b = all.get(j);
                double det = a[0] * b[1] - a[1] * b[0];
                if (Math.abs(det) < 1e-12) {
                    continue;
                }
                candidates.add(new double[] {(a[2] * b[1] - a[1] * b[2]) / det, (a[0] * b[2] - a[2] * b[0]) / det});
            }
        }
        for (double[] u : candidates) {
            boolean ok = true;
            for (double[] c : all) {
                if (c[0] * u[0] + c[1] * u[1] < c[2] - 1e-9) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                continue;
            }
            double cost = (u[0] - ref[0]) * (u[0] - ref[0]) + (u[1] - ref[1]) * (u[1] - ref[1]);
            if (cost < bestCost) {
                bestCost = cost;
                best = u;
            }
        }
        return best;
    }

    private static double[] leastViolating(double[] ref, List<double[]> constraints, double[] lower, double[] upper) {
        double[] best = ref.clone();
        double worst = Double.POSITIVE_INFINITY;
        int grid = 21;
        for (int i = 0; i < grid; i++) {
            for (int j = 0; j < grid; j++) {
                double[] u = {lower[0] + (upper[0] - lower[0]) * i / (grid - 1),
                    lower[1] + (upper[1] - lower[1]) * j / (grid - 1)};
                double viol = 0;
                for (double[] c : constraints) {
                    double norm = Math.max(1e-9, Math.hypot(c[0], c[1]));
                    viol = Math.max(viol, (c[2] - c[0] * u[0] - c[1] * u[1]) / norm);
                }
                if (viol < worst) {
                    worst = viol;
                    best = u;
                }
            }
        }
        return best;
    }
}
