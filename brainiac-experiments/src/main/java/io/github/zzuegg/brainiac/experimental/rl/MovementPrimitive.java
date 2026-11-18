package io.github.zzuegg.brainiac.experimental.rl;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * A dynamic movement primitive: a demonstrated trajectory encoded as a stable second-order
 * attractor per dimension,
 * {@code tau z' = alpha (beta (g - y) - z) + f(s)}, {@code tau y' = z}, driven by a phase
 * {@code tau s' = -alpha_s s} that decays from 1 to about 0. The forcing term
 * {@code f(s) = sum_i psi_i(s) w_i / sum_i psi_i(s) * s * (g - y0)} is a normalised mix of
 * Gaussian basis functions in the phase, scaled by the movement's amplitude, so it vanishes
 * as the phase decays and the system converges to the goal {@code g} whatever the weights;
 * new goals and start points rescale the demonstrated shape.
 * <ul>
 *   <li>{@link #fit} learns the weights from one or more demonstrations by locally weighted
 *       regression of the forcing term the demonstrations imply;</li>
 *   <li>{@link #rollout} integrates the system to a trajectory;</li>
 *   <li>{@link #improve} refines the weights by reward-weighted averaging of perturbed
 *       rollouts (the policy-improvement step of PI^2 and of reward-weighted regression).</li>
 * </ul>
 *
 * <p>Ijspeert, Nakanishi, Hoffmann, Pastor and Schaal 2013, "Dynamical Movement Primitives:
 * Learning Attractor Models for Motor Behaviors", Neural Computation 25(2).
 */
public final class MovementPrimitive {

    private static final double ALPHA_S = 4;

    private final int dims;
    private final int basis;
    private final double alpha;
    private final double beta;
    private final double[] centres;
    private final double[] widths;
    private double[][] weights;

    private MovementPrimitive(int dims, int basis, double alpha) {
        this.dims = dims;
        this.basis = basis;
        this.alpha = alpha;
        this.beta = alpha / 4;
        this.centres = new double[basis];
        this.widths = new double[basis];
        for (int i = 0; i < basis; i++) {
            centres[i] = Math.exp(-ALPHA_S * i / (double) Math.max(1, basis - 1));
        }
        for (int i = 0; i < basis; i++) {
            double gap = i + 1 < basis ? centres[i + 1] - centres[i] : centres[i] - centres[i - 1];
            widths[i] = 1 / (gap * gap);
        }
        this.weights = new double[dims][basis];
    }

    /**
     * @param dims the dimensions of the movement
     * @param basis basis functions per dimension
     * @param alpha the spring gain {@code alpha} (critically damped with {@code beta = alpha / 4}),
     *     for example 25
     * @return a primitive with zero weights (a plain point attractor)
     */
    public static MovementPrimitive of(int dims, int basis, double alpha) {
        return new MovementPrimitive(dims, basis, alpha);
    }

    /** @return the weights, dimension by dimension (a copy) */
    public double[] weights() {
        double[] out = new double[dims * basis];
        for (int d = 0; d < dims; d++) {
            System.arraycopy(weights[d], 0, out, d * basis, basis);
        }
        return out;
    }

    /**
     * @param flat weights, dimension by dimension
     * @return a copy of this primitive with those weights
     */
    public MovementPrimitive withWeights(double[] flat) {
        MovementPrimitive p = new MovementPrimitive(dims, basis, alpha);
        for (int d = 0; d < dims; d++) {
            System.arraycopy(flat, d * basis, p.weights[d], 0, basis);
        }
        return p;
    }

    private double forcing(int d, double s) {
        double num = 0;
        double den = 1e-10;
        for (int i = 0; i < basis; i++) {
            double psi = Math.exp(-widths[i] * (s - centres[i]) * (s - centres[i]));
            num += psi * weights[d][i];
            den += psi;
        }
        return num / den * s;
    }

    /**
     * Fits the weights to demonstrations sampled every {@code dt}, each from its own start
     * (first row) to its own goal (last row); all demonstrations share the forcing shape.
     *
     * @param demos demonstrations, {@code [steps][dims]} each
     * @param dt the sampling interval
     * @return this
     */
    public MovementPrimitive fit(List<double[][]> demos, double dt) {
        double[][] num = new double[dims][basis];
        double[][] den = new double[dims][basis];
        for (double[][] y : demos) {
            int steps = y.length;
            double tau = (steps - 1) * dt;
            double[] y0 = y[0];
            double[] g = y[steps - 1];
            for (int t = 0; t < steps; t++) {
                double s = Math.exp(-ALPHA_S * t * dt / tau);
                int prev = Math.max(0, t - 1);
                int next = Math.min(steps - 1, t + 1);
                for (int d = 0; d < dims; d++) {
                    double v = (y[next][d] - y[prev][d]) / ((next - prev) * dt);
                    double acc = (y[next][d] - 2 * y[t][d] + y[prev][d]) / (dt * dt);
                    if (t == 0 || t == steps - 1) {
                        acc = 0;
                    }
                    double target = tau * tau * acc - alpha * (beta * (g[d] - y[t][d]) - tau * v);
                    double xi = s * (g[d] - y0[d]);
                    for (int i = 0; i < basis; i++) {
                        double psi = Math.exp(-widths[i] * (s - centres[i]) * (s - centres[i]));
                        num[d][i] += psi * xi * target;
                        den[d][i] += psi * xi * xi;
                    }
                }
            }
        }
        // the forcing term is psi-normalised, so the per-basis regression targets the term
        // before normalisation; this is the standard LWR fit of Ijspeert et al.
        for (int d = 0; d < dims; d++) {
            for (int i = 0; i < basis; i++) {
                weights[d][i] = num[d][i] / (den[d][i] + 1e-10);
            }
        }
        return this;
    }

    /**
     * Integrates the primitive (Euler, {@code steps - 1} steps of {@code dt}, so the duration
     * {@code tau = (steps - 1) dt}).
     *
     * @param start the start position
     * @param goal the goal
     * @param steps samples, including the start
     * @param dt the time step
     * @return the trajectory {@code [steps][dims]}
     */
    public double[][] rollout(double[] start, double[] goal, int steps, double dt) {
        double tau = (steps - 1) * dt;
        double[][] out = new double[steps][];
        double[] y = start.clone();
        double[] z = new double[dims];
        double s = 1;
        out[0] = y.clone();
        for (int t = 1; t < steps; t++) {
            for (int d = 0; d < dims; d++) {
                double f = forcing(d, s) * (goal[d] - start[d]);
                double zd = (alpha * (beta * (goal[d] - y[d]) - z[d]) + f) / tau;
                y[d] += z[d] / tau * dt;
                z[d] += zd * dt;
            }
            s += -ALPHA_S * s / tau * dt;
            out[t] = y.clone();
        }
        return out;
    }

    /**
     * Reward-weighted policy improvement: each iteration perturbs the weights with Gaussian
     * noise, rolls out every candidate, and moves to their average weighted by
     * {@code exp((R - max R) / (temperature * std R))}.
     *
     * @param reward the reward of a trajectory (higher is better)
     * @param start the start
     * @param goal the goal
     * @param steps trajectory samples
     * @param dt the time step
     * @param iterations iterations
     * @param rollouts perturbed rollouts per iteration
     * @param sigma the exploration noise on the weights
     * @param temperature the softmax temperature in standard deviations of the rewards
     * @param seed the seed
     * @return the improved primitive
     */
    public MovementPrimitive improve(ToDoubleFunction<double[][]> reward, double[] start, double[] goal, int steps,
            double dt, int iterations, int rollouts, double sigma, double temperature, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] mean = weights();
        for (int it = 0; it < iterations; it++) {
            double[][] cand = new double[rollouts][];
            double[] rew = new double[rollouts];
            for (int k = 0; k < rollouts; k++) {
                cand[k] = mean.clone();
                if (k > 0) {
                    for (int j = 0; j < cand[k].length; j++) {
                        cand[k][j] += sigma * r.nextGaussian();
                    }
                }
                rew[k] = reward.applyAsDouble(withWeights(cand[k]).rollout(start, goal, steps, dt));
            }
            double max = Double.NEGATIVE_INFINITY;
            double mu = 0;
            for (double v : rew) {
                max = Math.max(max, v);
                mu += v / rollouts;
            }
            double sd = 0;
            for (double v : rew) {
                sd += (v - mu) * (v - mu) / rollouts;
            }
            sd = Math.sqrt(sd) + 1e-12;
            double[] next = new double[mean.length];
            double z = 0;
            for (int k = 0; k < rollouts; k++) {
                double w = Math.exp((rew[k] - max) / (temperature * sd));
                z += w;
                for (int j = 0; j < next.length; j++) {
                    next[j] += w * cand[k][j];
                }
            }
            for (int j = 0; j < next.length; j++) {
                next[j] /= z;
            }
            mean = next;
        }
        return withWeights(mean);
    }

    /**
     * The root-mean-square jerk (third derivative, by finite differences) of a trajectory.
     *
     * @param y the trajectory {@code [steps][dims]}
     * @param dt the sampling interval
     * @return the RMS jerk
     */
    public static double jerk(double[][] y, double dt) {
        double total = 0;
        int count = 0;
        for (int t = 0; t + 3 < y.length; t++) {
            for (int d = 0; d < y[t].length; d++) {
                double j = (y[t + 3][d] - 3 * y[t + 2][d] + 3 * y[t + 1][d] - y[t][d]) / (dt * dt * dt);
                total += j * j;
                count++;
            }
        }
        return Math.sqrt(total / Math.max(1, count));
    }
}
