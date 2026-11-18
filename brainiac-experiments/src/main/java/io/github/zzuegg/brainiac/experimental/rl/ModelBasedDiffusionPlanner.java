package io.github.zzuegg.brainiac.experimental.rl;

import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Model-based diffusion: the reverse diffusion process used as a training-free trajectory
 * optimiser on a known model ({@link ControlSystem}). The target is the distribution
 * {@code p(U) ∝ exp(-J(U) / lambda)} over control sequences; at each noise level {@code i} the
 * score of its noised version is estimated by Monte Carlo: sample candidates
 * {@code U ~ N(Y_i / sqrt(abar_i), (1 - abar_i) / abar_i)}, weight them by
 * {@code exp(-J / lambda)} from model rollouts, and the weighted mean {@code Ubar} gives
 * {@code score = (sqrt(abar_i) Ubar - Y_i) / (1 - abar_i)}; a reverse step then moves
 * {@code Y_(i-1) = (Y_i + (1 - alpha_i) score) / sqrt(alpha_i)}. Early, high-noise levels
 * average over wide regions of trajectory space, which escapes local minima where MPPI and CEM
 * (which sample around one mean at one scale) get stuck. Costs are standardised within each
 * batch, so {@code lambda} is a temperature in units of the batch's standard deviation.
 *
 * <p>{@link #mppi} is the single-scale special case kept for comparison: model-predictive
 * path integral control (Williams et al. 2017) samples around the warm-started mean with a
 * fixed noise scale and moves to the exponentially weighted mean, for several rounds.
 *
 * <p>Pan et al. 2024, "Model-Based Diffusion for Trajectory Optimization",
 * <a href="https://arxiv.org/abs/2407.01573">arXiv:2407.01573</a>.
 */
public final class ModelBasedDiffusionPlanner implements ControlSystem.Planner {

    private final ControlSystem system;
    private final int horizon;
    private final int samples;
    private final int levels;
    private final boolean diffusion;
    private final double sigma;
    private final double temperature;
    private final double[] alphaBar;
    private final double[] alphaBarWarm;
    private final SplittableRandom random;
    private double[][] plan;

    private ModelBasedDiffusionPlanner(ControlSystem system, int horizon, int samples, int levels, boolean diffusion,
            double sigma, double temperature, long seed) {
        this.system = Objects.requireNonNull(system);
        this.horizon = horizon;
        this.samples = samples;
        this.levels = levels;
        this.diffusion = diffusion;
        this.sigma = sigma;
        this.temperature = temperature;
        this.random = new SplittableRandom(seed);
        // the candidate spread sqrt((1 - abar_i) / abar_i) grows geometrically from 0.05 at the
        // last level to 3 (beyond the action range, nearly pure noise) at the first; a
        // warm-started (receding-horizon) solve only partially re-noises the previous plan, up
        // to a spread of 0.5, as in the paper's MPC variant
        this.alphaBar = schedule(levels, 3);
        this.alphaBarWarm = schedule(levels, 0.5);
    }

    private static double[] schedule(int levels, double maxSpread) {
        double[] ab = new double[levels + 1];
        ab[0] = 1;
        for (int i = 1; i <= levels; i++) {
            double spread = levels == 1 ? 0.05 : 0.05 * Math.pow(maxSpread / 0.05, (i - 1) / (double) (levels - 1));
            ab[i] = 1 / (1 + spread * spread);
        }
        return ab;
    }

    /**
     * @param system the model
     * @param horizon planning horizon in steps
     * @param samples rollouts per noise level
     * @param levels noise levels (rollouts per decision = samples x levels)
     * @param temperature the softmax temperature on standardised costs
     * @param seed the seed
     * @return a model-based diffusion planner
     */
    public static ModelBasedDiffusionPlanner of(ControlSystem system, int horizon, int samples, int levels,
            double temperature, long seed) {
        return new ModelBasedDiffusionPlanner(system, horizon, samples, levels, true, 0, temperature, seed);
    }

    /**
     * @param system the model
     * @param horizon planning horizon in steps
     * @param samples rollouts per round
     * @param rounds refinement rounds (rollouts per decision = samples x rounds)
     * @param sigma the sampling noise, in units of half the action range
     * @param temperature the softmax temperature on standardised costs
     * @param seed the seed
     * @return an MPPI planner
     */
    public static ModelBasedDiffusionPlanner mppi(ControlSystem system, int horizon, int samples, int rounds,
            double sigma, double temperature, long seed) {
        return new ModelBasedDiffusionPlanner(system, horizon, samples, rounds, false, sigma, temperature, seed);
    }

    /** @return rollouts of the model per decision */
    public int rolloutsPerDecision() {
        return samples * levels;
    }

    /** Controls in normalised units {@code [-1, 1]} mapped to the action box. */
    private double[] action(double[] y) {
        double[] lo = system.actionLow();
        double[] hi = system.actionHigh();
        double[] u = new double[y.length];
        for (int i = 0; i < u.length; i++) {
            double c = Math.max(-1, Math.min(1, y[i]));
            u[i] = (lo[i] + hi[i]) / 2 + c * (hi[i] - lo[i]) / 2;
        }
        return u;
    }

    private double cost(double[] x0, double[][] y) {
        double[] x = x0;
        double total = 0;
        for (int t = 0; t < horizon; t++) {
            double[] u = action(y[t]);
            total += system.cost(x, u);
            x = system.step(x, u);
        }
        return total + system.finalCost(x);
    }

    /** The weighted mean of candidates by exp(-standardised cost / temperature). */
    private double[][] weightedMean(double[][][] candidates, double[] costs) {
        double mean = 0;
        for (double c : costs) {
            mean += c / costs.length;
        }
        double var = 0;
        double min = Double.POSITIVE_INFINITY;
        for (double c : costs) {
            var += (c - mean) * (c - mean) / costs.length;
            min = Math.min(min, c);
        }
        double std = Math.sqrt(var) + 1e-9;
        int m = system.actionSize();
        double[][] out = new double[horizon][m];
        double z = 0;
        for (int k = 0; k < candidates.length; k++) {
            double w = Math.exp(-(costs[k] - min) / std / temperature);
            z += w;
            for (int t = 0; t < horizon; t++) {
                for (int j = 0; j < m; j++) {
                    out[t][j] += w * candidates[k][t][j];
                }
            }
        }
        for (double[] row : out) {
            for (int j = 0; j < m; j++) {
                row[j] /= z;
            }
        }
        return out;
    }

    /**
     * Optimises a control sequence from a state (open loop).
     *
     * @param x0 the start state
     * @param warm a warm-start sequence in normalised units, or null
     * @return the optimised sequence in normalised units {@code [-1, 1]}
     */
    public double[][] optimize(double[] x0, double[][] warm) {
        int m = system.actionSize();
        double[][] y = new double[horizon][m];
        if (diffusion) {
            double[] alphaBar = warm == null ? this.alphaBar : alphaBarWarm;
            double s = Math.sqrt(alphaBar[levels]);
            for (int t = 0; t < horizon; t++) {
                for (int j = 0; j < m; j++) {
                    // a warm start begins at the previous plan itself, which the first level then
                    // evaluates as its zero-noise candidate
                    y[t][j] = warm == null ? Math.sqrt(1 - alphaBar[levels]) * random.nextGaussian() : s * warm[t][j];
                }
            }
            for (int i = levels; i >= 1; i--) {
                double ab = alphaBar[i];
                double spread = Math.sqrt((1 - ab) / ab);
                double[][][] cand = new double[samples][horizon][m];
                double[] costs = new double[samples];
                for (int k = 0; k < samples; k++) {
                    for (int t = 0; t < horizon; t++) {
                        for (int j = 0; j < m; j++) {
                            cand[k][t][j] = Math.max(-1, Math.min(1, y[t][j] / Math.sqrt(ab)
                                    + (k == 0 ? 0 : spread * random.nextGaussian())));
                        }
                    }
                    costs[k] = cost(x0, cand[k]);
                }
                double[][] ubar = weightedMean(cand, costs);
                double alpha = ab / alphaBar[i - 1];
                for (int t = 0; t < horizon; t++) {
                    for (int j = 0; j < m; j++) {
                        double score = (Math.sqrt(ab) * ubar[t][j] - y[t][j]) / (1 - ab);
                        y[t][j] = (y[t][j] + (1 - alpha) * score) / Math.sqrt(alpha);
                    }
                }
                if (i == 1) {
                    y = ubar;
                }
            }
            return y;
        }
        double[][] mean = warm == null ? y : warm;
        for (int round = 0; round < levels; round++) {
            double[][][] cand = new double[samples][horizon][m];
            double[] costs = new double[samples];
            for (int k = 0; k < samples; k++) {
                for (int t = 0; t < horizon; t++) {
                    for (int j = 0; j < m; j++) {
                        cand[k][t][j] = Math.max(-1, Math.min(1,
                                mean[t][j] + (k == 0 ? 0 : sigma * random.nextGaussian())));
                    }
                }
                costs[k] = cost(x0, cand[k]);
            }
            mean = weightedMean(cand, costs);
        }
        return mean;
    }

    @Override
    public double[] act(double[] state) {
        double[][] warm = null;
        if (plan != null) {
            warm = new double[horizon][];
            for (int t = 0; t < horizon; t++) {
                warm[t] = plan[Math.min(t + 1, horizon - 1)].clone();
            }
        }
        plan = optimize(state, warm);
        return action(plan[0]);
    }

    @Override
    public void reset() {
        plan = null;
    }
}
