package io.github.zzuegg.brainiac.experiments.evolution;

import io.github.zzuegg.brainiac.experimental.evolve.CmaEs;
import io.github.zzuegg.brainiac.experimental.evolve.DifferentialEvolution;
import io.github.zzuegg.brainiac.experimental.evolve.EvolutionStrategy;
import io.github.zzuegg.brainiac.experimental.evolve.ParticleSwarm;
import io.github.zzuegg.brainiac.rl.CartPole;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Pendulum;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Population-based black-box optimisers on the classic benchmark functions and on
 * neuroevolution of control policies, at equal function-evaluation budgets: differential
 * evolution (DE/rand/1/bin; Storn and Price 1997) and particle swarm optimisation (global and
 * ring neighbourhoods; Kennedy and Eberhart 1995) against CMA-ES (Hansen,
 * <a href="https://arxiv.org/abs/1604.00772">arXiv:1604.00772</a>), the OpenAI evolution
 * strategy (Salimans et al., <a href="https://arxiv.org/abs/1703.03864">arXiv:1703.03864</a>) and
 * uniform random search.
 *
 * <p>Every method searches the box mapped to {@code [-1, 1]^d}; the unbounded methods (CMA-ES,
 * ES) start at a uniform random point and have their proposals clipped to the box.
 */
public final class BlackBoxOptimizers {

    private BlackBoxOptimizers() {
    }

    /** The benchmark functions, all with global minimum 0. */
    public enum Function {
        /** Highly multimodal, separable, in [-5.12, 5.12]^d. */
        RASTRIGIN(5.12),
        /** A narrow curved valley, non-separable, in [-5, 5]^d (minimum at (1, ..., 1)). */
        ROSENBROCK(5),
        /** A nearly flat outer region with a central funnel, in [-32.768, 32.768]^d. */
        ACKLEY(32.768),
        /** Deceptive: the second-best minimum is far from the best, in [-500, 500]^d. */
        SCHWEFEL(500);

        private final double bound;

        Function(double bound) {
            this.bound = bound;
        }

        /**
         * @param x the point (in the function's own box)
         * @return the value
         */
        public double value(double[] x) {
            int d = x.length;
            double s = 0;
            switch (this) {
                case RASTRIGIN -> {
                    for (double v : x) {
                        s += v * v - 10 * Math.cos(2 * Math.PI * v) + 10;
                    }
                }
                case ROSENBROCK -> {
                    for (int i = 0; i + 1 < d; i++) {
                        s += 100 * Math.pow(x[i + 1] - x[i] * x[i], 2) + Math.pow(1 - x[i], 2);
                    }
                }
                case ACKLEY -> {
                    double sq = 0;
                    double cos = 0;
                    for (double v : x) {
                        sq += v * v;
                        cos += Math.cos(2 * Math.PI * v);
                    }
                    s = -20 * Math.exp(-0.2 * Math.sqrt(sq / d)) - Math.exp(cos / d) + 20 + Math.E;
                }
                case SCHWEFEL -> {
                    s = 418.9828872724338 * d;
                    for (double v : x) {
                        s -= v * Math.sin(Math.sqrt(Math.abs(v)));
                    }
                }
                default -> throw new IllegalStateException();
            }
            return Math.max(0, s);
        }
    }

    /**
     * One optimiser's result, means over seeds.
     *
     * @param method the optimiser
     * @param curve best value found after 10%, 25%, 50% and 100% of the budget
     * @param success the fraction of runs reaching the target
     */
    public record Score(String method, double[] curve, double success) {

        /** @return the best value at the full budget */
        public double best() {
            return curve[curve.length - 1];
        }
    }

    /** The methods compared, in order. */
    public static final List<String> METHODS = List.of("random", "OpenAI-ES", "CMA-ES", "DE", "PSO", "PSO-ring");

    /** Tracks the best-so-far value at budget checkpoints. */
    private static final class Tracker implements ToDoubleFunction<double[]> {
        private final ToDoubleFunction<double[]> cost;
        private final int[] checkpoints;
        private final double[] curve;
        private double best = Double.POSITIVE_INFINITY;
        private int evaluations;

        Tracker(ToDoubleFunction<double[]> cost, int budget) {
            this.cost = cost;
            this.checkpoints = new int[] {budget / 10, budget / 4, budget / 2, budget};
            this.curve = new double[checkpoints.length];
            java.util.Arrays.fill(curve, Double.NaN);
        }

        @Override
        public double applyAsDouble(double[] u) {
            double[] x = u.clone();
            for (int i = 0; i < x.length; i++) {
                x[i] = Math.max(-1, Math.min(1, x[i]));
            }
            double c = cost.applyAsDouble(x);
            evaluations++;
            best = Math.min(best, c);
            for (int k = 0; k < checkpoints.length; k++) {
                if (evaluations == checkpoints[k]) {
                    curve[k] = best;
                }
            }
            return c;
        }

        double[] finish() {
            for (int k = 0; k < curve.length; k++) {
                if (Double.isNaN(curve[k])) {
                    curve[k] = best;
                }
            }
            return curve;
        }
    }

    /**
     * Runs one method on a cost over {@code [-1, 1]^d} with the given budget.
     *
     * @param method one of {@link #METHODS}
     * @param cost the cost
     * @param d the dimension
     * @param budget evaluations
     * @param seed the seed
     * @return the best-so-far curve at the checkpoints
     */
    public static double[] optimise(String method, ToDoubleFunction<double[]> cost, int d, int budget, long seed) {
        Tracker t = new Tracker(cost, budget);
        double[] lo = new double[d];
        double[] hi = new double[d];
        java.util.Arrays.fill(lo, -1);
        java.util.Arrays.fill(hi, 1);
        SplittableRandom r = new SplittableRandom(seed);
        double[] start = new double[d];
        for (int i = 0; i < d; i++) {
            start[i] = r.nextDouble(-1, 1);
        }
        switch (method) {
            case "random" -> {
                for (int e = 0; e < budget; e++) {
                    double[] x = new double[d];
                    for (int i = 0; i < d; i++) {
                        x[i] = r.nextDouble(-1, 1);
                    }
                    t.applyAsDouble(x);
                }
            }
            case "OpenAI-ES" -> {
                int pairs = Math.max(8, 2 * d);
                EvolutionStrategy.of(start).sigma(0.05).learningRate(0.02).pairs(pairs).seed(seed)
                        .run(x -> -t.applyAsDouble(x), budget / (2 * pairs));
            }
            case "CMA-ES" -> CmaEs.of(start, 0.3).seed(seed).minimize(t, budget);
            case "DE" -> DifferentialEvolution.of(lo, hi).seed(seed).minimize(t, budget);
            case "PSO" -> ParticleSwarm.of(lo, hi).seed(seed).minimize(t, budget);
            case "PSO-ring" -> ParticleSwarm.of(lo, hi).ring(true).seed(seed).minimize(t, budget);
            default -> throw new IllegalArgumentException(method);
        }
        return t.finish();
    }

    /**
     * Every method on a benchmark function.
     *
     * @param f the function
     * @param d the dimension
     * @param budget evaluations per run
     * @param target the value counted as success
     * @param seeds seeds
     * @return method name to its score
     */
    public static Map<String, Score> benchmark(Function f, int d, int budget, double target, int seeds) {
        ToDoubleFunction<double[]> cost = u -> {
            double[] x = new double[u.length];
            for (int i = 0; i < x.length; i++) {
                x[i] = u[i] * f.bound;
            }
            return f.value(x);
        };
        return compare(cost, d, budget, target, seeds);
    }

    private static Map<String, Score> compare(ToDoubleFunction<double[]> cost, int d, int budget, double target,
            int seeds) {
        Map<String, Score> out = new LinkedHashMap<>();
        for (String m : METHODS) {
            double[] curve = new double[4];
            double success = 0;
            for (long seed = 1; seed <= seeds; seed++) {
                double[] c = optimise(m, cost, d, budget, seed);
                for (int k = 0; k < 4; k++) {
                    curve[k] += c[k] / seeds;
                }
                success += c[3] <= target ? 1.0 / seeds : 0;
            }
            out.put(m, new Score(m, curve, success));
        }
        return out;
    }

    /** The control tasks for neuroevolution. */
    public enum Control {
        /** Core cart-pole (500 steps), a linear policy {@code push right iff w . x + b > 0}: 5 weights. */
        CART_POLE,
        /** Core pendulum swing-up (200 steps), a 3-8-1 tanh MLP torque policy: 41 weights. */
        PENDULUM
    }

    private static double policyReturn(Control task, double[] w, long episodeSeed) {
        Environment env = task == Control.CART_POLE ? new CartPole() : new Pendulum();
        float[] obs = env.reset(new SplittableRandom(episodeSeed));
        double total = 0;
        while (true) {
            float[] action;
            if (task == Control.CART_POLE) {
                double s = w[4];
                for (int i = 0; i < 4; i++) {
                    s += w[i] * obs[i];
                }
                action = new float[] {s > 0 ? 1 : 0};
            } else {
                double out = w[40];
                for (int h = 0; h < 8; h++) {
                    double a = w[24 + h];
                    for (int i = 0; i < 3; i++) {
                        a += w[h * 3 + i] * obs[i];
                    }
                    out += w[32 + h] * Math.tanh(a);
                }
                action = new float[] {(float) (2 * Math.tanh(out))};
            }
            Environment.Step s = env.step(action);
            total += s.reward();
            if (s.done()) {
                return total;
            }
            obs = s.observation();
        }
    }

    /**
     * Neuroevolution: every method maximises the mean return of {@code episodes} fixed-seed
     * episodes over policy weights in {@code [-scale, scale]}; the score is the negated return
     * of the best weights found, re-evaluated on 10 fresh episodes.
     *
     * @param task the control task
     * @param budget fitness evaluations
     * @param episodes episodes per fitness evaluation
     * @param seeds seeds
     * @return method name to the mean held-out return (higher is better)
     */
    public static Map<String, Double> neuroevolution(Control task, int budget, int episodes, int seeds) {
        int d = task == Control.CART_POLE ? 5 : 41;
        double scale = task == Control.CART_POLE ? 1 : 3;
        Map<String, Double> out = new LinkedHashMap<>();
        for (String m : METHODS) {
            double sum = 0;
            for (long seed = 1; seed <= seeds; seed++) {
                long base = seed * 1000;
                double[] bestU = {Double.POSITIVE_INFINITY};
                double[][] bestW = {null};
                ToDoubleFunction<double[]> cost = u -> {
                    double[] w = new double[d];
                    for (int i = 0; i < d; i++) {
                        w[i] = u[i] * scale;
                    }
                    double ret = 0;
                    for (int e = 0; e < episodes; e++) {
                        ret += policyReturn(task, w, base + e) / episodes;
                    }
                    if (-ret < bestU[0]) {
                        bestU[0] = -ret;
                        bestW[0] = w;
                    }
                    return -ret;
                };
                optimise(m, cost, d, budget, seed);
                double test = 0;
                for (int e = 0; e < 10; e++) {
                    test += policyReturn(task, bestW[0], 777_000 + seed * 100 + e) / 10;
                }
                sum += test / seeds;
            }
            out.put(m, sum);
        }
        return out;
    }
}
