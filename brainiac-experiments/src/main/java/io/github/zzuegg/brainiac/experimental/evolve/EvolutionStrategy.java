package io.github.zzuegg.brainiac.experimental.evolve;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

/**
 * OpenAI-style evolution strategy: estimates the gradient of the expected reward from
 * random perturbations of the parameters and climbs it, using only reward values.
 *
 * <ul>
 *   <li><b>Antithetic sampling</b>: every noise vector {@code e} is evaluated as
 *       {@code theta + sigma e} and {@code theta - sigma e}, which cancels the even terms of
 *       the estimate and halves its variance.</li>
 *   <li><b>Seed-regenerated noise</b>: a perturbation is identified by a seed; the update
 *       regenerates it, so memory stays at one vector and distributed workers would only
 *       exchange seeds and rewards.</li>
 *   <li><b>Rank shaping</b>: rewards are replaced by centred ranks in {@code [-0.5, 0.5]},
 *       which makes the step invariant to the reward's scale and robust to outliers.</li>
 *   <li>The gradient estimate drives Adam (or plain SGD), with optional weight decay.</li>
 * </ul>
 *
 * <pre>{@code
 * EvolutionStrategy es = EvolutionStrategy.of(new double[10]).sigma(0.1).learningRate(0.05)
 *         .pairs(20).seed(1);
 * es.run(theta -> -sphere(theta), 200);
 * double[] best = es.center();
 * }</pre>
 *
 * <p>Salimans et al. 2017, "Evolution Strategies as a Scalable Alternative to
 * Reinforcement Learning", <a href="https://arxiv.org/abs/1703.03864">arXiv:1703.03864</a>.
 * Rewards are maximised.
 */
public final class EvolutionStrategy {

    /**
     * What one iteration saw.
     *
     * @param iteration the iteration, from 0
     * @param meanReward the mean reward of the perturbed candidates
     * @param bestReward the best reward among them
     * @param gradientNorm the length of the gradient estimate
     */
    public record Step(int iteration, double meanReward, double bestReward, double gradientNorm) {
    }

    private final double[] theta;
    private double sigma = 0.1;
    private double learningRate = 0.01;
    private int pairs = 16;
    private boolean antithetic = true;
    private boolean rankShaping = true;
    private boolean adam = true;
    private double weightDecay;
    private long seed = 1;
    private boolean parallel;
    private int iteration;
    private final double[] m;
    private final double[] v;
    private final List<Step> history = new ArrayList<>();

    private EvolutionStrategy(double[] start) {
        this.theta = start.clone();
        this.m = new double[start.length];
        this.v = new double[start.length];
    }

    /**
     * Starts a search from a parameter vector.
     *
     * @param start the initial parameters (copied)
     * @return the strategy, to configure
     */
    public static EvolutionStrategy of(double[] start) {
        return new EvolutionStrategy(Objects.requireNonNull(start, "start"));
    }

    /**
     * Sets the noise scale.
     *
     * @param s the standard deviation of the perturbations
     * @return this
     */
    public EvolutionStrategy sigma(double s) {
        if (!(s > 0)) {
            throw new IllegalArgumentException("sigma must be positive");
        }
        this.sigma = s;
        return this;
    }

    /**
     * Sets the step size of the optimizer that follows the gradient estimate.
     *
     * @param rate the learning rate
     * @return this
     */
    public EvolutionStrategy learningRate(double rate) {
        this.learningRate = rate;
        return this;
    }

    /**
     * Sets how many noise vectors are drawn per iteration; with antithetic sampling each is
     * evaluated twice.
     *
     * @param count the number of noise vectors
     * @return this
     */
    public EvolutionStrategy pairs(int count) {
        if (count < 1) {
            throw new IllegalArgumentException("need at least one perturbation");
        }
        this.pairs = count;
        return this;
    }

    /**
     * Turns mirrored (antithetic) sampling on or off.
     *
     * @param on whether to evaluate {@code -e} as well as {@code e}
     * @return this
     */
    public EvolutionStrategy antithetic(boolean on) {
        this.antithetic = on;
        return this;
    }

    /**
     * Turns centred-rank fitness shaping on or off.
     *
     * @param on whether to use ranks instead of raw rewards
     * @return this
     */
    public EvolutionStrategy rankShaping(boolean on) {
        this.rankShaping = on;
        return this;
    }

    /**
     * Chooses Adam (the default) or plain gradient ascent for the update.
     *
     * @param on whether to use Adam
     * @return this
     */
    public EvolutionStrategy adam(boolean on) {
        this.adam = on;
        return this;
    }

    /**
     * Sets a decoupled weight decay, which keeps the parameters small.
     *
     * @param decay the decay per step, relative to the learning rate
     * @return this
     */
    public EvolutionStrategy weightDecay(double decay) {
        this.weightDecay = decay;
        return this;
    }

    /**
     * Sets the seed all noise seeds are derived from.
     *
     * @param s the seed
     * @return this
     */
    public EvolutionStrategy seed(long s) {
        this.seed = s;
        return this;
    }

    /**
     * Evaluates the perturbed candidates of an iteration in parallel. Results stay
     * reproducible; the objective must be thread-safe.
     *
     * @param on whether to evaluate in parallel
     * @return this
     */
    public EvolutionStrategy parallel(boolean on) {
        this.parallel = on;
        return this;
    }

    /** @return a copy of the current parameters (the centre of the search distribution) */
    public double[] center() {
        return theta.clone();
    }

    /** @return one entry per iteration so far */
    public List<Step> history() {
        return List.copyOf(history);
    }

    /**
     * The seed of perturbation {@code k} in iteration {@code it}; regenerate the noise with
     * {@link ParameterVector#noise(long, int)}.
     *
     * @param it the iteration
     * @param k the perturbation
     * @return its seed
     */
    public long noiseSeed(int it, int k) {
        return ParameterVector.seed(seed, it, k);
    }

    /**
     * Runs one iteration: perturb, evaluate, estimate the gradient, update.
     *
     * @param objective the reward to maximise
     * @return what happened
     */
    public Step step(ToDoubleFunction<double[]> objective) {
        int n = theta.length;
        int evals = antithetic ? 2 * pairs : pairs;
        int it = iteration;
        double[] rewards = new double[evals];
        IntStream range = IntStream.range(0, evals);
        (parallel ? range.parallel() : range).forEach(e -> {
            int k = antithetic ? e / 2 : e;
            double sign = antithetic && (e & 1) == 1 ? -1 : 1;
            double[] x = theta.clone();
            ParameterVector.addNoise(x, noiseSeed(it, k), sign * sigma);
            rewards[e] = objective.applyAsDouble(x);
        });
        double[] shaped = rankShaping ? centredRanks(rewards) : standardise(rewards);
        double[] grad = new double[n];
        for (int k = 0; k < pairs; k++) {
            double w = antithetic ? shaped[2 * k] - shaped[2 * k + 1] : shaped[k];
            if (w != 0) {
                ParameterVector.addNoise(grad, noiseSeed(it, k), w);
            }
        }
        double scale = 1.0 / (evals * sigma);
        double norm = 0;
        for (int i = 0; i < n; i++) {
            grad[i] *= scale;
            norm += grad[i] * grad[i];
        }
        apply(grad);
        iteration++;
        double mean = Arrays.stream(rewards).average().orElse(0);
        double best = Arrays.stream(rewards).max().orElse(0);
        Step s = new Step(it, mean, best, Math.sqrt(norm));
        history.add(s);
        return s;
    }

    /**
     * Runs several iterations.
     *
     * @param objective the reward to maximise
     * @param iterations how many
     * @return the final parameters
     */
    public double[] run(ToDoubleFunction<double[]> objective, int iterations) {
        for (int i = 0; i < iterations; i++) {
            step(objective);
        }
        return center();
    }

    private void apply(double[] grad) {
        int t = iteration + 1;
        for (int i = 0; i < theta.length; i++) {
            double g = grad[i];
            double update;
            if (adam) {
                m[i] = 0.9 * m[i] + 0.1 * g;
                v[i] = 0.999 * v[i] + 0.001 * g * g;
                double mh = m[i] / (1 - Math.pow(0.9, t));
                double vh = v[i] / (1 - Math.pow(0.999, t));
                update = mh / (Math.sqrt(vh) + 1e-8);
            } else {
                update = g;
            }
            theta[i] = theta[i] * (1 - learningRate * weightDecay) + learningRate * update;
        }
    }

    /**
     * Replaces values by their centred ranks: the smallest becomes -0.5, the largest 0.5.
     *
     * @param values the values
     * @return the ranks, in the same order
     */
    public static double[] centredRanks(double[] values) {
        int n = values.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(values[a], values[b]));
        double[] out = new double[n];
        for (int r = 0; r < n; r++) {
            out[order[r]] = n == 1 ? 0 : r / (double) (n - 1) - 0.5;
        }
        return out;
    }

    private static double[] standardise(double[] values) {
        double mean = Arrays.stream(values).average().orElse(0);
        double var = 0;
        for (double x : values) {
            var += (x - mean) * (x - mean);
        }
        double sd = Math.sqrt(var / values.length) + 1e-12;
        double[] out = new double[values.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = (values[i] - mean) / sd;
        }
        return out;
    }
}
