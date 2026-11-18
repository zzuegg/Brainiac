package io.github.zzuegg.brainiac.experimental.explore;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.rl.Td3;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Evolutionary RL on top of the step-by-step {@link Td3} driver ({@link Td3#begin},
 * {@link Td3#actUnit}, {@link Td3#observe}, {@link Td3#updateOnce}):
 *
 * <ul>
 *   <li>{@link Method#ERL}: evolutionary reinforcement learning (Khadka &amp; Tumer, NeurIPS 2018,
 *       <a href="https://arxiv.org/abs/1805.07917">arXiv:1805.07917</a>): a genetic population of
 *       deterministic actors (elitism, uniform crossover between elites, Gaussian mutation of a
 *       share of the weights) is evaluated each generation; every evaluation step goes into the
 *       TD3 learner's replay buffer, the learner runs its own exploring episode with one update
 *       per step, and every {@code sync} generations its actor replaces the weakest individual;</li>
 *   <li>{@link Method#CEM_RL}: CEM-RL (Pourchot &amp; Sigaud, ICLR 2019,
 *       <a href="https://arxiv.org/abs/1810.01222">arXiv:1810.01222</a>): a diagonal Gaussian over
 *       actor parameters (the cross-entropy method, like {@code CmaEs} with a diagonal covariance);
 *       half of each generation's samples receive TD3 gradient steps (critic and actor) before
 *       evaluation, all evaluations fill the replay buffer, and the distribution is refitted to the
 *       elites with weights {@code log(1 + elite) - log(rank)} and a decaying noise floor.</li>
 * </ul>
 *
 * <p>Both need TD3 driven one step and one update at a time; before wave 12 that was the local
 * {@code StepwiseTd3}.
 */
public final class EvolutionaryRl {

    /** The method. */
    public enum Method {
        /** genetic population plus a TD3 learner sharing its replay buffer */
        ERL,
        /** cross-entropy method with TD3 steps on half of the samples */
        CEM_RL
    }

    /**
     * The outcome of a run.
     *
     * @param steps environment steps at each generation's end
     * @param bestReturn the best evaluation return of each generation
     * @param champion the parameters of the final best actor (CEM-RL: the mean)
     * @param stepsUsed environment steps used
     */
    public record Result(long[] steps, double[] bestReturn, float[] champion, long stepsUsed) {
    }

    private final Method method;
    private final Td3 td3;
    private final Model evalActor;
    private final SplittableRandom random;
    private int population = 6;
    private double sigma = 0.1;
    private int gradientSteps = 50;
    private int sync = 2;
    private double mutationShare = 0.1;
    private int episodeCap = 200;

    private EvolutionaryRl(Method method, Td3 td3, Model evalActor, long seed) {
        this.method = Objects.requireNonNull(method, "method");
        this.td3 = td3;
        this.evalActor = evalActor;
        this.random = new SplittableRandom(seed);
    }

    /**
     * @param method the method
     * @param td3 a TD3 learner (its actor architecture is copied for the population)
     * @param networks the factory that built {@code td3}'s networks
     * @param observationDims observation values
     * @param actionDims action values
     * @param seed the seed
     * @return the runner
     */
    public static EvolutionaryRl of(Method method, Td3 td3, RlNetworkFactory networks, int observationDims,
            int actionDims, long seed) {
        return new EvolutionaryRl(method, Objects.requireNonNull(td3), networks.create(observationDims, actionDims, 21),
                seed);
    }

    /** @param n individuals per generation @return this runner */
    public EvolutionaryRl population(int n) {
        this.population = Math.max(2, n);
        return this;
    }

    /** @param s the initial standard deviation (CEM-RL) or mutation scale (ERL) @return this runner */
    public EvolutionaryRl sigma(double s) {
        this.sigma = s;
        return this;
    }

    /** @param n TD3 updates per gradient-receiving CEM-RL sample @return this runner */
    public EvolutionaryRl gradientSteps(int n) {
        this.gradientSteps = n;
        return this;
    }

    /** @param generations ERL generations between learner-to-population copies @return this runner */
    public EvolutionaryRl sync(int generations) {
        this.sync = Math.max(1, generations);
        return this;
    }

    /** @param steps the step cap of every evaluation episode @return this runner */
    public EvolutionaryRl episodeCap(int steps) {
        this.episodeCap = steps;
        return this;
    }

    /**
     * Runs until the environment-step budget is used.
     *
     * @param env the environment (a box action space)
     * @param budget environment steps
     * @return the result
     */
    public Result run(Environment env, long budget) {
        td3.begin(env);
        if (!(env.actionSpace() instanceof Space.Box)) {
            throw new IllegalArgumentException("needs continuous actions");
        }
        float[] start = flatten(td3.actor());
        int dims = start.length;
        long[] used = {0};
        List<Long> stepLog = new ArrayList<>();
        List<Double> bestLog = new ArrayList<>();
        float[] champion = start.clone();
        if (method == Method.CEM_RL) {
            double[] mean = new double[dims];
            double[] var = new double[dims];
            for (int i = 0; i < dims; i++) {
                mean[i] = start[i];
                var[i] = sigma * sigma;
            }
            double noise = sigma * sigma;
            int elites = Math.max(1, population / 2);
            while (used[0] < budget) {
                float[][] pop = new float[population][dims];
                double[] fit = new double[population];
                for (int p = 0; p < population; p++) {
                    for (int i = 0; i < dims; i++) {
                        pop[p][i] = (float) (mean[i] + Math.sqrt(var[i]) * random.nextGaussian());
                    }
                    if (p < population / 2 && td3.stepsObserved() > 0) {
                        assign(td3.actor(), pop[p]);
                        for (int g = 0; g < gradientSteps; g++) {
                            td3.updateOnce();
                        }
                        pop[p] = flatten(td3.actor());
                    }
                }
                for (int p = 0; p < population && used[0] < budget; p++) {
                    fit[p] = episode(env, pop[p], used, budget);
                }
                Integer[] order = rank(fit);
                double[] w = new double[elites];
                double ws = 0;
                for (int e = 0; e < elites; e++) {
                    w[e] = Math.log(1 + elites) - Math.log(1 + e);
                    ws += w[e];
                }
                double[] newMean = new double[dims];
                for (int e = 0; e < elites; e++) {
                    float[] x = pop[order[e]];
                    for (int i = 0; i < dims; i++) {
                        newMean[i] += w[e] / ws * x[i];
                    }
                }
                for (int i = 0; i < dims; i++) {
                    double v = 0;
                    for (int e = 0; e < elites; e++) {
                        double d = pop[order[e]][i] - mean[i];
                        v += w[e] / ws * d * d;
                    }
                    var[i] = v + noise;
                }
                noise = 0.95 * noise + 0.05 * 1e-5;
                mean = newMean;
                for (int i = 0; i < dims; i++) {
                    champion[i] = (float) mean[i];
                }
                stepLog.add(used[0]);
                bestLog.add(fit[order[0]]);
            }
        } else {
            float[][] pop = new float[population][];
            for (int p = 0; p < population; p++) {
                pop[p] = start.clone();
                for (int i = 0; i < dims; i++) {
                    pop[p][i] += (float) (sigma * random.nextGaussian());
                }
            }
            int generation = 0;
            while (used[0] < budget) {
                double[] fit = new double[population];
                for (int p = 0; p < population && used[0] < budget; p++) {
                    fit[p] = episode(env, pop[p], used, budget);
                }
                learnerEpisode(env, used, budget);
                Integer[] order = rank(fit);
                champion = pop[order[0]].clone();
                stepLog.add(used[0]);
                bestLog.add(fit[order[0]]);
                int elites = Math.max(1, population / 3);
                float[][] next = new float[population][];
                for (int e = 0; e < elites; e++) {
                    next[e] = pop[order[e]].clone();
                }
                for (int p = elites; p < population; p++) {
                    float[] a = pop[order[random.nextInt(elites)]];
                    float[] b = pop[order[random.nextInt(population / 2 + 1)]];
                    float[] child = new float[dims];
                    for (int i = 0; i < dims; i++) {
                        child[i] = random.nextBoolean() ? a[i] : b[i];
                        if (random.nextDouble() < mutationShare) {
                            child[i] += (float) (sigma * random.nextGaussian());
                        }
                    }
                    next[p] = child;
                }
                generation++;
                if (generation % sync == 0) {
                    next[population - 1] = flatten(td3.actor());
                }
                pop = next;
            }
        }
        return new Result(stepLog.stream().mapToLong(Long::longValue).toArray(),
                bestLog.stream().mapToDouble(Double::doubleValue).toArray(), champion, used[0]);
    }

    /**
     * @param parameters actor parameters (as in {@link Result#champion})
     * @return a deterministic agent acting with them (scaled to the environment of the last run)
     */
    public Agent agent(float[] parameters) {
        float[] copy = parameters.clone();
        return (obs, explore) -> {
            assign(evalActor, copy);
            return td3.toEnvironment(unitAction(evalActor, obs));
        };
    }

    private double episode(Environment env, float[] params, long[] used, long budget) {
        assign(evalActor, params);
        float[] obs = env.reset(random.split());
        double ret = 0;
        for (int t = 0; t < episodeCap && used[0] < budget; t++) {
            float[] u = unitAction(evalActor, obs);
            Environment.Step s = env.step(td3.toEnvironment(u));
            td3.observe(obs, u, s);
            used[0]++;
            ret += s.reward();
            if (s.done()) {
                break;
            }
            obs = s.observation();
        }
        return ret;
    }

    private void learnerEpisode(Environment env, long[] used, long budget) {
        float[] obs = env.reset(random.split());
        for (int t = 0; t < episodeCap && used[0] < budget; t++) {
            float[] u = td3.actUnit(obs);
            Environment.Step s = env.step(td3.toEnvironment(u));
            td3.observe(obs, u, s);
            used[0]++;
            td3.updateOnce();
            if (s.done()) {
                break;
            }
            obs = s.observation();
        }
    }

    private static float[] unitAction(Model actor, float[] obs) {
        Tensor x = Tensor.of(Shape.of(1, obs.length), obs).to(actor.device());
        return actor.forward(Var.of(x), false).tanh().value().to(DType.FLOAT32).toArray();
    }

    private static Integer[] rank(double[] fit) {
        Integer[] order = new Integer[fit.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (x, y) -> Double.compare(fit[y], fit[x]));
        return order;
    }

    /**
     * @param model a model
     * @return its parameters, concatenated
     */
    public static float[] flatten(Model model) {
        List<Var> params = model.parameters();
        int total = 0;
        for (Var p : params) {
            total += (int) p.value().shape().size();
        }
        float[] out = new float[total];
        int k = 0;
        for (Var p : params) {
            float[] v = p.value().to(DType.FLOAT32).toArray();
            System.arraycopy(v, 0, out, k, v.length);
            k += v.length;
        }
        return out;
    }

    /**
     * @param model a model
     * @param values parameters in {@link #flatten} order
     */
    public static void assign(Model model, float[] values) {
        int k = 0;
        for (Var p : model.parameters()) {
            Shape shape = p.value().shape();
            int n = (int) shape.size();
            p.assign(Tensor.of(shape, Arrays.copyOfRange(values, k, k + n)).to(p.value().device()));
            k += n;
        }
    }
}
