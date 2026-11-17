package io.github.zzuegg.brainiac.experimental.evolve;

import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * POET, the Paired Open-Ended Trailblazer (a light version): co-evolves a population of
 * environments together with an agent for each. Environments come from a parameterised
 * family; every few iterations, environments whose agent does well spawn mutated children,
 * which are kept only when they are neither too easy nor too hard for the current agents
 * (the <em>minimal criterion</em>) and, among those, the most novel ones first. Agents are
 * optimised on their own environment with an {@link EvolutionStrategy}, and periodically
 * every agent is tried in every other environment: a better one is <em>transferred</em>,
 * so skills found in one place become stepping stones elsewhere.
 *
 * <p>Works with any {@link Environment} from {@code brainiac-rl}: an
 * {@link EnvironmentFamily} turns parameters into an environment, and a policy factory turns
 * a parameter vector into an {@link Agent}.
 *
 * <p>Wang, Lehman, Clune and Stanley 2019, "Paired Open-Ended Trailblazer (POET)",
 * <a href="https://arxiv.org/abs/1901.01753">arXiv:1901.01753</a>.
 */
public final class Poet {

    /** A parameterised family of environments. */
    public interface EnvironmentFamily {
        /**
         * Builds the environment with given parameters.
         *
         * @param parameters the parameters
         * @return the environment
         */
        Environment create(double[] parameters);

        /**
         * Makes a mutated copy of parameters, usually a bit harder.
         *
         * @param parameters the parent's parameters
         * @param random the random source
         * @return the child's parameters
         */
        double[] mutate(double[] parameters, SplittableRandom random);
    }

    /**
     * An environment with its paired agent.
     *
     * @param id a running number
     * @param parameters the environment's parameters
     * @param agent the agent's parameters
     * @param score the agent's return in the environment
     * @param born the iteration the pair was created
     */
    public record Pair(int id, double[] parameters, double[] agent, double score, int born) {
    }

    /**
     * One iteration's summary.
     *
     * @param iteration the iteration
     * @param pairs active pairs
     * @param created pairs created this iteration
     * @param transfers agents transferred this iteration
     * @param meanScore the mean score of the active pairs
     */
    public record Iteration(int iteration, int pairs, int created, int transfers, double meanScore) {
    }

    private final EnvironmentFamily family;
    private final Function<double[], Agent> policy;
    private final int agentSize;
    private double minScore = Double.NEGATIVE_INFINITY;
    private double maxScore = Double.POSITIVE_INFINITY;
    private double reproduceScore = Double.NEGATIVE_INFINITY;
    private int capacity = 8;
    private int mutationInterval = 10;
    private int transferInterval = 5;
    private int childrenPerReproduction = 4;
    private int episodeSteps = 200;
    private int episodes = 1;
    private int esPairs = 8;
    private double esSigma = 0.1;
    private double esRate = 0.05;
    private long seed = 1;

    private final List<Pair> active = new ArrayList<>();
    private final List<double[]> archive = new ArrayList<>();
    private final List<EvolutionStrategy> optimizers = new ArrayList<>();
    private final List<Iteration> history = new ArrayList<>();
    private int nextId;
    private int iteration;
    private int totalTransfers;
    private SplittableRandom random;

    private Poet(EnvironmentFamily family, Function<double[], Agent> policy, int agentSize) {
        this.family = Objects.requireNonNull(family);
        this.policy = Objects.requireNonNull(policy);
        this.agentSize = agentSize;
    }

    /**
     * Sets up POET.
     *
     * @param family the environment family
     * @param policy turns agent parameters into an agent
     * @param agentSize the number of agent parameters
     * @return the search, to configure and {@link #start}
     */
    public static Poet of(EnvironmentFamily family, Function<double[], Agent> policy, int agentSize) {
        return new Poet(family, policy, agentSize);
    }

    /**
     * Sets the minimal criterion: a new environment is kept only if the best current
     * agent's return in it is in {@code [min, max]}.
     *
     * @param min below this the environment is too hard
     * @param max above this it is too easy
     * @return this
     */
    public Poet minimalCriterion(double min, double max) {
        this.minScore = min;
        this.maxScore = max;
        return this;
    }

    /**
     * Sets the return an agent needs before its environment may reproduce.
     *
     * @param score the threshold
     * @return this
     */
    public Poet reproduceScore(double score) {
        this.reproduceScore = score;
        return this;
    }

    /**
     * Sets how many pairs are active at once; the oldest are retired first.
     *
     * @param pairs the capacity
     * @return this
     */
    public Poet capacity(int pairs) {
        this.capacity = pairs;
        return this;
    }

    /**
     * Sets the schedule.
     *
     * @param mutateEvery iterations between attempts to create new environments
     * @param transferEvery iterations between transfer attempts
     * @return this
     */
    public Poet intervals(int mutateEvery, int transferEvery) {
        this.mutationInterval = mutateEvery;
        this.transferInterval = transferEvery;
        return this;
    }

    /**
     * Sets how agents are scored.
     *
     * @param steps the step limit of an episode
     * @param count episodes averaged per score
     * @return this
     */
    public Poet episodes(int steps, int count) {
        this.episodeSteps = steps;
        this.episodes = count;
        return this;
    }

    /**
     * Sets the inner evolution strategy.
     *
     * @param pairs antithetic pairs per step
     * @param sigma noise scale
     * @param rate learning rate
     * @return this
     */
    public Poet strategy(int pairs, double sigma, double rate) {
        this.esPairs = pairs;
        this.esSigma = sigma;
        this.esRate = rate;
        return this;
    }

    /**
     * Sets the seed.
     *
     * @param s the seed
     * @return this
     */
    public Poet seed(long s) {
        this.seed = s;
        return this;
    }

    /**
     * Starts with one environment and an agent.
     *
     * @param parameters the first environment
     * @param agent the first agent's parameters (often zeros)
     * @return this
     */
    public Poet start(double[] parameters, double[] agent) {
        if (agent.length != agentSize) {
            throw new IllegalArgumentException("agent must have " + agentSize + " parameters");
        }
        random = new SplittableRandom(seed);
        add(parameters.clone(), agent.clone());
        return this;
    }

    /**
     * The average return of an agent in an environment.
     *
     * @param parameters the environment
     * @param agent the agent
     * @return the average return
     */
    public double evaluate(double[] parameters, double[] agent) {
        Environment env = family.create(parameters);
        Agent a = policy.apply(agent);
        double total = 0;
        for (int e = 0; e < episodes; e++) {
            float[] obs = env.reset(new SplittableRandom(seed * 31 + e));
            for (int t = 0; t < episodeSteps; t++) {
                Environment.Step s = env.step(a.act(obs, false));
                total += s.reward();
                obs = s.observation();
                if (s.done()) {
                    break;
                }
            }
        }
        return total / episodes;
    }

    /**
     * Runs iterations.
     *
     * @param count how many
     * @return the active pairs afterwards
     */
    public List<Pair> run(int count) {
        if (random == null) {
            throw new IllegalStateException("call start() first");
        }
        for (int i = 0; i < count; i++) {
            step();
        }
        return pairs();
    }

    /**
     * Runs one iteration: maybe create environments, optimise every agent, maybe transfer.
     *
     * @return its summary
     */
    public Iteration step() {
        int created = 0;
        if (iteration > 0 && iteration % mutationInterval == 0) {
            created = reproduce();
        }
        for (int i = 0; i < active.size(); i++) {
            Pair p = active.get(i);
            EvolutionStrategy es = optimizers.get(i);
            double[] env = p.parameters();
            es.step(theta -> evaluate(env, theta));
            double[] agent = es.center();
            active.set(i, new Pair(p.id(), env, agent, evaluate(env, agent), p.born()));
        }
        int transfers = 0;
        if (iteration > 0 && iteration % transferInterval == 0) {
            transfers = transfer();
        }
        totalTransfers += transfers;
        double mean = active.stream().mapToDouble(Pair::score).average().orElse(0);
        Iteration it = new Iteration(iteration++, active.size(), created, transfers, mean);
        history.add(it);
        return it;
    }

    private void add(double[] env, double[] agent) {
        active.add(new Pair(nextId++, env, agent, evaluate(env, agent), iteration));
        optimizers.add(EvolutionStrategy.of(agent).sigma(esSigma).learningRate(esRate).pairs(esPairs)
                .seed(ParameterVector.seed(seed, nextId)));
        archive.add(env);
        while (active.size() > capacity) {
            active.removeFirst();
            optimizers.removeFirst();
        }
    }

    private int reproduce() {
        List<Pair> parents = active.stream().filter(p -> p.score() >= reproduceScore).toList();
        if (parents.isEmpty()) {
            return 0;
        }
        record Candidate(double[] env, double[] agent, double novelty) {
        }
        List<Candidate> ok = new ArrayList<>();
        BehaviourArchive known = new BehaviourArchive(archive.getFirst().length);
        archive.forEach(known::add);
        for (int c = 0; c < childrenPerReproduction * parents.size(); c++) {
            Pair parent = parents.get(random.nextInt(parents.size()));
            double[] child = family.mutate(parent.parameters(), random);
            double[] bestAgent = null;
            double best = Double.NEGATIVE_INFINITY;
            for (Pair p : active) {
                double s = evaluate(child, p.agent());
                if (s > best) {
                    best = s;
                    bestAgent = p.agent();
                }
            }
            if (best >= minScore && best <= maxScore) {
                ok.add(new Candidate(child, bestAgent, known.novelty(child, 5, null)));
            }
        }
        ok.sort(Comparator.comparingDouble(c -> -c.novelty()));
        int added = 0;
        for (Candidate c : ok) {
            if (added >= Math.max(1, capacity / 4)) {
                break;
            }
            add(c.env(), c.agent());
            added++;
        }
        return added;
    }

    private int transfer() {
        int transfers = 0;
        List<Pair> snapshot = List.copyOf(active);
        for (int i = 0; i < active.size(); i++) {
            Pair target = active.get(i);
            double[] bestAgent = null;
            double best = target.score();
            for (Pair source : snapshot) {
                if (source.id() == target.id()) {
                    continue;
                }
                double s = evaluate(target.parameters(), source.agent());
                if (s > best) {
                    best = s;
                    bestAgent = source.agent();
                }
            }
            if (bestAgent != null) {
                active.set(i, new Pair(target.id(), target.parameters(), bestAgent.clone(), best, target.born()));
                optimizers.set(i, EvolutionStrategy.of(bestAgent).sigma(esSigma).learningRate(esRate).pairs(esPairs)
                        .seed(ParameterVector.seed(seed, target.id(), iteration)));
                transfers++;
            }
        }
        return transfers;
    }

    /** @return the active pairs */
    public List<Pair> pairs() {
        return List.copyOf(active);
    }

    /** @return the parameters of every environment ever accepted */
    public List<double[]> archive() {
        return archive.stream().map(double[]::clone).toList();
    }

    /** @return the iterations so far */
    public List<Iteration> history() {
        return List.copyOf(history);
    }

    /** @return how many agents were transferred in total */
    public int transfers() {
        return totalTransfers;
    }

    /**
     * A linear policy with a tanh output: {@code tanh(W obs + b)}, parameters row by row
     * with the bias last in each row.
     *
     * @param observations the observation size
     * @param actions the action size
     * @return the policy factory, taking {@code actions * (observations + 1)} parameters
     */
    public static Function<double[], Agent> linearPolicy(int observations, int actions) {
        return theta -> (obs, explore) -> {
            float[] out = new float[actions];
            for (int a = 0; a < actions; a++) {
                int row = a * (observations + 1);
                double s = theta[row + observations];
                for (int i = 0; i < observations; i++) {
                    s += theta[row + i] * obs[i];
                }
                out[a] = (float) Math.tanh(s);
            }
            return out;
        };
    }
}
