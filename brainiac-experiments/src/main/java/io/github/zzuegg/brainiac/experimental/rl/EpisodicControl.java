package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.experimental.nn.KnnIndex;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Episode;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Model-free episodic control (Blundell et al. 2016, <a href="https://arxiv.org/abs/1606.04460">arXiv:1606.04460</a>):
 * an agent that remembers, per action, the states it acted in and the best discounted return
 * that followed, in a {@link KnnIndex} per action. A state seen before is valued by its
 * remembered return; a new one by the mean of its {@code k} nearest remembered states. After
 * each episode every visited state-action pair is written back with
 * {@code Q(s, a) = max(Q(s, a), R)}. The memory persists across episodes (and across
 * sessions with {@link #save}/{@link #load}), so one lucky trajectory is latched immediately:
 * far more sample-efficient than gradient-based value learning on deterministic tasks, and
 * far better than a memoryless policy.
 *
 * <p>Observations are used as keys directly, or through a fixed random projection to
 * {@code embedding} dimensions for large observations (as in the paper). Discrete actions only.
 */
public final class EpisodicControl implements Learner {

    private final int actions;
    private final KnnIndex[] memory;
    private final float[][] projection;
    private final int keySize;
    private int k = 5;
    private double radius = Double.POSITIVE_INFINITY;
    private double unknownValue;
    private double truncationValue;
    private double gamma = 0.99;
    private double epsilon = 0.1;
    private int capacity = 100_000;
    private boolean remember = true;
    private SplittableRandom random = new SplittableRandom(0);
    private final List<Double> returns = new ArrayList<>();

    private EpisodicControl(int observationSize, int actions, int embedding, long seed) {
        this.actions = actions;
        this.random = new SplittableRandom(seed);
        if (embedding > 0 && embedding < observationSize) {
            projection = new float[embedding][observationSize];
            SplittableRandom r = new SplittableRandom(seed ^ 0x5DEECE66DL);
            for (float[] row : projection) {
                for (int i = 0; i < row.length; i++) {
                    row[i] = (float) (r.nextGaussian() / Math.sqrt(embedding));
                }
            }
            keySize = embedding;
        } else {
            projection = null;
            keySize = observationSize;
        }
        memory = new KnnIndex[actions];
        for (int a = 0; a < actions; a++) {
            memory[a] = KnnIndex.create(keySize, 1, KnnIndex.Metric.L2);
        }
    }

    /**
     * Creates an agent for an environment with discrete actions and box observations.
     *
     * @param env the environment
     * @param embedding the random-projection size, or 0 to use observations as they are
     * @param seed the random seed
     * @return the agent
     */
    public static EpisodicControl of(Environment env, int embedding, long seed) {
        if (!(env.actionSpace() instanceof Space.Discrete d)) {
            throw new IllegalArgumentException("episodic control needs discrete actions, got " + env.actionSpace());
        }
        return new EpisodicControl(env.observationSpace().size(), d.n(), embedding, seed);
    }

    /**
     * @param neighbours the remembered states averaged to value a new one
     * @return this agent
     */
    public EpisodicControl neighbours(int neighbours) {
        this.k = neighbours;
        return this;
    }

    /**
     * Limits generalization: a new state is valued from remembered neighbours only within
     * this Euclidean distance (in key space); further away, or with no memory at all, an
     * action gets {@code unknownValue}. A finite radius with an optimistic unknown value
     * makes the agent try every action of a new state once: systematic exploration.
     *
     * @param distance the radius, infinite by default
     * @param unknownValue the value of an action with no memory in reach, 0 by default
     * @return this agent
     */
    public EpisodicControl generalization(double distance, double unknownValue) {
        this.radius = distance;
        this.unknownValue = unknownValue;
        return this;
    }

    /**
     * The value assumed after an episode cut off by a time limit (not terminated), from which
     * the returns of its last states are computed. Max-return memories latch whatever they
     * are given, so a cut-off tail valued at 0 looks better than real progress when every step
     * costs; set it to the cost of never finishing, such as {@code stepReward / (1 - discount)}.
     *
     * @param value the bootstrap value of a truncated episode, 0 by default
     * @return this agent
     */
    public EpisodicControl truncationValue(double value) {
        this.truncationValue = value;
        return this;
    }

    /**
     * @param discount the return discount
     * @return this agent
     */
    public EpisodicControl discount(double discount) {
        this.gamma = discount;
        return this;
    }

    /**
     * @param rate the share of random actions while exploring
     * @return this agent
     */
    public EpisodicControl epsilon(double rate) {
        this.epsilon = rate;
        return this;
    }

    /**
     * @param rows the most memories per action; the least recently used go first
     * @return this agent
     */
    public EpisodicControl capacity(int rows) {
        this.capacity = rows;
        return this;
    }

    /**
     * @param on whether memory survives from one episode to the next; off makes a memoryless
     *     baseline that forgets everything at every reset
     * @return this agent
     */
    public EpisodicControl rememberAcrossEpisodes(boolean on) {
        this.remember = on;
        return this;
    }

    /** @return the remembered state-action pairs */
    public int memories() {
        int n = 0;
        for (KnnIndex m : memory) {
            n += m.size();
        }
        return n;
    }

    /** @return the total reward of each episode {@link #fit} played, oldest first */
    public List<Double> episodeReturns() {
        return List.copyOf(returns);
    }

    private float[] key(float[] observation) {
        if (projection == null) {
            return observation.clone();
        }
        float[] e = new float[keySize];
        for (int i = 0; i < keySize; i++) {
            double s = 0;
            for (int j = 0; j < observation.length; j++) {
                s += projection[i][j] * observation[j];
            }
            e[i] = (float) s;
        }
        return e;
    }

    /**
     * The value estimate of every action in a state.
     *
     * @param observation the state
     * @return one estimate per action
     */
    public double[] actionValues(float[] observation) {
        float[] key = key(observation);
        double[] q = new double[actions];
        for (int a = 0; a < actions; a++) {
            KnnIndex.Hit[] hits = memory[a].search(key, k);
            if (hits.length == 0 || -hits[0].score() > radius * radius) {
                q[a] = unknownValue;
                continue;
            }
            if (hits[0].score() > -1e-9f) {
                q[a] = memory[a].value(hits[0].row())[0];
                continue;
            }
            double s = 0;
            int n = 0;
            for (KnnIndex.Hit h : hits) {
                if (-h.score() <= radius * radius) {
                    s += memory[a].value(h.row())[0];
                    n++;
                }
            }
            q[a] = s / n;
        }
        return q;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        if (explore && random.nextDouble() < epsilon) {
            return new float[] {random.nextInt(actions)};
        }
        double[] q = actionValues(observation);
        int best = 0;
        int ties = 1;
        for (int a = 1; a < actions; a++) {
            if (q[a] > q[best]) {
                best = a;
                ties = 1;
            } else if (q[a] == q[best] && random.nextInt(++ties) == 0) {
                best = a;
            }
        }
        return new float[] {best};
    }

    /**
     * Writes an episode into memory: each visited state-action pair keeps the best discounted
     * return seen after it.
     *
     * @param observations the states, in order
     * @param chosen the action taken in each
     * @param rewards the reward after each
     */
    public void remember(List<float[]> observations, List<Integer> chosen, List<Float> rewards) {
        remember(observations, chosen, rewards, 0);
    }

    /**
     * Writes an episode into memory, continuing from a bootstrap value after its last step.
     *
     * @param observations the states, in order
     * @param chosen the action taken in each
     * @param rewards the reward after each
     * @param bootstrap the value after the last step: 0 for a finished episode
     */
    public void remember(List<float[]> observations, List<Integer> chosen, List<Float> rewards, double bootstrap) {
        double ret = bootstrap;
        for (int t = observations.size() - 1; t >= 0; t--) {
            ret = rewards.get(t) + gamma * ret;
            int a = chosen.get(t);
            float[] key = key(observations.get(t));
            KnnIndex.Hit[] hit = memory[a].search(key, 1);
            if (hit.length > 0 && hit[0].score() > -1e-9f) {
                int row = hit[0].row();
                float old = memory[a].value(row)[0];
                memory[a].setValue(row, new float[] {(float) Math.max(old, ret)});
                memory[a].touch(row);
            } else {
                memory[a].add(key, new float[] {(float) ret});
                memory[a].pruneLeastRecentlyUsed(capacity);
            }
        }
    }

    /** Forgets every memory. */
    public void forget() {
        for (KnnIndex m : memory) {
            m.clear();
        }
    }

    /**
     * Plays and remembers episodes. The returned {@link RlResult} has them (the last one cut
     * short if the step budget runs out), the listener gets every episode and
     * {@link #episodeReturns()} has their totals.
     */
    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        RlResult result = new RlResult();
        long total = 0;
        int episodes = 0;
        while (total < steps) {
            if (!remember) {
                forget();
            }
            List<float[]> obs = new ArrayList<>();
            List<Integer> acts = new ArrayList<>();
            List<Float> rewards = new ArrayList<>();
            float[] o = env.reset(random.split());
            Environment.Step s;
            double sum = 0;
            int n = 0;
            do {
                int a = Math.round(act(o, true)[0]);
                s = env.step(a);
                obs.add(o);
                acts.add(a);
                rewards.add(s.reward());
                sum += s.reward();
                o = s.observation();
                n++;
                total++;
            } while (!s.done() && total < steps);
            remember(obs, acts, rewards, s.terminated() ? 0 : truncationValue);
            returns.add(sum);
            Episode e = new Episode(++episodes, n, sum, total);
            result.add(e);
            result.steps(total);
            listener.onEpisode(e, result);
        }
        return result;
    }

    /**
     * Saves the memory.
     *
     * @param path the file
     * @throws IOException if writing fails
     */
    public void save(Path path) throws IOException {
        try (var out = new java.io.BufferedOutputStream(Files.newOutputStream(path))) {
            for (KnnIndex m : memory) {
                m.write(out);
            }
        }
    }

    /**
     * Loads a memory {@link #save} wrote, for the same environment.
     *
     * @param path the file
     * @throws IOException if reading fails
     */
    public void load(Path path) throws IOException {
        try (var in = new java.io.BufferedInputStream(Files.newInputStream(path))) {
            for (int a = 0; a < actions; a++) {
                memory[a] = KnnIndex.read(in);
            }
        }
    }

    @Override
    public String toString() {
        return "EpisodicControl(" + memories() + " memories)";
    }
}
