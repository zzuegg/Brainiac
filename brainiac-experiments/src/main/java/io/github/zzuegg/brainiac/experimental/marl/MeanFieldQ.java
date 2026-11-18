package io.github.zzuegg.brainiac.experimental.marl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.rl.RlTensors;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Mean-field Q-learning (Yang et al., ICML 2018,
 * <a href="https://arxiv.org/abs/1802.05438">arXiv:1802.05438</a>): every agent's Q-function
 * depends on its own action and the mean action of its neighbours, {@code Q(o_i, a_i, abar_i)},
 * so its input stays the same size however many agents there are. One network is shared by all
 * agents and trained on per-agent rewards with the mean-field target
 * {@code r_i + gamma sum_a pi(a | o'_i, abar_i) Q_target(o'_i, a, abar_i)} under the Boltzmann
 * policy {@code pi ~ exp(Q / tau)}.
 *
 * <p>Acting runs the Boltzmann consistency iteration: starting from the last step's mean
 * actions, every agent's policy is computed from its neighbours' mean, the means move halfway
 * to the neighbours' policies (damped, so that anti-coordination does not oscillate), and so on for a few rounds; the actions are then sampled and the
 * realised neighbour means are stored with the transition. {@link #independent()} drops the
 * mean action from the input, which gives independent Q-learners with parameter sharing.
 *
 * <p>{@link MeanFieldActorCritic} trains an explicit policy against the same mean-field critic.
 */
public final class MeanFieldQ {

    private record Sample(float[] obs, float[] mean, int action, float reward, float[] next,
            boolean terminated) {
    }

    private final int agents;
    private final int obsSize;
    private final int actions;
    private final int[][] neighbours;
    private final Model q;
    private final Model qTarget;
    private final Model actor;
    private final Optimizer optimizer = Optimizer.adam(3e-3f);
    private final Optimizer actorOptimizer = Optimizer.adam(3e-3f);
    private boolean independent;
    private double tau = 0.5;
    private double tauEnd = 0.05;
    private long tauSteps = 400;
    private int iterations = 2;
    private float gamma = 0.9f;
    private float entropy = 0.01f;
    private int batchSize = 128;
    private int bufferSize = 20_000;
    private int targetEvery = 50;
    private SplittableRandom random = new SplittableRandom(0);
    private final List<Sample> buffer = new ArrayList<>();
    private int position;
    private float[][] lastMean;
    private long steps;
    private long updates;

    MeanFieldQ(LocalRewardEnvironment env, RlNetworkFactory nets, boolean withActor, long seed) {
        this.agents = env.agents();
        this.obsSize = env.observationSize();
        this.actions = env.actions();
        this.neighbours = new int[agents][];
        for (int i = 0; i < agents; i++) {
            neighbours[i] = env.neighbours(i);
        }
        q = nets.create(obsSize + actions, actions, seed);
        qTarget = nets.create(obsSize + actions, actions, seed);
        RlTensors.copy(q.parameters(), qTarget.parameters());
        actor = withActor ? nets.create(obsSize, actions, seed + 1) : null;
        this.random = new SplittableRandom(seed);
    }

    /**
     * @param env the environment (for sizes and neighbourhoods)
     * @param networks builds the shared Q-network
     * @param seed the seed
     * @return the learner
     */
    public static MeanFieldQ of(LocalRewardEnvironment env, RlNetworkFactory networks, long seed) {
        return new MeanFieldQ(Objects.requireNonNull(env), Objects.requireNonNull(networks), false, seed);
    }

    /**
     * Drops the mean action from the Q-function's input: independent Q-learners with a shared
     * network, the baseline.
     *
     * @return this learner
     */
    public MeanFieldQ independent() {
        this.independent = true;
        return this;
    }

    /**
     * @param start the Boltzmann temperature at the start
     * @param end the final temperature
     * @param anneal steps over which it falls linearly
     * @return this learner
     */
    public MeanFieldQ temperature(double start, double end, long anneal) {
        this.tau = start;
        this.tauEnd = end;
        this.tauSteps = anneal;
        return this;
    }

    /** @param rounds consistency iterations per step (0 uses the last step's means) @return this learner */
    public MeanFieldQ consistencyIterations(int rounds) {
        this.iterations = rounds;
        return this;
    }

    /** @param discount gamma @return this learner */
    public MeanFieldQ discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param seed the exploration seed @return this learner */
    public MeanFieldQ seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return gradient updates taken */
    public long updates() {
        return updates;
    }

    private double temperature() {
        double f = Math.min(1, steps / (double) Math.max(1, tauSteps));
        return tau + f * (tauEnd - tau);
    }

    private float[] input(float[] obs, float[] mean) {
        float[] x = new float[obsSize + actions];
        System.arraycopy(obs, 0, x, 0, obsSize);
        if (!independent) {
            System.arraycopy(mean, 0, x, obsSize, actions);
        }
        return x;
    }

    private float[][] policies(float[][] obs, float[][] mean, double temp) {
        int n = obs.length;
        float[] logits;
        if (actor != null) {
            float[] flat = new float[n * obsSize];
            for (int i = 0; i < n; i++) {
                System.arraycopy(obs[i], 0, flat, i * obsSize, obsSize);
            }
            logits = actor.forward(Var.of(Tensor.of(Shape.of(n, obsSize), flat)), false).value().toArray();
            temp = 1;
        } else {
            float[] flat = new float[n * (obsSize + actions)];
            for (int i = 0; i < n; i++) {
                System.arraycopy(input(obs[i], mean[i]), 0, flat, i * (obsSize + actions), obsSize + actions);
            }
            logits = q.forward(Var.of(Tensor.of(Shape.of(n, obsSize + actions), flat)), false).value().toArray();
        }
        float[][] p = new float[n][actions];
        for (int i = 0; i < n; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int a = 0; a < actions; a++) {
                max = Math.max(max, logits[i * actions + a] / temp);
            }
            double z = 0;
            for (int a = 0; a < actions; a++) {
                double e = Math.exp(logits[i * actions + a] / temp - max);
                p[i][a] = (float) e;
                z += e;
            }
            for (int a = 0; a < actions; a++) {
                p[i][a] /= (float) z;
            }
        }
        return p;
    }

    private float[][] neighbourMeans(float[][] perAgent) {
        float[][] m = new float[agents][actions];
        for (int i = 0; i < agents; i++) {
            for (int j : neighbours[i]) {
                for (int a = 0; a < actions; a++) {
                    m[i][a] += perAgent[j][a] / neighbours[i].length;
                }
            }
        }
        return m;
    }

    /** Forgets the last step's mean actions; call at the start of an episode. */
    public void beginEpisode() {
        lastMean = null;
    }

    /**
     * @param observations one per agent
     * @param explore sample from the Boltzmann policy rather than take the most likely action
     * @return the joint action
     */
    public int[] act(float[][] observations, boolean explore) {
        if (lastMean == null) {
            lastMean = new float[agents][actions];
            for (float[] row : lastMean) {
                java.util.Arrays.fill(row, 1f / actions);
            }
        }
        double temp = explore ? temperature() : tauEnd;
        float[][] mean = lastMean;
        float[][] p = policies(observations, mean, temp);
        for (int k = 0; k < iterations && actor == null; k++) {
            float[][] next = neighbourMeans(p);
            for (int i = 0; i < agents; i++) {
                for (int a = 0; a < actions; a++) {
                    next[i][a] = 0.5f * (next[i][a] + mean[i][a]);
                }
            }
            mean = next;
            p = policies(observations, mean, temp);
        }
        int[] a = new int[agents];
        float[][] hot = new float[agents][actions];
        for (int i = 0; i < agents; i++) {
            a[i] = explore ? sample(p[i]) : RlTensors.argmax(p[i]);
            hot[i][a[i]] = 1;
        }
        lastMean = neighbourMeans(hot);
        return a;
    }

    private int sample(float[] p) {
        double u = random.nextDouble();
        double c = 0;
        for (int i = 0; i < p.length; i++) {
            c += p[i];
            if (u < c) {
                return i;
            }
        }
        return p.length - 1;
    }

    /**
     * Trains for a number of joint steps, one gradient update per step.
     *
     * @param env the environment
     * @param totalSteps joint steps
     * @return the mean team return per episode over training
     */
    public double fit(LocalRewardEnvironment env, long totalSteps) {
        float[][] obs = env.reset(random.split());
        beginEpisode();
        double episodeReturn = 0;
        double sum = 0;
        int episodes = 0;
        for (long t = 0; t < totalSteps; t++) {
            int[] a = act(obs, true);
            float[][] mean = lastMean;
            LocalRewardEnvironment.Step s = env.step(a);
            float[] r = env.agentRewards();
            for (int i = 0; i < agents; i++) {
                remember(new Sample(obs[i], mean[i], a[i], r[i], s.observations()[i], s.terminated()));
            }
            steps++;
            episodeReturn += s.reward();
            if (s.done()) {
                sum += episodeReturn;
                episodes++;
                episodeReturn = 0;
                obs = env.reset(random.split());
                beginEpisode();
            } else {
                obs = s.observations();
            }
            if (buffer.size() >= batchSize) {
                learn();
            }
        }
        return episodes == 0 ? episodeReturn : sum / episodes;
    }

    private void remember(Sample s) {
        if (buffer.size() < bufferSize) {
            buffer.add(s);
        } else {
            buffer.set(position, s);
            position = (position + 1) % bufferSize;
        }
    }

    private void learn() {
        int b = batchSize;
        int w = obsSize + actions;
        float[] x = new float[b * w];
        float[] nx = new float[b * w];
        float[] o = new float[b * obsSize];
        float[] no = new float[b * obsSize];
        float[] acts = new float[b];
        float[] r = new float[b];
        float[] notDone = new float[b];
        for (int i = 0; i < b; i++) {
            Sample s = buffer.get(random.nextInt(buffer.size()));
            System.arraycopy(input(s.obs(), s.mean()), 0, x, i * w, w);
            System.arraycopy(input(s.next(), s.mean()), 0, nx, i * w, w);
            System.arraycopy(s.obs(), 0, o, i * obsSize, obsSize);
            System.arraycopy(s.next(), 0, no, i * obsSize, obsSize);
            acts[i] = s.action();
            r[i] = s.reward();
            notDone[i] = s.terminated() ? 0 : 1;
        }
        Tensor nextQ = qTarget.forward(Var.of(Tensor.of(Shape.of(b, w), nx)), false).value();
        Tensor nextPi;
        if (actor != null) {
            nextPi = RlTensors.softmax(actor.forward(Var.of(Tensor.of(Shape.of(b, obsSize), no)), false)).value();
        } else {
            nextPi = RlTensors.softmax(Var.of(nextQ.div((float) tauEnd))).value();
        }
        Tensor v = nextQ.mul(nextPi).sum(1, true);
        Tensor y = Tensor.of(Shape.of(b, 1), r).add(Tensor.of(Shape.of(b, 1), notDone).mul(gamma).mul(v));
        Var qAll = q.forward(Var.of(Tensor.of(Shape.of(b, w), x)), true);
        Var chosen = qAll.mul(RlTensors.oneHot(acts, actions, q.device())).sum(1, true);
        Var loss = RlTensors.huber(chosen.sub(y), 1f).mean();
        loss.backward();
        RlTensors.clip(q.parameters(), 10f);
        optimizer.step(q.parameters());
        if (actor != null) {
            Tensor qValues = qAll.value();
            Var logPi = RlTensors.logSoftmax(actor.forward(Var.of(Tensor.of(Shape.of(b, obsSize), o)), true));
            Var pi = logPi.exp();
            Var objective = pi.mul(qValues).sum(1, true).sub(pi.mul(logPi).sum(1, true).mul(entropy)).mean();
            objective.neg().backward();
            RlTensors.clip(actor.parameters(), 10f);
            actorOptimizer.step(actor.parameters());
        }
        if (++updates % targetEvery == 0) {
            RlTensors.copy(q.parameters(), qTarget.parameters());
        }
    }

    /**
     * Plays one greedy episode.
     *
     * @param env the environment
     * @param seed the episode seed
     * @return the team return
     */
    public double evaluate(LocalRewardEnvironment env, long seed) {
        float[][] obs = env.reset(new SplittableRandom(seed));
        beginEpisode();
        double total = 0;
        LocalRewardEnvironment.Step s;
        do {
            s = env.step(act(obs, false));
            total += s.reward();
            obs = s.observations();
        } while (!s.done());
        return total;
    }

    @Override
    public String toString() {
        return actor != null ? "MF-AC" : independent ? "IQL" : "MF-Q";
    }
}
