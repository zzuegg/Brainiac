package io.github.zzuegg.brainiac.experimental.explore;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.rl.RlTensors;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Tracker;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Bootstrapped DQN with randomised prior functions (Osband et al., NeurIPS 2016,
 * <a href="https://arxiv.org/abs/1602.04621">arXiv:1602.04621</a>; NeurIPS 2018,
 * <a href="https://arxiv.org/abs/1806.03335">arXiv:1806.03335</a>). {@code K} Q-networks are
 * trained on bootstrapped views of one replay buffer: every transition carries a Bernoulli
 * mask per head, and head {@code k} learns only from the transitions its mask keeps. At the
 * start of each episode one head is chosen at random and followed greedily for the whole
 * episode, which explores in a temporally consistent way, like Thompson sampling over value
 * functions. Each head adds a fixed, randomly initialised prior network scaled by
 * {@code beta}: {@code Q_k = f_k + beta p_k}, which keeps the heads apart where there is no
 * data; with {@code beta = 0} the heads tend to collapse to the same function.
 *
 * <p>Greedy evaluation acts on the mean of the heads.
 */
public final class BootstrappedDqn implements Learner {

    private record Sample(float[] obs, int action, float reward, float[] next, boolean terminal, boolean[] mask) {
    }

    private final int obsSize;
    private final int actions;
    private final int heads;
    private final List<Model> nets = new ArrayList<>();
    private final List<Model> targets = new ArrayList<>();
    private final List<Model> priors = new ArrayList<>();
    private final List<Optimizer> optimizers = new ArrayList<>();
    private float priorScale = 3f;
    private double maskProbability = 0.5;
    private float gamma = 0.99f;
    private int batchSize = 32;
    private int bufferSize = 20_000;
    private int learningStarts = 64;
    private int trainEvery = 1;
    private int targetEvery = 100;
    private float lr = 1e-3f;
    private SplittableRandom random = new SplittableRandom(0);
    private final List<Sample> buffer = new ArrayList<>();
    private int position;
    private int active;
    private long updates;
    private final RlNetworkFactory factory;

    private BootstrappedDqn(int obsSize, int actions, int heads, RlNetworkFactory factory) {
        this.obsSize = obsSize;
        this.actions = actions;
        this.heads = heads;
        this.factory = factory;
    }

    /**
     * @param observations values per observation
     * @param actions discrete actions
     * @param heads the number of heads {@code K} (10 in the papers)
     * @param networks builds each head's network and prior
     * @return the learner
     */
    public static BootstrappedDqn of(int observations, int actions, int heads, RlNetworkFactory networks) {
        return new BootstrappedDqn(observations, actions, heads, Objects.requireNonNull(networks));
    }

    /** @param beta the prior scale, 0 for plain bootstrapped DQN @return this learner */
    public BootstrappedDqn priorScale(float beta) {
        this.priorScale = beta;
        return this;
    }

    /** @param p the probability a head sees a transition @return this learner */
    public BootstrappedDqn maskProbability(double p) {
        this.maskProbability = p;
        return this;
    }

    /** @param discount gamma @return this learner */
    public BootstrappedDqn discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param rate the learning rate @return this learner */
    public BootstrappedDqn learningRate(float rate) {
        this.lr = rate;
        return this;
    }

    /** @param steps environment steps between updates @return this learner */
    public BootstrappedDqn trainEvery(int steps) {
        this.trainEvery = steps;
        return this;
    }

    /** @param seed the seed @return this learner */
    public BootstrappedDqn seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return gradient updates taken (one per head per update round) */
    public long updates() {
        return updates;
    }

    private void build() {
        if (!nets.isEmpty()) {
            return;
        }
        long base = random.nextLong();
        for (int k = 0; k < heads; k++) {
            nets.add(factory.create(obsSize, actions, base + 3L * k));
            targets.add(factory.create(obsSize, actions, base + 3L * k));
            RlTensors.copy(nets.get(k).parameters(), targets.get(k).parameters());
            priors.add(factory.create(obsSize, actions, base + 3L * k + 1));
            optimizers.add(Optimizer.adam(lr));
        }
    }

    private float[] q(int k, float[] obs) {
        float[] f = RlTensors.row(nets.get(k), obs);
        if (priorScale != 0) {
            float[] p = RlTensors.row(priors.get(k), obs);
            for (int a = 0; a < f.length; a++) {
                f[a] += priorScale * p[a];
            }
        }
        return f;
    }

    @Override
    public void startEpisode() {
        active = random.nextInt(heads);
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        build();
        if (explore) {
            return new float[] {RlTensors.argmax(q(active, observation))};
        }
        float[] mean = new float[actions];
        for (int k = 0; k < heads; k++) {
            float[] v = q(k, observation);
            for (int a = 0; a < actions; a++) {
                mean[a] += v[a] / heads;
            }
        }
        return new float[] {RlTensors.argmax(mean)};
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        build();
        Tracker t = new Tracker(listener);
        float[] obs = env.reset(random.split());
        startEpisode();
        long step = 0;
        while (t.steps() < steps) {
            int a = Math.round(act(obs, true)[0]);
            Environment.Step s = env.step(a);
            boolean[] mask = new boolean[heads];
            for (int k = 0; k < heads; k++) {
                mask[k] = random.nextDouble() < maskProbability;
            }
            Sample sample = new Sample(obs, a, s.reward(), s.observation(), s.terminated(), mask);
            if (buffer.size() < bufferSize) {
                buffer.add(sample);
            } else {
                buffer.set(position, sample);
                position = (position + 1) % bufferSize;
            }
            if (t.step(s)) {
                obs = env.reset(random.split());
                startEpisode();
            } else {
                obs = s.observation();
            }
            if (buffer.size() >= learningStarts && ++step % trainEvery == 0) {
                learn();
            }
        }
        return t.result();
    }

    private void learn() {
        int b = batchSize;
        float[] o = new float[b * obsSize];
        float[] n = new float[b * obsSize];
        float[] r = new float[b];
        float[] notDone = new float[b];
        float[] acts = new float[b];
        Sample[] picked = new Sample[b];
        for (int i = 0; i < b; i++) {
            Sample s = buffer.get(random.nextInt(buffer.size()));
            picked[i] = s;
            System.arraycopy(s.obs(), 0, o, i * obsSize, obsSize);
            System.arraycopy(s.next(), 0, n, i * obsSize, obsSize);
            r[i] = s.reward();
            notDone[i] = s.terminal() ? 0 : 1;
            acts[i] = s.action();
        }
        Tensor obsT = Tensor.of(Shape.of(b, obsSize), o);
        Tensor nextT = Tensor.of(Shape.of(b, obsSize), n);
        Tensor hot = RlTensors.oneHot(acts, actions, nets.get(0).device());
        for (int k = 0; k < heads; k++) {
            float[] m = new float[b];
            float kept = 0;
            for (int i = 0; i < b; i++) {
                m[i] = picked[i].mask()[k] ? 1 : 0;
                kept += m[i];
            }
            if (kept == 0) {
                continue;
            }
            Tensor priorNext = priorScale == 0 ? null
                    : priors.get(k).forward(Var.of(nextT), false).value().mul(priorScale);
            Tensor nextQ = targets.get(k).forward(Var.of(nextT), false).value();
            Tensor nextOnline = nets.get(k).forward(Var.of(nextT), false).value();
            if (priorNext != null) {
                nextQ = nextQ.add(priorNext);
                nextOnline = nextOnline.add(priorNext);
            }
            float[] pick = nextOnline.argmax(1).to(DType.FLOAT32).toArray();
            Tensor boot = nextQ.mul(RlTensors.oneHot(pick, actions, nextQ.device())).sum(1, true);
            Tensor y = Tensor.of(Shape.of(b, 1), r).add(Tensor.of(Shape.of(b, 1), notDone).mul(gamma).mul(boot));
            Var pred = nets.get(k).forward(Var.of(obsT), true);
            if (priorScale != 0) {
                pred = pred.add(priors.get(k).forward(Var.of(obsT), false).value().mul(priorScale));
            }
            Var chosen = pred.mul(hot).sum(1, true);
            Var loss = RlTensors.huber(chosen.sub(y), 1f).mul(Tensor.of(Shape.of(b, 1), m)).sum().div(kept);
            loss.backward();
            RlTensors.clip(nets.get(k).parameters(), 10f);
            optimizers.get(k).step(nets.get(k).parameters());
            updates++;
        }
        if (updates / heads % targetEvery == 0) {
            for (int k = 0; k < heads; k++) {
                RlTensors.copy(nets.get(k).parameters(), targets.get(k).parameters());
            }
        }
    }

    @Override
    public String toString() {
        return "BootstrappedDqn(K=" + heads + ", beta=" + priorScale + ")";
    }
}
