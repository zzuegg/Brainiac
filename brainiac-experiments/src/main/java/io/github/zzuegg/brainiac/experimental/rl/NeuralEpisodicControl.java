package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Episode;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Neural Episodic Control (Pritzel et al., <a href="https://arxiv.org/abs/1703.01988">1703.01988</a>):
 * a fast-learning value function made of one <em>differentiable neural dictionary</em> per
 * action. Each dictionary stores embeddings of visited states with their N-step returns; the
 * value of a new state is the kernel-weighted mean ({@code 1 / (distance² + delta)}) of the
 * values of its {@code k} nearest keys. Writing a return into the dictionary changes the
 * estimate at once, so the agent learns from few episodes; a revisited key is moved towards
 * the new return with a high learning rate. The embedding network is trained slowly by
 * backpropagating the lookup error, and the stored keys are re-embedded after every update so
 * they never go stale (the paper instead updates the keys by gradient).
 *
 * <p>{@link #fit} returns an {@link RlResult} without episodes (its builder is package-private
 * in the core); listeners are told about every episode.
 */
public final class NeuralEpisodicControl implements Learner {

    private final int actions;
    private final Model embedding;
    private final Dictionary[] dictionaries;
    private Optimizer optimizer = Optimizer.adam(1e-3f);
    private int neighbours = 16;
    private int horizon = 20;
    private float gamma = 0.99f;
    private float writeRate = 0.5f;
    private double epsilon = 0.05;
    private int trainEvery = 32;
    private final List<float[]> replayObs = new ArrayList<>();
    private final List<Integer> replayAct = new ArrayList<>();
    private final List<Float> replayTarget = new ArrayList<>();
    private final List<Double> returns = new ArrayList<>();
    private SplittableRandom random;

    private NeuralEpisodicControl(int observations, int actions, int keySize, int hidden, int capacity, long seed) {
        this.actions = actions;
        this.embedding = AgentKit.mlp(observations, keySize, seed, hidden);
        this.dictionaries = new Dictionary[actions];
        for (int a = 0; a < actions; a++) {
            dictionaries[a] = new Dictionary(capacity);
        }
        this.random = new SplittableRandom(seed);
    }

    /**
     * @param observations observation size
     * @param actions the number of discrete actions
     * @param keySize the embedding size
     * @param hidden hidden width of the embedding network
     * @param capacity entries per dictionary (the oldest is replaced when full)
     * @param seed the seed
     * @return the agent
     */
    public static NeuralEpisodicControl of(int observations, int actions, int keySize, int hidden, int capacity,
            long seed) {
        return new NeuralEpisodicControl(observations, actions, keySize, hidden, capacity, seed);
    }

    /**
     * @param k neighbours per lookup
     * @return this agent
     */
    public NeuralEpisodicControl neighbours(int k) {
        this.neighbours = k;
        return this;
    }

    /**
     * @param n steps of the N-step return
     * @param discount the discount
     * @return this agent
     */
    public NeuralEpisodicControl returns(int n, float discount) {
        this.horizon = n;
        this.gamma = discount;
        return this;
    }

    /**
     * @param e the exploration rate
     * @return this agent
     */
    public NeuralEpisodicControl epsilon(double e) {
        this.epsilon = e;
        return this;
    }

    /**
     * @param rate how far a revisited key's value moves towards the new return
     * @return this agent
     */
    public NeuralEpisodicControl writeRate(float rate) {
        this.writeRate = rate;
        return this;
    }

    /**
     * @param o the optimizer of the embedding network
     * @return this agent
     */
    public NeuralEpisodicControl optimizer(Optimizer o) {
        this.optimizer = o;
        return this;
    }

    /** @return the number of stored entries over all actions */
    public int size() {
        int n = 0;
        for (Dictionary d : dictionaries) {
            n += d.size;
        }
        return n;
    }

    /** @return the return of every training episode */
    public List<Double> episodeReturns() {
        return List.copyOf(returns);
    }

    /**
     * @param observation a state
     * @return the estimated value of every action
     */
    public float[] values(float[] observation) {
        float[] h = AgentKit.predict(embedding, observation);
        float[] q = new float[actions];
        for (int a = 0; a < actions; a++) {
            q[a] = dictionaries[a].lookup(h, neighbours);
        }
        return q;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        if (explore && random.nextDouble() < epsilon) {
            return new float[] {random.nextInt(actions)};
        }
        return new float[] {AgentKit.argmax(values(observation))};
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Discrete d) || d.n() != actions) {
            throw new IllegalArgumentException("needs " + actions + " discrete actions");
        }
        RlResult result = new RlResult();
        List<float[]> obs = new ArrayList<>();
        List<Integer> acts = new ArrayList<>();
        List<Float> rewards = new ArrayList<>();
        float[] o = env.reset(random.split());
        double episodeReturn = 0;
        for (long t = 1; t <= steps; t++) {
            int a = Math.round(act(o, true)[0]);
            Environment.Step s = env.step(new float[] {a});
            obs.add(o);
            acts.add(a);
            rewards.add(s.reward());
            episodeReturn += s.reward();
            if (obs.size() > horizon) {
                write(obs, acts, rewards, 0, horizon, max(values(s.observation())));
                obs.removeFirst();
                acts.removeFirst();
                rewards.removeFirst();
            }
            if (s.done()) {
                float tail = s.terminated() ? 0 : max(values(s.observation()));
                while (!obs.isEmpty()) {
                    write(obs, acts, rewards, 0, obs.size(), tail);
                    obs.removeFirst();
                    acts.removeFirst();
                    rewards.removeFirst();
                }
                returns.add(episodeReturn);
                if (listener != null) {
                    listener.onEpisode(new Episode(returns.size(), 0, episodeReturn, t), result);
                }
                episodeReturn = 0;
                o = env.reset(random.split());
            } else {
                o = s.observation();
            }
            if (t % trainEvery == 0 && replayObs.size() >= 64) {
                trainEmbedding();
            }
        }
        return result;
    }

    private static float max(float[] v) {
        float m = v[0];
        for (float x : v) {
            m = Math.max(m, x);
        }
        return m;
    }

    private void write(List<float[]> obs, List<Integer> acts, List<Float> rewards, int from, int n, float tail) {
        double g = tail;
        for (int i = from + n - 1; i >= from; i--) {
            g = rewards.get(i) + gamma * g;
        }
        float[] x = obs.get(from);
        int a = acts.get(from);
        dictionaries[a].write(x, AgentKit.predict(embedding, x), (float) g, writeRate);
        replayObs.add(x);
        replayAct.add(a);
        replayTarget.add((float) g);
        if (replayObs.size() > 20_000) {
            replayObs.removeFirst();
            replayAct.removeFirst();
            replayTarget.removeFirst();
        }
    }

    /** One gradient step of the embedding through the dictionaries, then a key refresh. */
    private void trainEmbedding() {
        int batch = 32;
        List<Var> losses = new ArrayList<>();
        for (int a = 0; a < actions; a++) {
            Dictionary d = dictionaries[a];
            if (d.size < neighbours + 1) {
                continue;
            }
            List<float[]> xs = new ArrayList<>();
            List<Float> ys = new ArrayList<>();
            for (int i = 0; i < batch; i++) {
                int j = random.nextInt(replayObs.size());
                if (replayAct.get(j) == a) {
                    xs.add(replayObs.get(j));
                    ys.add(replayTarget.get(j));
                }
            }
            if (xs.isEmpty()) {
                continue;
            }
            float[] y = new float[ys.size()];
            for (int i = 0; i < y.length; i++) {
                y[i] = ys.get(i);
            }
            Var h = embedding.forward(Var.of(AgentKit.rows(xs)), true);
            losses.add(d.differentiableLookup(h, neighbours).sub(AgentKit.column(y)).square().mean());
        }
        if (losses.isEmpty()) {
            return;
        }
        Var loss = losses.getFirst();
        for (int i = 1; i < losses.size(); i++) {
            loss = loss.add(losses.get(i));
        }
        loss.backward();
        optimizer.step(embedding.parameters());
        for (Dictionary d : dictionaries) {
            d.refresh(embedding);
        }
    }

    /** One action's dictionary: keys, the observations they embed, and values. */
    private static final class Dictionary {
        private static final float DELTA = 1e-3f;
        private final float[][] keys;
        private final float[][] sources;
        private final float[] values;
        private int size;
        private int next;

        Dictionary(int capacity) {
            this.keys = new float[capacity][];
            this.sources = new float[capacity][];
            this.values = new float[capacity];
        }

        private static float distance(float[] a, float[] b) {
            float s = 0;
            for (int i = 0; i < a.length; i++) {
                float d = a[i] - b[i];
                s += d * d;
            }
            return s;
        }

        private int[] nearest(float[] h, int k) {
            int n = Math.min(k, size);
            int[] best = new int[n];
            float[] bestD = new float[n];
            Arrays.fill(bestD, Float.POSITIVE_INFINITY);
            for (int i = 0; i < size; i++) {
                float dist = distance(h, keys[i]);
                if (dist < bestD[n - 1]) {
                    int j = n - 1;
                    while (j > 0 && bestD[j - 1] > dist) {
                        bestD[j] = bestD[j - 1];
                        best[j] = best[j - 1];
                        j--;
                    }
                    bestD[j] = dist;
                    best[j] = i;
                }
            }
            return best;
        }

        float lookup(float[] h, int k) {
            if (size == 0) {
                return 0;
            }
            double num = 0;
            double den = 0;
            for (int i : nearest(h, k)) {
                double w = 1 / (distance(h, keys[i]) + DELTA);
                num += w * values[i];
                den += w;
            }
            return (float) (num / den);
        }

        void write(float[] source, float[] key, float value, float rate) {
            if (size > 0) {
                int i = nearest(key, 1)[0];
                if (distance(key, keys[i]) < 1e-8f) {
                    values[i] += rate * (value - values[i]);
                    return;
                }
            }
            keys[next] = key;
            sources[next] = source;
            values[next] = value;
            next = (next + 1) % keys.length;
            size = Math.min(size + 1, keys.length);
        }

        /** The lookup for a batch of embeddings, differentiable in the embeddings. */
        Var differentiableLookup(Var h, int k) {
            int b = h.shape().dim(0);
            int dim = h.shape().dim(1);
            float[] flat = new float[size * dim];
            for (int i = 0; i < size; i++) {
                System.arraycopy(keys[i], 0, flat, i * dim, dim);
            }
            Tensor keyT = Tensor.of(Shape.of(size, dim), flat);
            float[] hv = h.value().toArray();
            float[] mask = new float[b * size];
            for (int r = 0; r < b; r++) {
                for (int i : nearest(Arrays.copyOfRange(hv, r * dim, (r + 1) * dim), k)) {
                    mask[r * size + i] = 1;
                }
            }
            Var hk = h.matmul(keyT.transpose());
            Tensor kk = keyT.square().sum(1, true).transpose();
            Var dist = h.square().sum(1, true).sub(hk.mul(2f)).add(kk).relu();
            Var w = dist.add(DELTA).pow(-1f).mul(Tensor.of(Shape.of(b, size), mask));
            Tensor v = Tensor.of(Shape.of(size, 1), Arrays.copyOf(values, size));
            return w.matmul(v).div(w.sum(1, true));
        }

        void refresh(Model embedding) {
            if (size == 0) {
                return;
            }
            List<float[]> xs = new ArrayList<>(Arrays.asList(sources).subList(0, size));
            float[] all = embedding.forward(Var.of(AgentKit.rows(xs)), false).value().toArray();
            int dim = all.length / size;
            for (int i = 0; i < size; i++) {
                keys[i] = Arrays.copyOfRange(all, i * dim, (i + 1) * dim);
            }
        }
    }
}
