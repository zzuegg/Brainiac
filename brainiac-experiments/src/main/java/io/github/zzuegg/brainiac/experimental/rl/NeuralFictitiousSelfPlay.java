package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ReservoirReplay;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Neural fictitious self-play: fictitious play (each player best-responds to the opponent's
 * <em>average</em> past strategy, which converges in two-player zero-sum games, Brown 1951)
 * with both halves learned from sampled play. Each player has a best-response Q-network,
 * trained by DQN on a circular replay of its own transitions, and an average-policy network,
 * trained by supervised learning on a {@link ReservoirReplay} of the actions its
 * best-response network took (the reservoir keeps a uniform sample of all of them, so the
 * network imitates the true average). In each episode a player acts with its
 * best-response network with probability {@code eta} (the anticipatory parameter) and with its
 * average network otherwise. The average networks approach a Nash equilibrium; the
 * best-response networks alone are DQN self-play, which cycles.
 *
 * <p>Heinrich and Silver 2016, "Deep Reinforcement Learning from Self-Play in
 * Imperfect-Information Games", <a href="https://arxiv.org/abs/1603.01121">arXiv:1603.01121</a>.
 */
public final class NeuralFictitiousSelfPlay {

    private final GameTree tree;
    private final int features;
    private final int actions;
    private final Model[] q = new Model[2];
    private final Model[] target = new Model[2];
    private final Model[] average = new Model[2];
    private final Optimizer[] rlOpt = new Optimizer[2];
    private final Optimizer[] slOpt = new Optimizer[2];
    private final ReservoirReplay[] reservoir = new ReservoirReplay[2];
    private final Ring[] replay = new Ring[2];
    private final SplittableRandom random;
    private double eta = 0.1;
    private double epsilon = 0.06;
    private int batch = 128;
    private int trainEvery = 32;
    private int targetEvery = 300;
    private long steps;
    private long updates;

    /** A circular replay of (features, action, reward, next features, next legal, done). */
    private static final class Ring {
        final float[][] s;
        final int[] a;
        final float[] r;
        final float[][] next;
        final boolean[][] legal;
        final boolean[] done;
        int size;
        int pos;

        Ring(int capacity) {
            s = new float[capacity][];
            a = new int[capacity];
            r = new float[capacity];
            next = new float[capacity][];
            legal = new boolean[capacity][];
            done = new boolean[capacity];
        }

        void add(float[] state, int action, float reward, float[] nextState, boolean[] nextLegal, boolean end) {
            s[pos] = state;
            a[pos] = action;
            r[pos] = reward;
            next[pos] = nextState;
            legal[pos] = nextLegal;
            done[pos] = end;
            pos = (pos + 1) % s.length;
            size = Math.min(size + 1, s.length);
        }
    }

    private NeuralFictitiousSelfPlay(GameTree tree, int hidden, long seed) {
        this.tree = tree;
        this.features = tree.game().featureSize();
        this.actions = tree.maxActions();
        this.random = new SplittableRandom(seed);
        for (int p = 0; p < 2; p++) {
            q[p] = Nets.mlp(features, actions, Activation.RELU, seed + 10 * p, hidden);
            target[p] = Nets.mlp(features, actions, Activation.RELU, seed + 10 * p, hidden);
            RlTensors.copy(q[p].parameters(), target[p].parameters());
            average[p] = Nets.mlp(features, actions, Activation.RELU, seed + 10 * p + 5, hidden);
            rlOpt[p] = Optimizer.adam(1e-3f);
            slOpt[p] = Optimizer.adam(1e-3f);
            reservoir[p] = new ReservoirReplay(20_000, seed + p);
            replay[p] = new Ring(20_000);
        }
    }

    /**
     * @param tree the game
     * @param hidden hidden units of every network
     * @param seed the seed
     * @return a learner
     */
    public static NeuralFictitiousSelfPlay of(GameTree tree, int hidden, long seed) {
        return new NeuralFictitiousSelfPlay(Objects.requireNonNull(tree), hidden, seed);
    }

    /**
     * @param anticipatory the probability {@code eta} of acting with the best-response network
     *     in an episode; 1 gives plain DQN self-play
     * @return this
     */
    public NeuralFictitiousSelfPlay anticipatory(double anticipatory) {
        this.eta = anticipatory;
        return this;
    }

    /**
     * @param e the exploration rate of the best-response networks
     * @return this
     */
    public NeuralFictitiousSelfPlay epsilon(double e) {
        this.epsilon = e;
        return this;
    }

    /**
     * @param rl the best-response (DQN) learning rate
     * @param sl the average-policy (supervised) learning rate
     * @return this
     */
    public NeuralFictitiousSelfPlay learningRates(float rl, float sl) {
        for (int p = 0; p < 2; p++) {
            rlOpt[p] = Optimizer.adam(rl);
            slOpt[p] = Optimizer.adam(sl);
        }
        return this;
    }

    /**
     * @param stepsPerUpdate decisions between gradient updates of all four networks
     * @return this
     */
    public NeuralFictitiousSelfPlay trainEvery(int stepsPerUpdate) {
        this.trainEvery = stepsPerUpdate;
        return this;
    }

    /** @return decisions taken so far (environment steps) */
    public long steps() {
        return steps;
    }

    /**
     * Plays and learns.
     *
     * @param episodes episodes of self-play
     * @return this
     */
    public NeuralFictitiousSelfPlay fit(int episodes) {
        for (int e = 0; e < episodes; e++) {
            episode();
        }
        return this;
    }

    private void episode() {
        boolean[] best = {eta >= 1 || random.nextDouble() < eta, eta >= 1 || random.nextDouble() < eta};
        float[][] prev = new float[2][];
        int[] prevAction = new int[2];
        int node = 0;
        while (!tree.terminal(node)) {
            int p = tree.player(node);
            if (p == ExtensiveFormGame.CHANCE) {
                node = tree.children(node)[tree.sampleChance(node, random)];
                continue;
            }
            int set = tree.infoSet(node);
            float[] f = tree.features(set);
            boolean[] legal = tree.legal(set);
            if (prev[p] != null) {
                replay[p].add(prev[p], prevAction[p], 0, f, legal, false);
            }
            int a;
            if (best[p]) {
                a = random.nextDouble() < epsilon ? randomLegal(legal) : greedy(RlTensors.row(q[p], f), legal);
                float[] y = new float[actions];
                y[a] = 1;
                reservoir[p].add(Tensor.of(Shape.of(1, features), f), Tensor.of(Shape.of(1, actions), y));
            } else {
                a = sampleSoftmax(RlTensors.row(average[p], f), legal);
            }
            prev[p] = f;
            prevAction[p] = a;
            node = tree.children(node)[a];
            if (++steps % trainEvery == 0) {
                train();
            }
        }
        double payoff = tree.payoff(node);
        for (int p = 0; p < 2; p++) {
            if (prev[p] != null) {
                replay[p].add(prev[p], prevAction[p], (float) (p == 0 ? payoff : -payoff), prev[p], tree.legal(0), true);
            }
        }
    }

    private int randomLegal(boolean[] legal) {
        int count = 0;
        for (boolean l : legal) {
            count += l ? 1 : 0;
        }
        int k = random.nextInt(count);
        for (int a = 0; a < legal.length; a++) {
            if (legal[a] && k-- == 0) {
                return a;
            }
        }
        return 0;
    }

    private static int greedy(float[] values, boolean[] legal) {
        int arg = -1;
        for (int a = 0; a < values.length; a++) {
            if (legal[a] && (arg < 0 || values[a] > values[arg])) {
                arg = a;
            }
        }
        return arg;
    }

    private int sampleSoftmax(float[] logits, boolean[] legal) {
        double[] p = maskedSoftmax(logits, 0, legal);
        double u = random.nextDouble();
        int last = 0;
        for (int a = 0; a < p.length; a++) {
            if (legal[a]) {
                last = a;
                u -= p[a];
                if (u <= 0) {
                    return a;
                }
            }
        }
        return last;
    }

    private static double[] maskedSoftmax(float[] logits, int offset, boolean[] legal) {
        double max = Double.NEGATIVE_INFINITY;
        for (int a = 0; a < legal.length; a++) {
            if (legal[a]) {
                max = Math.max(max, logits[offset + a]);
            }
        }
        double[] p = new double[legal.length];
        double z = 0;
        for (int a = 0; a < legal.length; a++) {
            p[a] = legal[a] ? Math.exp(logits[offset + a] - max) : 0;
            z += p[a];
        }
        for (int a = 0; a < legal.length; a++) {
            p[a] /= z;
        }
        return p;
    }

    private void train() {
        for (int p = 0; p < 2; p++) {
            Ring r = replay[p];
            if (r.size >= batch) {
                float[] xs = new float[batch * features];
                float[] nx = new float[batch * features];
                int[] idx = new int[batch];
                for (int i = 0; i < batch; i++) {
                    idx[i] = random.nextInt(r.size);
                    System.arraycopy(r.s[idx[i]], 0, xs, i * features, features);
                    System.arraycopy(r.next[idx[i]], 0, nx, i * features, features);
                }
                float[] nextQ = target[p].forward(Var.of(Tensor.of(Shape.of(batch, features), nx)), false).value()
                        .toArray();
                float[] y = new float[batch];
                float[] mask = new float[batch * actions];
                for (int i = 0; i < batch; i++) {
                    int k = idx[i];
                    float bootstrap = 0;
                    if (!r.done[k]) {
                        bootstrap = Float.NEGATIVE_INFINITY;
                        for (int a = 0; a < actions; a++) {
                            if (r.legal[k][a]) {
                                bootstrap = Math.max(bootstrap, nextQ[i * actions + a]);
                            }
                        }
                    }
                    y[i] = r.r[k] + bootstrap;
                    mask[i * actions + r.a[k]] = 1;
                }
                Var out = q[p].forward(Var.of(Tensor.of(Shape.of(batch, features), xs)), true);
                Var chosen = out.mul(Tensor.of(Shape.of(batch, actions), mask)).sum(1, true);
                RlTensors.huber(chosen.sub(Tensor.of(Shape.of(batch, 1), y)), 1f).mean().backward();
                rlOpt[p].step(q[p].parameters());
            }
            if (reservoir[p].size() >= batch) {
                Tensor[] b = reservoir[p].sample(batch);
                Var logits = average[p].forward(Var.of(b[0]), true);
                RlTensors.logSoftmax(logits).mul(b[1]).sum(1, true).neg().mean().backward();
                slOpt[p].step(average[p].parameters());
            }
        }
        if (++updates % (targetEvery / trainEvery + 1) == 0) {
            for (int p = 0; p < 2; p++) {
                RlTensors.copy(q[p].parameters(), target[p].parameters());
            }
        }
    }

    /** @return the policy table of the average-policy networks (the NFSP output) */
    public double[][] averagePolicy() {
        return table(average, false);
    }

    /** @return the greedy policy table of the best-response networks (DQN self-play) */
    public double[][] bestResponsePolicy() {
        return table(q, true);
    }

    private double[][] table(Model[] nets, boolean greedy) {
        int m = tree.infoSets();
        double[][] out = new double[m][];
        for (int p = 0; p < 2; p++) {
            float[] xs = new float[m * features];
            for (int s = 0; s < m; s++) {
                System.arraycopy(tree.features(s), 0, xs, s * features, features);
            }
            float[] y = nets[p].forward(Var.of(Tensor.of(Shape.of(m, features), xs)), false).value().toArray();
            for (int s = 0; s < m; s++) {
                if (tree.infoSetPlayer(s) != p) {
                    continue;
                }
                if (greedy) {
                    float[] row = java.util.Arrays.copyOfRange(y, s * actions, (s + 1) * actions);
                    out[s] = new double[actions];
                    out[s][greedy(row, tree.legal(s))] = 1;
                } else {
                    out[s] = maskedSoftmax(y, s * actions, tree.legal(s));
                }
            }
        }
        return out;
    }
}
