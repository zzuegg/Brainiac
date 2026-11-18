package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GumbelChannel;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Agents that learn to communicate with continuous messages, trained end to end: gradients
 * of the team objective flow through the messages into the senders. Each agent encodes its
 * observation into a hidden state; for several rounds every agent then receives a message and
 * updates its hidden state, and finally acts from it. All agents share one set of weights.
 * <ul>
 *   <li>{@link Channel#COMMNET}: the message is the mean of the other agents' hidden states
 *       (Sukhbaatar et al. 2016, <a href="https://arxiv.org/abs/1605.07736">arXiv:1605.07736</a>).</li>
 *   <li>{@link Channel#TARMAC}: targeted communication: each sender emits a key and a value,
 *       each receiver a query, and the message is the attention-weighted sum of the values,
 *       so the network learns whom to listen to (Das et al. 2019,
 *       <a href="https://arxiv.org/abs/1810.11187">arXiv:1810.11187</a>).</li>
 *   <li>{@link Channel#GUMBEL}: a discrete channel: each agent sends one symbol through a
 *       straight-through Gumbel-softmax ({@link GumbelChannel}), and receives the mean of the
 *       others' symbols, as in the referential game.</li>
 *   <li>{@link Channel#NONE}: no communication, the baseline.</li>
 * </ul>
 * Actions are trained with REINFORCE on the shared team reward against the batch-mean
 * baseline, plus an entropy bonus.
 */
public final class CommunicatingAgents {

    /** The communication channel. */
    public enum Channel {
        /** No messages. */
        NONE,
        /** Mean of the other agents' hidden states. */
        COMMNET,
        /** Attention over keys and values from every agent. */
        TARMAC,
        /** One discrete Gumbel-softmax symbol per agent. */
        GUMBEL
    }

    private final Channel channel;
    private final int agents;
    private final int obsSize;
    private final int actions;
    private final int hidden;
    private final int rounds;
    private final Var we;
    private final Var be;
    private final Var wh;
    private final Var wc;
    private final Var bh;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var ws;
    private final Var wo;
    private final List<Var> params = new ArrayList<>();
    private final GumbelChannel gumbel;
    private final Optimizer optimizer = Optimizer.adam(3e-3f);
    private final SplittableRandom random;
    private float[] lastAttention;

    private CommunicatingAgents(Channel channel, MultiAgentEnvironment env, int hidden, int rounds, long seed) {
        this.channel = channel;
        this.agents = env.agents();
        this.obsSize = env.observationSize();
        this.actions = env.actions();
        this.hidden = hidden;
        this.rounds = rounds;
        this.random = new SplittableRandom(seed);
        SplittableRandom init = new SplittableRandom(seed + 1);
        int key = Math.max(4, hidden / 4);
        int msg = channel == Channel.GUMBEL ? actions + 1 : hidden;
        we = weight(obsSize, hidden, init);
        be = Var.param(Tensor.zeros(Shape.of(1, hidden)));
        wh = weight(hidden, hidden, init);
        wc = weight(msg, hidden, init);
        bh = Var.param(Tensor.zeros(Shape.of(1, hidden)));
        wq = weight(hidden, key, init);
        wk = weight(hidden, key, init);
        wv = weight(hidden, hidden, init);
        ws = weight(hidden, actions + 1, init);
        wo = weight(hidden, actions, init);
        params.addAll(List.of(we, be, wh, wc, bh, wo));
        if (channel == Channel.TARMAC) {
            params.addAll(List.of(wq, wk, wv));
        } else if (channel == Channel.GUMBEL) {
            params.add(ws);
        }
        gumbel = GumbelChannel.of(1, actions + 1, 1f, seed + 2);
    }

    private static Var weight(int in, int out, SplittableRandom r) {
        return Var.param(Tensor.normal(Shape.of(in, out), 0, (float) Math.sqrt(1.0 / in), r));
    }

    /**
     * @param channel the channel
     * @param env the environment (for its sizes)
     * @param hidden hidden units
     * @param rounds communication rounds
     * @param seed the seed
     * @return the agents
     */
    public static CommunicatingAgents of(Channel channel, MultiAgentEnvironment env, int hidden, int rounds,
            long seed) {
        return new CommunicatingAgents(Objects.requireNonNull(channel), env, hidden, rounds, seed);
    }

    /** @return the channel */
    public Channel channel() {
        return channel;
    }

    /** Logits {@code [batch * agents, actions]} from observations {@code [batch * agents, obs]}. */
    private Var forward(Tensor obs, int batch, boolean training) {
        int n = agents;
        Var h = Var.of(obs).matmul(we).add(be).tanh();
        for (int c = 0; c < rounds; c++) {
            Var message = switch (channel) {
                case NONE -> null;
                case COMMNET -> {
                    Var h3 = h.reshape(batch, n, hidden);
                    Var others = h3.sum(1, true).expand(Shape.of(batch, n, hidden)).sub(h3).div(n - 1);
                    yield others.reshape(batch * n, hidden);
                }
                case TARMAC -> {
                    int key = wq.shape().dim(1);
                    Var q = h.matmul(wq).reshape(batch, n, key);
                    Var k = h.matmul(wk).reshape(batch, n, key);
                    Var v = h.matmul(wv).reshape(batch, n, hidden);
                    Var scores = q.matmul(k.permute(0, 2, 1)).div((float) Math.sqrt(key));
                    Var attention = scores.sub(scores.logsumexp(2, true)).exp();
                    if (c == rounds - 1 && !training) {
                        lastAttention = attention.value().toArray();
                    }
                    yield attention.matmul(v).reshape(batch * n, hidden);
                }
                case GUMBEL -> {
                    Var logits = h.matmul(ws);
                    Var symbols = training ? gumbel.sample(logits, true) : Var.of(gumbel.greedy(logits.value()));
                    Var s3 = symbols.reshape(batch, n, actions + 1);
                    Var others = s3.sum(1, true).expand(Shape.of(batch, n, actions + 1)).sub(s3).div(n - 1);
                    yield others.reshape(batch * n, actions + 1);
                }
            };
            Var pre = h.matmul(wh).add(bh);
            if (message != null) {
                pre = pre.add(message.matmul(wc));
            }
            h = pre.tanh();
        }
        return h.matmul(wo);
    }

    private Tensor observations(float[][][] obs) {
        int batch = obs.length;
        float[] x = new float[batch * agents * obsSize];
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < agents; i++) {
                System.arraycopy(obs[b][i], 0, x, (b * agents + i) * obsSize, obsSize);
            }
        }
        return Tensor.of(Shape.of(batch * agents, obsSize), x);
    }

    /**
     * Trains on batches of one-step episodes run in parallel environments.
     *
     * @param factory creates the environments (episodes must last one step)
     * @param iterations gradient steps
     * @param batch episodes per step
     * @return this
     */
    public CommunicatingAgents fit(Supplier<? extends MultiAgentEnvironment> factory, int iterations, int batch) {
        MultiAgentEnvironment[] envs = new MultiAgentEnvironment[batch];
        for (int b = 0; b < batch; b++) {
            envs[b] = factory.get();
        }
        for (int it = 0; it < iterations; it++) {
            float[][][] obs = new float[batch][][];
            for (int b = 0; b < batch; b++) {
                obs[b] = envs[b].reset(random.split());
            }
            Var logp = RlTensors.logSoftmax(forward(observations(obs), batch, true));
            float[] p = logp.value().exp().toArray();
            float[] mask = new float[batch * agents * actions];
            float[] reward = new float[batch];
            float mean = 0;
            for (int b = 0; b < batch; b++) {
                int[] a = new int[agents];
                for (int i = 0; i < agents; i++) {
                    int row = b * agents + i;
                    double u = random.nextDouble();
                    a[i] = actions - 1;
                    for (int j = 0; j < actions; j++) {
                        u -= p[row * actions + j];
                        if (u <= 0) {
                            a[i] = j;
                            break;
                        }
                    }
                    mask[row * actions + a[i]] = 1;
                }
                reward[b] = envs[b].step(a).reward();
                mean += reward[b] / batch;
            }
            float[] adv = new float[batch * agents];
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < agents; i++) {
                    adv[b * agents + i] = reward[b] - mean;
                }
            }
            Var chosen = logp.mul(Tensor.of(Shape.of(batch * agents, actions), mask)).sum(1, true);
            Var entropy = logp.exp().mul(logp).sum(1, true).neg();
            chosen.mul(Tensor.of(Shape.of(batch * agents, 1), adv)).add(entropy.mul(0.01f)).mean().neg().backward();
            optimizer.step(params);
        }
        return this;
    }

    /**
     * Greedy evaluation.
     *
     * @param env the environment
     * @param episodes episodes
     * @param seed the seed
     * @return {@code {mean team reward, fraction of episodes all agents were right}}
     */
    public double[] evaluate(MultiAgentEnvironment env, int episodes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double total = 0;
        double success = 0;
        for (int e = 0; e < episodes; e++) {
            float[][][] obs = {env.reset(r.split())};
            float[] logits = forward(observations(obs), 1, false).value().toArray();
            int[] a = new int[agents];
            for (int i = 0; i < agents; i++) {
                float[] row = java.util.Arrays.copyOfRange(logits, i * actions, (i + 1) * actions);
                a[i] = RlTensors.argmax(row);
            }
            float rew = env.step(a).reward();
            total += rew;
            success += rew >= 1 ? 1 : 0;
        }
        return new double[] {total / episodes, success / episodes};
    }

    /**
     * For {@link Channel#TARMAC} on a {@link HiddenGoalGame}: how the uninformed agents attend
     * in the last round.
     *
     * @param game the game
     * @param episodes episodes
     * @param seed the seed
     * @return {@code {mean attention weight on the informed agent, mean attention entropy in
     *     nats}} over the uninformed receivers
     */
    public double[] attentionOnInformed(HiddenGoalGame game, int episodes, long seed) {
        if (channel != Channel.TARMAC) {
            throw new IllegalStateException("only TarMAC has attention");
        }
        SplittableRandom r = new SplittableRandom(seed);
        double weight = 0;
        double entropy = 0;
        int count = 0;
        for (int e = 0; e < episodes; e++) {
            float[][][] obs = {game.reset(r.split())};
            forward(observations(obs), 1, false);
            int inf = game.informed();
            for (int i = 0; i < agents; i++) {
                if (i == inf) {
                    continue;
                }
                double h = 0;
                for (int j = 0; j < agents; j++) {
                    double a = lastAttention[i * agents + j];
                    h -= a > 0 ? a * Math.log(a) : 0;
                }
                weight += lastAttention[i * agents + inf];
                entropy += h;
                count++;
            }
        }
        return new double[] {weight / count, entropy / count};
    }
}
