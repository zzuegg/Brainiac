package io.github.zzuegg.brainiac.experimental.combopt;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GruCell;
import io.github.zzuegg.brainiac.experimental.nn.PointerNetwork;
import io.github.zzuegg.brainiac.experimental.nn.Seq2SeqAttention;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Pointer networks on the TSP, trained two ways.
 *
 * <ul>
 *   <li>{@link Supervised} wraps the repo's {@link PointerNetwork} unchanged and trains it,
 *       as Vinyals et al. did, by teacher-forced cross-entropy on optimal (or best-known)
 *       tours, each rotated to start at node 0 and oriented so the second node has the lower
 *       index; decoding is greedy without repeats.</li>
 *   <li>The outer class is Bello et al.'s actor-critic pointer network: the same building
 *       blocks (a bidirectional GRU {@link Seq2SeqAttention.Encoder}, a {@link GruCell}
 *       decoder and additive {@link Seq2SeqAttention.Scorer}s, here with one glimpse before
 *       the pointer and a {@code 10 tanh} logit clip), with sampling and the log-likelihood
 *       of the sampled tour, trained by REINFORCE against a learned critic that predicts the
 *       tour length from the mean encoder state. No solver is needed for training.</li>
 * </ul>
 *
 * <p>Papers: Vinyals, Fortunato and Jaitly, "Pointer Networks" (NeurIPS 2015),
 * <a href="https://arxiv.org/abs/1506.03134">arXiv:1506.03134</a>; Bello, Pham, Le, Norouzi and
 * Bengio, "Neural Combinatorial Optimization with Reinforcement Learning" (2017),
 * <a href="https://arxiv.org/abs/1611.09940">arXiv:1611.09940</a>.
 */
public final class PointerRouting {

    /**
     * Rollouts of the actor.
     *
     * @param tours the tours
     * @param lengths their lengths
     * @param logp the summed log-likelihoods {@code [batch, 1]}
     * @param value the critic's predicted lengths {@code [batch, 1]}
     */
    public record Rollout(int[][] tours, double[] lengths, Var logp, Var value) {
    }

    private final int units;
    private final Var projection;
    private final Seq2SeqAttention.Encoder encoder;
    private final GruCell decoder;
    private final Seq2SeqAttention.Scorer glimpse;
    private final Seq2SeqAttention.Scorer pointer;
    private final Var start;
    private final Var critic1;
    private final Var criticBias;
    private final Var critic2;

    private PointerRouting(int units, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.units = units;
        this.projection = AttentionModel.matrix(r, 2, units);
        this.encoder = new Seq2SeqAttention.Encoder(units, units, r);
        this.decoder = new GruCell(units, units, r);
        this.glimpse = new Seq2SeqAttention.Scorer(Seq2SeqAttention.Attention.BAHDANAU, units, units, units, r);
        this.pointer = new Seq2SeqAttention.Scorer(Seq2SeqAttention.Attention.BAHDANAU, units, units, units, r);
        this.start = Var.param(Tensor.normal(Shape.of(1, units), 0, 0.3f, r));
        this.critic1 = AttentionModel.matrix(r, units, units);
        this.criticBias = Var.param(Tensor.zeros(Shape.of(units)));
        this.critic2 = AttentionModel.matrix(r, units, 1);
    }

    /**
     * Creates the actor-critic pointer network.
     *
     * @param units the hidden width
     * @param seed the seed
     * @return the model
     */
    public static PointerRouting of(int units, long seed) {
        return new PointerRouting(units, seed);
    }

    static Var coordinates(List<RoutingEnv> envs) {
        int b = envs.size();
        int n = envs.get(0).nodes();
        float[] x = new float[b * n * 2];
        for (int i = 0; i < b; i++) {
            double[][] c = envs.get(i).coords();
            for (int j = 0; j < n; j++) {
                x[(i * n + j) * 2] = (float) (c[j][0] - 0.5);
                x[(i * n + j) * 2 + 1] = (float) (c[j][1] - 0.5);
            }
        }
        return Var.of(Tensor.of(Shape.of(b, n, 2), x));
    }

    /**
     * Rolls out the actor.
     *
     * @param envs TSP instances of one size
     * @param sample sample (else greedy)
     * @param random the source
     * @return the rollouts with the critic's predictions
     */
    public Rollout rollout(List<RoutingEnv> envs, boolean sample, SplittableRandom random) {
        int b = envs.size();
        int n = envs.get(0).nodes();
        Var x = coordinates(envs).matmul(projection);
        Var[] enc = encoder.encode(x);
        Var keys = enc[0];
        Var value = keys.mean(1, false).matmul(critic1).add(criticBias).relu().matmul(critic2);
        Var flat = x.reshape(b * n, units);
        Var h = enc[1].tanh();
        Var in = start.expand(Shape.of(b, units));
        boolean[][] used = new boolean[b][n];
        int[][] tours = new int[b][n];
        Var logp = null;
        for (int t = 0; t < n; t++) {
            h = decoder.step(in, h);
            float[] mask = new float[b * n];
            for (int i = 0; i < b; i++) {
                for (int j = 0; j < n; j++) {
                    mask[i * n + j] = used[i][j] ? -1e9f : 0;
                }
            }
            Tensor m = Tensor.of(Shape.of(b, n), mask);
            Var g = Seq2SeqAttention.Scorer.context(glimpse.scores(h, keys).add(m).softmax(1), keys);
            Var lp = pointer.scores(g, keys).tanh().mul(10f).add(m).logSoftmax(1);
            float[] p = lp.value().toArray();
            float[] idx = new float[b];
            float[] gather = new float[b];
            for (int i = 0; i < b; i++) {
                int pick = -1;
                if (sample) {
                    double u = random.nextDouble();
                    for (int j = 0; j < n; j++) {
                        if (!used[i][j]) {
                            pick = j;
                            u -= Math.exp(p[i * n + j]);
                            if (u <= 0) {
                                break;
                            }
                        }
                    }
                } else {
                    for (int j = 0; j < n; j++) {
                        if (!used[i][j] && (pick < 0 || p[i * n + j] > p[i * n + pick])) {
                            pick = j;
                        }
                    }
                }
                used[i][pick] = true;
                tours[i][t] = pick;
                idx[i] = pick;
                gather[i] = i * n + pick;
            }
            Var step = lp.takeAlong(Tensor.of(Shape.of(b, 1), idx));
            logp = logp == null ? step : logp.add(step);
            in = flat.gather(0, Tensor.of(Shape.of(b), gather));
        }
        double[] len = new double[b];
        for (int i = 0; i < b; i++) {
            len[i] = envs.get(i).length(tours[i]);
        }
        return new Rollout(tours, len, logp, value);
    }

    /**
     * One actor-critic step: REINFORCE with the critic's prediction as baseline, and a mean
     * squared error for the critic.
     *
     * @param batch instances
     * @param optimizer the optimizer of actor and critic (the critic shares the encoder)
     * @param random the source
     * @return the batch's mean sampled length
     */
    public double actorCriticStep(List<RoutingEnv> batch, Optimizer optimizer, SplittableRandom random) {
        Rollout ro = rollout(batch, true, random);
        int b = batch.size();
        float[] len = new float[b];
        float[] adv = new float[b];
        float[] v = ro.value().value().toArray();
        double mean = 0;
        for (int i = 0; i < b; i++) {
            len[i] = (float) ro.lengths()[i];
            adv[i] = len[i] - v[i];
            mean += len[i];
        }
        Var actorLoss = ro.logp().mul(Tensor.of(Shape.of(b, 1), adv)).mean();
        Var criticLoss = ro.value().sub(Tensor.of(Shape.of(b, 1), len)).square().mean();
        AttentionModel.step(actorLoss.add(criticLoss), parameters(), optimizer, 1.0);
        return mean / b;
    }

    /**
     * Greedy tour lengths.
     *
     * @param envs instances
     * @return lengths
     */
    public double[] greedyLengths(List<RoutingEnv> envs) {
        return Var.noGrad(() -> rollout(envs, false, null)).lengths();
    }

    /** @return the actor's (and shared encoder's) trainable values */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(projection, start));
        p.addAll(encoder.parameters());
        p.addAll(decoder.parameters());
        p.addAll(glimpse.parameters());
        p.addAll(pointer.parameters());
        p.addAll(criticParameters());
        return p;
    }

    private List<Var> criticParameters() {
        return List.of(critic1, criticBias, critic2);
    }

    /**
     * Vinyals et al.'s supervised pointer network: the repo's {@link PointerNetwork} trained
     * on reference tours.
     */
    public static final class Supervised {
        private final PointerNetwork net;

        /**
         * Creates the model.
         *
         * @param units the hidden width
         * @param seed the seed
         */
        public Supervised(int units, long seed) {
            this.net = PointerNetwork.of(2, units, false, seed);
        }

        /**
         * A tour in canonical form: starting at node 0, second node below the last.
         *
         * @param tour a tour
         * @return the canonical tour
         */
        public static int[] canonical(int[] tour) {
            int n = tour.length;
            int z = 0;
            while (tour[z] != 0) {
                z++;
            }
            int[] out = new int[n];
            boolean reverse = tour[(z + 1) % n] > tour[(z - 1 + n) % n];
            for (int k = 0; k < n; k++) {
                out[k] = reverse ? tour[(z - k + n) % n] : tour[(z + k) % n];
            }
            return out;
        }

        /**
         * One teacher-forced step.
         *
         * @param envs instances
         * @param tours their reference tours
         * @param optimizer the optimizer
         * @return the loss
         */
        public float step(List<RoutingEnv> envs, int[][] tours, Optimizer optimizer) {
            int[][] targets = new int[tours.length][];
            for (int i = 0; i < tours.length; i++) {
                targets[i] = canonical(tours[i]);
            }
            Var loss = net.loss(coordinates(envs), targets);
            float v = loss.value().item();
            AttentionModel.step(loss, net.parameters(), optimizer, 1.0);
            return v;
        }

        /**
         * Greedy tours without repeats.
         *
         * @param envs instances
         * @return lengths
         */
        public double[] greedyLengths(List<RoutingEnv> envs) {
            int n = envs.get(0).nodes();
            int[][] tours = Var.noGrad(() -> net.decode(coordinates(envs), n, true));
            double[] len = new double[envs.size()];
            for (int i = 0; i < len.length; i++) {
                len[i] = envs.get(i).length(tours[i]);
            }
            return len;
        }

        /** @return the wrapped network */
        public PointerNetwork network() {
            return net;
        }
    }
}
