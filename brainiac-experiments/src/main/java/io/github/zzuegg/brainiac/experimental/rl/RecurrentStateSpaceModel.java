package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The recurrent state-space model of PlaNet (Hafner et al., 2019,
 * <a href="https://arxiv.org/abs/1811.04551">arXiv:1811.04551</a>): a latent state with a
 * deterministic path {@code h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])} that remembers, and a
 * stochastic path {@code z_t} that models uncertainty, with a prior {@code p(z_t | h_t)} for
 * imagination and a posterior {@code q(z_t | h_t, o_t)} for filtering real observations.
 * Observations and rewards are decoded from {@code (h_t, z_t)}.
 *
 * <p>Trained on sequences by the ELBO: reconstruction and reward errors plus
 * {@code KL(q || p)} clipped below at a number of free nats, and latent overshooting: the
 * prior rolled open loop for up to {@code D} steps must also match the (fixed) later
 * posteriors, which is what multi-step planning needs. {@link LatentPlanner} plans in this
 * latent space. Observations are standardized internally with the training data's
 * statistics.
 */
public final class RecurrentStateSpaceModel {

    /** One recorded episode: {@code T + 1} observations, {@code T} actions and rewards. */
    public record Sequence(float[][] observations, float[][] actions, float[] rewards) {
    }

    /** A batch of latent states (no gradients). @param h deterministic {@code [n, deter]} @param z stochastic {@code [n, stoch]} */
    public record Belief(Tensor h, Tensor z) {
    }

    private final int obsDims;
    private final int actDims;
    private final int stoch;
    private final RlGruCell cell;
    private final Model prior;
    private final Model posterior;
    private final Model decoder;
    private final Model rewardHead;
    private final List<Var> parameters = new ArrayList<>();
    private final Optimizer optimizer = Optimizer.adam(3e-3f);
    private final float[] mean;
    private final float[] std;
    private float freeNats = 1f;
    private float klDynamics;
    private float klRepresentation;
    private int overshooting = 3;
    private float overshootWeight = 0.5f;
    private SplittableRandom random;
    private double lastLoss = Double.NaN;

    private RecurrentStateSpaceModel(int obsDims, int actDims, int deter, int stoch, int hidden, long seed) {
        this.obsDims = obsDims;
        this.actDims = actDims;
        this.stoch = stoch;
        this.random = new SplittableRandom(seed);
        this.cell = new RlGruCell(stoch + actDims, deter, seed);
        this.prior = Nets.mlp(deter, 2 * stoch, Activation.RELU, seed + 1, hidden);
        this.posterior = Nets.mlp(deter + obsDims, 2 * stoch, Activation.RELU, seed + 2, hidden);
        this.decoder = Nets.mlp(deter + stoch, obsDims, Activation.RELU, seed + 3, hidden);
        this.rewardHead = Nets.mlp(deter + stoch, 1, Activation.RELU, seed + 4, hidden);
        parameters.addAll(cell.parameters());
        parameters.addAll(prior.parameters());
        parameters.addAll(posterior.parameters());
        parameters.addAll(decoder.parameters());
        parameters.addAll(rewardHead.parameters());
        this.mean = new float[obsDims];
        this.std = new float[obsDims];
        java.util.Arrays.fill(std, 1f);
    }

    /**
     * @param obsDims values per observation
     * @param actDims values per action
     * @param deter deterministic state size
     * @param stoch stochastic state size
     * @param hidden hidden width of the prior, posterior and decoders
     * @param seed the seed
     * @return the model
     */
    public static RecurrentStateSpaceModel of(int obsDims, int actDims, int deter, int stoch, int hidden, long seed) {
        return new RecurrentStateSpaceModel(obsDims, actDims, deter, stoch, hidden, seed);
    }

    /**
     * @param nats the KL below which the KL term gives no gradient
     * @return this model
     */
    public RecurrentStateSpaceModel freeNats(float nats) {
        this.freeNats = nats;
        return this;
    }

    /**
     * DreamerV3's KL balancing (Hafner et al., <a href="https://arxiv.org/abs/2301.04104">2301.04104</a>):
     * the KL term is split into a dynamics loss {@code KL(sg(q) || p)} that trains the prior
     * towards the posterior and a representation loss {@code KL(q || sg(p))} that keeps the
     * posterior predictable, each clipped at the free nats and weighted separately (the paper
     * uses 0.5 and 0.1 with 1 free nat). Off by default (the plain {@code KL(q || p)}).
     *
     * @param dynamics the dynamics-loss weight, 0 to switch balancing off
     * @param representation the representation-loss weight
     * @return this model
     */
    public RecurrentStateSpaceModel klBalance(float dynamics, float representation) {
        this.klDynamics = dynamics;
        this.klRepresentation = representation;
        return this;
    }

    /**
     * @param distance the longest open-loop prior rollout matched to the posterior (1 for none)
     * @param weight the weight of the overshooting KL terms
     * @return this model
     */
    public RecurrentStateSpaceModel overshooting(int distance, float weight) {
        this.overshooting = Math.max(1, distance);
        this.overshootWeight = weight;
        return this;
    }

    /** @return the action size */
    public int actionDims() {
        return actDims;
    }

    /** @return the loss of the last batch */
    public double lastLoss() {
        return lastLoss;
    }

    /**
     * Records episodes with uniformly random actions.
     *
     * @param env the environment
     * @param episodes episodes to record
     * @param seed the seed
     * @return the episodes
     */
    public static List<Sequence> collectRandom(Environment env, int episodes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Sequence> out = new ArrayList<>();
        for (int e = 0; e < episodes; e++) {
            List<float[]> obs = new ArrayList<>();
            List<float[]> acts = new ArrayList<>();
            List<Float> rews = new ArrayList<>();
            obs.add(env.reset(r.split()));
            Environment.Step s;
            do {
                float[] a = env.actionSpace().sample(r);
                s = env.step(a);
                acts.add(a);
                rews.add(s.reward());
                obs.add(s.observation());
            } while (!s.done());
            float[] rw = new float[rews.size()];
            for (int i = 0; i < rw.length; i++) {
                rw[i] = rews.get(i);
            }
            out.add(new Sequence(obs.toArray(new float[0][]), acts.toArray(new float[0][]), rw));
        }
        return out;
    }

    private float[] normalize(float[] o) {
        float[] n = new float[o.length];
        for (int i = 0; i < o.length; i++) {
            n[i] = (o[i] - mean[i]) / std[i];
        }
        return n;
    }

    private float[] denormalize(float[] n) {
        float[] o = new float[n.length];
        for (int i = 0; i < n.length; i++) {
            o[i] = n[i] * std[i] + mean[i];
        }
        return o;
    }

    private void fitStatistics(List<Sequence> data) {
        double[] s = new double[obsDims];
        double[] sq = new double[obsDims];
        long n = 0;
        for (Sequence q : data) {
            for (float[] o : q.observations()) {
                for (int i = 0; i < obsDims; i++) {
                    s[i] += o[i];
                    sq[i] += o[i] * o[i];
                }
                n++;
            }
        }
        for (int i = 0; i < obsDims; i++) {
            mean[i] = (float) (s[i] / n);
            std[i] = (float) Math.sqrt(Math.max(1e-4, sq[i] / n - mean[i] * mean[i]));
        }
    }

    /** Splits a {@code [n, 2 stoch]} output into mean and positive std. */
    private Var[] gaussian(Var out) {
        Var mu = out.slice(1, 0, stoch);
        Var sd = out.slice(1, stoch, 2 * stoch).softplus().add(0.1f);
        return new Var[] {mu, sd};
    }

    private static Var kl(Var[] q, Var[] p) {
        // KL(N(mq, sq) || N(mp, sp)) per row, summed over dimensions
        Var ratio = q[1].div(p[1]).square();
        Var diff = q[0].sub(p[0]).div(p[1]).square();
        return ratio.add(diff).sub(1f).sub(ratio.log()).mul(0.5f).sum(1, true);
    }

    private static Var[] detached(Var[] g) {
        return new Var[] {g[0].detach(), g[1].detach()};
    }

    private Var sample(Var[] g, int rows) {
        return g[0].add(g[1].mul(Tensor.normal(Shape.of(rows, stoch), 0, 1, random)));
    }

    /**
     * Trains on random windows of the episodes.
     *
     * @param data the episodes
     * @param steps gradient steps
     * @param batch windows per step
     * @param length steps per window
     * @return this model
     */
    public RecurrentStateSpaceModel train(List<Sequence> data, int steps, int batch, int length) {
        fitStatistics(data);
        for (int it = 0; it < steps; it++) {
            float[][] o = new float[length + 1][batch * obsDims];
            float[][] a = new float[length][batch * actDims];
            float[][] r = new float[length][batch];
            for (int b = 0; b < batch; b++) {
                Sequence q = data.get(random.nextInt(data.size()));
                int start = random.nextInt(q.actions().length - length + 1);
                for (int t = 0; t <= length; t++) {
                    System.arraycopy(normalize(q.observations()[start + t]), 0, o[t], b * obsDims, obsDims);
                    if (t < length) {
                        System.arraycopy(q.actions()[start + t], 0, a[t], b * actDims, actDims);
                        r[t][b] = q.rewards()[start + t];
                    }
                }
            }
            lastLoss = step(o, a, r, batch, length);
        }
        return this;
    }

    private double step(float[][] o, float[][] a, float[][] r, int batch, int length) {
        Var h = cell.zeroState(batch);
        Tensor o0 = Tensor.of(Shape.of(batch, obsDims), o[0]);
        Var[] q0 = gaussian(posterior.forward(Var.of(Tensor.concat(1, h.value(), o0)), true));
        Var z = sample(q0, batch);
        Var loss = decoder.forward(Var.concat(1, h, z), true).sub(o0).square().sum(1, true).mean();
        List<Var> hs = new ArrayList<>();
        List<Var> zs = new ArrayList<>();
        List<Var[]> posts = new ArrayList<>();
        hs.add(h);
        zs.add(z);
        posts.add(q0);
        for (int t = 1; t <= length; t++) {
            Tensor act = Tensor.of(Shape.of(batch, actDims), a[t - 1]);
            h = cell.step(Var.concat(1, z, Var.of(act)), h);
            Tensor ot = Tensor.of(Shape.of(batch, obsDims), o[t]);
            Var[] p = gaussian(prior.forward(h, true));
            Var[] q = gaussian(posterior.forward(Var.concat(1, h, Var.of(ot)), true));
            z = sample(q, batch);
            Var feat = Var.concat(1, h, z);
            Var rec = decoder.forward(feat, true).sub(ot).square().sum(1, true).mean();
            Var rew = rewardHead.forward(feat, true).sub(Tensor.of(Shape.of(batch, 1), r[t - 1])).square().mean();
            Var klTerm = klDynamics > 0
                    ? kl(detached(q), p).mean().maximum(freeNats).mul(klDynamics)
                            .add(kl(q, detached(p)).mean().maximum(freeNats).mul(klRepresentation))
                    : kl(q, p).mean().maximum(freeNats);
            loss = loss.add(rec).add(rew).add(klTerm);
            hs.add(h);
            zs.add(z);
            posts.add(q);
        }
        // latent overshooting: open-loop priors from step t must match the fixed posteriors at t + d
        if (overshooting > 1) {
            for (int t = 0; t < length - 1; t++) {
                Var oh = hs.get(t);
                Var oz = zs.get(t);
                for (int d = 1; d <= overshooting && t + d <= length; d++) {
                    Tensor act = Tensor.of(Shape.of(batch, actDims), a[t + d - 1]);
                    oh = cell.step(Var.concat(1, oz, Var.of(act)), oh);
                    Var[] p = gaussian(prior.forward(oh, true));
                    oz = sample(p, batch);
                    if (d >= 2) {
                        Var[] target = posts.get(t + d);
                        Var[] fixed = {target[0].detach(), target[1].detach()};
                        loss = loss.add(kl(fixed, p).mean().maximum(freeNats).mul(overshootWeight / (overshooting - 1)));
                    }
                }
            }
        }
        loss = loss.mul(1f / length);
        RlTensors.zeroGrad(parameters);
        loss.backward();
        RlTensors.clip(parameters, 100f);
        optimizer.step(parameters);
        return loss.value().item();
    }

    /**
     * Filters one real observation into the belief.
     *
     * @param previous the belief before, or {@code null} at the start of an episode
     * @param action the action taken since (ignored at the start)
     * @param observation the new observation
     * @return the belief, using the posterior mean
     */
    public Belief observe(Belief previous, float[] action, float[] observation) {
        Var h;
        if (previous == null) {
            h = cell.zeroState(1);
        } else {
            h = cell.step(Var.of(Tensor.concat(1, previous.z(), Tensor.of(Shape.of(1, actDims), action))),
                    Var.of(previous.h()));
        }
        Tensor o = Tensor.of(Shape.of(1, obsDims), normalize(observation));
        Var[] q = gaussian(posterior.forward(Var.of(Tensor.concat(1, h.value(), o)), false));
        return new Belief(h.value(), q[0].value());
    }

    /**
     * One imagined step with the prior mean.
     *
     * @param belief the beliefs, {@code n} rows
     * @param actions {@code [n, act]}
     * @return the next beliefs
     */
    public Belief imagine(Belief belief, Tensor actions) {
        Var h = cell.step(Var.of(Tensor.concat(1, belief.z(), actions)), Var.of(belief.h()));
        Var[] p = gaussian(prior.forward(h, false));
        return new Belief(h.value(), p[0].value());
    }

    /**
     * @param belief the beliefs
     * @return the predicted rewards of reaching them, {@code [n, 1]}
     */
    public Tensor reward(Belief belief) {
        return rewardHead.forward(Var.of(Tensor.concat(1, belief.h(), belief.z())), false).value();
    }

    /**
     * @param belief one belief
     * @return the decoded observation
     */
    public float[] decode(Belief belief) {
        return denormalize(decoder.forward(Var.of(Tensor.concat(1, belief.h(), belief.z())), false).value()
                .to(DType.FLOAT32).toArray());
    }

    /**
     * Repeats a single belief.
     *
     * @param belief one belief
     * @param n copies
     * @return {@code n} rows
     */
    public static Belief repeat(Belief belief, int n) {
        return new Belief(repeatRow(belief.h(), n), repeatRow(belief.z(), n));
    }

    private static Tensor repeatRow(Tensor row, int n) {
        float[] v = row.to(DType.FLOAT32).toArray();
        float[] out = new float[n * v.length];
        for (int i = 0; i < n; i++) {
            System.arraycopy(v, 0, out, i * v.length, v.length);
        }
        return Tensor.of(Shape.of(n, v.length), out);
    }
}
