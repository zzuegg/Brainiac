package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Bandit;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Meta-reinforcement learning on a distribution of multi-armed {@link Bandit} tasks: the
 * agent is trained across many tasks so that, on a new one, it explores and exploits within
 * a short horizon with no weight updates.
 * <ul>
 *   <li><b>RL²</b> (Duan et al., 2016, <a href="https://arxiv.org/abs/1611.02779">arXiv:1611.02779</a>):
 *       a recurrent policy whose input is the previous action, the previous reward and the
 *       time; its hidden state carries over the whole trial and learns to be the bandit
 *       algorithm. Trained end to end by advantage actor-critic over whole trials.</li>
 *   <li><b>VariBAD</b> (Zintgraf et al., 2020, <a href="https://arxiv.org/abs/1910.08348">arXiv:1910.08348</a>):
 *       the recurrent network is a variational encoder whose Gaussian posterior over a task
 *       latent is trained by an ELBO to decode the rewards of every arm pulled in the whole
 *       trial (past and future); the policy is a feed-forward network on the time and the
 *       belief (posterior mean and standard deviation), trained by actor-critic without
 *       gradients into the encoder.</li>
 * </ul>
 * As an {@link Agent}, it acts on the encoded previous step ({@link #input}); call
 * {@link #newTask()} before the first pull of a task.
 */
public final class MetaRlAgent implements Agent {

    /** The meta-RL method. */
    public enum Kind {
        /** Recurrent policy trained end to end. */
        RL2,
        /** Variational task belief plus a belief-conditioned policy. */
        VARIBAD
    }

    private final Kind kind;
    private final int arms;
    private final int horizon;
    private final int latent;
    private final RlGruCell cell;
    private final List<Var> encoderParams = new ArrayList<>();
    private final List<Var> policyParams = new ArrayList<>();
    private final Var muW;
    private final Var muB;
    private final Var lvW;
    private final Var lvB;
    private final Var decW1;
    private final Var decB1;
    private final Var decW2;
    private final Var decB2;
    private final Var polW1;
    private final Var polB1;
    private final Var polW2;
    private final Var polB2;
    private final Var valW;
    private final Var valB;
    private Optimizer optimizer;
    private float gamma = 0.9f;
    private float lambda = 0.95f;
    private float entropy = 0.05f;
    private float klWeight = 0.1f;
    private int batch = 32;
    private SplittableRandom random;
    private Var state;

    private MetaRlAgent(Kind kind, int arms, int horizon, int units, int latent, long seed) {
        this.kind = kind;
        this.arms = arms;
        this.horizon = horizon;
        this.latent = latent;
        this.random = new SplittableRandom(seed);
        SplittableRandom init = new SplittableRandom(seed + 1);
        this.cell = new RlGruCell(arms + 2, units, seed);
        int policyIn = kind == Kind.RL2 ? units : 1 + 2 * latent;
        this.muW = weight(units, latent, init);
        this.muB = Var.param(Tensor.zeros(Shape.of(1, latent)));
        this.lvW = weight(units, latent, init);
        this.lvB = Var.param(Tensor.zeros(Shape.of(1, latent)));
        this.decW1 = weight(latent, 32, init);
        this.decB1 = Var.param(Tensor.zeros(Shape.of(1, 32)));
        this.decW2 = weight(32, arms, init);
        this.decB2 = Var.param(Tensor.zeros(Shape.of(1, arms)));
        this.polW1 = weight(policyIn, units, init);
        this.polB1 = Var.param(Tensor.zeros(Shape.of(1, units)));
        this.polW2 = weight(units, arms, init);
        this.polB2 = Var.param(Tensor.zeros(Shape.of(1, arms)));
        this.valW = weight(units, 1, init);
        this.valB = Var.param(Tensor.zeros(Shape.of(1, 1)));
        if (kind == Kind.RL2) {
            policyParams.addAll(cell.parameters());
        } else {
            encoderParams.addAll(cell.parameters());
            encoderParams.addAll(List.of(muW, muB, lvW, lvB, decW1, decB1, decW2, decB2));
        }
        policyParams.addAll(List.of(polW1, polB1, polW2, polB2, valW, valB));
        this.optimizer = Optimizer.adam(5e-3f);
    }

    private static Var weight(int in, int out, SplittableRandom r) {
        double limit = Math.sqrt(6.0 / (in + out));
        float[] w = new float[in * out];
        for (int i = 0; i < w.length; i++) {
            w[i] = (float) r.nextDouble(-limit, limit);
        }
        return Var.param(Tensor.of(Shape.of(in, out), w));
    }

    /**
     * @param arms arms per bandit
     * @param horizon pulls per task
     * @param units recurrent state size
     * @param seed the seed
     * @return an RL² agent
     */
    public static MetaRlAgent rl2(int arms, int horizon, int units, long seed) {
        return new MetaRlAgent(Kind.RL2, arms, horizon, units, 1, seed);
    }

    /**
     * @param arms arms per bandit
     * @param horizon pulls per task
     * @param units encoder state size
     * @param latent task latent size
     * @param seed the seed
     * @return a VariBAD agent
     */
    public static MetaRlAgent variBad(int arms, int horizon, int units, int latent, long seed) {
        return new MetaRlAgent(Kind.VARIBAD, arms, horizon, units, latent, seed);
    }

    /** @param tasks tasks per meta-training batch @return this agent */
    public MetaRlAgent batchSize(int tasks) {
        this.batch = tasks;
        return this;
    }

    /** @param discount the discount within a trial @param gaeLambda the GAE lambda @return this agent */
    public MetaRlAgent discount(float discount, float gaeLambda) {
        this.gamma = discount;
        this.lambda = gaeLambda;
        return this;
    }

    /** @param rate the Adam learning rate @return this agent */
    public MetaRlAgent learningRate(float rate) {
        this.optimizer = Optimizer.adam(rate);
        return this;
    }

    /** @param coefficient the entropy bonus @return this agent */
    public MetaRlAgent entropyBonus(float coefficient) {
        this.entropy = coefficient;
        return this;
    }

    /**
     * Encodes one step for {@link #act}: one-hot previous arm (none at the start), previous
     * reward and the time as a fraction of the horizon.
     *
     * @param previousArm the previous arm, or -1 at the start of a task
     * @param previousReward the previous reward
     * @param t pulls made so far in this task
     * @return the input row
     */
    public float[] input(int previousArm, float previousReward, int t) {
        float[] x = new float[arms + 2];
        if (previousArm >= 0) {
            x[previousArm] = 1;
        }
        x[arms] = previousReward;
        x[arms + 1] = (float) t / horizon;
        return x;
    }

    /** Forgets the current task. */
    public void newTask() {
        state = cell.zeroState(1);
    }

    private Var policyInput(Var h, Tensor time) {
        if (kind == Kind.RL2) {
            return h;
        }
        Var mu = h.matmul(muW).add(muB);
        Var sd = h.matmul(lvW).add(lvB).mul(0.5f).exp();
        return Var.concat(1, Var.of(time), mu.detach(), sd.detach());
    }

    private Var hidden(Var in) {
        return in.matmul(polW1).add(polB1).tanh();
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        if (state == null) {
            newTask();
        }
        Var x = Var.of(Tensor.of(Shape.of(1, arms + 2), observation));
        state = cell.step(x, state).detach();
        Tensor time = Tensor.of(Shape.of(1, 1), new float[] {observation[arms + 1]});
        float[] logits = hidden(policyInput(state, time)).matmul(polW2).add(polB2).value()
                .to(DType.FLOAT32).toArray();
        return new float[] {explore ? sample(logits, random) : RlTensors.argmax(logits)};
    }

    private static int sample(float[] logits, SplittableRandom r) {
        double max = Double.NEGATIVE_INFINITY;
        for (float l : logits) {
            max = Math.max(max, l);
        }
        double[] p = new double[logits.length];
        double z = 0;
        for (int i = 0; i < p.length; i++) {
            p[i] = Math.exp(logits[i] - max);
            z += p[i];
        }
        double u = r.nextDouble() * z;
        for (int i = 0; i < p.length; i++) {
            u -= p[i];
            if (u <= 0) {
                return i;
            }
        }
        return p.length - 1;
    }

    /**
     * Meta-trains on tasks drawn from a distribution, one batch of whole trials per
     * iteration.
     *
     * @param tasks draws a task from a seed source
     * @param iterations the number of batches
     * @return the mean reward per pull of the last batch
     */
    public double metaTrain(Function<SplittableRandom, Bandit> tasks, int iterations) {
        double last = 0;
        for (int it = 0; it < iterations; it++) {
            last = trainBatch(tasks);
        }
        return last;
    }

    private double trainBatch(Function<SplittableRandom, Bandit> tasks) {
        List<Bandit> envs = new ArrayList<>();
        for (int b = 0; b < batch; b++) {
            envs.add(tasks.apply(random));
        }
        Var h = cell.zeroState(batch);
        float[] x = new float[batch * (arms + 2)];
        float[][] rewards = new float[horizon][batch];
        List<Var> logps = new ArrayList<>();
        List<Var> ents = new ArrayList<>();
        List<Var> values = new ArrayList<>();
        List<Var> states = new ArrayList<>();
        float[] successes = new float[batch * arms];
        float[] counts = new float[batch * arms];
        double total = 0;
        for (int t = 0; t <= horizon; t++) {
            h = cell.step(Var.of(Tensor.of(Shape.of(batch, arms + 2), x)), h);
            states.add(h);
            if (t == horizon) {
                break;
            }
            float[] tf = new float[batch];
            java.util.Arrays.fill(tf, (float) t / horizon);
            Var hid = hidden(policyInput(h, Tensor.of(Shape.of(batch, 1), tf)));
            Var logits = hid.matmul(polW2).add(polB2);
            Var logp = RlTensors.logSoftmax(logits);
            values.add(hid.matmul(valW).add(valB));
            float[] lv = logits.value().to(DType.FLOAT32).toArray();
            float[] chosen = new float[batch];
            x = new float[batch * (arms + 2)];
            for (int b = 0; b < batch; b++) {
                int a = sample(java.util.Arrays.copyOfRange(lv, b * arms, (b + 1) * arms), random);
                chosen[b] = a;
                Environment env = envs.get(b);
                env.reset(random.split());
                float r = env.step(new float[] {a}).reward();
                rewards[t][b] = r;
                total += r;
                counts[b * arms + a]++;
                successes[b * arms + a] += r;
                x[b * (arms + 2) + a] = 1;
                x[b * (arms + 2) + arms] = r;
                x[b * (arms + 2) + arms + 1] = (float) (t + 1) / horizon;
            }
            Tensor onehot = RlTensors.oneHot(chosen, arms, logits.value().device());
            logps.add(logp.mul(onehot).sum(1, true));
            ents.add(logp.exp().mul(logp).sum(1, true).neg());
        }
        // advantage actor-critic over the whole trial
        // (GAE advantages, normalized over the batch; Monte Carlo value targets)
        float[][] v = new float[horizon + 1][];
        for (int t = 0; t < horizon; t++) {
            v[t] = values.get(t).value().to(DType.FLOAT32).toArray();
        }
        v[horizon] = new float[batch];
        float[][] adv = new float[horizon][batch];
        float[][] ret = new float[horizon][batch];
        float[] gae = new float[batch];
        float[] g = new float[batch];
        double sum = 0;
        double sq = 0;
        for (int t = horizon - 1; t >= 0; t--) {
            for (int b = 0; b < batch; b++) {
                g[b] = rewards[t][b] + gamma * g[b];
                ret[t][b] = g[b];
                float delta = rewards[t][b] + gamma * v[t + 1][b] - v[t][b];
                gae[b] = delta + gamma * lambda * gae[b];
                adv[t][b] = gae[b];
                sum += gae[b];
                sq += gae[b] * gae[b];
            }
        }
        int n = horizon * batch;
        double mean = sum / n;
        double sd = Math.sqrt(Math.max(1e-8, sq / n - mean * mean));
        Var loss = null;
        for (int t = 0; t < horizon; t++) {
            float[] a = new float[batch];
            for (int b = 0; b < batch; b++) {
                a[b] = (float) ((adv[t][b] - mean) / sd);
            }
            Tensor at = Tensor.of(Shape.of(batch, 1), a);
            Tensor gt = Tensor.of(Shape.of(batch, 1), ret[t]);
            Var term = logps.get(t).mul(at).neg().add(values.get(t).sub(gt).square().mul(0.5f))
                    .sub(ents.get(t).mul(entropy)).mean();
            loss = loss == null ? term : loss.add(term);
        }
        if (kind == Kind.VARIBAD) {
            loss = loss.add(elbo(states, Tensor.of(Shape.of(batch, arms), counts),
                    Tensor.of(Shape.of(batch, arms), successes)));
        }
        List<Var> all = new ArrayList<>(policyParams);
        all.addAll(encoderParams);
        RlTensors.zeroGrad(all);
        loss.mul(1f / horizon).backward();
        RlTensors.clip(all, 5f);
        optimizer.step(all);
        return total / (batch * horizon);
    }

    /**
     * Negative ELBO summed over the trial: for every belief, the squared error of the
     * decoded arm means on every pull of the whole trial, plus the KL to the prior.
     */
    private Var elbo(List<Var> states, Tensor counts, Tensor sums) {
        Var loss = null;
        for (Var h : states) {
            Var mu = h.matmul(muW).add(muB);
            Var logVar = h.matmul(lvW).add(lvB);
            Tensor eps = Tensor.normal(Shape.of(batch, latent), 0, 1, random);
            Var z = mu.add(logVar.mul(0.5f).exp().mul(eps));
            Var pred = z.matmul(decW1).add(decB1).tanh().matmul(decW2).add(decB2);
            // sum_j (pred[a_j] - r_j)^2 up to a constant = sum_k n_k pred_k^2 - 2 pred_k s_k
            Var rec = pred.square().mul(counts).sub(pred.mul(sums).mul(2f)).sum(1, true).mean();
            Var kl = mu.square().add(logVar.exp()).sub(logVar).sub(1f).mul(0.5f).sum(1, true).mean();
            Var term = rec.add(kl.mul(klWeight));
            loss = loss == null ? term : loss.add(term);
        }
        return loss;
    }

    /**
     * Plays fresh tasks greedily-or-sampled, without learning.
     *
     * @param tasks draws a task
     * @param count tasks to play
     * @param seed the task seed
     * @param explore whether to sample actions (the trained stochastic policy) or take the argmax
     * @return the mean reward per pull
     */
    public double evaluate(Function<SplittableRandom, Bandit> tasks, int count, long seed, boolean explore) {
        SplittableRandom r = new SplittableRandom(seed);
        double total = 0;
        for (int i = 0; i < count; i++) {
            Bandit env = tasks.apply(r);
            newTask();
            int prev = -1;
            float prevR = 0;
            for (int t = 0; t < horizon; t++) {
                int a = Math.round(act(input(prev, prevR, t), explore)[0]);
                env.reset(r.split());
                prevR = env.step(new float[] {a}).reward();
                prev = a;
                total += prevR;
            }
        }
        return total / (count * (double) horizon);
    }

    @Override
    public String toString() {
        return kind == Kind.RL2 ? "RL2" : "VariBAD";
    }
}
