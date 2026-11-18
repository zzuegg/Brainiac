package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.ReplayBuffer;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tracker;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Maximum a-posteriori policy optimisation (Abdolmaleki et al., 2018,
 * <a href="https://arxiv.org/abs/1806.06920">arXiv:1806.06920</a>) for continuous actions:
 * policy improvement as expectation-maximization.
 * <ul>
 *   <li><b>E-step</b>: for each replayed state, sample {@code K} actions from the target
 *       policy and weight them by {@code softmax(Q / eta)}; the temperature {@code eta}
 *       minimizes the dual {@code eta epsilon + eta mean_s log mean_k exp(Q / eta)}, so the
 *       non-parametric improved policy stays within KL {@code epsilon} of the old one.</li>
 *   <li><b>M-step</b>: fit the parametric Gaussian policy to the weighted actions by
 *       maximum likelihood, with separate KL trust regions on the mean and on the standard
 *       deviation enforced by Lagrange multipliers (decoupled updates).</li>
 * </ul>
 * The critic is a {@link CriticEnsemble} of two, trained by expected-SARSA targets under
 * the target policy. Actions are Gaussian samples clipped to the unit box and scaled to the
 * environment's bounds.
 */
public final class Mpo implements Learner {

    private static final float LOG_STD_MIN = -5f;
    private static final float LOG_STD_MAX = 2f;

    private final int obsDims;
    private final int actDims;
    private final Model actor;
    private final Model targetActor;
    private final Var logStd;
    private Tensor targetLogStd;
    private final CriticEnsemble critics;
    private Optimizer actorOpt = Optimizer.adam(1e-3f);
    private Optimizer criticOpt = Optimizer.adam(3e-3f);
    private float gamma = 0.99f;
    private float tau = 0.005f;
    private int batchSize = 64;
    private int samples = 10;
    private int learningStarts = 300;
    private int targetEvery = 50;
    private double epsilon = 0.1;
    private double epsilonMean = 0.01;
    private double epsilonStd = 1e-4;
    private double eta = 1;
    private double alphaMean = 1;
    private double alphaStd = 100;
    private double lastKlMean;
    private float[] low;
    private float[] high;
    private SplittableRandom random = new SplittableRandom(0);
    private ReplayBuffer buffer;
    private long updates;

    private Mpo(int obsDims, int actDims, int[] hidden) {
        this.obsDims = obsDims;
        this.actDims = actDims;
        this.actor = Nets.mlp(obsDims, actDims, Activation.RELU, 11, hidden);
        this.targetActor = Nets.mlp(obsDims, actDims, Activation.RELU, 11, hidden);
        RlTensors.copy(actor.parameters(), targetActor.parameters());
        this.logStd = Var.param(Tensor.full(Shape.of(1, actDims), (float) Math.log(0.5)));
        this.targetLogStd = logStd.value();
        this.critics = new CriticEnsemble(obsDims + actDims, 2, 0, false, 12, hidden);
    }

    /**
     * @param obsDims values per observation
     * @param actDims values per action
     * @param hidden the hidden layer sizes of actor and critics
     * @return the learner
     */
    public static Mpo of(int obsDims, int actDims, int... hidden) {
        return new Mpo(obsDims, actDims, hidden.clone());
    }

    /**
     * @param e the E-step KL bound
     * @param mean the M-step KL bound on the mean
     * @param std the M-step KL bound on the standard deviation
     * @return this learner
     */
    public Mpo klBounds(double e, double mean, double std) {
        this.epsilon = e;
        this.epsilonMean = mean;
        this.epsilonStd = std;
        return this;
    }

    /** @param k actions sampled per state in the E-step @return this learner */
    public Mpo actionSamples(int k) {
        this.samples = k;
        return this;
    }

    /** @param actorLr the policy learning rate @param criticLr the critic learning rate @return this learner */
    public Mpo learningRates(float actorLr, float criticLr) {
        actorOpt = Optimizer.adam(actorLr);
        criticOpt = Optimizer.adam(criticLr);
        return this;
    }

    /** @param updates updates between copies of the policy into the target (old) policy @return this learner */
    public Mpo targetUpdateEvery(int updates) {
        this.targetEvery = Math.max(1, updates);
        return this;
    }

    /** @param steps random steps before learning starts @return this learner */
    public Mpo learningStarts(int steps) {
        this.learningStarts = steps;
        return this;
    }

    /** @param seed the seed @return this learner */
    public Mpo seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return the E-step temperature */
    public double temperature() {
        return eta;
    }

    /** @return the KL between the old and new policy mean in the last M-step */
    public double lastKlMean() {
        return lastKlMean;
    }

    private static double clampLogStd(double ls) {
        return Math.max(LOG_STD_MIN, Math.min(LOG_STD_MAX, ls));
    }

    private float[] unit(float[] observation, boolean explore) {
        float[] out = RlTensors.row(actor, observation);
        float[] u = new float[actDims];
        for (int i = 0; i < actDims; i++) {
            double m = Math.tanh(out[i]);
            if (explore) {
                double ls = clampLogStd(logStd.value().to(DType.FLOAT32).toArray()[i]);
                m += Math.exp(ls) * random.nextGaussian();
            }
            u[i] = (float) Math.max(-1, Math.min(1, m));
        }
        return u;
    }

    private float[] scale(float[] unit) {
        if (low == null) {
            return unit;
        }
        float[] a = new float[unit.length];
        for (int i = 0; i < a.length; i++) {
            a[i] = low[i] + (unit[i] + 1) * 0.5f * (high[i] - low[i]);
        }
        return a;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        return scale(unit(observation, explore));
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Box box) || box.size() != actDims
                || env.observationSpace().size() != obsDims) {
            throw new IllegalArgumentException("needs " + obsDims + " observation values and a box of "
                    + actDims + " actions");
        }
        low = box.low();
        high = box.high();
        if (buffer == null) {
            buffer = new ReplayBuffer((int) Math.max(1000, steps), obsDims, actDims);
        }
        Tracker recorder = new Tracker(listener);
        float[] obs = env.reset(random.split());
        long taken = 0;
        while (recorder.steps() < steps) {
            float[] u;
            if (taken < learningStarts) {
                u = new float[actDims];
                for (int i = 0; i < actDims; i++) {
                    u[i] = (float) random.nextDouble(-1, 1);
                }
            } else {
                u = unit(obs, true);
            }
            Environment.Step s = env.step(scale(u));
            taken++;
            buffer.add(new Transition(obs, u, s.reward(), s.observation(), s.terminated(), gamma));
            obs = recorder.step(s) ? env.reset(random.split()) : s.observation();
            if (taken >= learningStarts && buffer.size() >= batchSize) {
                update();
            }
        }
        return recorder.result();
    }

    /** Samples K clipped actions per row from the target policy: [B*K, actDims] plus mean and std. */
    private Tensor[] sampleTarget(Tensor obs) {
        float[] out = targetActor.forward(Var.of(obs), false).value().to(DType.FLOAT32).toArray();
        int b = obs.shape().dim(0);
        float[] tls = targetLogStd.to(DType.FLOAT32).toArray();
        float[] acts = new float[b * samples * actDims];
        float[] mean = new float[b * actDims];
        float[] std = new float[b * actDims];
        for (int i = 0; i < b; i++) {
            for (int d = 0; d < actDims; d++) {
                mean[i * actDims + d] = (float) Math.tanh(out[i * actDims + d]);
                double ls = clampLogStd(tls[d]);
                std[i * actDims + d] = (float) Math.exp(ls);
                for (int k = 0; k < samples; k++) {
                    double a = mean[i * actDims + d] + std[i * actDims + d] * random.nextGaussian();
                    acts[(i * samples + k) * actDims + d] = (float) Math.max(-1, Math.min(1, a));
                }
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(b * samples, actDims), acts),
            Tensor.of(Shape.of(b, actDims), mean), Tensor.of(Shape.of(b, actDims), std)};
    }

    private Tensor repeatRows(Tensor x) {
        float[] v = x.to(DType.FLOAT32).toArray();
        int b = x.shape().dim(0);
        int w = x.shape().dim(1);
        float[] r = new float[b * samples * w];
        for (int i = 0; i < b; i++) {
            for (int k = 0; k < samples; k++) {
                System.arraycopy(v, i * w, r, (i * samples + k) * w, w);
            }
        }
        return Tensor.of(Shape.of(b * samples, w), r);
    }

    private void update() {
        ReplayBuffer.Batch b = buffer.sample(batchSize, random, actor.device());
        // critic: expected SARSA under the target policy
        Tensor nextRep = repeatRows(b.nextObservations());
        Tensor nextActs = sampleTarget(b.nextObservations())[0];
        float[] qn = critics.targetMin(Tensor.concat(1, nextRep, nextActs), 2, random).to(DType.FLOAT32).toArray();
        float[] v = new float[batchSize];
        for (int i = 0; i < batchSize; i++) {
            for (int k = 0; k < samples; k++) {
                v[i] += qn[i * samples + k] / samples;
            }
        }
        Tensor y = b.rewards().add(b.discounts().mul(b.notDone()).mul(Tensor.of(Shape.of(batchSize, 1), v)));
        RlTensors.zeroGrad(actor.parameters());
        critics.update(Tensor.concat(1, b.observations(), b.actions()), y, criticOpt);

        // E-step: Q of sampled actions, dual temperature, weights
        Tensor obsRep = repeatRows(b.observations());
        Tensor[] t = sampleTarget(b.observations());
        float[] q = critics.targetMin(Tensor.concat(1, obsRep, t[0]), 2, random).to(DType.FLOAT32).toArray();
        eta = solveTemperature(q);
        float[] w = new float[batchSize * samples];
        for (int i = 0; i < batchSize; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < samples; k++) {
                max = Math.max(max, q[i * samples + k]);
            }
            double z = 0;
            for (int k = 0; k < samples; k++) {
                z += Math.exp((q[i * samples + k] - max) / eta);
            }
            for (int k = 0; k < samples; k++) {
                w[i * samples + k] = (float) (Math.exp((q[i * samples + k] - max) / eta) / z);
            }
        }

        // M-step: decoupled weighted maximum likelihood with KL penalties on mean and std
        Var out = actor.forward(Var.of(obsRep), true);
        Var mean = out.tanh();
        Var ls = logStd;
        Tensor oldMean = repeatRows(t[1]);
        Tensor oldStd = repeatRows(t[2]);
        Var acts = Var.of(t[0]);
        Var std = ls.exp();
        Var llMean = acts.sub(mean).div(oldStd).square().mul(-0.5f);
        Var llStd = acts.sub(Var.of(oldMean)).div(std).square().mul(-0.5f).sub(ls);
        Tensor weights = Tensor.of(Shape.of(batchSize * samples, 1), w);
        Var ll = llMean.add(llStd).sum(1, true).mul(weights).sum().mul(1f / batchSize);
        Var klMean = mean.sub(Var.of(oldMean)).div(oldStd).square().mul(0.5f).sum(1, true).mean();
        Var ratio = Var.of(oldStd).div(std).square();
        Var klStd = ratio.sub(1f).sub(ratio.log()).mul(0.5f).sum(1, true).mean();
        Var loss = ll.neg().add(klMean.mul((float) alphaMean)).add(klStd.mul((float) alphaStd));
        loss.backward();
        RlTensors.clip(actor.parameters(), 40f);
        List<Var> policyParams = new ArrayList<>(actor.parameters());
        policyParams.add(logStd);
        actorOpt.step(policyParams);
        RlTensors.zeroGrad(critics.parameters());
        lastKlMean = klMean.value().item();
        double kls = klStd.value().item();
        alphaMean = Math.max(0, Math.min(1e4, alphaMean + 10 * (lastKlMean - epsilonMean) / epsilonMean * 0.1));
        alphaStd = Math.max(0, Math.min(1e5, alphaStd + 10 * (kls - epsilonStd) / epsilonStd * 0.1));

        critics.blend(tau);
        updates++;
        if (updates % targetEvery == 0) {
            RlTensors.copy(actor.parameters(), targetActor.parameters());
            targetLogStd = logStd.value();
        }
    }

    /** Minimizes the convex dual g(eta) = eta eps + eta mean_i log mean_k exp(q_ik / eta) by Newton steps on log eta. */
    private double solveTemperature(float[] q) {
        double logEta = Math.log(Math.max(1e-4, eta));
        for (int it = 0; it < 20; it++) {
            double h = 1e-3;
            double g0 = dual(q, Math.exp(logEta));
            double gp = dual(q, Math.exp(logEta + h));
            double gm = dual(q, Math.exp(logEta - h));
            double d1 = (gp - gm) / (2 * h);
            double d2 = (gp - 2 * g0 + gm) / (h * h);
            double step = d2 > 1e-12 ? d1 / d2 : Math.signum(d1) * 0.5;
            logEta -= Math.max(-1, Math.min(1, step));
            logEta = Math.max(Math.log(1e-4), Math.min(Math.log(1e4), logEta));
            if (Math.abs(step) < 1e-4) {
                break;
            }
        }
        return Math.exp(logEta);
    }

    private double dual(float[] q, double e) {
        double sum = 0;
        for (int i = 0; i < batchSize; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < samples; k++) {
                max = Math.max(max, q[i * samples + k]);
            }
            double z = 0;
            for (int k = 0; k < samples; k++) {
                z += Math.exp((q[i * samples + k] - max) / e);
            }
            sum += max / e + Math.log(z / samples);
        }
        return e * epsilon + e * sum / batchSize;
    }

    @Override
    public String toString() {
        return "MPO(eta " + String.format("%.3f", eta) + ")";
    }
}
