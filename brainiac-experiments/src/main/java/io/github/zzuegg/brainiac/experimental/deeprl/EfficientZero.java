package io.github.zzuegg.brainiac.experimental.deeprl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.Mcts;
import io.github.zzuegg.brainiac.experimental.rl.RlGruCell;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.rl.RlTensors;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tracker;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * EfficientZero (Ye et al., "Mastering Atari Games with Limited Data", NeurIPS 2021,
 * <a href="https://arxiv.org/abs/2111.00210">arXiv:2111.00210</a>): the three additions that make
 * MuZero sample-efficient, each a switch, on the same structure as
 * {@link io.github.zzuegg.brainiac.experimental.rl.MuZeroLite} (representation, dynamics and
 * prediction networks planned over with {@link Mcts}); with all three off it is MuZero-lite.
 *
 * <ul>
 *   <li>{@link Components#consistency()}: a SimSiam-style self-supervised loss makes the
 *       dynamics model's predicted latent, after a projection and a predictor, match the
 *       (stop-gradient) projection of the encoder's latent of the real next observation, so the
 *       model learns from every pixel of every transition rather than from scalar targets
 *       alone.</li>
 *   <li>{@link Components#valuePrefix()}: a recurrent cell ({@link RlGruCell}) over the
 *       unrolled latents predicts the cumulative reward since the root instead of each step's
 *       reward, which removes the hard problem of predicting exactly when a reward arrives; the
 *       search uses differences of consecutive prefixes as rewards.</li>
 *   <li>{@link Components#offPolicyCorrection()}: value targets of older trajectories bootstrap
 *       after fewer real rewards (the horizon shrinks with the trajectory's age) and from a fresh
 *       value of the current model instead of the stale root value stored at collection (this
 *       implementation re-evaluates the prediction network rather than re-running the full
 *       search, to keep the reanalysis cheap on a CPU).</li>
 * </ul>
 */
public final class EfficientZero implements Learner {

    /**
     * Which additions are on.
     *
     * @param consistency the self-supervised consistency loss
     * @param valuePrefix the recurrent value-prefix reward head
     * @param offPolicyCorrection adaptive horizons and fresh bootstrap values
     */
    public record Components(boolean consistency, boolean valuePrefix, boolean offPolicyCorrection) {
        /** @return all three */
        public static Components all() {
            return new Components(true, true, true);
        }

        /** @return none: MuZero-lite */
        public static Components none() {
            return new Components(false, false, false);
        }
    }

    private static final class Trajectory {
        final List<float[]> observations = new ArrayList<>();
        final List<Integer> actions = new ArrayList<>();
        final List<Float> rewards = new ArrayList<>();
        final List<float[]> policies = new ArrayList<>();
        final List<Float> rootValues = new ArrayList<>();
        boolean terminated;
        long collectedAt;
    }

    /** A search state: the latent, the value-prefix cell's hidden state and the prefix so far. */
    private record Node(float[] latent, float[] hidden, float prefix) {
    }

    private final int obsDims;
    private final int actions;
    private final int latent;
    private final Components parts;
    private final Model representation;
    private final Model dynamics;
    private final Model prediction;
    private final Model projection;
    private final Model predictor;
    private final RlGruCell prefixCell;
    private final Model prefixHead;
    private Optimizer optimizer = Optimizer.adam(3e-3f);
    private int simulations = 8;
    private int unroll = 3;
    private int tdSteps = 5;
    private float gamma = 0.97f;
    private int batchSize = 32;
    private int trainEvery = 2;
    private int window = 200;
    private int warmup = 200;
    private SplittableRandom random = new SplittableRandom(0);
    private final List<Trajectory> replay = new ArrayList<>();
    private long stepsTaken;
    private long updates;
    private double lastLoss = Double.NaN;

    private EfficientZero(int obsDims, int actions, RlNetworkFactory nets, int latent, Components parts, long seed) {
        this.obsDims = obsDims;
        this.actions = actions;
        this.latent = latent;
        this.parts = parts;
        long s = seed * 1000;
        representation = nets.create(obsDims, latent, s + 31);
        dynamics = nets.create(latent + actions, latent + 1, s + 32);
        prediction = nets.create(latent, actions + 1, s + 33);
        projection = nets.create(latent, latent, s + 34);
        predictor = nets.create(latent, latent, s + 35);
        prefixCell = new RlGruCell(latent, latent, s + 36);
        prefixHead = Model.sequential().input(latent).dense(1).seed(s + 37).build();
        random = new SplittableRandom(seed);
    }

    /**
     * @param observationDims values per observation
     * @param actions discrete actions
     * @param networks builds the representation, dynamics, prediction, projection and predictor
     * @param latentSize the latent size
     * @param components the additions in use
     * @param seed the seed of networks, search and replay
     * @return the learner
     */
    public static EfficientZero of(int observationDims, int actions, RlNetworkFactory networks, int latentSize,
            Components components, long seed) {
        return new EfficientZero(observationDims, actions, Objects.requireNonNull(networks), latentSize, components,
                seed);
    }

    /** @param n MCTS simulations per move @return this learner */
    public EfficientZero simulations(int n) {
        this.simulations = n;
        return this;
    }

    /** @param k model unroll steps @return this learner */
    public EfficientZero unroll(int k) {
        this.unroll = k;
        return this;
    }

    /** @param n real rewards before bootstrapping (for fresh data) @return this learner */
    public EfficientZero tdSteps(int n) {
        this.tdSteps = n;
        return this;
    }

    /** @param discount gamma @return this learner */
    public EfficientZero discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param steps environment steps between training batches @return this learner */
    public EfficientZero trainEvery(int steps) {
        this.trainEvery = Math.max(1, steps);
        return this;
    }

    /** @param steps random steps first @return this learner */
    public EfficientZero warmup(int steps) {
        this.warmup = steps;
        return this;
    }

    /** @param rate the learning rate @return this learner */
    public EfficientZero learningRate(float rate) {
        this.optimizer = Optimizer.adam(rate);
        return this;
    }

    /** @return the gradient updates taken */
    public long updates() {
        return updates;
    }

    /** @return the loss of the last batch */
    public double lastLoss() {
        return lastLoss;
    }

    private static float[] row(Model m, float[] in) {
        Tensor x = Tensor.of(Shape.of(1, in.length), in);
        return Var.noGrad(() -> m.forward(Var.of(x), false)).value().toArray();
    }

    private float[] latentOf(float[] observation) {
        float[] s = row(representation, observation);
        for (int i = 0; i < s.length; i++) {
            s[i] = (float) Math.tanh(s[i]);
        }
        return s;
    }

    private final Mcts.Model<Node> model = new Mcts.Model<>() {
        @Override
        public int actions() {
            return actions;
        }

        @Override
        public Mcts.Evaluation evaluate(Node state) {
            float[] out = row(prediction, state.latent());
            float max = Float.NEGATIVE_INFINITY;
            for (int a = 0; a < actions; a++) {
                max = Math.max(max, out[a]);
            }
            float[] p = new float[actions];
            float sum = 0;
            for (int a = 0; a < actions; a++) {
                p[a] = (float) Math.exp(out[a] - max);
                sum += p[a];
            }
            for (int a = 0; a < actions; a++) {
                p[a] /= sum;
            }
            return new Mcts.Evaluation(p, out[actions], false);
        }

        @Override
        public Mcts.Step<Node> step(Node state, int action) {
            float[] in = new float[latent + actions];
            System.arraycopy(state.latent(), 0, in, 0, latent);
            in[latent + action] = 1;
            float[] out = row(dynamics, in);
            float[] next = new float[latent];
            for (int i = 0; i < latent; i++) {
                next[i] = (float) Math.tanh(out[i]);
            }
            if (!parts.valuePrefix()) {
                return new Mcts.Step<>(new Node(next, null, 0), out[latent]);
            }
            Tensor x = Tensor.of(Shape.of(1, latent), next);
            Tensor h = Tensor.of(Shape.of(1, latent), state.hidden());
            float[] hidden = Var.noGrad(() -> prefixCell.step(Var.of(x), Var.of(h))).value().toArray();
            float prefix = row(prefixHead, hidden)[0];
            return new Mcts.Step<>(new Node(next, hidden, prefix), prefix - state.prefix());
        }
    };

    private Mcts.Result plan(float[] observation, boolean explore) {
        Mcts<Node> m = Mcts.of(model).simulations(simulations).discount(gamma);
        if (explore) {
            m.rootNoise(0.3f, 0.25f);
        }
        return m.search(new Node(latentOf(observation), new float[latent], 0), random);
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        Mcts.Result r = plan(observation, explore);
        return new float[] {explore ? r.sample(1.0, random) : r.best()};
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Discrete d) || d.n() != actions
                || env.observationSpace().size() != obsDims) {
            throw new IllegalArgumentException("EfficientZero needs " + obsDims + " observation values and "
                    + actions + " discrete actions");
        }
        Tracker recorder = new Tracker(listener);
        float[] obs = env.reset(random.split());
        Trajectory t = new Trajectory();
        t.collectedAt = stepsTaken;
        while (recorder.steps() < steps) {
            Mcts.Result r = plan(obs, true);
            double temperature = recorder.steps() < steps / 2 ? 1.0 : 0.5;
            int a = stepsTaken < warmup ? random.nextInt(actions) : r.sample(temperature, random);
            Environment.Step s = env.step(a);
            stepsTaken++;
            t.observations.add(obs);
            t.actions.add(a);
            t.rewards.add(s.reward());
            t.policies.add(r.policy());
            t.rootValues.add(r.value());
            if (recorder.step(s)) {
                t.terminated = s.terminated();
                t.observations.add(s.observation());
                replay.add(t);
                if (replay.size() > window) {
                    replay.removeFirst();
                }
                t = new Trajectory();
                t.collectedAt = stepsTaken;
                obs = env.reset(random.split());
            } else {
                obs = s.observation();
            }
            if (stepsTaken >= warmup && !replay.isEmpty() && stepsTaken % trainEvery == 0) {
                learn();
            }
        }
        return recorder.result();
    }

    /** The bootstrap horizon of a trajectory: shorter the older it is (off-policy correction). */
    private int horizon(Trajectory t) {
        if (!parts.offPolicyCorrection()) {
            return tdSteps;
        }
        double age = (stepsTaken - t.collectedAt) / (double) Math.max(1, stepsTaken);
        return Math.max(1, tdSteps - (int) Math.floor(age * tdSteps));
    }

    private float bootstrapValue(Trajectory t, int index) {
        if (!parts.offPolicyCorrection()) {
            return t.rootValues.get(index);
        }
        return row(prediction, latentOf(t.observations.get(index)))[actions];
    }

    private float valueTarget(Trajectory t, int index) {
        int len = t.actions.size();
        if (index >= len) {
            return 0;
        }
        int n = horizon(t);
        int boot = index + n;
        float v = 0;
        if (boot < len) {
            v = bootstrapValue(t, boot) * (float) Math.pow(gamma, n);
        } else if (!t.terminated) {
            v = bootstrapValue(t, len - 1) * (float) Math.pow(gamma, len - 1 - index);
        }
        for (int i = index; i < Math.min(boot, len); i++) {
            v += t.rewards.get(i) * (float) Math.pow(gamma, i - index);
        }
        return v;
    }

    private Var represent(Var obs, boolean training) {
        return representation.forward(obs, training).tanh();
    }

    private static Var l2(Var x) {
        return x.div(x.square().sum(1, true).add(1e-6f).sqrt());
    }

    private void learn() {
        int b = batchSize;
        float[] obs = new float[b * obsDims];
        float[][] nextObs = new float[unroll][b * obsDims];
        float[][] nextMask = new float[unroll][b];
        float[][] actionHot = new float[unroll][b * actions];
        float[][] rewardT = new float[unroll][b];
        float[][] valueT = new float[unroll + 1][b];
        float[][] policyT = new float[unroll + 1][b * actions];
        float[][] policyMask = new float[unroll + 1][b];
        for (int i = 0; i < b; i++) {
            Trajectory t = replay.get(random.nextInt(replay.size()));
            int len = t.actions.size();
            int start = random.nextInt(len);
            System.arraycopy(t.observations.get(start), 0, obs, i * obsDims, obsDims);
            for (int k = 0; k <= unroll; k++) {
                int idx = start + k;
                valueT[k][i] = valueTarget(t, idx);
                if (idx < len) {
                    System.arraycopy(t.policies.get(idx), 0, policyT[k], i * actions, actions);
                    policyMask[k][i] = 1;
                }
                if (k < unroll) {
                    int a = idx < len ? t.actions.get(idx) : random.nextInt(actions);
                    actionHot[k][i * actions + a] = 1;
                    rewardT[k][i] = idx < len ? t.rewards.get(idx) : 0;
                    if (idx + 1 < t.observations.size()) {
                        System.arraycopy(t.observations.get(idx + 1), 0, nextObs[k], i * obsDims, obsDims);
                        nextMask[k][i] = idx < len ? 1 : 0;
                    }
                }
            }
        }
        Var state = represent(Var.of(Tensor.of(Shape.of(b, obsDims), obs)), true);
        Var loss = predictionLoss(state, valueT[0], policyT[0], policyMask[0]);
        Var hidden = prefixCell.zeroState(b);
        float[] prefixTarget = new float[b];
        for (int k = 0; k < unroll; k++) {
            Var out = dynamics.forward(Var.concat(1, state, Var.of(Tensor.of(Shape.of(b, actions), actionHot[k]))), true);
            state = out.slice(1, 0, latent).tanh();
            if (parts.valuePrefix()) {
                hidden = prefixCell.step(state, hidden);
                for (int i = 0; i < b; i++) {
                    prefixTarget[i] += rewardT[k][i];
                }
                Var prefix = prefixHead.forward(hidden, true);
                loss = loss.add(prefix.sub(Tensor.of(Shape.of(b, 1), prefixTarget.clone())).square().mean());
            } else {
                loss = loss.add(out.slice(1, latent, latent + 1).sub(Tensor.of(Shape.of(b, 1), rewardT[k]))
                        .square().mean());
            }
            if (parts.consistency()) {
                Tensor real = Tensor.of(Shape.of(b, obsDims), nextObs[k]);
                Tensor target = Var.noGrad(() -> l2(projection.forward(represent(Var.of(real), false), false))).value();
                Var pred = l2(predictor.forward(projection.forward(state, true), true));
                Var cos = pred.mul(target).sum(1, true).mul(Tensor.of(Shape.of(b, 1), nextMask[k])).mean().neg();
                loss = loss.add(cos.mul(2f));
            }
            loss = loss.add(predictionLoss(state, valueT[k + 1], policyT[k + 1], policyMask[k + 1]));
        }
        loss = loss.div(unroll + 1);
        loss.backward();
        List<Var> params = new ArrayList<>(representation.parameters());
        params.addAll(dynamics.parameters());
        params.addAll(prediction.parameters());
        if (parts.consistency()) {
            params.addAll(projection.parameters());
            params.addAll(predictor.parameters());
        }
        if (parts.valuePrefix()) {
            params.addAll(prefixCell.parameters());
            params.addAll(prefixHead.parameters());
        }
        RlTensors.clip(params, 5f);
        optimizer.step(params);
        RlTensors.zeroGrad(projection.parameters());
        RlTensors.zeroGrad(predictor.parameters());
        RlTensors.zeroGrad(prefixCell.parameters());
        RlTensors.zeroGrad(prefixHead.parameters());
        lastLoss = loss.value().item();
        updates++;
    }

    private Var predictionLoss(Var state, float[] value, float[] policy, float[] mask) {
        int b = value.length;
        Var out = prediction.forward(state, true);
        Var v = out.slice(1, actions, actions + 1);
        Var valueLoss = v.sub(Tensor.of(Shape.of(b, 1), value)).square().mean();
        Var ce = RlTensors.logSoftmax(out.slice(1, 0, actions)).mul(Tensor.of(Shape.of(b, actions), policy))
                .sum(1, true).neg().mul(Tensor.of(Shape.of(b, 1), mask)).mean();
        return valueLoss.add(ce);
    }

    @Override
    public String toString() {
        return "EfficientZero" + parts;
    }
}
