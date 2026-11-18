package io.github.zzuegg.brainiac.experimental.marl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.MultiAgentEnvironment;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.rl.RlTensors;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Schedule;
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
 * Value factorisation that spans the whole class of joint values with the
 * individual-global-max (IGM) property, beyond VDN and QMIX, on the same centralised-training
 * loop as {@code ValueDecomposition} (shared agent Q-network, replay, double-Q targets) with
 * pluggable {@link FactorizedMixer}s:
 *
 * <ul>
 *   <li>{@link Method#VDN} and {@link Method#QMIX}: the sum and the monotonic hypernetwork mixer,
 *       for reference;</li>
 *   <li>{@link Method#QTRAN_BASE} (Son et al., ICML 2019,
 *       <a href="https://arxiv.org/abs/1905.05408">arXiv:1905.05408</a>): an unrestricted joint
 *       {@code Q_jt(s, u)} trained by TD, a state value {@code V(s)}, and soft constraints that
 *       make {@code sum_i Q_i - Q_jt + V} zero at the greedy joint action and non-negative
 *       elsewhere;</li>
 *   <li>{@link Method#QPLEX} (Wang et al., ICLR 2021,
 *       <a href="https://arxiv.org/abs/2008.01062">arXiv:2008.01062</a>): the duplex dueling
 *       form {@code Q_tot = sum_i (w_i V_i + b_i) + sum_i lambda_i(s, u) w_i A_i} with positive
 *       state-dependent {@code w_i} and positive joint-action weights {@code lambda_i} (one
 *       head here instead of the paper's multi-head attention);</li>
 *   <li>{@link Method#QFIX_SUM} and {@link Method#QFIX_MONO} (Baisero et al., RLJ 2026,
 *       <a href="https://arxiv.org/abs/2505.10484">arXiv:2505.10484</a>): a thin fixing layer
 *       {@code Q = w(s, u) A_fixee(s, u) + b(s)} with {@code w > 0} over the fixee's advantage,
 *       {@code sum_i A_i} for VDN or {@code f_mono(q) - f_mono(v)} for QMIX.</li>
 * </ul>
 */
public final class CompleteFactorization {

    /** The mixer. */
    public enum Method {
        /** sum of utilities */
        VDN,
        /** monotonic hypernetwork mixer */
        QMIX,
        /** QTRAN-base with soft consistency constraints */
        QTRAN_BASE,
        /** duplex dueling with positive joint-action weights */
        QPLEX,
        /** QFIX over VDN */
        QFIX_SUM,
        /** QFIX over QMIX */
        QFIX_MONO
    }

    private record Sample(float[][] obs, float[] state, int[] actions, float reward, float[][] next,
            float[] nextState, boolean terminated) {
    }

    private final Method method;
    private final int agents;
    private final int obsSize;
    private final int stateSize;
    private final int actions;
    private final Model q;
    private final Model qTarget;
    private final FactorizedMixer mixer;
    private final FactorizedMixer mixerTarget;
    private Optimizer optimizer = Optimizer.adam(5e-3f);
    private Schedule epsilon = Schedule.linear(1, 0.05, 2_000);
    private float gamma = 0.99f;
    private int batchSize = 32;
    private int bufferSize = 5_000;
    private int targetEvery = 50;
    private SplittableRandom random = new SplittableRandom(0);
    private final List<Sample> buffer = new ArrayList<>();
    private int position;
    private long steps;
    private long updates;

    private CompleteFactorization(Method method, MultiAgentEnvironment env, RlNetworkFactory nets, long seed) {
        this.method = method;
        this.agents = env.agents();
        this.obsSize = env.observationSize();
        this.stateSize = env.stateSize();
        this.actions = env.actions();
        q = nets.create(obsSize, actions, seed);
        qTarget = nets.create(obsSize, actions, seed);
        RlTensors.copy(q.parameters(), qTarget.parameters());
        mixer = mixer(method, seed + 1);
        mixerTarget = mixer(method, seed + 1);
        RlTensors.copy(mixer.parameters(), mixerTarget.parameters());
    }

    /**
     * @param method the mixer
     * @param env the environment (for sizes)
     * @param networks builds the shared agent network ({@code observation -> actions})
     * @param seed the initialisation seed
     * @return the learner
     */
    public static CompleteFactorization of(Method method, MultiAgentEnvironment env, RlNetworkFactory networks,
            long seed) {
        return new CompleteFactorization(Objects.requireNonNull(method), env, Objects.requireNonNull(networks), seed);
    }

    /** @param s exploration by step @return this learner */
    public CompleteFactorization epsilon(Schedule s) {
        this.epsilon = Objects.requireNonNull(s);
        return this;
    }

    /** @param discount gamma @return this learner */
    public CompleteFactorization discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param seed the exploration seed @return this learner */
    public CompleteFactorization seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return gradient updates taken */
    public long updates() {
        return updates;
    }

    private FactorizedMixer mixer(Method m, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = agents;
        int a = actions;
        int s = stateSize;
        return switch (m) {
            case VDN -> new Sum();
            case QMIX -> new Monotonic(n, s, r);
            case QTRAN_BASE -> new Qtran(n, a, s, r);
            case QPLEX -> new Qplex(n, a, s, r);
            case QFIX_SUM -> new Qfix(n, a, s, null, r);
            case QFIX_MONO -> new Qfix(n, a, s, new Monotonic(n, s, r), r);
        };
    }

    private static Var chosenValues(Var utilities, Tensor chosen) {
        return utilities.mul(chosen).sum(2, false);
    }

    /** {@code sum_i Q_i(u_i)}. */
    static final class Sum implements FactorizedMixer {
        @Override
        public Var mix(Var utilities, Tensor chosen, Tensor state) {
            return chosenValues(utilities, chosen).sum(1, true);
        }

        @Override
        public List<Var> parameters() {
            return List.of();
        }
    }

    /** The QMIX mixer: non-negative hypernetwork weights over the state. */
    static final class Monotonic implements FactorizedMixer {
        private final int n;
        private final int embed = 8;
        private final MixerNet w1;
        private final MixerNet b1;
        private final MixerNet w2;
        private final MixerNet v;

        Monotonic(int n, int s, SplittableRandom r) {
            this.n = n;
            w1 = new MixerNet(r, s, n * embed);
            b1 = new MixerNet(r, s, embed);
            w2 = new MixerNet(r, s, embed);
            v = new MixerNet(r, s, embed, 1);
        }

        /** The mixer over per-agent values {@code [b, n]}. */
        Var monotonic(Var qs, Tensor state) {
            int b = qs.shape().dim(0);
            Var st = Var.of(state);
            Var w = w1.apply(st).abs().reshape(b, n, embed);
            Var hidden = qs.reshape(b, n, 1).mul(w).sum(1, false).add(b1.apply(st)).relu();
            return hidden.mul(w2.apply(st).abs()).sum(1, true).add(v.apply(st));
        }

        @Override
        public Var mix(Var utilities, Tensor chosen, Tensor state) {
            return monotonic(chosenValues(utilities, chosen), state);
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(w1.parameters());
            p.addAll(b1.parameters());
            p.addAll(w2.parameters());
            p.addAll(v.parameters());
            return p;
        }
    }

    private static Tensor flat(Tensor chosen) {
        int b = chosen.shape().dim(0);
        return chosen.reshape(b, chosen.shape().dim(1) * chosen.shape().dim(2));
    }

    /** QTRAN-base: an unrestricted joint value with consistency constraints on the utility sum. */
    static final class Qtran implements FactorizedMixer {
        private final MixerNet joint;
        private final MixerNet value;

        Qtran(int n, int a, int s, SplittableRandom r) {
            joint = new MixerNet(r, s + n * a, 32, 32, 1);
            value = new MixerNet(r, s, 16, 1);
        }

        @Override
        public Var mix(Var utilities, Tensor chosen, Tensor state) {
            return joint.apply(Var.of(Tensor.concat(1, state, flat(chosen))));
        }

        @Override
        public Var auxiliaryLoss(Var utilities, Tensor chosen, Tensor greedy, Tensor state) {
            Var v = value.apply(Var.of(state));
            Tensor jtGreedy = joint.apply(Var.of(Tensor.concat(1, state, flat(greedy)))).value();
            Tensor jtChosen = joint.apply(Var.of(Tensor.concat(1, state, flat(chosen)))).value();
            Var opt = chosenValues(utilities, greedy).sum(1, true).sub(jtGreedy).add(v).square().mean();
            Var nopt = chosenValues(utilities, chosen).sum(1, true).sub(jtChosen).add(v).minimum(0f).square().mean();
            return opt.add(nopt);
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(joint.parameters());
            p.addAll(value.parameters());
            return p;
        }
    }

    /** QPLEX's duplex dueling mixer with a single positive attention head. */
    static final class Qplex implements FactorizedMixer {
        private final int n;
        private final MixerNet w;
        private final MixerNet bias;
        private final MixerNet lambda;

        Qplex(int n, int a, int s, SplittableRandom r) {
            this.n = n;
            w = new MixerNet(r, s, 16, n);
            bias = new MixerNet(r, s, 16, n);
            lambda = new MixerNet(r, s + n * a, 32, n);
        }

        @Override
        public Var mix(Var utilities, Tensor chosen, Tensor state) {
            Var st = Var.of(state);
            Var qs = chosenValues(utilities, chosen);
            Var vs = utilities.max(2, false);
            Var adv = qs.sub(vs);
            Var wi = w.apply(st).abs().add(1e-3f);
            Var v = vs.mul(wi).add(bias.apply(st)).sum(1, true);
            Var lam = lambda.apply(Var.of(Tensor.concat(1, state, flat(chosen)))).softplus().add(1e-3f);
            return v.add(adv.mul(wi).mul(lam).sum(1, true));
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(w.parameters());
            p.addAll(bias.parameters());
            p.addAll(lambda.parameters());
            return p;
        }
    }

    /** QFIX: a positive weight on the fixee's advantage plus a state bias. */
    static final class Qfix implements FactorizedMixer {
        private final Monotonic fixee;
        private final MixerNet weight;
        private final MixerNet bias;

        Qfix(int n, int a, int s, Monotonic fixee, SplittableRandom r) {
            this.fixee = fixee;
            weight = new MixerNet(r, s + n * a, 32, 1);
            bias = new MixerNet(r, s, 16, 1);
        }

        @Override
        public Var mix(Var utilities, Tensor chosen, Tensor state) {
            Var qs = chosenValues(utilities, chosen);
            Var vs = utilities.max(2, false);
            Var adv = fixee == null ? qs.sub(vs).sum(1, true)
                    : fixee.monotonic(qs, state).sub(fixee.monotonic(vs, state));
            Var w = weight.apply(Var.of(Tensor.concat(1, state, flat(chosen)))).softplus().add(1e-3f);
            return adv.mul(w).add(bias.apply(Var.of(state)));
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(weight.parameters());
            p.addAll(bias.parameters());
            if (fixee != null) {
                p.addAll(fixee.parameters());
            }
            return p;
        }
    }

    /**
     * @param observations one observation per agent
     * @param explore whether to act epsilon-greedily
     * @return the joint action (each agent greedy on its own utility)
     */
    public int[] act(float[][] observations, boolean explore) {
        int[] a = new int[agents];
        for (int i = 0; i < agents; i++) {
            if (explore && random.nextDouble() < epsilon.at(steps)) {
                a[i] = random.nextInt(actions);
            } else {
                a[i] = RlTensors.argmax(RlTensors.row(q, observations[i]));
            }
        }
        return a;
    }

    /**
     * Trains for a number of joint steps.
     *
     * @param env the environment
     * @param totalSteps joint steps
     * @return the mean team return of the last 50 episodes
     */
    public double fit(MultiAgentEnvironment env, long totalSteps) {
        Tracker rec = new Tracker(null);
        float[][] obs = env.reset(random.split());
        float[] state = env.state();
        while (rec.steps() < totalSteps) {
            int[] a = act(obs, true);
            MultiAgentEnvironment.Step s = env.step(a);
            steps++;
            float[] nextState = env.state();
            Sample sample = new Sample(obs, state, a, s.reward(), s.observations(), nextState, s.terminated());
            if (buffer.size() < bufferSize) {
                buffer.add(sample);
            } else {
                buffer.set(position, sample);
                position = (position + 1) % bufferSize;
            }
            if (rec.step(new Environment.Step(new float[0], s.reward(), s.terminated(), s.truncated()))) {
                obs = env.reset(random.split());
                state = env.state();
            } else {
                obs = s.observations();
                state = nextState;
            }
            if (buffer.size() >= batchSize) {
                learn();
            }
        }
        return rec.result().meanReturn(50);
    }

    private Tensor greedyHot(Tensor utilities, int b) {
        float[] u = utilities.to(DType.FLOAT32).toArray();
        float[] hot = new float[b * agents * actions];
        for (int i = 0; i < b * agents; i++) {
            int best = 0;
            for (int k = 1; k < actions; k++) {
                if (u[i * actions + k] > u[i * actions + best]) {
                    best = k;
                }
            }
            hot[i * actions + best] = 1;
        }
        return Tensor.of(Shape.of(b, agents, actions), hot);
    }

    private void learn() {
        int b = batchSize;
        float[] o = new float[b * agents * obsSize];
        float[] n = new float[b * agents * obsSize];
        float[] st = new float[b * stateSize];
        float[] nst = new float[b * stateSize];
        float[] hot = new float[b * agents * actions];
        float[] r = new float[b];
        float[] notDone = new float[b];
        for (int i = 0; i < b; i++) {
            Sample s = buffer.get(random.nextInt(buffer.size()));
            for (int k = 0; k < agents; k++) {
                System.arraycopy(s.obs()[k], 0, o, (i * agents + k) * obsSize, obsSize);
                System.arraycopy(s.next()[k], 0, n, (i * agents + k) * obsSize, obsSize);
                hot[(i * agents + k) * actions + s.actions()[k]] = 1;
            }
            System.arraycopy(s.state(), 0, st, i * stateSize, stateSize);
            System.arraycopy(s.nextState(), 0, nst, i * stateSize, stateSize);
            r[i] = s.reward();
            notDone[i] = s.terminated() ? 0 : 1;
        }
        Tensor stateT = Tensor.of(Shape.of(b, stateSize), st);
        Tensor nextStateT = Tensor.of(Shape.of(b, stateSize), nst);
        Tensor nextObs = Tensor.of(Shape.of(b * agents, obsSize), n);
        Tensor nextOnline = q.forward(Var.of(nextObs), false).value().reshape(b, agents, actions);
        Tensor nextGreedy = greedyHot(nextOnline, b);
        Var nextUtil = Var.of(qTarget.forward(Var.of(nextObs), false).value().reshape(b, agents, actions));
        Tensor nextTot = mixerTarget.mix(nextUtil, nextGreedy, nextStateT).value();
        Tensor y = Tensor.of(Shape.of(b, 1), r).add(Tensor.of(Shape.of(b, 1), notDone).mul(gamma).mul(nextTot));
        Var util = q.forward(Var.of(Tensor.of(Shape.of(b * agents, obsSize), o)), true).reshape(b, agents, actions);
        Tensor chosen = Tensor.of(Shape.of(b, agents, actions), hot);
        Var loss = mixer.mix(util, chosen, stateT).sub(y).square().mean();
        Var aux = mixer.auxiliaryLoss(util, chosen, greedyHot(util.value(), b), stateT);
        if (aux != null) {
            loss = loss.add(aux);
        }
        loss.backward();
        List<Var> params = new ArrayList<>(q.parameters());
        params.addAll(mixer.parameters());
        RlTensors.clip(params, 10f);
        optimizer.step(params);
        if (++updates % targetEvery == 0) {
            RlTensors.copy(q.parameters(), qTarget.parameters());
            RlTensors.copy(mixer.parameters(), mixerTarget.parameters());
        }
    }

    /**
     * Plays one greedy episode.
     *
     * @param env the environment
     * @param seed the episode seed
     * @return the team return
     */
    public double evaluate(MultiAgentEnvironment env, long seed) {
        float[][] obs = env.reset(new SplittableRandom(seed));
        double total = 0;
        MultiAgentEnvironment.Step s;
        do {
            s = env.step(act(obs, false));
            total += s.reward();
            obs = s.observations();
        } while (!s.done());
        return total;
    }

    @Override
    public String toString() {
        return method.toString();
    }
}
