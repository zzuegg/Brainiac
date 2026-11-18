package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GradientPenalty;
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
 * Three imitation recipes that fix the usual failure modes of plain {@link Gail}, for discrete
 * actions.
 * <ul>
 *   <li>{@link Method#GAIL}: GAIL with the two fixes of Discriminator-Actor-Critic (Kostrikov
 *       et al., <a href="https://arxiv.org/abs/1809.02925">arXiv:1809.02925</a>): a gradient
 *       penalty (weight 10, target norm 1 on expert/policy interpolates) that stops the
 *       discriminator from saturating, and an absorbing state after each true termination, so
 *       the learned reward of "having ended" is learned too instead of being fixed at 0 (which
 *       biases the policy towards long or short episodes, depending on the reward's sign).</li>
 *   <li>{@link Method#AIRL}: adversarial inverse RL (Fu et al.,
 *       <a href="https://arxiv.org/abs/1710.11248">arXiv:1710.11248</a>). The discriminator is
 *       {@code D = exp(f) / (exp(f) + pi(a|s))} with {@code f = g(s) + gamma h(s') - h(s)}, so its
 *       logit is {@code f - log pi(a|s)}, which is the policy's reward. The state-only term
 *       {@code g} is the recovered reward ({@link #reward}); {@code h} is a shaping potential, and
 *       {@code g} transfers to changed dynamics.</li>
 *   <li>{@link Method#IQ_LEARN}: inverse soft-Q learning (Garg et al.,
 *       <a href="https://arxiv.org/abs/2106.12142">arXiv:2106.12142</a>) without an adversary: a
 *       single soft Q-function, policy {@code softmax(Q / temperature)}, trained on
 *       {@code -E_expert[Q(s,a) - gamma V(s')] + E_mix[V(s) - gamma V(s')]
 *       + E_mix[(Q(s,a) - gamma V(s'))^2] / (4 chi)}: the "value" form of the initial-state term
 *       and the chi-squared regulariser of the paper, with expert and policy samples mixed
 *       online. The implied reward is {@code Q(s, a) - gamma V(s')}.</li>
 * </ul>
 * GAIL and AIRL train the policy like {@link Gail} (advantage actor-critic with a value
 * baseline and an entropy bonus); truncated episodes bootstrap from the value. Observations
 * get one extra input, the absorbing-state flag.
 */
public final class AdversarialImitation implements Learner {

    /** The imitation recipe. */
    public enum Method {
        /** GAIL with gradient penalty and absorbing states (DAC) */
        GAIL,
        /** AIRL's shaped discriminator */
        AIRL,
        /** IQ-Learn's inverse soft-Q objective */
        IQ_LEARN
    }

    private record Sample(float[] s, int a, float[] next, boolean terminal) {
    }

    private final Method method;
    private final List<OfflineDataset.Step> expert;
    private final int obsDims;
    private final int actions;
    private final Model policy;
    private final Model value;
    private final Model discriminator;
    private final Model g;
    private final Model h;
    private final Model q;
    private final Optimizer piOpt = Optimizer.adam(3e-3f);
    private final Optimizer vOpt = Optimizer.adam(3e-3f);
    private final Optimizer dOpt = Optimizer.adam(1e-3f);
    private final Optimizer qOpt = Optimizer.adam(1e-3f);
    private int episodesPerIteration = 8;
    private int discriminatorSteps = 2;
    private int qUpdates = 20;
    private int batchSize = 64;
    private float gamma = 0.95f;
    private float entropy = 0.01f;
    private float penalty = 10;
    private float temperature = 0.1f;
    private float chi = 0.5f;
    private boolean absorbing = true;
    private SplittableRandom random = new SplittableRandom(0);
    private final List<Sample> policySamples = new ArrayList<>();

    private AdversarialImitation(Method method, OfflineDataset expert, int obsDims, int actions,
            RlNetworkFactory nets) {
        this.method = method;
        this.expert = List.copyOf(expert.steps());
        this.obsDims = obsDims;
        this.actions = actions;
        int in = obsDims + 1;
        this.policy = nets.create(in, actions, 61);
        this.value = nets.create(in, 1, 62);
        this.discriminator = nets.create(in + actions, 1, 63);
        this.g = nets.create(in, 1, 64);
        this.h = nets.create(in, 1, 65);
        this.q = nets.create(in, actions, 66);
    }

    /**
     * @param method the recipe
     * @param expert expert demonstrations (action indices)
     * @param observationDims values per observation
     * @param actions discrete actions
     * @param networks builds every network
     * @return the learner
     */
    public static AdversarialImitation of(Method method, OfflineDataset expert, int observationDims, int actions,
            RlNetworkFactory networks) {
        return new AdversarialImitation(Objects.requireNonNull(method, "method"), Objects.requireNonNull(expert),
                observationDims, actions, Objects.requireNonNull(networks, "networks"));
    }

    /** @param weight the gradient-penalty weight of GAIL's discriminator (10; 0 is off) @return this learner */
    public AdversarialImitation gradientPenalty(float weight) {
        this.penalty = weight;
        return this;
    }

    /** @param on whether terminations lead to a learned absorbing state (default on) @return this learner */
    public AdversarialImitation absorbingStates(boolean on) {
        this.absorbing = on;
        return this;
    }

    /** @param discount gamma @return this learner */
    public AdversarialImitation discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param n policy episodes per iteration @return this learner */
    public AdversarialImitation episodesPerIteration(int n) {
        this.episodesPerIteration = n;
        return this;
    }

    /** @param t IQ-Learn's soft-value temperature (0.1) @return this learner */
    public AdversarialImitation temperature(float t) {
        this.temperature = t;
        return this;
    }

    /** @param n IQ-Learn gradient steps per iteration (20) @return this learner */
    public AdversarialImitation qUpdates(int n) {
        this.qUpdates = n;
        return this;
    }

    /** @param seed the seed @return this learner */
    public AdversarialImitation seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return the recipe */
    public Method method() {
        return method;
    }

    private float[] aug(float[] obs) {
        float[] x = new float[obsDims + 1];
        System.arraycopy(obs, 0, x, 0, obsDims);
        return x;
    }

    private float[] absorbingState() {
        float[] x = new float[obsDims + 1];
        x[obsDims] = 1;
        return x;
    }

    private Tensor states(List<float[]> s) {
        return RlTensors.rows(s.toArray(float[][]::new), obsDims + 1, policy.device());
    }

    private Tensor oneHot(List<Sample> b) {
        float[] a = new float[b.size()];
        for (int i = 0; i < a.length; i++) {
            a[i] = b.get(i).a();
        }
        return RlTensors.oneHot(a, actions, policy.device());
    }

    private float[] logits(float[] augmented) {
        Model m = method == Method.IQ_LEARN ? q : policy;
        float[] l = RlTensors.row(m, augmented);
        if (method == Method.IQ_LEARN) {
            for (int i = 0; i < l.length; i++) {
                l[i] /= temperature;
            }
        }
        return l;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        float[] l = logits(aug(observation));
        if (!explore) {
            return new float[] {RlTensors.argmax(l)};
        }
        float[] p = RlTensors.softmax(Var.of(Tensor.of(l))).value().toArray();
        double u = random.nextDouble();
        for (int a = 0; a < p.length; a++) {
            u -= p[a];
            if (u <= 0) {
                return new float[] {a};
            }
        }
        return new float[] {p.length - 1};
    }

    private List<Sample> expertSamples() {
        List<Sample> out = new ArrayList<>();
        for (OfflineDataset.Step st : expert) {
            boolean end = st.terminated();
            out.add(new Sample(aug(st.observation()), Math.round(st.action()[0]),
                    end && absorbing ? absorbingState() : aug(st.nextObservation()), end));
            if (end && absorbing) {
                out.add(new Sample(absorbingState(), 0, absorbingState(), false));
            }
        }
        return out;
    }

    private List<Sample> sample(List<Sample> pool, int n) {
        List<Sample> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            out.add(pool.get(random.nextInt(pool.size())));
        }
        return out;
    }

    private Tensor pairs(List<Sample> b) {
        List<float[]> s = new ArrayList<>();
        for (Sample x : b) {
            s.add(x.s());
        }
        return Tensor.concat(1, states(s), oneHot(b));
    }

    private Tensor notDone(List<Sample> b) {
        float[] m = new float[b.size()];
        for (int i = 0; i < m.length; i++) {
            m[i] = b.get(i).terminal() && !absorbing ? 0 : 1;
        }
        return RlTensors.column(m, policy.device());
    }

    private Tensor logPi(List<Sample> b) {
        List<float[]> s = new ArrayList<>();
        for (Sample x : b) {
            s.add(x.s());
        }
        Tensor lp = Var.noGrad(() -> RlTensors.logSoftmax(policy.forward(Var.of(states(s)), false))).value();
        return lp.mul(oneHot(b)).sum(1, true);
    }

    /** AIRL's f(s, a, s') = g(s) + gamma h(s') - h(s). */
    private Var shaped(List<Sample> b, boolean training) {
        List<float[]> s = new ArrayList<>();
        List<float[]> n = new ArrayList<>();
        for (Sample x : b) {
            s.add(x.s());
            n.add(x.next());
        }
        Var hs = h.forward(Var.of(states(s)), training);
        Var hn = h.forward(Var.of(states(n)), training).mul(Var.of(notDone(b)));
        return g.forward(Var.of(states(s)), training).add(hn.mul(gamma)).sub(hs);
    }

    private Var softV(Var qValues) {
        // temperature * logsumexp(Q / temperature), stabilised by the row maximum
        Tensor max = qValues.value().div(temperature).max(1, true);
        return qValues.div(temperature).sub(Var.of(max)).exp().sum(1, true).log().add(Var.of(max)).mul(temperature);
    }

    private void trainDiscriminator(List<Sample> fresh, List<Sample> expertPool) {
        for (int k = 0; k < discriminatorSteps; k++) {
            List<Sample> e = sample(expertPool, fresh.size());
            Var loss;
            if (method == Method.GAIL) {
                Tensor ep = pairs(e);
                Tensor pp = pairs(fresh);
                Var le = discriminator.forward(Var.of(ep), true);
                Var lp = discriminator.forward(Var.of(pp), true);
                loss = le.neg().softplus().mean().add(lp.softplus().mean());
                if (penalty > 0) {
                    loss = loss.add(GradientPenalty.wgan(x -> discriminator.forward(x, true), ep, pp, random)
                            .mul(penalty));
                }
                loss.backward();
                dOpt.step(discriminator.parameters());
            } else {
                Var le = shaped(e, true).sub(Var.of(logPi(e)));
                Var lp = shaped(fresh, true).sub(Var.of(logPi(fresh)));
                loss = le.neg().softplus().mean().add(lp.softplus().mean());
                loss.backward();
                List<Var> params = new ArrayList<>(g.parameters());
                params.addAll(h.parameters());
                dOpt.step(params);
            }
        }
    }

    private float[] surrogate(List<Sample> b) {
        if (method == Method.GAIL) {
            return Var.noGrad(() -> discriminator.forward(Var.of(pairs(b)), false)).value().toArray();
        }
        Tensor f = Var.noGrad(() -> shaped(b, false)).value();
        return f.sub(logPi(b)).toArray();
    }

    private void trainQ(List<Sample> expertPool) {
        for (int k = 0; k < qUpdates; k++) {
            List<Sample> e = sample(expertPool, batchSize);
            List<Sample> mix = new ArrayList<>(e);
            if (!policySamples.isEmpty()) {
                mix.addAll(sample(policySamples, batchSize));
            }
            List<float[]> s = new ArrayList<>();
            List<float[]> n = new ArrayList<>();
            for (Sample x : mix) {
                s.add(x.s());
                n.add(x.next());
            }
            Var qs = q.forward(Var.of(states(s)), true);
            Var qa = qs.mul(Var.of(oneHot(mix))).sum(1, true);
            Var vs = softV(qs);
            Var vn = softV(q.forward(Var.of(states(n)), true)).mul(Var.of(notDone(mix)));
            Var r = qa.sub(vn.mul(gamma));
            int ne = e.size();
            Var expertR = r.slice(0, 0, ne);
            Var loss = expertR.mean().neg().add(vs.sub(vn.mul(gamma)).mean()).add(r.square().mean().div(4 * chi));
            loss.backward();
            qOpt.step(q.parameters());
        }
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Discrete d) || d.n() != actions) {
            throw new IllegalArgumentException("needs " + actions + " discrete actions");
        }
        List<Sample> expertPool = expertSamples();
        Tracker recorder = new Tracker(listener);
        while (recorder.steps() < steps) {
            List<Sample> fresh = new ArrayList<>();
            List<Integer> ends = new ArrayList<>();
            List<Boolean> terminated = new ArrayList<>();
            for (int e = 0; e < episodesPerIteration && recorder.steps() < steps; e++) {
                float[] o = env.reset(random.split());
                Environment.Step s;
                boolean over;
                do {
                    int a = Math.round(act(o, true)[0]);
                    s = env.step(a);
                    fresh.add(new Sample(aug(o), a, s.terminated() && absorbing ? absorbingState()
                            : aug(s.observation()), s.terminated()));
                    o = s.observation();
                    over = recorder.step(s);
                } while (!over);
                ends.add(fresh.size());
                terminated.add(s.terminated());
            }
            int real = fresh.size();
            List<Sample> withAbsorbing = new ArrayList<>(fresh);
            if (absorbing) {
                for (boolean t : terminated) {
                    if (t) {
                        withAbsorbing.add(new Sample(absorbingState(), 0, absorbingState(), false));
                    }
                }
            }
            if (method == Method.IQ_LEARN) {
                policySamples.addAll(withAbsorbing);
                while (policySamples.size() > 20_000) {
                    policySamples.remove(0);
                }
                trainQ(expertPool);
                continue;
            }
            trainDiscriminator(withAbsorbing, expertPool);
            float[] reward = surrogate(fresh);
            float rAbs = absorbing ? surrogate(List.of(new Sample(absorbingState(), 0, absorbingState(), false)))[0]
                    : 0;
            List<float[]> obs = new ArrayList<>();
            for (Sample x : fresh) {
                obs.add(x.s());
            }
            Tensor x = states(obs);
            float[] returns = new float[real];
            int start = 0;
            for (int k = 0; k < ends.size(); k++) {
                int end = ends.get(k);
                float ret;
                if (terminated.get(k)) {
                    ret = absorbing ? rAbs / (1 - gamma) : 0;
                } else {
                    float[] last = fresh.get(end - 1).next();
                    ret = RlTensors.row(value, last)[0];
                }
                for (int i = end - 1; i >= start; i--) {
                    ret = reward[i] + gamma * ret;
                    returns[i] = ret;
                }
                start = end;
            }
            Tensor ret = Tensor.of(Shape.of(real, 1), returns);
            Var v = value.forward(Var.of(x), true);
            Tensor adv = ret.sub(v.value());
            Tensor advNorm = adv.sub(adv.mean()).div(adv.sub(adv.mean()).square().mean().sqrt().add(1e-6f).item());
            v.sub(ret).square().mean().backward();
            vOpt.step(value.parameters());
            Var logp = RlTensors.logSoftmax(policy.forward(Var.of(x), true));
            Var chosen = logp.mul(Var.of(oneHot(fresh))).sum(1, true);
            Var ent = logp.exp().mul(logp).sum(1, true).mean();
            chosen.mul(Var.of(advNorm)).mean().neg().add(ent.mul(entropy)).backward();
            piOpt.step(policy.parameters());
        }
        return recorder.result();
    }

    /**
     * The recovered reward of one transition: GAIL's discriminator logit {@code log D - log(1 - D)}
     * of {@code (s, a)}, AIRL's state-only {@code g(s)}, or IQ-Learn's {@code Q(s, a) - gamma V(s')}.
     *
     * @param observation the state
     * @param action the action
     * @param next the next state
     * @param terminal whether the transition ended the episode
     * @return the reward
     */
    public double reward(float[] observation, int action, float[] next, boolean terminal) {
        Sample x = new Sample(aug(observation), action, terminal && absorbing ? absorbingState() : aug(next),
                terminal);
        return switch (method) {
            case GAIL -> surrogate(List.of(x))[0];
            case AIRL -> RlTensors.row(g, x.s())[0];
            case IQ_LEARN -> {
                List<Sample> b = List.of(x);
                Tensor qs = Var.noGrad(() -> q.forward(Var.of(states(List.of(x.s()))), false)).value();
                double qa = qs.mul(oneHot(b)).sum().item();
                double vn = Var.noGrad(() -> softV(q.forward(Var.of(states(List.of(x.next()))), false))).value()
                        .item();
                yield qa - gamma * vn * notDone(b).item();
            }
        };
    }

    /**
     * @return the recovered reward of the absorbing state's self-loop, received forever after a
     *         termination; 0 without absorbing states
     */
    public double absorbingReward() {
        if (!absorbing) {
            return 0;
        }
        return switch (method) {
            case GAIL -> surrogate(List.of(new Sample(absorbingState(), 0, absorbingState(), false)))[0];
            case AIRL -> RlTensors.row(g, absorbingState())[0];
            case IQ_LEARN -> {
                Tensor qs = Var.noGrad(() -> q.forward(Var.of(states(List.of(absorbingState()))), false)).value();
                double v = Var.noGrad(() -> softV(q.forward(Var.of(states(List.of(absorbingState()))), false)))
                        .value().item();
                yield qs.toArray()[0] - gamma * v;
            }
        };
    }

    @Override
    public String toString() {
        return switch (method) {
            case GAIL -> "GAIL" + (penalty > 0 ? "+GP" : "") + (absorbing ? "+absorbing" : "");
            case AIRL -> "AIRL";
            case IQ_LEARN -> "IQ-Learn";
        };
    }
}
