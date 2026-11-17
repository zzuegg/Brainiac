package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
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
 * Generative adversarial imitation learning, a compact version (Ho and Ermon, 2016,
 * <a href="https://arxiv.org/abs/1606.03476">arXiv:1606.03476</a>) for discrete actions.
 *
 * <p>A discriminator {@code D(s, a)} learns to tell expert state-action pairs from the
 * policy's; the policy is trained by policy gradient (advantage actor-critic with a value
 * baseline and an entropy bonus) on the surrogate reward {@code -log(1 - D(s, a))},
 * ignoring the environment's reward entirely. At the equilibrium the policy's occupancy
 * matches the expert's: imitation that, unlike behaviour cloning, corrects its own
 * compounding errors because it practises in the environment.
 */
public final class Gail implements Learner {

    private final OfflineDataset expert;
    private final int obsDims;
    private final int actions;
    private final Model policy;
    private final Model value;
    private final Model discriminator;
    private Optimizer piOpt = Optimizer.adam(3e-3f);
    private Optimizer vOpt = Optimizer.adam(3e-3f);
    private Optimizer dOpt = Optimizer.adam(1e-3f);
    private int episodesPerIteration = 8;
    private int discriminatorSteps = 2;
    private float gamma = 0.95f;
    private float entropy = 0.01f;
    private SplittableRandom random = new SplittableRandom(0);

    private Gail(OfflineDataset expert, int obsDims, int actions, RlNetworkFactory nets) {
        this.expert = expert;
        this.obsDims = obsDims;
        this.actions = actions;
        policy = nets.create(obsDims, actions, 51);
        value = nets.create(obsDims, 1, 52);
        discriminator = nets.create(obsDims + actions, 1, 53);
    }

    /**
     * @param expert expert demonstrations
     * @param observationDims values per observation
     * @param actions discrete actions
     * @param networks builds policy, value and discriminator networks
     * @return the learner
     */
    public static Gail of(OfflineDataset expert, int observationDims, int actions, RlNetworkFactory networks) {
        return new Gail(Objects.requireNonNull(expert), observationDims, actions, Objects.requireNonNull(networks));
    }

    /** @param n policy episodes per iteration @return this learner */
    public Gail episodesPerIteration(int n) {
        this.episodesPerIteration = n;
        return this;
    }

    /** @param discount gamma for the surrogate reward @return this learner */
    public Gail discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param seed the seed @return this learner */
    public Gail seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    private float[] probabilities(float[] obs) {
        float[] l = RlTensors.row(policy, obs);
        return RlTensors.softmax(Var.of(Tensor.of(l))).value().toArray();
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        float[] p = probabilities(observation);
        if (!explore) {
            return new float[] {RlTensors.argmax(p)};
        }
        double u = random.nextDouble();
        for (int a = 0; a < p.length; a++) {
            u -= p[a];
            if (u <= 0) {
                return new float[] {a};
            }
        }
        return new float[] {p.length - 1};
    }

    private Tensor pairs(List<float[]> obs, List<Integer> acts) {
        int n = obs.size();
        float[] x = new float[n * (obsDims + actions)];
        for (int i = 0; i < n; i++) {
            System.arraycopy(obs.get(i), 0, x, i * (obsDims + actions), obsDims);
            x[i * (obsDims + actions) + obsDims + acts.get(i)] = 1;
        }
        return Tensor.of(Shape.of(n, obsDims + actions), x);
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Discrete d) || d.n() != actions) {
            throw new IllegalArgumentException("GAIL needs " + actions + " discrete actions");
        }
        List<OfflineDataset.Step> expertSteps = expert.steps();
        Tracker recorder = new Tracker(listener);
        while (recorder.steps() < steps) {
            List<float[]> obs = new ArrayList<>();
            List<Integer> acts = new ArrayList<>();
            List<Integer> episodeEnds = new ArrayList<>();
            for (int e = 0; e < episodesPerIteration && recorder.steps() < steps; e++) {
                float[] o = env.reset(random.split());
                Environment.Step s;
                do {
                    int a = Math.round(act(o, true)[0]);
                    s = env.step(a);
                    obs.add(o);
                    acts.add(a);
                    o = s.observation();
                } while (!recorder.step(s));
                episodeEnds.add(obs.size());
            }
            int n = obs.size();
            Tensor policyPairs = pairs(obs, acts);
            for (int k = 0; k < discriminatorSteps; k++) {
                List<float[]> eo = new ArrayList<>();
                List<Integer> ea = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    OfflineDataset.Step st = expertSteps.get(random.nextInt(expertSteps.size()));
                    eo.add(st.observation());
                    ea.add(Math.round(st.action()[0]));
                }
                Var le = discriminator.forward(Var.of(pairs(eo, ea)), true);
                Var lp = discriminator.forward(Var.of(policyPairs), true);
                // binary cross-entropy: expert -> 1, policy -> 0
                Var loss = le.neg().softplus().mean().add(lp.softplus().mean());
                loss.backward();
                dOpt.step(discriminator.parameters());
            }
            // log D - log(1 - D), the discriminator logit: unlike -log(1 - D) it is not always
            // positive, so it carries no bias towards long (or short) episodes
            float[] reward = discriminator.forward(Var.of(policyPairs), false).value().toArray();
            float[] returns = new float[n];
            int start = 0;
            for (int end : episodeEnds) {
                float g = 0;
                for (int i = end - 1; i >= start; i--) {
                    g = reward[i] + gamma * g;
                    returns[i] = g;
                }
                start = end;
            }
            Tensor x = RlTensors.rows(obs.toArray(float[][]::new), obsDims, policy.device());
            Tensor ret = Tensor.of(Shape.of(n, 1), returns);
            Var v = value.forward(Var.of(x), true);
            Tensor adv = ret.sub(v.value());
            Tensor advNorm = adv.sub(adv.mean()).div(adv.square().mean().sqrt().add(1e-6f).item());
            Var vLoss = v.sub(ret).square().mean();
            vLoss.backward();
            vOpt.step(value.parameters());
            Var logp = RlTensors.logSoftmax(policy.forward(Var.of(x), true));
            Var chosen = logp.mul(RlTensors.oneHot(toFloats(acts), actions, policy.device())).sum(1, true);
            Var ent = logp.exp().mul(logp).sum(1, true).mean();
            Var piLoss = chosen.mul(advNorm).mean().neg().add(ent.mul(entropy));
            piLoss.backward();
            piOpt.step(policy.parameters());
        }
        return recorder.result();
    }

    private static float[] toFloats(List<Integer> values) {
        float[] f = new float[values.size()];
        for (int i = 0; i < f.length; i++) {
            f[i] = values.get(i);
        }
        return f;
    }
}
