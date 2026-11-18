package io.github.zzuegg.brainiac.experimental.deeprl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tracker;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Model-based policy optimisation (Janner et al., NeurIPS 2019,
 * <a href="https://arxiv.org/abs/1906.08253">arXiv:1906.08253</a>): an ensemble of
 * probabilistic dynamics models, each predicting a Gaussian over the state change and the
 * reward, is trained on real transitions; short {@code k}-step rollouts branched from real
 * states with a random elite member at every step are added to the replay buffer of a
 * {@link ModernSac}, which trains at a high update-to-data ratio on the mix of real and model
 * data. Short branches keep model error from compounding while multiplying the data.
 *
 * <p>The model data goes into SAC's own buffer (its first-in first-out capacity bounds how
 * long model transitions are kept), and SAC is driven in chunks of whole episodes, so the chunk
 * length should be the task's episode length. The models predict no termination (suited to
 * the swing-up and mountain-car tasks, which only end by time limit).
 */
public final class Mbpo implements Learner {

    private final int obsDims;
    private final int actDims;
    private final ModernSac sac;
    private final List<Model> members = new ArrayList<>();
    private final List<Optimizer> optimizers = new ArrayList<>();
    private final List<Transition> real = new ArrayList<>();
    private final SplittableRandom random;
    private int elites = 3;
    private int[] eliteIndex;
    private int rolloutStart = 1;
    private int rolloutEnd = 1;
    private int rolloutsPerStep = 20;
    private int chunk = 200;
    private int modelSteps = 100;
    private float gamma = 0.99f;
    private double[] inMean;
    private double[] inStd;
    private double[] outMean;
    private double[] outStd;
    private long realSteps;
    private long plannedSteps = 1;
    private long modelTransitions;

    private Mbpo(int obsDims, int actDims, ModernSac sac, int ensemble, int hidden, long seed) {
        this.obsDims = obsDims;
        this.actDims = actDims;
        this.sac = sac;
        this.random = new SplittableRandom(seed);
        for (int m = 0; m < ensemble; m++) {
            members.add(Nets.mlp(obsDims + actDims, 2 * (obsDims + 1), Activation.RELU, seed + 13L * m, hidden,
                    hidden));
            optimizers.add(Optimizer.adam(1e-3f));
        }
    }

    /**
     * @param observationDims values per observation
     * @param actionDims values per action
     * @param actor SAC's actor factory
     * @param critic SAC's critic factory
     * @param ensemble dynamics models (7 in the paper)
     * @param hidden the models' hidden width
     * @param seed the seed
     * @return the learner
     */
    public static Mbpo of(int observationDims, int actionDims, RlNetworkFactory actor, RlNetworkFactory critic,
            int ensemble, int hidden, long seed) {
        ModernSac sac = ModernSac.of(observationDims, actionDims, actor, critic).seed(seed);
        return new Mbpo(observationDims, actionDims, sac, ensemble, hidden, seed);
    }

    /** @return the SAC learner, for its options (update-to-data ratio, learning rate and so on) */
    public ModernSac sac() {
        return sac;
    }

    /** @param n elite models used for rollouts @return this learner */
    public Mbpo elites(int n) {
        this.elites = n;
        return this;
    }

    /**
     * A linear rollout-length schedule over the training run.
     *
     * @param first the length at the start
     * @param last the length at the end
     * @return this learner
     */
    public Mbpo rolloutLength(int first, int last) {
        this.rolloutStart = first;
        this.rolloutEnd = last;
        return this;
    }

    /** @param m model rollouts branched per real step @return this learner */
    public Mbpo rolloutsPerStep(int m) {
        this.rolloutsPerStep = m;
        return this;
    }

    /** @param steps real steps between model refits and rollouts (an episode length) @return this learner */
    public Mbpo chunk(int steps) {
        this.chunk = steps;
        return this;
    }

    /** @param discount gamma (also set on SAC) @return this learner */
    public Mbpo discount(float discount) {
        this.gamma = discount;
        sac.discount(discount);
        return this;
    }

    /** @return SAC's gradient updates */
    public long updates() {
        return sac.updates();
    }

    /** @return model transitions added so far */
    public long modelTransitions() {
        return modelTransitions;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        return sac.act(observation, explore);
    }

    private final class Recorder implements Environment {
        private final Environment inner;
        private final Tracker tracker;
        private float[] last;

        Recorder(Environment inner, Tracker tracker) {
            this.inner = inner;
            this.tracker = tracker;
        }

        @Override
        public Space observationSpace() {
            return inner.observationSpace();
        }

        @Override
        public Space actionSpace() {
            return inner.actionSpace();
        }

        @Override
        public float[] reset(SplittableRandom r) {
            last = inner.reset(r);
            return last;
        }

        @Override
        public Step step(float[] action) {
            Step s = inner.step(action);
            real.add(new Transition(last, unit(action), s.reward(), s.observation(), s.terminated(), gamma));
            realSteps++;
            tracker.step(s);
            last = s.observation();
            return s;
        }
    }

    private Space.Box box;

    private float[] unit(float[] a) {
        float[] u = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            float lo = box.low()[i];
            float hi = box.high()[i];
            u[i] = Math.max(-1, Math.min(1, 2 * (a[i] - lo) / (hi - lo) - 1));
        }
        return u;
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        box = (Space.Box) env.actionSpace();
        Tracker tracker = new Tracker(listener);
        Recorder rec = new Recorder(env, tracker);
        plannedSteps = realSteps + steps;
        long done = 0;
        while (done < steps) {
            long n = Math.min(chunk, steps - done);
            sac.fit(rec, n);
            done += n;
            if (real.size() >= 200 && done < steps) {
                trainModels();
                rollouts((int) n);
            }
        }
        return tracker.result();
    }

    private void normalise() {
        int in = obsDims + actDims;
        int out = obsDims + 1;
        inMean = new double[in];
        inStd = new double[in];
        outMean = new double[out];
        outStd = new double[out];
        for (Transition t : real) {
            double[] x = input(t.observation(), t.action());
            double[] y = target(t);
            for (int i = 0; i < in; i++) {
                inMean[i] += x[i] / real.size();
                inStd[i] += x[i] * x[i] / real.size();
            }
            for (int i = 0; i < out; i++) {
                outMean[i] += y[i] / real.size();
                outStd[i] += y[i] * y[i] / real.size();
            }
        }
        for (int i = 0; i < in; i++) {
            inStd[i] = Math.sqrt(Math.max(1e-6, inStd[i] - inMean[i] * inMean[i]));
        }
        for (int i = 0; i < out; i++) {
            outStd[i] = Math.sqrt(Math.max(1e-6, outStd[i] - outMean[i] * outMean[i]));
        }
    }

    private double[] input(float[] obs, float[] action) {
        double[] x = new double[obsDims + actDims];
        for (int i = 0; i < obsDims; i++) {
            x[i] = obs[i];
        }
        for (int i = 0; i < actDims; i++) {
            x[obsDims + i] = action[i];
        }
        return x;
    }

    private double[] target(Transition t) {
        double[] y = new double[obsDims + 1];
        for (int i = 0; i < obsDims; i++) {
            y[i] = t.nextObservation()[i] - t.observation()[i];
        }
        y[obsDims] = t.reward();
        return y;
    }

    private Tensor inputs(List<float[]> obs, List<float[]> actions) {
        int w = obsDims + actDims;
        float[] flat = new float[obs.size() * w];
        for (int r = 0; r < obs.size(); r++) {
            double[] x = input(obs.get(r), actions.get(r));
            for (int i = 0; i < w; i++) {
                flat[r * w + i] = (float) ((x[i] - inMean[i]) / inStd[i]);
            }
        }
        return Tensor.of(Shape.of(obs.size(), w), flat);
    }

    /** The Gaussian negative log-likelihood of a member on normalised targets. */
    private Var nll(Model m, Tensor x, Tensor y) {
        int out = obsDims + 1;
        Var o = m.forward(Var.of(x), true);
        Var mean = o.slice(1, 0, out);
        Var logVar = o.slice(1, out, 2 * out).tanh().mul(4f);
        return mean.sub(y).square().mul(logVar.neg().exp()).add(logVar).mean();
    }

    private void trainModels() {
        normalise();
        int n = real.size();
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
        int holdout = Math.max(1, n / 10);
        int out = obsDims + 1;
        for (int step = 0; step < modelSteps; step++) {
            for (int m = 0; m < members.size(); m++) {
                List<float[]> o = new ArrayList<>();
                List<float[]> a = new ArrayList<>();
                float[] y = new float[64 * out];
                for (int b = 0; b < 64; b++) {
                    Transition t = real.get(order[holdout + random.nextInt(n - holdout)]);
                    o.add(t.observation());
                    a.add(t.action());
                    double[] tg = target(t);
                    for (int i = 0; i < out; i++) {
                        y[b * out + i] = (float) ((tg[i] - outMean[i]) / outStd[i]);
                    }
                }
                Var loss = nll(members.get(m), inputs(o, a), Tensor.of(Shape.of(64, out), y));
                loss.backward();
                optimizers.get(m).step(members.get(m).parameters());
            }
        }
        // elites by held-out mean squared error
        List<float[]> o = new ArrayList<>();
        List<float[]> a = new ArrayList<>();
        float[] y = new float[holdout * out];
        for (int b = 0; b < holdout; b++) {
            Transition t = real.get(order[b]);
            o.add(t.observation());
            a.add(t.action());
            double[] tg = target(t);
            for (int i = 0; i < out; i++) {
                y[b * out + i] = (float) ((tg[i] - outMean[i]) / outStd[i]);
            }
        }
        Tensor x = inputs(o, a);
        Tensor yt = Tensor.of(Shape.of(holdout, out), y);
        double[] mse = new double[members.size()];
        for (int m = 0; m < members.size(); m++) {
            Tensor pred = members.get(m).forward(Var.of(x), false).value().slice(1, 0, out);
            mse[m] = pred.sub(yt).square().mean().item();
        }
        Integer[] idx = new Integer[members.size()];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (p, q) -> Double.compare(mse[p], mse[q]));
        eliteIndex = new int[Math.min(elites, members.size())];
        for (int i = 0; i < eliteIndex.length; i++) {
            eliteIndex[i] = idx[i];
        }
    }

    /** @return the rollout length at the current point of the run */
    public int rolloutLength() {
        double frac = Math.min(1, realSteps / (double) plannedSteps);
        return (int) Math.round(rolloutStart + frac * (rolloutEnd - rolloutStart));
    }

    private void rollouts(int realChunk) {
        int count = rolloutsPerStep * realChunk;
        int k = Math.max(1, rolloutLength());
        int out = obsDims + 1;
        List<float[]> states = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            states.add(real.get(random.nextInt(real.size())).observation());
        }
        for (int step = 0; step < k; step++) {
            List<float[]> actions = new ArrayList<>();
            for (float[] s : states) {
                actions.add(unit(sac.act(s, true)));
            }
            Tensor x = inputs(states, actions);
            float[][] preds = new float[eliteIndex.length][];
            for (int e = 0; e < eliteIndex.length; e++) {
                preds[e] = members.get(eliteIndex[e]).forward(Var.of(x), false).value().toArray();
            }
            List<float[]> next = new ArrayList<>();
            for (int r = 0; r < states.size(); r++) {
                float[] p = preds[random.nextInt(preds.length)];
                float[] s = states.get(r);
                float[] ns = new float[obsDims];
                float reward = 0;
                for (int i = 0; i < out; i++) {
                    double mean = p[r * 2 * out + i];
                    double logVar = 4 * Math.tanh(p[r * 2 * out + out + i]);
                    double v = (mean + Math.exp(0.5 * logVar) * random.nextGaussian()) * outStd[i] + outMean[i];
                    if (i < obsDims) {
                        ns[i] = (float) (s[i] + v);
                    } else {
                        reward = (float) v;
                    }
                }
                sac.buffer().add(new Transition(s, actions.get(r), reward, ns, false, gamma));
                modelTransitions++;
                next.add(ns);
            }
            states = next;
        }
    }

    @Override
    public String toString() {
        return "MBPO";
    }
}
