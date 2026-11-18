package io.github.zzuegg.brainiac.experimental.explore;

import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Rl;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Predicate;
import java.util.function.Supplier;

/**
 * Interactive imitation learning for discrete actions, with a softmax-regression learner:
 *
 * <ul>
 *   <li>{@link Method#BEHAVIOR_CLONING}: expert rollouts only (DAgger with {@code beta = 1} in
 *       every round), the baseline at the same number of expert labels;</li>
 *   <li>{@link Method#DAGGER}: DAgger (Ross, Gordon &amp; Bagnell, AISTATS 2011,
 *       <a href="https://arxiv.org/abs/1011.0686">arXiv:1011.0686</a>): round {@code i} rolls out
 *       the mixture {@code beta_i expert + (1 - beta_i) learner} with {@code beta_i = p^i}, asks the
 *       expert to label every visited state, aggregates and retrains, so the learner sees the
 *       states its own mistakes lead to;</li>
 *   <li>{@link Method#HG_DAGGER}: HG-DAgger (Kelly et al., ICRA 2019,
 *       <a href="https://arxiv.org/abs/1810.02890">arXiv:1810.02890</a>): the learner drives and
 *       the expert takes over while an intervention rule flags the state as risky; only the
 *       expert-controlled states are labelled, and each take-over counts as an intervention;</li>
 *   <li>{@link Method#SAFE_DAGGER}: SafeDAgger-style learned gating (Zhang &amp; Cho, AAAI 2017):
 *       a logistic gate trained to predict where the learner disagrees with the expert decides
 *       when to query (and hand control to) the expert.</li>
 * </ul>
 *
 * <p>The aggregated dataset is a plain list of (observation, expert action) pairs; core
 * {@code ReplayBuffer.relabel} had not landed when this was written.
 */
public final class InteractiveImitation {

    /** The imitation scheme. */
    public enum Method {
        /** behaviour cloning on expert rollouts */
        BEHAVIOR_CLONING,
        /** dataset aggregation with a decaying expert mixture */
        DAGGER,
        /** human-gated DAgger with an intervention rule */
        HG_DAGGER,
        /** DAgger with a learned query gate */
        SAFE_DAGGER
    }

    /**
     * The outcome of a run.
     *
     * @param labels expert labels used per round (cumulative)
     * @param returns the learner's evaluation return after each round
     * @param interventions expert take-overs (HG-DAgger, SafeDAgger), 0 otherwise
     */
    public record Result(int[] labels, double[] returns, int interventions) {
        /** @return the final evaluation return */
        public double finalReturn() {
            return returns[returns.length - 1];
        }
    }

    private final Method method;
    private final int obsDims;
    private final int actions;
    private final SplittableRandom random;
    private int rounds = 5;
    private double betaDecay = 0.5;
    private int evalEpisodes = 10;
    private final List<float[]> xs = new ArrayList<>();
    private final List<Integer> ys = new ArrayList<>();
    private final SoftmaxPolicy policy;
    private final SoftmaxPolicy gate;

    private InteractiveImitation(Method method, int obsDims, int actions, long seed) {
        this.method = Objects.requireNonNull(method, "method");
        this.obsDims = obsDims;
        this.actions = actions;
        this.random = new SplittableRandom(seed);
        this.policy = new SoftmaxPolicy(obsDims, actions);
        this.gate = new SoftmaxPolicy(obsDims, 2);
    }

    /**
     * @param method the scheme
     * @param obsDims observation values
     * @param actions discrete actions
     * @param seed the seed
     * @return the learner
     */
    public static InteractiveImitation of(Method method, int obsDims, int actions, long seed) {
        return new InteractiveImitation(method, obsDims, actions, seed);
    }

    /** @param n data-collection rounds @return this learner */
    public InteractiveImitation rounds(int n) {
        this.rounds = n;
        return this;
    }

    /** @param p DAgger's {@code beta_i = p^i} @return this learner */
    public InteractiveImitation betaDecay(double p) {
        this.betaDecay = p;
        return this;
    }

    /** @param n evaluation episodes per round @return this learner */
    public InteractiveImitation evalEpisodes(int n) {
        this.evalEpisodes = n;
        return this;
    }

    /** @return the learner's greedy policy */
    public Agent policy() {
        return (obs, explore) -> new float[] {policy.argmax(obs)};
    }

    /**
     * Runs the rounds.
     *
     * @param environments fresh environments
     * @param expert the expert (queried greedily)
     * @param risky the HG-DAgger intervention rule (ignored by the other methods)
     * @param labelBudget total expert labels over all rounds
     * @return the result
     */
    public Result run(Supplier<? extends Environment> environments, Agent expert, Predicate<float[]> risky,
            int labelBudget) {
        int perRound = Math.max(1, labelBudget / rounds);
        int[] labels = new int[rounds];
        double[] returns = new double[rounds];
        int interventions = 0;
        List<float[]> gateX = new ArrayList<>();
        List<Integer> gateY = new ArrayList<>();
        for (int round = 0; round < rounds; round++) {
            Environment env = environments.get();
            int got = 0;
            double beta = method == Method.BEHAVIOR_CLONING || round == 0 ? 1 : Math.pow(betaDecay, round);
            boolean learnerDrives = round > 0 && (method == Method.HG_DAGGER || method == Method.SAFE_DAGGER);
            int guard = 0;
            while (got < perRound && guard < 20 * perRound) {
                float[] obs = env.reset(random.split());
                boolean expertInControl = false;
                Environment.Step st;
                do {
                    int expertAction = Math.round(expert.act(obs, false)[0]);
                    int act;
                    if (!learnerDrives) {
                        xs.add(obs.clone());
                        ys.add(expertAction);
                        got++;
                        act = random.nextDouble() < beta ? expertAction : policy.argmax(obs);
                    } else {
                        boolean query = method == Method.HG_DAGGER ? risky.test(obs) : gate.argmax(obs) == 1;
                        if (query) {
                            if (!expertInControl) {
                                interventions++;
                            }
                            expertInControl = true;
                            xs.add(obs.clone());
                            ys.add(expertAction);
                            got++;
                            act = expertAction;
                        } else {
                            expertInControl = false;
                            act = policy.argmax(obs);
                        }
                    }
                    if (method == Method.SAFE_DAGGER) {
                        gateX.add(obs.clone());
                        gateY.add(policy.argmax(obs) == expertAction ? 0 : 1);
                    }
                    st = env.step(act);
                    obs = st.observation();
                    guard++;
                } while (!st.done() && got < perRound && guard < 20 * perRound);
            }
            policy.fit(xs, ys, 300, random);
            if (method == Method.SAFE_DAGGER && !gateX.isEmpty()) {
                // the gate learns where the learner erred, on everything seen so far
                gate.fit(gateX, gateY, 200, random);
            }
            labels[round] = xs.size();
            returns[round] = Rl.evaluate(policy(), environments.get(), evalEpisodes, 1000L + round);
        }
        return new Result(labels, returns, interventions);
    }

    /** A multinomial logistic regression on standardised observations plus a bias. */
    static final class SoftmaxPolicy {
        private final int dims;
        private final int classes;
        private final double[][] w;
        private final double[] mean;
        private final double[] scale;

        SoftmaxPolicy(int dims, int classes) {
            this.dims = dims;
            this.classes = classes;
            this.w = new double[classes][dims + 1];
            this.mean = new double[dims];
            this.scale = new double[dims];
            java.util.Arrays.fill(scale, 1);
        }

        private double[] logits(float[] x) {
            double[] z = new double[classes];
            for (int c = 0; c < classes; c++) {
                double s = w[c][dims];
                for (int j = 0; j < dims; j++) {
                    s += w[c][j] * (x[j] - mean[j]) / scale[j];
                }
                z[c] = s;
            }
            return z;
        }

        int argmax(float[] x) {
            double[] z = logits(x);
            int best = 0;
            for (int c = 1; c < classes; c++) {
                if (z[c] > z[best]) {
                    best = c;
                }
            }
            return best;
        }

        void fit(List<float[]> x, List<Integer> y, int iterations, SplittableRandom random) {
            int n = x.size();
            if (n == 0) {
                return;
            }
            for (int j = 0; j < dims; j++) {
                double m = 0;
                double v = 0;
                for (float[] row : x) {
                    m += row[j];
                }
                m /= n;
                for (float[] row : x) {
                    v += (row[j] - m) * (row[j] - m);
                }
                mean[j] = m;
                scale[j] = Math.sqrt(v / n) + 1e-6;
            }
            for (double[] row : w) {
                java.util.Arrays.fill(row, 0);
            }
            double lr = 0.5;
            double[][] g = new double[classes][dims + 1];
            for (int it = 0; it < iterations; it++) {
                for (double[] row : g) {
                    java.util.Arrays.fill(row, 0);
                }
                for (int i = 0; i < n; i++) {
                    float[] xi = x.get(i);
                    double[] z = logits(xi);
                    double max = Double.NEGATIVE_INFINITY;
                    for (double v : z) {
                        max = Math.max(max, v);
                    }
                    double sum = 0;
                    for (int c = 0; c < classes; c++) {
                        z[c] = Math.exp(z[c] - max);
                        sum += z[c];
                    }
                    int yi = y.get(i);
                    for (int c = 0; c < classes; c++) {
                        double err = z[c] / sum - (c == yi ? 1 : 0);
                        for (int j = 0; j < dims; j++) {
                            g[c][j] += err * (xi[j] - mean[j]) / scale[j];
                        }
                        g[c][dims] += err;
                    }
                }
                for (int c = 0; c < classes; c++) {
                    for (int j = 0; j <= dims; j++) {
                        w[c][j] -= lr * (g[c][j] / n + (j < dims ? 1e-3 * w[c][j] : 0));
                    }
                }
            }
        }
    }
}
