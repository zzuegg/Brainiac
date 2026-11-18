package io.github.zzuegg.brainiac.experimental.explore;

import io.github.zzuegg.brainiac.experimental.bandit.BanditMath;
import io.github.zzuegg.brainiac.experimental.rl.IntrinsicReward;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tracker;

import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Provably efficient exploration for episodic tabular MDPs, as core {@link Learner}s on
 * environments with one-hot state observations (such as {@link DeepSeaChain} and
 * {@link io.github.zzuegg.brainiac.experimental.rl.TabularMdp}):
 *
 * <ul>
 *   <li>{@link Method#EPSILON_GREEDY_Q}: horizon-indexed Q-learning with epsilon-greedy dithering,
 *       optionally with an {@link IntrinsicReward} count bonus (the baselines);</li>
 *   <li>{@link Method#RMAX}: R-max (Brafman &amp; Tennenholtz, JMLR 2002): a pair visited fewer
 *       than {@code m} times is assumed to pay the maximum reward forever; the agent replans on
 *       the empirical model whenever a pair becomes known;</li>
 *   <li>{@link Method#UCRL2}: UCRL2 (Jaksch, Ortner &amp; Auer, JMLR 2010): extended value iteration
 *       in the most optimistic MDP within L1 confidence sets around the empirical transitions and
 *       Hoeffding sets around the rewards; it replans only when some pair's count has doubled;</li>
 *   <li>{@link Method#PSRL}: posterior sampling (Osband, Russo &amp; Van Roy, NeurIPS 2013,
 *       <a href="https://arxiv.org/abs/1306.0940">arXiv:1306.0940</a>): one MDP per episode from
 *       Dirichlet transition and normal-gamma reward posteriors, then greedy acting in it;</li>
 *   <li>{@link Method#UCB_Q_HOEFFDING} and {@link Method#UCB_Q_BERNSTEIN}: optimistic Q-learning
 *       (Jin et al., NeurIPS 2018, <a href="https://arxiv.org/abs/1807.03765">arXiv:1807.03765</a>)
 *       with the learning rate {@code (H + 1) / (H + t)} and a Hoeffding or an empirical-Bernstein
 *       bonus (scaled by {@link #bonusScale}; the theoretical constants are far too loose at
 *       this size).</li>
 * </ul>
 *
 * <p>All methods plan over the episode horizon {@code H} with step-indexed values (the step is
 * counted from the episode start). Terminal transitions go to an absorbing zero-value state.
 * Transition statistics are sparse, so Deep Sea sizes of 10-30 plan in milliseconds; PSRL draws
 * the Dirichlet mass of never-seen successors as one lump spread uniformly (exact aggregation
 * of the Dirichlet, uniform within the lump).
 */
public final class TabularExplorers implements Learner {

    /** The exploration strategy. */
    public enum Method {
        /** Q-learning with epsilon-greedy dithering */
        EPSILON_GREEDY_Q,
        /** R-max optimism for unknown pairs */
        RMAX,
        /** UCRL2 extended value iteration with doubling epochs */
        UCRL2,
        /** posterior sampling for RL */
        PSRL,
        /** optimistic Q-learning with a Hoeffding bonus */
        UCB_Q_HOEFFDING,
        /** optimistic Q-learning with an empirical Bernstein bonus */
        UCB_Q_BERNSTEIN
    }

    private final Method method;
    private final int states;
    private final int actions;
    private final int horizon;
    private SplittableRandom random;
    private double epsilon = 0.1;
    private double learningRate = 0.1;
    private int knownVisits = 3;
    private double rewardMax = 1;
    private double bonusScale = 1;
    private double delta = 0.1;
    private double dirichletPrior = -1;
    private IntrinsicReward intrinsic;
    private double intrinsicScale;

    // model statistics; next-state index states is the absorbing terminal
    private final int[] visits;
    private final int[] visitsAtPlan;
    private final double[] rewardSum;
    private final double[] rewardSq;
    private final int[][] successors;
    private final int[][] successorCounts;
    private final int[] successorSize;

    // horizon-indexed values
    private final double[][] q;
    private final int[][] stepVisits;
    private final double[][] vSum;
    private final double[][] vSq;
    private long totalSteps;
    private int step;
    private long plans;
    private boolean planDirty = true;

    private TabularExplorers(Method method, int states, int actions, int horizon, long seed) {
        this.method = Objects.requireNonNull(method, "method");
        this.states = states;
        this.actions = actions;
        this.horizon = horizon;
        this.random = new SplittableRandom(seed);
        int pairs = states * actions;
        visits = new int[pairs];
        visitsAtPlan = new int[pairs];
        rewardSum = new double[pairs];
        rewardSq = new double[pairs];
        successors = new int[pairs][];
        successorCounts = new int[pairs][];
        successorSize = new int[pairs];
        q = new double[horizon][pairs];
        stepVisits = new int[horizon][pairs];
        vSum = new double[horizon][pairs];
        vSq = new double[horizon][pairs];
        double init = switch (method) {
            case UCB_Q_HOEFFDING, UCB_Q_BERNSTEIN -> rewardMax * horizon;
            default -> 0;
        };
        for (double[] row : q) {
            Arrays.fill(row, init);
        }
    }

    /**
     * @param method the strategy
     * @param states the number of states (the observation length)
     * @param actions the number of discrete actions
     * @param horizon the episode length {@code H}
     * @param seed the seed
     * @return the learner
     */
    public static TabularExplorers of(Method method, int states, int actions, int horizon, long seed) {
        return new TabularExplorers(method, states, actions, horizon, seed);
    }

    /** @param eps the dithering rate of {@link Method#EPSILON_GREEDY_Q} @return this learner */
    public TabularExplorers epsilon(double eps) {
        this.epsilon = eps;
        return this;
    }

    /** @param lr the step size of {@link Method#EPSILON_GREEDY_Q} @return this learner */
    public TabularExplorers learningRate(double lr) {
        this.learningRate = lr;
        return this;
    }

    /** @param m visits after which R-max trusts a pair @return this learner */
    public TabularExplorers knownVisits(int m) {
        this.knownVisits = m;
        return this;
    }

    /** @param max the largest per-step reward (R-max optimism and value caps) @return this learner */
    public TabularExplorers rewardMax(double max) {
        this.rewardMax = max;
        if (method == Method.UCB_Q_HOEFFDING || method == Method.UCB_Q_BERNSTEIN) {
            for (double[] row : q) {
                Arrays.fill(row, max * horizon);
            }
        }
        return this;
    }

    /** @param c multiplies every confidence radius and bonus @return this learner */
    public TabularExplorers bonusScale(double c) {
        this.bonusScale = c;
        return this;
    }

    /** @param d the confidence parameter of the bonuses @return this learner */
    public TabularExplorers delta(double d) {
        this.delta = d;
        return this;
    }

    /**
     * @param alpha the Dirichlet prior count per successor of {@link Method#PSRL} (default
     *     {@code 1 / (states + 1)})
     * @return this learner
     */
    public TabularExplorers dirichletPrior(double alpha) {
        this.dirichletPrior = alpha;
        return this;
    }

    /**
     * Adds an intrinsic reward (for example a {@link CountBonuses} count bonus) to the rewards of
     * {@link Method#EPSILON_GREEDY_Q}.
     *
     * @param bonus the intrinsic reward
     * @param scale its weight
     * @return this learner
     */
    public TabularExplorers intrinsic(IntrinsicReward bonus, double scale) {
        this.intrinsic = bonus;
        this.intrinsicScale = scale;
        return this;
    }

    /** @return the number of times a model-based method planned */
    public long plans() {
        return plans;
    }

    /** @return the strategy */
    public Method method() {
        return method;
    }

    private int state(float[] observation) {
        int best = 0;
        for (int i = 1; i < observation.length; i++) {
            if (observation[i] > observation[best]) {
                best = i;
            }
        }
        return best;
    }

    private int greedy(int h, int s) {
        double[] row = q[Math.min(h, horizon - 1)];
        int best = 0;
        int ties = 1;
        for (int a = 1; a < actions; a++) {
            double v = row[s * actions + a];
            double b = row[s * actions + best];
            if (v > b + 1e-12) {
                best = a;
                ties = 1;
            } else if (Math.abs(v - b) <= 1e-12 && random.nextInt(++ties) == 0) {
                best = a;
            }
        }
        return best;
    }

    @Override
    public void startEpisode() {
        step = 0;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        int s = state(observation);
        int a = explore && method == Method.EPSILON_GREEDY_Q && random.nextDouble() < epsilon
                ? random.nextInt(actions) : greedy(step, s);
        step++;
        return new float[] {a};
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Discrete d) || d.n() != actions
                || env.observationSpace().size() != states) {
            throw new IllegalArgumentException("needs " + states + " one-hot states and " + actions + " actions");
        }
        Tracker tracker = new Tracker(listener);
        float[] obs = env.reset(random.split());
        episodeStart();
        while (tracker.steps() < steps) {
            int s = state(obs);
            int h = Math.min(step, horizon - 1);
            float[] action = act(obs, true);
            int a = (int) action[0];
            Environment.Step st = env.step(action);
            totalSteps++;
            double r = st.reward();
            if (intrinsic != null) {
                r += intrinsicScale * intrinsic.observe(obs, action, st.observation());
            }
            int next = st.terminated() ? states : state(st.observation());
            record(s, a, r, next);
            learn(h, s, a, r, next, st.terminated() || h + 1 >= horizon);
            if (tracker.step(st)) {
                obs = env.reset(random.split());
                episodeStart();
            } else {
                obs = st.observation();
            }
        }
        return tracker.result();
    }

    private void episodeStart() {
        step = 0;
        switch (method) {
            case PSRL -> plan();
            case UCRL2 -> {
                if (planDirty || doubled()) {
                    plan();
                }
            }
            case RMAX -> {
                if (planDirty) {
                    plan();
                }
            }
            default -> {
            }
        }
    }

    private boolean doubled() {
        for (int i = 0; i < visits.length; i++) {
            if (visits[i] >= Math.max(1, 2 * visitsAtPlan[i])) {
                return true;
            }
        }
        return false;
    }

    private void record(int s, int a, double r, int next) {
        int i = s * actions + a;
        visits[i]++;
        rewardSum[i] += r;
        rewardSq[i] += r * r;
        if (successors[i] == null) {
            successors[i] = new int[2];
            successorCounts[i] = new int[2];
        }
        int k = 0;
        while (k < successorSize[i] && successors[i][k] != next) {
            k++;
        }
        if (k == successorSize[i]) {
            if (k == successors[i].length) {
                successors[i] = Arrays.copyOf(successors[i], 2 * k);
                successorCounts[i] = Arrays.copyOf(successorCounts[i], 2 * k);
            }
            successors[i][k] = next;
            successorSize[i]++;
        }
        successorCounts[i][k]++;
        if (method == Method.RMAX && visits[i] == knownVisits) {
            planDirty = true;
            plan();
        }
    }

    private double value(int h, int s) {
        if (h >= horizon || s >= states) {
            return 0;
        }
        double[] row = q[h];
        double best = Double.NEGATIVE_INFINITY;
        for (int a = 0; a < actions; a++) {
            best = Math.max(best, row[s * actions + a]);
        }
        return best;
    }

    private void learn(int h, int s, int a, double r, int next, boolean last) {
        int i = s * actions + a;
        switch (method) {
            case EPSILON_GREEDY_Q -> {
                double target = r + (last ? 0 : value(h + 1, next));
                q[h][i] += learningRate * (target - q[h][i]);
            }
            case UCB_Q_HOEFFDING, UCB_Q_BERNSTEIN -> {
                int t = ++stepVisits[h][i];
                double lr = (horizon + 1.0) / (horizon + t);
                double vNext = last ? 0 : Math.min(rewardMax * (horizon - h - 1), value(h + 1, next));
                vSum[h][i] += vNext;
                vSq[h][i] += vNext * vNext;
                double iota = Math.log(states * actions * Math.max(2, totalSteps) / delta);
                double bonus;
                if (method == Method.UCB_Q_HOEFFDING) {
                    bonus = bonusScale * rewardMax * horizon * Math.sqrt(iota / t);
                } else {
                    double mean = vSum[h][i] / t;
                    double var = Math.max(0, vSq[h][i] / t - mean * mean);
                    bonus = bonusScale * (Math.sqrt(var * iota / t) + rewardMax * horizon * iota / t);
                }
                double target = r + vNext + bonus;
                q[h][i] = Math.min(rewardMax * (horizon - h), (1 - lr) * q[h][i] + lr * target);
            }
            default -> {
            }
        }
    }

    /** Plans the step-indexed Q-values for the current model (or posterior sample). */
    private void plan() {
        plans++;
        planDirty = false;
        System.arraycopy(visits, 0, visitsAtPlan, 0, visits.length);
        double[] vNext = new double[states + 1];
        double[] vCur = new double[states + 1];
        double logTerm = Math.log(2.0 * states * actions * Math.max(2, totalSteps) / delta);
        double alpha0 = dirichletPrior > 0 ? dirichletPrior : 1.0 / (states + 1);
        // per-pair sampled models for PSRL, shared over the horizon
        double[] sampledReward = null;
        double[][] sampledProbs = null;
        double[] sampledLump = null;
        if (method == Method.PSRL) {
            sampledReward = new double[visits.length];
            sampledProbs = new double[visits.length][];
            sampledLump = new double[visits.length];
            for (int i = 0; i < visits.length; i++) {
                int n = visits[i];
                double mean = n == 0 ? 0 : rewardSum[i] / n;
                double kappa = 1 + n;
                double mu = n * mean / kappa;
                double alpha = 1 + n / 2.0;
                double ss = n == 0 ? 0 : rewardSq[i] - n * mean * mean;
                double beta = 1 + 0.5 * ss + n * mean * mean / (2 * kappa);
                double tau = BanditMath.gamma(alpha, random) / beta;
                sampledReward[i] = mu + random.nextGaussian() / Math.sqrt(kappa * tau);
                int k = successorSize[i];
                double[] g = new double[k];
                double total = 0;
                for (int j = 0; j < k; j++) {
                    g[j] = BanditMath.gamma(successorCounts[i][j] + alpha0, random);
                    total += g[j];
                }
                double lump = BanditMath.gamma(alpha0 * (states + 1 - k), random);
                total += lump;
                for (int j = 0; j < k; j++) {
                    g[j] /= total;
                }
                sampledProbs[i] = g;
                sampledLump[i] = lump / total;
            }
        }
        Integer[] order = new Integer[states + 1];
        for (int h = horizon - 1; h >= 0; h--) {
            double cap = rewardMax * (horizon - h);
            double lumpMean = 0;
            for (int s = 0; s <= states; s++) {
                lumpMean += vNext[s];
            }
            lumpMean /= states + 1;
            if (method == Method.UCRL2) {
                for (int s = 0; s <= states; s++) {
                    order[s] = s;
                }
                final double[] vn = vNext;
                Arrays.sort(order, (x, y) -> Double.compare(vn[y], vn[x]));
            }
            for (int s = 0; s < states; s++) {
                double best = Double.NEGATIVE_INFINITY;
                for (int a = 0; a < actions; a++) {
                    int i = s * actions + a;
                    int n = visits[i];
                    double qv;
                    switch (method) {
                        case RMAX -> {
                            if (n < knownVisits) {
                                qv = cap;
                            } else {
                                qv = rewardSum[i] / n + expectation(i, vNext);
                            }
                        }
                        case UCRL2 -> {
                            double rHat = n == 0 ? rewardMax : rewardSum[i] / n
                                    + bonusScale * Math.sqrt(3.5 * logTerm / n);
                            double radius = n == 0 ? 2 : bonusScale * Math.sqrt(14 * (states + 1) * logTerm / n);
                            qv = Math.min(rewardMax, rHat) + optimisticExpectation(i, vNext, order, radius);
                        }
                        case PSRL -> qv = sampledReward[i] + sampledLump[i] * lumpMean
                                + sampledExpectation(i, sampledProbs[i], vNext);
                        default -> qv = q[h][i];
                    }
                    if (method != Method.PSRL) {
                        qv = Math.min(cap, qv);
                    }
                    q[h][i] = qv;
                    best = Math.max(best, qv);
                }
                vCur[s] = best;
            }
            vCur[states] = 0;
            double[] tmp = vNext;
            vNext = vCur;
            vCur = tmp;
        }
    }

    private double expectation(int i, double[] v) {
        int n = visits[i];
        double e = 0;
        for (int j = 0; j < successorSize[i]; j++) {
            e += successorCounts[i][j] * v[successors[i][j]];
        }
        return n == 0 ? 0 : e / n;
    }

    private double sampledExpectation(int i, double[] probs, double[] v) {
        double e = 0;
        for (int j = 0; j < successorSize[i]; j++) {
            e += probs[j] * v[successors[i][j]];
        }
        return e;
    }

    /** The inner maximisation of extended value iteration over an L1 ball of the given radius. */
    private double optimisticExpectation(int i, double[] v, Integer[] order, double radius) {
        int n = visits[i];
        double[] p = new double[states + 1];
        if (n > 0) {
            for (int j = 0; j < successorSize[i]; j++) {
                p[successors[i][j]] = successorCounts[i][j] / (double) n;
            }
        }
        int top = order[0];
        double add = Math.min(1 - p[top], radius / 2);
        if (n == 0) {
            return v[top];
        }
        p[top] += add;
        double excess = add;
        for (int k = order.length - 1; k >= 0 && excess > 1e-15; k--) {
            int s = order[k];
            if (s == top) {
                continue;
            }
            double take = Math.min(p[s], excess);
            p[s] -= take;
            excess -= take;
        }
        double e = 0;
        for (int s = 0; s <= states; s++) {
            e += p[s] * v[s];
        }
        return e;
    }

    @Override
    public String toString() {
        return method.name();
    }
}
