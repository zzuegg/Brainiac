package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tracker;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A tabular Agent57 (Badia et al., 2020, <a href="https://arxiv.org/abs/2003.13350">arXiv:2003.13350</a>):
 * a family of policies, arm {@code j} maximizing {@code r_extrinsic + beta_j r_intrinsic}
 * with its own discount {@code gamma_j} (more exploratory arms get a larger {@code beta}
 * and a shorter horizon), all learned off-policy from the same experience, with a
 * meta-controller, a sliding-window UCB bandit over the arms rewarded by the extrinsic
 * return, choosing which arm acts in each episode. The intrinsic reward is
 * {@link NeverGiveUp}'s. Arm 0 ({@code beta = 0}) is the exploitative policy used when
 * acting greedily. The intrinsic reward is centred by its running mean, which tabular
 * learners with zero-initialized values need (see {@link CuriosityEnvironment#centered}).
 *
 * <p>The paper's arms share one network conditioned on the arm index and are trained by
 * distributed R2D2 actors; here every arm has its own Q-table, which keeps the idea (many
 * exploration levels, one shared stream of experience, a bandit picking among them) in a
 * form small enough to test.
 */
public final class Agent57Learner implements Learner {

    private final int actions;
    private final NeverGiveUp bonus;
    private final double[] betas;
    private final double[] gammas;
    private final List<Map<String, double[]>> tables = new ArrayList<>();
    private final Deque<double[]> window = new ArrayDeque<>();
    private int windowSize = 50;
    private double ucb = 0.5;
    private double banditEpsilon = 0.1;
    private double learningRate = 0.5;
    private double epsilon = 0.05;
    private SplittableRandom random = new SplittableRandom(0);
    private final int[] plays;
    private final AgentKit.Running intrinsic = new AgentKit.Running();

    private Agent57Learner(int actions, NeverGiveUp bonus, int arms, double maxBeta) {
        this.actions = actions;
        this.bonus = Objects.requireNonNull(bonus);
        this.betas = new double[arms];
        this.gammas = new double[arms];
        this.plays = new int[arms];
        for (int j = 0; j < arms; j++) {
            double x = arms == 1 ? 0 : j / (double) (arms - 1);
            betas[j] = maxBeta * x;
            gammas[j] = 1 - Math.exp(Math.log(1 - 0.99) * (1 - x) + Math.log(1 - 0.95) * x);
            tables.add(new HashMap<>());
        }
    }

    /**
     * @param actions the number of discrete actions
     * @param bonus the intrinsic reward
     * @param arms how many (beta, gamma) policies
     * @param maxBeta the intrinsic weight of the most exploratory arm
     * @return the learner
     */
    public static Agent57Learner of(int actions, NeverGiveUp bonus, int arms, double maxBeta) {
        return new Agent57Learner(actions, bonus, arms, maxBeta);
    }

    /** @param rate the Q-learning rate @return this learner */
    public Agent57Learner learningRate(double rate) {
        this.learningRate = rate;
        return this;
    }

    /** @param episodes the bandit's sliding window @return this learner */
    public Agent57Learner window(int episodes) {
        this.windowSize = episodes;
        return this;
    }

    /** @param seed the seed @return this learner */
    public Agent57Learner seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return how many episodes each arm acted in */
    public int[] plays() {
        return plays.clone();
    }

    /** @return each arm's intrinsic weight */
    public double[] betas() {
        return betas.clone();
    }

    private double[] q(int arm, float[] obs) {
        return tables.get(arm).computeIfAbsent(Arrays.toString(obs), k -> new double[actions]);
    }

    private int greedy(double[] v) {
        int best = 0;
        int ties = 1;
        for (int a = 1; a < v.length; a++) {
            if (v[a] > v[best]) {
                best = a;
                ties = 1;
            } else if (v[a] == v[best] && random.nextInt(++ties) == 0) {
                best = a;
            }
        }
        return best;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        return new float[] {greedy(q(0, observation))};
    }

    private int chooseArm() {
        int arms = betas.length;
        double[] sum = new double[arms];
        int[] count = new int[arms];
        for (double[] e : window) {
            sum[(int) e[0]] += e[1];
            count[(int) e[0]]++;
        }
        for (int j = 0; j < arms; j++) {
            if (count[j] == 0) {
                return j;
            }
        }
        if (random.nextDouble() < banditEpsilon) {
            return random.nextInt(arms);
        }
        double[] score = new double[arms];
        for (int j = 0; j < arms; j++) {
            score[j] = sum[j] / count[j] + ucb * Math.sqrt(Math.log(window.size()) / count[j]);
        }
        return RlTensors.argmax(score);
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Discrete d) || d.n() != actions) {
            throw new IllegalArgumentException("needs " + actions + " discrete actions");
        }
        Tracker tracker = new Tracker(listener);
        float[] obs = env.reset(random.split());
        bonus.newEpisode();
        int arm = chooseArm();
        double extrinsic = 0;
        while (tracker.steps() < steps) {
            double[] qa = q(arm, obs);
            int a = random.nextDouble() < epsilon ? random.nextInt(actions) : greedy(qa);
            float[] action = {a};
            Environment.Step s = env.step(action);
            extrinsic += s.reward();
            double raw = bonus.observe(obs, action, s.observation());
            intrinsic.add(raw);
            // centred, so that with zero-initialized tables familiar transitions look worse than untried ones
            double ri = raw - intrinsic.mean();
            for (int j = 0; j < betas.length; j++) {
                double[] here = q(j, obs);
                double next = 0;
                if (!s.terminated()) {
                    double[] qn = q(j, s.observation());
                    next = Arrays.stream(qn).max().orElse(0);
                }
                double target = s.reward() + betas[j] * ri + gammas[j] * next;
                here[a] += learningRate * (target - here[a]);
            }
            if (tracker.step(s)) {
                window.addLast(new double[] {arm, extrinsic});
                plays[arm]++;
                while (window.size() > windowSize) {
                    window.removeFirst();
                }
                obs = env.reset(random.split());
                bonus.newEpisode();
                arm = chooseArm();
                extrinsic = 0;
            } else {
                obs = s.observation();
            }
        }
        return tracker.result();
    }

    @Override
    public String toString() {
        return "Agent57(" + betas.length + " arms)";
    }
}
