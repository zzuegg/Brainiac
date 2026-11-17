package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tracker;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * The option-critic architecture (Bacon, Harb and Precup, 2017,
 * <a href="https://arxiv.org/abs/1609.05140">arXiv:1609.05140</a>), tabular as in the paper's
 * four-rooms experiments: {@code k} options, each with a softmax intra-option policy
 * {@code pi_w(a|s)} and a sigmoid termination function {@code beta_w(s)}, plus an
 * epsilon-greedy policy over options from the option values {@code Q_Omega(s, w)}. All are
 * learned end to end, online: the critic {@code Q_U(s, w, a)} by intra-option Q-learning,
 * the intra-option policies by the intra-option policy gradient
 * {@code d log pi_w(a|s) (Q_U(s, w, a) - Q_Omega(s, w))}, and the terminations by the
 * termination gradient {@code -d beta_w(s') (Q_Omega(s', w) - V(s') + xi)}, which ends an
 * option where it is worse than the best one (the margin {@code xi} favours longer options).
 * Observations are used as table keys.
 */
public final class OptionCritic implements Learner {

    private final int actions;
    private final int options;
    private final Map<String, double[][]> qu = new HashMap<>();
    private final Map<String, double[][]> theta = new HashMap<>();
    private final Map<String, double[]> beta = new HashMap<>();
    private double criticRate = 0.5;
    private double policyRate = 0.25;
    private double terminationRate = 0.25;
    private double gamma = 0.99;
    private double epsilon = 0.05;
    private double margin = 0.01;
    private double temperature = 1;
    private SplittableRandom random = new SplittableRandom(0);
    private long switches;
    private long optionSteps;
    private int current = -1;

    private OptionCritic(int actions, int options) {
        this.actions = actions;
        this.options = options;
    }

    /**
     * @param actions the number of discrete actions
     * @param options the number of options
     * @return the learner
     */
    public static OptionCritic of(int actions, int options) {
        return new OptionCritic(actions, options);
    }

    /**
     * @param critic the critic learning rate
     * @param policy the intra-option policy learning rate
     * @param termination the termination learning rate
     * @return this learner
     */
    public OptionCritic learningRates(double critic, double policy, double termination) {
        this.criticRate = critic;
        this.policyRate = policy;
        this.terminationRate = termination;
        return this;
    }

    /** @param discount the discount @return this learner */
    public OptionCritic discount(double discount) {
        this.gamma = discount;
        return this;
    }

    /** @param eps exploration of the policy over options @return this learner */
    public OptionCritic epsilon(double eps) {
        this.epsilon = eps;
        return this;
    }

    /** @param xi the termination margin @return this learner */
    public OptionCritic terminationMargin(double xi) {
        this.margin = xi;
        return this;
    }

    /** @param seed the seed @return this learner */
    public OptionCritic seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return the mean number of steps an option ran before terminating, during training */
    public double meanOptionDuration() {
        return switches == 0 ? optionSteps : optionSteps / (double) switches;
    }

    private static String key(float[] obs) {
        return Arrays.toString(obs);
    }

    private double[][] qu(String s) {
        return qu.computeIfAbsent(s, k -> new double[options][actions]);
    }

    private double[][] theta(String s) {
        return theta.computeIfAbsent(s, k -> new double[options][actions]);
    }

    private double[] beta(String s) {
        return beta.computeIfAbsent(s, k -> new double[options]);
    }

    private double[] policy(String s, int w) {
        double[] l = theta(s)[w];
        double max = Arrays.stream(l).max().orElse(0);
        double[] p = new double[actions];
        double sum = 0;
        for (int a = 0; a < actions; a++) {
            p[a] = Math.exp((l[a] - max) / temperature);
            sum += p[a];
        }
        for (int a = 0; a < actions; a++) {
            p[a] /= sum;
        }
        return p;
    }

    private double qOmega(String s, int w) {
        double[] p = policy(s, w);
        double[] q = qu(s)[w];
        double v = 0;
        for (int a = 0; a < actions; a++) {
            v += p[a] * q[a];
        }
        return v;
    }

    private double termination(String s, int w) {
        return 1 / (1 + Math.exp(-beta(s)[w]));
    }

    private int chooseOption(String s, boolean explore) {
        if (explore && random.nextDouble() < epsilon) {
            return random.nextInt(options);
        }
        int best = 0;
        double bestV = Double.NEGATIVE_INFINITY;
        int ties = 0;
        for (int w = 0; w < options; w++) {
            double v = qOmega(s, w);
            if (v > bestV) {
                bestV = v;
                best = w;
                ties = 1;
            } else if (v == bestV && random.nextInt(++ties) == 0) {
                best = w;
            }
        }
        return best;
    }

    private int sample(double[] p) {
        double u = random.nextDouble();
        for (int a = 0; a < p.length; a++) {
            u -= p[a];
            if (u <= 0) {
                return a;
            }
        }
        return p.length - 1;
    }

    @Override
    public void startEpisode() {
        current = -1;
    }

    /** Acts with the current option, terminating and re-choosing options as during training. */
    @Override
    public float[] act(float[] observation, boolean explore) {
        String s = key(observation);
        if (current < 0 || random.nextDouble() < termination(s, current)) {
            current = chooseOption(s, explore);
        }
        double[] p = policy(s, current);
        if (explore) {
            return new float[] {sample(p)};
        }
        return new float[] {AgentKit.argmax(toFloat(p))};
    }

    private static float[] toFloat(double[] d) {
        float[] f = new float[d.length];
        for (int i = 0; i < d.length; i++) {
            f[i] = (float) d[i];
        }
        return f;
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Discrete d) || d.n() != actions) {
            throw new IllegalArgumentException("needs " + actions + " discrete actions");
        }
        Tracker tracker = new Tracker(listener);
        float[] obs = env.reset(random.split());
        String s = key(obs);
        int w = chooseOption(s, true);
        while (tracker.steps() < steps) {
            double[] p = policy(s, w);
            int a = sample(p);
            Environment.Step st = env.step(new float[] {a});
            String next = key(st.observation());
            optionSteps++;
            // critic: intra-option Q-learning
            double target = st.reward();
            double b = termination(next, w);
            if (!st.terminated()) {
                double best = Double.NEGATIVE_INFINITY;
                for (int o = 0; o < options; o++) {
                    best = Math.max(best, qOmega(next, o));
                }
                target += gamma * ((1 - b) * qOmega(next, w) + b * best);
                // termination gradient
                double advantage = qOmega(next, w) - best + margin;
                beta(next)[w] -= terminationRate * b * (1 - b) * advantage;
            }
            double[] q = qu(s)[w];
            q[a] += criticRate * (target - q[a]);
            // intra-option policy gradient with the option value as baseline
            double baseline = qOmega(s, w);
            double[] l = theta(s)[w];
            double adv = q[a] - baseline;
            for (int x = 0; x < actions; x++) {
                l[x] += policyRate * ((x == a ? 1 : 0) - p[x]) * adv;
            }
            if (tracker.step(st)) {
                switches++;
                obs = env.reset(random.split());
                s = key(obs);
                w = chooseOption(s, true);
            } else {
                s = next;
                if (random.nextDouble() < termination(s, w)) {
                    switches++;
                    w = chooseOption(s, true);
                }
            }
        }
        return tracker.result();
    }

    @Override
    public String toString() {
        return "OptionCritic(" + options + " options)";
    }
}
