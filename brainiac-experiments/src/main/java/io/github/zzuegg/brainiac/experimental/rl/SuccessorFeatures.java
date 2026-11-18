package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Tracker;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Successor features with generalized policy improvement (Barreto et al., 2017,
 * <a href="https://arxiv.org/abs/1606.05312">arXiv:1606.05312</a>), tabular. For tasks whose
 * reward is linear in features, {@code r = phi(s, a, s') . w} ({@link FeatureEnvironment}),
 * the action values of a policy factor as {@code Q(s, a) = psi(s, a) . w}, where the
 * successor features {@code psi(s, a)} are the expected discounted sum of future features,
 * learned by TD like a value: {@code psi(s, a) <- phi + gamma psi(s', a')}. Each
 * {@link #fit} learns one policy (greedy for that environment's {@code w}) and keeps its
 * {@code psi}. For a <em>new</em> {@code w} the agent acts zero-shot by GPI:
 * {@code a = argmax_a max_i psi_i(s, a) . w}, which is at least as good as every stored
 * policy on the new task. {@link #inferWeights} recovers an unknown {@code w} from a few
 * observed rewards by least squares.
 */
public final class SuccessorFeatures implements Learner {

    private final int actions;
    private final int features;
    private final List<Map<String, double[][]>> psis = new ArrayList<>();
    private float[] task;
    private double learningRate = 0.5;
    private double gamma = 0.95;
    private double epsilon = 0.1;
    private SplittableRandom random = new SplittableRandom(0);

    private SuccessorFeatures(int actions, int features) {
        this.actions = actions;
        this.features = features;
    }

    /**
     * @param actions the number of discrete actions
     * @param features the number of features
     * @return the learner, with no policies yet
     */
    public static SuccessorFeatures of(int actions, int features) {
        return new SuccessorFeatures(actions, features);
    }

    /** @param rate the TD learning rate @return this learner */
    public SuccessorFeatures learningRate(double rate) {
        this.learningRate = rate;
        return this;
    }

    /** @param discount the discount @return this learner */
    public SuccessorFeatures discount(double discount) {
        this.gamma = discount;
        return this;
    }

    /** @param eps exploration while learning a policy @return this learner */
    public SuccessorFeatures epsilon(double eps) {
        this.epsilon = eps;
        return this;
    }

    /** @param seed the seed @return this learner */
    public SuccessorFeatures seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /**
     * Sets the task the agent acts for, by GPI over all stored policies.
     *
     * @param w the reward weights
     * @return this learner
     */
    public SuccessorFeatures task(float[] w) {
        if (w.length != features) {
            throw new IllegalArgumentException("need " + features + " weights");
        }
        this.task = w.clone();
        return this;
    }

    /** @return how many policies are stored */
    public int policies() {
        return psis.size();
    }

    private static double dot(double[] a, float[] w) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * w[i];
        }
        return s;
    }

    private double[][] psi(Map<String, double[][]> table, String s) {
        return table.computeIfAbsent(s, k -> new double[actions][features]);
    }

    /**
     * @param observation the observation
     * @param w the task
     * @return for every action, the best value any stored policy promises: {@code max_i psi_i . w}
     */
    public double[] gpiValues(float[] observation, float[] w) {
        String s = Arrays.toString(observation);
        double[] best = new double[actions];
        Arrays.fill(best, Double.NEGATIVE_INFINITY);
        for (Map<String, double[][]> table : psis) {
            double[][] p = psi(table, s);
            for (int a = 0; a < actions; a++) {
                best[a] = Math.max(best[a], dot(p[a], w));
            }
        }
        return best;
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
        if (task == null || psis.isEmpty()) {
            return new float[] {random.nextInt(actions)};
        }
        if (explore && random.nextDouble() < epsilon) {
            return new float[] {random.nextInt(actions)};
        }
        return new float[] {greedy(gpiValues(observation, task))};
    }

    /**
     * Learns a new policy, greedy for the environment's task, and stores its successor
     * features. The environment must be a {@link FeatureEnvironment}.
     */
    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        if (!(env instanceof FeatureEnvironment fe) || fe.features() != features) {
            throw new IllegalArgumentException("needs a FeatureEnvironment with " + features + " features");
        }
        float[] w = fe.weights();
        Map<String, double[][]> table = new HashMap<>();
        psis.add(table);
        Tracker tracker = new Tracker(listener);
        float[] obs = env.reset(random.split());
        while (tracker.steps() < steps) {
            String s = Arrays.toString(obs);
            double[][] p = psi(table, s);
            double[] q = new double[actions];
            for (int a = 0; a < actions; a++) {
                q[a] = dot(p[a], w);
            }
            int a = random.nextDouble() < epsilon ? random.nextInt(actions) : greedy(q);
            Environment.Step st = env.step(new float[] {a});
            float[] phi = fe.lastFeatures();
            double[] target = new double[features];
            for (int i = 0; i < features; i++) {
                target[i] = phi[i];
            }
            if (!st.terminated()) {
                double[][] pn = psi(table, Arrays.toString(st.observation()));
                double[] qn = new double[actions];
                for (int b = 0; b < actions; b++) {
                    qn[b] = dot(pn[b], w);
                }
                int next = greedy(qn);
                for (int i = 0; i < features; i++) {
                    target[i] += gamma * pn[next][i];
                }
            }
            for (int i = 0; i < features; i++) {
                p[a][i] += learningRate * (target[i] - p[a][i]);
            }
            obs = tracker.step(st) ? env.reset(random.split()) : st.observation();
        }
        task = w.clone();
        return tracker.result();
    }

    /**
     * Estimates a task's weights from observed transitions by ridge regression of the
     * rewards on the features.
     *
     * @param phis the features of each transition
     * @param rewards the rewards
     * @param ridge the regularization
     * @return the weights
     */
    public static float[] inferWeights(List<float[]> phis, List<Float> rewards, double ridge) {
        Objects.requireNonNull(phis);
        int d = phis.get(0).length;
        double[][] a = new double[d][d + 1];
        for (int k = 0; k < phis.size(); k++) {
            float[] f = phis.get(k);
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    a[i][j] += f[i] * f[j];
                }
                a[i][d] += f[i] * rewards.get(k);
            }
        }
        for (int i = 0; i < d; i++) {
            a[i][i] += ridge;
        }
        for (int c = 0; c < d; c++) {
            int pivot = c;
            for (int r = c + 1; r < d; r++) {
                if (Math.abs(a[r][c]) > Math.abs(a[pivot][c])) {
                    pivot = r;
                }
            }
            double[] t = a[c];
            a[c] = a[pivot];
            a[pivot] = t;
            for (int r = 0; r < d; r++) {
                if (r != c && a[c][c] != 0) {
                    double f = a[r][c] / a[c][c];
                    for (int k = c; k <= d; k++) {
                        a[r][k] -= f * a[c][k];
                    }
                }
            }
        }
        float[] w = new float[d];
        for (int i = 0; i < d; i++) {
            w[i] = a[i][i] == 0 ? 0 : (float) (a[i][d] / a[i][i]);
        }
        return w;
    }

    @Override
    public String toString() {
        return "SuccessorFeatures(" + psis.size() + " policies)";
    }
}
