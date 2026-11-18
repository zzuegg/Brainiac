package io.github.zzuegg.brainiac.experimental.explore;

import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tracker;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Randomised least-squares value iteration (Osband, Van Roy &amp; Wen, ICML 2016,
 * <a href="https://arxiv.org/abs/1402.0635">arXiv:1402.0635</a>): at the start of every episode
 * the agent samples a value function from a Bayesian linear-regression posterior fitted to all
 * data by value iteration, and acts greedily on that sample for the whole episode. Like
 * Thompson sampling it explores deeply and consistently, without dithering.
 *
 * <ul>
 *   <li>{@link #tabular}: one-hot features over the distinct observations seen, where the
 *       posterior of each {@code Q(s, a)} is independent: {@code Q~(s, a) = (sum r + xi +
 *       gamma sum max Q~(s')) / (n + lambda) + prior}, with {@code xi ~ N(0, n sigma^2)} the
 *       summed reward perturbation and a prior draw {@code N(0, sigma_p^2)} weighted by
 *       {@code lambda}; the noise is fixed per episode while the value-iteration sweeps run.</li>
 *   <li>{@link #linear}: Bayesian linear regression per action on the observation vector
 *       (plus a bias), {@code theta~ = mean + sigma L z} with the posterior's Cholesky factor
 *       and one {@code z} per episode, refitted over several sweeps of targets
 *       {@code r + gamma max_a' phi(s')^T theta~_a'}.</li>
 * </ul>
 */
public final class Rlsvi implements Learner {

    private record Sample(float[] obs, int action, float reward, float[] next, boolean terminal) {
    }

    private record Key(float[] v) {
        @Override
        public boolean equals(Object o) {
            return o instanceof Key k && Arrays.equals(v, k.v);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(v);
        }
    }

    private final boolean tabular;
    private final int actions;
    private double sigma = 0.1;
    private double priorScale = 1;
    private double lambda = 1;
    private double gamma = 0.99;
    private int sweeps = 30;
    private SplittableRandom random = new SplittableRandom(0);
    private final List<Sample> data = new ArrayList<>();
    private final Map<Key, Integer> states = new HashMap<>();
    private final List<double[]> q = new ArrayList<>();
    private double[][] theta;

    private Rlsvi(boolean tabular, int actions) {
        this.tabular = tabular;
        this.actions = actions;
    }

    /**
     * @param actions discrete actions
     * @return tabular RLSVI
     */
    public static Rlsvi tabular(int actions) {
        return new Rlsvi(true, actions);
    }

    /**
     * @param actions discrete actions
     * @return linear RLSVI on the observation vector
     */
    public static Rlsvi linear(int actions) {
        return new Rlsvi(false, actions);
    }

    /**
     * @param noise the reward noise scale of the posterior
     * @param prior the prior standard deviation of the values
     * @return this learner
     */
    public Rlsvi noise(double noise, double prior) {
        this.sigma = noise;
        this.priorScale = prior;
        return this;
    }

    /** @param discount gamma @return this learner */
    public Rlsvi discount(double discount) {
        this.gamma = discount;
        return this;
    }

    /** @param n value-iteration sweeps per episode @return this learner */
    public Rlsvi sweeps(int n) {
        this.sweeps = n;
        return this;
    }

    /** @param seed the seed @return this learner */
    public Rlsvi seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    private int state(float[] obs) {
        return states.computeIfAbsent(new Key(obs.clone()), k -> {
            q.add(new double[actions]);
            return states.size();
        });
    }

    private double[] values(float[] obs) {
        if (tabular) {
            Integer s = states.get(new Key(obs));
            return s == null ? new double[actions] : q.get(s);
        }
        double[] v = new double[actions];
        if (theta == null) {
            return v;
        }
        for (int a = 0; a < actions; a++) {
            double z = theta[a][obs.length];
            for (int i = 0; i < obs.length; i++) {
                z += theta[a][i] * obs[i];
            }
            v[a] = z;
        }
        return v;
    }

    private int greedy(double[] v) {
        int best = 0;
        int ties = 1;
        for (int a = 1; a < v.length; a++) {
            if (v[a] > v[best] + 1e-12) {
                best = a;
                ties = 1;
            } else if (Math.abs(v[a] - v[best]) <= 1e-12 && random.nextInt(++ties) == 0) {
                best = a;
            }
        }
        return best;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        return new float[] {greedy(values(observation))};
    }

    /** Samples a new value function from the posterior (both modes). */
    private void resample(boolean noisy) {
        if (tabular) {
            resampleTabular(noisy);
        } else {
            resampleLinear(noisy);
        }
    }

    private void resampleTabular(boolean noisy) {
        for (Sample s : data) {
            state(s.obs());
            state(s.next());
        }
        int n = states.size();
        double[][] sumR = new double[n][actions];
        int[][] count = new int[n][actions];
        List<Map<Integer, Integer>> next = new ArrayList<>();
        for (int i = 0; i < n * actions; i++) {
            next.add(new HashMap<>());
        }
        for (Sample s : data) {
            int i = state(s.obs());
            sumR[i][s.action()] += s.reward();
            count[i][s.action()]++;
            if (!s.terminal()) {
                next.get(i * actions + s.action()).merge(state(s.next()), 1, Integer::sum);
            }
        }
        n = states.size();
        double[][] xi = new double[n][actions];
        double[][] prior = new double[n][actions];
        for (int i = 0; i < n; i++) {
            for (int a = 0; a < actions; a++) {
                if (noisy) {
                    xi[i][a] = i < count.length ? Math.sqrt(count[i][a]) * sigma * random.nextGaussian() : 0;
                    prior[i][a] = priorScale * random.nextGaussian();
                }
            }
        }
        double[][] v = new double[n][actions];
        for (int sweep = 0; sweep < sweeps; sweep++) {
            double[] max = new double[n];
            for (int i = 0; i < n; i++) {
                max[i] = Arrays.stream(v[i]).max().orElse(0);
            }
            double[][] nv = new double[n][actions];
            for (int i = 0; i < n; i++) {
                for (int a = 0; a < actions; a++) {
                    int c = i < count.length ? count[i][a] : 0;
                    double boot = 0;
                    if (i < count.length) {
                        for (Map.Entry<Integer, Integer> e : next.get(i * actions + a).entrySet()) {
                            boot += e.getValue() * max[e.getKey()];
                        }
                    }
                    double sr = i < count.length ? sumR[i][a] : 0;
                    nv[i][a] = (sr + xi[i][a] + gamma * boot + lambda * prior[i][a]) / (c + lambda);
                }
            }
            v = nv;
        }
        for (int i = 0; i < n; i++) {
            System.arraycopy(v[i], 0, q.get(i), 0, actions);
        }
    }

    private void resampleLinear(boolean noisy) {
        if (data.isEmpty()) {
            return;
        }
        int d = data.get(0).obs().length + 1;
        double[][][] gram = new double[actions][d][d];
        for (int a = 0; a < actions; a++) {
            for (int i = 0; i < d; i++) {
                gram[a][i][i] = lambda * sigma * sigma / (priorScale * priorScale);
            }
        }
        for (Sample s : data) {
            double[] f = phi(s.obs());
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    gram[s.action()][i][j] += f[i] * f[j];
                }
            }
        }
        double[][][] chol = new double[actions][][];
        double[][][] covChol = new double[actions][][];
        double[][] z = new double[actions][d];
        for (int a = 0; a < actions; a++) {
            chol[a] = MlMatrices.cholesky(gram[a]);
            covChol[a] = MlMatrices.cholesky(MlMatrices.choleskyInverse(chol[a]));
            for (int i = 0; i < d; i++) {
                z[a][i] = noisy ? random.nextGaussian() : 0;
            }
        }
        theta = new double[actions][d];
        for (int sweep = 0; sweep < sweeps; sweep++) {
            double[][] b = new double[actions][d];
            for (Sample s : data) {
                double y = s.reward();
                if (!s.terminal()) {
                    y += gamma * Arrays.stream(values(s.next())).max().orElse(0);
                }
                double[] f = phi(s.obs());
                for (int i = 0; i < d; i++) {
                    b[s.action()][i] += y * f[i];
                }
            }
            double[][] nt = new double[actions][];
            for (int a = 0; a < actions; a++) {
                double[] mean = MlMatrices.choleskySolve(chol[a], b[a]);
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j <= i; j++) {
                        mean[i] += sigma * covChol[a][i][j] * z[a][j];
                    }
                }
                nt[a] = mean;
            }
            theta = nt;
        }
    }

    private static double[] phi(float[] obs) {
        double[] f = new double[obs.length + 1];
        for (int i = 0; i < obs.length; i++) {
            f[i] = obs[i];
        }
        f[obs.length] = 1;
        return f;
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Discrete dd) || dd.n() != actions) {
            throw new IllegalArgumentException("needs " + actions + " discrete actions");
        }
        Tracker t = new Tracker(listener);
        float[] obs = env.reset(random.split());
        resample(true);
        while (t.steps() < steps) {
            if (tabular) {
                state(obs);
            }
            int a = greedy(values(obs));
            Environment.Step s = env.step(a);
            data.add(new Sample(obs, a, s.reward(), s.observation(), s.terminated()));
            if (t.step(s)) {
                obs = env.reset(random.split());
                resample(true);
            } else {
                obs = s.observation();
            }
        }
        resample(false);
        return t.result();
    }

    @Override
    public String toString() {
        return tabular ? "RLSVI(tabular)" : "RLSVI(linear)";
    }
}
