package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Contextual bandits for online recommendation with exploration. Each round the policy sees a
 * context {@code x}, picks an arm and observes only that arm's reward.
 * <ul>
 *   <li>{@link #linUcb} (Li et al., 2010, <a href="https://arxiv.org/abs/1003.0146">
 *       arXiv:1003.0146</a>): a ridge-regression posterior per arm; pick
 *       {@code argmax theta_a . x + alpha sqrt(x^T A_a^-1 x)} (optimism under uncertainty);</li>
 *   <li>{@link #linearThompson} (Agrawal and Goyal, 2013, <a href="https://arxiv.org/abs/1209.3352">
 *       arXiv:1209.3352</a>): sample {@code theta_a ~ N(theta_hat_a, v^2 A_a^-1)} and act greedily
 *       on the sample;</li>
 *   <li>{@link #neuralLinear} (Riquelme et al., 2018, <a href="https://arxiv.org/abs/1802.09127">
 *       arXiv:1802.09127</a>): Bayesian linear regression with Thompson sampling on the last hidden
 *       layer of a network that is retrained on all data every few rounds;</li>
 *   <li>baselines: {@link #epsilonGreedy} on the same ridge models, {@link #ucb1} that ignores the
 *       context (core {@code Bandit}-style), and {@link #neuralGreedy}, the neural model acting
 *       greedily on its own logs.</li>
 * </ul>
 * Rewards are modelled linearly per arm (on the context, or on learned features). As an
 * {@link Agent} the policy maps an observation to {@code {arm}}; call {@link #update} with the
 * observed reward.
 */
public final class ContextualBandit implements Agent {

    private enum Kind {
        LIN_UCB, THOMPSON, EPSILON_GREEDY, UCB1, NEURAL_LINEAR, NEURAL_GREEDY
    }

    private final Kind kind;
    private final int arms;
    private final double param;
    private final double lambda;
    private final SplittableRandom random;
    private final int hidden;
    private final int retrainEvery;
    private double[][][] inverse;
    private double[][] b;
    private final int[] pulls;
    private final double[] sums;
    private int rounds;
    private final List<double[]> contexts = new ArrayList<>();
    private final List<Integer> chosen = new ArrayList<>();
    private final List<Double> rewards = new ArrayList<>();
    private Var w1;
    private Var b1;
    private Var w2;
    private Var b2;

    private ContextualBandit(Kind kind, int dim, int arms, double param, double lambda, int hidden, int retrainEvery,
            long seed) {
        this.kind = kind;
        this.arms = arms;
        this.param = param;
        this.lambda = lambda;
        this.hidden = hidden;
        this.retrainEvery = retrainEvery;
        this.random = new SplittableRandom(seed);
        this.pulls = new int[arms];
        this.sums = new double[arms];
        int features = (kind == Kind.NEURAL_LINEAR || kind == Kind.NEURAL_GREEDY ? hidden : dim) + 1;
        resetPosterior(features);
        if (hidden > 0) {
            w1 = RecsysOps.matrix(random, dim, hidden);
            b1 = RecsysOps.zeros(hidden);
            w2 = RecsysOps.matrix(random, hidden, arms);
            b2 = RecsysOps.zeros(arms);
        }
    }

    private void resetPosterior(int features) {
        inverse = new double[arms][][];
        b = new double[arms][features];
        for (int a = 0; a < arms; a++) {
            inverse[a] = MlMatrices.identity(features);
            for (int i = 0; i < features; i++) {
                inverse[a][i][i] = 1 / lambda;
            }
        }
    }

    /**
     * LinUCB with disjoint linear models.
     *
     * @param dim the context width
     * @param arms the number of arms
     * @param alpha the exploration bonus scale
     * @param lambda the ridge penalty
     * @return the policy
     */
    public static ContextualBandit linUcb(int dim, int arms, double alpha, double lambda) {
        return new ContextualBandit(Kind.LIN_UCB, dim, arms, alpha, lambda, 0, 0, 0);
    }

    /**
     * Linear Thompson sampling.
     *
     * @param dim the context width
     * @param arms the number of arms
     * @param v the posterior scale
     * @param lambda the ridge penalty
     * @param seed the sampling seed
     * @return the policy
     */
    public static ContextualBandit linearThompson(int dim, int arms, double v, double lambda, long seed) {
        return new ContextualBandit(Kind.THOMPSON, dim, arms, v, lambda, 0, 0, seed);
    }

    /**
     * Epsilon-greedy on ridge models.
     *
     * @param dim the context width
     * @param arms the number of arms
     * @param epsilon the share of random arms
     * @param lambda the ridge penalty
     * @param seed the seed
     * @return the policy
     */
    public static ContextualBandit epsilonGreedy(int dim, int arms, double epsilon, double lambda, long seed) {
        return new ContextualBandit(Kind.EPSILON_GREEDY, dim, arms, epsilon, lambda, 0, 0, seed);
    }

    /**
     * UCB1, which ignores the context.
     *
     * @param arms the number of arms
     * @return the policy
     */
    public static ContextualBandit ucb1(int arms) {
        return new ContextualBandit(Kind.UCB1, 1, arms, 0, 1, 0, 0, 0);
    }

    /**
     * Neural-linear Thompson sampling.
     *
     * @param dim the context width
     * @param arms the number of arms
     * @param hidden the network's last hidden width
     * @param retrainEvery rounds between retrainings of the network
     * @param v the posterior scale
     * @param seed the seed
     * @return the policy
     */
    public static ContextualBandit neuralLinear(int dim, int arms, int hidden, int retrainEvery, double v,
            long seed) {
        return new ContextualBandit(Kind.NEURAL_LINEAR, dim, arms, v, 1, hidden, retrainEvery, seed);
    }

    /**
     * The neural model acting greedily on its predictions (no exploration), retrained on its own
     * logs.
     *
     * @param dim the context width
     * @param arms the number of arms
     * @param hidden the hidden width
     * @param retrainEvery rounds between retrainings
     * @param seed the seed
     * @return the policy
     */
    public static ContextualBandit neuralGreedy(int dim, int arms, int hidden, int retrainEvery, long seed) {
        return new ContextualBandit(Kind.NEURAL_GREEDY, dim, arms, 0, 1, hidden, retrainEvery, seed);
    }

    private double[] features(double[] x) {
        if (hidden == 0) {
            double[] out = java.util.Arrays.copyOf(x, x.length + 1);
            out[x.length] = 1;
            return out;
        }
        float[] f = new float[x.length];
        for (int i = 0; i < f.length; i++) {
            f[i] = (float) x[i];
        }
        float[] h = Var.noGrad(() -> Var.of(Tensor.of(Shape.of(1, x.length), f)).matmul(w1).add(b1).relu()).value()
                .toArray();
        double[] out = new double[hidden + 1];
        for (int i = 0; i < hidden; i++) {
            out[i] = h[i];
        }
        out[hidden] = 1;
        return out;
    }

    private double[] mean(int a) {
        return multiply(inverse[a], b[a]);
    }

    private static double[] multiply(double[][] m, double[] v) {
        double[] out = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = MlMatrices.dot(m[i], v);
        }
        return out;
    }

    /**
     * Picks an arm.
     *
     * @param x the context
     * @return the arm
     */
    public int select(double[] x) {
        if (kind == Kind.UCB1) {
            for (int a = 0; a < arms; a++) {
                if (pulls[a] == 0) {
                    return a;
                }
            }
            return argmax(a -> sums[a] / pulls[a] + Math.sqrt(2 * Math.log(rounds) / pulls[a]));
        }
        if (kind == Kind.EPSILON_GREEDY && random.nextDouble() < param) {
            return random.nextInt(arms);
        }
        double[] f = features(x);
        return argmax(a -> {
            double[] theta = mean(a);
            return switch (kind) {
                case LIN_UCB -> MlMatrices.dot(theta, f) + param * Math.sqrt(MlMatrices.dot(f,
                        multiply(inverse[a], f)));
                case THOMPSON, NEURAL_LINEAR -> MlMatrices.dot(sample(theta, inverse[a]), f);
                default -> MlMatrices.dot(theta, f);
            };
        });
    }

    private double[] sample(double[] theta, double[][] cov) {
        int d = theta.length;
        double[][] scaled = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                scaled[i][j] = param * param * (cov[i][j] + cov[j][i]) / 2 + (i == j ? 1e-9 : 0);
            }
        }
        double[][] l = MlMatrices.cholesky(scaled);
        double[] z = new double[d];
        for (int i = 0; i < d; i++) {
            z[i] = random.nextGaussian();
        }
        double[] out = theta.clone();
        for (int i = 0; i < d; i++) {
            for (int j = 0; j <= i; j++) {
                out[i] += l[i][j] * z[j];
            }
        }
        return out;
    }

    private int argmax(java.util.function.IntToDoubleFunction score) {
        int best = 0;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int a = 0; a < arms; a++) {
            double s = score.applyAsDouble(a);
            if (s > bestScore) {
                bestScore = s;
                best = a;
            }
        }
        return best;
    }

    /**
     * Records the reward of the chosen arm.
     *
     * @param x the context
     * @param arm the arm played
     * @param reward the observed reward
     */
    public void update(double[] x, int arm, double reward) {
        rounds++;
        pulls[arm]++;
        sums[arm] += reward;
        if (kind == Kind.UCB1) {
            return;
        }
        if (hidden > 0) {
            contexts.add(x.clone());
            chosen.add(arm);
            rewards.add(reward);
            if (rounds % retrainEvery == 0) {
                retrain();
                return;
            }
        }
        absorb(features(x), arm, reward);
    }

    /** Sherman-Morrison update of the arm's inverse Gram matrix. */
    private void absorb(double[] f, int arm, double reward) {
        double[][] inv = inverse[arm];
        double[] u = multiply(inv, f);
        double denom = 1 + MlMatrices.dot(f, u);
        for (int i = 0; i < u.length; i++) {
            for (int j = 0; j < u.length; j++) {
                inv[i][j] -= u[i] * u[j] / denom;
            }
        }
        for (int i = 0; i < f.length; i++) {
            b[arm][i] += reward * f[i];
        }
    }

    private void retrain() {
        int n = contexts.size();
        int dim = contexts.get(0).length;
        Optimizer opt = Optimizer.adam(0.01f);
        List<Var> params = List.of(w1, b1, w2, b2);
        for (int s = 0; s < 150; s++) {
            int batch = Math.min(64, n);
            float[] x = new float[batch * dim];
            float[] mask = new float[batch * arms];
            float[] y = new float[batch * arms];
            for (int i = 0; i < batch; i++) {
                int j = random.nextInt(n);
                for (int k = 0; k < dim; k++) {
                    x[i * dim + k] = (float) contexts.get(j)[k];
                }
                mask[i * arms + chosen.get(j)] = 1;
                y[i * arms + chosen.get(j)] = rewards.get(j).floatValue();
            }
            Var pred = Var.of(Tensor.of(Shape.of(batch, dim), x)).matmul(w1).add(b1).relu().matmul(w2).add(b2);
            pred.sub(Tensor.of(Shape.of(batch, arms), y)).mul(Tensor.of(Shape.of(batch, arms), mask)).square().sum()
                    .div(batch).backward();
            opt.step(params);
        }
        resetPosterior(hidden + 1);
        float[] all = new float[n * dim];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < dim; k++) {
                all[i * dim + k] = (float) contexts.get(i)[k];
            }
        }
        float[] h = Var.noGrad(() -> Var.of(Tensor.of(Shape.of(n, dim), all)).matmul(w1).add(b1).relu()).value()
                .toArray();
        for (int i = 0; i < n; i++) {
            double[] f = new double[hidden + 1];
            for (int k = 0; k < hidden; k++) {
                f[k] = h[i * hidden + k];
            }
            f[hidden] = 1;
            absorb(f, chosen.get(i), rewards.get(i));
        }
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        double[] x = new double[observation.length];
        for (int i = 0; i < x.length; i++) {
            x[i] = observation[i];
        }
        return new float[] {select(x)};
    }

    @Override
    public String toString() {
        return kind + "(" + arms + " arms)";
    }
}
