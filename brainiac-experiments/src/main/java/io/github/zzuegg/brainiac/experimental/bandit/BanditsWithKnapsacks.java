package io.github.zzuegg.brainiac.experimental.bandit;

import io.github.zzuegg.brainiac.experimental.data.OperationsSims.KnapsackArms;
import io.github.zzuegg.brainiac.experimental.ml.LinearProgram;

import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Bandits with knapsacks: every pull pays a reward and consumes resources, and play stops once any
 * budget is spent, so the benchmark is the best distribution over arms (a linear program), not
 * the best arm. A {@link BanditPolicy} over the {@code K} real arms plus a null arm (index
 * {@code K}: no reward, no consumption); {@link #observe} reports the consumption.
 *
 * <ul>
 *   <li>{@link Method#PRIMAL_DUAL_BWK}: PD-BwK (Badanidiyuru, Kleinberg &amp; Slivkins, FOCS 2013,
 *       <a href="https://arxiv.org/abs/1305.2545">arXiv:1305.2545</a>): multiplicative weights over
 *       the resources (time included) act as prices, and the arm with the best ratio of
 *       optimistic reward to priced pessimistic cost is pulled;</li>
 *   <li>{@link Method#UCB_BWK}: the optimistic LP (Agrawal &amp; Devanur, EC 2014): each round solve
 *       {@code max sum x_k UCB(r_k)} subject to {@code sum x_k LCB(c_kj) <= B / T} and
 *       {@code sum x_k <= 1} with the wave 6 {@link LinearProgram}, and sample the arm from {@code x};</li>
 *   <li>{@link Method#LAGRANGIAN_ADVERSARIAL}: the Lagrangian game of adversarial BwK (Immorlica
 *       et al., FOCS 2019): EXP3 over arms against Hedge over resources on the payoff
 *       {@code r + sum_j lambda_j (1 - (T / B) c_j)}.</li>
 * </ul>
 */
public final class BanditsWithKnapsacks implements BanditPolicy {

    /** The policy. */
    public enum Method {
        /** primal-dual multiplicative weights over resources */
        PRIMAL_DUAL_BWK,
        /** an optimistic LP per round */
        UCB_BWK,
        /** EXP3 against Hedge on the Lagrangian */
        LAGRANGIAN_ADVERSARIAL
    }

    private final Method method;
    private final int k;
    private final int d;
    private final double budget;
    private final int horizon;
    private final SplittableRandom random;
    private final int[] pulls;
    private final double[] rewardSum;
    private final double[][] costSum;
    private final double[] resourceWeights;
    private final double[] armWeights;
    private final double[] dualWeights;
    private double[] lastProbs;
    private long t;

    private BanditsWithKnapsacks(Method method, int arms, int resources, double budget, int horizon, long seed) {
        this.method = Objects.requireNonNull(method, "method");
        this.k = arms;
        this.d = resources;
        this.budget = budget;
        this.horizon = horizon;
        this.random = new SplittableRandom(seed);
        this.pulls = new int[arms];
        this.rewardSum = new double[arms];
        this.costSum = new double[arms][resources];
        this.resourceWeights = new double[resources + 1];
        Arrays.fill(resourceWeights, 1);
        this.armWeights = new double[arms + 1];
        Arrays.fill(armWeights, 1);
        this.dualWeights = new double[resources + 1];
        Arrays.fill(dualWeights, 1);
    }

    /**
     * @param method the policy
     * @param arms real arms
     * @param resources resources
     * @param budget the budget of every resource
     * @param horizon the number of rounds
     * @param seed the seed
     * @return the policy
     */
    public static BanditsWithKnapsacks of(Method method, int arms, int resources, double budget, int horizon,
            long seed) {
        return new BanditsWithKnapsacks(method, arms, resources, budget, horizon, seed);
    }

    @Override
    public int arms() {
        return k + 1;
    }

    private double ucb(int a) {
        if (pulls[a] == 0) {
            return 1;
        }
        double m = rewardSum[a] / pulls[a];
        return Math.min(1, m + Math.sqrt(2 * Math.log(Math.max(2, t)) / pulls[a]));
    }

    private double lcb(int a, int j) {
        if (pulls[a] == 0) {
            return 0;
        }
        double m = costSum[a][j] / pulls[a];
        return Math.max(0, m - Math.sqrt(2 * Math.log(Math.max(2, t)) / pulls[a]));
    }

    @Override
    public int select() {
        t++;
        for (int a = 0; a < k; a++) {
            if (pulls[a] == 0 && method != Method.LAGRANGIAN_ADVERSARIAL) {
                return a;
            }
        }
        double rate = budget / horizon;
        return switch (method) {
            case PRIMAL_DUAL_BWK -> {
                double total = 0;
                for (double w : resourceWeights) {
                    total += w;
                }
                int best = k;
                double bestRatio = 0;
                for (int a = 0; a < k; a++) {
                    double price = resourceWeights[d] / total * rate;
                    for (int j = 0; j < d; j++) {
                        price += resourceWeights[j] / total * lcb(a, j);
                    }
                    double ratio = ucb(a) / Math.max(1e-9, price);
                    if (ratio > bestRatio) {
                        bestRatio = ratio;
                        best = a;
                    }
                }
                yield best;
            }
            case UCB_BWK -> {
                double[] c = new double[k];
                double[][] a = new double[d + 1][k];
                double[] b = new double[d + 1];
                for (int arm = 0; arm < k; arm++) {
                    c[arm] = ucb(arm);
                    for (int j = 0; j < d; j++) {
                        a[j][arm] = lcb(arm, j);
                    }
                    a[d][arm] = 1;
                }
                Arrays.fill(b, 0, d, rate);
                b[d] = 1;
                double[] x = LinearProgram.maximize(c, a, b).x();
                yield sample(x);
            }
            case LAGRANGIAN_ADVERSARIAL -> {
                double gamma = Math.min(1, Math.sqrt((k + 1) * Math.log(k + 1) / ((Math.E - 1) * horizon)));
                double total = 0;
                for (double w : armWeights) {
                    total += w;
                }
                lastProbs = new double[k + 1];
                for (int arm = 0; arm <= k; arm++) {
                    lastProbs[arm] = (1 - gamma) * armWeights[arm] / total + gamma / (k + 1);
                }
                double u = random.nextDouble();
                int pick = k;
                for (int arm = 0; arm <= k; arm++) {
                    u -= lastProbs[arm];
                    if (u <= 0) {
                        pick = arm;
                        break;
                    }
                }
                yield pick;
            }
        };
    }

    private int sample(double[] x) {
        double u = random.nextDouble();
        for (int a = 0; a < x.length; a++) {
            u -= x[a];
            if (u <= 0) {
                return a;
            }
        }
        return k;
    }

    @Override
    public void update(int arm, double reward) {
        observe(arm, reward, new double[d]);
    }

    /**
     * Records a pull's reward and resource consumption.
     *
     * @param arm the arm ({@code K} for the null arm)
     * @param reward its reward in {@code [0, 1]}
     * @param costs its consumption of every resource, each in {@code [0, 1]}
     */
    public void observe(int arm, double reward, double[] costs) {
        double rate = budget / horizon;
        if (arm < k) {
            pulls[arm]++;
            rewardSum[arm] += reward;
            for (int j = 0; j < d; j++) {
                costSum[arm][j] += costs[j];
            }
        }
        switch (method) {
            case PRIMAL_DUAL_BWK -> {
                double eps = Math.sqrt(Math.log(d + 1) / budget);
                for (int j = 0; j < d; j++) {
                    resourceWeights[j] *= Math.pow(1 + eps, costs[j]);
                }
                resourceWeights[d] *= Math.pow(1 + eps, rate);
                double max = Arrays.stream(resourceWeights).max().orElse(1);
                for (int j = 0; j <= d; j++) {
                    resourceWeights[j] /= max;
                }
            }
            case LAGRANGIAN_ADVERSARIAL -> {
                double scale = horizon / budget;
                double total = Arrays.stream(dualWeights).sum();
                double lagrangian = reward;
                for (int j = 0; j < d; j++) {
                    lagrangian += dualWeights[j] / total * (1 - scale * costs[j]);
                }
                double lo = 1 - scale;
                double normalised = (lagrangian - lo) / (1 + 1 - lo);
                double gamma = Math.min(1, Math.sqrt((k + 1) * Math.log(k + 1) / ((Math.E - 1) * horizon)));
                double estimate = normalised / lastProbs[arm];
                armWeights[arm] *= Math.exp(gamma * estimate / (k + 1));
                double maxW = Arrays.stream(armWeights).max().orElse(1);
                for (int a = 0; a <= k; a++) {
                    armWeights[a] /= maxW;
                }
                double eta = Math.sqrt(Math.log(d + 1) / horizon);
                for (int j = 0; j < d; j++) {
                    // the dual player minimises the Lagrangian: weight grows when a resource is overspent
                    dualWeights[j] *= Math.exp(-eta * (1 - scale * costs[j]) / scale);
                }
                double maxD = Arrays.stream(dualWeights).max().orElse(1);
                for (int j = 0; j <= d; j++) {
                    dualWeights[j] /= maxD;
                }
            }
            default -> {
            }
        }
    }

    /**
     * The LP benchmark per round: {@code max sum x_k r_k} with {@code sum x_k c_kj <= B / T}.
     *
     * @param arms the arms
     * @param budget the budget per resource
     * @param horizon the rounds
     * @return the best expected reward per round of any fixed arm distribution
     */
    public static double lpValue(KnapsackArms arms, double budget, int horizon) {
        int k = arms.arms();
        int d = arms.resources();
        double[][] a = new double[d + 1][k];
        double[] b = new double[d + 1];
        for (int arm = 0; arm < k; arm++) {
            for (int j = 0; j < d; j++) {
                a[j][arm] = arms.cost()[arm][j];
            }
            a[d][arm] = 1;
        }
        Arrays.fill(b, 0, d, budget / horizon);
        b[d] = 1;
        return LinearProgram.maximize(arms.reward(), a, b).objective();
    }

    /**
     * Plays a policy until the horizon or until a budget is exceeded.
     *
     * @param arms the arms
     * @param policy any policy over the real arms (indices {@code >= K} are the null arm)
     * @param budget the budget per resource
     * @param horizon the rounds
     * @param seed the seed of the rewards
     * @return the total reward
     */
    public static double play(KnapsackArms arms, BanditPolicy policy, double budget, int horizon, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = arms.resources();
        double[] spent = new double[d];
        double total = 0;
        for (int round = 0; round < horizon; round++) {
            int a = policy.select();
            double reward = 0;
            double[] costs = new double[d];
            if (a < arms.arms()) {
                reward = r.nextDouble() < arms.reward()[a] ? 1 : 0;
                for (int j = 0; j < d; j++) {
                    costs[j] = r.nextDouble() < arms.cost()[a][j] ? 1 : 0;
                }
            }
            boolean over = false;
            for (int j = 0; j < d; j++) {
                spent[j] += costs[j];
                over |= spent[j] > budget;
            }
            if (over) {
                break;
            }
            total += reward;
            if (policy instanceof BanditsWithKnapsacks bwk) {
                bwk.observe(a, reward, costs);
            } else if (a < arms.arms()) {
                policy.update(a, reward);
            }
        }
        return total;
    }

    @Override
    public String toString() {
        return method.name();
    }
}
