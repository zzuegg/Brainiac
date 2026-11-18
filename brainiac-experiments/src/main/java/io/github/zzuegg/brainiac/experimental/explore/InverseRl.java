package io.github.zzuegg.brainiac.experimental.explore;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.OfflineDataset;
import io.github.zzuegg.brainiac.experimental.rl.RlTensors;
import io.github.zzuegg.brainiac.experimental.rl.TabularMdp;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Inverse reinforcement learning beyond tabular linear MaxEnt IRL, on known tabular dynamics
 * ({@link TabularMdp}, rewards for entering states):
 *
 * <ul>
 *   <li>{@link Method#APPRENTICESHIP_PROJECTION}: apprenticeship learning by the projection method
 *       (Abbeel &amp; Ng, ICML 2004): alternate between the optimal policy for the reward
 *       {@code w . phi} and moving the running feature-expectation estimate towards the expert's,
 *       with {@code w} the remaining gap;</li>
 *   <li>{@link Method#BAYESIAN_IRL}: Bayesian IRL (Ramachandran &amp; Amir, IJCAI 2007): a
 *       Metropolis random walk over per-state rewards on a grid of step {@code delta}, under a
 *       Boltzmann-rational expert {@code P(a | s) ~ exp(alpha Q*(s, a))}; it returns the posterior
 *       mean (a per-state reward, so it does not transfer to new layouts);</li>
 *   <li>{@link Method#DEEP_MAXENT_IRL}: deep maximum-entropy IRL (Wulfmeier, Ondruska &amp; Posner,
 *       2015, <a href="https://arxiv.org/abs/1507.04888">arXiv:1507.04888</a>): the reward is an MLP
 *       of the state features (core {@link Nets#mlp}), trained with the MaxEnt gradient
 *       {@code dL/dr(s) = mu_expert(s) - E_pi[mu(s)]} back-propagated through the network.</li>
 * </ul>
 *
 * <p>{@link ObjectWorld} is Levine et al.'s object world (NeurIPS 2011): coloured objects on a grid,
 * a reward that is a nonlinear function of the distances to the nearest object of each colour,
 * and binary "within distance d of colour c" features. {@link #expectedValueDifference} scores a
 * learned reward by the true return lost by the policy that is optimal for it. The MaxEnt
 * soft value iteration is {@link TabularMdp#softQ}, as in the wave 6 {@code MaxEntIrl}.
 */
public final class InverseRl {

    private InverseRl() {
    }

    /** The IRL method. */
    public enum Method {
        /** feature-expectation matching by projection */
        APPRENTICESHIP_PROJECTION,
        /** Metropolis policy walk over per-state rewards */
        BAYESIAN_IRL,
        /** an MLP reward trained by the MaxEnt gradient */
        DEEP_MAXENT_IRL
    }

    /**
     * A learned reward.
     *
     * @param reward the reward of every state of the training world
     * @param transfer maps the feature matrix of another world to its rewards ({@code null} when
     *     the reward is per state and cannot transfer)
     */
    public record Fit(double[] reward, Function<double[][], double[]> transfer) {
    }

    /**
     * An object world.
     *
     * @param size the grid side
     * @param objectCells the cells holding objects
     * @param objectColours their colours (0 or 1)
     * @param features binary features {@code [states][2 * maxDistance]}
     * @param trueReward the true reward per state
     * @param mdp the dynamics (5 actions: stay, up, right, down, left; slip to a random action)
     */
    public record ObjectWorld(int size, int[] objectCells, int[] objectColours, double[][] features,
            double[] trueReward, TabularMdp mdp) {
    }

    /** Distances 1..5 for the binary features. */
    private static final int MAX_DISTANCE = 5;

    /**
     * Builds an object world: reward +1 within distance 3 of colour 0 and within 2 of colour 1,
     * -1 within 3 of colour 0 only, 0 elsewhere (Euclidean distances).
     *
     * @param size the grid side
     * @param objects the number of objects
     * @param slip the chance an action is replaced by a random one
     * @param horizon the episode length
     * @param seed the layout seed
     * @return the world (uniform start distribution)
     */
    public static ObjectWorld objectWorld(int size, int objects, double slip, int horizon, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = size * size;
        int[] cells = new int[objects];
        int[] colours = new int[objects];
        for (int k = 0; k < objects; k++) {
            cells[k] = r.nextInt(n);
            colours[k] = k % 2;
        }
        double[][] features = new double[n][2 * MAX_DISTANCE];
        double[] reward = new double[n];
        for (int s = 0; s < n; s++) {
            double[] nearest = {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY};
            for (int k = 0; k < objects; k++) {
                double dr = s / size - cells[k] / size;
                double dc = s % size - cells[k] % size;
                nearest[colours[k]] = Math.min(nearest[colours[k]], Math.sqrt(dr * dr + dc * dc));
            }
            for (int c = 0; c < 2; c++) {
                for (int d = 1; d <= MAX_DISTANCE; d++) {
                    features[s][c * MAX_DISTANCE + d - 1] = nearest[c] <= d ? 1 : 0;
                }
            }
            boolean near0 = nearest[0] <= 3;
            boolean near1 = nearest[1] <= 2;
            reward[s] = near0 && near1 ? 1 : near0 ? -1 : 0;
        }
        int[][] moves = {{0, 0}, {-1, 0}, {0, 1}, {1, 0}, {0, -1}};
        double[][][] p = new double[n][5][n];
        for (int s = 0; s < n; s++) {
            for (int a = 0; a < 5; a++) {
                for (int m = 0; m < 5; m++) {
                    double prob = (m == a ? 1 - slip : 0) + slip / 5;
                    int rr = Math.max(0, Math.min(size - 1, s / size + moves[m][0]));
                    int cc = Math.max(0, Math.min(size - 1, s % size + moves[m][1]));
                    p[s][a][rr * size + cc] += prob;
                }
            }
        }
        double[] start = new double[n];
        java.util.Arrays.fill(start, 1.0 / n);
        TabularMdp mdp = TabularMdp.of(p, reward, start, new boolean[n], horizon);
        return new ObjectWorld(size, cells, colours, features, reward, mdp);
    }

    /**
     * Expert demonstrations: the optimal policy for the true reward from the start distribution.
     *
     * @param world the world
     * @param trajectories the number of demonstrations
     * @param seed the seed
     * @return the demonstrations (one-hot observations)
     */
    public static OfflineDataset demonstrations(ObjectWorld world, int trajectories, long seed) {
        TabularMdp mdp = world.mdp();
        int[] pi = greedy(mdp.optimalQ(0.99, mdp.maxSteps()));
        SplittableRandom r = new SplittableRandom(seed);
        int n = mdp.states();
        double[] start = mdp.start();
        List<List<OfflineDataset.Step>> out = new ArrayList<>();
        for (int t = 0; t < trajectories; t++) {
            int s = sample(start, r);
            List<OfflineDataset.Step> traj = new ArrayList<>();
            for (int k = 0; k < mdp.maxSteps(); k++) {
                int a = pi[s];
                int s2 = sample(mdp.transition(s, a), r);
                traj.add(new OfflineDataset.Step(oneHot(n, s), new float[] {a}, (float) world.trueReward()[s2],
                        oneHot(n, s2), false));
                s = s2;
            }
            out.add(traj);
        }
        return OfflineDataset.of(out);
    }

    /**
     * Learns a reward.
     *
     * @param method the method
     * @param world the world (dynamics and features; the true reward is not used)
     * @param demos the demonstrations
     * @param iterations iterations (projection steps, Metropolis steps or gradient steps)
     * @param seed the seed
     * @return the learned reward
     */
    public static Fit fit(Method method, ObjectWorld world, OfflineDataset demos, int iterations, long seed) {
        Objects.requireNonNull(method, "method");
        return switch (method) {
            case APPRENTICESHIP_PROJECTION -> projection(world, demos, iterations);
            case BAYESIAN_IRL -> bayesian(world, demos, iterations, 0.25, 5, seed);
            case DEEP_MAXENT_IRL -> deepMaxEnt(world, demos, iterations, seed);
        };
    }

    /**
     * The expected value difference: the true return of the optimal policy minus the true return
     * of the policy that is optimal for {@code learned}, from the start distribution.
     *
     * @param world the world
     * @param learned a reward per state
     * @return the difference (0 is perfect)
     */
    public static double expectedValueDifference(ObjectWorld world, double[] learned) {
        TabularMdp mdp = world.mdp();
        int[] best = greedy(mdp.optimalQ(0.99, mdp.maxSteps()));
        int[] mine = greedy(mdp.withReward(learned).optimalQ(0.99, mdp.maxSteps()));
        return policyReturn(world, best) - policyReturn(world, mine);
    }

    /**
     * @param world the world
     * @param policy a deterministic policy (action per state)
     * @return its expected true return over the horizon from the start distribution
     */
    public static double policyReturn(ObjectWorld world, int[] policy) {
        TabularMdp mdp = world.mdp();
        int n = mdp.states();
        double[][] pi = new double[n][mdp.actions()];
        for (int s = 0; s < n; s++) {
            pi[s][policy[s]] = 1;
        }
        double[] visits = visitation(mdp, pi, false);
        double total = 0;
        for (int s = 0; s < n; s++) {
            total += visits[s] * world.trueReward()[s];
        }
        return total;
    }

    // ------------------------------------------------------------------ methods

    private static Fit projection(ObjectWorld world, OfflineDataset demos, int iterations) {
        double[][] phi = world.features();
        int k = phi[0].length;
        double[] muE = expertFeatures(demos, phi);
        TabularMdp mdp = world.mdp();
        double[] w = new double[k];
        double[] muBar = featureExpectations(world, greedy(mdp.withReward(new double[mdp.states()])
                .optimalQ(0.99, mdp.maxSteps())));
        for (int it = 0; it < iterations; it++) {
            for (int j = 0; j < k; j++) {
                w[j] = muE[j] - muBar[j];
            }
            double[] mu = featureExpectations(world, greedy(mdp.withReward(linear(phi, w)).optimalQ(0.99,
                    mdp.maxSteps())));
            double num = 0;
            double den = 0;
            for (int j = 0; j < k; j++) {
                num += (mu[j] - muBar[j]) * (muE[j] - muBar[j]);
                den += (mu[j] - muBar[j]) * (mu[j] - muBar[j]);
            }
            if (den < 1e-12) {
                break;
            }
            double step = Math.max(0, Math.min(1, num / den));
            for (int j = 0; j < k; j++) {
                muBar[j] += step * (mu[j] - muBar[j]);
            }
        }
        double[] weights = w.clone();
        return new Fit(linear(phi, weights), f -> linear(f, weights));
    }

    private static Fit bayesian(ObjectWorld world, OfflineDataset demos, int iterations, double delta,
            double alpha, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        TabularMdp mdp = world.mdp();
        int n = mdp.states();
        List<int[]> pairs = new ArrayList<>();
        for (OfflineDataset.Step s : demos.steps()) {
            pairs.add(new int[] {argmax(s.observation()), Math.round(s.action()[0])});
        }
        double[] reward = new double[n];
        double logLik = logLikelihood(mdp, reward, pairs, alpha);
        double[] sum = new double[n];
        int kept = 0;
        for (int it = 0; it < iterations; it++) {
            int s = r.nextInt(n);
            double old = reward[s];
            double proposal = old + (r.nextBoolean() ? delta : -delta);
            if (proposal > 1 + 1e-9 || proposal < -1 - 1e-9) {
                continue;
            }
            reward[s] = proposal;
            double ll = logLikelihood(mdp, reward, pairs, alpha);
            if (Math.log(r.nextDouble()) < ll - logLik) {
                logLik = ll;
            } else {
                reward[s] = old;
            }
            if (it >= iterations / 2) {
                for (int u = 0; u < n; u++) {
                    sum[u] += reward[u];
                }
                kept++;
            }
        }
        for (int u = 0; u < n; u++) {
            sum[u] /= Math.max(1, kept);
        }
        return new Fit(sum, null);
    }

    private static double logLikelihood(TabularMdp mdp, double[] reward, List<int[]> pairs, double alpha) {
        double[][] q = mdp.withReward(reward).optimalQ(0.99, mdp.maxSteps());
        double ll = 0;
        for (int[] p : pairs) {
            double[] row = q[p[0]];
            double max = Double.NEGATIVE_INFINITY;
            for (double v : row) {
                max = Math.max(max, alpha * v);
            }
            double z = 0;
            for (double v : row) {
                z += Math.exp(alpha * v - max);
            }
            ll += alpha * row[p[1]] - max - Math.log(z);
        }
        return ll;
    }

    private static Fit deepMaxEnt(ObjectWorld world, OfflineDataset demos, int iterations, long seed) {
        TabularMdp mdp = world.mdp();
        int n = mdp.states();
        double[][] phi = world.features();
        int k = phi[0].length;
        Model net = Nets.mlp(k, 1, seed, 32, 32);
        Optimizer opt = Optimizer.adam(1e-2f);
        Tensor x = features(phi);
        double[] muE = new double[n];
        List<List<OfflineDataset.Step>> trajs = demos.trajectories();
        for (List<OfflineDataset.Step> t : trajs) {
            muE[argmax(t.getFirst().observation())]++;
            for (OfflineDataset.Step s : t) {
                muE[argmax(s.nextObservation())]++;
            }
        }
        for (int s = 0; s < n; s++) {
            muE[s] /= trajs.size();
        }
        for (int it = 0; it < iterations; it++) {
            Var r = net.forward(Var.of(x), true);
            double[] reward = r.value().toDoubleArray();
            double[][] q = mdp.withReward(reward).softQ(0.99, mdp.maxSteps(), 1);
            double[] mu = visitation(mdp, softmax(q), true);
            float[] g = new float[n];
            for (int s = 0; s < n; s++) {
                g[s] = (float) -(muE[s] - mu[s]) / mdp.maxSteps();
            }
            Var loss = r.mul(Tensor.of(Shape.of(n, 1), g)).sum();
            RlTensors.zeroGrad(net.parameters());
            loss.backward();
            opt.step(net.parameters());
        }
        Function<double[][], double[]> transfer = f -> net.forward(Var.of(features(f)), false).value()
                .toDoubleArray();
        return new Fit(transfer.apply(phi), transfer);
    }

    // ------------------------------------------------------------------ helpers

    private static Tensor features(double[][] phi) {
        int n = phi.length;
        int k = phi[0].length;
        float[] data = new float[n * k];
        for (int s = 0; s < n; s++) {
            for (int j = 0; j < k; j++) {
                data[s * k + j] = (float) phi[s][j];
            }
        }
        return Tensor.of(Shape.of(n, k), data).to(DType.FLOAT32);
    }

    private static double[][] softmax(double[][] q) {
        double[][] pi = new double[q.length][];
        for (int s = 0; s < q.length; s++) {
            double max = Double.NEGATIVE_INFINITY;
            for (double v : q[s]) {
                max = Math.max(max, v);
            }
            pi[s] = new double[q[s].length];
            double z = 0;
            for (int a = 0; a < q[s].length; a++) {
                pi[s][a] = Math.exp(q[s][a] - max);
                z += pi[s][a];
            }
            for (int a = 0; a < q[s].length; a++) {
                pi[s][a] /= z;
            }
        }
        return pi;
    }

    /**
     * Expected state counts over the horizon (start state included when {@code withStart}, as in
     * the MaxEnt feature counts; otherwise the states entered, which earn rewards).
     */
    private static double[] visitation(TabularMdp mdp, double[][] pi, boolean withStart) {
        int n = mdp.states();
        double[][][] p = new double[n][mdp.actions()][];
        for (int s = 0; s < n; s++) {
            for (int a = 0; a < mdp.actions(); a++) {
                p[s][a] = mdp.transition(s, a);
            }
        }
        double[] d = mdp.start();
        double[] mu = withStart ? d.clone() : new double[n];
        for (int t = 0; t < mdp.maxSteps(); t++) {
            double[] next = new double[n];
            for (int s = 0; s < n; s++) {
                if (d[s] == 0 || mdp.terminal(s)) {
                    continue;
                }
                for (int a = 0; a < pi[s].length; a++) {
                    double w = d[s] * pi[s][a];
                    if (w == 0) {
                        continue;
                    }
                    double[] row = p[s][a];
                    for (int u = 0; u < n; u++) {
                        next[u] += w * row[u];
                    }
                }
            }
            for (int s = 0; s < n; s++) {
                mu[s] += next[s];
            }
            d = next;
        }
        return mu;
    }

    private static double[] featureExpectations(ObjectWorld world, int[] policy) {
        TabularMdp mdp = world.mdp();
        int n = mdp.states();
        double[][] pi = new double[n][mdp.actions()];
        for (int s = 0; s < n; s++) {
            pi[s][policy[s]] = 1;
        }
        double[] visits = visitation(mdp, pi, true);
        double[][] phi = world.features();
        double[] mu = new double[phi[0].length];
        for (int s = 0; s < n; s++) {
            for (int j = 0; j < mu.length; j++) {
                mu[j] += visits[s] * phi[s][j];
            }
        }
        return mu;
    }

    private static double[] expertFeatures(OfflineDataset demos, double[][] phi) {
        double[] mu = new double[phi[0].length];
        List<List<OfflineDataset.Step>> trajs = demos.trajectories();
        for (List<OfflineDataset.Step> t : trajs) {
            add(mu, phi[argmax(t.getFirst().observation())]);
            for (OfflineDataset.Step s : t) {
                add(mu, phi[argmax(s.nextObservation())]);
            }
        }
        for (int j = 0; j < mu.length; j++) {
            mu[j] /= trajs.size();
        }
        return mu;
    }

    private static void add(double[] mu, double[] f) {
        for (int j = 0; j < mu.length; j++) {
            mu[j] += f[j];
        }
    }

    private static double[] linear(double[][] phi, double[] w) {
        double[] r = new double[phi.length];
        for (int s = 0; s < phi.length; s++) {
            for (int j = 0; j < w.length; j++) {
                r[s] += w[j] * phi[s][j];
            }
        }
        return r;
    }

    /**
     * @param q action values {@code [states][actions]}
     * @return the greedy action of every state
     */
    public static int[] greedy(double[][] q) {
        int[] pi = new int[q.length];
        for (int s = 0; s < q.length; s++) {
            int best = 0;
            for (int a = 1; a < q[s].length; a++) {
                if (q[s][a] > q[s][best] + 1e-12) {
                    best = a;
                }
            }
            pi[s] = best;
        }
        return pi;
    }

    private static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }

    private static float[] oneHot(int n, int s) {
        float[] o = new float[n];
        o[s] = 1;
        return o;
    }

    private static int sample(double[] p, SplittableRandom r) {
        double u = r.nextDouble();
        for (int k = 0; k < p.length; k++) {
            u -= p[k];
            if (u <= 0) {
                return k;
            }
        }
        return p.length - 1;
    }
}
