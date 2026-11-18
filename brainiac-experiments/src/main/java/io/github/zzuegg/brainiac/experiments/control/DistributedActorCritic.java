package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.AsyncActorCritic;
import io.github.zzuegg.brainiac.experimental.rl.VTraceLearner;
import io.github.zzuegg.brainiac.rl.CartPole;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Ppo;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * W6 3.2 and 3.3: distributed actor-critics on CartPole.
 * <ul>
 *   <li>{@link #a3c}: A3C with worker threads against the core synchronous A2C running the
 *       same number of environments, at equal environment steps: greedy return and wall-clock
 *       time.</li>
 *   <li>{@link #lag}: actors that lag the learner by a number of updates, as IMPALA's
 *       decoupled actors do, with V-trace, without correction (plain A2C) and with the
 *       untruncated one-step importance weight.</li>
 *   <li>{@link #retrace}: off-policy evaluation of {@code Q^pi} on a random MDP from uniform
 *       behaviour data: Retrace(lambda) against uncorrected n-step returns and one-step
 *       expected SARSA, as error to the exact {@code Q^pi} after a few passes over the data.</li>
 * </ul>
 *
 * <p>Papers: Mnih et al., <a href="https://arxiv.org/abs/1602.01783">arXiv:1602.01783</a>;
 * Espeholt et al., <a href="https://arxiv.org/abs/1802.01561">arXiv:1802.01561</a>; Munos et al.,
 * <a href="https://arxiv.org/abs/1606.02647">arXiv:1606.02647</a>.
 */
public final class DistributedActorCritic {

    /**
     * The outcome of one run.
     *
     * @param greedyReturn greedy return after training
     * @param millis wall-clock training time
     */
    public record Outcome(double greedyReturn, long millis) {
    }

    private DistributedActorCritic() {
    }

    /**
     * A3C against synchronous A2C.
     *
     * @param workers worker threads (A3C) and environments (A2C)
     * @param steps total environment steps
     * @param seed the seed
     * @return "A3C" and "A2C (core)" to their outcomes
     */
    public static Map<String, Outcome> a3c(int workers, int steps, long seed) {
        Map<String, Outcome> out = new LinkedHashMap<>();
        Space actions = Space.discrete(2);
        long t0 = System.nanoTime();
        AsyncActorCritic a3c = AsyncActorCritic.of(() -> Nets.mlp(4, 2, seed, 32), () -> Nets.mlp(4, 1, seed + 1, 32),
                actions).nSteps(16).optimizer(io.github.zzuegg.brainiac.train.Optimizer.adam(1e-3f)).seed(seed);
        a3c.fit(() -> new CartPole(200), workers, steps, RlListener.NONE);
        long t1 = System.nanoTime();
        out.put("A3C", new Outcome(Rl.evaluate(a3c, new CartPole(200), 5, seed), (t1 - t0) / 1_000_000));
        Ppo a2c = Ppo.a2c(Nets.mlp(4, 2, seed, 32), Nets.mlp(4, 1, seed + 1, 32), actions).seed(seed);
        long t2 = System.nanoTime();
        a2c.fit(() -> new CartPole(200), workers, steps, RlListener.NONE);
        long t3 = System.nanoTime();
        out.put("A2C (core)", new Outcome(Rl.evaluate(a2c, new CartPole(200), 5, seed), (t3 - t2) / 1_000_000));
        return out;
    }

    /**
     * Greedy return against actor lag for each correction.
     *
     * @param lags the lags, in learner updates
     * @param steps environment steps per run
     * @param seeds how many seeds to average
     * @return "correction, lag n" to the mean greedy return
     */
    public static Map<String, Double> lag(int[] lags, int steps, int seeds) {
        Map<String, Double> out = new LinkedHashMap<>();
        for (VTraceLearner.Correction c : VTraceLearner.Correction.values()) {
            for (int lag : lags) {
                double sum = 0;
                for (int s = 0; s < seeds; s++) {
                    long seed = 5 + s;
                    VTraceLearner l = VTraceLearner.of(() -> Nets.mlp(4, 2, seed, 32), Nets.mlp(4, 1, seed + 1, 32),
                            Space.discrete(2)).correction(c).lag(lag).unroll(16).seed(seed);
                    l.fit(() -> new CartPole(200), 4, steps, RlListener.NONE);
                    sum += Rl.evaluate(l, new CartPole(200), 5, seed);
                }
                out.put(c + ", lag " + lag, sum / seeds);
            }
        }
        return out;
    }

    /**
     * Off-policy evaluation with Retrace on a random MDP.
     *
     * @param states the number of states
     * @param sweeps passes over the behaviour data
     * @param seed the seed
     * @return method to root-mean-square error against the exact {@code Q^pi}
     */
    public static Map<String, Double> retrace(int states, int sweeps, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        int actions = 2;
        double gamma = 0.9;
        int[][] next = new int[states][actions];
        double[][] reward = new double[states][actions];
        for (int s = 0; s < states; s++) {
            for (int a = 0; a < actions; a++) {
                next[s][a] = random.nextInt(states);
                reward[s][a] = random.nextDouble(-1, 1);
            }
        }
        double[][] pi = new double[states][actions];
        for (int s = 0; s < states; s++) {
            int favourite = random.nextInt(actions);
            for (int a = 0; a < actions; a++) {
                pi[s][a] = a == favourite ? 0.9 : 0.1 / (actions - 1);
            }
        }
        double[][] exact = new double[states][actions];
        for (int it = 0; it < 500; it++) {
            double[][] q = new double[states][actions];
            for (int s = 0; s < states; s++) {
                for (int a = 0; a < actions; a++) {
                    int n = next[s][a];
                    double v = 0;
                    for (int b = 0; b < actions; b++) {
                        v += pi[n][b] * exact[n][b];
                    }
                    q[s][a] = reward[s][a] + gamma * v;
                }
            }
            exact = q;
        }
        // behaviour data: uniformly random actions, restarting from a random state every 16 steps
        int length = 2048;
        int[] st = new int[length];
        int[] ac = new int[length];
        int[] nx = new int[length];
        for (int t = 0; t < length; t++) {
            st[t] = t % 16 == 0 ? random.nextInt(states) : nx[t - 1];
            ac[t] = random.nextInt(actions);
            nx[t] = next[st[t]][ac[t]];
        }
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("Retrace(1)", learn(st, ac, nx, reward, pi, states, actions, gamma, 1.0, true, sweeps, exact));
        out.put("uncorrected n-step", learn(st, ac, nx, reward, pi, states, actions, gamma, 1.0, false, sweeps, exact));
        out.put("1-step expected SARSA", learn(st, ac, nx, reward, pi, states, actions, gamma, 0.0, true, sweeps, exact));
        return out;
    }

    private static double learn(int[] st, int[] ac, int[] nx, double[][] reward, double[][] pi, int states, int actions,
            double gamma, double lambda, boolean corrected, int sweeps, double[][] exact) {
        double[][] q = new double[states][actions];
        boolean[][] seen = new boolean[states][actions];
        int window = 8;
        double mu = 1.0 / actions;
        for (int sweep = 0; sweep < sweeps; sweep++) {
            for (int from = 0; from + window <= ac.length; from += window) {
                double[] qa = new double[window];
                double[] en = new double[window];
                double[] ratio = new double[window];
                double[] r = new double[window];
                double[] dones = new double[window];
                for (int t = 0; t < window; t++) {
                    int s = st[from + t];
                    int a = ac[from + t];
                    int n = nx[from + t];
                    seen[s][a] = true;
                    qa[t] = q[s][a];
                    double e = 0;
                    for (int b = 0; b < actions; b++) {
                        e += pi[n][b] * q[n][b];
                    }
                    en[t] = e;
                    ratio[t] = pi[s][a] / mu;
                    r[t] = reward[s][a];
                }
                double[] target;
                if (corrected) {
                    target = VTraceLearner.retrace(qa, en, ratio, r, dones, gamma, lambda);
                } else {
                    // plain n-step returns along the behaviour's actions, bootstrapped at the window's end
                    target = new double[window];
                    double g = en[window - 1];
                    for (int t = window - 1; t >= 0; t--) {
                        g = r[t] + gamma * g;
                        target[t] = g;
                    }
                }
                for (int t = 0; t < window; t++) {
                    int s = st[from + t];
                    int a = ac[from + t];
                    q[s][a] += 0.1 * (target[t] - q[s][a]);
                }
            }
        }
        double err = 0;
        int count = 0;
        for (int s = 0; s < states; s++) {
            for (int a = 0; a < actions; a++) {
                if (seen[s][a]) {
                    err += (q[s][a] - exact[s][a]) * (q[s][a] - exact[s][a]);
                    count++;
                }
            }
        }
        return Math.sqrt(err / Math.max(1, count));
    }
}
