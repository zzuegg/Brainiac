package io.github.zzuegg.brainiac.experiments.recsys;

import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.recsys.ContextualBandit;
import io.github.zzuegg.brainiac.experimental.recsys.OffPolicyEstimators;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Learning from logged and online bandit feedback (W7 2.7 off-policy evaluation and learning,
 * W7 2.8 contextual bandits) in a {@link SyntheticRecsys#banditWorld}, whose true expected rewards
 * are known.
 * <ul>
 *   <li>{@link #evaluation}: the RMSE of estimates of a target policy's value from logs of a
 *       different logging policy, over repeated logs, for the naive average of logged rewards, the
 *       direct method (a per-action ridge reward model), IPS, SNIPS, doubly robust and Switch;</li>
 *   <li>{@link #learning}: the true value of a policy learned from logs by counterfactual risk
 *       minimisation, against the logging policy and the policy that is greedy on the reward
 *       model;</li>
 *   <li>{@link #online}: the cumulative pseudo-regret of LinUCB, linear Thompson sampling and
 *       neural-linear against epsilon-greedy, context-free UCB1 and a greedy neural model.</li>
 * </ul>
 */
public final class LoggedBanditFeedback {

    private static final int DIM = 5;
    private static final int ACTIONS = 6;

    private LoggedBanditFeedback() {
    }

    private static double[] target(SyntheticRecsys.BanditWorld world, double[] x) {
        double[] p = new double[ACTIONS];
        double z = 0;
        for (int a = 0; a < ACTIONS; a++) {
            p[a] = Math.exp(6 * world.expected(x, a));
            z += p[a];
        }
        for (int a = 0; a < ACTIONS; a++) {
            p[a] /= z;
        }
        return p;
    }

    private static double trueValue(SyntheticRecsys.BanditWorld world,
            java.util.function.Function<double[], double[]> policy, int samples) {
        double v = 0;
        for (int i = 0; i < samples; i++) {
            double[] x = world.context();
            double[] p = policy.apply(x);
            for (int a = 0; a < ACTIONS; a++) {
                v += p[a] * world.expected(x, a);
            }
        }
        return v / samples;
    }

    /**
     * Off-policy evaluation.
     *
     * @param nonlinear whether rewards are non-linear (the reward model is then misspecified)
     * @param rounds rounds per log
     * @param inverseTemperature the logging policy's greediness (larger diverges more from uniform)
     * @param repeats independent logs
     * @param seed the seed
     * @return estimator name to RMSE against the true value; "true value" holds the value itself
     */
    public static Map<String, Double> evaluation(boolean nonlinear, int rounds, double inverseTemperature,
            int repeats, long seed) {
        SyntheticRecsys.BanditWorld world = SyntheticRecsys.banditWorld(DIM, ACTIONS, nonlinear, 1.0, seed);
        double truth = trueValue(world, x -> target(world, x), 20000);
        String[] names = {"naive", "DM", "IPS", "SNIPS", "DR", "Switch"};
        double[] se = new double[names.length];
        for (int r = 0; r < repeats; r++) {
            SyntheticRecsys.BanditLog log = world.log(rounds, inverseTemperature);
            double[][] pi = new double[rounds][];
            for (int i = 0; i < rounds; i++) {
                pi[i] = target(world, log.contexts()[i]);
            }
            double[][] q = OffPolicyEstimators.RewardModel.fit(log, ACTIONS, 1).predict(log);
            double[] est = {OffPolicyEstimators.naive(log), OffPolicyEstimators.directMethod(log, pi, q),
                OffPolicyEstimators.ips(log, pi, Double.POSITIVE_INFINITY), OffPolicyEstimators.snips(log, pi),
                OffPolicyEstimators.doublyRobust(log, pi, q), OffPolicyEstimators.switchDr(log, pi, q, 5)};
            for (int k = 0; k < est.length; k++) {
                se[k] += (est[k] - truth) * (est[k] - truth);
            }
        }
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("true value", truth);
        for (int k = 0; k < names.length; k++) {
            out.put(names[k], Math.sqrt(se[k] / repeats));
        }
        return out;
    }

    /**
     * Off-policy learning.
     *
     * @param rounds logged rounds
     * @param seed the seed
     * @return policy name to its true value
     */
    public static Map<String, Double> learning(int rounds, long seed) {
        SyntheticRecsys.BanditWorld world = SyntheticRecsys.banditWorld(DIM, ACTIONS, false, 1.0, seed);
        double beta = 1;
        SyntheticRecsys.BanditLog log = world.log(rounds, beta);
        double[][] w = OffPolicyEstimators.learnPolicy(log, ACTIONS, 20, 0.5, 300, seed);
        OffPolicyEstimators.RewardModel model = OffPolicyEstimators.RewardModel.fit(log, ACTIONS, 1);
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("uniform", trueValue(world, x -> uniform(), 20000));
        out.put("logging", trueValue(world, x -> world.loggingPolicy(x, beta), 20000));
        out.put("greedy on reward model", trueValue(world, x -> {
            double[] p = new double[ACTIONS];
            int best = 0;
            for (int a = 1; a < ACTIONS; a++) {
                best = model.predict(x, a) > model.predict(x, best) ? a : best;
            }
            p[best] = 1;
            return p;
        }, 20000));
        out.put("CRM", trueValue(world, x -> OffPolicyEstimators.policy(w, x), 20000));
        out.put("oracle", trueValue(world, x -> {
            double[] p = new double[ACTIONS];
            int best = 0;
            for (int a = 1; a < ACTIONS; a++) {
                best = world.expected(x, a) > world.expected(x, best) ? a : best;
            }
            p[best] = 1;
            return p;
        }, 20000));
        return out;
    }

    private static double[] uniform() {
        double[] p = new double[ACTIONS];
        java.util.Arrays.fill(p, 1.0 / ACTIONS);
        return p;
    }

    /**
     * Online contextual bandits.
     *
     * @param nonlinear whether rewards are non-linear
     * @param rounds rounds
     * @param seed the seed
     * @return policy name to cumulative pseudo-regret (expected reward of the best arm minus that
     *     of the chosen arm, summed)
     */
    public static Map<String, Double> online(boolean nonlinear, int rounds, long seed) {
        Map<String, Supplier<ContextualBandit>> policies = new LinkedHashMap<>();
        policies.put("UCB1 (no context)", () -> ContextualBandit.ucb1(ACTIONS));
        policies.put("epsilon-greedy", () -> ContextualBandit.epsilonGreedy(DIM, ACTIONS, 0.1, 1, seed));
        policies.put("LinUCB", () -> ContextualBandit.linUcb(DIM, ACTIONS, 1.0, 1));
        policies.put("linear TS", () -> ContextualBandit.linearThompson(DIM, ACTIONS, 0.3, 1, seed));
        policies.put("neural greedy", () -> ContextualBandit.neuralGreedy(DIM, ACTIONS, 16, 500, seed));
        policies.put("neural-linear", () -> ContextualBandit.neuralLinear(DIM, ACTIONS, 16, 500, 0.3, seed));
        SyntheticRecsys.BanditWorld contextSource = SyntheticRecsys.banditWorld(DIM, ACTIONS, nonlinear, 1.0, seed);
        double[][] contexts = new double[rounds][];
        for (int t = 0; t < rounds; t++) {
            contexts[t] = contextSource.context();
        }
        Map<String, Double> out = new LinkedHashMap<>();
        policies.forEach((name, make) -> {
            SyntheticRecsys.BanditWorld world = SyntheticRecsys.banditWorld(DIM, ACTIONS, nonlinear, 1.0, seed);
            ContextualBandit policy = make.get();
            double regret = 0;
            for (double[] x : contexts) {
                int a = policy.select(x);
                double best = 0;
                for (int b = 0; b < ACTIONS; b++) {
                    best = Math.max(best, world.expected(x, b));
                }
                regret += best - world.expected(x, a);
                policy.update(x, a, world.pull(x, a));
            }
            out.put(name, regret);
        });
        return out;
    }
}
