package io.github.zzuegg.brainiac.experimental.bandit;

import io.github.zzuegg.brainiac.experimental.data.BanditSuites;

import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.function.LongFunction;

/**
 * Runs bandit policies on {@link BanditSuites.Instance}s and measures regret: pseudo-regret
 * from the arm means on stochastic instances, regret against the best fixed arm in hindsight
 * (from the full reward vectors) on adversarial ones, and the Lai-Robbins lower bound.
 */
public final class BanditRuns {

    private BanditRuns() {
    }

    /**
     * One run's regret.
     *
     * @param checkpoints the rounds at which regret was recorded
     * @param regret the cumulative regret at each checkpoint
     * @param pulls pulls per arm at the end
     */
    public record Run(int[] checkpoints, double[] regret, int[] pulls) {

        /** @return the regret at the horizon */
        public double finalRegret() {
            return regret[regret.length - 1];
        }
    }

    /**
     * Plays a policy for {@code horizon} rounds.
     *
     * @param instance the bandit
     * @param policy the policy (fresh)
     * @param horizon the rounds
     * @param checkpoints rounds at which to record the regret, increasing, the last at most {@code horizon}
     * @param seed the seed of the rewards
     * @return the run
     */
    public static Run run(BanditSuites.Instance instance, BanditPolicy policy, int horizon, int[] checkpoints,
            long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        int k = instance.arms();
        int[] pulls = new int[k];
        double[] sums = new double[k];
        double[] armTotals = new double[k];
        double collected = 0;
        double[] regret = new double[checkpoints.length];
        int next = 0;
        for (int t = 0; t < horizon; t++) {
            double[] r = instance.rewards(t, pulls, sums, random);
            int a = policy.select();
            policy.update(a, r[a]);
            pulls[a]++;
            sums[a] += r[a];
            if (instance.stochastic()) {
                double[] m = instance.means(t);
                for (int i = 0; i < k; i++) {
                    armTotals[i] += m[i];
                }
                collected += m[a];
            } else {
                for (int i = 0; i < k; i++) {
                    armTotals[i] += r[i];
                }
                collected += r[a];
            }
            while (next < checkpoints.length && t + 1 == checkpoints[next]) {
                regret[next++] = Arrays.stream(armTotals).max().orElse(0) - collected;
            }
        }
        return new Run(checkpoints.clone(), regret, pulls);
    }

    /**
     * Mean and standard error of the final regret over seeds.
     *
     * @param instance the bandit
     * @param policies builds a fresh policy for a seed
     * @param horizon the rounds
     * @param seeds the number of seeds
     * @return {@code {mean, standard error}}
     */
    public static double[] meanRegret(BanditSuites.Instance instance, LongFunction<BanditPolicy> policies,
            int horizon, int seeds) {
        double[] x = new double[seeds];
        for (int s = 0; s < seeds; s++) {
            x[s] = run(instance, policies.apply(s + 1), horizon, new int[] {horizon}, 1000 + s).finalRegret();
        }
        double mean = Arrays.stream(x).average().orElse(0);
        double var = 0;
        for (double v : x) {
            var += (v - mean) * (v - mean);
        }
        var /= Math.max(1, seeds - 1);
        return new double[] {mean, Math.sqrt(var / seeds)};
    }

    /**
     * The Lai-Robbins asymptotic lower bound on pseudo-regret for Bernoulli arms:
     * {@code sum over suboptimal arms of gap / KL(mu_a, mu*) * log T}.
     *
     * @param means the arm means
     * @param horizon the horizon
     * @return the bound
     */
    public static double laiRobbins(double[] means, int horizon) {
        double best = Arrays.stream(means).max().orElse(0);
        double c = 0;
        for (double m : means) {
            if (m < best) {
                c += (best - m) / BanditMath.klBernoulli(m, best);
            }
        }
        return c * Math.log(horizon);
    }
}
