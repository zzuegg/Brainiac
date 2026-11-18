package io.github.zzuegg.brainiac.experiments.bandits;

import io.github.zzuegg.brainiac.experimental.bandit.AdversarialBandits;
import io.github.zzuegg.brainiac.experimental.bandit.BanditPolicy;
import io.github.zzuegg.brainiac.experimental.bandit.BanditRuns;
import io.github.zzuegg.brainiac.experimental.bandit.StochasticBandits.Method;
import io.github.zzuegg.brainiac.experimental.data.BanditSuites;
import io.github.zzuegg.brainiac.experimental.recsys.ContextualBandit;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.LongFunction;

/**
 * Wave 11 items 11.6.1-11.6.2: regret of bandit policies at equal pulls.
 *
 * <ul>
 *   <li>{@link #stochastic} (11.6.1): KL-UCB, Beta Thompson sampling, MOSS and IMED against
 *       UCB1, epsilon-greedy and the existing {@code ContextualBandit} UCB1, with the
 *       Lai-Robbins lower bound, on Bernoulli arms with low means and small gaps.</li>
 *   <li>{@link #regimes} (11.6.2): EXP3, EXP3-IX, Tsallis-INF, {@code BanditMixer}'s EXP3 and
 *       sliding-window UCB against KL-UCB and Thompson sampling on a stochastic instance, a
 *       switching instance and an adaptive adversary that punishes the empirical leader.</li>
 * </ul>
 */
public final class StochasticBandits {

    private StochasticBandits() {
    }

    private static BanditPolicy contextualUcb1(int arms) {
        ContextualBandit c = ContextualBandit.ucb1(arms);
        double[] none = new double[0];
        return new BanditPolicy() {
            @Override
            public int arms() {
                return arms;
            }

            @Override
            public int select() {
                return c.select(none);
            }

            @Override
            public void update(int arm, double reward) {
                c.update(none, arm, reward);
            }
        };
    }

    private static LongFunction<BanditPolicy> stochastic(
            Method m, int arms, int horizon) {
        return s -> io.github.zzuegg.brainiac.experimental.bandit.StochasticBandits.of(m, arms, horizon, s);
    }

    /**
     * 11.6.1: regret at the horizon of every stochastic policy, plus the Lai-Robbins bound
     * (as a row with zero standard error).
     *
     * @param instance Bernoulli arms
     * @param means their means (for the bound)
     * @param horizon pulls
     * @param seeds seeds
     * @return row name to {@code {mean regret, standard error}}
     */
    public static Map<String, double[]> stochastic(BanditSuites.Instance instance, double[] means, int horizon,
            int seeds) {
        int k = instance.arms();
        Map<String, double[]> out = new LinkedHashMap<>();
        for (Method m : Method.values()) {
            if (m == Method.THOMPSON_GAUSSIAN) {
                continue;
            }
            out.put(m.toString(), BanditRuns.meanRegret(instance, stochastic(m, k, horizon), horizon, seeds));
        }
        out.put("ContextualBandit UCB1", BanditRuns.meanRegret(instance, s -> contextualUcb1(k), horizon, seeds));
        out.put("Lai-Robbins bound", new double[] {BanditRuns.laiRobbins(means, horizon), 0});
        return out;
    }

    /** The reward regimes of 11.6.2. */
    public enum Regime {
        /** i.i.d. Bernoulli arms */
        STOCHASTIC,
        /** the best arm rotates every {@code horizon / 5} pulls */
        SWITCHING,
        /** the empirical leader pays nothing */
        ADAPTIVE_ADVERSARY
    }

    /**
     * @param regime the regime
     * @param arms arms
     * @param horizon pulls
     * @return the instance
     */
    public static BanditSuites.Instance instance(Regime regime, int arms, int horizon) {
        double[] means = new double[arms];
        for (int i = 0; i < arms; i++) {
            means[i] = i == 0 ? 0.6 : 0.45;
        }
        return switch (regime) {
            case STOCHASTIC -> BanditSuites.bernoulli(means);
            case SWITCHING -> BanditSuites.switching(Math.max(1, horizon / 5), means);
            case ADAPTIVE_ADVERSARY -> BanditSuites.adaptiveAdversary(arms);
        };
    }

    /**
     * 11.6.2: regret against the best fixed arm in each regime.
     *
     * @param regime the regime
     * @param arms arms
     * @param horizon pulls
     * @param seeds seeds
     * @return row name to {@code {mean regret, standard error}}
     */
    public static Map<String, double[]> regimes(Regime regime, int arms, int horizon, int seeds) {
        BanditSuites.Instance inst = instance(regime, arms, horizon);
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("KL_UCB", BanditRuns.meanRegret(inst, stochastic(Method.KL_UCB, arms, horizon), horizon,
                seeds));
        out.put("THOMPSON_BETA", BanditRuns.meanRegret(inst, stochastic(Method.THOMPSON_BETA, arms,
                horizon), horizon, seeds));
        for (AdversarialBandits.Method m : AdversarialBandits.Method.values()) {
            out.put(m.toString(), BanditRuns.meanRegret(inst,
                    s -> AdversarialBandits.of(m, arms, horizon, s).window(Math.max(50, horizon / 10)), horizon, seeds));
        }
        out.put("BanditMixer EXP3", BanditRuns.meanRegret(inst, s -> AdversarialBandits.banditMixer(arms, s), horizon,
                seeds));
        return out;
    }

    /**
     * @param table a table
     * @return one line {@code name mean+-se; ...}
     */
    public static String line(Map<String, double[]> table) {
        StringBuilder b = new StringBuilder();
        table.forEach((k, v) -> b.append(String.format("%s %.1f+-%.1f; ", k, v[0], v[1])));
        return b.toString();
    }

    /**
     * Prints the tables at a larger budget.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        int h = 20_000;
        BanditSuites.Instance low = BanditSuites.lowMeanBernoulli(10, 0.05, 0.02);
        System.out.println(line(stochastic(low, low.means(0), h, 50)));
        for (Regime r : Regime.values()) {
            System.out.println(r + ": " + line(regimes(r, 5, h, 50)));
        }
    }
}
