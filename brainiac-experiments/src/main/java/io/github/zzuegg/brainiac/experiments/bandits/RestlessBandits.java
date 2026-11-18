package io.github.zzuegg.brainiac.experiments.bandits;

import io.github.zzuegg.brainiac.experimental.bandit.BanditPolicy;
import io.github.zzuegg.brainiac.experimental.bandit.BanditsWithKnapsacks;
import io.github.zzuegg.brainiac.experimental.bandit.StochasticBandits;
import io.github.zzuegg.brainiac.experimental.data.OperationsSims;
import io.github.zzuegg.brainiac.experimental.recsys.BudgetPacer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Wave 12 items 12.6.15-12.6.16: restless bandits and bandits with knapsacks.
 *
 * <ul>
 *   <li>{@link #restless} (12.6.15): the mean reward per step of the Whittle index policy (exact,
 *       Q-learned and neural), the LP-relaxation policy, the myopic, round-robin and random
 *       baselines and the exact joint DP (small instances), against the LP upper bound;</li>
 *   <li>{@link #knapsacks} (12.6.16): the reward of primal-dual BwK, UCB-BwK and the Lagrangian
 *       adversarial-robust variant against UCB1 and Thompson sampling that ignore the budgets, UCB1
 *       throttled by {@code BudgetPacer}'s participation coin, and the LP benchmark
 *       {@code T * LP(B / T)}.</li>
 * </ul>
 */
public final class RestlessBandits {

    private RestlessBandits() {
    }

    /**
     * 12.6.15.
     *
     * @param arms arms
     * @param budget arms activated per step
     * @param steps simulated steps per policy
     * @param learnEpisodes training episodes of the neural index ({@code 0} to skip)
     * @param seed the seed
     * @return policy to the mean reward per step; the row {@code "LP upper bound"} holds the bound
     */
    public static Map<String, Double> restless(int arms, int budget, int steps, int learnEpisodes, long seed) {
        List<OperationsSims.RestlessArm> a = OperationsSims.restlessArms(arms, seed);
        double gamma = 0.95;
        Map<String, Supplier<io.github.zzuegg.brainiac.experimental.bandit.RestlessBandits.Policy>> policies =
                new LinkedHashMap<>();
        policies.put("Whittle index (exact)",
                () -> io.github.zzuegg.brainiac.experimental.bandit.RestlessBandits.whittle(a, gamma));
        policies.put("LP relaxation",
                () -> io.github.zzuegg.brainiac.experimental.bandit.RestlessBandits.lpRelaxation(a, budget));
        policies.put("Whittle Q-learning", () -> io.github.zzuegg.brainiac.experimental.bandit.RestlessBandits
                .whittleQLearning(arms, gamma, 0.1, seed));
        if (learnEpisodes > 0) {
            policies.put("neural index", () -> io.github.zzuegg.brainiac.experimental.bandit.RestlessBandits
                    .neuralIndex(a, learnEpisodes, gamma, seed));
        }
        if (arms <= 6) {
            policies.put("joint DP (optimal, discounted)", () -> io.github.zzuegg.brainiac.experimental.bandit
                    .RestlessBandits.optimal(a, budget, 0.99));
        }
        policies.put("myopic", () -> io.github.zzuegg.brainiac.experimental.bandit.RestlessBandits.myopic(a));
        policies.put("round robin",
                () -> io.github.zzuegg.brainiac.experimental.bandit.RestlessBandits.roundRobin());
        policies.put("random", () -> io.github.zzuegg.brainiac.experimental.bandit.RestlessBandits.random(seed));
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("LP upper bound", io.github.zzuegg.brainiac.experimental.bandit.RestlessBandits.lpBound(a, budget));
        for (Map.Entry<String, Supplier<io.github.zzuegg.brainiac.experimental.bandit.RestlessBandits.Policy>> e
                : policies.entrySet()) {
            out.put(e.getKey(), io.github.zzuegg.brainiac.experimental.bandit.RestlessBandits.simulate(a, budget,
                    e.getValue().get(), steps, seed + 1));
        }
        return out;
    }

    /** UCB1 whose participation in each round is decided by {@code BudgetPacer}'s throttling coin. */
    private static BanditPolicy throttled(BanditPolicy inner, double probability, long seed) {
        BudgetPacer.Policy coin = BudgetPacer.throttle(probability, seed);
        return new BanditPolicy() {
            private int t;

            @Override
            public int arms() {
                return inner.arms() + 1;
            }

            @Override
            public int select() {
                return coin.bid(1, t++) > 0 ? inner.select() : inner.arms();
            }

            @Override
            public void update(int arm, double reward) {
                inner.update(arm, reward);
            }
        };
    }

    /**
     * 12.6.16.
     *
     * @param arms arms ({@code 0} for the inventory-pricing arms of 12.6.7)
     * @param resources resources (random arms only)
     * @param budgetShare the budget as a share of the horizon
     * @param horizon rounds
     * @param seeds runs
     * @return policy to the mean total reward; the row {@code "LP benchmark"} holds {@code T * LP(B / T)}
     */
    public static Map<String, Double> knapsacks(int arms, int resources, double budgetShare, int horizon, int seeds) {
        Map<String, Double> out = new LinkedHashMap<>();
        double budget = budgetShare * horizon;
        double lp = 0;
        for (int s = 0; s < seeds; s++) {
            OperationsSims.KnapsackArms a = arms == 0
                    ? OperationsSims.pricingArms(OperationsSims.pricing(OperationsSims.DemandModel.LOGIT, 8, s))
                    : OperationsSims.knapsackArms(arms, resources, s);
            int k = a.arms();
            lp += horizon * BanditsWithKnapsacks.lpValue(a, budget, horizon) / seeds;
            for (BanditsWithKnapsacks.Method m : BanditsWithKnapsacks.Method.values()) {
                out.merge(m.name(), BanditsWithKnapsacks.play(a, BanditsWithKnapsacks.of(m, k, a.resources(), budget,
                        horizon, s), budget, horizon, 100 + s) / seeds, Double::sum);
            }
            out.merge("UCB1 (ignores budget)", BanditsWithKnapsacks.play(a,
                    StochasticBandits.of(StochasticBandits.Method.UCB1, k, horizon, s), budget, horizon, 100 + s)
                    / seeds, Double::sum);
            out.merge("Thompson (ignores budget)", BanditsWithKnapsacks.play(a,
                    StochasticBandits.of(StochasticBandits.Method.THOMPSON_BETA, k, horizon, s), budget, horizon,
                    100 + s) / seeds, Double::sum);
            double maxCost = 0;
            for (double[] c : a.cost()) {
                for (double v : c) {
                    maxCost = Math.max(maxCost, v);
                }
            }
            double p = Math.min(1, budgetShare / Math.max(1e-9, maxCost));
            out.merge("UCB1 + BudgetPacer throttle", BanditsWithKnapsacks.play(a, throttled(
                    StochasticBandits.of(StochasticBandits.Method.UCB1, k, horizon, s), p, s), budget, horizon,
                    100 + s) / seeds, Double::sum);
        }
        Map<String, Double> ordered = new LinkedHashMap<>();
        ordered.put("LP benchmark", lp);
        ordered.putAll(out);
        return ordered;
    }
}
