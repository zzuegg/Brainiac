package io.github.zzuegg.brainiac.experiments.operations;

import io.github.zzuegg.brainiac.experimental.bandit.StochasticBandits;
import io.github.zzuegg.brainiac.experimental.data.OperationsSims;
import io.github.zzuegg.brainiac.experimental.ops.DynamicPricing;
import io.github.zzuegg.brainiac.experimental.ops.Inventory;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Ppo;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Wave 12 items 12.6.6-12.6.7: inventory control and dynamic pricing.
 *
 * <ul>
 *   <li>{@link #newsvendor} (12.6.6): the out-of-sample newsvendor cost of feature-based ERM (linear
 *       and random Fourier features) against predict-then-optimise (least squares plus a normal
 *       quantile), relative to the oracle that orders the true conditional quantile;</li>
 *   <li>{@link #inventory} (12.6.6): the average cost of the best base-stock and (s, S) policies and
 *       core {@link Ppo} on lost-sales inventory with lead times, relative to the DP-optimal policy;</li>
 *   <li>{@link #pricing} (12.6.7): revenue regret of certainty-equivalent pricing, parametric Thompson
 *       sampling and (with inventory) the LP-based Thompson sampling against UCB1 and Bernoulli
 *       Thompson sampling over the price grid;</li>
 *   <li>{@link #contextualPricing} (12.6.7): the logistic valuation model against LinUCB over a price
 *       grid.</li>
 * </ul>
 */
public final class InventoryAndPricing {

    private InventoryAndPricing() {
    }

    /**
     * 12.6.6: the newsvendor with features.
     *
     * @param n training samples (the test set has the same size)
     * @param tau the critical ratio
     * @param heavyTailed Pareto rather than log-normal demand noise
     * @param seed the seed
     * @return method to the mean test cost relative to the oracle ({@code 1} is optimal); the row
     *     {@code "oracle"} holds the oracle's absolute cost
     */
    public static Map<String, Double> newsvendor(int n, double tau, boolean heavyTailed, long seed) {
        OperationsSims.Demand train = OperationsSims.newsvendor(n, 4, heavyTailed, seed);
        OperationsSims.Demand test = OperationsSims.newsvendor(n, 4, heavyTailed, seed + 1);
        double oracle = 0;
        for (int i = 0; i < test.size(); i++) {
            oracle += Inventory.newsvendorCost(test.quantile(i, tau), test.demand()[i], tau);
        }
        Map<String, Inventory.QuantityModel> models = new LinkedHashMap<>();
        models.put("ERM linear", Inventory.newsvendorErm(train.features(), train.demand(), tau, 0, 1, 60, seed));
        models.put("ERM random features", Inventory.newsvendorErm(train.features(), train.demand(), tau, 100, 1.5,
                60, seed));
        models.put("predict-then-optimise", Inventory.predictThenOptimise(train.features(), train.demand(), tau));
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("oracle", oracle / test.size());
        for (Map.Entry<String, Inventory.QuantityModel> e : models.entrySet()) {
            double cost = 0;
            for (int i = 0; i < test.size(); i++) {
                cost += Inventory.newsvendorCost(e.getValue().order(test.features()[i]), test.demand()[i], tau);
            }
            out.put(e.getKey(), cost / oracle);
        }
        return out;
    }

    /**
     * 12.6.6: lost-sales inventory.
     *
     * @param leadTime the lead time (the DP oracle needs at most 3)
     * @param penalty the lost-sales penalty per unit (holding cost 1)
     * @param periods simulated periods per evaluation
     * @param ppoSteps PPO environment steps ({@code 0} to skip)
     * @param seed the seed
     * @return policy to {@code {average cost, cost relative to the reference}}; the reference is the
     *     DP optimum when the lead time is at most 3, else the best base-stock policy
     */
    public static Map<String, double[]> inventory(int leadTime, double penalty, int periods, int ppoSteps,
            long seed) {
        OperationsSims.Inventory problem = OperationsSims.lostSales(leadTime, penalty);
        Map<String, Double> costs = new LinkedHashMap<>();
        if (leadTime <= 3) {
            costs.put("DP optimal", Inventory.averageCost(problem, Inventory.optimal(problem, 0.99, 400), periods,
                    seed));
        }
        double[] bs = Inventory.bestBaseStock(problem, periods, seed);
        costs.put("best base-stock", bs[1]);
        costs.put("best (s, S)", Inventory.bestSS(problem, periods, seed)[2]);
        costs.put("order the mean demand", Inventory.averageCost(problem, (onHand, pipe) ->
                (int) Math.round(problem.meanDemand()), periods, seed));
        if (ppoSteps > 0) {
            int in = leadTime + 1;
            Ppo ppo = Ppo.of(Nets.mlp(in, problem.maxOrder() + 1, seed, 32, 32), Nets.mlp(in, 1, seed + 1, 32, 32),
                    Space.discrete(problem.maxOrder() + 1)).rolloutSteps(512).epochs(4).minibatchSize(64)
                    .discount(0.99f).learningRate(3e-3f).seed(seed);
            ppo.fit(new Inventory.Env(problem, 100), ppoSteps);
            costs.put("PPO", Inventory.averageCost(problem, (onHand, pipe) -> {
                float[] o = new float[in];
                o[0] = onHand / (float) problem.maxStock();
                for (int k = 0; k < pipe.length; k++) {
                    o[k + 1] = pipe[k] / (float) problem.maxStock();
                }
                return Math.max(0, Math.min(problem.maxOrder(), Math.round(ppo.act(o, false)[0])));
            }, periods, seed));
        }
        double ref = costs.getOrDefault("DP optimal", bs[1]);
        Map<String, double[]> out = new LinkedHashMap<>();
        costs.forEach((k, v) -> out.put(k, new double[] {v, v / ref}));
        return out;
    }

    /**
     * 12.6.7: pricing with demand learning.
     *
     * @param model the demand curve
     * @param prices price points
     * @param horizon periods
     * @param inventory stock ({@code < 0} for unlimited)
     * @param seeds runs
     * @return policy to {@code {mean regret, mean stock-out period (-1 if never)}}
     */
    public static Map<String, double[]> pricing(OperationsSims.DemandModel model, int prices, int horizon,
            double inventory, int seeds) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (DynamicPricing.Method m : DynamicPricing.Method.values()) {
            if (m == DynamicPricing.Method.THOMPSON_INVENTORY_LP && inventory < 0) {
                continue;
            }
            double regret = 0;
            double stockOut = 0;
            for (int s = 0; s < seeds; s++) {
                DynamicPricing.Run run = DynamicPricing.run(m, OperationsSims.pricing(model, prices, s), horizon,
                        inventory, 1000 + s);
                regret += run.regret();
                stockOut += run.stockOutPeriod();
            }
            out.put(m.name(), new double[] {regret / seeds, stockOut / seeds});
        }
        for (StochasticBandits.Method m : new StochasticBandits.Method[] {StochasticBandits.Method.UCB1,
            StochasticBandits.Method.THOMPSON_BETA}) {
            double regret = 0;
            double stockOut = 0;
            for (int s = 0; s < seeds; s++) {
                DynamicPricing.Run run = DynamicPricing.runBandit(StochasticBandits.of(m, prices, horizon, s),
                        OperationsSims.pricing(model, prices, s), horizon, inventory, 1000 + s);
                regret += run.regret();
                stockOut += run.stockOutPeriod();
            }
            out.put(m.name(), new double[] {regret / seeds, stockOut / seeds});
        }
        return out;
    }

    /**
     * 12.6.7: contextual pricing.
     *
     * @param dims context dimensions
     * @param horizon periods
     * @param seeds runs
     * @return policy to the mean regret against the per-context clairvoyant price
     */
    public static Map<String, Double> contextualPricing(int dims, int horizon, int seeds) {
        Map<String, Double> out = new LinkedHashMap<>();
        for (DynamicPricing.Contextual c : DynamicPricing.Contextual.values()) {
            double regret = 0;
            for (int s = 0; s < seeds; s++) {
                regret += DynamicPricing.contextual(c, OperationsSims.valuations(dims, s), horizon, 12, 100 + s)
                        .regret();
            }
            out.put(c.name(), regret / seeds);
        }
        return out;
    }
}
