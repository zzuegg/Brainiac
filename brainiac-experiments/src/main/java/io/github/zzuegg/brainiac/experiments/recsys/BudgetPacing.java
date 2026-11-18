package io.github.zzuegg.brainiac.experiments.recsys;

import io.github.zzuegg.brainiac.experimental.data.ClickLogs;
import io.github.zzuegg.brainiac.experimental.recsys.BudgetPacer;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Budget pacing (9.1.9) on {@link ClickLogs#auctions}, a day of second-price auctions whose
 * competition peaks in the working day. The budget is a share of what unconstrained truthful
 * bidding would spend over the whole day. Policies: truthful bidding until the budget is gone,
 * a fixed throttle at the budget share, PID throttling and dual-gradient bid shading, each scored
 * as a fraction of the hindsight optimum ({@link BudgetPacer#hindsight}).
 */
public final class BudgetPacing {

    private BudgetPacing() {
    }

    /**
     * @param auctions auctions in the day, such as 20,000
     * @param budgetShare the budget as a share of unconstrained truthful spend, such as 0.2
     * @param seed the seed
     * @return policy name to {@code {value / hindsight value, spend / budget, depletion time
     *     (fraction of the day)}}; the entry {@code "hindsight"} holds {@code {1, 1, 1}}
     */
    public static Map<String, double[]> run(int auctions, double budgetShare, long seed) {
        ClickLogs.AuctionDay day = ClickLogs.auctions(auctions, seed);
        double unconstrained = BudgetPacer.simulate(day.value(), day.competing(), Double.MAX_VALUE,
                BudgetPacer.truthful()).spend();
        double budget = budgetShare * unconstrained;
        double best = BudgetPacer.hindsight(day.value(), day.competing(), budget);
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("hindsight", new double[] {1, 1, 1});
        Map<String, BudgetPacer.Policy> policies = new LinkedHashMap<>();
        policies.put("truthful until exhausted", BudgetPacer.truthful());
        policies.put("fixed throttle", BudgetPacer.throttle(budgetShare, seed));
        policies.put("PID throttle", BudgetPacer.pid(budget, auctions, Math.max(1, auctions / 200), 0.5, 0.05, 0.1,
                seed));
        policies.put("dual bid shading", BudgetPacer.dual(budget, auctions, 0.01));
        policies.forEach((name, p) -> {
            BudgetPacer.Outcome o = BudgetPacer.simulate(day.value(), day.competing(), budget, p);
            out.put(name, new double[] {o.value() / best, o.spend() / budget, o.depletion()});
        });
        return out;
    }

    /**
     * Checks the greedy hindsight optimum against the simplex solution on a small day.
     *
     * @param auctions auctions, a few hundred at most
     * @param budgetShare the budget share
     * @param seed the seed
     * @return {@code {greedy optimum, simplex optimum}}
     */
    public static double[] hindsightCheck(int auctions, double budgetShare, long seed) {
        ClickLogs.AuctionDay day = ClickLogs.auctions(auctions, seed);
        double unconstrained = BudgetPacer.simulate(day.value(), day.competing(), Double.MAX_VALUE,
                BudgetPacer.truthful()).spend();
        double budget = budgetShare * unconstrained;
        return new double[] {BudgetPacer.hindsight(day.value(), day.competing(), budget),
            BudgetPacer.hindsightLp(day.value(), day.competing(), budget)};
    }
}
