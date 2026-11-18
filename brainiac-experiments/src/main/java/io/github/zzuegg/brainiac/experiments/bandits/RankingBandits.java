package io.github.zzuegg.brainiac.experiments.bandits;

import io.github.zzuegg.brainiac.experimental.bandit.RankingBandits.Method;
import io.github.zzuegg.brainiac.experimental.bandit.RankingBandits.Run;
import io.github.zzuegg.brainiac.experimental.data.BanditSuites;
import io.github.zzuegg.brainiac.experimental.data.ClickLogs;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Wave 11 item 11.6.5: cascade-aware ranking bandits against item-wise linear bandits that
 * ignore the cascade, combinatorial UCB with naive semi-bandit feedback, offline unbiased
 * learning to rank from logged clicks, and random lists, under the cascade and the
 * position-based click models.
 */
public final class RankingBandits {

    private RankingBandits() {
    }

    /**
     * Regret and click-through over seeds.
     *
     * @param model the click model
     * @param items items
     * @param k list length
     * @param rounds rounds
     * @param seeds seeds
     * @return policy name to {@code {mean cumulative regret, click-through rate}}
     */
    public static Map<String, double[]> table(ClickLogs.ClickModel model, int items, int k, int rounds, int seeds) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (Method m : Method.values()) {
            double regret = 0;
            double ctr = 0;
            for (int s = 0; s < seeds; s++) {
                BanditSuites.ClickItems it = BanditSuites.clickItems(items, k, 5, model, 1, 1, 40 + s);
                Run r = io.github.zzuegg.brainiac.experimental.bandit.RankingBandits.run(it,
                        io.github.zzuegg.brainiac.experimental.bandit.RankingBandits.of(m, it, s).alpha(0.5), rounds,
                        60 + s);
                regret += r.regret();
                ctr += r.clickThrough();
            }
            out.put(m.toString(), new double[] {regret / seeds, ctr / seeds});
        }
        double regret = 0;
        double ctr = 0;
        for (int s = 0; s < seeds; s++) {
            BanditSuites.ClickItems it = BanditSuites.clickItems(items, k, 5, model, 1, 1, 40 + s);
            Run r = io.github.zzuegg.brainiac.experimental.bandit.RankingBandits.offlineUltr(it, rounds / 2, 200,
                    rounds - rounds / 2, 80 + s);
            // the logging half is played with random lists
            Run logging = io.github.zzuegg.brainiac.experimental.bandit.RankingBandits.run(it,
                    io.github.zzuegg.brainiac.experimental.bandit.RankingBandits.of(Method.RANDOM, it, s), rounds / 2,
                    90 + s);
            regret += r.regret() + logging.regret();
            ctr += (r.clickThrough() + logging.clickThrough()) / 2;
        }
        out.put("OFFLINE_ULTR (IPW, random logging)", new double[] {regret / seeds, ctr / seeds});
        return out;
    }

    /**
     * Prints both click models at a larger budget.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        for (ClickLogs.ClickModel m : ClickLogs.ClickModel.values()) {
            System.out.println(m + ": " + StochasticBandits.line(table(m, 200, 8, 50_000, 10)));
        }
    }
}
