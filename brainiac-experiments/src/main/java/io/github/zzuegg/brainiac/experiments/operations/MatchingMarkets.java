package io.github.zzuegg.brainiac.experiments.operations;

import io.github.zzuegg.brainiac.experimental.data.OperationsSims;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Wave 12 item 12.6.11: bandit learning in two-sided matching markets. Every scheme of
 * {@code experimental.ops.MatchingMarkets} (centralised ETC and UCB, decentralised CA-UCB, and the
 * greedy, random and conflict-ignoring UCB baselines) is run on serial-dictatorship and random
 * markets; reported are the mean and largest per-player stable regret against the player-pessimal
 * stable matching and the share of rounds with an unstable matching.
 */
public final class MatchingMarkets {

    private MatchingMarkets() {
    }

    /**
     * Runs every scheme.
     *
     * @param players players
     * @param arms arms
     * @param gap the preference gap
     * @param structure the market structure
     * @param horizon rounds
     * @param seeds runs
     * @return scheme to {@code {mean per-player regret, mean largest per-player regret, unstable share}}
     */
    public static Map<String, double[]> run(int players, int arms, double gap,
            OperationsSims.MarketStructure structure, int horizon, int seeds) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (var m : io.github.zzuegg.brainiac.experimental.ops.MatchingMarkets.Method.values()) {
            double mean = 0;
            double max = 0;
            double unstable = 0;
            for (int s = 0; s < seeds; s++) {
                OperationsSims.Market market = OperationsSims.market(players, arms, gap, structure, s);
                var run = io.github.zzuegg.brainiac.experimental.ops.MatchingMarkets.run(market, m, horizon, 50,
                        100 + s);
                double sum = 0;
                for (double v : run.regret()) {
                    sum += v;
                }
                mean += sum / players;
                max += run.maxRegret();
                unstable += run.unstableShare();
            }
            out.put(m.name(), new double[] {mean / seeds, max / seeds, unstable / seeds});
        }
        return out;
    }
}
