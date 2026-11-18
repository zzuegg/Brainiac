package io.github.zzuegg.brainiac.experiments.markets;

import io.github.zzuegg.brainiac.experimental.data.MarketPaths;
import io.github.zzuegg.brainiac.experimental.markets.LearnedAuctions;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Learned auctions against known optimal mechanisms (wave 12, item 12.4.17): RegretNet and
 * RegretFormer against second price with and without a reserve, Myerson (one item) or separate
 * per-item Myerson, and Manelli-Vincent (one bidder, two items).
 */
public final class MechanismDesign {

    private MechanismDesign() {
    }

    /**
     * Trains and evaluates mechanisms on one setting.
     *
     * @param bidders bidders
     * @param items items
     * @param kinds the learned architectures to train
     * @param steps training steps
     * @param seed the seed
     * @return mechanism to {@code {revenue, mean regret, max regret, learned parameters}}
     */
    public static Map<String, double[]> auction(int bidders, int items, LearnedAuctions.Kind[] kinds, int steps,
            long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        double[][][] train = MarketPaths.privateValues(20000, bidders, items, rnd);
        double[][][] test = MarketPaths.privateValues(300, bidders, items, rnd);
        double[][][] large = MarketPaths.privateValues(100000, bidders, items, rnd);
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("second price (no reserve)", new double[] {LearnedAuctions.secondPrice(large, 0), 0, 0, 0});
        out.put(items == 1 ? "Myerson (reserve 1/2)" : "separate Myerson (reserve 1/2)",
                new double[] {LearnedAuctions.secondPrice(large, 0.5), 0, 0, 0});
        if (bidders == 1 && items == 2) {
            out.put("Manelli-Vincent", new double[] {LearnedAuctions.manelliVincent(large), 0, 0, 0});
        }
        for (LearnedAuctions.Kind kind : kinds) {
            LearnedAuctions net = new LearnedAuctions(kind, bidders, items,
                    kind == LearnedAuctions.Kind.REGRET_NET ? 32 : 16, seed);
            net.train(train, steps, 64, 4, 20, 3e-3f, seed + 1);
            LearnedAuctions.Result r = net.evaluate(test, 15, seed + 2);
            out.put(kind.name(), new double[] {r.revenue(), r.meanRegret(), r.maxRegret(), net.parameterCount()});
        }
        return out;
    }
}
