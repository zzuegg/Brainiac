package io.github.zzuegg.brainiac.experiments.operations;

import io.github.zzuegg.brainiac.experimental.data.OperationsSims;
import io.github.zzuegg.brainiac.experimental.ops.MatchingMarkets.Method;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class MatchingMarketsTest {

    @Test
    void stableRegret() {
        Map<String, double[]> serial = MatchingMarkets.run(5, 5, 0.1, OperationsSims.MarketStructure.SERIAL_DICTATORSHIP,
                3_000, 3);
        Map<String, double[]> random = MatchingMarkets.run(5, 5, 0.1, OperationsSims.MarketStructure.RANDOM, 3_000, 3);
        System.out.printf("12.6.11 5x5 markets regret mean/max/unstable share, serial dictatorship: %s | random: %s%n",
                InventoryAndPricingTest.table(serial), InventoryAndPricingTest.table(random));
        double ucb = serial.get(Method.CENTRALIZED_UCB.name())[1];
        assertTrue(ucb < serial.get(Method.RANDOM.name())[1], "centralised UCB beats random matching");
        assertTrue(ucb < serial.get(Method.INDEPENDENT_UCB.name())[1], "and UCB that ignores conflicts");
    }

    @Test
    void deferredAcceptanceIsStable() {
        OperationsSims.Market m = OperationsSims.market(6, 6, 0.1, OperationsSims.MarketStructure.RANDOM, 3);
        int[] players = io.github.zzuegg.brainiac.experimental.ops.MatchingMarkets.deferredAcceptance(m.means(),
                m.armRanks(), true);
        int[] arms = io.github.zzuegg.brainiac.experimental.ops.MatchingMarkets.deferredAcceptance(m.means(),
                m.armRanks(), false);
        assertTrue(io.github.zzuegg.brainiac.experimental.ops.MatchingMarkets.stable(players, m.means(), m.armRanks()));
        assertTrue(io.github.zzuegg.brainiac.experimental.ops.MatchingMarkets.stable(arms, m.means(), m.armRanks()));
        for (int i = 0; i < players.length; i++) {
            assertTrue(m.means()[i][players[i]] >= m.means()[i][arms[i]] - 1e-12, "player-proposing is player-optimal");
        }
    }
}
