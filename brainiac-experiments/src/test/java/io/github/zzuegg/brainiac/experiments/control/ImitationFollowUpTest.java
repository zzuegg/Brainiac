package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class ImitationFollowUpTest {

    @Test
    void absorbingStatesFixGailOnCartPoleFromOneDemonstration() {
        Map<String, Double> r = GoalsImitationAndTeams.imitationFollowUpCartPole(new int[] {1}, 30_000, 1);
        StringBuilder b = new StringBuilder();
        r.forEach((k, v) -> b.append(String.format("%s %.1f, ", k, v)));
        System.out.printf("W10 4.13 cart-pole imitation, 30k steps: %s%n", b);
        assertTrue(r.get("1 demos: GAIL+GP+absorbing") > r.get("1 demos: random"), b.toString());
    }

    @Test
    void airlRewardTransfersToRotatedMoves() {
        Map<String, Double> r = GoalsImitationAndTeams.imitationTransfer(5, 10_000, 1);
        StringBuilder b = new StringBuilder();
        r.forEach((k, v) -> b.append(String.format("%s %.2f, ", k, v)));
        System.out.printf("W10 4.13 transfer to rotated moves, 10k steps: %s%n", b);
        assertTrue(r.get("expert policy copied") < r.get("oracle (true reward re-solved)"), b.toString());
        for (double v : r.values()) {
            assertTrue(Double.isFinite(v), b.toString());
        }
    }
}
