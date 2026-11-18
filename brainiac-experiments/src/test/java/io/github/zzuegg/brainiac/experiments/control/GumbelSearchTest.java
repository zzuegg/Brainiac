package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.rl.NInARow;
import org.junit.jupiter.api.Test;

import java.util.Map;

class GumbelSearchTest {

    @Test
    void gumbelSearchHoldsUpAtTinyBudgets() {
        Map<Integer, GumbelSearch.Outcome> r = GumbelSearch.run(NInARow.ticTacToe(), new int[] {2, 4, 8, 16, 32}, 400, 1);
        StringBuilder b = new StringBuilder();
        r.forEach((n, o) -> b.append(String.format(" n=%d %.2f (Elo %+.0f, vs random %.2f/%.2f)", n, o.gumbelVsPuct(),
                o.eloDifference(), o.gumbelVsRandom(), o.puctVsRandom())));
        System.out.printf("W6 3.18 tic-tac-toe Gumbel vs PUCT score (Elo, Gumbel/PUCT vs random):%s%n", b);
        assertTrue(r.get(2).gumbelVsPuct() >= 0.45, r.toString());
        assertTrue(r.get(32).gumbelVsRandom() > r.get(2).gumbelVsRandom() + 0.1, r.toString());
    }
}
