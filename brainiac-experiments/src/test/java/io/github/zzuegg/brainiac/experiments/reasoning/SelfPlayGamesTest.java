package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Wave 6, 6.13: SPIRAL self-play with role-conditioned advantages on text TicTacToe. */
class SelfPlayGamesTest {

    @Test
    void selfPlayLearnsTheGameButForgetsArithmetic() {
        SelfPlayGames.Result r = SelfPlayGames.run(600, 32, 1);
        StringBuilder sb = new StringBuilder();
        for (SelfPlayGames.Player p : r.players()) {
            sb.append(String.format(" %s Elo %.0f vsRandom %.2f vsMinimax %.2f add %.2f->%.2f;", p.mode(), p.elo(),
                    p.winVsRandom(), p.drawVsMinimax(), p.arithmeticBefore(), p.arithmeticAfter()));
        }
        System.out.printf("TicTacToe league (random %.0f, minimax %.0f):%s%n", r.randomElo(), r.minimaxElo(), sb);
        SelfPlayGames.Player spiral = r.players().get(0);
        assertTrue(spiral.elo() > r.randomElo() + 50, "self-play beats random in the league");
        assertTrue(spiral.winVsRandom() > 0.6, "self-play wins against random: " + spiral.winVsRandom());
    }
}
