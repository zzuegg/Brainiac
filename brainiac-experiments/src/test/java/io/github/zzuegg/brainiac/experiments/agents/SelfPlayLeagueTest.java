package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.rl.ConnectThree;
import io.github.zzuegg.brainiac.experimental.rl.EloLeague;
import io.github.zzuegg.brainiac.experimental.rl.GameEnvironment;
import io.github.zzuegg.brainiac.experimental.rl.GamePlayer;
import io.github.zzuegg.brainiac.experimental.rl.TicTacToe;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.Schedule;
import io.github.zzuegg.brainiac.rl.Tabular;
import org.junit.jupiter.api.Test;

class SelfPlayLeagueTest {

    @Test
    void gamesFollowTheRules() {
        TicTacToe t = TicTacToe.start().play(0).play(3).play(1).play(4).play(2);
        assertEquals(1, t.winner());
        assertTrue(t.legalMoves().isEmpty());
        ConnectThree c = ConnectThree.start().play(0).play(1).play(0).play(1).play(0);
        assertEquals(1, c.winner());
        assertEquals(0.5, EloLeague.expected(1200, 1200), 1e-12);
    }

    @Test
    void ticTacToeSelfPlayClimbsTheLeague() {
        SelfPlayLeague.Result r = SelfPlayLeague.run(TicTacToe::start, 6000, 4, 1);
        System.out.println("tic-tac-toe " + r);
        assertTrue(r.firstEloEnd() > r.firstEloStart() + 100, r.toString());
        assertTrue(r.secondEloEnd() > r.secondEloStart() + 100, r.toString());
        assertTrue(r.firstVsRandom() > 0.9, r.toString());
        assertTrue(r.secondVsRandom() > 0.7, r.toString());
    }

    @Test
    void connectThreeSelfPlayClimbsTheLeague() {
        SelfPlayLeague.Result r = SelfPlayLeague.run(ConnectThree::start, 4000, 4, 2);
        System.out.println("connect-3 " + r);
        assertTrue(r.firstEloEnd() > r.firstEloStart() + 100, r.toString());
        assertTrue(r.firstVsRandom() > 0.8, r.toString());
    }

    @Test
    void gameEnvironmentLetsCoreLearnersPlay() {
        GameEnvironment env = GameEnvironment.of(TicTacToe::start, GamePlayer.random(), false);
        Tabular q = Tabular.qLearning(9).learningRate(0.3).epsilon(Schedule.constant(0.1)).seed(1);
        double before = Rl.evaluate(q, env, 200, 3);
        q.fit(env, 30_000);
        double after = Rl.evaluate(q, env, 200, 3);
        System.out.println("Q-learning vs random " + before + " -> " + after);
        assertTrue(after > before + 0.3, before + " -> " + after);
    }
}
