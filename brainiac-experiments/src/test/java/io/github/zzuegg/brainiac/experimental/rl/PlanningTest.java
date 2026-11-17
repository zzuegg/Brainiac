package io.github.zzuegg.brainiac.experimental.rl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Nets;
import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

class PlanningTest {

    private static int[] match(TwoPlayerGame start, TwoPlayerGame.Player p, TwoPlayerGame.Player other, int games) {
        int[] wdl = new int[3];
        for (int g = 0; g < games; g++) {
            int r = g % 2 == 0 ? TwoPlayerGame.playOut(start, p, other) : -TwoPlayerGame.playOut(start, other, p);
            wdl[r > 0 ? 0 : r == 0 ? 1 : 2]++;
        }
        return wdl;
    }

    @Test
    void gamesDetectWinsAndGravity() {
        TwoPlayerGame g = NInARow.ticTacToe().play(0).play(3).play(1).play(4).play(2);
        assertTrue(g.over());
        assertEquals(1, g.winner());
        TwoPlayerGame c = NInARow.connectThree();
        for (int i = 0; i < 4; i++) {
            c = c.play(0);
        }
        assertTrue(!c.legal()[0] && c.legal()[1]);
    }

    @Test
    void mctsFindsTheWinningMove() {
        // X has 0 and 1, O has 3 and 4; X to move must take 2
        TwoPlayerGame g = NInARow.ticTacToe().play(0).play(3).play(1).play(4);
        Mcts<TwoPlayerGame> m = Mcts.of(TwoPlayerGame.searchModel(9, TwoPlayerGame.randomRollouts(1)))
                .simulations(200);
        assertEquals(2, m.search(g, new SplittableRandom(1)).best());
    }

    @Test
    void pureMctsBeatsRandom() {
        Mcts<TwoPlayerGame> m = Mcts.of(TwoPlayerGame.searchModel(9, TwoPlayerGame.randomRollouts(1)))
                .simulations(100);
        SplittableRandom r = new SplittableRandom(2);
        int[] wdl = match(NInARow.ticTacToe(), g -> m.search(g, r).best(), TwoPlayerGame.Player.random(3), 20);
        assertTrue(wdl[2] <= 1 && wdl[0] >= 14, java.util.Arrays.toString(wdl));
    }

    @Test
    void alphaZeroLearnsTicTacToeFromSelfPlay() {
        AlphaZero az = AlphaZero.of(NInARow.ticTacToe(), Nets.mlp(18, 10, Activation.RELU, 1, 64, 64))
                .simulations(48).epochs(8).seed(1);
        int[] untrained = match(NInARow.ticTacToe(), az.player(0), TwoPlayerGame.Player.random(5), 200);
        az.train(20, 25);
        int[] trained = match(NInARow.ticTacToe(), az.player(0), TwoPlayerGame.Player.random(5), 200);
        System.out.println("AlphaZero raw policy vs random (w/d/l): untrained " + java.util.Arrays.toString(untrained)
                + " trained " + java.util.Arrays.toString(trained) + " loss " + az.lastLoss());
        assertTrue(trained[2] * 2 < untrained[2] && trained[0] > untrained[0], java.util.Arrays.toString(trained));
    }

    @Test
    void muZeroLitePlansWithALearnedModel() {
        io.github.zzuegg.brainiac.rl.GridWorld env = io.github.zzuegg.brainiac.rl.GridWorld.of("S...", "....", "...G")
                .maxSteps(20);
        double random = io.github.zzuegg.brainiac.rl.Rl.evaluate(
                io.github.zzuegg.brainiac.rl.Agent.random(env.actionSpace(), 1), env, 10, 3);
        MuZeroLite mz = MuZeroLite.of(12, 4, RlNetworkFactory.mlp(Activation.RELU, 48), 12).simulations(12).trainEvery(2).warmup(600).seed(2);
        long t = System.nanoTime();
        mz.fit(env, 4000);
        double trained = io.github.zzuegg.brainiac.rl.Rl.evaluate(mz, env, 3, 3);
        System.out.println("MuZeroLite random " + random + " trained " + trained + " loss " + mz.lastLoss() + " "
                + (System.nanoTime() - t) / 1e9 + "s");
        assertTrue(trained > 0.9, "return " + trained);
    }
}
