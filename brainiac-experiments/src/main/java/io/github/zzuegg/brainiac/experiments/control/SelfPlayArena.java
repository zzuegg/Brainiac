package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.AlphaZero;
import io.github.zzuegg.brainiac.experimental.rl.Mcts;
import io.github.zzuegg.brainiac.experimental.rl.NInARow;
import io.github.zzuegg.brainiac.experimental.rl.TwoPlayerGame;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Nets;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * AlphaZero learns a board game from self-play, then plays a round robin: the raw policy
 * network (no search), the network with a small MCTS budget, pure MCTS with random
 * rollouts at the same budget, and a random player. The expected order shows the AlphaZero
 * thesis: a learned prior and value make a small search far stronger than the same
 * search with random playouts.
 *
 * <p>Paper: Silver et al., <a href="https://arxiv.org/abs/1712.01815">arXiv:1712.01815</a>.
 */
public final class SelfPlayArena {

    private SelfPlayArena() {
    }

    /**
     * Plays a match, alternating who moves first.
     *
     * @param start the opening position
     * @param a one player
     * @param b the other
     * @param games games to play
     * @return {@code a}'s score: 1 per win, 0.5 per draw, divided by games
     */
    public static double score(TwoPlayerGame start, TwoPlayerGame.Player a, TwoPlayerGame.Player b, int games) {
        double s = 0;
        for (int g = 0; g < games; g++) {
            int r = g % 2 == 0 ? TwoPlayerGame.playOut(start, a, b) : -TwoPlayerGame.playOut(start, b, a);
            s += r > 0 ? 1 : r == 0 ? 0.5 : 0;
        }
        return s / games;
    }

    /**
     * Trains AlphaZero and scores the players against each other.
     *
     * @param game the opening position, for example {@link NInARow#connectThree()}
     * @param iterations self-play iterations
     * @param gamesPerIteration games per iteration
     * @param simulations MCTS simulations for training and for the searching players
     * @param matchGames games per pairing
     * @param seed the seed
     * @return pairing ("A vs B") to A's score
     */
    public static Map<String, Double> run(TwoPlayerGame game, int iterations, int gamesPerIteration, int simulations,
            int matchGames, long seed) {
        int inputs = game.encode().length;
        AlphaZero az = AlphaZero.of(game, Nets.mlp(inputs, game.actions() + 1, Activation.RELU, seed, 64, 64))
                .simulations(simulations).epochs(8).seed(seed);
        az.train(iterations, gamesPerIteration);
        TwoPlayerGame.Player prior = az.player(0);
        TwoPlayerGame.Player guided = az.player(simulations);
        Mcts<TwoPlayerGame> pure = Mcts.of(TwoPlayerGame.searchModel(game.actions(),
                TwoPlayerGame.randomRollouts(seed))).simulations(simulations);
        SplittableRandom r = new SplittableRandom(seed);
        TwoPlayerGame.Player rollouts = g -> pure.search(g, r).best();
        TwoPlayerGame.Player random = TwoPlayerGame.Player.random(seed);
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("AlphaZero+MCTS vs pure MCTS", score(game, guided, rollouts, matchGames));
        out.put("AlphaZero+MCTS vs random", score(game, guided, random, matchGames));
        out.put("AlphaZero prior vs random", score(game, prior, random, matchGames));
        out.put("pure MCTS vs random", score(game, rollouts, random, matchGames));
        return out;
    }
}
