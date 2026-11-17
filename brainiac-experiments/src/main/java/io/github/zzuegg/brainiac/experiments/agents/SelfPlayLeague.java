package io.github.zzuegg.brainiac.experiments.agents;

import io.github.zzuegg.brainiac.experimental.rl.AfterstateLearner;
import io.github.zzuegg.brainiac.experimental.rl.BoardGame;
import io.github.zzuegg.brainiac.experimental.rl.EloLeague;
import io.github.zzuegg.brainiac.experimental.rl.GamePlayer;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Experiment X26, self-play: two learners play each other, one always first and one always
 * second, and improve together; every few hundred games both are frozen into an
 * {@link EloLeague}. A final round robin rates all snapshots, so the progress shows as rising
 * Elo, and both are tested against a random player.
 */
public final class SelfPlayLeague {

    /**
     * The outcome.
     *
     * @param firstEloStart the Elo of the first player's earliest snapshot
     * @param firstEloEnd of its last snapshot
     * @param secondEloStart the Elo of the second player's earliest snapshot
     * @param secondEloEnd of its last snapshot
     * @param firstVsRandom the final first player's non-loss rate against a random player
     * @param secondVsRandom the final second player's non-loss rate against a random player
     */
    public record Result(double firstEloStart, double firstEloEnd, double secondEloStart, double secondEloEnd,
            double firstVsRandom, double secondVsRandom) {
    }

    private SelfPlayLeague() {
    }

    /**
     * @param game the game
     * @param games training games
     * @param snapshots how many snapshots of each player
     * @param seed the seed
     * @return the outcome
     */
    public static Result run(Supplier<BoardGame> game, int games, int snapshots, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        AfterstateLearner first = AfterstateLearner.create().epsilon(0.1);
        AfterstateLearner second = AfterstateLearner.create().epsilon(0.1);
        EloLeague league = EloLeague.of(game).kFactor(8);
        int[] firstIds = new int[snapshots];
        int[] secondIds = new int[snapshots];
        int every = games / snapshots;
        for (int g = 0; g < games; g++) {
            if (g % every == 0) {
                int s = g / every;
                firstIds[s] = league.add("first@" + g, first.snapshot());
                secondIds[s] = league.add("second@" + g, second.snapshot());
            }
            AfterstateLearner learner = g % 2 == 0 ? first : second;
            learner.train(game.get(), first, second, random);
        }
        firstIds[snapshots - 1] = league.add("first@end", first.snapshot());
        secondIds[snapshots - 1] = league.add("second@end", second.snapshot());
        league.roundRobin(10, random);
        return new Result(league.rating(firstIds[0]), league.rating(firstIds[snapshots - 1]),
                league.rating(secondIds[0]), league.rating(secondIds[snapshots - 1]),
                nonLoss(game, first.snapshot(), true, random), nonLoss(game, second.snapshot(), false, random));
    }

    static double nonLoss(Supplier<BoardGame> game, GamePlayer p, boolean asFirst, SplittableRandom random) {
        int ok = 0;
        int n = 200;
        for (int i = 0; i < n; i++) {
            int w = asFirst ? GamePlayer.playGame(game.get(), p, GamePlayer.random(), random)
                    : GamePlayer.playGame(game.get(), GamePlayer.random(), p, random);
            if (w == -1 || w == (asFirst ? 1 : 2)) {
                ok++;
            }
        }
        return (double) ok / n;
    }
}
