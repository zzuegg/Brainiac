package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.GumbelMcts;
import io.github.zzuegg.brainiac.experimental.rl.Mcts;
import io.github.zzuegg.brainiac.experimental.rl.TwoPlayerGame;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * W6 3.18: Gumbel search against PUCT {@link Mcts} at small simulation budgets, both guided by
 * the same evaluator (uniform prior, one random rollout as the value), on a
 * {@link TwoPlayerGame} such as tic-tac-toe or Connect-3. For each budget {@code n} it plays
 * Gumbel against PUCT and both against a random player, alternating colours, and converts the
 * head-to-head score into an Elo difference.
 *
 * <p>Paper: Danihelka et al., <a href="https://openreview.net/forum?id=bERaNdoegnO">ICLR 2022</a>.
 */
public final class GumbelSearch {

    /**
     * The results at one budget.
     *
     * @param gumbelVsPuct Gumbel's score against PUCT (win 1, draw 0.5)
     * @param eloDifference the Elo difference that score implies (positive: Gumbel stronger)
     * @param gumbelVsRandom Gumbel's score against random play
     * @param puctVsRandom PUCT's score against random play
     */
    public record Outcome(double gumbelVsPuct, double eloDifference, double gumbelVsRandom, double puctVsRandom) {
    }

    private GumbelSearch() {
    }

    /**
     * @param game the opening position
     * @param budgets the simulation counts to compare
     * @param games games per match
     * @param seed the seed
     * @return budget to outcome
     */
    public static Map<Integer, Outcome> run(TwoPlayerGame game, int[] budgets, int games, long seed) {
        Map<Integer, Outcome> out = new LinkedHashMap<>();
        Mcts.Model<TwoPlayerGame> model = TwoPlayerGame.searchModel(game.actions(), TwoPlayerGame.randomRollouts(seed));
        for (int n : budgets) {
            SplittableRandom r = new SplittableRandom(seed + n);
            Mcts<TwoPlayerGame> puct = Mcts.of(model).simulations(n);
            GumbelMcts<TwoPlayerGame> gumbel = GumbelMcts.of(model).simulations(n);
            TwoPlayerGame.Player puctPlayer = g -> puct.search(g, r).best();
            TwoPlayerGame.Player gumbelPlayer = g -> gumbel.search(g, r).action();
            TwoPlayerGame.Player random = TwoPlayerGame.Player.random(seed + n);
            double h2h = SelfPlayArena.score(game, gumbelPlayer, puctPlayer, games);
            double clipped = Math.max(0.01, Math.min(0.99, h2h));
            out.put(n, new Outcome(h2h, 400 * Math.log10(clipped / (1 - clipped)),
                    SelfPlayArena.score(game, gumbelPlayer, random, games),
                    SelfPlayArena.score(game, puctPlayer, random, games)));
        }
        return out;
    }
}
