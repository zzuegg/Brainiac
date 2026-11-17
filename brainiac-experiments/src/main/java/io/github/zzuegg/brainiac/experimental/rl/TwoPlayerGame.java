package io.github.zzuegg.brainiac.experimental.rl;

import java.util.SplittableRandom;

/**
 * An immutable position of a two-player, zero-sum, perfect-information board game with
 * discrete moves, such as tic-tac-toe or Connect Four: the counterpart of
 * {@link io.github.zzuegg.brainiac.rl.Environment} for self-play methods like AlphaZero.
 * Players are {@code +1} (moves first) and {@code -1}.
 */
public interface TwoPlayerGame {

    /** @return the number of distinct moves (legal or not) */
    int actions();

    /** @return which moves are legal now */
    boolean[] legal();

    /**
     * @param action a legal move
     * @return the position after it
     */
    TwoPlayerGame play(int action);

    /** @return whether the game is over */
    boolean over();

    /** @return {@code +1} or {@code -1} for the winner, 0 for a draw or an unfinished game */
    int winner();

    /** @return the player to move, {@code +1} or {@code -1} */
    int toMove();

    /** @return the position as seen by the player to move, as network input */
    float[] encode();

    /** Picks moves. */
    @FunctionalInterface
    interface Player {
        /**
         * @param position a position that is not over
         * @return a legal move
         */
        int move(TwoPlayerGame position);

        /**
         * @param seed the random seed
         * @return a player that moves uniformly at random
         */
        static Player random(long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            return g -> {
                boolean[] legal = g.legal();
                int count = 0;
                for (boolean b : legal) {
                    count += b ? 1 : 0;
                }
                int k = r.nextInt(count);
                for (int a = 0; a < legal.length; a++) {
                    if (legal[a] && k-- == 0) {
                        return a;
                    }
                }
                throw new IllegalStateException("no legal move");
            };
        }
    }

    /**
     * Plays one game.
     *
     * @param start the opening position
     * @param first the player moving first
     * @param second the other player
     * @return the winner: {@code +1} first, {@code -1} second, 0 draw
     */
    static int playOut(TwoPlayerGame start, Player first, Player second) {
        TwoPlayerGame g = start;
        while (!g.over()) {
            g = g.play(g.toMove() == start.toMove() ? first.move(g) : second.move(g));
        }
        return g.winner() * start.toMove();
    }

    /**
     * Adapts the game for {@link Mcts}: rewards {@code +1} to the player whose move wins,
     * judging non-terminal positions with an evaluator.
     *
     * @param actions the number of moves
     * @param evaluator the prior and value (for the player to move) of a live position
     * @return the search model
     */
    static Mcts.Model<TwoPlayerGame> searchModel(int actions,
            java.util.function.Function<TwoPlayerGame, Mcts.Evaluation> evaluator) {
        return new Mcts.Model<>() {
            @Override
            public int actions() {
                return actions;
            }

            @Override
            public Mcts.Evaluation evaluate(TwoPlayerGame state) {
                if (state.over()) {
                    return new Mcts.Evaluation(new float[actions], 0, true);
                }
                return evaluator.apply(state);
            }

            @Override
            public Mcts.Step<TwoPlayerGame> step(TwoPlayerGame state, int action) {
                TwoPlayerGame next = state.play(action);
                return new Mcts.Step<>(next, next.winner() == state.toMove() ? 1f : 0f);
            }

            @Override
            public boolean alternating() {
                return true;
            }
        };
    }

    /**
     * An evaluator for pure MCTS: uniform priors over legal moves and the result of one
     * random playout as value.
     *
     * @param seed the playout seed
     * @return the evaluator
     */
    static java.util.function.Function<TwoPlayerGame, Mcts.Evaluation> randomRollouts(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Player random = Player.random(r.nextLong());
        return g -> {
            boolean[] legal = g.legal();
            float[] p = new float[legal.length];
            for (int a = 0; a < p.length; a++) {
                p[a] = legal[a] ? 1 : 0;
            }
            int result = playOut(g, random, random);
            return new Mcts.Evaluation(p, result, false);
        };
    }
}
