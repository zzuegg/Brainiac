package io.github.zzuegg.brainiac.experimental.rl;

import java.util.List;
import java.util.SplittableRandom;

/** Picks moves in a {@link BoardGame}. */
@FunctionalInterface
public interface GamePlayer {

    /**
     * @param position the position, with this player to move
     * @param random the random source
     * @return a legal move
     */
    int move(BoardGame position, SplittableRandom random);

    /** @return a player that moves uniformly at random */
    static GamePlayer random() {
        return (position, random) -> {
            List<Integer> moves = position.legalMoves();
            return moves.get(random.nextInt(moves.size()));
        };
    }

    /**
     * Plays one game.
     *
     * @param start the first position
     * @param first the player who moves first (player 1)
     * @param second player 2
     * @param random the random source
     * @return the winner: 1, 2 or -1 for a draw
     */
    static int playGame(BoardGame start, GamePlayer first, GamePlayer second, SplittableRandom random) {
        BoardGame g = start;
        while (!g.isOver()) {
            g = g.play((g.toMove() == 1 ? first : second).move(g, random));
        }
        return g.winner();
    }
}
