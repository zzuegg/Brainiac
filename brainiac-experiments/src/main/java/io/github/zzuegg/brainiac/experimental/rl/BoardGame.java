package io.github.zzuegg.brainiac.experimental.rl;

import java.util.List;

/**
 * A two-player, zero-sum, turn-based game with perfect information, such as
 * {@link TicTacToe} or {@link ConnectThree}. Positions are immutable: {@link #play} returns a
 * new position. Players are numbered 1 and 2.
 */
public interface BoardGame {

    /** @return the player to move, 1 or 2 */
    int toMove();

    /** @return the legal moves in this position */
    List<Integer> legalMoves();

    /**
     * @param move a legal move
     * @return the position after it
     */
    BoardGame play(int move);

    /** @return 1 or 2 for a winner, -1 for a draw, 0 while the game goes on */
    int winner();

    /** @return whether the game is over */
    default boolean isOver() {
        return winner() != 0;
    }

    /** @return the number of possible moves (the action space size) */
    int moveCount();

    /**
     * Encodes the position for a network, from the point of view of a player: for every
     * cell, one-hot {own piece, opponent piece}.
     *
     * @param player the viewer
     * @return the features
     */
    float[] features(int player);

    /** @return a key that identifies the position, for tables */
    String key();
}
