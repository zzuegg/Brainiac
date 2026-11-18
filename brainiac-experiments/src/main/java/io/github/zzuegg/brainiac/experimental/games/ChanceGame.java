package io.github.zzuegg.brainiac.experimental.games;

/**
 * An immutable position of a two-player zero-sum game with chance nodes (dice, card draws):
 * the stochastic counterpart of {@link io.github.zzuegg.brainiac.experimental.rl.TwoPlayerGame},
 * searched by {@link MinimaxSearch.Expectimax}. Players are {@code +1} and {@code -1}.
 */
public interface ChanceGame {

    /** @return the number of distinct decisions (legal or not) */
    int actions();

    /** @return which decisions are legal (decision nodes only) */
    boolean[] legal();

    /**
     * @param action a legal decision
     * @return the position after it (often a chance node)
     */
    ChanceGame play(int action);

    /** @return whether nature moves next */
    boolean chanceNode();

    /** @return the probability of every chance outcome (chance nodes only) */
    double[] outcomeProbabilities();

    /**
     * @param k a chance outcome
     * @return the position after it
     */
    ChanceGame outcome(int k);

    /** @return whether the game is over */
    boolean over();

    /** @return the winner ({@code +1} or {@code -1}), 0 for none */
    int winner();

    /** @return the player to decide */
    int toMove();

    /** Scores decision positions for the side to move. */
    @FunctionalInterface
    interface Evaluator {
        /**
         * @param position an unfinished decision position
         * @return its value for the side to move, in {@code [-1, 1]}
         */
        double evaluate(ChanceGame position);
    }
}
