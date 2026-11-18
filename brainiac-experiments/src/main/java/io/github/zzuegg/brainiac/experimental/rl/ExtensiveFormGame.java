package io.github.zzuegg.brainiac.experimental.rl;

/**
 * A finite two-player zero-sum game with imperfect information and chance, in extensive
 * form: a tree of histories where either a player or chance acts, and where a player sees only
 * an <em>information set</em>, the part of the history they observe (their own cards and the
 * public actions, not the opponent's cards). Poker is the standard example; a matrix game is
 * the special case where the second player cannot see the first player's move.
 *
 * <p>Actions are indexed {@code 0..maxActions()-1} at every node, with {@link State#legal}
 * marking those allowed, so a network can have one output per action. Payoffs are to player
 * 0; player 1 receives the negative. {@link GameTree#compile} expands the whole tree once into
 * flat arrays, which is what regret minimisation, best responses and equilibrium learners run
 * on.
 *
 * <p>Zinkevich et al. 2007, "Regret Minimization in Games with Incomplete Information",
 * <a href="https://papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information">NeurIPS 2007</a>.
 */
public interface ExtensiveFormGame {

    /** The {@link State#player} of a chance node. */
    int CHANCE = -1;

    /** A history of the game. States are immutable: {@link #play} returns a new state. */
    interface State {

        /** @return whether the game is over */
        boolean terminal();

        /** @return the player to act, 0 or 1, or {@link #CHANCE} */
        int player();

        /**
         * @return the number of action indices at this node: {@link #maxActions} for a
         *     player, the number of outcome indices for chance
         */
        int actionCount();

        /**
         * @param action an action index
         * @return whether it is allowed here
         */
        boolean legal(int action);

        /**
         * @param action a chance outcome
         * @return its probability (only at chance nodes)
         */
        double chanceProbability(int action);

        /**
         * @param action a legal action or chance outcome
         * @return the following history
         */
        State play(int action);

        /** @return the payoff to player 0 at a terminal history */
        double payoff();

        /**
         * @return a key naming the acting player's information set: equal for exactly the
         *     histories that player cannot tell apart
         */
        String infoSet();

        /** @return a fixed-length encoding of the acting player's information set */
        float[] features();
    }

    /** @return the empty history */
    State root();

    /** @return the number of action indices (the most actions at any node) */
    int maxActions();

    /** @return the length of {@link State#features} */
    int featureSize();
}
