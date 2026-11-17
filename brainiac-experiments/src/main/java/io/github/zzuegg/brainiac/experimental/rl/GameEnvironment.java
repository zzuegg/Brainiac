package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Turns a {@link BoardGame} against a fixed opponent into an {@link Environment}, so any
 * learner of the RL module (DQN, PPO, tabular) can learn to play it. The agent plays one
 * side (alternating sides between episodes if asked); observations are
 * {@link BoardGame#features} from the agent's view; the reward is +1 for a win, -1 for a
 * loss, 0 otherwise; an illegal move loses at once.
 */
public final class GameEnvironment implements Environment {

    private final Supplier<BoardGame> start;
    private final GamePlayer opponent;
    private final boolean alternate;
    private BoardGame game;
    private int me = 1;
    private int episodes;
    private SplittableRandom random = new SplittableRandom(0);

    private GameEnvironment(Supplier<BoardGame> start, GamePlayer opponent, boolean alternate) {
        this.start = Objects.requireNonNull(start);
        this.opponent = Objects.requireNonNull(opponent);
        this.alternate = alternate;
        this.game = start.get();
    }

    /**
     * @param start makes the first position
     * @param opponent the other player
     * @param alternateSides play first and second in turn
     * @return the environment
     */
    public static GameEnvironment of(Supplier<BoardGame> start, GamePlayer opponent, boolean alternateSides) {
        return new GameEnvironment(start, opponent, alternateSides);
    }

    @Override
    public Space observationSpace() {
        return Space.box(game.features(1).length, 0, 1);
    }

    @Override
    public Space actionSpace() {
        return Space.discrete(game.moveCount());
    }

    @Override
    public float[] reset(SplittableRandom random) {
        this.random = random;
        game = start.get();
        me = alternate && episodes++ % 2 == 1 ? 2 : 1;
        if (game.toMove() != me) {
            game = game.play(opponent.move(game, random));
        }
        return game.features(me);
    }

    @Override
    public Step step(float[] action) {
        int move = Math.round(action[0]);
        if (!game.legalMoves().contains(move)) {
            return new Step(game.features(me), -1, true, false);
        }
        game = game.play(move);
        if (!game.isOver()) {
            game = game.play(opponent.move(game, random));
        }
        int w = game.winner();
        float reward = w == me ? 1 : w == 3 - me ? -1 : 0;
        return new Step(game.features(me), reward, game.isOver(), false);
    }

    @Override
    public String render() {
        return game.toString();
    }
}
