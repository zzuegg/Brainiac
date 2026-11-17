package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.IntFunction;

/**
 * An environment over procedurally generated levels whose every {@link #reset} plays the
 * level a {@link LevelReplay} picks. All levels must share observation and action spaces.
 * The learner reports each episode's score with {@link #score(double)}.
 */
public final class ReplayedLevels implements Environment {

    private final IntFunction<Environment> generator;
    private final LevelReplay sampler;
    private final Map<Integer, Environment> cache = new HashMap<>();
    private final Environment first;
    private Environment current;
    private int level = -1;

    private ReplayedLevels(IntFunction<Environment> generator, LevelReplay sampler) {
        this.generator = Objects.requireNonNull(generator);
        this.sampler = Objects.requireNonNull(sampler);
        this.first = generator.apply(0);
        cache.put(0, first);
    }

    /**
     * @param generator makes level {@code i}
     * @param sampler picks the levels
     * @return the environment
     */
    public static ReplayedLevels of(IntFunction<Environment> generator, LevelReplay sampler) {
        return new ReplayedLevels(generator, sampler);
    }

    /** @return the level being played, or -1 before the first reset */
    public int currentLevel() {
        return level;
    }

    /**
     * Reports the score of the episode just played on {@link #currentLevel()}.
     *
     * @param learningPotential such as the mean absolute TD error
     */
    public void score(double learningPotential) {
        sampler.update(level, learningPotential);
    }

    @Override
    public Space observationSpace() {
        return first.observationSpace();
    }

    @Override
    public Space actionSpace() {
        return first.actionSpace();
    }

    @Override
    public float[] reset(SplittableRandom random) {
        level = sampler.next();
        current = cache.computeIfAbsent(level, generator::apply);
        return current.reset(random);
    }

    @Override
    public Step step(float[] action) {
        return current.step(action);
    }
}
