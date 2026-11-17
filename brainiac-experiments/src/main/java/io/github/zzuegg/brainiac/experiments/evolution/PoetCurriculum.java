package io.github.zzuegg.brainiac.experiments.evolution;

import io.github.zzuegg.brainiac.experimental.evolve.EvolutionStrategy;
import io.github.zzuegg.brainiac.experimental.evolve.Poet;
import io.github.zzuegg.brainiac.experimental.rl.WindyReach;
import io.github.zzuegg.brainiac.rl.Environment;

import java.util.Comparator;
import java.util.SplittableRandom;

/**
 * POET on the {@link WindyReach} family: starting from a near, calm target, POET grows a
 * curriculum of farther targets and stronger head winds while optimising a linear agent
 * for each, then the agent POET has for its hardest environment is compared with one
 * optimised directly on that environment from scratch for the same number of evolution
 * strategy steps.
 */
public final class PoetCurriculum {

    /**
     * The outcome.
     *
     * @param environments environments POET accepted
     * @param hardest the hardest active environment, {@code [target, wind]}
     * @param poetScore the score of POET's agent there
     * @param directScore the score of an agent optimised only there
     * @param transfers agent transfers that happened
     */
    public record Outcome(int environments, double[] hardest, double poetScore, double directScore, int transfers) {
    }

    private PoetCurriculum() {
    }

    /**
     * How hard an environment is: farther targets and stronger head winds are harder.
     *
     * @param p {@code [target, wind]}
     * @return the difficulty
     */
    public static double difficulty(double[] p) {
        return p[0] + 3 * Math.max(0, -p[1]);
    }

    /** @return the family: mutation moves the target out and strengthens the head wind */
    static Poet.EnvironmentFamily family() {
        return new Poet.EnvironmentFamily() {
            @Override
            public Environment create(double[] parameters) {
                return WindyReach.of(parameters);
            }

            @Override
            public double[] mutate(double[] p, SplittableRandom r) {
                return new double[] {Math.min(12, Math.max(0.5, p[0] + r.nextDouble(-0.5, 2))),
                    Math.max(-1.8, Math.min(0.5, p[1] + r.nextDouble(-0.5, 0.15)))};
            }
        };
    }

    /**
     * Runs the comparison.
     *
     * @param iterations POET iterations
     * @param seed the seed
     * @return the outcome
     */
    public static Outcome run(int iterations, long seed) {
        Poet poet = Poet.of(family(), Poet.linearPolicy(2, 1), 3)
                .minimalCriterion(-0.75, -0.15).reproduceScore(-0.3)
                .capacity(6).intervals(8, 4).episodes(60, 1).strategy(8, 0.1, 0.05).seed(seed)
                .start(new double[] {1, 0}, new double[3]);
        poet.run(iterations);
        Poet.Pair hardest = poet.pairs().stream()
                .max(Comparator.comparingDouble(p -> difficulty(p.parameters()))).orElseThrow();
        EvolutionStrategy direct = EvolutionStrategy.of(new double[3]).sigma(0.1).learningRate(0.05).pairs(8)
                .seed(seed);
        double[] env = hardest.parameters();
        direct.run(theta -> poet.evaluate(env, theta), iterations);
        return new Outcome(poet.archive().size(), env, hardest.score(), poet.evaluate(env, direct.center()),
                poet.transfers());
    }
}
