package io.github.zzuegg.brainiac.experiments.evolution;

import io.github.zzuegg.brainiac.experimental.evolve.CmaEs;
import io.github.zzuegg.brainiac.experimental.train.HebbianRule;

import java.util.SplittableRandom;

/**
 * Meta-evolves a Hebbian plasticity rule: the coefficients of {@link HebbianRule} are
 * evolved with CMA-ES so that a network whose weights start at zero learns new
 * pattern-to-class associations within its lifetime, by experiencing each pattern once
 * with its class clamped on the output layer, then recalls them without the teacher.
 * Nothing is trained by gradient descent; evolution only finds <em>how</em> to learn.
 */
public final class EvolvedPlasticity {

    private static final int INPUTS = 16;
    private static final int CLASSES = 4;
    private static final int PATTERNS = 6;

    /**
     * The outcome.
     *
     * @param rule the evolved coefficients, {@code [eta, A, B, C, D, decay]}
     * @param evolvedAccuracy recall accuracy of the evolved rule on new tasks
     * @param randomAccuracy mean recall accuracy of random rules
     */
    public record Outcome(double[] rule, double evolvedAccuracy, double randomAccuracy) {
    }

    private EvolvedPlasticity() {
    }

    /**
     * Lifetime learning: recall accuracy of a rule averaged over random association tasks.
     *
     * @param genome the rule's coefficients
     * @param tasks the number of tasks
     * @param seed the task seed
     * @return the mean recall accuracy
     */
    public static double lifetimeAccuracy(double[] genome, int tasks, long seed) {
        HebbianRule rule = HebbianRule.of(genome);
        SplittableRandom r = new SplittableRandom(seed);
        int correct = 0;
        for (int t = 0; t < tasks; t++) {
            double[][] x = new double[PATTERNS][INPUTS];
            int[] label = new int[PATTERNS];
            for (int p = 0; p < PATTERNS; p++) {
                for (int i = 0; i < INPUTS; i++) {
                    x[p][i] = r.nextBoolean() ? 1 : -1;
                }
                label[p] = r.nextInt(CLASSES);
            }
            double[][] w = new double[INPUTS][CLASSES];
            for (int p = 0; p < PATTERNS; p++) {
                double[] post = new double[CLASSES];
                for (int c = 0; c < CLASSES; c++) {
                    post[c] = c == label[p] ? 1 : -1;
                }
                rule.update(w, x[p], post);
            }
            for (int p = 0; p < PATTERNS; p++) {
                int best = 0;
                double bestValue = Double.NEGATIVE_INFINITY;
                for (int c = 0; c < CLASSES; c++) {
                    double s = 0;
                    for (int i = 0; i < INPUTS; i++) {
                        s += x[p][i] * w[i][c];
                    }
                    if (s > bestValue) {
                        bestValue = s;
                        best = c;
                    }
                }
                correct += best == label[p] ? 1 : 0;
            }
        }
        return correct / (double) (tasks * PATTERNS);
    }

    /**
     * Evolves a rule from a random start and compares it with random rules.
     *
     * @param evaluations the CMA-ES budget
     * @param seed the seed
     * @return the outcome
     */
    public static Outcome run(int evaluations, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] start = new double[6];
        for (int i = 0; i < 6; i++) {
            start[i] = r.nextGaussian() * 0.5;
        }
        CmaEs cma = CmaEs.of(start, 0.5).seed(seed);
        int[] calls = {0};
        double[] best = cma.minimize(g -> -lifetimeAccuracy(g, 20, seed * 1000 + calls[0]++ / 10), evaluations);
        double random = 0;
        for (int k = 0; k < 20; k++) {
            double[] g = new double[6];
            for (int i = 0; i < 6; i++) {
                g[i] = r.nextGaussian() * 0.5;
            }
            random += lifetimeAccuracy(g, 50, 99);
        }
        return new Outcome(best, lifetimeAccuracy(best, 200, 4242), random / 20);
    }
}
