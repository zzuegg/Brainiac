package io.github.zzuegg.brainiac.experiments.evolution;

import io.github.zzuegg.brainiac.experimental.evolve.LibraryLearner;
import io.github.zzuegg.brainiac.experiments.reasoning.ListPrograms;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Predicate;

/**
 * DreamCoder-style library learning on {@link ListPrograms} tasks (item 5.8). The task set
 * holds every concept alone, each concept followed by one extra primitive, and pairs and
 * triples of concepts. An enumerator with a fixed budget per task solves the short ones;
 * {@link LibraryLearner} then compresses the solutions into new primitives (the concepts
 * reappear across tasks, so they are what compresses) and refits its prior, and the next
 * wake reaches the pairs and triples as two- and three-token programs. The baseline is the
 * same enumerator over the fixed DSL, for the same rounds.
 */
public final class LearnedLibraries {

    /**
     * Per round, for the learned and the fixed library.
     *
     * @param tasks the number of tasks
     * @param learnedSolved tasks solved after each round, with library learning
     * @param fixedSolved tasks solved after each round, fixed DSL
     * @param learnedEvaluations programs evaluated in each round, with library learning
     * @param fixedEvaluations programs evaluated in each round, fixed DSL
     * @param learnedEvalsPerSolve mean evaluations per newly solved task in each round, learned
     * @param fixedEvalsPerSolve the same for the fixed DSL
     * @param abstractions entries the library learned
     * @param learnedMillis wall time of the learned run
     * @param fixedMillis wall time of the fixed run
     */
    public record Result(int tasks, int[] learnedSolved, int[] fixedSolved, long[] learnedEvaluations,
            long[] fixedEvaluations, double[] learnedEvalsPerSolve, double[] fixedEvalsPerSolve, int abstractions,
            long learnedMillis, long fixedMillis) {
    }

    private LearnedLibraries() {
    }

    /**
     * @param seed the seed
     * @return the task set: concepts, concepts plus one primitive, pairs and triples
     */
    public static List<ListPrograms.Task> tasks(long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        List<ListPrograms.Task> out = new ArrayList<>();
        List<String> prims = ListPrograms.primitives();
        for (ListPrograms.Concept c : ListPrograms.CONCEPTS) {
            out.add(ListPrograms.task(List.of(c), 6, random));
            for (int k = 0; k < 3; k++) {
                String p = prims.get(random.nextInt(prims.size()));
                out.add(ListPrograms.task(List.of(c, new ListPrograms.Concept(p, List.of(p))), 6, random));
            }
        }
        for (int i = 0; i < 15; i++) {
            out.add(ListPrograms.task(2, 6, random));
        }
        for (int i = 0; i < 10; i++) {
            out.add(ListPrograms.task(3, 6, random));
        }
        return out;
    }

    /**
     * @param rounds wake-sleep rounds
     * @param budget evaluations per task per wake
     * @param seed the seed
     * @return the solve counts and costs per round
     */
    public static Result run(int rounds, int budget, long seed) {
        List<ListPrograms.Task> tasks = tasks(seed);
        List<Predicate<List<String>>> checks = new ArrayList<>();
        for (ListPrograms.Task t : tasks) {
            checks.add(t::solvedBy);
        }
        LibraryLearner learned = LibraryLearner.of(ListPrograms.primitives()).budget(budget).maxLength(3);
        LibraryLearner fixed = LibraryLearner.of(ListPrograms.primitives()).budget(budget).maxLength(3);
        int[] ls = new int[rounds];
        int[] fs = new int[rounds];
        long[] le = new long[rounds];
        long[] fe = new long[rounds];
        double[] lp = new double[rounds];
        double[] fp = new double[rounds];
        long t0 = System.nanoTime();
        for (int r = 0; r < rounds; r++) {
            LibraryLearner.Wake w = learned.wake(checks);
            ls[r] = w.solved();
            le[r] = w.evaluations();
            lp[r] = w.evaluationsPerSolve();
            learned.sleep();
        }
        long t1 = System.nanoTime();
        for (int r = 0; r < rounds; r++) {
            LibraryLearner.Wake w = fixed.wake(checks);
            fs[r] = w.solved();
            fe[r] = w.evaluations();
            fp[r] = w.evaluationsPerSolve();
        }
        long t2 = System.nanoTime();
        return new Result(tasks.size(), ls, fs, le, fe, lp, fp,
                learned.library().size() - ListPrograms.primitives().size(), (t1 - t0) / 1_000_000,
                (t2 - t1) / 1_000_000);
    }
}
