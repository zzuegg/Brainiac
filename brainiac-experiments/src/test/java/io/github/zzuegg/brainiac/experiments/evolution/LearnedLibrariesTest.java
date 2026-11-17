package io.github.zzuegg.brainiac.experiments.evolution;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.evolve.LibraryLearner;
import java.util.List;
import java.util.function.Predicate;
import org.junit.jupiter.api.Test;

class LearnedLibrariesTest {

    @Test
    void sleepCompressesARecurringFragment() {
        LibraryLearner l = LibraryLearner.of(List.of("a", "b", "c")).maxLength(4).budget(500);
        List<Predicate<List<String>>> tasks = List.of(p -> p.equals(List.of("a", "b", "c")),
                p -> p.equals(List.of("c", "a", "b")), p -> p.equals(List.of("a", "b", "a", "b")));
        assertEquals(3, l.wake(tasks).solved());
        List<LibraryLearner.Entry> added = l.sleep();
        assertEquals(List.of("a", "b"), added.getFirst().expansion());
        LibraryLearner.Solution s = l.solutions().get(2);
        assertNotNull(s);
        assertEquals(2, s.program().size(), "a b a b is now two tokens: " + s);
    }

    @Test
    void learnedLibrarySolvesMoreTasksThanTheFixedDsl() {
        LearnedLibraries.Result r = LearnedLibraries.run(4, 2000, 1);
        int last = r.learnedSolved().length - 1;
        System.out.printf("solved of %d per round: learned library %s (+%d abstractions, evals/solve %s, %d ms) vs "
                + "fixed DSL %s (evals/solve %s, %d ms)%n", r.tasks(), java.util.Arrays.toString(r.learnedSolved()),
                r.abstractions(), fmt(r.learnedEvalsPerSolve()), r.learnedMillis(),
                java.util.Arrays.toString(r.fixedSolved()), fmt(r.fixedEvalsPerSolve()), r.fixedMillis());
        assertEquals(r.learnedSolved()[0], r.fixedSolved()[0], "the first wake is the same search");
        assertTrue(r.learnedSolved()[last] > r.fixedSolved()[last] + 10, r.toString());
    }

    private static String fmt(double[] v) {
        StringBuilder b = new StringBuilder("[");
        for (double x : v) {
            b.append(b.length() > 1 ? ", " : "").append(Math.round(x));
        }
        return b.append(']').toString();
    }
}
