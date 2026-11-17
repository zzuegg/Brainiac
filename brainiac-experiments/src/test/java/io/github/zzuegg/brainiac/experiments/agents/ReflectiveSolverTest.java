package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.ReflexionLoop;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class ReflectiveSolverTest {

    @Test
    void loopRemembersOnlyTheLatestLessons() {
        int[] guesses = {1, 2, 3, 4};
        int[] next = {0};
        ReflexionLoop.Outcome<Integer> o = ReflexionLoop.<Integer>of((memory, r) -> guesses[next[0]++],
                v -> new ReflexionLoop.Feedback(v == 4, ""), (v, f) -> "not " + v).memorySize(2)
                .run(5, new SplittableRandom(1));
        assertTrue(o.success());
        assertEquals(4, o.trials());
        assertEquals(List.of("not 2", "not 3"), o.memory());
    }

    @Test
    void reflectionsBeatPlainRetrying() {
        ReflectiveSolver.Result r = ReflectiveSolver.run(40, 4, 1);
        System.out.println("reflexion " + r);
        assertTrue(r.retry() > r.firstTry(), r.toString());
        assertTrue(r.reflexion() > r.retry() + 0.1, r.toString());
    }

    @Test
    void pagedReflectionMemoryBeatsReflexionAloneOverLongSequences() {
        ReflectiveSolver.Lifelong r = ReflectiveSolver.lifelong(40, 240, 2, 1);
        System.out.printf("lifelong solve rate: memory agent %.3f (first try %.3f, last quarter %.3f) vs Reflexion "
                + "alone %.3f (first try %.3f, last quarter %.3f)%n", r.memoryAgent(), r.memoryFirstTry(),
                r.lastQuarterMemory(), r.reflexion(), r.reflexionFirstTry(), r.lastQuarterReflexion());
        assertTrue(r.memoryAgent() > r.reflexion() + 0.1, r.toString());
        assertTrue(r.lastQuarterMemory() > r.lastQuarterReflexion() + 0.2, r.toString());
        assertTrue(r.memoryFirstTry() > r.reflexionFirstTry(), r.toString());
    }
}
