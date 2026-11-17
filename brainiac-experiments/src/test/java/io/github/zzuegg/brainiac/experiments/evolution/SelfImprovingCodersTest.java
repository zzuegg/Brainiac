package io.github.zzuegg.brainiac.experiments.evolution;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.evolve.SelfModifyingArchive;
import java.util.OptionalDouble;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class SelfImprovingCodersTest {

    @Test
    void archiveCrossesAValleyThatStopsHillClimbing() {
        // score peaks at 0 and 3, with a valley at 1 and 2: the step +1 is the only edit
        double[] f = {0.5, 0.2, 0.1, 1.0, 0.0};
        SelfModifyingArchive<Integer> a = SelfModifyingArchive.of(0, x -> OptionalDouble.of(f[x]),
                (x, r) -> Math.min(4, x + 1));
        a.run(10, 2, new SplittableRandom(1));
        assertTrue(a.best().score() == 1.0, a.archive().toString());
        SelfModifyingArchive.Climb<Integer> g = SelfModifyingArchive.hillClimb(0, x -> OptionalDouble.of(f[x]),
                (x, r) -> Math.min(4, x + 1), 10, 2, new SplittableRandom(1));
        assertTrue(g.agent() == 0, "hill-climbing stays on the first peak");
    }

    @Test
    void openEndedArchiveBeatsGreedyHillClimbing() {
        SelfImprovingCoders.Result r = SelfImprovingCoders.run(25, 4, 200000, 1.7, 1);
        int last = r.archive().length - 1;
        System.out.printf("self-modifying coders, benchmark score after %d generations: DGM archive %.3f (%s, %d "
                + "agents) vs greedy hill-climbing %.3f (%s); start %.3f%n", r.archive().length, r.archive()[last],
                r.archiveBest(), r.archiveSize(), r.greedy()[last], r.greedyBest(), r.initial());
        assertTrue(r.archive()[last] > r.greedy()[last] + 0.03, r.toString());
    }
}
