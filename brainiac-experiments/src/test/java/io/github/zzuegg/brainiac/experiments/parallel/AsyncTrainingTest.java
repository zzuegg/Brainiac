package io.github.zzuegg.brainiac.experiments.parallel;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class AsyncTrainingTest {

    @Test
    void hogwildMatchesOneThreadOnSparseData() {
        List<AsyncTraining.HogwildRow> rows = AsyncTraining.hogwild(40_000, 1);
        for (AsyncTraining.HogwildRow r : rows) {
            System.out.printf("9.6.7 %-22s %d threads: log loss %.4f, %.3f s, speed-up %.2fx%n", r.method(),
                    r.threads(), r.loss(), r.seconds(), r.speedUp());
        }
        double one = rows.get(0).loss();
        double eight = rows.get(3).loss();
        assertTrue(Math.abs(eight - one) < 0.05, rows.toString());
        assertTrue(one < Math.log(2), "learns something");
    }

    @Test
    void delayCompensationHelpsStaleDenseGradients() {
        List<AsyncTraining.StaleRow> rows = AsyncTraining.staleness(new int[] {0, 4, 16, 32}, 2000, 3);
        for (AsyncTraining.StaleRow r : rows) {
            System.out.printf("9.6.7 %-40s tau %2d: loss %.4f, accuracy %.3f%n", r.method(), r.delay(), r.loss(),
                    r.accuracy());
        }
        AsyncTraining.StaleRow sync = rows.stream().filter(r -> r.method().equals("ASGD") && r.delay() == 0)
                .findFirst().orElseThrow();
        AsyncTraining.StaleRow plain32 = rows.stream().filter(r -> r.method().equals("ASGD") && r.delay() == 32)
                .findFirst().orElseThrow();
        AsyncTraining.StaleRow dc32 = rows.stream().filter(r -> r.method().startsWith("DC") && r.delay() == 32)
                .findFirst().orElseThrow();
        assertTrue(sync.accuracy() > 0.5, sync.toString());
        assertTrue(Double.isFinite(dc32.loss()) && dc32.loss() <= 1.1 * plain32.loss() + 0.05,
                dc32 + " vs " + plain32);
    }
}
