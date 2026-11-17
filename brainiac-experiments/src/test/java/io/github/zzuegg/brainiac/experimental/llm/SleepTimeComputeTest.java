package io.github.zzuegg.brainiac.experimental.llm;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.assertEquals;

class SleepTimeComputeTest {

    /** The "model": answers questions about a list of numbers. */
    static Integer answer(int[] doc, String q) {
        return switch (q.trim().toLowerCase()) {
            case "sum" -> Arrays.stream(doc).sum();
            case "max" -> Arrays.stream(doc).max().orElse(0);
            case "min" -> Arrays.stream(doc).min().orElse(0);
            case "count" -> doc.length;
            default -> -1;
        };
    }

    @Test
    void anticipatedQueriesCostNothingAtTestTime() {
        try (ExecutorService ex = Executors.newSingleThreadExecutor()) {
            SleepTimeCompute<int[], Integer> stc = SleepTimeCompute.of(doc -> List.of("sum", "max", "min"),
                    SleepTimeComputeTest::answer, ex);
            stc.contextChanged(new int[] {3, 9, 4});
            stc.awaitIdle();
            assertEquals(16, stc.ask("sum"));
            assertEquals(9, stc.ask("  MAX "));
            assertEquals(3, stc.ask("min"));
            assertEquals(3, stc.ask("count"));
            assertEquals(new SleepTimeCompute.Stats(3, 1, 3, 1, 0), stc.stats());
        }
    }

    @Test
    void aNewContextMakesEarlierWorkStale() throws InterruptedException {
        try (ExecutorService ex = Executors.newSingleThreadExecutor()) {
            CountDownLatch gate = new CountDownLatch(1);
            ex.execute(() -> {
                try {
                    gate.await();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
            SleepTimeCompute<int[], Integer> stc = SleepTimeCompute.of(doc -> List.of("sum", "max"),
                    SleepTimeComputeTest::answer, ex);
            stc.contextChanged(new int[] {1, 2});
            stc.contextChanged(new int[] {10, 20});
            gate.countDown();
            stc.awaitIdle();
            assertEquals(30, stc.ask("sum"));
            assertEquals(20, stc.ask("max"));
            SleepTimeCompute.Stats s = stc.stats();
            assertEquals(2, s.skipped(), s.toString());
            assertEquals(2, s.sleepCalls());
            assertEquals(2, s.hits());
            assertEquals(0, s.testCalls());
        }
    }
}
