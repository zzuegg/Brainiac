package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class GenerativeAgentsRecallTest {

    @Test
    void memoryStreamRecallsMoreThanWindowAndPager() {
        GenerativeAgentsRecall.Result r = GenerativeAgentsRecall.run(5, 6, 6, 1);
        System.out.printf("memory stream recall %.2f (latest %.2f, event %.2f, usual %.2f) vs relevance-only %.2f, "
                + "sliding window %.2f, MemGPT pager %.2f (%d memories)%n", r.stream(), r.streamLatest(),
                r.streamImportant(), r.streamUsual(), r.relevanceOnly(), r.window(), r.pager(), r.memories());
        assertTrue(r.stream() >= 0.8, r.toString());
        assertTrue(r.stream() > r.window() + 0.2, r.toString());
        assertTrue(r.stream() > r.pager() + 0.2, r.toString());
        assertTrue(r.stream() > r.relevanceOnly(), r.toString());
    }
}
