package io.github.zzuegg.brainiac.experimental.code;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/** Wave 9 WP2: deterministic checks of the code-model building blocks. */
class CodeBlocksTest {

    private static final ExecutionTracer TRACER = ExecutionTracer.of(List.of("inc", "rev", "tail"),
            CodeBlocksTest::apply, 10);

    private static int[] apply(String op, int[] in) {
        int[] out = in.clone();
        switch (op) {
            case "inc" -> {
                for (int i = 0; i < out.length; i++) {
                    out[i]++;
                }
            }
            case "rev" -> {
                for (int i = 0; i < out.length; i++) {
                    out[i] = in[in.length - 1 - i];
                }
            }
            case "tail" -> {
                return in.length == 0 ? null : Arrays.copyOfRange(in, 1, in.length);
            }
            default -> throw new IllegalArgumentException(op);
        }
        return out;
    }

    @Test
    void tracesRecordEveryLine() {
        ExecutionTracer.Trace t = TRACER.trace(List.of("inc", "rev"), new int[] {1, 2, 3});
        assertEquals(2, t.frames().size());
        assertArrayEquals(new int[] {4, 3, 2}, t.output());
        assertNull(TRACER.run(List.of("tail", "tail"), new int[] {1}));
    }

    @Test
    void rerankersPickTheConsensus() {
        ExecutionReranker rr = ExecutionReranker.of(TRACER);
        List<ExecutionReranker.Candidate> cs = new ArrayList<>();
        cs.add(new ExecutionReranker.Candidate(List.of("rev"), -0.1));
        cs.add(new ExecutionReranker.Candidate(List.of("inc", "inc"), -2));
        cs.add(new ExecutionReranker.Candidate(List.of("rev", "inc", "rev", "inc"), -3));
        int[][] in = {{1, 2}};
        int[][] out = {{3, 4}};
        assertEquals(2, rr.filter(cs, in, out).size());
        assertEquals(List.of("rev"), rr.likelihood(cs).program());
        assertEquals(List.of("inc", "inc"), rr.cluster(cs, new int[][] {{5, 1}}).program());
        List<ExecutionReranker.TestCase> tests = List.of(new ExecutionReranker.TestCase(in[0], out[0]),
                new ExecutionReranker.TestCase(new int[] {0}, new int[] {2}));
        assertEquals(List.of("inc", "inc"), rr.dualAgreement(cs, tests).program());
    }

    @Test
    void repairCountsEveryCall() {
        int[][] in = {{1, 2}};
        int[][] out = {{2, 3}};
        RepairLoop loop = RepairLoop.of(TRACER, (i, o, r) -> List.of("rev"),
                (i, o, failed, traces, fb, r) -> List.of("inc"));
        RepairLoop.Result fixed = loop.run(in, out, 4, RepairLoop.Feedback.OUTPUT_DIFF, new SplittableRandom(1));
        RepairLoop.Result resampled = loop.run(in, out, 4, RepairLoop.Feedback.NONE, new SplittableRandom(1));
        assertEquals(2, fixed.solvedAt());
        assertFalse(resampled.solvedWithin(4));
        System.out.printf("code: perfect repairer solves at call %d, resampling a wrong generator never%n",
                fixed.solvedAt());
    }
}
