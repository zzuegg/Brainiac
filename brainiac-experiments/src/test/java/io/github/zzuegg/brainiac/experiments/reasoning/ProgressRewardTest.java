package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Wave 9, 9.2.10: process advantage verifiers as dense rewards and beam scores. */
class ProgressRewardTest {

    private static final StepChainTask TASK = new StepChainTask(2, 3, 4);
    private static ProcessSupervision.Setup setup;

    @BeforeAll
    static void start() {
        setup = ProcessSupervision.setup(TASK, 120, 1);
    }

    @Test
    void progressRewards() {
        ProcessSupervision.ProgressResult r = ProcessSupervision.progress(setup, 40, 2, 1f, 3);
        StringBuilder sb = new StringBuilder();
        r.accuracy().forEach((k, a) -> sb.append(String.format(" %s %s;", k, fmt(a))));
        System.out.printf("9.2.10 PAV accuracy by checkpoint (outcome-only GRPO is the baseline):%s beam %s%n", sb,
                r.beam());
        assertFalse(r.accuracy().isEmpty());
    }

    private static String fmt(double[] xs) {
        StringBuilder sb = new StringBuilder("[");
        Arrays.stream(xs).forEach(x -> sb.append(String.format(" %.3f", x)));
        return sb.append(" ]").toString();
    }
}
