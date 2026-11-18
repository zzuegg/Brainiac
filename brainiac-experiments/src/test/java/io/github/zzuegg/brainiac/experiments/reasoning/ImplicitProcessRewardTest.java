package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Map;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Wave 9, 9.2.11: PRIME's online implicit process reward. */
class ImplicitProcessRewardTest {

    private static final StepChainTask TASK = new StepChainTask(2, 3, 4);
    private static ProcessSupervision.Setup setup;

    @BeforeAll
    static void start() {
        setup = ProcessSupervision.setup(TASK, 120, 1);
    }

    @Test
    void primeImplicitRewards() {
        Map<String, ProcessSupervision.PrimeRow> r = ProcessSupervision.prime(setup, 40, 2, 4);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, row) -> sb.append(String.format(" %s acc %s frozen-PRM %s;", k, fmt(row.accuracy()),
                fmt(row.frozenPrmScore()))));
        System.out.printf("9.2.11 PRIME accuracy by checkpoint (outcome-only RLOO is the baseline):%s%n", sb);
        assertTrue(r.size() >= 2);
    }

    private static String fmt(double[] xs) {
        StringBuilder sb = new StringBuilder("[");
        Arrays.stream(xs).forEach(x -> sb.append(String.format(" %.3f", x)));
        return sb.append(" ]").toString();
    }
}
