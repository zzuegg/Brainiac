package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Map;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Wave 9, 9.2.12: Step-DPO against solution-level DPO, RFT and STaR. */
class StepDpoTest {

    private static final StepChainTask TASK = new StepChainTask(2, 3, 4);
    private static ProcessSupervision.Setup setup;

    @BeforeAll
    static void start() {
        setup = ProcessSupervision.setup(TASK, 120, 1);
    }

    @Test
    void stepDpo() {
        Map<String, ProcessSupervision.StepDpoRow> r = ProcessSupervision.stepDpo(setup, 120, 40, 5);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, row) -> sb.append(String.format(" %s %.3f (1-step %.3f 2-step %.3f, %d pairs);", k,
                row.accuracy(), row.shortChains(), row.longChains(), row.pairs())));
        System.out.printf("9.2.12 Step-DPO accuracy (solution-level DPO is the baseline):%s%n", sb);
        assertTrue(r.size() >= 2);
    }

    private static String fmt(double[] xs) {
        StringBuilder sb = new StringBuilder("[");
        Arrays.stream(xs).forEach(x -> sb.append(String.format(" %.3f", x)));
        return sb.append(" ]").toString();
    }
}
