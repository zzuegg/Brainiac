package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Map;
import java.util.SplittableRandom;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Wave 9, 9.2.8 and 9.2.9: VinePPO and Math-Shepherd on step-by-step running sums. */
class ProcessSupervisionTest {

    private static final StepChainTask TASK = new StepChainTask(2, 3, 4);
    private static ProcessSupervision.Setup setup;

    @BeforeAll
    static void start() {
        setup = ProcessSupervision.setup(TASK, 120, 1);
    }

    @Test
    void stepChainTaskIsConsistent() {
        int[] p = TASK.samplePrompt(new SplittableRandom(3));
        int[] s = TASK.solution(p);
        assertTrue(TASK.correct(p, s));
        for (float l : TASK.stepLabels(p, s)) {
            assertEquals(1f, l);
        }
        assertFalse(TASK.correct(p, TASK.inject(p, 0, new SplittableRandom(4))));
    }

    @Test
    void vinePpoAgainstACritic() {
        ProcessSupervision.ValueError[] errors = new ProcessSupervision.ValueError[1];
        Map<String, ProcessSupervision.VineRow> r = ProcessSupervision.vine(setup, 30, 1, errors);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, row) -> sb.append(String.format(" %s %.3f (%d tok);", k, row.accuracy(), row.tokens())));
        System.out.printf("9.2.8 VinePPO accuracy (PPO with critic is the baseline):%s value MSE vs MC64 by step:"
                + " critic %s MC4 %s constant %s%n", sb, fmt(errors[0].criticByStep()),
                fmt(errors[0].monteCarloByStep()), fmt(errors[0].constantByStep()));
        assertTrue(r.size() >= 2);
    }

    @Test
    void mathShepherdLabels() {
        ProcessSupervision.ShepherdResult r = ProcessSupervision.shepherd(setup, 80, 4, 2);
        System.out.printf("9.2.9 Math-Shepherd step-label agreement soft %.3f hard %.3f; best-of-16 %s%n",
                r.softAgreement(), r.hardAgreement(), r.bestOf16());
        assertTrue(r.softAgreement() >= 0 && r.softAgreement() <= 1);
    }

    private static String fmt(double[] xs) {
        StringBuilder sb = new StringBuilder("[");
        Arrays.stream(xs).forEach(x -> sb.append(String.format(" %.3f", x)));
        return sb.append(" ]").toString();
    }
}
