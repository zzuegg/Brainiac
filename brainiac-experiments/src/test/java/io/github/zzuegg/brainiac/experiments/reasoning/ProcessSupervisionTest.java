package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/** Wave 9, 9.2.9 to 9.2.14: process supervision for step-by-step math. */
class ProcessSupervisionTest {

    private static final StepChainTask TASK = new StepChainTask(2, 4, 9);

    @Test
    void stepChainTaskIsConsistent() {
        int[] p = TASK.samplePrompt(new SplittableRandom(3));
        int[] s = TASK.solution(p);
        assertTrue(TASK.correct(p, s));
        float[] labels = TASK.stepLabels(p, s);
        for (float l : labels) {
            assertEquals(1f, l);
        }
        assertFalse(TASK.correct(p, TASK.inject(p, 0, new SplittableRandom(4))));
    }

    @Test
    void vinePpoAgainstACritic() {
        ProcessSupervision.Setup setup = ProcessSupervision.setup(TASK, 1);
        ProcessSupervision.ValueError[] errors = new ProcessSupervision.ValueError[1];
        Map<String, ProcessSupervision.VineRow> r = ProcessSupervision.vine(setup, 15, 1, errors);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, row) -> sb.append(String.format(" %s %.3f (%d tokens);", k, row.accuracy(), row.tokens())));
        System.out.printf("9.2.9 VinePPO test accuracy (PPO with critic is the baseline, SFT %.3f):%s%n",
                ProcessSupervision.accuracy(setup.base(), TASK, setup.test()), sb);
        assertTrue(r.size() >= 2);
    }

    @Test
    void mathShepherdLabels() {
        ProcessSupervision.Setup setup = ProcessSupervision.setup(TASK, 2);
        ProcessSupervision.ShepherdResult r = ProcessSupervision.shepherd(setup, 60, 8, 2);
        System.out.printf("9.2.10 Math-Shepherd step-label agreement soft %.3f hard %.3f; best-of-16 %s%n",
                r.softAgreement(), r.hardAgreement(), r.bestOf16());
        assertTrue(r.softAgreement() >= 0 && r.softAgreement() <= 1);
    }

    @Test
    void progressRewards() {
        ProcessSupervision.Setup setup = ProcessSupervision.setup(TASK, 3);
        ProcessSupervision.ProgressResult r = ProcessSupervision.progress(setup, 30, 3, 1f, 3);
        StringBuilder sb = new StringBuilder();
        r.accuracy().forEach((k, a) -> sb.append(String.format(" %s %.3f;", k, a[a.length - 1])));
        System.out.printf("9.2.11 PAV final accuracy (outcome-only GRPO is the baseline):%s beam %s%n", sb, r.beam());
        assertFalse(r.accuracy().isEmpty());
    }

    @Test
    void primeImplicitRewards() {
        ProcessSupervision.Setup setup = ProcessSupervision.setup(TASK, 4);
        Map<String, ProcessSupervision.PrimeRow> r = ProcessSupervision.prime(setup, 30, 3, 4);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, row) -> sb.append(String.format(" %s %.3f;", k, row.accuracy()[row.accuracy().length - 1])));
        System.out.printf("9.2.12 PRIME final accuracy (outcome-only RLOO is the baseline):%s%n", sb);
        assertTrue(r.size() >= 2);
    }

    @Test
    void stepDpo() {
        ProcessSupervision.Setup setup = ProcessSupervision.setup(TASK, 5);
        Map<String, ProcessSupervision.StepDpoRow> r = ProcessSupervision.stepDpo(setup, 120, 60, 5);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, row) -> sb.append(String.format(" %s %.3f (short %.3f long %.3f, %d pairs);", k, row.accuracy(),
                row.shortChains(), row.longChains(), row.pairs())));
        System.out.printf("9.2.13 Step-DPO accuracy (response-level DPO is the baseline):%s%n", sb);
        assertTrue(r.size() >= 2);
    }
}
