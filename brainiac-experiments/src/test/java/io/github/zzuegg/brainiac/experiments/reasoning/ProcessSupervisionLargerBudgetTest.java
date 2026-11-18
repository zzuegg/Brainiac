package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** W10 4.16: VinePPO, PRIME and Step-DPO from a held-out start (a small corner of the Run). */
class ProcessSupervisionLargerBudgetTest {

    @Test
    void heldOutStartRunsAllThreeComparisons() {
        ProcessSupervision.LargerBudget r = ProcessSupervision.largerBudget(new StepChainTask(3, 4, 9), 400, 100, 2,
                2, 16, 1);
        StringBuilder b = new StringBuilder();
        r.vine().forEach((k, v) -> b.append(String.format("%s %.3f, ", k, v.accuracy())));
        r.prime().forEach((k, v) -> b.append(String.format("%s %.3f, ", k, v.accuracy()[v.accuracy().length - 1])));
        r.stepDpo().forEach((k, v) -> b.append(String.format("%s %.3f, ", k, v.accuracy())));
        System.out.printf("W10 4.16 held-out start (%d SFT steps, train %.3f, held-out %.3f): %s%n", r.sftSteps(),
                r.startTrain(), r.startHeldOut(), b);
        assertTrue(r.startHeldOut() >= 0 && r.startHeldOut() <= 1, b.toString());
        assertTrue(r.vine().containsKey("VinePPO (MC values)") && r.stepDpo().containsKey("Step-DPO"), b.toString());
    }
}
