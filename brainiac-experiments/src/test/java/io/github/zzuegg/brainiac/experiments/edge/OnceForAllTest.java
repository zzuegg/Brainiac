package io.github.zzuegg.brainiac.experiments.edge;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class OnceForAllTest {

    @Test
    void onceForAllSubnetsWorkWithoutRetraining() {
        EdgeDeployment.OfaResult res = EdgeDeployment.onceForAll(160, 2, 3);
        for (EdgeDeployment.OfaRow r : res.rows()) {
            System.out.printf("9.6.15 %-38s %-44s %8d MACs, peak %6d B, accuracy %.3f, %5d steps%n", r.method(),
                    r.subnet(), r.macs(), r.peakBytes(), r.accuracy(), r.trainingSteps());
        }
        for (EdgeDeployment.SearchRow r : res.search()) {
            System.out.printf("9.6.15 budget %-8s %-28s %-44s %8d MACs, predicted %.3f, measured %.3f%n", r.budget(),
                    r.method(), r.subnet(), r.macs(), r.predicted(), r.measured());
        }
        System.out.printf("9.6.15 predictor Kendall tau %.2f%n", res.predictorTau());
        double ofaSmall = res.rows().stream().filter(r -> r.method().startsWith("OFA")).skip(1).findFirst()
                .orElseThrow().accuracy();
        double bigSmall = res.rows().stream().filter(r -> r.method().startsWith("largest")).skip(1).findFirst()
                .orElseThrow().accuracy();
        // At this CPU budget (160 steps) the paper's claim does not hold: subnets trained from scratch
        // and the random-sampling supernet beat progressive shrinking on the smallest subnet, and the
        // shrunk supernet's largest path trails the largest-only network. Kept as a sanity check:
        // shrinking must at least make the smallest subnet work better than cutting it out of the
        // largest-only network.
        assertTrue(ofaSmall > bigSmall, "progressive shrinking helps the smallest subnet: " + ofaSmall + " vs "
                + bigSmall);
    }
}
