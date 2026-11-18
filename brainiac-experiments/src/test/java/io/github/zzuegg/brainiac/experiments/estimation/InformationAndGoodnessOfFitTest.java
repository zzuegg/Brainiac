package io.github.zzuegg.brainiac.experiments.estimation;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class InformationAndGoodnessOfFitTest {

    private static String flat(Map<String, Double> r) {
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format("%s %.3f; ", k, v)));
        return sb.toString();
    }

    @Test
    void mutualInformationEstimators() {
        Map<String, Double> r = InferenceAndTesting.mutualInformation(2000, 400, 3);
        System.out.printf("mutual information (nats): %s%n", flat(r));
        for (String s : new String[] {"gaussian I=2", "gaussian I=6", "cubic I=2"}) {
            for (String e : new String[] {"KSG k=3", "MINE", "NWJ", "INFONCE", "SMILE"}) {
                assertTrue(Double.isFinite(r.get(s + " / " + e)), flat(r));
            }
        }
        assertTrue(Math.abs(r.get("gaussian I=2 / INFONCE") - 2) < 0.6, flat(r));
        assertTrue(r.get("gaussian I=6 / INFONCE") < Math.log(64) + 1e-6, flat(r));
        assertTrue(r.get("gaussian I=6 / SMILE") > r.get("gaussian I=6 / INFONCE"), flat(r));
    }

    @Test
    void steinAndTwoSampleTests() {
        Map<String, Double> r = InferenceAndTesting.goodnessOfFit(200, 20, 4);
        System.out.printf("rejection rate at 0.05: %s%n", flat(r));
        assertTrue(r.get("KSD (IMQ) / null") <= 0.15, flat(r));
        assertTrue(r.get("FSSD (opt, J=5) / null") <= 0.15, flat(r));
        assertTrue(r.get("KSD (IMQ) / variance 2") > 0.5, flat(r));
        assertTrue(r.get("KSD (IMQ) / laplace") > r.get("MMD (median) / laplace"), flat(r));
    }
}
