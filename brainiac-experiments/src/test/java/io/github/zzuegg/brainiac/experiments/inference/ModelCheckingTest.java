package io.github.zzuegg.brainiac.experiments.inference;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class ModelCheckingTest {

    @Test
    void psisLooAndStacking() {
        Map<String, double[]> loo = ModelChecking.loo(40, 3, 2000, 1);
        System.out.printf("PSIS-LOO (13.5.12) {|elpd - exact|, elpd}: %s%n", NeuralLikelihoodTest.line(loo));
        Map<String, double[]> st = ModelChecking.stacking(60, 1000, 1000, 2);
        System.out.printf("Stacking M-open (13.5.12) {held-out log score, w_linear}: %s%n",
                NeuralLikelihoodTest.line(st));
        assertTrue(loo.get("PSIS-LOO")[0] < loo.get("IS-LOO")[0] + 0.5, "PSIS no worse than plain IS");
        assertTrue(loo.get("k-hat>0.7 {recall, false flags}")[0] >= 0.5, "k-hat flags the outliers");
        assertTrue(st.get("stacking")[0] >= st.get("BMA")[0] - 0.02, "stacking at least matches BMA");
    }

    @Test
    void amortisedModelComparison() {
        Map<String, double[]> m = ModelChecking.amortised(4000, 1000, 3);
        System.out.printf("Amortised comparison (13.5.16) {|lbf err| <5, 5-20, >20, accuracy, ECE}: %s%n",
                NeuralLikelihoodTest.line(m));
        assertTrue(m.get("EVIDENCE_NETWORK")[3] > 0.7, "the evidence network chooses the model");
        assertTrue(m.get("CLASSIFIER")[3] > 0.7, "the classifier chooses the model");
    }
}
