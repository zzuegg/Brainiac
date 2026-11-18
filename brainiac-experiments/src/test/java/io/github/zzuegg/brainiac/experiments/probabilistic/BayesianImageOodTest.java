package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class BayesianImageOodTest {

    @Test
    void singleModelUncertaintyAgainstEnsembles() {
        Map<String, double[]> r = BayesianNeuralNetworks.imageOod(1);
        System.out.printf("ShapeImages OOD {acc, NLL, ECE, AUROC, passes, joint NLL}: %s%n",
                ProbabilisticProgramsTest.line(r));
        r.forEach((k, v) -> assertTrue(v[0] > 0.6, k + " accuracy"));
        assertTrue(r.get("DDU")[3] > 0.6, "DDU detects OOD");
        assertTrue(r.get("epinet (10 indices)")[5] < r.get("MAP")[5], "the epinet's joint predictions beat MAP");
    }
}
