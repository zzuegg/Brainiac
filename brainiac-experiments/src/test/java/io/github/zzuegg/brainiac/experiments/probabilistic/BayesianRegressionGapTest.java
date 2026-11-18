package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class BayesianRegressionGapTest {

    @Test
    void weightPosteriorsAgainstHmc() {
        Map<String, double[]> r = BayesianNeuralNetworks.regressionGap(1);
        System.out.printf("BNN gap {NLL, gap NLL, cov data, cov gap, TV to HMC}: %s%n",
                ProbabilisticProgramsTest.line(r));
        assertTrue(r.get("HMC")[3] > 0.5, "HMC covers the gap");
        r.forEach((k, v) -> assertTrue(Double.isFinite(v[0]), k));
    }

    @Test
    void coldPosteriorSweep() {
        Map<String, double[]> r = BayesianNeuralNetworks.coldPosterior(2);
        System.out.printf("BNN cold posterior {NLL, gap NLL, cov data, cov gap}: %s%n",
                ProbabilisticProgramsTest.line(r));
        assertTrue(Double.isFinite(r.get("T=0.1")[0]));
    }
}
