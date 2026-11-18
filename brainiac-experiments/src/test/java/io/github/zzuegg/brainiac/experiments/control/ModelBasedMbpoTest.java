package io.github.zzuegg.brainiac.experiments.control;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelBasedMbpoTest {

    @Test
    void mbpo() {
        Map<String, double[]> r = ModelBasedAndHierarchical.mbpo(600, 1, 32, 3);
        System.out.printf("11.6.9 pendulum IQM return at 600 steps: %s%n", ModelBasedAndHierarchical.fmt(r));
        assertTrue(r.get("CEM (true model)")[0] > r.get("SAC (UTD 1)")[0]);
    }
}
