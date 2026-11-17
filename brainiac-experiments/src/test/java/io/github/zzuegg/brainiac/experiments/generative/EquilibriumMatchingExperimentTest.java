package io.github.zzuegg.brainiac.experiments.generative;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class EquilibriumMatchingExperimentTest {

    @Test
    void equilibriumMatchingSamplesByDescendingALearnedLandscape() {
        Map<String, Double> r = EquilibriumMatchingExperiment.run(1500, new int[] {1, 2, 4, 16}, 1);
        StringBuilder line = new StringBuilder("ToyPoints ring, distance to nearest centre by steps:");
        for (String m : new String[] {"flow matching", "consistency", "EqM"}) {
            line.append(String.format(" %s %.3f/%.3f/%.3f/%.3f;", m, r.get(m + " 1"), r.get(m + " 2"), r.get(m + " 4"),
                    r.get(m + " 16")));
        }
        line.append(String.format(" EqM adaptive %.3f in %.1f steps", r.get("EqM adaptive"),
                r.get("EqM adaptive steps")));
        System.out.println(line);
        assertTrue(r.get("EqM 16") < 0.2, "descent reaches the blobs: " + r);
        assertTrue(r.get("EqM 4") < r.get("flow matching 4"), "and does so in few steps: " + r);
        assertTrue(r.get("EqM adaptive") < 0.2, r.toString());
    }
}
