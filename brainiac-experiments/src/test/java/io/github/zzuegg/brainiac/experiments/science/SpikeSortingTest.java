package io.github.zzuegg.brainiac.experiments.science;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class SpikeSortingTest {

    @Test
    void sortersSeparateUnitsAndTemplateMatchingResolvesOverlaps() {
        Map<String, double[]> r = Neuroscience.sorting(4, 4, 20, 0.15, false, 3);
        System.out.printf("13.3.17 spike sorting 4 ch x 4 units {accuracy, FP rate, FN rate, overlap recall,"
                + " refractory violations}: %s%n", Lines.of(r));
        Map<String, double[]> s = Neuroscience.sorting(4, 4, 20, 0.15, true, 2);
        System.out.printf("13.3.17 spike sorting, similar template pairs: %s%n", Lines.of(s));
        assertTrue(r.get("PCA_GMM")[0] > 0.4, Lines.of(r));
        assertTrue(s.get("TEMPLATE_MATCHING")[3] > s.get("PCA_GMM")[3], Lines.of(s));
        assertTrue(r.get("TEMPLATE_MATCHING")[3] >= r.get("PCA_GMM")[3], Lines.of(r));
    }
}
