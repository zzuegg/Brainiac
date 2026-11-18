package io.github.zzuegg.brainiac.experiments.algorithms;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.KeyStreams;
import io.github.zzuegg.brainiac.experiments.algorithms.LearnedStructures.IndexRow;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

/** Item 11.2.16: learned indexes and learned/sandwiched Bloom filters. */
class LearnedStructuresTest {

    static IndexRow find(List<IndexRow> rows, String method) {
        return rows.stream().filter(r -> r.method().equals(method)).findFirst().orElseThrow();
    }

    @Test
    void learnedIndexesAndFilters() {
        StringBuilder line = new StringBuilder("11.2.16");
        boolean allExact = true;
        for (KeyStreams.KeyDistribution d : KeyStreams.KeyDistribution.values()) {
            List<IndexRow> rows = LearnedStructures.indexes(200_000, d, 100_000, 7);
            System.out.println(d + " 200k keys: " + rows);
            allExact &= rows.stream().allMatch(IndexRow::exact);
            line.append(String.format(" %s cmp/B: binary %.1f, B-tree %.1f/%d, RMI-lin %.1f/%d, PLA-64 %.1f/%d;", d,
                    find(rows, "binary search").comparisons(), find(rows, "B-tree fan-out 64").comparisons(),
                    find(rows, "B-tree fan-out 64").bytes(), find(rows, "RMI linear 1024").comparisons(),
                    find(rows, "RMI linear 1024").bytes(), find(rows, "PLA eps 64").comparisons(),
                    find(rows, "PLA eps 64").bytes()));
        }
        List<List<String>> urls = LearnedStructures.urls(16, 600, 8);
        Map<Double, double[]> bloom = LearnedStructures.bloomFilters(urls.get(0), urls.get(1), new double[] {6, 10},
                256, 8);
        bloom.forEach((b, v) -> System.out.printf("Bloom %.0f bits/key (%d keys): plain FPR %.4f, learned %.4f,"
                + " sandwiched %.4f, classifier alone %.4f%n", b, urls.get(0).size(), v[0], v[1], v[2], v[3]));
        double[] at8 = bloom.get(10.0);
        line.append(String.format(" Bloom 10 bits/key FPR plain %.4f, learned %.4f, sandwiched %.4f", at8[0], at8[1],
                at8[2]));
        System.out.println(line);
        assertTrue(allExact, "every index finds every key");
        assertTrue(at8[0] < 0.1 && at8[2] < 1, "the filters run without false negatives");
    }
}
