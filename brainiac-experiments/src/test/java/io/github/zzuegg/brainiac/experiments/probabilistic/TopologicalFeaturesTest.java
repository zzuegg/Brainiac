package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.ml.PersistentHomology;

import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class TopologicalFeaturesTest {

    @Test
    void barcodesSeeLoops() {
        double[][] circle = TopologicalFeatures.cloud(1, 60, 0.02, new SplittableRandom(1));
        PersistentHomology.Barcode b = PersistentHomology.rips(circle, 2.5);
        long loops = b.h1().stream().filter(p -> p.persistence() > 0.5).count();
        assertEquals(1, loops, "one long H1 bar on a circle");
        assertEquals(59, b.h0().size());
        List<PersistentHomology.Pair> d = PersistentHomology.cap(b.h1(), 2.5);
        assertEquals(0, PersistentHomology.bottleneckDistance(d, d), 1e-12);
    }

    @Test
    void topologyClassifiesUnderRotation() {
        Map<String, double[]> t = TopologicalFeatures.timing(new int[] {50, 100, 150}, 1);
        System.out.printf("Rips barcode {ms, H1 pairs}: %s%n", ProbabilisticProgramsTest.line(t));
        Map<String, double[]> r = TopologicalFeatures.classify(24, 2);
        System.out.printf("Topology classes {acc, rotated acc}: %s%n", ProbabilisticProgramsTest.line(r));
        assertTrue(r.get("barcode features + MLP")[1] > 0.8, "barcodes are rotation invariant");
        assertTrue(r.get("barcode features + MLP")[1] > r.get("raw-point MLP")[1], "beats raw points under rotation");
    }
}
