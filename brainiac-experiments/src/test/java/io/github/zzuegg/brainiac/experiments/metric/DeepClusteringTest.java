package io.github.zzuegg.brainiac.experiments.metric;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.ml.DeepEmbeddedClustering;
import io.github.zzuegg.brainiac.experimental.ml.NeighbourClustering;

import java.util.Map;

import org.junit.jupiter.api.Test;

/** Items 10.3.11-10.3.12: DEC, IDEC, IIC and SCAN against k-means, a GMM and spectral clustering. */
class DeepClusteringTest {

    @Test
    void decRefinesAutoencoderClusters() {
        Map<String, DeepClustering.Score> s = DeepClustering.embedded(40, DeepEmbeddedClustering.Settings.small(), 1);
        System.out.println("DEC/IDEC on digits: " + s);
        assertTrue(s.get("DEC").accuracy() > 0.18, "DEC is above chance (0.1)");
        assertTrue(s.get("IDEC").accuracy() > 0.18, "IDEC is above chance (0.1)");
    }

    @Test
    void scanAndIicClusterFromInvariances() {
        Map<String, DeepClustering.Score> s = DeepClustering.neighbourBased(40, 100, 200,
                NeighbourClustering.ScanSettings.small(), 1);
        System.out.println("IIC/SCAN on digits: " + s);
        assertTrue(s.get("SCAN + self-labelling").accuracy() > 0.2, "SCAN is above chance (0.1)");
        assertTrue(s.get("IIC raw").nmi() > 0.1, "IIC finds structure");
    }
}
