package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class ObjectDiscoveryTest {

    @Test
    void slotsSegmentWithoutLabels() {
        double[] slots = ObjectDiscovery.run(true, 600, 1);
        double[] plain = ObjectDiscovery.run(false, 600, 1);
        double kmeans = ObjectDiscovery.kMeans(1);
        System.out.printf("Object discovery: slot attention FG-ARI %.3f MSE %.4f; plain AE MSE %.4f;"
                + " spatial k-means FG-ARI %.3f%n", slots[1], slots[0], plain[0], kmeans);
        assertTrue(slots[1] > 0.2, "the slots' alpha masks separate the shapes better than chance");
        assertTrue(slots[1] > kmeans, "and better than clustering pixels by intensity and position");
    }
}
