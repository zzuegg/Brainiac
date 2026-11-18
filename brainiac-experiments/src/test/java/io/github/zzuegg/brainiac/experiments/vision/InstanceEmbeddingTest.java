package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class InstanceEmbeddingTest {

    @Test
    void embeddingsSeparateTouchingSameClassShapes() {
        DetectionShootout.InstanceScore emb = DetectionShootout.instanceEmbedding(16, 400, 1);
        DetectionShootout.InstanceScore cc = DetectionShootout.instanceComponents(16, 400, 1);
        System.out.printf("touching instances mask AP@0.5/FG-ARI: discriminative embedding %.3f/%.3f,"
                + " U-Net + components %.3f/%.3f (slot attention FG-ARI 0.303 on ObjectDiscovery)%n", emb.maskAp(),
                emb.fgAri(), cc.maskAp(), cc.fgAri());
        assertTrue(emb.fgAri() > cc.fgAri(), "embeddings split touching same-class shapes");
    }
}
