package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.vision.VisionShootout.FlexiRow;

import java.util.List;

import org.junit.jupiter.api.Test;

class FlexiVitTest {

    @Test
    void oneVitForEveryPatchSize() {
        List<FlexiRow> r = VisionShootout.flexiViT(16, 150, false, 1);
        System.out.printf("FlexiViT on 16x16 shapes, 150 steps: %s%n", r);
        for (int i = 0; i < 3; i++) {
            assertTrue(r.get(i).accuracy() > 0.25, "FlexiViT works at every patch size: " + r.get(i));
            assertTrue(r.get(i).accuracy() > r.get(i + 3).accuracy(),
                    "one FlexiViT beats separate models on a third of the budget: " + r.get(i) + " vs " + r.get(i + 3));
        }
    }
}
