package io.github.zzuegg.brainiac.experiments.data;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.data.SyntheticTraining.Augmentation;
import io.github.zzuegg.brainiac.experiments.data.SyntheticTraining.GuidancePoint;

import java.util.List;

import org.junit.jupiter.api.Test;

class DiffusionAugmentationTest {

    @Test
    void guidanceSweepOfDiffusionSamples() {
        double real = SyntheticTraining.augmentation(Augmentation.REAL, 5, 1f, 0, 2, 1).accuracy();
        List<GuidancePoint> pts = SyntheticTraining.guidanceSweep(5, new float[] {1f, 3f}, 1500, 2, 1);
        System.out.printf("diffusion augmentation 5 img/class: real only %.3f; w=1 acc %.3f P %.3f R %.3f; "
                + "w=3 acc %.3f P %.3f R %.3f%n", real, pts.get(0).accuracy(), pts.get(0).precision(),
                pts.get(0).recall(), pts.get(1).accuracy(), pts.get(1).precision(), pts.get(1).recall());
        for (GuidancePoint p : pts) {
            assertTrue(p.accuracy() > 0.12, "above chance");
            assertTrue(Math.abs(p.accuracy() - real) < 0.1, "samples neither rescue nor wreck the classifier");
            assertTrue(p.precision() >= 0 && p.precision() <= 1 && p.recall() >= 0 && p.recall() <= 1);
        }
    }
}
