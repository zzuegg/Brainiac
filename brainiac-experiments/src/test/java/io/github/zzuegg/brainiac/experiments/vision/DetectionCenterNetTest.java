package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DetectionCenterNetTest {

    @Test
    void centerNetDetectsWithoutAnchorsOrNms() {
        DetectionShootout.Score[] cn = DetectionShootout.centerNet(300, 1);
        DetectionShootout.Score retina = DetectionShootout.anchorCrowded(250, 1);
        System.out.printf("CenterNet mAP@0.5 sparse/crowded %.3f/%.3f (%d MACs, decode %.2f ms/image);"
                + " RetinaNet-lite crowded %.3f (%d MACs, decode %.2f ms/image)%n", cn[0].map50(), cn[1].map50(),
                cn[0].macs(), cn[0].decodeMillis(), retina.map50(), retina.macs(), retina.decodeMillis());
        assertTrue(cn[0].map50() > 0.3, "CenterNet learns to detect");
        assertTrue(cn[1].map50() < cn[0].map50(), "overlapping glyphs are harder");
        assertTrue(cn[0].macs() < retina.macs(), "CenterNet is cheaper than the anchor detector");
    }
}
