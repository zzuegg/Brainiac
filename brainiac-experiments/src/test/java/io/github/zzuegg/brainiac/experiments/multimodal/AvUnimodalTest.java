package io.github.zzuegg.brainiac.experiments.multimodal;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.multimodal.BottleneckFusion;

import java.util.List;

import org.junit.jupiter.api.Test;

class AvUnimodalTest {

    @Test
    void fusesSightAndSound() {
        List<FusionShootout.AvScores> rows = FusionShootout.audioVisual((BottleneckFusion.Mode) null, 150, 0.15, 0.6, 1);
        for (FusionShootout.AvScores s : rows) {
            System.out.printf("Audio-visual %-36s both %.3f image only %.3f audio only %.3f MACs %d%n", s.method(),
                    s.both(), s.imageOnly(), s.audioOnly(), s.macs());
        }
        assertTrue(rows.get(1).both() > 0.2, "audio alone beats chance (0.1)");
    }
}
