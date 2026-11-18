package io.github.zzuegg.brainiac.experiments.multimodal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class FusionShootoutTest {

    @Test
    void bridgesToAFrozenLanguageModel() {
        FusionShootout.VqaSetup setup = FusionShootout.vqaSetup(300, 200, 512, 200, 1);
        double frozen = setup.textLoss();
        Map<FusionShootout.Bridge, FusionShootout.VqaScores> all = new EnumMap<>(FusionShootout.Bridge.class);
        for (FusionShootout.Bridge b : FusionShootout.Bridge.values()) {
            FusionShootout.VqaScores s = FusionShootout.vqa(setup, b, 150, 1);
            all.put(b, s);
            System.out.printf("VQA %-12s acc %.3f (count %.3f exists %.3f left %.3f) trainable %d text loss %.3f"
                    + " (frozen LM %.3f) %d ms%n", b, s.accuracy(), s.count(), s.exists(), s.left(), s.trainable(),
                    s.textLoss(), frozen, s.millis());
        }
        assertEquals(frozen, all.get(FusionShootout.Bridge.GATED).textLoss(), 1e-6, "the frozen LM is intact");
        assertTrue(all.get(FusionShootout.Bridge.PROJECTOR).accuracy()
                >= all.get(FusionShootout.Bridge.TEXT_ONLY).accuracy() - 0.05, "seeing the image does not hurt");
    }

    @Test
    void transfusionAgainstVqTokens() {
        List<FusionShootout.GenerationScores> rows = FusionShootout.generation(200, 3, 1);
        for (FusionShootout.GenerationScores s : rows) {
            System.out.printf("Generation %-46s caption %.3f generation %.3f distance %.3f params %d (%d ms)%n",
                    s.method(), s.captionAccuracy(), s.generationAccuracy(), s.distance(), s.parameters(), s.millis());
        }
        assertTrue(rows.get(0).captionAccuracy() > 0.15, "Transfusion learns to caption above chance");
    }

    @Test
    void bottleneckFusionAndModalityDropout() {
        List<FusionShootout.AvScores> rows = FusionShootout.audioVisual(200, 0.25, 0.6, 1);
        for (FusionShootout.AvScores s : rows) {
            System.out.printf("Audio-visual %-36s both %.3f image only %.3f audio only %.3f MACs %d%n", s.method(),
                    s.both(), s.imageOnly(), s.audioOnly(), s.macs());
        }
        assertTrue(rows.get(0).both() > 0.3, "bottleneck fusion learns");
        assertTrue(rows.get(0).macs() < rows.get(2).macs(), "bottlenecks are cheaper than full attention");
    }
}
