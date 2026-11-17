package io.github.zzuegg.brainiac.experiments.safety;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class ObfuscationAtlasTest {

    @Test
    void probePenaltyUnderRlPressure() {
        List<ObfuscationAtlas.Result> out = ObfuscationAtlas.run(200, 0.6f, 1f, 1);
        for (ObfuscationAtlas.Result r : out) {
            System.out.printf("Atlas %-14s honest %.2f  hacks %.2f (HACK2 %.2f)  probe AUROC %.2f  retrained %.2f"
                    + "  -> %s%n", r.condition(), r.honesty(), r.hackRate(), r.hack2Share(), r.originalProbeAuroc(),
                    r.retrainedProbeAuroc(), r.category());
        }
        ObfuscationAtlas.Result start = out.get(0);
        ObfuscationAtlas.Result plain = out.get(1);
        assertTrue(start.originalProbeAuroc() > 0.9, start.toString());
        ObfuscationAtlas.Result penalised = out.get(2);
        ObfuscationAtlas.Result withKl = out.get(3);
        assertTrue(plain.hackRate() > 0.9, plain.toString());
        assertTrue(penalised.hack2Share() > 0.5, penalised.toString());
        assertTrue(withKl.hack2Share() < penalised.hack2Share() && withKl.originalProbeAuroc() > 0.9,
                withKl.toString());
    }
}
