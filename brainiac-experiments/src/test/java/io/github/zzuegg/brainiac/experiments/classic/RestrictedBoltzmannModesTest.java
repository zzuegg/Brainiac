package io.github.zzuegg.brainiac.experiments.classic;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class RestrictedBoltzmannModesTest {

    @Test
    void modeCoverageOnPlantedBimodal() {
        List<ClassicLearners.ModeRow> rows = ClassicLearners.rbmModes(800, 1);
        StringBuilder b = new StringBuilder("11.4.12 bimodal share A/B (exact balance):");
        rows.forEach(r -> b.append(String.format(" %s %.2f/%.2f (%.2f);", r.method(), r.shareA(), r.shareB(),
                r.exactBalance())));
        System.out.println(b);
        ClassicLearners.ModeRow pt = rows.stream().filter(r -> r.method().equals("PT")).findFirst().orElseThrow();
        assertTrue(pt.modesHit() == 2, "parallel tempering samples both modes");
        assertTrue(pt.exactBalance() > 0.5, "parallel tempering keeps the two modes' mass balanced");
    }
}
