package io.github.zzuegg.brainiac.experiments.music;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class PitchShootoutTest {

    @Test
    void classicalSupervisedAndSelfSupervisedPitch() {
        List<PitchShootout.Row> rows = PitchShootout.run(
                new PitchShootout.Budget(10, 2, 2.0, 220, new double[] {20, 5}), 11);
        String cond = "";
        StringBuilder line = new StringBuilder();
        for (PitchShootout.Row r : rows) {
            if (!r.condition().equals(cond)) {
                if (!line.isEmpty()) {
                    System.out.println(line);
                }
                cond = r.condition();
                line = new StringBuilder("11.5.1-2 " + cond + " RPA/RCA/VF1/oct:");
            }
            line.append(String.format(" %s %.2f/%.2f/%.2f/%.2f;", r.method(), r.score().rawPitch(),
                    r.score().rawChroma(), r.score().voicingF1(), r.score().octaveErrors()));
        }
        System.out.println(line);
        PitchShootout.Row yin = rows.stream().filter(r -> r.condition().equals("melody 20dB") && r.method().equals("YIN"))
                .findFirst().orElseThrow();
        assertTrue(yin.score().rawPitch() > 0.7, "YIN raw pitch at 20 dB " + yin.score().rawPitch());
        PitchShootout.Row pesto = rows.stream().filter(r -> r.condition().equals("melody 20dB")
                && r.method().equals("PESTO")).findFirst().orElseThrow();
        assertTrue(pesto.score().rawChroma() > 0.4, "label-free PESTO raw chroma " + pesto.score().rawChroma());
        for (PitchShootout.Row r : rows) {
            assertTrue(Double.isFinite(r.score().rawPitch()), r.toString());
        }
    }
}
