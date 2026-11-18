package io.github.zzuegg.brainiac.experiments.audio;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class VocoderShootoutTest {

    @Test
    void ganAndDiffusionVocodersAgainstGriffinLim() {
        List<AudioGeneration.VocoderRow> rows = AudioGeneration.vocoders(6, 10, 2, new int[] {6}, 1);
        StringBuilder line = new StringBuilder("9.5.9/9.5.10 vocoders, MR-STFT / DER / ms per audio s:");
        for (AudioGeneration.VocoderRow r : rows) {
            line.append(String.format(" %s %.2f/%.2f/%.0f;", r.vocoder(), r.stftDistance(), r.digitErrorRate(),
                    r.millisPerSecond()));
            assertTrue(Double.isFinite(r.stftDistance()) && Double.isFinite(r.digitErrorRate()), line.toString());
        }
        System.out.println(line);
    }
}
