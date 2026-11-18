package io.github.zzuegg.brainiac.experiments.audio;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class FlowMatchingTtsTest {

    @Test
    void monotonicAlignmentAgainstFillerPadding() {
        List<AudioGeneration.TtsRow> rows = AudioGeneration.tts(60, 4, 4, 1);
        StringBuilder line = new StringBuilder("9.5.11 TTS, DER 1-3 / 4-5 digits, boundary err (frames):");
        for (AudioGeneration.TtsRow r : rows) {
            line.append(String.format(" %s %.2f/%.2f %.2f;", r.model(), r.derShort(), r.derLong(),
                    r.alignmentError()));
            assertTrue(Double.isFinite(r.derShort()) && Double.isFinite(r.derLong()), line.toString());
        }
        System.out.println(line);
    }
}
