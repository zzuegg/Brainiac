package io.github.zzuegg.brainiac.experiments.audio;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class CodecQuantizerTest {

    @Test
    void finiteScalarQuantizationAgainstVqAndRvq() {
        List<AudioGeneration.CodecRow> rows = AudioGeneration.codecs(120, 1);
        StringBuilder line = new StringBuilder("9.5.13 toy codec, SNR dB / perplexity / code use:");
        for (AudioGeneration.CodecRow r : rows) {
            line.append(String.format(" %s %.1f/%.0f/%.2f;", r.quantizer(), r.snr(), r.perplexity(),
                    r.utilisation()));
            assertTrue(Double.isFinite(r.snr()), line.toString());
        }
        System.out.println(line);
    }
}
