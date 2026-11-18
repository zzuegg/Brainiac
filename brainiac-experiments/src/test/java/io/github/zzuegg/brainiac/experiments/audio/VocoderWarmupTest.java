package io.github.zzuegg.brainiac.experiments.audio;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.audio.GanVocoder;

import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** A short smoke run of the 10.6.16 vocoder follow-up; the long training is a main() Run. */
class VocoderWarmupTest {

    @TempDir
    Path dir;

    @Test
    void warmupCheckpointAndResume() {
        Path ckpt = dir.resolve("gan.ckpt");
        List<AudioGeneration.VocoderRow> rows = AudioGeneration.vocoderFollowUp(10, new int[] {10, 20}, 2, ckpt,
                false, 1);
        for (AudioGeneration.VocoderRow r : rows) {
            System.out.printf("%-48s STFT %.3f DER %.3f %.1f ms/s%n", r.vocoder(), r.stftDistance(), r.digitErrorRate(),
                    r.millisPerSecond());
        }
        assertTrue(rows.size() == 7, rows.toString());
        GanVocoder resumed = GanVocoder.of(AudioGeneration.MEL, 32, 51).load(ckpt);
        assertEquals(20, resumed.trainedSteps(), "the checkpoint keeps the step counter");
        GanVocoder again = GanVocoder.of(AudioGeneration.MEL, 32, 7).load(ckpt);
        float[][] mel = new float[8][AudioGeneration.MEL.mels()];
        assertArrayEquals(resumed.vocode(mel, 8 * AudioGeneration.MEL.hop()),
                again.vocode(mel, 8 * AudioGeneration.MEL.hop()), 0f, "a loaded checkpoint is the same vocoder");
    }
}
