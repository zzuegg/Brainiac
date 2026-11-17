package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ToyAudioCodecTest {

    @Test
    void moreStagesMeanHigherFidelity() {
        ToyAudioCodec codec = ToyAudioCodec.of(32, 8, 4, 32, 1);
        SplittableRandom r = new SplittableRandom(2);
        codec.restart(codec.frames(512, r));
        for (int step = 0; step < 500; step++) {
            codec.trainStep(codec.frames(64, r));
            if (step % 100 == 99 && step < 400) {
                codec.restart(codec.frames(512, r));
            }
        }
        float[][] probe = codec.frames(256, r);
        double[] snr = new double[4];
        for (int s = 1; s <= 4; s++) {
            snr[s - 1] = codec.snr(probe, s);
        }
        System.out.printf("audio codec SNR dB by RVQ stages 1-4: %.2f %.2f %.2f %.2f%n", snr[0], snr[1], snr[2], snr[3]);
        assertEquals(4, codec.encode(probe, 4).length);
        assertTrue(snr[3] > snr[0] + 3, "SNR by stages " + java.util.Arrays.toString(snr));
        assertTrue(snr[3] > 6, "4 stages (20 bits per 32 samples) SNR " + snr[3]);
    }
}
