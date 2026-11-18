package io.github.zzuegg.brainiac.experiments.speech;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.speech.SpeakerShootout.VerificationRow;
import io.github.zzuegg.brainiac.experiments.speech.SpeakerShootout.Verifier;

import java.util.List;

import org.junit.jupiter.api.Test;

class SpeakerVerificationTest {

    @Test
    void xvectorsAgainstMfccAndGmmUbm() {
        List<VerificationRow> r = SpeakerShootout.verification(40, 12, 6, 20, 120, 0, 1, Verifier.MEAN_MFCC,
                Verifier.GMM_UBM, Verifier.XVECTOR_AAM, Verifier.GE2E);
        System.out.printf("speaker verification, 40 train / 12 held-out speakers, 20 dB, 120 steps: %s%n", r);
        assertTrue(r.get(0).eer() < 0.4, "the voices are separable by a mean-MFCC baseline: " + r.get(0));
        assertTrue(r.get(2).eer() < 0.45, "the x-vector learns speakers: " + r.get(2));
    }
}
