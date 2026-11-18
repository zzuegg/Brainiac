package io.github.zzuegg.brainiac.experiments.speech;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.speech.SpeakerShootout.DiarizationRow;
import io.github.zzuegg.brainiac.experiments.speech.SpeakerShootout.Diarizer;

import java.util.List;

import org.junit.jupiter.api.Test;

class SpeakerDiarizationTest {

    @Test
    void clusteringAndEendAgainstOneSpeaker() {
        List<DiarizationRow> r = SpeakerShootout.diarization(2, 0.0, 2, 30, 1, Diarizer.ONE_SPEAKER,
                Diarizer.MFCC_SPECTRAL, Diarizer.XVECTOR_SPECTRAL, Diarizer.XVECTOR_ORACLE, Diarizer.EEND);
        System.out.printf("diarisation, 2 speakers, no overlap, 30 steps: %s%n", r);
        assertTrue(r.get(0).der() > 0.2, "one speaker for everything confuses half the speech: " + r.get(0));
        for (DiarizationRow row : r) {
            assertTrue(Double.isFinite(row.der()), row.toString());
        }
    }
}
