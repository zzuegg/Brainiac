package io.github.zzuegg.brainiac.experiments.science;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class ChirpSearchTest {

    @Test
    void matchedFilterDetectsChirpsAndVetoesGlitches() {
        Map<String, double[]> r = SignalSearch.chirps(300, 30, 600, 250, 1);
        System.out.printf("13.3.14 chirps {Pd@FAR1e-2 SNR 4,6,8,10,12, Pd@FAR1/300 SNR8, Pd out-of-range SNR8,"
                + " glitch FA, us per s}: %s%n", Lines.of(r));
        assertTrue(r.get("matched filter")[3] > 0.8, Lines.of(r));
        assertTrue(r.get("matched filter + chi2 veto")[7] <= r.get("matched filter")[7], Lines.of(r));
    }
}
