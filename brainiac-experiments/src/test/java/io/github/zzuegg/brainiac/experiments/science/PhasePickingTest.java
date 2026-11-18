package io.github.zzuegg.brainiac.experiments.science;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class PhasePickingTest {

    @Test
    void pickersFindPhaseArrivals() {
        Map<String, double[]> r = SignalSearch.picking(50, 300, 16, 200, 1);
        System.out.printf("13.3.13 phase picking {P recall, P prec, P sd s, S recall, S prec, S sd s, noise false"
                + " picks, S recall SNR<=0}: %s%n", Lines.of(r));
        // only the 10 and 20 dB fifths are reliably pickable on this SNR scale
        assertTrue(r.get("AR-AIC")[0] > 0.15, Lines.of(r));
        assertTrue(r.get("AR-AIC")[2] < r.get("STA/LTA")[2], Lines.of(r));
        assertTrue(r.get("PhaseNet-lite U-Net")[3] > r.get("STA/LTA")[3], Lines.of(r));
    }
}
