package io.github.zzuegg.brainiac.experiments.science;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class TransitSearchTest {

    @Test
    void boxLeastSquaresRecoversPeriodsAndTheVetterSeparatesBinaries() {
        Map<String, double[]> r = SignalSearch.transits(21, 160, 80, 150, 1);
        System.out.printf("13.3.15 transits: %s%n", Lines.of(r));
        double[] bls = r.get("recovered period within 1% {SNR 4-7, 7-10, 10-20}: BLS + moving median");
        double[] fft = r.get("recovered period within 1% {SNR 4-7, 7-10, 10-20}: FFT periodogram");
        assertTrue(bls[2] > 0.6 && bls[2] > fft[2], Lines.of(r));
        assertTrue(r.get("vetting {AUROC, precision at 95% recall}: two-view CNN")[0] > 0.7, Lines.of(r));
    }
}
