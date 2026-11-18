package io.github.zzuegg.brainiac.tensor.spi;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

/** The double-precision transforms match a direct DFT and round-trip to 1e-12. */
class FftTest {

    @Test
    void doublePrecisionMatchesTheDefinition() {
        for (int n : new int[] {8, 12, 17}) {
            double[] re = new double[n];
            double[] im = new double[n];
            for (int j = 0; j < n; j++) {
                re[j] = Math.cos(0.3 * j * j) + 1e-9 * j;
                im[j] = Math.sin(0.7 * j);
            }
            double[] wantRe = new double[n];
            double[] wantIm = new double[n];
            for (int k = 0; k < n; k++) {
                for (int j = 0; j < n; j++) {
                    double a = -2 * Math.PI * j * k / n;
                    wantRe[k] += re[j] * Math.cos(a) - im[j] * Math.sin(a);
                    wantIm[k] += re[j] * Math.sin(a) + im[j] * Math.cos(a);
                }
            }
            double[] data = new double[2 * n];
            for (int j = 0; j < n; j++) {
                data[2 * j] = re[j];
                data[2 * j + 1] = im[j];
            }
            double[] fr = re.clone();
            double[] fi = im.clone();
            Fft.transform(fr, fi, false);
            assertArrayEquals(wantRe, fr, 1e-10);
            assertArrayEquals(wantIm, fi, 1e-10);
            Fft.transform(data, 1, n, false);
            Fft.transform(data, 1, n, true);
            for (int j = 0; j < n; j++) {
                assertEquals(re[j], data[2 * j], 1e-12);
                assertEquals(im[j], data[2 * j + 1], 1e-12);
            }
        }
    }
}
