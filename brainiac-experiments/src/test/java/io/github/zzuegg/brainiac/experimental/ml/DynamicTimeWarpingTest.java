package io.github.zzuegg.brainiac.experimental.ml;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.nn.Mfcc;

import org.junit.jupiter.api.Test;

class DynamicTimeWarpingTest {

    private static float[][] ramp(int n, double freq) {
        float[][] x = new float[n][2];
        for (int i = 0; i < n; i++) {
            x[i][0] = (float) Math.sin(freq * i / n * 6);
            x[i][1] = (float) Math.cos(freq * i / n * 6);
        }
        return x;
    }

    @Test
    void warpedCopiesAreCloseAndTheDctIsOrthonormal() {
        float[][] a = ramp(30, 1);
        float[][] stretched = ramp(45, 1);
        float[][] other = ramp(30, 2.5);
        double same = DynamicTimeWarping.distance(a, a, 5);
        double warped = DynamicTimeWarping.distance(a, stretched, 5);
        double different = DynamicTimeWarping.distance(a, other, 5);
        float[] d = Mfcc.dctMatrix(16, 16).toArray();
        double worst = 0;
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                double dot = 0;
                for (int m = 0; m < 16; m++) {
                    dot += d[m * 16 + i] * d[m * 16 + j];
                }
                worst = Math.max(worst, Math.abs(dot - (i == j ? 1 : 0)));
            }
        }
        System.out.printf("DTW: self %.3f, time-stretched %.3f, different %.3f; DCT orthonormality error %.1e%n",
                same, warped, different, worst);
        assertEquals(0, same, 1e-9);
        assertTrue(warped < different / 3, "warping absorbs a stretch");
        assertTrue(worst < 1e-5);
    }
}
