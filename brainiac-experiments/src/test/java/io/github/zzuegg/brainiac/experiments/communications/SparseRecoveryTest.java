package io.github.zzuegg.brainiac.experiments.communications;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Map;
import java.util.stream.Collectors;

class SparseRecoveryTest {

    private static String grid(Map<String, double[][]> m) {
        return m.entrySet().stream()
                .map(e -> e.getKey() + " " + Arrays.stream(e.getValue()).map(r -> Arrays.stream(r)
                        .mapToObj(v -> String.format("%.2f", v)).collect(Collectors.joining("/")))
                        .collect(Collectors.joining(" | ")))
                .collect(Collectors.joining("; "));
    }

    @Test
    void sparseRecoveryAndPhaseTransition() {
        double[] deltas = {0.25, 0.5};
        double[] rhos = {0.1, 0.25, 0.4};
        Map<String, double[][]> p = SparseRecovery.phaseDiagram(128, deltas, rhos, 8, 300, 1);
        System.out.printf("12.4.11 success rate n 128, delta 0.25|0.5 x rho 0.1/0.25/0.4: %s%n", grid(p));
        Map<String, double[]> c = SparseRecovery.convergence(64, 128, 8, 1000, 3, 2);
        System.out.printf("12.4.11 {NMSE dB at 20 it, iterations to 1e-4} m 64 n 128 s 8: %s%n",
                ChannelDecodingTest.line(c));
        assertTrue(p.get("AMP")[1][0] >= 0.75, grid(p));
        assertTrue(p.get("MIN_NORM_LS")[1][0] == 0, grid(p));
        assertTrue(c.get("AMP")[1] < c.get("ISTA")[1], ChannelDecodingTest.line(c));
    }

    @Test
    void unrolledSolvers() {
        Map<String, double[]> u = SparseRecovery.unrolled(32, 64, 4, 8, 200, 64, 100, 3);
        System.out.printf("12.4.12 {NMSE dB in-dist, 2x sparsity, noise 0.02, params} 8 layers m 32 n 64 s 4: %s%n",
                ChannelDecodingTest.line(u));
        assertTrue(u.get("LISTA")[0] < u.get("ISTA (8 iterations)")[0], ChannelDecodingTest.line(u));
        assertTrue(u.get("ALISTA")[3] < u.get("LISTA")[3] / 50, ChannelDecodingTest.line(u));
    }

    @Test
    void dictionaryLearning() {
        double[] a = SparseRecovery.plantedAtoms(20, 50, 1500, 3, 15, 4);
        System.out.printf("12.4.13 planted atoms recovered: K-SVD %.2f (%.0f ms), online DL %.2f (%.0f ms)%n", a[0],
                a[1], a[2], a[3]);
        Map<String, double[]> d = SparseRecovery.denoising(new double[] {0.1, 0.2}, 6, 96, 8, 5);
        System.out.printf("12.4.13 patch denoising PSNR dB at sigma 0.1/0.2: %s%n", ChannelDecodingTest.line(d));
        assertTrue(a[0] > 0.7, "K-SVD recovered " + a[0]);
        assertTrue(d.get("K-SVD (96)")[0] > d.get("noisy")[0] + 3, ChannelDecodingTest.line(d));
    }

    @Test
    void robustPca() {
        Map<String, double[]> r = SparseRecovery.corruptedMatrices(60, 3, new double[] {0.05, 0.1, 0.2}, 6);
        System.out.printf("12.4.14 relative error of L (and support F1) at 5/10/20%% corruption, 60x60 r 3: %s%n",
                ChannelDecodingTest.line(r));
        double[] v = SparseRecovery.video(3, 0.4, 7);
        double[] still = SparseRecovery.video(3, 0, 7);
        System.out.printf("12.4.14 video foreground IoU (drifting light / static): PCP %.3f/%.3f, AltProj %.3f/%.3f,"
                + " median %.3f/%.3f%n", v[0], still[0], v[1], still[1], v[2], still[2]);
        assertTrue(r.get("PCP")[1] < r.get("PCA (true rank)")[1] / 3, ChannelDecodingTest.line(r));
        assertTrue(v[0] > v[2], Arrays.toString(v));
    }
}
