package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.experimental.nn.PBitIsingMachine;
import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ProbabilisticBitsTest {
    // codes: bit 0 = A, bit 1 = B, bit 2 = C
    private static final int[] TRUTH_TABLE = {0b000, 0b001, 0b010, 0b111};

    @Test
    void gibbsSamplingMatchesTheExactBoltzmannDistribution() {
        SplittableRandom r = new SplittableRandom(1);
        int n = 6;
        double[][] j = new double[n][n];
        double[] h = new double[n];
        for (int a = 0; a < n; a++) {
            h[a] = r.nextDouble(-0.5, 0.5);
            for (int b = a + 1; b < n; b++) {
                j[a][b] = j[b][a] = r.nextDouble(-1, 1);
            }
        }
        PBitIsingMachine m = PBitIsingMachine.of(j, h);
        double[] sampled = ProbabilisticBits.histogram(m, 0.8, 40000, 100, 2);
        double tv = ProbabilisticBits.totalVariation(sampled, m.boltzmann(0.8));
        System.out.printf("p-bit sampler TV distance to Boltzmann: %.4f%n", tv);
        assertTrue(tv < 0.03, "TV " + tv);
    }

    @Test
    void andGateRunsForwardsAndBackwards() {
        PBitIsingMachine gate = ProbabilisticBits.andGate();
        double[] exact = gate.boltzmann(2);
        double valid = 0;
        for (int c : TRUTH_TABLE) {
            assertEquals(-3, gate.energy(new int[] {(c & 1) == 1 ? 1 : -1, (c & 2) == 2 ? 1 : -1, (c & 4) == 4 ? 1 : -1}),
                    1e-12);
            valid += exact[c];
        }
        assertTrue(valid > 0.95);
        // forwards: A = 1, B = 1 gives C = 1
        gate.clamp(0, 1);
        gate.clamp(1, 1);
        double[] fwd = ProbabilisticBits.histogram(gate, 2, 2000, 10, 3);
        assertTrue(fwd[0b111] > 0.95, "forward " + fwd[0b111]);
        // backwards: C = 1 forces A = B = 1
        gate.release();
        gate.clamp(2, 1);
        double[] back = ProbabilisticBits.histogram(gate, 2, 2000, 10, 4);
        assertTrue(back[0b111] > 0.95, "backward " + back[0b111]);
        // C = 0 leaves the three input pairs that give false, about equally likely
        gate.release();
        gate.clamp(2, -1);
        double[] no = ProbabilisticBits.histogram(gate, 2, 6000, 10, 5);
        System.out.printf("p-bit AND gate: valid states %.4f (exact), forward P(111) %.3f, backward P(111) %.3f, "
                + "C=0 -> P(000) %.3f P(001) %.3f P(010) %.3f%n", valid, fwd[0b111], back[0b111], no[0b000], no[0b001],
                no[0b010]);
        for (int c : new int[] {0b000, 0b001, 0b010}) {
            assertEquals(1.0 / 3, no[c], 0.05, "state " + c);
        }
    }

    @Test
    void latticeOrdersBelowTheCriticalTemperature() {
        double hot = ProbabilisticBits.latticeMagnetization(16, 0.3, 400, 200, 6);
        double cold = ProbabilisticBits.latticeMagnetization(16, 0.6, 400, 200, 7);
        System.out.printf("Ising 16x16 <|m|>: beta 0.3 -> %.3f, beta 0.6 -> %.3f%n", hot, cold);
        assertTrue(hot < 0.25, "disordered phase " + hot);
        assertTrue(cold > 0.9, "ordered phase " + cold);
    }
}
