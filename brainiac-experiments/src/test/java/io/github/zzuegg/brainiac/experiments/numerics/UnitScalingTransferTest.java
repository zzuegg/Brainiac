package io.github.zzuegg.brainiac.experiments.numerics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Map;

import org.junit.jupiter.api.Test;

class UnitScalingTransferTest {

    private static final int[] WIDTHS = {32, 128};
    private static final int[] RATES = {-8, -6, -4, -2, 0, 2};

    @Test
    void unitMupKeepsTheOptimalLearningRateAcrossWidths() {
        StringBuilder s = new StringBuilder("Digits MLP, LR sweep 2^-8..2^2 in steps of 4x, 80 steps, optimal log2 LR per width:");
        int[] shift = new int[2];
        for (UnitScalingTransfer.Scheme scheme : UnitScalingTransfer.Scheme.values()) {
            Map<Integer, double[]> sweep = UnitScalingTransfer.sweep(scheme, WIDTHS, RATES, 80, 1);
            Map<Integer, Integer> opt = UnitScalingTransfer.optimum(sweep, RATES);
            int lo = opt.values().stream().min(Integer::compare).orElseThrow();
            int hi = opt.values().stream().max(Integer::compare).orElseThrow();
            shift[scheme.ordinal()] = hi - lo;
            s.append(String.format(" %s %s (shift %d, best losses %s);", scheme, opt, hi - lo,
                    Arrays.toString(sweep.values().stream().mapToDouble(l -> Arrays.stream(l).min().orElseThrow())
                            .map(v -> Math.round(v * 1000) / 1000.0).toArray())));
        }
        System.out.println(s);
        assertTrue(shift[1] <= shift[0], s.toString());
        assertTrue(shift[1] <= 2, s.toString());
    }

    @Test
    void unitScaledGradientsSurviveFp8WithoutLossScaling() {
        UnitScalingTransfer.Fp8Row sp = UnitScalingTransfer.fp8(UnitScalingTransfer.Scheme.STANDARD, 64, 0x1p-5f, 150, 2);
        UnitScalingTransfer.Fp8Row mup = UnitScalingTransfer.fp8(UnitScalingTransfer.Scheme.UNIT_MUP, 64, 0x1p-2f, 150,
                2);
        String s = String.format("Digits MLP width 64, FP8 (E4M3 fwd, E5M2 bwd, no scaling), fp32 loss/fp8 loss/"
                + "gradients flushed to zero: standard %.3f/%.3f/%.3f; u-muP %.3f/%.3f/%.3f", sp.fp32Loss(),
                sp.fp8Loss(), sp.underflow(), mup.fp32Loss(), mup.fp8Loss(), mup.underflow());
        System.out.println(s);
        assertTrue(mup.underflow() < sp.underflow(), s);
        assertTrue(mup.fp8Loss() < mup.fp32Loss() + 0.2, s);
    }
}
