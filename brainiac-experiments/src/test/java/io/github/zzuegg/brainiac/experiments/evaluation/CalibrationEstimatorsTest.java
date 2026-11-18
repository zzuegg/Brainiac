package io.github.zzuegg.brainiac.experiments.evaluation;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.techniques.UncertaintyBenchmark;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class CalibrationEstimatorsTest {

    private static CalibrationEstimators.Error find(List<CalibrationEstimators.Error> e, String name, int n) {
        return e.stream().filter(x -> x.estimator().equals(name) && x.n() == n).findFirst().orElseThrow();
    }

    @Test
    void estimatorsAgainstAKnownCalibrationError() {
        List<CalibrationEstimators.Error> e = CalibrationEstimators.synthetic(1.3, new int[] {200, 1000}, 100, 3);
        e.forEach(x -> System.out.println("calibration estimators: " + x));
        CalibrationEstimators.Error width = find(e, "equal-width 15", 200);
        CalibrationEstimators.Error smooth = find(e, "smooth ECE", 200);
        CalibrationEstimators.Error sweep = find(e, "monotone sweep", 200);
        CalibrationEstimators.Error plugin = find(e, "plug-in L2 15", 200);
        CalibrationEstimators.Error debiased = find(e, "debiased L2 15", 200);
        System.out.printf("calibration estimators n=200: equal-width ECE %.2fx truth, smooth ECE %.2fx, sweep %.2fx; "
                + "debiased L2 bias %+.4f vs plug-in %+.4f%n", width.ratio(), smooth.ratio(), sweep.ratio(),
                debiased.bias(), plugin.bias());
        assertTrue(width.bias() > 0, "binned ECE overstates at n = 200");
        assertTrue(Math.abs(debiased.bias()) < Math.abs(plugin.bias()), debiased + " vs " + plugin);
        assertTrue(Math.abs(find(e, "equal-width 15", 1000).bias()) < width.bias(), "bias shrinks with n");
    }

    @Test
    void perfectCalibrationHasZeroTruthAndSmallSmoothEce() {
        assertEquals(0, CalibrationEstimators.truth(1.0)[0], 1e-12);
        double[][] s = CalibrationEstimators.sample(1.0, 5000, new SplittableRandom(4));
        double smooth = io.github.zzuegg.brainiac.experimental.eval.CalibrationEstimators.smooth(s[0], s[1]);
        double width = io.github.zzuegg.brainiac.experimental.eval.CalibrationEstimators.equalWidth(s[0], s[1], 15);
        System.out.printf("calibrated predictor, n=5000: smooth ECE %.4f, equal-width ECE %.4f (truth 0)%n", smooth,
                width);
        assertTrue(smooth < 0.05 && width < 0.05);
    }

    @Test
    void benchmarkRankingsUnderEachEstimator() {
        CalibrationEstimators.Rankings r = CalibrationEstimators.rankings(List.of(UncertaintyBenchmark.Method.MAP,
                UncertaintyBenchmark.Method.MC_DROPOUT, UncertaintyBenchmark.Method.LAPLACE), 1);
        r.scores().forEach((k, v) -> System.out.println("calibration ranking by " + k + ": " + v));
        System.out.printf("calibration rankings: %d distinct equal-width rankings over 5-50 bins; differ from 15 bins: "
                + "%s%n", r.distinctEqualWidth(), r.flipsVsBaseline());
        assertTrue(r.distinctEqualWidth() >= 1);
        assertEquals(3, r.scores().get("smooth ECE").size());
    }
}
