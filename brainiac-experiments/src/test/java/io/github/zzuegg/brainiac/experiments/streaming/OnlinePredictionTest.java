package io.github.zzuegg.brainiac.experiments.streaming;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class OnlinePredictionTest {

    @Test
    void budgetedKernelsBeatLinearLearnersOnSpirals() {
        List<OnlinePrediction.KernelRow> rows = OnlinePrediction.kernels(2_000, new int[] {50, 200}, 1);
        for (OnlinePrediction.KernelRow r : rows) {
            System.out.printf("12.2.5 %-12s %-42s budget %4d: mistakes %.3f, held-out %.3f, %4d SVs, %7.1f us/point%n",
                    r.task(), r.method(), r.budget(), r.mistakeRate(), r.accuracy(), r.supportVectors(),
                    r.microsPerPoint());
        }
        assertTrue(pick(rows, "spirals", "BSGD (merge)", 200).accuracy()
                >= pick(rows, "spirals", "FTRL", -1).accuracy(), "kernels match binned FTRL on spirals");
        assertTrue(pick(rows, "digits", "BSGD (merge)", 200).accuracy()
                > pick(rows, "digits", "FTRL", -1).accuracy() + 0.2, "kernels beat FTRL on digits");
        assertTrue(pick(rows, "digits", "online RFF", 500).accuracy() > 0.5);
    }

    private static OnlinePrediction.KernelRow pick(List<OnlinePrediction.KernelRow> rows, String task, String method,
            int budget) {
        return rows.stream().filter(r -> r.task().equals(task) && r.method().startsWith(method)
                && (budget < 0 || r.budget() == budget)).findFirst().orElseThrow();
    }

    @Test
    void expertAdviceTracksTheBestForecaster() {
        List<OnlinePrediction.ExpertRow> rows = OnlinePrediction.experts(1_200, 2);
        for (OnlinePrediction.ExpertRow r : rows) {
            System.out.printf("12.2.6 %-40s %-36s loss %.4f, regret %+.4f, vs best 3-switch sequence %+.4f%n",
                    r.setting(), r.method(), r.meanLoss(), r.regret(), r.switchingRegret());
        }
        for (OnlinePrediction.ExpertRow r : rows) {
            if (r.method().equals("AdaHedge")) {
                assertTrue(r.regret() < 0.05, r.toString());
            }
        }
    }

    @Test
    void conformalPidKeepsCoverage() {
        List<OnlinePrediction.IntervalRow> rows = OnlinePrediction.intervals(2_000, 0.1, 3);
        for (OnlinePrediction.IntervalRow r : rows) {
            System.out.printf("12.2.8 %-20s %-44s coverage %.3f, local deviation %.3f, worst window %.2f, width %.3f%n",
                    r.series(), r.method(), r.coverage(), r.localDeviation(), r.worstWindow(), r.width());
        }
        for (OnlinePrediction.IntervalRow r : rows) {
            if (r.method().startsWith("PI") || r.method().startsWith("quantile")) {
                assertTrue(Math.abs(r.coverage() - 0.9) < 0.03, r.toString());
            }
        }
    }
}
