package io.github.zzuegg.brainiac.experiments.evaluation;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.eval.EvalStatistics;

import java.util.List;

import org.junit.jupiter.api.Test;

class StatisticalReportingTest {

    @Test
    void normalHelpersAndMcNemar() {
        assertEquals(0.975, EvalStatistics.normalCdf(1.959964), 1e-6);
        assertEquals(1.959964, EvalStatistics.normalQuantile(0.975), 1e-5);
        double[] a = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0};
        double[] b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        assertEquals(2 * Math.pow(0.5, 8), EvalStatistics.mcNemar(a, b), 1e-12);
    }

    @Test
    void clusteredErrorBarsPairedTestsAndPower() {
        StatisticalReporting.ErrorBars e = StatisticalReporting.errorBars(50, 10, 1.5, 200, 1);
        System.out.printf("error bars, 50 clusters x 10 questions, 200 reruns: coverage iid %.2f, clustered %.2f, "
                + "cluster bootstrap %.2f (SE ratio %.2f); false wins /100 unpaired %.1f, paired %.1f, paired "
                + "clustered %.1f, McNemar %.1f; identical models >= 1 point apart %.0f/100; 2-point power paired "
                + "%.2f vs unpaired %.2f; questions for 80%% power paired %d vs unpaired %d%n", e.iidCoverage(),
                e.clusteredCoverage(), e.bootstrapCoverage(), e.seRatio(), e.falseWinsUnpaired(), e.falseWinsPaired(),
                e.falseWinsPairedClustered(), e.falseWinsMcNemar(), e.pointDifferences(), e.powerPaired(),
                e.powerUnpaired(), e.questionsPaired(), e.questionsUnpaired());
        assertTrue(e.iidCoverage() < 0.85, "iid SEs under-cover clustered questions: " + e.iidCoverage());
        assertTrue(e.clusteredCoverage() > 0.85, "clustered: " + e.clusteredCoverage());
        assertTrue(e.falseWinsPairedClustered() <= 10, "paired test keeps its level");
        assertTrue(e.powerPaired() > e.powerUnpaired(), "pairing adds power");
        assertTrue(e.questionsPaired() < e.questionsUnpaired());
    }

    @Test
    void boostingAttackClimbsThePlainHoldoutOnly() {
        List<StatisticalReporting.Adaptive> r = StatisticalReporting.boosting(500, 500, 4, 2);
        r.forEach(x -> System.out.println("boosting attack on random labels, 500 queries: " + x));
        StatisticalReporting.Adaptive plain = r.get(0);
        StatisticalReporting.Adaptive ladder = r.get(1);
        StatisticalReporting.Adaptive thresholdout = r.get(3);
        System.out.printf("reusable holdout: boosting attack reported %.3f plain, %.3f Ladder, %.3f Thresholdout "
                + "(truth %.3f)%n", plain.reported(), ladder.reported(), thresholdout.reported(), plain.truth());
        assertTrue(plain.reported() > 0.56, "plain holdout is overfit: " + plain);
        assertTrue(Math.abs(plain.truth() - 0.5) < 0.04, "true accuracy stays at chance: " + plain);
        assertTrue(ladder.reported() < plain.reported());
        assertTrue(thresholdout.reported() < plain.reported());
    }

    @Test
    void honestSearchLosesLittleUnderTheMechanisms() {
        List<StatisticalReporting.Adaptive> r = StatisticalReporting.honestSearch(300, 100, 3, 3);
        r.forEach(x -> System.out.println("honest greedy search over 100 classifiers: " + x));
        double plainTruth = r.get(0).truth();
        for (StatisticalReporting.Adaptive a : r) {
            assertTrue(a.truth() > plainTruth - 0.05, a.toString());
        }
    }
}
