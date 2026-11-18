package io.github.zzuegg.brainiac.experiments.streaming;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class StreamShootoutTest {

    @Test
    void ftrlIsSparseAtAdaGradLogLoss() {
        List<StreamShootout.ClickRow> rows = StreamShootout.clicks(150_000, 1);
        for (StreamShootout.ClickRow r : rows) {
            System.out.printf("9.6.8 %-46s progressive log loss %.4f (after drifts %.4f), %6d non-zeros, %8d B%n",
                    r.method(), r.logLoss(), r.lossAfterDrift(), r.nonZeros(), r.bytes());
        }
        StreamShootout.ClickRow ftrl = rows.get(0);
        StreamShootout.ClickRow adagrad = rows.get(3);
        StreamShootout.ClickRow daily = rows.get(4);
        assertTrue(ftrl.logLoss() < adagrad.logLoss() + 0.02, ftrl + " vs " + adagrad);
        assertTrue(ftrl.nonZeros() < adagrad.nonZeros(), "FTRL is sparser");
        assertTrue(daily.lossAfterDrift() > ftrl.lossAfterDrift(), "daily retraining lags after a drift");
    }

    @Test
    void forestsAndTreesTrackDriftingStreams() {
        List<StreamShootout.TreeRow> rows = StreamShootout.trees(12_000, 2);
        for (StreamShootout.TreeRow r : rows) {
            System.out.printf("9.6.9 %-18s %-46s accuracy %.3f, recovery %6.0f, %8d B, %.2f s%n", r.stream(),
                    r.method(), r.accuracy(), r.recovery(), r.bytes(), r.seconds());
        }
        for (String s : new String[] {"SEA abrupt", "hyperplane gradual", "RBF recurring"}) {
            double arf = rows.stream().filter(r -> r.stream().equals(s) && r.method().startsWith("adaptive"))
                    .findFirst().orElseThrow().accuracy();
            double majority = rows.stream().filter(r -> r.stream().equals(s) && r.method().startsWith("majority"))
                    .findFirst().orElseThrow().accuracy();
            assertTrue(arf > majority + 0.1, s + ": ARF " + arf + " vs majority " + majority);
        }
    }

    @Test
    void driftDetectorsFindKnownDrifts() {
        List<StreamShootout.DetectionRow> rows = StreamShootout.detection(9_000, 3);
        for (StreamShootout.DetectionRow r : rows) {
            System.out.printf("9.6.10 %-28s %-36s delay %7.1f, false alarms %6.1f per 10k, missed %d of %d%n",
                    r.series(), r.method(), r.meanDelay(), r.falseAlarmsPer10k(), r.missed(), r.drifts());
        }
        StreamShootout.DetectionRow adwin = rows.stream()
                .filter(r -> r.series().startsWith("SEA") && r.method().startsWith("ADWIN")).findFirst().orElseThrow();
        assertTrue(adwin.missed() <= adwin.drifts() / 2, adwin.toString());
    }

    @Test
    void asymmetricReplayBeatsNoReplayOnline() {
        List<StreamShootout.ContinualRow> rows = StreamShootout.continual("digits", new int[] {20, 200}, 4);
        rows.addAll(StreamShootout.continual("shapes", new int[] {20}, 4));
        for (StreamShootout.ContinualRow r : rows) {
            System.out.printf("9.6.11 %-6s buffer %3d %-24s final acc %.3f, forgetting %.3f, old-task acc after switch %.3f%n",
                    r.dataset(), r.buffer(), r.method(), r.finalAccuracy(), r.forgetting(), r.afterSwitch());
        }
        double ace = rows.stream().filter(r -> r.dataset().equals("digits") && r.buffer() == 20
                && r.method().equals("ER-ACE")).findFirst().orElseThrow().finalAccuracy();
        double none = rows.stream().filter(r -> r.dataset().equals("digits") && r.method().startsWith("fine-tune"))
                .findFirst().orElseThrow().finalAccuracy();
        assertTrue(ace > none, "ER-ACE " + ace + " vs no replay " + none);
    }

    @Test
    void coinBettingNeedsNoLearningRate() {
        List<StreamShootout.OptimizerRow> rows = StreamShootout.optimizers(300, 5);
        for (StreamShootout.OptimizerRow r : rows) {
            System.out.printf("9.6.12 %-26s %-30s held-out loss %.4f%n", r.task(), r.method(), r.loss());
        }
        double cocob = rows.stream().filter(r -> r.task().startsWith("click") && r.method().startsWith("COCOB"))
                .findFirst().orElseThrow().loss();
        assertTrue(cocob < Math.log(2), "COCOB learns the convex task: " + cocob);
    }
}
