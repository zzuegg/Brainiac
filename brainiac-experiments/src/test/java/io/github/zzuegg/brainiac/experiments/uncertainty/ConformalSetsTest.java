package io.github.zzuegg.brainiac.experiments.uncertainty;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ConformalSetsTest {

    private static String line(List<ConformalSets.Row> rows) {
        StringBuilder b = new StringBuilder();
        for (ConformalSets.Row r : rows) {
            b.append(String.format(" %s cov %.3f worst %.3f size %.2f ssc %.3f;", r.method(), r.coverage(),
                    r.worstClass(), r.size(), r.sizeStratified()));
        }
        return b.toString();
    }

    private static ConformalSets.Row row(List<ConformalSets.Row> rows, String m) {
        return rows.stream().filter(r -> r.method().equals(m)).findFirst().orElseThrow();
    }

    @Test
    void rapsAndMondrianAgainstLacAndAps() {
        List<ConformalSets.Row> rows = ConformalSets.sets(UncertaintyFixture.Task.DIGITS, 0.1, 1);
        System.out.println("11.4.4 digits alpha 0.1:" + line(rows));
        List<ConformalSets.Row> tail = ConformalSets.longTail(30, 0.1, 400, 1);
        System.out.println("11.4.4 long-tailed digits (30x):" + line(tail));
        for (ConformalSets.Row r : rows) {
            if (!r.method().startsWith("top")) {
                assertTrue(r.coverage() > 0.85, "conformal coverage near 0.9: " + r);
            }
        }
        assertTrue(row(rows, "raps").size() <= row(rows, "aps").size() + 1e-9, "RAPS sets no larger than APS");
        assertTrue(row(tail, "mondrian aps").worstClass() >= row(tail, "aps").worstClass() - 0.05,
                "Mondrian protects the worst class");
    }

    @Test
    void riskControlAndConformalTraining() {
        double[] crc = ConformalSets.riskControl(0.1, 600, 1);
        System.out.printf("11.4.4 multilabel FNR: crc %.3f (size %.2f, lambda %.3f) vs threshold 0.5 %.3f (size %.2f)%n",
                crc[0], crc[1], crc[2], crc[3], crc[4]);
        assertTrue(crc[0] < 0.15, "risk control holds the FNR near 0.1");
        List<ConformalSets.TrainingRow> rows = ConformalSets.conformalTraining(UncertaintyFixture.Task.DIGITS, 0.1,
                150, 1);
        StringBuilder b = new StringBuilder("11.4.5 fine-tune then calibrate, acc / LAC / APS / RAPS size:");
        for (ConformalSets.TrainingRow r : rows) {
            b.append(String.format(" %s %.3f / %.2f / %.2f / %.2f;", r.method(), r.accuracy(), r.lacSize(), r.apsSize(),
                    r.rapsSize()));
        }
        System.out.println(b);
        for (ConformalSets.TrainingRow r : rows) {
            assertTrue(r.accuracy() > 0.3 && r.lacCoverage() > 0.85, r.toString());
        }
    }
}
