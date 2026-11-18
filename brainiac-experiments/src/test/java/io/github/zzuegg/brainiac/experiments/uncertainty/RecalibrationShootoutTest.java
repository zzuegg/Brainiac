package io.github.zzuegg.brainiac.experiments.uncertainty;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class RecalibrationShootoutTest {

    private static RecalibrationShootout.Row row(List<RecalibrationShootout.Row> rows, String m, int n) {
        return rows.stream().filter(r -> r.method().equals(m) && r.validation() == n).findFirst().orElseThrow();
    }

    @Test
    void recalibrationMapsAgainstTemperatureScaling() {
        List<RecalibrationShootout.Row> rows = RecalibrationShootout.recalibration(UncertaintyFixture.Task.DIGITS,
                new int[] {100, 1000}, null, 1);
        for (int n : new int[] {100, 1000}) {
            StringBuilder line = new StringBuilder("11.4.1-3 digits recalibration n=" + n + " ECE/cwECE/NLL:");
            for (RecalibrationShootout.Row r : rows) {
                if (r.validation() == n) {
                    line.append(String.format(" %s %.3f/%.3f/%.3f;", r.method(), r.ece(), r.classwiseEce(), r.nll()));
                }
            }
            System.out.println(line);
        }
        RecalibrationShootout.Row none = row(rows, "none", 1000);
        RecalibrationShootout.Row temp = row(rows, "temperature", 1000);
        System.out.printf("11.4.1 shifted ECE: none %.3f temperature %.3f dirichlet %.3f sms %.3f%n", none.shiftedEce(),
                temp.shiftedEce(), row(rows, "dirichlet", 1000).shiftedEce(), row(rows, "sms", 1000).shiftedEce());
        assertTrue(temp.nll() < none.nll(), "temperature scaling lowers NLL of the overconfident model");
        for (RecalibrationShootout.Row r : rows) {
            assertTrue(Double.isFinite(r.nll()) && r.accuracy() > 0.3, r.toString());
        }
        assertTrue(row(rows, "sms", 1000).nll() < none.nll() && row(rows, "dirichlet", 1000).nll() < none.nll());
    }

    @Test
    void vennAbersAgainstIsotonicAndPlatt() {
        List<RecalibrationShootout.BinaryRow> rows = RecalibrationShootout.vennAbers(new int[] {100, 1000}, 1);
        StringBuilder line = new StringBuilder("11.4.3 tabular binary NLL/Brier/width (shift NLL/width):");
        for (RecalibrationShootout.BinaryRow r : rows) {
            line.append(String.format(" %s@%d %.3f/%.3f/%.3f (%.3f/%.3f);", r.method(), r.validation(), r.nll(),
                    r.brier(), r.width(), r.shiftedNll(), r.shiftedWidth()));
        }
        System.out.println(line);
        for (RecalibrationShootout.BinaryRow r : rows) {
            assertTrue(Double.isFinite(r.nll()) && r.brier() < 0.5, r.toString());
            if (r.method().equals("ivap")) {
                assertTrue(r.width() >= 0 && r.width() < 0.5, "intervals are narrow but non-negative: " + r);
            }
        }
    }
}
