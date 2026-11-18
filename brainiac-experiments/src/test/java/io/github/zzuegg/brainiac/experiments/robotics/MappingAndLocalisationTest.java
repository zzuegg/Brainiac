package io.github.zzuegg.brainiac.experiments.robotics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class MappingAndLocalisationTest {

    @Test
    void slam() {
        List<MappingAndLocalisation.SlamRow> rows = MappingAndLocalisation.slam(30, 400, 1291);
        StringBuilder line = new StringBuilder("12.3.9 ATE/landmark-RMSE/ms-per-step:");
        for (MappingAndLocalisation.SlamRow r : rows) {
            line.append(String.format(" %s %.3f/%.3f/%.3f;", r.method(), r.trajectoryError(), r.landmarkError(),
                r.millisPerStep()));
        }
        System.out.println(line);
        double dead = rows.get(0).trajectoryError();
        assertTrue(rows.get(3).trajectoryError() < dead, line.toString());
        assertTrue(rows.get(6).trajectoryError() < dead, line.toString());
    }

    @Test
    void mapping() {
        List<MappingAndLocalisation.MappingRow> rows = MappingAndLocalisation.mapping(32, 40, 1292);
        StringBuilder line = new StringBuilder("12.3.10 accuracy/AUROC:");
        for (MappingAndLocalisation.MappingRow r : rows) {
            line.append(String.format(" %s %s %.3f/%.3f;", r.sensor(), r.method(), r.accuracy(), r.auroc()));
        }
        List<MappingAndLocalisation.RecoveryRow> rec = MappingAndLocalisation.recovery(32, 60, 60, 1293);
        line.append(" recovery scans:");
        for (MappingAndLocalisation.RecoveryRow r : rec) {
            line.append(String.format(" %s %d;", r.method(), r.scans()));
        }
        System.out.println(line);
        assertTrue(rows.get(2).accuracy() > 0.8, line.toString());
        assertTrue(rec.get(2).scans() <= rec.get(1).scans(), line.toString());
    }
}
