package io.github.zzuegg.brainiac.experiments.robotics;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.List;

class DeepClosestPointSvdTest {

    @Test
    void registrationSvd() {
        List<MultiViewGeometry.RegistrationRow> rows = MultiViewGeometry.registrationSvd(new double[] {15, 45, 90}, 8,
            48, 120, 1281);
        StringBuilder line = new StringBuilder("13.4.17 DCP Procrustes, success/median-deg:");
        for (MultiViewGeometry.RegistrationRow r : rows) {
            line.append(String.format(" %.0f %s %.2f/%.2f;", r.angle(), r.method(), r.success(), r.rotationError()));
        }
        System.out.println(line);
        assertEquals(12, rows.size(), line.toString());
        for (MultiViewGeometry.RegistrationRow r : rows) {
            assertTrue(Double.isFinite(r.rotationError()), line.toString());
        }
    }
}
