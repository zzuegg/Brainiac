package io.github.zzuegg.brainiac.experiments.robotics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class MultiViewGeometryTest {

    @Test
    void twoViewAndPnp() {
        List<MultiViewGeometry.TwoViewRow> rows = MultiViewGeometry.twoView(6, 1261);
        StringBuilder line = new StringBuilder("12.3.6 rot/trans-deg/recall:");
        for (MultiViewGeometry.TwoViewRow r : rows) {
            line.append(String.format(" %s %s %.2f/%.2f/%.2f;", r.scene(), r.method(), r.rotationError(),
                r.translationError(), r.inlierRecall()));
        }
        System.out.println(line);
        assertTrue(rows.get(1).rotationError() < 1.0, line.toString());
        assertTrue(rows.get(7).rotationError() < 3.0, line.toString());
        List<MultiViewGeometry.PnpRow> pnp = MultiViewGeometry.pnp(new int[] {6, 10, 50}, 10, 1271);
        StringBuilder l2 = new StringBuilder("12.3.7 PnP rot-deg/rel-trans:");
        for (MultiViewGeometry.PnpRow r : pnp) {
            l2.append(String.format(" n=%d %s %.3f/%.4f;", r.points(), r.method(), r.rotationError(),
                r.translationError()));
        }
        System.out.println(l2);
        assertTrue(pnp.get(pnp.size() - 1).rotationError() < 0.5, l2.toString());
        List<MultiViewGeometry.BundleRow> ba = MultiViewGeometry.bundle(12, 150, 15, 1272);
        StringBuilder l3 = new StringBuilder("12.3.7 BA rmse before/after, iters, ms/iter, point err:");
        for (MultiViewGeometry.BundleRow r : ba) {
            l3.append(String.format(" %s %.2f/%.3f %d %.1f %.4f;", r.method(), r.initialRmse(), r.finalRmse(),
                r.iterations(), r.millisPerIteration(), r.pointError()));
        }
        System.out.println(l3);
        assertTrue(ba.get(0).finalRmse() < 2.0, l3.toString());
        assertTrue(Math.abs(ba.get(0).finalRmse() - ba.get(1).finalRmse()) < 0.05, l3.toString());
    }

    @Test
    void registration() {
        List<MultiViewGeometry.RegistrationRow> rows = MultiViewGeometry.registration(new double[] {15, 45, 90}, 8,
            48, 120, 1281);
        StringBuilder line = new StringBuilder("12.3.8 success/median-deg:");
        for (MultiViewGeometry.RegistrationRow r : rows) {
            line.append(String.format(" %.0f %s %.2f/%.2f;", r.angle(), r.method(), r.success(), r.rotationError()));
        }
        System.out.println(line);
        assertTrue(rows.get(0).success() > 0.99, line.toString());
        assertTrue(rows.get(1).success() >= 0.75, line.toString());
    }
}
