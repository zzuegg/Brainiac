package io.github.zzuegg.brainiac.experiments.security;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.security.Backdoors;

import java.util.List;

import org.junit.jupiter.api.Test;

class PoisonDetectionTest {

    @Test
    void representationDetectorsFindBadNetsPoisonsBetterThanRandom() {
        for (Backdoors.Kind kind : new Backdoors.Kind[] {Backdoors.Kind.PATCH, Backdoors.Kind.BLENDED}) {
            List<BackdoorShootout.Detection> rows = BackdoorShootout.detection(kind, 0.05, 1);
            for (BackdoorShootout.Detection r : rows) {
                System.out.printf("Poison detection %-7s 5%% %-21s AUROC %.3f (in target class %.3f) recall %.3f  "
                        + "retrained clean %.3f ASR %.3f %s%n", kind, r.detector(), r.auroc(), r.targetAuroc(),
                        r.recall(), r.clean(), r.success(), r.detail());
            }
            BackdoorShootout.Detection spectral = rows.get(0);
            BackdoorShootout.Detection clustering = rows.get(1);
            BackdoorShootout.Detection random = rows.get(5);
            System.out.printf("Poison detection %s: in-class AUROC spectral %.3f clustering %.3f AGPD %.3f random %.3f;"
                    + " ASR after filtering spectral %.3f clustering %.3f random %.3f%n", kind, spectral.targetAuroc(),
                    clustering.targetAuroc(), rows.get(2).targetAuroc(), random.targetAuroc(), spectral.success(),
                    clustering.success(), random.success());
            if (kind == Backdoors.Kind.PATCH) {
                assertTrue(Math.max(spectral.targetAuroc(), clustering.targetAuroc()) > 0.9, "detection AUROC");
                assertTrue(spectral.targetAuroc() > random.targetAuroc(), "spectral beats random");
                assertTrue(Math.min(spectral.success(), clustering.success()) <= random.success() + 0.05,
                        "filtering no worse than random");
            }
            assertTrue(Math.abs(random.auroc() - 0.5) < 0.15, "random AUROC " + random.auroc());
        }
    }
}
