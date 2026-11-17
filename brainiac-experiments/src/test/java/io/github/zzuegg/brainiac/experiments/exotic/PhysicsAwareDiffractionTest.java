package io.github.zzuegg.brainiac.experiments.exotic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Wave 6, 6.7: physics-aware training of a mis-fabricated diffractive classifier. */
class PhysicsAwareDiffractionTest {

    private static PhysicsAwareDiffraction trained(PhysicsAwareDiffraction.Mode mode) {
        PhysicsAwareDiffraction d = PhysicsAwareDiffraction.of(2, 0.2, 0.1, 0.6, 1);
        d.train(mode, 150, 0.2, 1);
        return d;
    }

    @Test
    void trainingThroughTheDeviceClosesTheSimToRealGap() {
        PhysicsAwareDiffraction silico = trained(PhysicsAwareDiffraction.Mode.IN_SILICO);
        PhysicsAwareDiffraction noisy = trained(PhysicsAwareDiffraction.Mode.NOISE_AWARE);
        PhysicsAwareDiffraction pat = trained(PhysicsAwareDiffraction.Mode.PHYSICS_AWARE);
        System.out.printf("diffractive device acc (twin acc): in-silico %.2f (%.2f), noise-aware %.2f (%.2f),"
                + " physics-aware %.2f (%.2f)%n", silico.accuracy(true), silico.accuracy(false),
                noisy.accuracy(true), noisy.accuracy(false), pat.accuracy(true), pat.accuracy(false));
        assertTrue(pat.accuracy(true) > silico.accuracy(true) + 0.2, "PAT beats in-silico on the device");
        assertTrue(pat.accuracy(true) > 0.6, "PAT device accuracy " + pat.accuracy(true));
    }
}
