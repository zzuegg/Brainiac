package io.github.zzuegg.brainiac.experiments.exotic;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class EquivariantPointCloudsTest {
    @Test
    void egnnGeneralizesToRotationsWhereAnMlpDoesNot() {
        EquivariantPointClouds egnn = EquivariantPointClouds.egnn(1e-2f, 1);
        EquivariantPointClouds mlp = EquivariantPointClouds.mlp(1e-2f, 1);
        egnn.train(400, 32);
        mlp.train(150, 32);
        double egnnPlain = egnn.accuracy(false, 200, 7);
        double egnnRotated = egnn.accuracy(true, 200, 8);
        double mlpPlain = mlp.accuracy(false, 200, 7);
        double mlpRotated = mlp.accuracy(true, 200, 8);
        System.out.printf("EGNN %.2f / rotated %.2f; MLP %.2f / rotated %.2f%n", egnnPlain, egnnRotated, mlpPlain,
                mlpRotated);
        assertTrue(egnnPlain > 0.9 && mlpPlain > 0.9);
        assertTrue(egnnRotated > 0.9, "EGNN rotated " + egnnRotated);
        assertTrue(mlpRotated < 0.85, "MLP rotated " + mlpRotated);
    }
}
