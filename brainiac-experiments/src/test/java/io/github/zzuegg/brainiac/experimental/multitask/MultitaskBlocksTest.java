package io.github.zzuegg.brainiac.experimental.multitask;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class MultitaskBlocksTest {

    private static final double[][] CONFLICT = {{1, 0, 0}, {-0.8, 1, 0}, {0, 0.2, 1}};

    @Test
    void surgeryRemovesConflicts() {
        double[] pc = GradientSurgery.pcGrad(CONFLICT, new SplittableRandom(1));
        double[] ops = GradientSurgery.gradOps(CONFLICT, 0);
        for (double[] g : CONFLICT) {
            assertTrue(TaskGradients.dot(ops, g) >= -1e-9, "GradOPS conflicts with no task");
        }
        assertTrue(TaskGradients.dot(pc, CONFLICT[0]) > TaskGradients.dot(sum(CONFLICT), CONFLICT[0]) - 1e-9);
        double[] weighted = GradientSurgery.gradOps(CONFLICT, 1);
        for (double[] g : CONFLICT) {
            assertTrue(TaskGradients.dot(weighted, g) >= -1e-9, "reweighting keeps it non-conflicting");
        }
    }

    private static double[] sum(double[][] g) {
        return TaskGradients.combine(g, new double[] {1, 1, 1});
    }

    @Test
    void paretoDirectionsDescendEveryTask() {
        double[] w = ParetoDescent.mgdaWeights(new double[][] {{1, 0}, {0, 1}});
        assertEquals(0.5, w[0], 1e-6);
        double[][] scaled = {{10, 0}, {0, 1}};
        double[] mgda = ParetoDescent.mgda(scaled);
        double[] ca = ParetoDescent.caGrad(CONFLICT, 0.4);
        double[] nash = ParetoDescent.nashMtl(CONFLICT);
        for (double[] g : CONFLICT) {
            assertTrue(TaskGradients.dot(nash, g) > 0, "Nash-MTL descends every task");
        }
        assertTrue(TaskGradients.dot(mgda, scaled[0]) > 0 && TaskGradients.dot(mgda, scaled[1]) > 0);
        assertTrue(TaskGradients.norm(ca) > 0);
        double[] a = ParetoDescent.nashWeights(scaled);
        // Nash bargaining: alpha_i * |g_i|^2 * alpha_i = 1 for orthogonal gradients
        assertEquals(1, a[0] * a[0] * 100, 1e-4);
        assertEquals(1, a[1] * a[1], 1e-4);
    }

    @Test
    void famoShiftsWeightToTheSlowerTask() {
        Famo famo = new Famo(2, 0.1, 0);
        for (int i = 0; i < 20; i++) {
            famo.weightedLoss(java.util.List.of(
                    io.github.zzuegg.brainiac.autograd.Var.of(io.github.zzuegg.brainiac.tensor.Tensor.scalar(1f)),
                    io.github.zzuegg.brainiac.autograd.Var.of(io.github.zzuegg.brainiac.tensor.Tensor.scalar(1f))));
            famo.update(new double[] {0.5, 0.99});
        }
        double[] z = famo.weights();
        assertTrue(z[1] > z[0], "the task that improved less gets more weight: " + java.util.Arrays.toString(z));
    }
}
