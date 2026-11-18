package io.github.zzuegg.brainiac.experimental.deeprl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.rl.KeyDoorGrid;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.GridWorld;
import io.github.zzuegg.brainiac.rl.Pendulum;
import io.github.zzuegg.brainiac.rl.Rl;
import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

class LearningCurveHarnessTest {

    @Test
    void statisticsMatchHandComputedValues() {
        double[] x = {1, 2, 3, 4, 5, 6, 7, 100};
        assertEquals(4.5, LearningCurveHarness.iqm(x), 1e-9);   // drops 1, 2 and 7, 100
        assertEquals(1.0, LearningCurveHarness.kendallTau(new double[] {1, 2, 3}, new double[] {10, 20, 30}), 1e-9);
        assertEquals(-1.0, LearningCurveHarness.kendallTau(new double[] {1, 2, 3}, new double[] {3, 2, 1}), 1e-9);
        double poi = LearningCurveHarness.probabilityOfImprovement(new double[][] {{2, 3}}, new double[][] {{1, 3}});
        assertEquals(0.625, poi, 1e-9);   // 2>1, 2<3, 3>1, 3=3: (1 + 0 + 1 + 0.5) / 4
        double[] ci = LearningCurveHarness.stratifiedBootstrapIqm(new double[][] {{1, 2, 3, 4, 5, 6, 7, 8}}, 500, 0.95, 1);
        assertTrue(ci[0] <= 4.5 && ci[1] >= 4.5 && ci[0] < ci[1], ci[0] + " " + ci[1]);
        System.out.printf("W10 4.14 harness stats: IQM %.2f (mean 16.0), bootstrap 95%% [%.2f, %.2f], P(improve) %.3f%n",
                LearningCurveHarness.iqm(x), ci[0], ci[1], poi);
    }

    @Test
    void pixelEnvironmentsRenderTheState() {
        PixelObservations grid = PixelObservations.grid(GridWorld.of("S..", ".#.", "..G"), 2);
        float[] img = grid.reset(new SplittableRandom(1));
        assertEquals(3 * 6 * 6, img.length);
        assertEquals(1f, img[2 * 36]);                 // agent in the top-left cell, channel 2
        assertEquals(1f, img[36 + 4 * 6 + 4]);         // goal, channel 1, bottom right
        assertEquals(1f, img[2 * 6 + 2]);              // wall in the centre, channel 0
        Environment.Step s = grid.step(1);             // right
        assertEquals(1f, s.observation()[2 * 36 + 2]);

        PixelObservations kd = PixelObservations.keyDoor(new KeyDoorGrid(4, 20, 5), 4, 2);
        float[] k = kd.reset(new SplittableRandom(2));
        double sum = 0;
        for (float v : k) {
            sum += v;
        }
        assertEquals(12.0, sum, 1e-6);                 // agent, key, door: 4 pixels each

        PixelObservations pend = PixelObservations.pendulum(new Pendulum(50), 9);
        float[] p = pend.reset(new SplittableRandom(3));
        assertEquals(2 * 81, p.length);
        assertEquals(1f, p[4 * 9 + 4]);                // the rod starts at the centre

        ContinuousMountainCar car = new ContinuousMountainCar(300);
        double random = Rl.evaluate(Agent.random(car.actionSpace(), 1), car, 3, 1);
        double push = Rl.evaluate((o, e) -> new float[] {o[1] >= 0 ? 1 : -1}, car, 3, 1);
        System.out.printf("W10 continuous mountain car: random %.1f, bang-bang energy pumping %.1f%n", random, push);
        assertTrue(push > 50 && random < 0, random + " " + push);
    }
}
