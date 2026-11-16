package io.github.zzuegg.brainiac.rl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/** The lander can be solved by a hand-written pilot, and doing nothing crashes. */
class LanderTest {

    /** A hand-written pilot: aim the craft's tilt at the pad, and brake the fall. */
    static int pilot(float[] o) {
        double x = o[0] * 10;
        double y = o[1] * 10;
        double vx = o[2] * 5;
        double vy = o[3] * 5;
        double angle = o[4];
        double spin = o[5] * 2;
        if (o[6] > 0 && o[7] > 0) {
            return 0;
        }
        double targetAngle = Math.max(-0.4, Math.min(0.4, 0.1 * x + 0.35 * vx));   // tilt to push back over the pad
        double angleTodo = (targetAngle - angle) * 1.0 - spin * 0.6;
        double targetVy = -Math.max(0.4, 0.35 * y);
        double hoverTodo = (targetVy - vy) * 1.0;
        if (hoverTodo > Math.abs(angleTodo) && hoverTodo > 0.05) {
            return 2;
        }
        if (angleTodo < -0.05) {
            return 1;
        }
        if (angleTodo > 0.05) {
            return 3;
        }
        return 0;
    }

    private static double score(Agent agent, int episodes) {
        return Rl.evaluate(agent, new Lander(), episodes, 11);
    }

    @Test
    void aPilotLandsAndIdlingCrashes() {
        Agent pilot = (obs, explore) -> new float[] {pilot(obs)};
        double good = score(pilot, 20);
        double idle = score((obs, explore) -> new float[] {0}, 20);
        assertTrue(good > 200, "pilot scores " + good);
        assertTrue(idle < -100, "idling scores " + idle);
    }

    @Test
    void stepsChecksItsInput() {
        Lander lander = new Lander();
        assertThrows(IllegalStateException.class, () -> lander.step(0));
        float[] obs = lander.reset(new SplittableRandom(1));
        assertEquals(8, obs.length);
        assertThrows(IllegalArgumentException.class, () -> lander.step(4));
    }
}
