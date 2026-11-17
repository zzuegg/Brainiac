package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.experimental.nn.SensorimotorColumn;
import org.junit.jupiter.api.Test;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ThousandBrainsVotingTest {
    @Test
    void oneColumnDisambiguatesByMovingAndPathIntegrating() {
        SensorimotorColumn c = SensorimotorColumn.create();
        // two objects share feature 7 at different places; what lies one step right tells them apart
        c.learn(0, 0, 0, 7);
        c.learn(0, 1, 0, 1);
        c.learn(1, 0, 0, 2);
        c.learn(1, 1, 0, 7);
        c.learn(1, 2, 0, 3);
        c.sense(7);
        assertEquals(Set.of(0, 1), c.candidates());
        assertEquals(2, c.hypothesisCount());
        c.move(1, 0);
        c.sense(3);   // only "object 1, at (1,0)" moved to a 3
        assertEquals(Set.of(1), c.candidates());
    }

    @Test
    void votingColumnsRecognizeWithFewerSensations() {
        ThousandBrainsVoting t = ThousandBrainsVoting.of(40, 4, 4, 1);
        double[] one = t.meanSensations(1, 5, 2);
        double[] five = t.meanSensations(5, 5, 2);
        System.out.printf("thousand brains: 1 column %.2f sensations (%.2f correct), 5 voting columns %.2f (%.2f)%n",
                one[0], one[1], five[0], five[1]);
        assertEquals(1.0, one[1], 1e-9);
        assertEquals(1.0, five[1], 1e-9);
        assertTrue(five[0] < 0.6 * one[0], "voting should need fewer sensations");
        assertTrue(five[0] < 2.5);
    }
}
