package io.github.zzuegg.brainiac.experiments.safety;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class PersonaSteeringTest {

    @Test
    void personaVectorMonitorsAndPreventsTheTraitShift() {
        List<PersonaSteering.Result> out = PersonaSteering.run(400, 150, 1f, 1);
        for (PersonaSteering.Result r : out) {
            System.out.printf("Persona %-17s trait %.2f  task %.2f  projection shift %+.2f%n", r.method(), r.trait(),
                    r.taskAccuracy(), r.projectionShift());
        }
        PersonaSteering.Result before = out.get(0);
        PersonaSteering.Result plain = out.get(1);
        PersonaSteering.Result prevented = out.get(4);
        assertTrue(plain.trait() > before.trait() + 0.3, plain + " vs " + before);
        assertTrue(plain.projectionShift() > 0, plain.toString());
        assertTrue(prevented.trait() < plain.trait(), prevented + " vs " + plain);
        assertTrue(prevented.taskAccuracy() >= 0.75, prevented.toString());
    }
}
