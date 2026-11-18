package io.github.zzuegg.brainiac.experiments.exotic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import org.junit.jupiter.api.Test;

/** Wave 6, 6.6: Diehl and Cook STDP on the toy digits. */
class StdpDigitsTest {

    @Test
    void stdpLearnsDigitPrototypesWithoutGradients() {
        ToyTasks.Data train = ToyTasks.digits(10, 30, 8, 0.05, 1);
        ToyTasks.Data test = ToyTasks.digits(10, 20, 8, 0.05, 1, 999);
        StdpDigits small = StdpDigits.of(64, 20, 0.05, 1);
        small.train(train, 2);
        StdpDigits large = StdpDigits.of(64, 60, 0.05, 1);
        large.train(train, 2);
        double hebb = StdpDigits.softHebb(train, test, 60, 20, 1);
        double surrogate = StdpDigits.surrogate(train, test, 60, 300, 1);
        double small1 = small.accuracy(test);
        double large1 = large.accuracy(test);
        System.out.printf("STDP digits acc: 20 neurons %.2f, 60 neurons %.2f; SoftHebb(60)+readout %.2f;"
                + " surrogate-gradient LIF(60) %.2f%n", small1, large1, hebb, surrogate);
        assertTrue(large1 > 0.5, "well above the 10% of chance: " + large1);
    }
}
