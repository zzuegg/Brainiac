package io.github.zzuegg.brainiac.experiments.security;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.security.Backdoors;

import org.junit.jupiter.api.Test;

class BackdoorShootoutTest {

    @Test
    void badNetsImplantsTheTriggerAtOnePercentWhileCleanLabelNeedsMore() {
        BackdoorShootout.Attack clean = BackdoorShootout.attack(Backdoors.Kind.PATCH, 0, 1);
        BackdoorShootout.Attack noise = BackdoorShootout.attack(null, 0.05, 1);
        print(clean);
        print(noise);
        double[] rates = {0.01, 0.05};
        double[][] success = new double[3][2];
        for (Backdoors.Kind k : Backdoors.Kind.values()) {
            for (int i = 0; i < rates.length; i++) {
                BackdoorShootout.Attack a = BackdoorShootout.attack(k, rates[i], 1);
                print(a);
                success[k.ordinal()][i] = a.success();
                assertTrue(a.clean() > clean.clean() - 0.08, "clean accuracy " + a);
            }
        }
        System.out.printf("Backdoor ASR at 1%%/5%%: patch %.3f/%.3f blended %.3f/%.3f clean-label %.3f/%.3f; "
                + "no-trigger control %.3f, clean model %.3f%n", success[0][0], success[0][1], success[1][0],
                success[1][1], success[2][0], success[2][1], noise.success(), clean.success());
        assertTrue(success[0][1] > 0.9, "BadNets at 5%");
        assertTrue(success[0][0] > clean.success() + 0.3, "BadNets at 1%");
        assertTrue(success[2][0] < success[0][0], "clean-label weaker than BadNets at 1%");
    }

    static void print(BackdoorShootout.Attack a) {
        System.out.printf("Backdoor %-13s rate %.3f clean acc %.3f attack success %.3f%n", a.attack(), a.rate(),
                a.clean(), a.success());
    }
}
