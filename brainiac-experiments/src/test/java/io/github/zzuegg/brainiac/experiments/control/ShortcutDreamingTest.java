package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class ShortcutDreamingTest {

    @Test
    void shortcutWorldModelSamplesTheModesInFewSteps() {
        Map<String, Double> r = ShortcutDreaming.run(100, 2000, 150, 1);
        StringBuilder b = new StringBuilder();
        for (int k : ShortcutDreaming.STEPS) {
            b.append(String.format("%d: %.3f vs FM %.3f, ", k, r.get("shortcut, " + k + " steps"),
                    r.get("flow matching, " + k + " steps")));
        }
        System.out.printf("W6 3.9 slippery track, distance of sampled next state to nearest real outcome by flow steps:"
                + " shortcut %s MLP world model %.3f; imagination-trained policy return: shortcut model %.1f, MLP model"
                + " %.1f, random %.1f%n", b, r.get("MLP world model"), r.get("policy in shortcut model, return"),
                r.get("policy in MLP world model, return"), r.get("random policy, return"));
        assertTrue(r.get("shortcut, 2 steps") < r.get("flow matching, 2 steps"), r.toString());
        assertTrue(r.get("shortcut, 2 steps") < r.get("MLP world model"), r.toString());
        assertTrue(r.get("policy in shortcut model, return") > r.get("random policy, return") + 10, r.toString());
    }
}
