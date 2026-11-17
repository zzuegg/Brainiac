package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.experimental.nn.HdcClassifier;
import io.github.zzuegg.brainiac.experimental.tensor.Hypervector;
import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class HyperdimensionalDemoTest {

    @Test
    void theDollarOfMexicoIsThePeso() {
        String answer = HyperdimensionalDemo.dollarOfMexico(10_048, 1);
        System.out.printf("HDC analogy at 10048 bits: dollar of Mexico = %s%n", answer);
        assertEquals("peso", answer);
    }

    @Test
    void identifiesLanguagesFromOneExampleEach() {
        double acc = HyperdimensionalDemo.languageId(6, 400, 20, 120, 4096, 2);
        System.out.printf("HDC one-shot language ID (6 languages): accuracy %.3f (chance %.3f)%n", acc, 1.0 / 6);
        assertTrue(acc > 0.9, "one-shot accuracy " + acc + " (chance 1/6)");
    }

    @Test
    void retrainingFixesMistakes() {
        SplittableRandom r = new SplittableRandom(5);
        int d = 1024;
        Hypervector[] centres = {Hypervector.random(d, r), Hypervector.random(d, r)};
        HdcClassifier c = new HdcClassifier(2, d);
        // one misleading example: a class-0-looking vector labelled 1
        c.add(centres[0], 0);
        c.add(centres[0], 1);
        c.add(centres[1], 1);
        assertTrue(c.retrain(centres[0], 0) || c.predict(centres[0]) == 0);
        assertEquals(0, c.predict(centres[0]));
        assertEquals(1, c.predict(centres[1]));
    }
}
