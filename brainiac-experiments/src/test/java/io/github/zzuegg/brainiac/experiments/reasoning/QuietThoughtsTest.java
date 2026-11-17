package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class QuietThoughtsTest {

    @Test
    void quietStarAgainstNoThoughtsOnTwoHops() {
        QuietThoughts.Result r = QuietThoughts.run(300, 1);
        System.out.printf("Quiet-STaR held-out accuracy %.3f vs no thoughts %.3f (thought = hop f(x) on %.3f;"
                + " Coconut latent: see CoconutCurriculumTest)%n", r.quietStar(), r.noThought(), r.hopThoughts());
        assertTrue(r.quietStar() > 0.25, r.toString());
    }
}
