package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class CoconutCurriculumTest {

    /**
     * Runs the whole curriculum down to fully latent reasoning. At this tiny CPU budget the
     * latent model stays well below a model trained to answer directly (the paper needs
     * many epochs per stage), so this only checks that every stage trains and the latent
     * hops carry information (above the 25% of guessing).
     */
    @Test
    void curriculumRunsDownToLatentReasoning() {
        CoconutCurriculum.Result r = CoconutCurriculum.run(80, 1);
        System.out.println("Coconut " + r);
        assertTrue(r.language() > 0.3, r.toString());
        assertTrue(r.latent() > 0.25, r.toString());
        assertTrue(r.noThought() > 0.3, r.toString());
    }
}
