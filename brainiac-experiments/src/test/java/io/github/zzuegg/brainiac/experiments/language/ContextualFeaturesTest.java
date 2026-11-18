package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.language.ContextualFeatures.Features;

import org.junit.jupiter.api.Test;

class ContextualFeaturesTest {

    @Test
    void contextResolvesAmbiguousTags() {
        double[] elmo = ContextualFeatures.run(Features.ELMO, 300, 1);
        double[] stat = ContextualFeatures.run(Features.STATIC, 300, 1);
        double[] mlm = ContextualFeatures.run(Features.MASKED_LM, 300, 1);
        System.out.printf("Tagging macro-F1 / ambiguous-token accuracy: ELMo %.3f/%.3f, static SGNS %.3f/%.3f,"
                + " masked-LM encoder %.3f/%.3f%n", elmo[0], elmo[1], stat[0], stat[1], mlm[0], mlm[1]);
        assertTrue(elmo[1] > stat[1] + 0.2, "biLM features tag words whose tag depends on context");
        assertTrue(elmo[0] > 0.9, "and tag the rest");
    }
}
