package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.nn.ConceptLm;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ConceptVsTokenLmTest {

    @Test
    void sentenceEmbeddingAutoregression() {
        ConceptVsTokenLm e = ConceptVsTokenLm.create();
        int steps = 400;
        int batch = 16;
        ConceptLm lcm = e.trainConcept(steps, batch * ConceptVsTokenLm.positionRatio(), 1);
        HookedDecoder lm = e.trainTokens(steps, batch, 1);
        double concept = e.conceptAccuracy(lcm, 9);
        double token = e.tokenAccuracy(lm, 9);
        System.out.printf("next sentence, equal positions trained: concept LM %.3f in 1 call per sentence, token LM "
                + "%.3f in %d calls per sentence%n", concept, token, ConceptVsTokenLm.positionRatio() - 1);
        assertTrue(concept > 0.5, "the concept model predicts whole sentences: " + concept);
        assertTrue(token > 0.3, "the token model learns too: " + token);
    }
}
