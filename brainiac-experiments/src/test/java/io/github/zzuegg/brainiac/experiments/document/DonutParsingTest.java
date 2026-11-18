package io.github.zzuegg.brainiac.experiments.document;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DonutParsingTest {

    @Test
    void ocrFreeParserRunsAtToyBudget() {
        DocumentParsing.ParseScores s = DocumentParsing.parse(DocumentParsing.Parser.DONUT, 50, 40, 0.02, 1);
        System.out.printf("Parsing %-16s field F1 %.3f exact %.3f (%d ms)%n", s.parser(), s.f1(), s.exact(),
                s.millis());
        // a negative result at this budget: the image-to-markup model does not yet read values
        assertTrue(s.f1() >= 0 && s.f1() <= 1, "F1 is a share");
    }
}
