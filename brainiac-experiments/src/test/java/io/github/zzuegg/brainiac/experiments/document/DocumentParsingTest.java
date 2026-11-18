package io.github.zzuegg.brainiac.experiments.document;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DocumentParsingTest {

    @Test
    void layoutTaggersUseBoxesWhenReadingOrderIsShuffled() {
        DocumentParsing.TagScores rules = DocumentParsing.tagging(DocumentParsing.Tagger.RULES, 100, 0, 0, false,
                null, 100, 1);
        System.out.printf("Tagging %-12s shuffled %-5s F1 %.3f%n", rules.tagger(), rules.shuffled(), rules.f1());
        assertTrue(rules.f1() > 0.95, "nearest-key rules are near exact on ground-truth tokens");
        double[] f1 = new double[2];
        int k = 0;
        for (DocumentParsing.Tagger t : new DocumentParsing.Tagger[] {DocumentParsing.Tagger.LAYOUT_2D,
            DocumentParsing.Tagger.POSITION_1D}) {
            DocumentParsing.TagScores s = DocumentParsing.tagging(t, 100, 250, 0, true, null, 100, 1);
            f1[k++] = s.f1();
            System.out.printf("Tagging %-12s shuffled %-5s F1 %.3f (%d ms)%n", s.tagger(), s.shuffled(), s.f1(),
                    s.millis());
        }
        assertTrue(f1[0] > f1[1], "2-D boxes help when reading order is shuffled");
    }

    @Test
    void pipelineAndOcrFreeParsing() {
        for (DocumentParsing.Parser p : new DocumentParsing.Parser[] {DocumentParsing.Parser.OCR_RULES,
            DocumentParsing.Parser.DONUT}) {
            DocumentParsing.ParseScores s = DocumentParsing.parse(p, 200, 40, 0.02, 1);
            System.out.printf("Parsing %-16s field F1 %.3f exact %.3f (%d ms)%n", s.parser(), s.f1(), s.exact(),
                    s.millis());
            assertTrue(s.f1() >= 0 && s.f1() <= 1, "F1 is a share");
        }
    }
}
