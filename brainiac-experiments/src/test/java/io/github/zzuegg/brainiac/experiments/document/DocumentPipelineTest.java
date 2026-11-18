package io.github.zzuegg.brainiac.experiments.document;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class DocumentPipelineTest {

    @Test
    void ocrPipelinesExtractFields() {
        List<DocumentParsing.ParseScores> rows = DocumentParsing.pipelines(350, 150, 40, 0.02, 1);
        for (DocumentParsing.ParseScores s : rows) {
            System.out.printf("Parsing %-16s field F1 %.3f exact %.3f (%d ms)%n", s.parser(), s.f1(), s.exact(),
                    s.millis());
        }
        assertTrue(rows.get(0).f1() > 0.6, "OCR plus nearest-key rules recovers fields");
        assertTrue(rows.get(1).f1() > 0.6, "OCR plus the layout tagger recovers fields");
    }
}
