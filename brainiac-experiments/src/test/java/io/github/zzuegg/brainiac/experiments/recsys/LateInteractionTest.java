package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.SyntheticCorpus.Kind;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;

class LateInteractionTest {

    @Test
    void colbertMatchesTermsSeparatelyAndCompressesWell() {
        Map<String, Double> bytes = new LinkedHashMap<>();
        Map<String, Map<Kind, RetrievalShootout.Metrics>> r = new RetrievalShootout(1).lateInteraction(250, 32, bytes);
        System.out.println(RetrievalShootoutTest.table("ColBERT", r));
        System.out.println("bytes per document: " + bytes);
        Map<Kind, RetrievalShootout.Metrics> fp32 = r.get("ColBERT fp32");
        Map<Kind, RetrievalShootout.Metrics> small = r.get("ColBERT 2-bit residual");
        assertTrue(fp32.get(Kind.PARAPHRASE).mrr10() > r.get("BM25").get(Kind.PARAPHRASE).mrr10() + 0.1, r.toString());
        assertTrue(small.get(Kind.MULTI_ASPECT).mrr10() > 0.7 * fp32.get(Kind.MULTI_ASPECT).mrr10(), r.toString());
        assertTrue(bytes.get("ColBERT 2-bit residual") < bytes.get("ColBERT fp32") / 8, bytes.toString());
    }
}
