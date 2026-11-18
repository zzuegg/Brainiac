package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.SyntheticCorpus.Kind;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;

class SparseRetrievalTest {

    @Test
    void spladeLearnsExpansionTermsForParaphrases() {
        Map<String, Double> nnz = new LinkedHashMap<>();
        Map<String, Map<Kind, RetrievalShootout.Metrics>> r = new RetrievalShootout(1).sparse(200,
                new float[] {0.01f, 0.1f}, nnz);
        System.out.println(RetrievalShootoutTest.table("SPLADE", r));
        System.out.println("non-zero terms per document: " + nnz);
        double splade = r.get("SPLADE flops=0.01").get(Kind.PARAPHRASE).mrr10();
        assertTrue(splade > r.get("BM25").get(Kind.PARAPHRASE).mrr10() + 0.2, r.toString());
        assertTrue(nnz.get("SPLADE flops=0.1") < nnz.get("SPLADE flops=0.01"), nnz.toString());
    }
}
