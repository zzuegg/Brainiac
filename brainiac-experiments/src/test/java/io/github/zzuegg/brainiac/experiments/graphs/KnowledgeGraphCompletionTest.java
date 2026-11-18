package io.github.zzuegg.brainiac.experiments.graphs;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.graph.KnowledgeGraphEmbedding;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

/** Wave 7 item 7.1.13. */
class KnowledgeGraphCompletionTest {

    @Test
    void scoringFunctionsMatchRelationPatterns() {
        Map<String, Double> r = KnowledgeGraphCompletion.compare(List.of(KnowledgeGraphEmbedding.Model.values()), 10,
                300, 1);
        System.out.printf("7.1.13 filtered MRR: ComplEx %.3f, RotatE %.3f, TransE %.3f, DistMult %.3f; two-tower %.3f,"
                + " frequency %.3f. Sibling (symmetric): DistMult %.3f vs TransE %.3f. Parent (antisymmetric): ComplEx"
                + " %.3f vs DistMult %.3f. Grandparent: RotatE %.3f%n", r.get("COMPLEX MRR"), r.get("ROTATE MRR"),
                r.get("TRANSE MRR"), r.get("DISTMULT MRR"), r.get("two-tower MRR"), r.get("frequency MRR"),
                r.get("DISTMULT MRR sibling"), r.get("TRANSE MRR sibling"), r.get("COMPLEX MRR parent"),
                r.get("DISTMULT MRR parent"), r.get("ROTATE MRR grandparent"));
        assertTrue(r.get("COMPLEX MRR") > r.get("two-tower MRR") + 0.3, r.toString());
        assertTrue(r.get("DISTMULT MRR sibling") > r.get("TRANSE MRR sibling") + 0.2, r.toString());
        assertTrue(r.get("COMPLEX MRR parent") > r.get("DISTMULT MRR parent") + 0.2, r.toString());
    }
}
