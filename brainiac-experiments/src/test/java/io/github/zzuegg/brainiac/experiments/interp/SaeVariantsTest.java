package io.github.zzuegg.brainiac.experiments.interp;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.nn.SparseAutoencoder.Sparsity;

import java.util.List;

import org.junit.jupiter.api.Test;

class SaeVariantsTest {

    @Test
    void matryoshkaAbsorbsLessThanPlainTopK() {
        List<SuperpositionRecovery.HierarchyOutcome> out = SuperpositionRecovery.hierarchical(4, 3, 16, 1500, 3);
        for (SuperpositionRecovery.HierarchyOutcome o : out) {
            System.out.printf("SAE %-11s recovered %d/%d  absorption %.3f  dead %.2f  L0 %.2f  EV %.3f%n", o.mode(),
                    o.recovered(), o.trueFeatures(), o.absorption(), o.dead(), o.l0(), o.explainedVariance());
            assertTrue(o.explainedVariance() > 0.5, o.toString());
        }
        SuperpositionRecovery.HierarchyOutcome topK = out.get(Sparsity.TOP_K.ordinal());
        SuperpositionRecovery.HierarchyOutcome mat = out.get(Sparsity.MATRYOSHKA.ordinal());
        assertTrue(mat.absorption() <= topK.absorption(), mat + " vs " + topK);
        assertTrue(out.get(Sparsity.JUMP_RELU.ordinal()).l0() < 6, out.get(Sparsity.JUMP_RELU.ordinal()).toString());
    }
}
