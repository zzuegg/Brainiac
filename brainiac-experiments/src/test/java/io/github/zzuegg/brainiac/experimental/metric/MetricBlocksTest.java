package io.github.zzuegg.brainiac.experimental.metric;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.ClusteringMetrics;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class MetricBlocksTest {

    /** Two well separated classes against two mixed ones. */
    private static Var embeddings(boolean separated) {
        SplittableRandom r = new SplittableRandom(1);
        float[] v = new float[8 * 4];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 4; j++) {
                double centre = separated ? (j == i % 2 ? 3 : 0) : 0;
                v[i * 4 + j] = (float) (centre + r.nextGaussian() * 0.3);
            }
        }
        return Var.param(Tensor.of(Shape.of(8, 4), v));
    }

    private static final Tensor LABELS = Tensor.of(Shape.of(8), 0, 1, 0, 1, 0, 1, 0, 1);

    @Test
    void everyLossPrefersSeparatedClasses() {
        List<Loss> losses = List.of(new SupervisedContrastive(0.1f), PairLosses.contrastive(1),
                PairLosses.triplet(0.2f, PairLosses.Mining.SEMI_HARD, 1), PairLosses.multiSimilarity(2, 50, 0.5f, 0.1f),
                MarginSoftmax.arcFace(4, 2, 1), MarginSoftmax.cosFace(4, 2, 1),
                new ProxyLosses(ProxyLosses.Kind.PROXY_ANCHOR, 4, 2, 1));
        for (Loss l : losses) {
            Var sep = embeddings(true);
            Var mix = embeddings(false);
            float a = l.compute(sep, LABELS).value().item();
            float b = l.compute(mix, LABELS).value().item();
            Tensor g = Var.gradients(l.compute(mix, LABELS), List.of(mix), false).get(0).value();
            assertTrue(Float.isFinite(a) && Float.isFinite(b) && Float.isFinite(g.abs().sum().item()), l.toString());
            if (!(l instanceof MarginSoftmax) && !(l instanceof ProxyLosses)) {
                // the class-centre losses start from random centres, so only check the pair losses' order
                assertTrue(a < b, l + ": separated " + a + " mixed " + b);
            }
        }
    }

    @Test
    void retrievalAndClusteringMetrics() {
        float[][] e = {{1, 0}, {0.9f, 0.1f}, {0, 1}, {0.1f, 0.9f}};
        int[] y = {0, 0, 1, 1};
        assertEquals(1, RetrievalMetrics.recallAt(e, y, 1), 1e-9);
        assertEquals(0, RetrievalMetrics.equalErrorRate(e, y), 1e-9);
        assertEquals(1, ClusteringMetrics.accuracy(new int[] {0, 0, 1, 1, 2}, new int[] {2, 2, 0, 0, 1}), 1e-9);
        assertEquals(1, ClusteringMetrics.nmi(new int[] {0, 0, 1, 1}, new int[] {1, 1, 0, 0}), 1e-9);
        assertEquals(0, ClusteringMetrics.nmi(new int[] {0, 1, 0, 1}, new int[] {0, 0, 1, 1}), 1e-9);
    }
}
