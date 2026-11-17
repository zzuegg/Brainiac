package io.github.zzuegg.brainiac.experimental.evolve;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.Batch;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class TransformerBlockSearchTest {

    /** Predict the previous token. */
    private static Batch shift(SplittableRandom r) {
        float[] x = new float[8 * 6];
        float[] y = new float[8 * 6];
        for (int i = 0; i < x.length; i++) {
            x[i] = r.nextInt(6);
        }
        for (int b = 0; b < 8; b++) {
            for (int t = 0; t < 6; t++) {
                y[b * 6 + t] = t == 0 ? -1 : x[b * 6 + t - 1];
            }
        }
        return new Batch(Tensor.of(Shape.of(8, 6), x), Tensor.of(Shape.of(8, 6), y));
    }

    @Test
    void mutationAndCrossoverKeepGenomesValid() {
        SplittableRandom r = new SplittableRandom(1);
        TransformerBlockGenome a = TransformerBlockGenome.random(r);
        TransformerBlockGenome b = TransformerBlockGenome.random(r);
        for (int i = 0; i < 500; i++) {
            a = a.mutate(r);
            b = a.cross(b, r).mutate(r);
            assertEquals(0, a.heads() % a.kvHeads());
            assertEquals(0, b.heads() % b.kvHeads());
            a.config(8);   // throws if invalid
        }
    }

    @Test
    void candidatesTrainedTogetherMatchTrainingAlone() {
        SplittableRandom r = new SplittableRandom(2);
        TransformerBlockGenome a = TransformerBlockGenome.random(r);
        TransformerBlockGenome b = TransformerBlockGenome.random(r);
        TransformerBlockSearch search = TransformerBlockSearch.of(6, 6, 8, TransformerBlockSearchTest::shift).steps(10);
        double alone = search.trainTogether(List.of(a), 3).getFirst().validationLoss();
        double together = search.trainTogether(List.of(b, a), 3).get(1).validationLoss();
        assertEquals(alone, together, 1e-4);
    }

    @Test
    void searchImprovesOnItsFirstGeneration() {
        TransformerBlockSearch search = TransformerBlockSearch.of(6, 6, 8, TransformerBlockSearchTest::shift)
                .population(6, 2).steps(40).seed(4);
        List<TransformerBlockSearch.Candidate> last = search.run(3, List.of());
        double firstMedian = search.history().getFirst().medianLoss();
        assertTrue(last.getFirst().validationLoss() < firstMedian,
                last.getFirst().validationLoss() + " vs first median " + firstMedian);
        assertTrue(search.history().getLast().trained() <= 6);
    }
}
