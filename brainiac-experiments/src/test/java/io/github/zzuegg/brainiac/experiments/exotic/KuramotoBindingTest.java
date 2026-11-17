package io.github.zzuegg.brainiac.experiments.exotic;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.nn.TinyRecursiveModel;
import io.github.zzuegg.brainiac.experiments.reasoning.LatinSquarePuzzles;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Wave 6, 6.1: AKOrN, binding by synchrony and Latin-square reasoning. */
class KuramotoBindingTest {

    @Test
    void ariIsOneForIdenticalPartitions() {
        assertEquals(1, KuramotoBinding.ari(new int[] {1, 1, 2, 2}, new int[] {5, 5, 0, 0}), 1e-9);
    }

    @Test
    void synchronyBindsObjectsAndOscillatorsSolveLatinSquares() {
        KuramotoBinding.Discovery d = KuramotoBinding.discover(20, 40, 1);
        List<List<LatinSquarePuzzles.Puzzle>> data = LatinSquarePuzzles.split(6, 400, 1);
        KuramotoBinding.LatinModel akorn = KuramotoBinding.latinSquares(8, 6, 1);
        LatinSquarePuzzles.train(akorn::supervised, akorn.parameters(), data.get(0), 200, 64, 1);
        LatinSquarePuzzles.Evaluation ae = LatinSquarePuzzles.evaluate(akorn::supervised, data.get(1));
        TinyRecursiveModel trm = TinyRecursiveModel.of(LatinSquarePuzzles.INPUTS, 64, LatinSquarePuzzles.OUTPUTS, 3,
                2, 3, 1);
        LatinSquarePuzzles.train(trm::supervised, trm.parameters(), data.get(0), 200, 64, 1);
        LatinSquarePuzzles.Evaluation te = LatinSquarePuzzles.evaluate(trm::supervised, data.get(1));
        double akAcc = ae.blankAccuracy()[ae.blankAccuracy().length - 1];
        double trmAcc = te.blankAccuracy()[te.blankAccuracy().length - 1];
        System.out.printf("AKOrN objects FG-ARI %.2f vs k-means(xy) %.2f, soft slots(xy) %.2f (energy drop %.2f);"
                + " Latin squares blank acc / solved: AKOrN %.2f/%.2f vs TRM %.2f/%.2f%n", d.kuramoto(), d.kmeans(),
                d.slots(), d.energyDrop(), akAcc, ae.solved()[ae.solved().length - 1], trmAcc,
                te.solved()[te.solved().length - 1]);
        assertTrue(d.kuramoto() > d.kmeans(), "synchrony segments better than position clustering");
        assertTrue(d.kuramoto() > 0.8, "FG-ARI " + d.kuramoto());
        assertTrue(akAcc > 0.4, "AKOrN well above the 25% of guessing: " + akAcc);
    }
}
