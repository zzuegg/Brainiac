package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SatNetLayer;
import io.github.zzuegg.brainiac.experimental.train.SemanticLoss;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/** Items 7.6.12 (semantic loss) and 7.6.14 (SATNet) as new LatinSquarePuzzles rows, against the plain MLP. */
class ConstraintLatinSquaresTest {

    private final List<List<LatinSquarePuzzles.Puzzle>> data = LatinSquarePuzzles.split(6, 400, 1);

    private LatinSquarePuzzles.Evaluation mlp(float semanticWeight) {
        Model mlp = Model.sequential().input(LatinSquarePuzzles.INPUTS).dense(128, Activation.TANH)
                .dense(LatinSquarePuzzles.OUTPUTS).seed(4).build();
        LatinSquarePuzzles.train(x -> List.of(mlp.forward(x, true)), mlp.parameters(), data.get(0), 300, 64, 4,
                semanticWeight == 0 ? null : logits -> {
                    int b = logits.shape().dim(0);
                    Var z = logits.reshape(b, 16, 4);
                    Var p = z.sub(z.logsumexp(2, true)).exp().reshape(b, 64);
                    return SemanticLoss.latinSquare(p, 4, false).mul(semanticWeight);
                });
        return LatinSquarePuzzles.evaluate(x -> List.of(mlp.forward(x, false)), data.get(1));
    }

    @Test
    void semanticLossAndSatNetAgainstThePlainMlp() {
        LatinSquarePuzzles.Evaluation plain = mlp(0);
        LatinSquarePuzzles.Evaluation semantic = mlp(0.05f);
        SatNetLayer sat = new SatNetLayer(64, 96, 16, new SplittableRandom(5)).sweeps(10);
        long t0 = System.nanoTime();
        LatinSquarePuzzles.trainBits(sat::forward, sat.parameters(), data.get(0), 60, 32, 0.05f, 5);
        long satMs = (System.nanoTime() - t0) / 1_000_000;
        LatinSquarePuzzles.Evaluation satEval = LatinSquarePuzzles.evaluateBits(sat::forward, data.get(1));
        System.out.printf("Latin squares (6 blanks) cell acc / solved: MLP %.3f/%.3f, MLP + semantic loss %.3f/%.3f,"
                + " SATNet %.3f/%.3f (%d ms training; wave 6: TRM/HRM ~0.72, AKOrN 0.62 solved)%n",
                plain.blankAccuracy()[0], plain.solved()[0], semantic.blankAccuracy()[0], semantic.solved()[0],
                satEval.blankAccuracy()[0], satEval.solved()[0], satMs);
        assertTrue(semantic.blankAccuracy()[0] > 0.8, "the MLP still learns with the semantic loss");
        assertTrue(satEval.solved()[0] > plain.solved()[0], "SATNet solves more squares than the MLP");

    }
}
