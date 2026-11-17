package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.nn.HierarchicalReasoningModel;
import io.github.zzuegg.brainiac.experimental.nn.TinyRecursiveModel;

import java.util.List;

import org.junit.jupiter.api.Test;

class CompressArcTest {

    @Test
    void compressionSolvesPuzzlesWithoutPretraining() {
        String[] families = {"colour map", "mirror", "row fill", "column fill"};
        List<CompressArc.Puzzle> arc = CompressArc.arcLike(2, 1);
        int[] solvedPerFamily = new int[4];
        double arcCells = 0;
        double copyCells = 0;
        int copySolved = 0;
        for (int i = 0; i < arc.size(); i++) {
            CompressArc.Puzzle p = arc.get(i);
            int[] pred = CompressArc.solve(p, 400, 0.1f, i);
            solvedPerFamily[i / 2] += CompressArc.solved(p, pred) ? 1 : 0;
            arcCells += CompressArc.cellAccuracy(p, pred) / arc.size();
            int[] copy = p.inputs()[p.inputs().length - 1];
            copySolved += CompressArc.solved(p, copy) ? 1 : 0;
            copyCells += CompressArc.cellAccuracy(p, copy) / arc.size();
        }
        List<List<LatinSquarePuzzles.Puzzle>> data = LatinSquarePuzzles.split(6, 400, 1);
        List<LatinSquarePuzzles.Puzzle> tests = data.get(1).subList(0, 6);
        List<CompressArc.Puzzle> latin = CompressArc.latin(tests, data.get(0), 2);
        int latinSolved = 0;
        double latinCells = 0;
        for (int i = 0; i < latin.size(); i++) {
            int[] pred = CompressArc.solve(latin.get(i), 400, 0.1f, i);
            latinSolved += CompressArc.solved(latin.get(i), pred) ? 1 : 0;
            latinCells += CompressArc.cellAccuracy(latin.get(i), pred) / latin.size();
        }
        TinyRecursiveModel trm = TinyRecursiveModel.of(LatinSquarePuzzles.INPUTS, 64, LatinSquarePuzzles.OUTPUTS, 3,
                2, 3, 1);
        LatinSquarePuzzles.train(trm::supervised, trm.parameters(), data.get(0), 300, 64, 1);
        LatinSquarePuzzles.Evaluation te = LatinSquarePuzzles.evaluate(trm::supervised, tests);
        HierarchicalReasoningModel hrm = HierarchicalReasoningModel.of(LatinSquarePuzzles.INPUTS, 64,
                LatinSquarePuzzles.OUTPUTS, 3, 2, 3, 2);
        LatinSquarePuzzles.train(hrm::supervised, hrm.parameters(), data.get(0), 300, 64, 2);
        LatinSquarePuzzles.Evaluation he = LatinSquarePuzzles.evaluate(hrm::supervised, tests);
        int t = te.solved().length - 1;
        int h = he.solved().length - 1;
        StringBuilder line = new StringBuilder("CompressARC, ARC-like solved:");
        for (int f = 0; f < 4; f++) {
            line.append(String.format(" %s %d/2;", families[f], solvedPerFamily[f]));
        }
        line.append(String.format(" cells %.2f vs copy-input %.2f (solved %d/8). Latin squares (6 test, blank cells):"
                + " CompressARC solved %d/6 cells %.2f; TRM solved %.2f cells %.2f; HRM solved %.2f cells %.2f"
                + " (trained on 1600 puzzles)", arcCells, copyCells, copySolved, latinSolved, latinCells,
                te.solved()[t], te.blankAccuracy()[t], he.solved()[h], he.blankAccuracy()[h]));
        System.out.println(line);
        assertTrue(solvedPerFamily[0] >= 1, "colour mapping is compressed: " + line);
        assertTrue(arcCells > copyCells + 0.1, line.toString());
    }
}
