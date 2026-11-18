package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

/** AUROC, ROC and average precision against sklearn's documented examples. */
class ScoresTest {

    @Test
    void sklearnExamples() {
        double[] s = {0.1, 0.4, 0.35, 0.8};
        boolean[] y = {false, false, true, true};
        assertEquals(0.75, Scores.auroc(s, y), 1e-12);
        assertEquals(0.8333333333333333, Scores.averagePrecision(s, y), 1e-12);
        double[][] roc = Scores.roc(s, y);
        assertArrayEquals(new double[] {0, 0, 0.5, 0.5, 1}, roc[0], 1e-12);
        assertArrayEquals(new double[] {0, 0.5, 0.5, 1, 1}, roc[1], 1e-12);
        assertArrayEquals(new double[] {Double.POSITIVE_INFINITY, 0.8, 0.4, 0.35, 0.1}, roc[2]);
        // ties count half
        assertEquals(0.875, Scores.auroc(new double[] {0.5, 0.5, 0.2, 0.9}, new boolean[] {true, false, false, true}), 1e-12);
        assertThrows(IllegalArgumentException.class, () -> Scores.auroc(new double[] {1, 2}, new boolean[] {true, true}));
    }

    @Test
    void treesCanBeRead() {
        double[][] x = new double[40][2];
        double[] t = new double[40];
        for (int i = 0; i < 40; i++) {
            x[i][0] = i;
            x[i][1] = (i * 7) % 5;
            t[i] = i < 20 ? 1 : 3;
        }
        GradientBoostedTrees g = GradientBoostedTrees.fit(x, t, GradientBoostedTrees.Objective.REGRESSION,
                GradientBoostedTrees.Settings.defaults().withTrees(5).withDepth(2));
        for (double[] row : new double[][] {{3, 1}, {30, 4}}) {
            double raw = g.base();
            for (int k = 0; k < g.size(); k++) {
                GradientBoostedTrees.Tree tree = g.tree(k);
                int n = 0;
                while (tree.feature()[n] >= 0) {
                    n = row[tree.feature()[n]] < tree.threshold()[n] ? tree.left()[n] : tree.right()[n];
                }
                raw += g.learningRate() * tree.value()[n];
            }
            assertEquals(g.raw(row), raw, 1e-12);
        }
        g.tree(0).value()[0] = 1e9;   // a copy: the model is unchanged
        assertEquals(g.raw(new double[] {3, 1}), g.raw(new double[] {3, 1}));
    }
}
