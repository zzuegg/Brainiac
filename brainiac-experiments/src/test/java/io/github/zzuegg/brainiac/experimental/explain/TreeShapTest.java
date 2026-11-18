package io.github.zzuegg.brainiac.experimental.explain;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.PlantedFunctions;

import java.util.Arrays;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class TreeShapTest {

    private static final PlantedFunctions.Table DATA = PlantedFunctions.interactions(400, 7, 3);
    private static final TreeView MODEL = TreeView.boost(DATA.x(), DATA.y(), 20, 3, 0.2, 3);

    @Test
    void bothModesMatchBruteForceAndAreLocallyAccurate() {
        TreeShap shap = new TreeShap(MODEL);
        double[][] background = Arrays.copyOfRange(DATA.x(), 100, 110);
        double worstPath = 0;
        double worstInt = 0;
        for (int r = 0; r < 3; r++) {
            double[] x = DATA.x()[r];
            double[] path = shap.values(TreeShap.Mode.PATH_DEPENDENT, x, null);
            double[] pathTruth = TreeShap.bruteForce(7, m -> shap.conditionalExpectation(x,
                    CoalitionGame.members(m, 7)));
            double[] inter = shap.values(TreeShap.Mode.INTERVENTIONAL, x, background);
            double[] interTruth = TreeShap.bruteForce(7, m -> TreeShap.interventionalGame(MODEL::predict, x,
                    background, m));
            for (int j = 0; j < 7; j++) {
                worstPath = Math.max(worstPath, Math.abs(path[j] - pathTruth[j]));
                worstInt = Math.max(worstInt, Math.abs(inter[j] - interTruth[j]));
            }
            assertEquals(MODEL.predict(x) - shap.expectedValue(TreeShap.Mode.PATH_DEPENDENT, null),
                    Arrays.stream(path).sum(), 1e-9);
            double[][] phi = shap.interactionValues(x);
            for (int i = 0; i < 7; i++) {
                assertEquals(path[i], Arrays.stream(phi[i]).sum(), 1e-9);
            }
        }
        System.out.printf("TreeSHAP vs brute force: path-dependent max error %.2e, interventional %.2e%n", worstPath,
                worstInt);
        assertTrue(worstPath < 1e-9 && worstInt < 1e-9);
    }

    @Test
    void interactionIndicesAgreeAcrossMethods() {
        double[] x = DATA.x()[0];
        double[][] background = Arrays.copyOfRange(DATA.x(), 100, 108);
        CoalitionGame game = CoalitionGame.interventional(MODEL::predict, x, background);
        ShapleyInteractions.Interactions exact = ShapleyInteractions.exact(ShapleyInteractions.Index.SII, game, 7);
        ShapleyInteractions.Interactions tree = ShapleyInteractions.tree(ShapleyInteractions.Index.SII, MODEL, x,
                background);
        double worst = 0;
        for (int i = 0; i < 7; i++) {
            worst = Math.max(worst, Math.abs(exact.first()[i] - tree.first()[i]));
            for (int j = 0; j < 7; j++) {
                worst = Math.max(worst, Math.abs(exact.pairs()[i][j] - tree.pairs()[i][j]));
            }
        }
        double gap = game.value(full(7)) - game.value(new boolean[7]);
        for (ShapleyInteractions.Index index : new ShapleyInteractions.Index[] {ShapleyInteractions.Index.K_SII,
            ShapleyInteractions.Index.FAITH_SHAP}) {
            assertEquals(gap, ShapleyInteractions.exact(index, game, 7).total(), 1e-6, index + " is efficient");
        }
        // SHAP-IQ with the whole budget enumerates every coalition, hence is exact
        ShapleyInteractions.Interactions iq = ShapleyInteractions.shapIq(ShapleyInteractions.Index.SII, game, 7, 128,
                new SplittableRandom(1));
        double iqWorst = 0;
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) {
                iqWorst = Math.max(iqWorst, Math.abs(exact.pairs()[i][j] - iq.pairs()[i][j]));
            }
        }
        System.out.printf("SII: TREE vs EXACT max error %.2e, SHAP-IQ at full budget %.2e%n", worst, iqWorst);
        assertTrue(worst < 1e-9 && iqWorst < 1e-9);
    }

    @Test
    void estimatorsConvergeToTheExactValues() {
        double[] x = DATA.x()[1];
        double[][] background = Arrays.copyOfRange(DATA.x(), 100, 108);
        CoalitionGame game = CoalitionGame.interventional(MODEL::predict, x, background);
        double[] exact = new TreeShap(MODEL).values(TreeShap.Mode.INTERVENTIONAL, x, background);
        for (ShapleyEstimators.Estimator e : ShapleyEstimators.Estimator.values()) {
            double[] est = ShapleyEstimators.estimate(e, game, 7, 2000, new SplittableRandom(2));
            double err = 0;
            for (int j = 0; j < 7; j++) {
                err = Math.max(err, Math.abs(est[j] - exact[j]));
            }
            System.out.printf("Estimator %-22s at 2000 evaluations: max error %.4f%n", e, err);
            assertTrue(err < 0.05, e + " converges");
        }
    }

    private static boolean[] full(int n) {
        boolean[] s = new boolean[n];
        Arrays.fill(s, true);
        return s;
    }
}
