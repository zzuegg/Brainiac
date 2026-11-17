package io.github.zzuegg.brainiac.experiments.data;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.data.BenchmarkMethodology.ContaminationResult;
import io.github.zzuegg.brainiac.experiments.data.BenchmarkMethodology.FluidResult;
import io.github.zzuegg.brainiac.experiments.data.ModelCollapse.Generation;
import io.github.zzuegg.brainiac.experiments.data.ModelCollapse.Mode;

import org.junit.jupiter.api.Test;

class EvaluationAndCollapseTest {

    @Test
    void accumulatingRealDataAvoidsModelCollapse() {
        Generation[] replace = ModelCollapse.run(Mode.REPLACE, 5, 60, 1);
        Generation[] accumulate = ModelCollapse.run(Mode.ACCUMULATE, 5, 60, 1);
        StringBuilder line = new StringBuilder();
        for (int g = 0; g < replace.length; g++) {
            line.append(String.format(" g%d %.3f/%.3f (div %.2f/%.2f)", g, replace[g].testLoss(),
                    accumulate[g].testLoss(), replace[g].diversity(), accumulate[g].diversity()));
        }
        System.out.println("Model collapse, test loss replace/accumulate:" + line);
        Generation r5 = replace[5];
        Generation a5 = accumulate[5];
        assertTrue(r5.testLoss() > replace[0].testLoss() + 0.1, "replacing the data collapses the model");
        assertTrue(a5.testLoss() < r5.testLoss() && a5.diversity() > r5.diversity(),
                "accumulating keeps it closer to the real distribution and more diverse");
    }

    @Test
    void irtAbilityRanksCheckpointsBetterThanARandomSubset() {
        FluidResult f = BenchmarkMethodology.fluid(200, 20, 1);
        System.out.printf("Fluid benchmarking (Kendall tau with true quality, 10%% of items): IRT adaptive %.3f, "
                + "IRT on random items %.3f vs random-subset accuracy %.3f; all items: IRT %.3f, accuracy %.3f; "
                + "step-to-step drops %.2f vs %.2f (true %.2f)%n", f.irtTau(), f.irtRandomTau(), f.randomTau(),
                f.irtFullTau(), f.fullTau(), f.irtReversals(), f.randomReversals(), f.trueReversals());
        assertTrue(f.irtTau() > f.randomTau(), "IRT on 10% of the items ranks checkpoints more faithfully");
    }

    @Test
    void minKPlusPlusFindsThePlantedTestSet() {
        ContaminationResult c = BenchmarkMethodology.contamination(1);
        System.out.printf("Contamination AUROC: Min-K%%++ %.3f vs Min-K%% %.3f, loss threshold %.3f%n",
                c.minKPlusPlus(), c.minK(), c.lossThreshold());
        assertTrue(c.minKPlusPlus() > 0.6, "the planted half is detectable");
        assertTrue(c.minKPlusPlus() > c.lossThreshold(), "and Min-K%++ detects it better than the loss");
    }
}
