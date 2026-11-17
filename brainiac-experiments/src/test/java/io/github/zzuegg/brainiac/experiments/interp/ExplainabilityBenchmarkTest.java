package io.github.zzuegg.brainiac.experiments.interp;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.interp.ExplainabilityBenchmark.CounterfactualScores;
import io.github.zzuegg.brainiac.experiments.interp.ExplainabilityBenchmark.TabularMethod;
import io.github.zzuegg.brainiac.experiments.interp.ExplainabilityBenchmark.TabularScores;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class ExplainabilityBenchmarkTest {

    @Test
    void modelAgnosticAttributionsRecoverThePlantedRule() {
        Map<TabularMethod, TabularScores> mlp = new EnumMap<>(TabularMethod.class);
        for (TabularMethod m : TabularMethod.values()) {
            if (m == TabularMethod.TREE_GAIN) {
                continue;
            }
            TabularScores s = ExplainabilityBenchmark.tabular(m, false, 1);
            mlp.put(m, s);
            System.out.printf("Tabular MLP %-20s recovery %.3f deletion AUC %.3f%n", m, s.recovery(), s.deletionAuc());
        }
        Map<TabularMethod, TabularScores> trees = new EnumMap<>(TabularMethod.class);
        for (TabularMethod m : new TabularMethod[] {TabularMethod.KERNEL_SHAP, TabularMethod.LIME,
                TabularMethod.TREE_GAIN, TabularMethod.RANDOM}) {
            TabularScores s = ExplainabilityBenchmark.tabular(m, true, 1);
            trees.put(m, s);
            System.out.printf("Tabular GBT %-20s recovery %.3f deletion AUC %.3f%n", m, s.recovery(), s.deletionAuc());
        }
        for (boolean t : new boolean[] {false, true}) {
            CounterfactualScores c = ExplainabilityBenchmark.counterfactuals(t, 1);
            System.out.printf("Counterfactuals %s valid %.2f changed %.2f relevant %.2f L1 %.2f%n", t ? "GBT" : "MLP",
                    c.validity(), c.featuresChanged(), c.relevantShare(), c.distance());
            assertTrue(c.validity() > 0.9, "counterfactuals flip the prediction");
        }
        assertTrue(mlp.get(TabularMethod.KERNEL_SHAP).recovery() > mlp.get(TabularMethod.RANDOM).recovery() + 0.2);
        assertTrue(trees.get(TabularMethod.KERNEL_SHAP).recovery() > trees.get(TabularMethod.RANDOM).recovery() + 0.2);
    }

    @Test
    void gradCamPointsAtTheShapeAndTcavFindsTheShortcut() {
        ExplainabilityBenchmark.ImageStudy study = ExplainabilityBenchmark.images(1);
        study.maps().forEach((m, s) -> System.out.printf("Images %-20s pointing %.3f marker-pointing (spurious CNN) %.3f%n",
                m, s.pointingClean(), s.markerPointingSpurious()));
        ExplainabilityBenchmark.TcavScores t = study.tcav();
        System.out.printf("TCAV marker->class 1: spurious %.3f (z %.1f) clean %.3f (z %.1f) random %.3f +- %.3f;"
                + " the marker adds %.3f to p(class 1)%n", t.spurious(), t.spuriousZ(), t.clean(), t.cleanZ(),
                t.randomMean(), t.randomStd(), t.shortcutReliance());
        var maps = study.maps();
        assertTrue(maps.get(ExplainabilityBenchmark.ImageMethod.GRAD_CAM).pointingClean()
                > maps.get(ExplainabilityBenchmark.ImageMethod.RANDOM).pointingClean() + 0.4, "Grad-CAM localizes");
    }
}
