package io.github.zzuegg.brainiac.experiments.optim;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.PsgdKron;
import io.github.zzuegg.brainiac.experimental.train.Shampoo;
import io.github.zzuegg.brainiac.experimental.train.Soap;
import io.github.zzuegg.brainiac.experimental.train.StableAdamW;
import io.github.zzuegg.brainiac.experiments.optim.ModernOptimizerExperiment.MarkovSource;
import io.github.zzuegg.brainiac.experiments.optim.ModernOptimizerExperiment.Run;
import io.github.zzuegg.brainiac.experiments.optim.ModernOptimizerExperiment.Stress;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.Supplier;

import org.junit.jupiter.api.Test;

class PreconditionerStabilityTest {

    @Test
    void psgdKronAgainstShampooSoapAndMuon() {
        MarkovSource source = MarkovSource.random(10, 1);
        Predicate<Var> matrices = v -> v.shape().rank() == 2 && v.shape().dim(0) > 1;
        Map<String, Supplier<Optimizer>> zoo = new LinkedHashMap<>();
        zoo.put("psgd-kron", () -> PsgdKron.of(3e-3f));
        zoo.put("shampoo", () -> new Shampoo(0.05f, 0.9f, 10, 128));
        zoo.put("soap", () -> Soap.of(3e-3f));
        zoo.put("muon", () -> Optimizer.groups(Optimizer.adamW(3e-3f, 0.01f),
                new Optimizer.Group(matrices, Optimizer.muon(0.02f))));
        zoo.put("adamw", () -> Optimizer.adamW(3e-3f, 0.01f));
        StringBuilder line = new StringBuilder();
        Map<String, Run> runs = new LinkedHashMap<>();
        for (var e : zoo.entrySet()) {
            Run r = ModernOptimizerExperiment.train(e.getValue(), source, 64, 400, 32, 50, 1);
            runs.put(e.getKey(), r);
            line.append(String.format(" %s %.3f@100/%.3f@400 %.2fs;", e.getKey(), r.lossCurve()[1], r.finalLoss(),
                    r.seconds()));
        }
        System.out.println("PSGD-Kron (loss at step 100 / 400, train time):" + line);
        Run psgd = runs.get("psgd-kron");
        assertTrue(psgd.lossCurve()[1] < runs.get("adamw").lossCurve()[1], "the preconditioner speeds up early steps");
        assertTrue(psgd.finalLoss() < runs.get("shampoo").finalLoss() && psgd.finalLoss() < 1.15f,
                "PSGD trains to a good loss");
    }

    @Test
    void stableAdamWUnderAStaleSecondMoment() {
        List<Long> seeds = List.of(1L, 2L, 3L, 4L, 5L, 6L);
        Function<Float, Optimizer> adamW = lr -> Optimizer.adamW(lr, 0.01f);
        Function<Float, Optimizer> stable = lr -> StableAdamW.stable(lr, 0.01f);
        Function<Float, Optimizer> atan2 = lr -> StableAdamW.atan2(lr, 0.01f);
        Stress a = ModernOptimizerExperiment.stress(adamW, 0.03f, seeds, 150, 100, 300);
        Stress s = ModernOptimizerExperiment.stress(stable, 0.03f, seeds, 150, 100, 300);
        Stress t = ModernOptimizerExperiment.stress(atan2, 0.03f, seeds, 150, 100, 300);
        Stress aHigh = ModernOptimizerExperiment.stress(adamW, 0.1f, seeds, 150, 100, 300);
        Stress sHigh = ModernOptimizerExperiment.stress(stable, 0.1f, seeds, 150, 100, 300);
        Stress tHigh = ModernOptimizerExperiment.stress(atan2, 0.1f, seeds, 150, 100, 300);
        System.out.printf("StableAdamW / Adam-atan2 / AdamW after a 100x gradient jump: lr 0.03 final loss %.3f / %.3f"
                + " / %.3f, peak %.2f / %.2f / %.2f; lr 0.1 diverged %d / %d / %d of 6%n", s.meanFinalLoss(),
                t.meanFinalLoss(), a.meanFinalLoss(), s.meanPeakLoss(), t.meanPeakLoss(), a.meanPeakLoss(),
                sHigh.diverged(), tHigh.diverged(), aHigh.diverged());
        assertTrue(s.diverged() == 0 && s.meanFinalLoss() < a.meanFinalLoss(), "update clipping absorbs the jump");
        assertTrue(t.diverged() == 0 && Double.isFinite(t.meanFinalLoss()), "atan2 trains without epsilon");
    }
}
