package io.github.zzuegg.brainiac.experiments.numerics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class GradientCompressionShootoutTest {

    @Test
    void compressedGradientsTrainWithAFractionOfTheBytes() {
        Map<String, GradientCompressionShootout.Outcome> r = GradientCompressionShootout.run(150, 1);
        StringBuilder line = new StringBuilder("Spiral, 4 workers, 150 steps, loss/accuracy/kB:");
        r.forEach((k, v) -> line.append(String.format(" %s %.3f/%.2f/%d;", k, v.loss(), v.accuracy(), v.bytes() / 1000)));
        System.out.println(line);
        GradientCompressionShootout.Outcome dp = r.get("fp32 all-reduce");
        GradientCompressionShootout.Outcome power = r.get("PowerSGD r=1");
        assertTrue(dp.accuracy() > 0.85, line.toString());
        assertTrue(power.bytes() * 5 < dp.bytes(), line.toString());
        assertTrue(power.accuracy() > dp.accuracy() - 0.1, line.toString());
        assertTrue(r.get("EF21 1%").bytes() * 5 < dp.bytes(), line.toString());
        assertTrue(r.get("QSGD s=4").accuracy() > dp.accuracy() - 0.1, line.toString());
    }

    @Test
    void errorFeedbackHelpsSparsificationUnderPlainSgd() {
        Map<String, GradientCompressionShootout.Outcome> r = GradientCompressionShootout.run(150, 1, true,
                List.of("fp32 all-reduce", "top-1% no EF", "top-1% EF", "DGC 1%", "EF21 1%"));
        StringBuilder line = new StringBuilder("Spiral, 4 workers, 150 steps, plain SGD outer, loss/accuracy/kB:");
        r.forEach((k, v) -> line.append(String.format(" %s %.3f/%.2f/%d;", k, v.loss(), v.accuracy(), v.bytes() / 1000)));
        System.out.println(line);
        assertTrue(r.get("top-1% EF").loss() < r.get("top-1% no EF").loss(), line.toString());
    }

    @Test
    void quantizerVarianceShrinksWithMoreLevels() {
        Map<String, Double> v = GradientCompressionShootout.quantizerVariance(1);
        StringBuilder line = new StringBuilder("Quantizer relative variance E|Q(g)-g|^2/|g|^2:");
        v.forEach((k, x) -> line.append(String.format(" %s %.3f;", k, x)));
        System.out.println(line);
        assertTrue(v.get("QSGD s=16") < v.get("QSGD s=4"), line.toString());
        assertTrue(v.get("QSGD s=4") < v.get("QSGD s=1"), line.toString());
    }
}
