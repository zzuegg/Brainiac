package io.github.zzuegg.brainiac.experiments.numerics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class NaturalGradientDiagnosticsTest {

    @Test
    void exactNaturalGradientBeatsGradientDescentAndKfacAlignsBetterThanAdam() {
        int[] sizes = {2, 8, 3};
        Map<String, Double> loss = NaturalGradientDiagnostics.losses(sizes, 15, 1);
        StringBuilder s = new StringBuilder("Spirals 2-8-3 (51 params), loss after 15 steps:");
        loss.forEach((k, v) -> s.append(String.format(" %s %.3f;", k, v)));
        Map<String, Double> cos = NaturalGradientDiagnostics.alignment(sizes, 1);
        s.append(" cosine to the exact natural-gradient step:");
        cos.forEach((k, v) -> s.append(String.format(" %s %.2f;", k, v)));
        System.out.println(s);
        assertTrue(loss.get("natural gradient (Fisher)") < loss.get("gradient descent"), s.toString());
        assertTrue(cos.get("K-FAC") > cos.get("Adam (sqrt EF diagonal)"), s.toString());
        assertTrue(cos.get("K-FAC") > 0.3, s.toString());
    }

    @Test
    void logisticRegressionNaturalGradientConvergesInFewSteps() {
        Map<String, Double> loss = NaturalGradientDiagnostics.losses(new int[] {2, 3}, 10, 2);
        StringBuilder s = new StringBuilder("Spirals logistic regression (9 params), loss after 10 steps:");
        loss.forEach((k, v) -> s.append(String.format(" %s %.4f;", k, v)));
        System.out.println(s);
        assertTrue(loss.get("natural gradient (Fisher)") <= loss.get("gradient descent") + 1e-3, s.toString());
    }
}
