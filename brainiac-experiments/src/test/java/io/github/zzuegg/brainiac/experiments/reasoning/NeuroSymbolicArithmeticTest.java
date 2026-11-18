package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.reasoning.NeuroSymbolicArithmetic.Method;
import io.github.zzuegg.brainiac.experiments.reasoning.NeuroSymbolicArithmetic.Scores;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

/** Item 7.6.11: neural predicates trained from sums alone against an end-to-end MLP and the digit-label oracle. */
class NeuroSymbolicArithmeticTest {

    @Test
    void neuralPredicatesLearnDigitsFromSumsAlone() {
        Map<Method, Scores> all = new EnumMap<>(Method.class);
        StringBuilder line = new StringBuilder("Digit addition, sum acc / digit acc / 2-3-4 digit sums:");
        for (Method m : Method.values()) {
            Scores s = NeuroSymbolicArithmetic.run(m, 1000, 1);
            all.put(m, s);
            line.append(String.format(" | %s %.3f/%.3f/%.2f-%.2f-%.2f", m, s.sumAccuracy(), s.digitAccuracy(),
                    s.multiDigit()[0], s.multiDigit()[1], s.multiDigit()[2]));
        }
        System.out.println(line);
        assertTrue(all.get(Method.NEURAL_PREDICATE).sumAccuracy() > all.get(Method.END_TO_END).sumAccuracy(),
                "the neural predicate beats the end-to-end MLP on sums");
        assertTrue(all.get(Method.NEURAL_PREDICATE).digitAccuracy() > 0.6, "digits are learned from sums alone");
    }
}
