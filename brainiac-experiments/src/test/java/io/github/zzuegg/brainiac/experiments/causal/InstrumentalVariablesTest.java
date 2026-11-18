package io.github.zzuegg.brainiac.experiments.causal;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class InstrumentalVariablesTest {

    @Test
    void deepIvBeatsTheConfoundedRegressionUnderIntervention() {
        Map<String, double[]> r = InstrumentalVariables.run(3000, 0.5, 600, 1);
        StringBuilder line = new StringBuilder("airline demand obs MSE / counterfactual rel. MSE:");
        r.forEach((k, v) -> line.append(String.format(" %s %.1f/%.3f |", k, v[0], v[1])));
        System.out.println(line);
        assertTrue(r.get("DeepIV")[1] < 1, line.toString());
        assertTrue(r.get("DeepIV")[1] < r.get("linear 2SLS")[1], line.toString());
    }
}
