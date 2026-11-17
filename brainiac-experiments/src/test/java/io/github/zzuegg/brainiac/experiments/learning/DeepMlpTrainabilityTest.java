package io.github.zzuegg.brainiac.experiments.learning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.learning.DeepMlpTrainability.Stack;

import org.junit.jupiter.api.Test;

class DeepMlpTrainabilityTest {

    @Test
    void highwayLayersTrainAThirtyLayerMlp() {
        float[] plain = DeepMlpTrainability.run(Stack.PLAIN, 30, 400, 1);
        float[] residual = DeepMlpTrainability.run(Stack.RESIDUAL, 30, 400, 1);
        float[] highway = DeepMlpTrainability.run(Stack.HIGHWAY, 30, 400, 1);
        System.out.printf("30-layer MLP on spirals, loss/accuracy: plain %.3f/%.3f, residual %.3f/%.3f, highway "
                + "%.3f/%.3f%n", plain[0], plain[1], residual[0], residual[1], highway[0], highway[1]);
        assertTrue(highway[1] > plain[1] + 0.1f, "highway trains where the plain stack does not");
        assertTrue(highway[1] > 0.7f, "highway fits the spirals");
    }
}
