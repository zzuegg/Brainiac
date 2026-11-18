package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.reasoning.ArithmeticExtrapolation.BitModel;
import io.github.zzuegg.brainiac.experiments.reasoning.ArithmeticExtrapolation.Operation;

import org.junit.jupiter.api.Test;

/** Item 7.6.10: the Neural GPU against a transformer on binary addition, trained short and tested long. */
class NeuralGpuLengthTest {

    @Test
    void neuralGpuAddsLongerNumbersThanItSaw() {
        int[] lengths = {6, 12, 24};
        double[] gpu = ArithmeticExtrapolation.binaryLength(BitModel.NEURAL_GPU, Operation.ADD, 6, 400, lengths, 1);
        double[] tf = ArithmeticExtrapolation.binaryLength(BitModel.TRANSFORMER, Operation.ADD, 6, 400, lengths, 1);
        System.out.printf("Binary addition trained on 1-6 bits, exact accuracy at 6/12/24 bits: Neural GPU %.2f/%.2f/%.2f,"
                + " transformer %.2f/%.2f/%.2f%n", gpu[0], gpu[1], gpu[2], tf[0], tf[1], tf[2]);
        assertTrue(gpu[1] > tf[1], "the Neural GPU generalises to 12 bits better than the transformer");
    }
}
