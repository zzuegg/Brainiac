package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.SplittableRandom;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/** Stepping a recurrent layer one input at a time gives the states its sequence run gives. */
class RecurrentStepTest {

    static Stream<Recurrent.Spec> cells() {
        return Stream.of(Recurrent.rnn(5), Recurrent.lstm(5), Recurrent.gru(5), Recurrent.gru(5).resetAfter(false),
                Recurrent.lstm(5).learnInitialState(true));
    }

    @ParameterizedTest
    @MethodSource("cells")
    void stepsMatchTheSequence(Recurrent.Spec spec) {
        Recurrent layer = (Recurrent) spec.returnSequences(true).create(Shape.of(4, 3), new SplittableRandom(2));
        Tensor x = Tensor.uniform(Shape.of(2, 4, 3), -1, 1, new SplittableRandom(3));
        float[] all = layer.forward(Var.of(x)).value().toArray();
        Recurrent.State state = layer.initialState(2);
        Var input = Var.of(x);
        for (int t = 0; t < 4; t++) {
            state = layer.step(input.slice(1, t, t + 1).reshape(2, 3), state);
            float[] h = state.h().value().toArray();
            float[] want = new float[10];
            for (int b = 0; b < 2; b++) {
                System.arraycopy(all, (b * 4 + t) * 5, want, b * 5, 5);
            }
            assertArrayEquals(want, h, 1e-5f, "step " + t);
        }
    }
}
