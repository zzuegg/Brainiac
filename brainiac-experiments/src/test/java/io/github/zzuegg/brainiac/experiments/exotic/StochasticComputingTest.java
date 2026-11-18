package io.github.zzuegg.brainiac.experiments.exotic;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.StochasticBitstreamDense;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

/** Wave 6, 6.4: stochastic-computing inference against quantized formats. */
class StochasticComputingTest {

    @Test
    void xnorStreamsMultiplyAndCountersAdd() {
        Tensor w = Tensor.of(Shape.of(2, 1), new float[] {0.5f, -0.25f});
        StochasticBitstreamDense sc = StochasticBitstreamDense.of(w, null, 1 << 16,
                StochasticBitstreamDense.Adder.APC, 1);
        float y = sc.forward(Var.of(Tensor.of(Shape.of(1, 2), new float[] {0.8f, 0.4f}))).value().item();
        assertEquals(0.5 * 0.8 - 0.25 * 0.4, y, 0.02);
    }

    @Test
    void longerStreamsBuyAccuracy() {
        StochasticComputing e = StochasticComputing.train(400, 1);
        StochasticComputing.Result f = e.floating();
        StochasticComputing.Result apc16 = e.stochastic(16, StochasticBitstreamDense.Adder.APC, 1);
        StochasticComputing.Result apc256 = e.stochastic(256, StochasticBitstreamDense.Adder.APC, 1);
        StochasticComputing.Result apc1024 = e.stochastic(1024, StochasticBitstreamDense.Adder.APC, 1);
        StochasticComputing.Result mux256 = e.stochastic(256, StochasticBitstreamDense.Adder.MUX, 1);
        StochasticComputing.Result mux4096 = e.stochastic(4096, StochasticBitstreamDense.Adder.MUX, 1);
        StochasticComputing.Result int8 = e.int8();
        StochasticComputing.Result tern = e.ternary();
        StochasticComputing.Result bit = e.bitLinear(400, 1);
        System.out.printf("SC digits acc: float %.2f; APC L=16/256/1024 %.2f/%.2f/%.2f (%.0fk/%.0fk/%.0fk bit-ops);"
                + " MUX L=256/4096 %.2f/%.2f; int8 PTQ %.2f, ternary PTQ %.2f, BitLinear QAT %.2f%n", f.accuracy(),
                apc16.accuracy(), apc256.accuracy(), apc1024.accuracy(), apc16.bitOpsPerSample() / 1e3,
                apc256.bitOpsPerSample() / 1e3, apc1024.bitOpsPerSample() / 1e3, mux256.accuracy(),
                mux4096.accuracy(), int8.accuracy(), tern.accuracy(), bit.accuracy());
        assertTrue(f.accuracy() > 0.8, "float " + f.accuracy());
        assertTrue(apc1024.accuracy() > f.accuracy() - 0.1, "long APC streams approach float");
        assertTrue(apc1024.accuracy() > apc16.accuracy(), "stream length buys accuracy");
        assertTrue(mux4096.accuracy() > mux256.accuracy() - 0.02, "so it does for the mux");
    }
}
