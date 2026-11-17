package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.BitFit;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class PeftLayersTest {

    private static final SplittableRandom R = new SplittableRandom(1);

    private static float[] out(Layer l, Tensor x) {
        return l.forward(Var.of(x)).value().toArray();
    }

    @Test
    void everyAdapterStartsAsTheFrozenLayer() {
        Dense base = new Dense(8, 6, Activation.TANH, Init.XAVIER, R);
        base.bias().assign(Tensor.normal(Shape.of(6), 0, 0.3f, R));
        Tensor x = Tensor.normal(Shape.of(5, 8), 0, 1, R);
        float[] want = out(base, x);
        assertArrayEquals(want, out(LoraAdapterDense.lora(base, 2, 4, R), x), 1e-5f);
        assertArrayEquals(want, out(LoraAdapterDense.pissa(base, 2), x), 1e-4f);
        assertArrayEquals(want, out(new VeraDense.Shared(8, 6, 16, 3).adapt(base), x), 1e-5f);
        assertArrayEquals(want, out(new Ia3Scaling(base, 6), x), 1e-6f);
        Layer adapter = new HoulsbyAdapter(6, 2, Activation.GELU, R);
        assertArrayEquals(want, out(adapter, base.forward(Var.of(x)).value()), 1e-6f);
        assertEquals(2, new VeraDense.Shared(8, 6, 16, 3).adapt(base).parameters().size());
    }

    @Test
    void pissaHoldsThePrincipalDirectionsAndMergesBack() {
        Dense base = new Dense(10, 7, Activation.IDENTITY, Init.XAVIER, R);
        LoraAdapterDense p = LoraAdapterDense.pissa(base, 3);
        float normAB = p.a().matmul(p.b()).value().square().sum().item();
        float normW = base.weight().value().square().sum().item();
        assertTrue(normAB > 0.5f * normW, "rank 3 of 7 holds most of the energy: " + normAB / normW);
        p.b().assign(p.b().value().add(0.1f));
        Tensor x = Tensor.normal(Shape.of(4, 10), 0, 1, R);
        assertArrayEquals(out(p, x), out(p.merge(), x), 1e-4f);
    }

    @Test
    void adaptersLearnInsideASequentialModel() {
        Model model = Model.sequential().input(4)
                .layer(new Dense(4, 8, Activation.TANH, Init.XAVIER, R))
                .add(HoulsbyAdapter.spec(2, Activation.RELU))
                .build();
        assertEquals(8, model.layers().get(1).outputSize(8));
        assertTrue(BitFit.biases(model.parameters()).size() >= 3);
    }
}
