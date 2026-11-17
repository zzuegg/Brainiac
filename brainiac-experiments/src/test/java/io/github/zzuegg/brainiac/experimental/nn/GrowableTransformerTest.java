package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GrowableTransformerBlock.Config;
import io.github.zzuegg.brainiac.experimental.nn.GrowableTransformerBlock.NormPlacement;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.api.Test;

class GrowableTransformerTest {

    private static Var ids() {
        float[] v = new float[2 * 6];
        for (int i = 0; i < v.length; i++) {
            v[i] = (i * 7) % 10;
        }
        return Var.of(Tensor.of(Shape.of(2, 6), v));
    }

    private static Config config(NormPlacement norm) {
        return new Config(8, 2, 1, 4, true, norm, false,
                List.of(Activation.GELU, Activation.TANH, Activation.RELU, Activation.GELU, Activation.SILU,
                        Activation.TANH));
    }

    @ParameterizedTest
    @EnumSource(NormPlacement.class)
    void widthGrowthPreservesTheFunction(NormPlacement norm) {
        SplittableRandom r = new SplittableRandom(1);
        GrowableTransformer t = GrowableTransformer.of(10, 8, config(norm), 2, r);
        float[] before = t.forward(ids()).value().toArray();
        t.widenFfn(0, 11, r);
        t.addHeadGroup(1, r);
        t.addHeadGroup(1, r);
        float[] after = t.forward(ids()).value().toArray();
        assertArrayEquals(before, after, 2e-4f);
        assertEquals(11, t.blocks().getFirst().config().ffn());
        assertEquals(6, t.blocks().get(1).config().heads());
        assertEquals(3, t.blocks().get(1).config().kvHeads());
    }

    @Test
    void depthGrowthPreservesTheFunctionAndStackingDoublesDepth() {
        SplittableRandom r = new SplittableRandom(2);
        GrowableTransformer t = GrowableTransformer.of(10, 8, config(NormPlacement.PRE), 1, r);
        float[] before = t.forward(ids()).value().toArray();
        t.addBlock(1, config(NormPlacement.SANDWICH), r);
        t.addBlock(0, config(NormPlacement.PRE), r);
        assertArrayEquals(before, t.forward(ids()).value().toArray(), 1e-5f);
        t.stack();
        assertEquals(6, t.blocks().size());
    }

    @Test
    void grownNeuronsReceiveDifferentGradients() {
        SplittableRandom r = new SplittableRandom(3);
        GrowableTransformer t = GrowableTransformer.of(10, 8, Config.of(8, 1, 8, 2, Activation.GELU), 1, r);
        t.widenFfn(0, 4, r);
        Var loss = GrowableTransformer.loss(t.forward(ids()), Tensor.of(Shape.of(2, 6), new float[12]));
        loss.backward();
        for (Var p : t.parameters()) {
            assertNotNull(p.grad());
        }
        double[] pressure = t.blocks().getFirst().growthPressure();
        assertTrue(pressure[0] > 0 && pressure[1] > 0);
    }

    @Test
    void blockAndModelAreLayerSpecs() {
        Model m = Model.sequential().input(6).add(GrowableTransformer.spec(10, config(NormPlacement.PRE), 1))
                .seed(1).build();
        assertEquals(Shape.of(2, 6, 10), m.forward(ids()).shape());
        Model b = Model.sequential().input(5, 8).add(GrowableTransformerBlock.spec(config(NormPlacement.POST)))
                .seed(1).build();
        assertEquals(Shape.of(3, 5, 8), b.forward(Var.of(Tensor.ones(Shape.of(3, 5, 8)))).shape());
    }
}
