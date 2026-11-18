package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.MultiHeadAttention.Projection;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/** LoRA and other adapters on a transformer block's attention projections. */
class AttentionAdaptersTest {

    private static TransformerBlock block() {
        return (TransformerBlock) TransformerBlock.of(2).create(Shape.of(5, 4), new SplittableRandom(3));
    }

    private static Tensor x() {
        return Tensor.uniform(Shape.of(2, 5, 4), -1, 1, new SplittableRandom(4));
    }

    @Test
    void loraStartsUnchangedAndAddsItsParameters() {
        TransformerBlock b = block();
        float[] before = b.forward(Var.of(x())).value().toArray();
        int params = b.parameters().size();
        MultiHeadAttention attention = b.selfAttention();
        MultiHeadAttention.Adapter q = attention.lora(Projection.QUERY, 2, 2, new SplittableRandom(5));
        attention.lora(Projection.VALUE, 2, 2, new SplittableRandom(6));
        assertEquals(params + 4, b.parameters().size());
        assertArrayEquals(before, b.forward(Var.of(x())).value().toArray(), 0);
        assertEquals(Shape.of(4, 2), q.parameters().get(0).shape());
        assertSame(attention.weight(Projection.QUERY), attention.parameters().get(0));
    }

    @Test
    void anAdapterActsAsAChangedWeight() {
        TransformerBlock adapted = block();
        TransformerBlock edited = block();
        Tensor m = Tensor.uniform(Shape.of(4, 4), -0.5f, 0.5f, new SplittableRandom(7));
        for (Projection p : List.of(Projection.KEY, Projection.OUTPUT)) {
            adapted.selfAttention().adapt(p, new MultiHeadAttention.Adapter() {
                @Override
                public Var delta(Var in) {
                    return in.matmul(Var.of(m));
                }

                @Override
                public List<Var> parameters() {
                    return List.of();
                }
            });
            Var w = edited.selfAttention().weight(p);
            w.assign(w.value().add(m));
        }
        assertArrayEquals(edited.forward(Var.of(x())).value().toArray(), adapted.forward(Var.of(x())).value().toArray(), 1e-5f);
    }
}
