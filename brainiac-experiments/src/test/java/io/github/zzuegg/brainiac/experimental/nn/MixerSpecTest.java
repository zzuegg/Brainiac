package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/** Every experimental sequence mixer drops into {@code Model.sequential()} like the core layers. */
class MixerSpecTest {

    @Test
    void mixersBuildInsideSequentialModels() {
        List<LayerSpec> specs = List.of(
                DifferentialAttention.of(2, true, 1),
                GatedAttention.of(2, true, 10000f),
                ForgettingAttention.of(2, true),
                NormalizedTransformerBlock.of(2, 16, true, 0.1f),
                NativeSparseAttention.of(2, 2, 2, 2));
        for (LayerSpec spec : specs) {
            Model model = Model.sequential().input(6, 8).add(spec).dense(3).seed(1).build();
            Tensor y = model.predict(Tensor.uniform(Shape.of(2, 6, 8), -1, 1, new SplittableRandom(2)));
            assertEquals(Shape.of(2, 6, 3), y.shape(), spec.toString());
        }
        Model hyper = Model.sequential().input(6, 2, 8).add(ManifoldHyperConnection.of(GatedAttention.of(2, true, 0f)))
                .seed(1).build();
        assertEquals(Shape.of(2, 6, 2, 8),
                hyper.predict(Tensor.uniform(Shape.of(2, 6, 2, 8), -1, 1, new SplittableRandom(3))).shape());
    }
}
