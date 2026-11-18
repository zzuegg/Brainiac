package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.diffusion.AutoencoderKL;
import io.github.zzuegg.brainiac.nn.diffusion.ClipTextEncoder;
import io.github.zzuegg.brainiac.nn.diffusion.ImageAttention;
import io.github.zzuegg.brainiac.nn.diffusion.ResnetBlock;
import io.github.zzuegg.brainiac.nn.diffusion.SpatialTransformer;
import io.github.zzuegg.brainiac.nn.diffusion.TimestepEmbedding;
import io.github.zzuegg.brainiac.nn.diffusion.UNet;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.nio.file.Path;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The diffusion layers save and load like the built-in ones. */
class DiffusionCodecsTest {

    @TempDir
    Path dir;

    /** Writes the layer's settings, reads a new one, copies the parameters and compares outputs. */
    private static <L extends Layer> void roundTrip(L layer, Function<Layer, Var> run) {
        Registry.LayerType<?> type = Registry.layerType(layer.getClass());
        assertNotNull(type, layer.getClass() + " is registered");
        Config c = type.write(layer);
        Layer copy = Registry.layerType(type.name()).codec().read(c);
        List<Var> from = layer.parameters();
        List<Var> to = copy.parameters();
        assertEquals(from.size(), to.size());
        for (int i = 0; i < from.size(); i++) {
            to.get(i).assign(from.get(i).value());
        }
        assertArrayEquals(run.apply(layer).value().toArray(), run.apply(copy).value().toArray(), 1e-6f);
    }

    @Test
    void everyDiffusionLayerRoundTrips() {
        SplittableRandom r = new SplittableRandom(1);
        Tensor image = Tensor.uniform(Shape.of(2, 8, 8, 8), -1, 1, r);
        Tensor time = Tensor.of(Shape.of(2), new float[] {5, 500}, Device.cpu());
        Tensor ctx = Tensor.uniform(Shape.of(2, 3, 6), -1, 1, r);
        roundTrip(new TimestepEmbedding(TimestepEmbedding.Settings.of(8, 12), r), l -> l.forward(Var.of(time)));
        roundTrip(new ResnetBlock(new ResnetBlock.Settings(8, 16, 12, 4, 1e-5f, 0), r),
                l -> ((ResnetBlock) l).forward(List.of(Var.of(image), Var.of(Tensor.uniform(Shape.of(2, 12), -1, 1,
                        new SplittableRandom(2)))), false));
        roundTrip(new SpatialTransformer(new SpatialTransformer.Settings(8, 2, 2, 6, 4, true), r),
                l -> ((SpatialTransformer) l).forward(List.of(Var.of(image), Var.of(ctx)), false));
        roundTrip(new ImageAttention(new ImageAttention.Settings(8, 1, 4, 1e-6f), r), l -> l.forward(Var.of(image)));
        roundTrip(new UNet(UNet.Settings.of(8, 4, new int[] {8, 16}, 6), r),
                l -> ((UNet) l).forward(List.of(Var.of(image), Var.of(time), Var.of(ctx)), false));
        roundTrip(new AutoencoderKL(AutoencoderKL.Settings.of(8, 4, 8, 16), r), l -> l.forward(Var.of(image)));
        Tensor ids = Tensor.of(Shape.of(1, 4), new float[] {2, 9, 4, 0}, Device.cpu());
        roundTrip(new ClipTextEncoder(new ClipTextEncoder.Settings(10, 8, 2, 2, 16, 6, Activation.QUICK_GELU, 1e-5f,
                -1, 0), r), l -> l.forward(Var.of(ids)));
    }

    @Test
    void modelsWithDiffusionLayersSaveAndLoad() throws Exception {
        SplittableRandom r = new SplittableRandom(3);
        Model m = Model.sequential().input(3, 8, 8)
                .layer(new AutoencoderKL(AutoencoderKL.Settings.of(3, 2, 8, 16), r))
                .build();
        Path file = dir.resolve("vae.brainiac");
        m.save(file);
        Model loaded = Model.load(file);
        Tensor in = Tensor.uniform(Shape.of(1, 3, 8, 8), -1, 1, r);
        assertArrayEquals(m.predict(in).toArray(), loaded.predict(in).toArray(), 1e-6f);
    }
}
