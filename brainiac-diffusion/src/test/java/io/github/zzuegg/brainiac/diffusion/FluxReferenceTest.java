package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.SafeTensors;
import io.github.zzuegg.brainiac.nn.diffusion.FluxTransformer;
import io.github.zzuegg.brainiac.nn.diffusion.SigmaSampler;
import io.github.zzuegg.brainiac.nn.diffusion.T5Encoder;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static io.github.zzuegg.brainiac.diffusion.ReferenceTest.assertClose;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Compares the Flux parts against outputs that diffusers and transformers computed for tiny
 * random models with the same weights (made by {@code src/test/python/flux_fixtures.py}).
 */
class FluxReferenceTest {

    private static Path dir;
    private static SafeTensors reference;
    private static final Device DEVICE = TestDevice.get();

    @BeforeAll
    static void open() throws Exception {
        dir = Path.of(FluxReferenceTest.class.getResource("/tiny/flux_reference.safetensors").toURI()).getParent();
        reference = SafeTensors.open(dir.resolve("flux_reference.safetensors"));
    }

    @AfterAll
    static void close() {
        reference.close();
    }

    static Tensor ref(String name) {
        return reference.tensor(name, DEVICE, DType.FLOAT32);
    }

    private static Map<String, Object> config(String folder) throws Exception {
        return Json.object(Json.read(dir.resolve(folder).resolve("config.json")));
    }

    private static Weights weights(String folder, String file) {
        return new Weights(SafeTensors.open(dir.resolve(folder).resolve(file)), "", DEVICE, DType.FLOAT32);
    }

    @Test
    void t5Encoder() throws Exception {
        T5Encoder t5 = Pretrained.t5(weights("flux_dev/text_encoder_2", "model.safetensors"),
                Pretrained.t5Settings(config("flux_dev/text_encoder_2")));
        Tensor out = Var.noGrad(() -> t5.encode(Var.of(ref("t5.ids")), false).value());
        assertClose(ref("t5.last"), out, 1e-4);
        // the same with gradients, where the position biases take the other path
        assertClose(ref("t5.last"), t5.encode(Var.of(ref("t5.ids")), false).value(), 1e-4);
    }

    @Test
    void fluxTransformer() throws Exception {
        for (String name : new String[] {"dev", "schnell"}) {
            FluxTransformer.Settings s = Pretrained.fluxSettings(config("flux_" + name + "/transformer"));
            FluxTransformer f = Pretrained.flux(weights("flux_" + name + "/transformer",
                    "diffusion_pytorch_model.safetensors"), s);
            Tensor g = s.guidance() ? ref("dit.guidance") : null;
            Tensor out = Var.noGrad(() -> f.forward(Var.of(ref("dit.x")), ref("dit.sigma"), Var.of(ref("dit.ctx")),
                    Var.of(ref("dit.pooled")), g, false).value());
            assertClose(ref("dit." + name), out, 1e-4);
            Tensor withGrad = f.forward(Var.of(ref("dit.x")), ref("dit.sigma"), Var.of(ref("dit.ctx")),
                    Var.of(ref("dit.pooled")), g, false).value();
            assertClose(ref("dit." + name), withGrad, 1e-4);
        }
    }

    @Test
    void t5TokenizerMatchesTransformers() throws Exception {
        try (Flux flux = Flux.load(dir.resolve("flux_schnell"), DEVICE, DType.FLOAT32, new Flux.Options(0, 0, 16))) {
            for (Object o : Json.array(Json.read(dir.resolve("t5_tokenizer_cases.json")))) {
                Map<String, Object> c = Json.object(o);
                List<?> expected = (List<?>) c.get("ids");
                int[] e = expected.stream().mapToInt(x -> ((Number) x).intValue()).toArray();
                assertArrayEquals(e, flux.t5Ids((String) c.get("prompt")), (String) c.get("prompt"));
            }
        }
    }

    @Test
    void wholePipelinesMatchDiffusers() throws Exception {
        for (String name : new String[] {"dev", "schnell"}) {
            try (Flux flux = Flux.load(dir.resolve("flux_" + name), DEVICE, DType.FLOAT32,
                    new Flux.Options(0, 0, 16))) {
                assertEquals(name.equals("dev") ? Flux.Kind.DEV : Flux.Kind.SCHNELL, flux.kind());
                Flux.Conditioning cond = flux.encode("hello the cat");
                Tensor latents = flux.denoise(ref("pipe.noise"), cond, name.equals("dev") ? 4 : 3, 3.5f,
                        SigmaSampler.Listener.NONE);
                Tensor image = flux.decode(latents);
                assertClose(ref("pipe." + name), image, 2e-4);
            }
        }
    }

    @Test
    void quantizedWeightsStayClose() throws Exception {
        for (int bits : new int[] {8, 4}) {
            try (Flux flux = Flux.load(dir.resolve("flux_schnell"), DEVICE, DType.FLOAT32,
                    new Flux.Options(bits, bits, 16))) {
                assertTrue(flux.transformer().jointBlocks().get(0).image().q().weight().value().isPacked());
                Flux.Conditioning cond = flux.encode("hello the cat");
                Tensor image = flux.decode(flux.denoise(ref("pipe.noise"), cond, 3, 0, SigmaSampler.Listener.NONE));
                // 8 bits stay within a few levels of the full-precision image; 4 bits drift more
                assertClose(ref("pipe.schnell"), image, bits == 8 ? 0.02 : 0.15);
            }
        }
    }

    @Test
    void loadPicksTypeAndWeightsForTheDevice() throws Exception {
        Path schnell = dir.resolve("flux_schnell");
        assertEquals(DType.FLOAT32, ImageGenerator.dtype(schnell, Device.cpu()));
        assertEquals(Flux.Options.full(), Flux.Options.fitting(schnell, Device.cpu(), DType.FLOAT32));
        MemoryFit.Params t = MemoryFit.params(schnell.resolve("transformer"), "diffusion_pytorch_model", false);
        assertTrue(t.matrices() > 0 && t.other() > 0);
        assertTrue(t.bytes(DType.BFLOAT16, 4) < t.bytes(DType.BFLOAT16, 8));
        assertTrue(t.bytes(DType.BFLOAT16, 8) < t.bytes(DType.BFLOAT16, 0));
        // quantized projections still compute in the load type
        if (DEVICE.supports(DType.BFLOAT16)) {
            try (Flux q = Flux.load(schnell, DEVICE, DType.BFLOAT16, new Flux.Options(8, 8, 16))) {
                assertEquals(DType.BFLOAT16, q.transformer().dtype());
                // and its activations stay in it: no quantized layer lifts them to float32
                Tensor out = Var.noGrad(() -> q.transformer().forward(Var.of(ref("dit.x").to(DType.BFLOAT16)),
                        ref("dit.sigma"), Var.of(ref("dit.ctx").to(DType.BFLOAT16)),
                        Var.of(ref("dit.pooled").to(DType.BFLOAT16)), null, false).value());
                assertEquals(DType.BFLOAT16, out.dtype());
            }
        }
        // tiny models fit any device, so the most precise choice wins
        try (ImageGenerator g = ImageGenerator.load(schnell, DEVICE, DType.FLOAT32)) {
            assertEquals("FLUX_SCHNELL", g.family());
            assertEquals(0, ((Flux) g).options().transformerBits());
        }
    }
}
