package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.SafeTensors;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.nn.diffusion.FlowSampler;
import io.github.zzuegg.brainiac.nn.diffusion.SigmaSampler;
import io.github.zzuegg.brainiac.nn.diffusion.T5Encoder;
import io.github.zzuegg.brainiac.nn.diffusion.WanTransformer;
import io.github.zzuegg.brainiac.nn.diffusion.WanVae;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.UnaryOperator;

import static io.github.zzuegg.brainiac.diffusion.ReferenceTest.assertClose;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Compares the Wan parts against outputs that diffusers and transformers computed for tiny
 * random models with the same weights (made by {@code src/test/python/wan_fixtures.py}).
 */
class WanReferenceTest {

    private static Path dir;
    private static SafeTensors reference;
    private static final Device DEVICE = TestDevice.get();

    @BeforeAll
    static void open() throws Exception {
        dir = Path.of(WanReferenceTest.class.getResource("/tiny/wan_reference.safetensors").toURI()).getParent();
        reference = SafeTensors.open(dir.resolve("wan_reference.safetensors"));
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

    private static WanTransformer transformer(String folder) throws Exception {
        WeightSource s = SafeTensors.open(dir.resolve(folder).resolve("diffusion_pytorch_model.safetensors"));
        UnaryOperator<String> names = WanPretrained.transformerNames(s);
        return WanPretrained.transformer(new Weights(s, "", DEVICE, DType.FLOAT32).renamed(names),
                WanPretrained.transformerSettings(s, names, config(folder)));
    }

    @Test
    void umt5MatchesTransformersOnTheUnpaddedTokens() throws Exception {
        WeightSource s = SafeTensors.open(dir.resolve("wan/text_encoder/model.safetensors"));
        T5Encoder.Settings settings = WanPretrained.t5Settings(s, config("wan/text_encoder"));
        assertTrue(settings.biasPerLayer());
        T5Encoder t5 = WanPretrained.t5(new Weights(s, "", DEVICE, DType.FLOAT32), settings, false);
        float[] ids = ref("t5.ids").toArray();
        float[] mask = ref("t5.mask").toArray();
        float[] last = ref("t5.last").toArray();
        int n = 12;
        int d = 32;
        for (int b = 0; b < 2; b++) {
            int len = 0;
            while (len < n && mask[b * n + len] > 0) {
                len++;
            }
            float[] mine = new float[len];
            for (int i = 0; i < len; i++) {
                mine[i] = ids[b * n + i];
            }
            Tensor out = Var.noGrad(() -> t5.encode(Var.of(Tensor.of(Shape.of(1, mine.length), mine, DEVICE)), false)
                    .value());
            float[] expected = new float[len * d];
            System.arraycopy(last, b * n * d, expected, 0, len * d);
            assertClose(Tensor.of(Shape.of(1, len, d), expected), out, 1e-4);
        }
    }

    @Test
    void transformerMatchesDiffusers() throws Exception {
        WanTransformer t = transformer("wan/transformer");
        assertArrayEquals(new int[] {4, 4, 4}, t.settings().axes());
        Tensor out = Var.noGrad(() -> t.forward(Var.of(ref("dit.x")), ref("dit.t"), Var.of(ref("dit.ctx")), null,
                false).value());
        assertClose(ref("dit.out"), out, 1e-4);
        // a timestep per patch, as Wan 2.2 keeps its first frame clean
        Tensor perPatch = Var.noGrad(() -> t.forward(Var.of(ref("dit.x")), ref("dit.tp"), Var.of(ref("dit.ctx")),
                null, false).value());
        assertClose(ref("dit.out_tp"), perPatch, 1e-4);
        // the same with gradients
        assertClose(ref("dit.out"), t.forward(Var.of(ref("dit.x")), ref("dit.t"), Var.of(ref("dit.ctx")), null,
                false).value(), 1e-4);
    }

    @Test
    void imageToVideoTransformerMatchesDiffusers() throws Exception {
        WanTransformer t = transformer("wan_i2v_transformer");
        assertEquals(20, t.settings().imageFeatures());
        assertEquals(9, t.settings().inChannels());
        Tensor out = Var.noGrad(() -> t.forward(Var.of(ref("i2v.x")),
                Tensor.of(Shape.of(1), new float[] {640}, DEVICE), Var.of(ref("i2v.ctx")), Var.of(ref("i2v.img")),
                false).value());
        assertClose(ref("i2v.out"), out, 1e-4);
    }

    private static WanVae vae(String file, boolean encoder) throws Exception {
        WeightSource s = SafeTensors.open(dir.resolve(file));
        boolean diffusers = WanPretrained.diffusersVae(s);
        String prefix = diffusers ? "" : WanPretrained.vaePrefix(s);
        Map<String, Object> c = diffusers ? config("wan/vae") : config("wan/vae");
        return WanPretrained.vae(new Weights(s, prefix, DEVICE, DType.FLOAT32),
                WanPretrained.vaeSettings(s, prefix, c), diffusers, encoder);
    }

    @Test
    void vaeEncodesAndDecodesAsDiffusers() throws Exception {
        WanVae v = vae("wan/vae/diffusion_pytorch_model.safetensors", true);
        assertClose(ref("vae.latents"), v.encode(ref("vae.video")), 1e-4);
        assertClose(ref("vae.decoded"), v.decode(ref("vae.z")), 1e-4);
        // streamed: 1 frame, then 4 per latent frame
        List<Integer> counts = new ArrayList<>();
        v.decode(ref("vae.z"), f -> {
            counts.add(f.shape().dim(0));
            f.release();
        });
        assertEquals(List.of(1, 4, 4), counts);
    }

    @Test
    void originalLayoutLoadsTheSameNetworks() throws Exception {
        WanVae v = vae("wan_single/wan_vae_tiny.safetensors", false);
        assertClose(ref("vae.decoded"), v.decode(ref("vae.z")), 1e-4);
        WeightSource s = SafeTensors.open(dir.resolve("wan_single/wan_t2v_tiny.safetensors"));
        UnaryOperator<String> names = WanPretrained.transformerNames(s);
        // without a config, heads come from the transformer config's absence: give the tiny one's
        WanTransformer t = WanPretrained.transformer(new Weights(s, "", DEVICE, DType.FLOAT32).renamed(names),
                WanPretrained.transformerSettings(s, names, Map.of("num_attention_heads", 2)));
        Tensor out = Var.noGrad(() -> t.forward(Var.of(ref("dit.x")), ref("dit.t"), Var.of(ref("dit.ctx")), null,
                false).value());
        assertClose(ref("dit.out"), out, 1e-4);
    }

    @Test
    void filesAreToldApartByTheirWeights() throws Exception {
        Wan.Files f = Wan.Files.find(dir.resolve("wan_single"));
        assertEquals("wan_t2v_tiny.safetensors", f.transformers().get(0).getFileName().toString());
        assertEquals("wan_vae_tiny.safetensors", f.vae().getFileName().toString());
        assertEquals("umt5_tiny.safetensors", f.textEncoder().getFileName().toString());
        assertEquals("tokenizer.json", f.tokenizer().getFileName().toString());
        Wan.Files d = Wan.Files.find(dir.resolve("wan"));
        assertEquals(1, d.transformers().size());
        assertEquals(3.0, ((Number) d.scheduler().get("flow_shift")).doubleValue());
        assertTrue(Wan.detect(dir.resolve("wan")));
        assertTrue(!Wan.detect(dir.resolve("flux_dev")));
    }

    @Test
    void uniPcMatchesDiffusers() {
        double[] sigmas = FlowSampler.uniPcSigmas(6, 3.0);
        float[] expected = ref("unipc.sigmas").toArray();
        float[] t = ref("unipc.timesteps").toArray();
        for (int i = 0; i < sigmas.length; i++) {
            assertEquals(expected[i], sigmas[i], 1e-6, "sigma " + i);
            if (i < t.length) {
                assertEquals(t[i], Math.floor(sigmas[i] * 1000), 0, "timestep " + i);
            }
        }
        FlowSampler.Model model = (x, sigma, step) -> x.mul((float) (0.5 * sigma + 0.2)).add(0.1f);
        Tensor out = FlowSampler.sampleUniPc(model, ref("unipc.noise"), sigmas, SigmaSampler.Listener.NONE);
        assertClose(ref("unipc.out"), out, 1e-5);
    }

    @Test
    void wholePipelineMatchesDiffusers() throws Exception {
        try (Wan wan = Wan.load(dir.resolve("wan"), DEVICE, DType.FLOAT32, new Wan.Options(0, 0, 16))) {
            VideoRequest r = new VideoRequest("a red fox runs in snow", "blurry", 32, 32, 9, 3, 5f, 0f, 0);
            GeneratedVideo video = wan.generate(r, ref("pipe.noise"), Progress.NONE);
            assertEquals(9, video.frames().size());
            float[] e = ref("pipe.video").toArray();   // [1, frames, 3, h, w] in [0, 1]
            int plane = 32 * 32;
            double worst = 0;
            for (int f = 0; f < 9; f++) {
                GeneratedImage img = video.frames().get(f);
                for (int c = 0; c < 3; c++) {
                    for (int p = 0; p < plane; p++) {
                        int mine = img.rgb()[p * 3 + c] & 0xff;
                        worst = Math.max(worst, Math.abs(mine - e[(f * 3 + c) * plane + p] * 255));
                    }
                }
            }
            // 8-bit rounding of the same values
            assertTrue(worst <= 1.0, "largest difference " + worst + " of 255");
        }
    }

    @Test
    void singleFilesRunTheWholePipeline() throws Exception {
        Wan.Files f = Wan.Files.find(dir.resolve("wan_single"));
        // the tiny transformer's heads are not Wan's 128 features: give its config
        Wan.Files tiny = new Wan.Files(f.transformers(), f.vae(), f.textEncoder(), f.tokenizer(),
                Map.of("num_attention_heads", 2), config("wan/vae"), null, Map.of("flow_shift", 3.0), 0);
        try (Wan wan = Wan.load(tiny, DEVICE, DType.FLOAT32, new Wan.Options(0, 0, 16))) {
            VideoRequest r = new VideoRequest("a red fox runs in snow", "blurry", 32, 32, 9, 3, 5f, 0f, 0);
            GeneratedVideo video = wan.generate(r, ref("pipe.noise"), Progress.NONE);
            float[] e = ref("pipe.video").toArray();
            GeneratedImage last = video.frames().get(8);
            int plane = 32 * 32;
            for (int p = 0; p < plane; p += 37) {
                assertEquals(e[(8 * 3) * plane + p] * 255, last.rgb()[p * 3] & 0xff, 1.0);
            }
        }
    }

    @Test
    void videosWriteAsAnimatedPngAndAvi() {
        List<GeneratedImage> frames = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            byte[] rgb = new byte[4 * 2 * 3];
            rgb[0] = (byte) (i * 100);
            frames.add(new GeneratedImage(4, 2, rgb, 0));
        }
        GeneratedVideo v = new GeneratedVideo(frames, 16, 0);
        byte[] png = v.apng();
        String text = new String(png, java.nio.charset.StandardCharsets.ISO_8859_1);
        assertTrue(text.contains("acTL") && text.contains("fdAT") && text.endsWith("IEND®B`\u0082"));
        byte[] avi = v.avi();
        assertEquals("RIFF", new String(avi, 0, 4, java.nio.charset.StandardCharsets.US_ASCII));
        int riffSize = java.nio.ByteBuffer.wrap(avi, 4, 4).order(java.nio.ByteOrder.LITTLE_ENDIAN).getInt();
        assertEquals(avi.length - 8, riffSize);
        assertTrue(new String(avi, java.nio.charset.StandardCharsets.ISO_8859_1).contains("idx1"));
    }
}
