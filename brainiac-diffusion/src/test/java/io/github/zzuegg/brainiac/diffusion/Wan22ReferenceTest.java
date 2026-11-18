package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.SafeTensors;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.nn.diffusion.WanVae;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static io.github.zzuegg.brainiac.diffusion.ReferenceTest.assertClose;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Compares Wan 2.2's 16x autoencoder (residual levels, 2x2 patches) and a TI2V-style pipeline
 * against diffusers on tiny random models (made by {@code src/test/python/wan22_fixtures.py}).
 */
class Wan22ReferenceTest {

    private static Path dir;
    private static SafeTensors reference;
    private static final Device DEVICE = TestDevice.get();

    @BeforeAll
    static void open() throws Exception {
        dir = Path.of(Wan22ReferenceTest.class.getResource("/tiny/wan22_reference.safetensors").toURI()).getParent();
        reference = SafeTensors.open(dir.resolve("wan22_reference.safetensors"));
    }

    @AfterAll
    static void close() {
        reference.close();
    }

    private static Tensor ref(String name) {
        return reference.tensor(name, DEVICE, DType.FLOAT32);
    }

    private static Map<String, Object> vaeConfig() throws Exception {
        return Json.object(Json.read(dir.resolve("wan22/vae/config.json")));
    }

    private static WanVae vae(String file, boolean encoder) throws Exception {
        WeightSource s = SafeTensors.open(dir.resolve(file));
        boolean diffusers = WanPretrained.diffusersVae(s);
        String prefix = diffusers ? "" : WanPretrained.vaePrefix(s);
        return WanPretrained.vae(new Weights(s, prefix, DEVICE, DType.FLOAT32),
                WanPretrained.vaeSettings(s, prefix, vaeConfig()), diffusers, encoder);
    }

    @Test
    void settingsComeFromTheConfig() throws Exception {
        WanVae.Settings s = vae("wan22/vae/diffusion_pytorch_model.safetensors", false).settings();
        assertTrue(s.residual());
        assertEquals(2, s.patch());
        assertEquals(12, s.decoderBase());
        assertEquals(16, s.spatialFactor());
        assertEquals(12, s.videoChannels());
        WanVae.Settings full = WanVae.Settings.wan22();
        assertEquals(48, full.latentChannels());
        assertEquals(16, full.spatialFactor());
    }

    @Test
    void vaeEncodesAndDecodesAsDiffusers() throws Exception {
        WanVae v = vae("wan22/vae/diffusion_pytorch_model.safetensors", true);
        assertClose(ref("vae.latents"), v.encode(ref("vae.video")), 1e-4);
        assertClose(ref("vae.decoded"), v.decode(ref("vae.z")), 1e-4);
        List<Integer> counts = new ArrayList<>();
        v.decode(ref("vae.z"), f -> {
            counts.add(f.shape().dim(0));
            assertEquals(64, f.shape().dim(2));
            f.release();
        });
        assertEquals(List.of(1, 4, 4), counts);
    }

    @Test
    void originalLayoutLoadsTheSameDecoder() throws Exception {
        WanVae v = vae("wan22_single/wan22_vae_tiny.safetensors", true);
        assertClose(ref("vae.decoded"), v.decode(ref("vae.z")), 1e-4);
        assertClose(ref("vae.latents"), v.encode(ref("vae.video")), 1e-4);
    }

    @Test
    void wholePipelineMatchesDiffusers() throws Exception {
        try (Wan wan = Wan.load(dir.resolve("wan22"), DEVICE, DType.FLOAT32, new Wan.Options(0, 0, 16))) {
            assertEquals(Wan.FPS_5B, wan.fps());
            assertThrows(IllegalArgumentException.class, () -> wan.generate(
                    new VideoRequest("a fox", null, 48, 64, 9, 1, 1f, 0f, 0), Progress.NONE));
            VideoRequest r = new VideoRequest("a red fox runs in snow", "blurry", 64, 64, 9, 3, 5f, 0f, 0);
            GeneratedVideo video = wan.generate(r, ref("pipe.noise"), Progress.NONE);
            assertEquals(9, video.frames().size());
            assertEquals(24, video.fps());
            float[] e = ref("pipe.video").toArray();   // [1, frames, 3, h, w] in [0, 1]
            int plane = 64 * 64;
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
            assertTrue(worst <= 1.0, "largest difference " + worst + " of 255");
        }
    }
}
