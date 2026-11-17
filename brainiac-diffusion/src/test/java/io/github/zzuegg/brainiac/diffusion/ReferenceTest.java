package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.SafeTensors;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Compares each model against outputs that diffusers and transformers computed for tiny random
 * models with the same weights (made by {@code src/test/python/fixtures.py}).
 */
class ReferenceTest {

    static Path dir;
    private static SafeTensors reference;
    private static final Device DEVICE = TestDevice.get();

    @BeforeAll
    static void open() throws URISyntaxException {
        dir = Path.of(ReferenceTest.class.getResource("/tiny/reference.safetensors").toURI()).getParent();
        reference = SafeTensors.open(dir.resolve("reference.safetensors"));
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

    static void assertClose(Tensor expected, Tensor actual, double tolerance) {
        assertEquals(expected.shape(), actual.shape());
        float[] e = expected.toArray();
        float[] a = actual.toArray();
        double scale = 0;
        double worst = 0;
        for (int i = 0; i < e.length; i++) {
            scale = Math.max(scale, Math.abs(e[i]));
            worst = Math.max(worst, Math.abs(e[i] - a[i]));
        }
        assertTrue(worst <= tolerance * Math.max(1, scale),
                "largest difference " + worst + " (values up to " + scale + ")");
    }

    private static int[][] ids(String name) {
        float[] f = ref(name).toArray();
        int n = f.length / 2;
        int[][] ids = new int[2][n];
        for (int i = 0; i < f.length; i++) {
            ids[i / n][i % n] = (int) f[i];
        }
        return ids;
    }

    @Test
    void clipTextEncoder() throws Exception {
        for (String name : new String[] {"text_encoder", "text_encoder_2"}) {
            ClipTextModel.Config c = ClipTextModel.Config.fromJson(config("sdxl/" + name));
            ClipTextModel m = ClipTextModel.load(weights("sdxl/" + name, "model.safetensors"), c, c.layers());
            ClipTextModel.Output out = m.encode(ids(name + ".ids"));
            assertClose(ref(name + ".last"), out.last(), 1e-4);
            assertClose(ref(name + ".penultimate"), out.penultimate(), 1e-4);
            assertClose(ref(name + ".pooled"), out.pooled(), 1e-4);
        }
    }

    private static float[] floats(String name) {
        return ref(name).toArray();
    }

    @Test
    void unetSd15() throws Exception {
        UNet unet = UNet.load(weights("sd/unet", "diffusion_pytorch_model.safetensors"),
                UNet.Config.fromJson(config("sd/unet")));
        Tensor out = unet.forward(ref("unet.x"), floats("unet.t"), ref("unet.ctx"), null, null);
        assertClose(ref("unet.out"), out, 1e-4);
    }

    @Test
    void unetXl() throws Exception {
        UNet unet = UNet.load(weights("sdxl/unet", "diffusion_pytorch_model.safetensors"),
                UNet.Config.fromJson(config("sdxl/unet")));
        float[] ids = floats("unet_xl.time_ids");
        float[][] timeIds = {java.util.Arrays.copyOfRange(ids, 0, 6), java.util.Arrays.copyOfRange(ids, 6, 12)};
        Tensor out = unet.forward(ref("unet_xl.x"), floats("unet_xl.t"), ref("unet_xl.ctx"), ref("unet_xl.pooled"),
                timeIds);
        assertClose(ref("unet_xl.out"), out, 1e-4);
    }

    @Test
    void vae() throws Exception {
        Vae.Config c = Vae.Config.fromJson(config("sd/vae"));
        Vae vae = Vae.load(weights("sd/vae", "diffusion_pytorch_model.safetensors"), c, true);
        // decode takes latents as the UNet sees them; the reference decoded raw latents
        assertClose(ref("vae.decoded"), vae.decode(ref("vae.z").mul(c.scalingFactor())), 1e-4);
        assertClose(ref("vae.mean").mul(c.scalingFactor()), vae.encode(ref("vae.image")), 1e-4);
    }
}
