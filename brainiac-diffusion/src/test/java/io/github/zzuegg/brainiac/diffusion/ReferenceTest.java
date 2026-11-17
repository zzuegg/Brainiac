package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.nn.diffusion.AutoencoderKL;
import io.github.zzuegg.brainiac.nn.diffusion.ClipTextEncoder;
import io.github.zzuegg.brainiac.nn.diffusion.UNet;
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
            ClipTextEncoder.Settings c = Pretrained.clipSettings(config("sdxl/" + name));
            ClipTextEncoder m = Pretrained.clip(weights("sdxl/" + name, "model.safetensors"), c, c.layers());
            ClipTextEncoder.Output out = Var.noGrad(() -> m.encode(Var.of(StableDiffusion.idTensor(ids(name + ".ids"),
                    DEVICE)), false));
            assertClose(ref(name + ".last"), out.last().value(), 1e-4);
            assertClose(ref(name + ".penultimate"), out.penultimate().value(), 1e-4);
            assertClose(ref(name + ".pooled"), out.pooled().value(), 1e-4);
        }
    }

    private static Tensor steps(String name) {
        return ref(name);
    }

    private static Tensor unet(UNet unet, Tensor x, Tensor t, Tensor ctx, Tensor added) {
        return Var.noGrad(() -> unet.forward(Var.of(x), Var.of(t), Var.of(ctx), added == null ? null : Var.of(added),
                false).value());
    }

    @Test
    void unetSd15() throws Exception {
        UNet unet = Pretrained.unet(weights("sd/unet", "diffusion_pytorch_model.safetensors"),
                Pretrained.unetSettings(config("sd/unet")));
        Tensor out = unet(unet, ref("unet.x"), steps("unet.t"), ref("unet.ctx"), null);
        assertClose(ref("unet.out"), out, 1e-4);
    }

    @Test
    void unetXl() throws Exception {
        UNet unet = Pretrained.unet(weights("sdxl/unet", "diffusion_pytorch_model.safetensors"),
                Pretrained.unetSettings(config("sdxl/unet")));
        Tensor added = Tensor.concat(1, ref("unet_xl.pooled"), ref("unet_xl.time_ids").reshape(2, 6));
        Tensor out = unet(unet, ref("unet_xl.x"), steps("unet_xl.t"), ref("unet_xl.ctx"), added);
        assertClose(ref("unet_xl.out"), out, 1e-4);
    }

    @Test
    void vae() throws Exception {
        AutoencoderKL.Settings c = Pretrained.vaeSettings(config("sd/vae"), true);
        AutoencoderKL vae = Pretrained.vae(weights("sd/vae", "diffusion_pytorch_model.safetensors"), c);
        // fromLatents takes latents as the UNet sees them; the reference decoded raw latents
        Tensor decoded = Var.noGrad(() -> vae.fromLatents(Var.of(ref("vae.z").mul(c.scalingFactor()))).value());
        assertClose(ref("vae.decoded"), decoded, 1e-4);
        Tensor latents = Var.noGrad(() -> vae.toLatents(Var.of(ref("vae.image"))).value());
        assertClose(ref("vae.mean").mul(c.scalingFactor()), latents, 1e-4);
    }

    @Test
    void sameResultsWithGradients() throws Exception {
        // the layers give the same values when they record gradients (as while training)
        UNet unet = Pretrained.unet(weights("sd/unet", "diffusion_pytorch_model.safetensors"),
                Pretrained.unetSettings(config("sd/unet")));
        Var out = unet.forward(Var.of(ref("unet.x")), Var.of(steps("unet.t")), Var.of(ref("unet.ctx")), null, false);
        assertTrue(out.requiresGrad());
        assertClose(ref("unet.out"), out.value(), 1e-4);
    }
}
