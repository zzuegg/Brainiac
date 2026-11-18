package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.SafeTensors;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Whole text-to-image runs against diffusers' pipelines on the tiny models. */
class PipelineTest {

    private static final Device DEVICE = TestDevice.get();
    private static Path dir;

    @BeforeAll
    static void open() throws Exception {
        dir = Path.of(PipelineTest.class.getResource("/tiny/reference.safetensors").toURI()).getParent();
        ReferenceTest.open();
    }

    private static Tensor run(StableDiffusion sd, String negative, int steps, float guidance) {
        ImageRequest r = ImageRequest.of("hello the cat").negativePrompt(negative).steps(steps).guidance(guidance)
                .sampler(Sampler.EULER).size(128, 64).build();
        StableDiffusion.Conditioning cond = sd.encode(r.prompt(), r.negativePrompt(), guidance > 1);
        Tensor latents = sd.denoise(ReferenceTest.ref("pipe.noise"), cond, r, new SplittableRandom(0),
                (step, total, x0) -> true);
        return sd.decode(latents);
    }

    @Test
    void stableDiffusionMatchesDiffusers() throws Exception {
        assertEquals(StableDiffusion.Kind.SD1, StableDiffusion.detect(dir.resolve("sd")).orElseThrow());
        try (StableDiffusion sd = StableDiffusion.load(dir.resolve("sd"), DEVICE, DType.FLOAT32)) {
            ReferenceTest.assertClose(ReferenceTest.ref("pipe.sd"), run(sd, "a cat", 4, 3f), 2e-4);
            ReferenceTest.assertClose(ReferenceTest.ref("pipe.sd_unguided"), run(sd, "", 3, 1f), 2e-4);
        }
    }

    @Test
    void sdxlMatchesDiffusers() throws Exception {
        assertEquals(StableDiffusion.Kind.SDXL, StableDiffusion.detect(dir.resolve("sdxl")).orElseThrow());
        try (StableDiffusion sd = StableDiffusion.load(dir.resolve("sdxl"), DEVICE, DType.FLOAT32)) {
            // no negative prompt: SDXL conditions on zeros
            ReferenceTest.assertClose(ReferenceTest.ref("pipe.sdxl"), run(sd, "", 4, 3f), 2e-4);
            ReferenceTest.assertClose(ReferenceTest.ref("pipe.sdxl_neg"), run(sd, "the mat", 4, 3f), 2e-4);
        }
    }

    @Test
    void singleFileCheckpointsMatchDiffusersFolders(@TempDir Path tmp) throws Exception {
        SingleFile.Arch arch = new SingleFile.Arch(4, 8, 2, 8, 8, 0.18215f);
        Files.createDirectories(tmp.resolve("tokenizer"));
        for (String f : new String[] {"vocab.json", "merges.txt"}) {
            Files.copy(dir.resolve("vocab").resolve(f), tmp.resolve("tokenizer").resolve(f));
        }
        Path sd1 = tmp.resolve("tiny-sd.safetensors");
        Path sdxl = tmp.resolve("tiny-sdxl.safetensors");
        writeCheckpoint(dir.resolve("sd"), sd1, false);
        writeCheckpoint(dir.resolve("sdxl"), sdxl, true);
        assertEquals(StableDiffusion.Kind.SD1, StableDiffusion.detect(sd1).orElseThrow());
        assertEquals(StableDiffusion.Kind.SDXL, StableDiffusion.detect(sdxl).orElseThrow());
        try (StableDiffusion sd = SingleFile.load(sd1, DEVICE, DType.FLOAT32, arch)) {
            ReferenceTest.assertClose(ReferenceTest.ref("pipe.sd"), run(sd, "a cat", 4, 3f), 2e-4);
        }
        try (StableDiffusion sd = SingleFile.load(sdxl, DEVICE, DType.FLOAT32, arch)) {
            ReferenceTest.assertClose(ReferenceTest.ref("pipe.sdxl_neg"), run(sd, "the mat", 4, 3f), 2e-4);
        }
    }

    /** Writes a diffusers folder as an original-layout single file, as image tools share them. */
    private static void writeCheckpoint(Path folder, Path file, boolean xl) throws Exception {
        Map<String, Tensor> out = new LinkedHashMap<>();
        var uc = Pretrained.unetSettings(Json.object(Json.read(folder.resolve("unet/config.json"))));
        var unetNames = SingleFile.unetNames(uc);
        try (SafeTensors s = SafeTensors.open(folder.resolve("unet/diffusion_pytorch_model.safetensors"))) {
            for (String n : s.names()) {
                out.put(unetNames.apply(n), s.tensor(n, DEVICE, DType.FLOAT32));
            }
        }
        var vaeNames = SingleFile.vaeNames(2);
        try (SafeTensors s = SafeTensors.open(folder.resolve("vae/diffusion_pytorch_model.safetensors"))) {
            for (String n : s.names()) {
                Tensor t = s.tensor(n, DEVICE, DType.FLOAT32);
                String ldm = vaeNames.apply(n);
                if (ldm.contains(".mid.attn_1.") && t.rank() == 2) {   // the original uses 1x1 convolutions
                    t = t.reshape(t.shape().dim(0), t.shape().dim(1), 1, 1);
                }
                out.put(ldm, t);
            }
        }
        String clipL = xl ? "conditioner.embedders.0.transformer.text_model." : "cond_stage_model.transformer.text_model.";
        try (SafeTensors s = SafeTensors.open(folder.resolve("text_encoder/model.safetensors"))) {
            for (String n : s.names()) {
                out.put(clipL + n.replaceFirst("^text_model\\.", ""), s.tensor(n, DEVICE, DType.FLOAT32));
            }
        }
        if (xl) {
            String g = "conditioner.embedders.1.model.";
            try (SafeTensors s = SafeTensors.open(folder.resolve("text_encoder_2/model.safetensors"))) {
                String p = s.contains("text_model.final_layer_norm.weight") ? "text_model." : "";
                out.put(g + "token_embedding.weight", s.tensor(p + "embeddings.token_embedding.weight", DEVICE, DType.FLOAT32));
                out.put(g + "positional_embedding", s.tensor(p + "embeddings.position_embedding.weight", DEVICE, DType.FLOAT32));
                for (int i = 0; s.contains(p + "encoder.layers." + i + ".layer_norm1.weight"); i++) {
                    String l = p + "encoder.layers." + i + ".";
                    String r = g + "transformer.resblocks." + i + ".";
                    for (String[] pair : new String[][] {{"layer_norm1", "ln_1"}, {"layer_norm2", "ln_2"},
                            {"self_attn.out_proj", "attn.out_proj"}, {"mlp.fc1", "mlp.c_fc"}, {"mlp.fc2", "mlp.c_proj"}}) {
                        out.put(r + pair[1] + ".weight", s.tensor(l + pair[0] + ".weight", DEVICE, DType.FLOAT32));
                        out.put(r + pair[1] + ".bias", s.tensor(l + pair[0] + ".bias", DEVICE, DType.FLOAT32));
                    }
                    for (String kind : new String[] {"weight", "bias"}) {
                        out.put(r + "attn.in_proj_" + kind, Tensor.concat(0,
                                s.tensor(l + "self_attn.q_proj." + kind, DEVICE, DType.FLOAT32),
                                s.tensor(l + "self_attn.k_proj." + kind, DEVICE, DType.FLOAT32),
                                s.tensor(l + "self_attn.v_proj." + kind, DEVICE, DType.FLOAT32)));
                    }
                }
                out.put(g + "ln_final.weight", s.tensor(p + "final_layer_norm.weight", DEVICE, DType.FLOAT32));
                out.put(g + "ln_final.bias", s.tensor(p + "final_layer_norm.bias", DEVICE, DType.FLOAT32));
                out.put(g + "text_projection", s.tensor("text_projection.weight", DEVICE, DType.FLOAT32).transpose());
            }
        }
        SafeTensors.write(file, out);
    }

    @Test
    void generateReportsProgressAndCancels() throws Exception {
        try (StableDiffusion sd = StableDiffusion.load(dir.resolve("sd"), DEVICE, DType.FLOAT32)) {
            AtomicInteger previews = new AtomicInteger();
            ImageRequest r = ImageRequest.of("the cat").size(64, 64).steps(3).count(2).seed(5).build();
            List<GeneratedImage> images = sd.generate(r, Progress.withPreviews(2, (step, total, preview) -> {
                assertEquals(6, total);
                if (preview != null) {
                    previews.incrementAndGet();
                    assertEquals(8, preview.width());
                }
                return true;
            }));
            assertEquals(2, images.size());
            assertEquals(16, images.get(0).width());   // the tiny VAE upsamples only twice
            assertEquals(5, images.get(0).seed());
            assertEquals(6, images.get(1).seed());
            assertEquals(4, previews.get());   // steps 2 and 3 of each image
            byte[] png = images.get(0).png();
            assertEquals((byte) 0x89, png[0]);
            assertEquals('P', png[1]);
            // the same seed makes the same image
            List<GeneratedImage> again = sd.generate(ImageRequest.of("the cat").size(64, 64).steps(3).seed(6).build(),
                    Progress.NONE);
            assertArrayEquals(images.get(1).rgb(), again.get(0).rgb());
            for (Sampler s : Sampler.values()) {
                GeneratedImage img = sd.generate(ImageRequest.of("the cat").size(64, 64).steps(3).sampler(s).build(),
                        Progress.NONE).get(0);
                assertNotNull(img);
            }
            AtomicInteger calls = new AtomicInteger();
            assertThrows(CancellationException.class, () -> sd.generate(r, (step, total, p) -> calls.incrementAndGet() < 2));
            assertEquals(2, calls.get());
        }
    }

    @Test
    void aspectRatios() {
        assertArrayEquals(new int[] {1024, 1024}, AspectRatio.SQUARE.size(1024));
        assertArrayEquals(new int[] {1152, 896}, AspectRatio.LANDSCAPE_4_3.size(1024));
        assertArrayEquals(new int[] {896, 1152}, AspectRatio.PORTRAIT_3_4.size(1024));
        assertArrayEquals(new int[] {1344, 768}, AspectRatio.WIDE_16_9.size(1024));
        assertArrayEquals(new int[] {768, 1344}, AspectRatio.TALL_9_16.size(1024));
        assertArrayEquals(new int[] {704, 384}, AspectRatio.WIDE_16_9.size(512));
        assertArrayEquals(new int[] {576, 448}, AspectRatio.LANDSCAPE_4_3.size(512));
        for (AspectRatio a : AspectRatio.values()) {
            for (int base : new int[] {512, 768, 1024}) {
                int[] s = a.size(base);
                assertTrue(s[0] % 64 == 0 && s[1] % 64 == 0, a + " " + base);
            }
        }
    }
}
