package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.io.SafeTensors;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static io.github.zzuegg.brainiac.diffusion.ReferenceTest.assertClose;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Compares Wan's image-to-video pipelines with diffusers' {@code WanImageToVideoPipeline} on
 * tiny random models (made by {@code src/test/python/wan_i2v_fixtures.py}): Wan 2.2 TI2V's
 * first frame, Wan 2.2 I2V-A14B's extra channels with two experts, and Wan 2.1 I2V's CLIP
 * image features; and the Pillow resampling they rely on.
 */
class WanImageToVideoTest {

    private static Path dir;
    private static SafeTensors reference;
    private static final Device DEVICE = TestDevice.get();
    private static final Wan.Options TINY = new Wan.Options(0, 0, 16);

    @BeforeAll
    static void open() throws Exception {
        dir = Path.of(WanImageToVideoTest.class.getResource("/tiny/wan_i2v_reference.safetensors").toURI())
                .getParent();
        reference = SafeTensors.open(dir.resolve("wan_i2v_reference.safetensors"));
    }

    @AfterAll
    static void close() {
        reference.close();
    }

    private static Tensor ref(String name) {
        return reference.tensor(name, DEVICE, DType.FLOAT32);
    }

    /** A stored {@code [h, w, 3]} picture as an image. */
    private static GeneratedImage image(String name) {
        Tensor t = reference.tensor(name, Device.cpu(), DType.FLOAT32);
        Shape s = t.shape();
        float[] v = t.toArray();
        byte[] rgb = new byte[v.length];
        for (int i = 0; i < v.length; i++) {
            rgb[i] = (byte) Math.round(v[i]);
        }
        return new GeneratedImage(s.dim(1), s.dim(0), rgb, 0);
    }

    /** The largest difference in 0..255 between a video and diffusers' frames {@code [1, frames, 3, h, w]}. */
    private static double worst(GeneratedVideo video, String name) {
        float[] e = ref(name).toArray();
        GeneratedImage first = video.frames().get(0);
        int plane = first.width() * first.height();
        double worst = 0;
        for (int f = 0; f < video.frames().size(); f++) {
            GeneratedImage img = video.frames().get(f);
            for (int c = 0; c < 3; c++) {
                for (int p = 0; p < plane; p++) {
                    int mine = img.rgb()[p * 3 + c] & 0xff;
                    worst = Math.max(worst, Math.abs(mine - e[(f * 3 + c) * plane + p] * 255));
                }
            }
        }
        return worst;
    }

    @Test
    void resamplesLikePillow() {
        GeneratedImage src = image("pil.source");
        assertSame(src, Pixels.resize(src, src.width(), src.height(), Pixels.Filter.LANCZOS));
        for (String[] c : new String[][] {{"pil.bicubic", "BICUBIC"}, {"pil.lanczos", "LANCZOS"},
                {"pil.mixed", "LANCZOS"}}) {
            GeneratedImage want = image(c[0]);
            GeneratedImage got = Pixels.resize(src, want.width(), want.height(), Pixels.Filter.valueOf(c[1]));
            int worst = 0;
            for (int i = 0; i < want.rgb().length; i++) {
                worst = Math.max(worst, Math.abs((got.rgb()[i] & 0xff) - (want.rgb()[i] & 0xff)));
            }
            // Pillow sums in fixed point; one level apart at most
            assertTrue(worst <= 1, c[0] + " differs by " + worst);
        }
        GeneratedImage covered = Pixels.cover(src, 30, 30, Pixels.Filter.LANCZOS);
        assertEquals(30, covered.width());
        assertEquals(30, covered.height());
    }

    @Test
    void firstFrameModelMatchesDiffusers() throws Exception {
        try (Wan wan = Wan.load(dir.resolve("wan22"), DEVICE, DType.FLOAT32, TINY)) {
            assertEquals(Wan.Input.TEXT_OR_IMAGE, wan.input());
            VideoRequest r = new VideoRequest("a red fox runs in snow", "blurry", 64, 64, 9, 3, 5f, 0f, 0);
            GeneratedVideo video = wan.generate(r, image("ti2v.image"), ref("ti2v.noise"), Progress.NONE);
            assertEquals(9, video.frames().size());
            double worst = worst(video, "ti2v.video");
            assertTrue(worst <= 1.0, "largest difference " + worst + " of 255");
        }
    }

    @Test
    void channelModelWithTwoExpertsMatchesDiffusers() throws Exception {
        Wan.Files files = Wan.Files.find(dir.resolve("wan_i2v22"));
        assertEquals(2, files.transformers().size());
        assertEquals(0.9, files.boundary(), 1e-9);
        assertNull(files.imageEncoder());
        try (Wan wan = Wan.load(files, DEVICE, DType.FLOAT32, TINY)) {
            assertEquals(Wan.Input.IMAGE, wan.input());
            VideoRequest r = new VideoRequest("a lighthouse in a storm", "blurry", 48, 32, 9, 4, 4f, 0f, 0, 3f);
            assertThrows(IllegalArgumentException.class, () -> wan.generate(r, Progress.NONE));
            GeneratedVideo video = wan.generate(r, image("i2v22.image"), ref("i2v22.noise"), Progress.NONE);
            double worst = worst(video, "i2v22.video");
            assertTrue(worst <= 1.0, "largest difference " + worst + " of 255");
        }
    }

    @Test
    void clipModelMatchesDiffusers() throws Exception {
        Wan.Files files = Wan.Files.find(dir.resolve("wan_i2v21"));
        assertNotNull(files.imageEncoder());
        // diffusers tells image from text states by Wan's 512 text tokens
        try (Wan wan = Wan.load(files, DEVICE, DType.FLOAT32, new Wan.Options(0, 0, 512))) {
            assertEquals(Wan.Input.IMAGE, wan.input());
            assertEquals(2, wan.imageEncoder().settings().layers());   // all but the last of 3
            assertClose(ref("i2v21.features"), wan.imageFeatures(image("i2v21.image")), 1e-4);
            VideoRequest r = new VideoRequest("a lighthouse in a storm", "blurry", 48, 32, 9, 3, 5f, 0f, 0);
            GeneratedVideo video = wan.generate(r, image("i2v21.image"), ref("i2v21.noise"), Progress.NONE);
            double worst = worst(video, "i2v21.video");
            assertTrue(worst <= 1.0, "largest difference " + worst + " of 255");
        }
    }

    @Test
    void textModelRefusesPictures() throws Exception {
        try (Wan wan = Wan.load(dir.resolve("wan"), DEVICE, DType.FLOAT32, TINY)) {
            assertEquals(Wan.Input.TEXT, wan.input());
            VideoRequest r = new VideoRequest("a fox", null, 32, 32, 5, 1, 1f, 0f, 0);
            GeneratedImage picture = new GeneratedImage(32, 32, new byte[32 * 32 * 3], 0);
            assertThrows(IllegalArgumentException.class, () -> wan.generate(r, picture, Progress.NONE));
        }
    }
}
