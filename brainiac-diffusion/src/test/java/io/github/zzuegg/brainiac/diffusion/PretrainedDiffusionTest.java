package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeFalse;

/**
 * Generates an image with every Stable Diffusion model in the folder {@code -PtestModels=...}
 * names, and writes it to {@code build/pretrained-images}. Skipped without models.
 */
class PretrainedDiffusionTest {

    @Test
    void generatesWithPretrainedModels() throws Exception {
        String folder = System.getProperty("brainiac.test.models", "");
        assumeFalse(folder.isBlank(), "no -PtestModels folder");
        Device device = TestDevice.get();
        DType dtype = "f32".equals(System.getProperty("brainiac.test.dtype")) ? DType.FLOAT32 : DType.FLOAT16;
        Path out = Path.of("build", "pretrained-images");
        Files.createDirectories(out);
        List<Path> candidates;
        try (Stream<Path> s = Files.list(Path.of(folder))) {
            candidates = s.sorted().toList();
        }
        int ran = 0;
        for (Path p : candidates) {
            Optional<StableDiffusion.Kind> kind = StableDiffusion.detect(p);
            if (kind.isEmpty()) {
                continue;
            }
            long t0 = System.nanoTime();
            try (StableDiffusion sd = StableDiffusion.load(p, device, dtype)) {
                long t1 = System.nanoTime();
                ImageRequest r = ImageRequest.of("a photo of a red fox sitting in fresh snow, golden hour, detailed fur")
                        .negativePrompt("blurry, low quality").steps(25).seed(42).build();
                GeneratedImage img = sd.generate(r, Progress.NONE).get(0);
                long t2 = System.nanoTime();
                assertEquals(sd.nativeSize(), img.width());
                double mean = 0;
                double sq = 0;
                for (byte b : img.rgb()) {
                    mean += b & 0xff;
                    sq += (b & 0xff) * (b & 0xff);
                }
                mean /= img.rgb().length;
                double std = Math.sqrt(sq / img.rgb().length - mean * mean);
                assertTrue(std > 10, "the image is almost flat (std " + std + ")");
                Path file = out.resolve(p.getFileName().toString().replace(".safetensors", "") + ".png");
                Files.write(file, img.png());
                System.out.printf("%s: %s, loaded in %.1f s, 25 steps + decode in %.1f s -> %s%n", p.getFileName(),
                        sd, (t1 - t0) / 1e9, (t2 - t1) / 1e9, file.toAbsolutePath());
                ran++;
            }
        }
        assertTrue(ran > 0, "no Stable Diffusion model in " + folder);
    }
}
