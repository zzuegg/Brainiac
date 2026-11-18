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
 * Generates an image with every Stable Diffusion and Flux model in the folder {@code -PtestModels=...}
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
            Optional<String> family = ImageGenerator.detect(p);
            if (family.isEmpty()) {
                continue;
            }
            boolean flux = family.get().startsWith("FLUX");
            long t0 = System.nanoTime();
            // Flux overflows in float16; -PtestFluxBits=8,4 stores the transformer and T5 in fewer
            // bits, by default as few as the device's memory needs
            String bits = System.getProperty("brainiac.test.fluxBits", "auto");
            DType fluxType = dtype == DType.FLOAT32 ? dtype : DType.BFLOAT16;
            device.resetPeakMemory();
            try (ImageGenerator model = !flux ? StableDiffusion.load(p, device, dtype)
                    : bits.equals("auto") ? ImageGenerator.load(p, device, fluxType)
                    : Flux.load(p, device, fluxType, Flux.Options.bits(Integer.parseInt(bits.split(",")[0].trim()),
                            Integer.parseInt(bits.split(",")[1].trim())))) {
                long t1 = System.nanoTime();
                ImageGenerator.Defaults d = model.defaults();
                ImageRequest r = ImageRequest.of("a photo of a red fox sitting in fresh snow, golden hour, detailed fur")
                        .negativePrompt("blurry, low quality").steps(d.steps()).guidance(d.guidance()).seed(42)
                        .build();
                GeneratedImage img = model.generate(r, Progress.NONE).get(0);
                long t2 = System.nanoTime();
                assertEquals(model.nativeSize(), img.width());
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
                System.out.printf("%s: %s%s, loaded in %.1f s, %d steps + decode in %.1f s, peak %s -> %s%n",
                        p.getFileName(), model, model instanceof Flux f ? " " + f.options() : "", (t1 - t0) / 1e9,
                        d.steps(), (t2 - t1) / 1e9, device.memory().map(m -> MemoryFit.gb(m.peak())).orElse("n/a"),
                        file.toAbsolutePath());
                ran++;
            }
        }
        assertTrue(ran > 0, "no image model in " + folder);
    }
}
