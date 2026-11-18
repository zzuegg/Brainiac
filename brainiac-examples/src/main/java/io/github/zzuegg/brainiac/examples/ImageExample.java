package io.github.zzuegg.brainiac.examples;

import io.github.zzuegg.brainiac.diffusion.GeneratedImage;
import io.github.zzuegg.brainiac.diffusion.ImageRequest;
import io.github.zzuegg.brainiac.diffusion.Sampler;
import io.github.zzuegg.brainiac.diffusion.StableDiffusion;
import io.github.zzuegg.brainiac.io.HuggingFaceHub;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

/**
 * Generates images from text with a Stable Diffusion model: a diffusers folder (such as
 * {@code stable-diffusion-v1-5} or {@code stable-diffusion-xl-base-1.0}) or a single
 * {@code .safetensors} checkpoint.
 *
 * <pre>
 * ./gradlew :brainiac-examples:run --args="image models/stable-diffusion-v1-5 'a lighthouse at dusk, oil painting'"
 * ./gradlew :brainiac-examples:run --args="image models/sd_xl_base_1.0.safetensors 'a red fox in snow' --steps 30 --seed 7"
 * </pre>
 *
 * Options: {@code --negative text}, {@code --size WxH}, {@code --steps N}, {@code --guidance G},
 * {@code --seed S}, {@code --count N}, {@code --sampler euler|euler_a|dpmpp_2m|dpmpp_2m_karras},
 * {@code --device cpu|vulkan|<part of a name>}, {@code --dtype f32|f16}, {@code --out folder},
 * {@code --download-tokenizer} (fetches the CLIP vocabulary single files need, 1.6 MB).
 */
public final class ImageExample {

    private ImageExample() {
    }

    /**
     * Runs the example.
     *
     * @param args the model path, the prompt and options
     */
    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: image <diffusers folder or .safetensors> <prompt> [--negative text] [--size WxH]"
                    + " [--steps N] [--guidance G] [--seed S] [--count N] [--sampler name] [--device name]"
                    + " [--dtype f32|f16] [--out folder] [--download-tokenizer]");
            return;
        }
        Path path = Path.of(args[0]);
        ImageRequest.Builder request = ImageRequest.of(args[1]);
        String deviceName = "best";
        DType dtype = null;
        Path out = Path.of(".");
        try {
            for (int i = 2; i < args.length; i++) {
                switch (args[i]) {
                    case "--negative" -> request.negativePrompt(args[++i]);
                    case "--size" -> {
                        String[] wh = args[++i].toLowerCase(Locale.ROOT).split("x");
                        request.size(Integer.parseInt(wh[0]), Integer.parseInt(wh[1]));
                    }
                    case "--steps" -> request.steps(Integer.parseInt(args[++i]));
                    case "--guidance" -> request.guidance(Float.parseFloat(args[++i]));
                    case "--seed" -> request.seed(Long.parseLong(args[++i]));
                    case "--count" -> request.count(Integer.parseInt(args[++i]));
                    case "--sampler" -> request.sampler(switch (args[++i].toLowerCase(Locale.ROOT)) {
                        case "euler" -> Sampler.EULER;
                        case "euler_a" -> Sampler.EULER_ANCESTRAL;
                        case "dpmpp_2m" -> Sampler.DPM_PP_2M;
                        default -> Sampler.DPM_PP_2M_KARRAS;
                    });
                    case "--device" -> deviceName = args[++i];
                    case "--dtype" -> dtype = "f16".equals(args[++i]) ? DType.FLOAT16 : DType.FLOAT32;
                    case "--out" -> out = Path.of(args[++i]);
                    case "--download-tokenizer" -> System.out.println("CLIP vocabulary in "
                            + StableDiffusion.downloadTokenizer(HuggingFaceHub.standard()));
                    default -> throw new IllegalArgumentException("unknown option " + args[i]);
                }
            }
            Device device = device(deviceName);
            if (dtype == null) {
                dtype = device.kind() == Device.Kind.GPU && device.supports(DType.FLOAT16) ? DType.FLOAT16 : DType.FLOAT32;
            }
            long start = System.nanoTime();
            try (StableDiffusion sd = StableDiffusion.load(path, device, dtype)) {
                System.out.printf("Loaded %s in %.1f s%n", sd, (System.nanoTime() - start) / 1e9);
                ImageRequest r = request.build();
                long t0 = System.nanoTime();
                List<GeneratedImage> images = sd.generate(r, (step, total, preview) -> {
                    System.out.printf("\rstep %d / %d", step, total);
                    return true;
                });
                System.out.printf("%nGenerated %d image(s) in %.1f s%n", images.size(), (System.nanoTime() - t0) / 1e9);
                Files.createDirectories(out);
                for (GeneratedImage img : images) {
                    Path file = out.resolve("image-" + img.seed() + ".png");
                    Files.write(file, img.png());
                    System.out.println("Wrote " + file.toAbsolutePath());
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static Device device(String want) {
        if (want.equals("best")) {
            return Device.best();
        }
        if (want.equals("cpu")) {
            return Device.cpu();
        }
        return Device.available().stream().filter(d -> d.name().toLowerCase(Locale.ROOT).contains(want)).findFirst()
                .orElseThrow(() -> new IllegalArgumentException("no device matches '" + want + "': "
                        + Device.available()));
    }
}
