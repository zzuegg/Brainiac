package io.github.zzuegg.brainiac.examples;

import io.github.zzuegg.brainiac.diffusion.GeneratedImage;
import io.github.zzuegg.brainiac.diffusion.GeneratedVideo;
import io.github.zzuegg.brainiac.diffusion.VideoRequest;
import io.github.zzuegg.brainiac.diffusion.Wan;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Generates a short video from text, or from a picture and text, with Wan 2.1 or 2.2: a
 * diffusers folder or a folder of single files (ComfyUI's transformer, VAE and umT5 files plus
 * umT5's tokenizer, and {@code clip_vision_h.safetensors} for Wan 2.1's image-to-video models).
 *
 * <pre>
 * ./gradlew :brainiac-examples:run --args="video models/wan2.1-t2v-1.3b 'a red fox runs through fresh snow' --frames 33"
 * </pre>
 *
 * Options: {@code --negative text}, {@code --size WxH}, {@code --frames N} ({@code 1 + 4k}),
 * {@code --steps N}, {@code --guidance G}, {@code --shift S}, {@code --seed S},
 * {@code --device cpu|vulkan|<part of a name>} (several, comma-separated, spread the parts:
 * Wan 2.2's experts on different GPUs, umT5 where there is room), {@code --dtype f32|f16|bf16},
 * {@code --bits T,E} (transformer and text encoder bits, 0 for full), {@code --out folder},
 * {@code --avi} (also write an uncompressed AVI), {@code --image file} (the first frame, PNG or
 * JPEG, for image-to-video models and Wan 2.2 TI2V-5B), {@code --low-guidance G} (Wan 2.2's
 * low-noise expert's guidance).
 */
public final class VideoExample {

    private VideoExample() {
    }

    /**
     * Runs the example.
     *
     * @param args the model path, the prompt and options
     */
    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: video <model folder> <prompt> [--negative text] [--size WxH] [--frames N]"
                    + " [--steps N] [--guidance G] [--shift S] [--seed S] [--device name] [--dtype f32|f16|bf16]"
                    + " [--bits T,E] [--out folder] [--avi] [--image file] [--low-guidance G]");
            return;
        }
        Path path = Path.of(args[0]);
        VideoRequest request = VideoRequest.of(args[1]);
        String deviceName = "best";
        DType dtype = null;
        Wan.Options options = null;
        Path out = Path.of(".");
        boolean avi = false;
        GeneratedImage image = null;
        try {
            for (int i = 2; i < args.length; i++) {
                switch (args[i]) {
                    case "--negative" -> request = request.withNegativePrompt(args[++i]);
                    case "--size" -> {
                        String[] wh = args[++i].toLowerCase(Locale.ROOT).split("x");
                        request = request.withSize(Integer.parseInt(wh[0]), Integer.parseInt(wh[1]));
                    }
                    case "--frames" -> request = request.withFrames(Integer.parseInt(args[++i]));
                    case "--steps" -> request = request.withSteps(Integer.parseInt(args[++i]));
                    case "--guidance" -> request = request.withGuidance(Float.parseFloat(args[++i]));
                    case "--shift" -> request = request.withShift(Float.parseFloat(args[++i]));
                    case "--seed" -> request = request.withSeed(Long.parseLong(args[++i]));
                    case "--device" -> deviceName = args[++i];
                    case "--dtype" -> dtype = switch (args[++i]) {
                        case "f16" -> DType.FLOAT16;
                        case "bf16" -> DType.BFLOAT16;
                        default -> DType.FLOAT32;
                    };
                    case "--bits" -> {
                        String[] b = args[++i].split(",");
                        options = Wan.Options.bits(Integer.parseInt(b[0]), Integer.parseInt(b[1]));
                    }
                    case "--out" -> out = Path.of(args[++i]);
                    case "--avi" -> avi = true;
                    case "--image" -> image = read(Path.of(args[++i]));
                    case "--low-guidance" -> request = request.withLowNoiseGuidance(Float.parseFloat(args[++i]));
                    default -> throw new IllegalArgumentException("unknown option " + args[i]);
                }
            }
            List<Device> devices = new ArrayList<>();
            for (String name : deviceName.split(",")) {
                devices.add(device(name.strip()));
            }
            Device device = devices.get(0);
            if (dtype == null) {
                dtype = device.kind() != Device.Kind.GPU ? DType.FLOAT32
                        : device.supports(DType.BFLOAT16) ? DType.BFLOAT16
                        : device.supports(DType.FLOAT16) ? DType.FLOAT16 : DType.FLOAT32;
            }
            Wan.Files files = Wan.Files.find(path);
            Wan.Placement placement = Wan.Placement.single(device);
            if (options == null) {
                // the most precise weights that leave room for this video's size
                Wan.Plan plan = Wan.plan(files, devices, dtype, request);
                System.out.print(plan.describe());
                placement = plan.placement();
                options = plan.options();
            }
            System.out.println(placement + ", " + dtype + ", " + options);
            long start = System.nanoTime();
            try (Wan wan = Wan.load(files, placement, dtype, options)) {
                System.out.printf("Loaded %s in %.1f s%n", wan, (System.nanoTime() - start) / 1e9);
                // the peak of each phase: loading, the prompts and the first step, sampling, decoding
                List<Device> used = wan.placement().devices();
                peaks(used, "loading");
                long t0 = System.nanoTime();
                GeneratedVideo video = wan.generate(request, image, (step, total, preview) -> {
                    System.out.printf("\rstep %d / %d  (%.0f s)", step, total, (System.nanoTime() - t0) / 1e9);
                    if (step == 1 || step == total) {
                        System.out.println();
                        peaks(used, step == 1 ? "prompts and the first step" : "sampling");
                    }
                    return true;
                });
                peaks(used, "decoding");
                System.out.printf("%nGenerated %d frames (%dx%d, %.1f s of video) in %.1f s%n", video.frames().size(),
                        video.width(), video.height(), video.seconds(), (System.nanoTime() - t0) / 1e9);
                Files.createDirectories(out);
                Path file = out.resolve("video-" + video.seed() + ".png");
                Files.write(file, video.apng());
                System.out.println("Wrote " + file.toAbsolutePath());
                if (avi) {
                    Path a = out.resolve("video-" + video.seed() + ".avi");
                    Files.write(a, video.avi());
                    System.out.println("Wrote " + a.toAbsolutePath());
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** A PNG or JPEG file as 8-bit RGB. */
    private static GeneratedImage read(Path file) throws IOException {
        java.awt.image.BufferedImage b = javax.imageio.ImageIO.read(file.toFile());
        if (b == null) {
            throw new IOException(file + " is not an image ImageIO reads (PNG, JPEG, BMP, GIF)");
        }
        int w = b.getWidth();
        int h = b.getHeight();
        int[] argb = b.getRGB(0, 0, w, h, null, 0, w);
        byte[] rgb = new byte[w * h * 3];
        for (int i = 0; i < argb.length; i++) {
            rgb[i * 3] = (byte) (argb[i] >> 16);
            rgb[i * 3 + 1] = (byte) (argb[i] >> 8);
            rgb[i * 3 + 2] = (byte) argb[i];
        }
        return new GeneratedImage(w, h, rgb, -1);
    }

    static Device device(String want) {
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

    /** Prints each device's peak since the last call, next to what its pool reserved, and starts again. */
    private static void peaks(List<Device> devices, String phase) {
        for (Device d : devices) {
            d.memory().ifPresent(m -> System.out.printf("%s, %s: peak %.1f GB in use (%.1f GB now), %.1f GB reserved%n",
                    phase, d, m.peak() / 1e9, m.inUse() / 1e9, m.reserved() / 1e9));
            d.resetPeakMemory();
        }
    }
}
