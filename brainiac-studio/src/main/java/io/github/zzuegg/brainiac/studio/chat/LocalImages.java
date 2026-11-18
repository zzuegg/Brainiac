package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.diffusion.AspectRatio;
import io.github.zzuegg.brainiac.diffusion.GeneratedImage;
import io.github.zzuegg.brainiac.diffusion.ImageGenerator;
import io.github.zzuegg.brainiac.diffusion.ImageRequest;
import io.github.zzuegg.brainiac.diffusion.Progress;
import io.github.zzuegg.brainiac.diffusion.Sampler;
import io.github.zzuegg.brainiac.diffusion.StableDiffusion;
import io.github.zzuegg.brainiac.io.HuggingFaceHub;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;
import java.util.stream.Stream;

/**
 * The image models on this PC, found in the same places as the language models, and drawing with
 * them: Stable Diffusion 1.x, 2.x and XL (diffusers folders or single {@code .safetensors}
 * checkpoints) and FLUX.1 dev and schnell (diffusers folders). One model is kept loaded at a time,
 * in the type and, for Flux, the weight bits that fit the device.
 */
public final class LocalImages {

    private static final System.Logger LOG = System.getLogger(LocalImages.class.getName());

    /**
     * What one drawing asks for, beyond the prompt.
     *
     * @param negativePrompt what the pictures should not show
     * @param aspect {@code 1:1}, {@code 4:3}, {@code 3:4}, {@code 16:9} or {@code 9:16}
     * @param count how many pictures, 1 to 4
     * @param steps denoising steps, 1 to 150
     * @param guidance how closely to follow the prompt; 1 or less turns guidance off
     * @param seed the first picture's seed; picture {@code i} uses {@code seed + i}
     * @param sampler a sampler's name, such as {@code DPM_PP_2M_KARRAS}
     */
    public record Settings(String negativePrompt, String aspect, int count, int steps, double guidance, long seed,
            String sampler) {

        /** Checks the settings. */
        public Settings {
            negativePrompt = negativePrompt == null ? "" : negativePrompt;
            if (count < 1 || count > 4) {
                throw new IllegalArgumentException("draw 1 to 4 pictures at a time");
            }
            if (steps < 1 || steps > 150) {
                throw new IllegalArgumentException("choose 1 to 150 steps");
            }
            if (!(guidance >= 0) || guidance > 30) {
                throw new IllegalArgumentException("choose a guidance between 0 and 30");
            }
        }

        /**
         * The usual settings for a model.
         *
         * @param aspect the shape
         * @param count how many
         * @return the settings, with a random seed
         */
        public static Settings defaults(String aspect, int count) {
            return new Settings("", aspect, count, 25, 7, new java.util.Random().nextLong(1L << 32), "DPM_PP_2M_KARRAS");
        }
    }

    /** Hears each step of a drawing, with a rough preview now and then, and can stop it. */
    @FunctionalInterface
    public interface Listener {
        /**
         * Called after each step.
         *
         * @param step steps done over all pictures, from 1
         * @param total steps of the whole drawing
         * @param preview a rough preview at 1/8 size, or null
         * @return false to stop
         */
        boolean step(int step, int total, GeneratedImage preview);
    }

    /**
     * A picture made here.
     *
     * @param media the PNG
     * @param seed the seed it was made from
     */
    public record Picture(Media media, long seed) {
    }

    /** The samplers offered. */
    static final List<String> SAMPLERS = java.util.Arrays.stream(Sampler.values()).map(Enum::name).toList();

    private record Found(FileTime modified, Optional<String> family) {
    }

    private final Supplier<List<Path>> places;
    private final Supplier<HuggingFaceHub> hub;
    private final Map<Path, Found> seen = new ConcurrentHashMap<>();
    private ImageGenerator loaded;
    private Path loadedPath;

    LocalImages(Supplier<List<Path>> places, Supplier<HuggingFaceHub> hub) {
        this.places = places;
        this.hub = hub;
    }

    /**
     * Lists the image models, named by file or folder name; the first place with a name wins.
     *
     * @return the models
     */
    List<ModelInfo> models() {
        List<ModelInfo> out = new ArrayList<>();
        for (Map.Entry<String, Path> e : find().entrySet()) {
            String family = family(e.getValue()).orElseThrow();
            out.add(new ModelInfo("local", e.getKey(), e.getKey(), List.of("image"), 0,
                    label(family) + " · " + (Files.isDirectory(e.getValue()) ? "diffusers" : "checkpoint")));
        }
        return out;
    }

    /**
     * Says whether a name is one of these models.
     *
     * @param id the model's name
     * @return true if it is
     */
    boolean has(String id) {
        return find().containsKey(id);
    }

    static String label(String family) {
        return switch (family) {
            case "SD1" -> "Stable Diffusion 1.x";
            case "SD2" -> "Stable Diffusion 2.x";
            case "SDXL" -> "Stable Diffusion XL";
            case "FLUX_DEV" -> "FLUX.1 dev";
            case "FLUX_SCHNELL" -> "FLUX.1 schnell";
            default -> family;
        };
    }

    // what ImageGenerator.defaults() gives once a model is loaded, known here before loading it
    static ImageGenerator.Defaults defaults(String family) {
        return switch (family) {
            case "FLUX_DEV" -> new ImageGenerator.Defaults(28, 3.5f, false, false);
            case "FLUX_SCHNELL" -> new ImageGenerator.Defaults(4, 0f, false, false);
            case "SDXL" -> new ImageGenerator.Defaults(25, 5f, true, true);
            default -> new ImageGenerator.Defaults(25, 7f, true, true);
        };
    }

    private Map<String, Path> find() {
        Map<String, Path> found = new LinkedHashMap<>();
        for (Path place : places.get()) {
            for (Path p : candidates(place)) {
                if (family(p).isPresent()) {
                    found.putIfAbsent(p.getFileName().toString(), p);
                }
            }
        }
        return found;
    }

    // the place itself, or what is directly inside it
    private static List<Path> candidates(Path place) {
        if (looksLikeOne(place)) {
            return List.of(place);
        }
        if (!Files.isDirectory(place)) {
            return List.of();
        }
        try (Stream<Path> s = Files.list(place)) {
            return s.sorted().filter(LocalImages::looksLikeOne).toList();
        } catch (IOException e) {
            return List.of();
        }
    }

    private static boolean looksLikeOne(Path p) {
        return Files.isDirectory(p) ? Files.isRegularFile(p.resolve("unet/config.json"))
                        || Files.isRegularFile(p.resolve("transformer/config.json"))
                : p.getFileName() != null && p.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".safetensors");
    }

    // reading a checkpoint's header each time the models are listed would be slow: kept until the file changes
    private Optional<String> family(Path p) {
        FileTime t;
        try {
            Path marker = !Files.isDirectory(p) ? p
                    : Files.isRegularFile(p.resolve("transformer/config.json")) ? p.resolve("transformer/config.json")
                    : p.resolve("unet/config.json");
            t = Files.getLastModifiedTime(marker);
        } catch (IOException e) {
            return Optional.empty();
        }
        Found f = seen.get(p);
        if (f == null || !f.modified().equals(t)) {
            Optional<String> family;
            try {
                family = ImageGenerator.detect(p);
            } catch (RuntimeException e) {
                family = Optional.empty();
            }
            f = new Found(t, family);
            seen.put(p, f);
        }
        return f.family();
    }

    /**
     * Draws pictures.
     *
     * @param id the model's name
     * @param prompt what the pictures show
     * @param s the settings
     * @param listener hears each step and can stop
     * @return the pictures, as PNG
     * @throws IOException if the model cannot be loaded
     * @throws java.util.concurrent.CancellationException if the listener stopped it
     */
    synchronized List<Picture> draw(String id, String prompt, Settings s, Listener listener) throws IOException {
        Path path = find().get(id);
        if (path == null) {
            throw new IOException("no image model named " + id + " on this PC");
        }
        ImageGenerator model = load(path);
        ImageGenerator.Defaults d = model.defaults();
        AspectRatio ratio = switch (s.aspect()) {
            case "4:3" -> AspectRatio.LANDSCAPE_4_3;
            case "3:4" -> AspectRatio.PORTRAIT_3_4;
            case "16:9" -> AspectRatio.WIDE_16_9;
            case "9:16" -> AspectRatio.TALL_9_16;
            default -> AspectRatio.SQUARE;
        };
        int[] size = model.size(ratio);
        ImageRequest.Builder request = ImageRequest.of(prompt).size(size[0], size[1])
                .steps(s.steps()).guidance((float) s.guidance()).seed(s.seed()).count(s.count());
        if (d.negativePrompt()) {
            request.negativePrompt(s.negativePrompt());
        }
        if (d.samplers()) {
            try {
                request.sampler(Sampler.valueOf(s.sampler().toUpperCase(Locale.ROOT)));
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException("no sampler named " + s.sampler());
            }
        }
        List<GeneratedImage> images = model.generate(request.build(), Progress.withPreviews(2, listener::step));
        List<Picture> out = new ArrayList<>();
        for (GeneratedImage g : images) {
            out.add(new Picture(new Media(g.png(), "image/png"), g.seed()));
        }
        return out;
    }

    private ImageGenerator load(Path path) throws IOException {
        if (loaded != null && path.equals(loadedPath)) {
            return loaded;
        }
        if (loaded != null) {
            loaded.close();   // let the old weights go before loading new ones
            loaded = null;
            loadedPath = null;
        }
        // a single checkpoint has no CLIP vocabulary of its own; it is fetched once (about 1.6 MB)
        if (Files.isRegularFile(path) && !Files.isDirectory(path.resolveSibling("tokenizer"))
                && !Files.isRegularFile(StableDiffusion.tokenizerCache().resolve("vocab.json"))) {
            LOG.log(System.Logger.Level.INFO, "fetching the CLIP vocabulary that single-file checkpoints need");
            StableDiffusion.downloadTokenizer(hub.get());
        }
        Device device = Device.best();
        // the type suits the family and device; Flux's weights are quantized as far as memory needs,
        // and a model that cannot fit is refused before anything is loaded, saying how much it needs
        DType dtype = ImageGenerator.dtype(path, device);
        loaded = ImageGenerator.load(path, device, dtype);
        loadedPath = path;
        LOG.log(System.Logger.Level.INFO, "loaded " + path.getFileName() + " (" + label(loaded.family()) + ", " + dtype
                + ") on " + device);
        return loaded;
    }

    /**
     * Says what a model draws best at, for the page: its native square side.
     *
     * @param id the model's name
     * @return {@code {kind, label, steps, guidance, usesGuidance, usesNegative, usesSampler}} (the
     *     uses* say whether those settings do anything), or empty if unknown
     */
    Optional<Map<String, Object>> describe(String id) {
        Path p = find().get(id);
        if (p == null) {
            return Optional.empty();
        }
        String family = family(p).orElseThrow();
        ImageGenerator.Defaults d = defaults(family);
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("kind", family);
        m.put("label", label(family));
        m.put("steps", d.steps());
        m.put("guidance", (double) d.guidance());
        m.put("usesGuidance", !"FLUX_SCHNELL".equals(family));
        m.put("usesNegative", d.negativePrompt());
        m.put("usesSampler", d.samplers());
        return Optional.of(m);
    }

    synchronized void close() {
        if (loaded != null) {
            loaded.close();
            loaded = null;
            loadedPath = null;
        }
    }
}
