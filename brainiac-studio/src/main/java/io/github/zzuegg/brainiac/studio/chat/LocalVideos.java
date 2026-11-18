package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.diffusion.GeneratedImage;
import io.github.zzuegg.brainiac.diffusion.GeneratedVideo;
import io.github.zzuegg.brainiac.diffusion.VideoRequest;
import io.github.zzuegg.brainiac.diffusion.Wan;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;
import java.util.stream.Stream;

/**
 * The video models on this PC, found in the same places as the other models, and making clips
 * with them: Wan 2.1 and 2.2 text-to-video, from a diffusers folder or a folder of single files
 * (ComfyUI's transformer, VAE and umT5 files plus umT5's {@code tokenizer.json}). One model is
 * kept loaded at a time, in the type and weight bits that fit the device.
 */
public final class LocalVideos {

    private static final System.Logger LOG = System.getLogger(LocalVideos.class.getName());

    /** The clip shapes offered, as width and height for Wan 2.1 (832x480 class) models. */
    static final Map<String, int[]> SIZES_480 = Map.of("16:9", new int[] {832, 480}, "9:16", new int[] {480, 832},
            "1:1", new int[] {624, 624}, "4:3", new int[] {720, 544}, "3:4", new int[] {544, 720});
    /** The clip shapes for Wan 2.2 5B (1280x704 class), whose sides are multiples of 32. */
    static final Map<String, int[]> SIZES_720 = Map.of("16:9", new int[] {1280, 704}, "9:16", new int[] {704, 1280},
            "1:1", new int[] {960, 960}, "4:3", new int[] {1088, 832}, "3:4", new int[] {832, 1088});

    /**
     * What one clip asks for, beyond the prompt.
     *
     * @param negativePrompt what the clip should not show; empty for none
     * @param aspect {@code 16:9}, {@code 9:16}, {@code 1:1}, {@code 4:3} or {@code 3:4}
     * @param small whether to make it at half the side, much faster and lighter
     * @param seconds how long, 1 to 10
     * @param steps denoising steps, 1 to 100
     * @param guidance how closely to follow the prompt; 1 or less turns guidance off
     * @param seed the seed
     * @param width a width of its own, or 0 for the aspect's; rounded to the model's grid
     * @param height a height of its own, or 0 for the aspect's
     * @param start the first frame (a PNG or JPEG) for image-to-video models, or {@code null}
     */
    public record Settings(String negativePrompt, String aspect, boolean small, double seconds, int steps,
            double guidance, long seed, int width, int height, Media start) {

        /** Checks the settings. */
        public Settings {
            negativePrompt = negativePrompt == null ? "" : negativePrompt;
            aspect = SIZES_480.containsKey(aspect) ? aspect : "16:9";
            if (!(seconds >= 0.1) || seconds > 10) {
                throw new IllegalArgumentException("make clips of up to 10 seconds");
            }
            if (steps < 1 || steps > 100) {
                throw new IllegalArgumentException("choose 1 to 100 steps");
            }
            if (!(guidance >= 0) || guidance > 20) {
                throw new IllegalArgumentException("choose a guidance between 0 and 20");
            }
            if (width < 0 || height < 0 || width > 4096 || height > 4096 || (width == 0) != (height == 0)) {
                throw new IllegalArgumentException("give both sides, up to 4096 pixels, or neither");
            }
        }

        /**
         * Settings at the aspect's size.
         *
         * @param negativePrompt what the clip should not show
         * @param aspect the shape
         * @param small whether at half the side
         * @param seconds how long
         * @param steps denoising steps
         * @param guidance the guidance
         * @param seed the seed
         */
        public Settings(String negativePrompt, String aspect, boolean small, double seconds, int steps,
                double guidance, long seed) {
            this(negativePrompt, aspect, small, seconds, steps, guidance, seed, 0, 0, null);
        }

        /**
         * Settings without a first frame.
         *
         * @param negativePrompt what the clip should not show
         * @param aspect the shape
         * @param small whether at half the side
         * @param seconds how long
         * @param steps denoising steps
         * @param guidance the guidance
         * @param seed the seed
         * @param width a width of its own, or 0
         * @param height a height of its own, or 0
         */
        public Settings(String negativePrompt, String aspect, boolean small, double seconds, int steps,
                double guidance, long seed, int width, int height) {
            this(negativePrompt, aspect, small, seconds, steps, guidance, seed, width, height, null);
        }

        /**
         * The request for a model: its size class, the frame count nearest the length
         * ({@code 1 + 4k}), and the sides on the model's grid.
         *
         * @param prompt what the clip shows
         * @param large whether the model is a 720p one (Wan 2.2 5B)
         * @param fps the model's frame rate
         * @return the request
         */
        VideoRequest request(String prompt, boolean large, int fps) {
            int[] wh = (large ? SIZES_720 : SIZES_480).get(aspect);
            int grid = large ? 32 : 16;
            int w = width > 0 ? Math.max(grid, Math.round(width / (float) grid) * grid) : wh[0];
            int h = height > 0 ? Math.max(grid, Math.round(height / (float) grid) * grid) : wh[1];
            if (small) {
                w = Math.max(grid, Math.round(w / 2f / grid) * grid);
                h = Math.max(grid, Math.round(h / 2f / grid) * grid);
            }
            int frames = 1 + 4 * Math.max(0, (int) Math.round(seconds * fps / 4.0));
            VideoRequest r = new VideoRequest(prompt, negativePrompt.isBlank() ? null : negativePrompt, w, h, frames,
                    steps, (float) guidance, 0f, seed);
            return r;
        }
    }

    /** Hears each step and can stop the clip. */
    @FunctionalInterface
    public interface Listener {
        /**
         * Called after each step.
         *
         * @param step steps done, from 1
         * @param total all steps
         * @return false to stop
         */
        boolean step(int step, int total);
    }

    /**
     * A clip made here.
     *
     * @param media the animated PNG
     * @param seed the seed it was made from
     * @param width its width
     * @param height its height
     * @param frames its frame count
     * @param fps its frame rate
     */
    public record Clip(Media media, long seed, int width, int height, int frames, int fps) {
    }

    private record Found(FileTime modified, boolean wan, boolean large, Wan.Input input) {
    }

    private final Supplier<List<Path>> places;
    private final Map<Path, Found> seen = new ConcurrentHashMap<>();
    private final LocalDevice where;
    private final Runnable makeRoom;
    private volatile Wan loaded;
    private volatile Path loadedPath;
    private volatile String loadedOn;
    private final GpuSlot slot;

    /**
     * @param places where to look for models
     * @param where the device models run on
     * @param makeRoom frees other models' memory before a video model loads
     */
    LocalVideos(Supplier<List<Path>> places, LocalDevice where, Runnable makeRoom) {
        this.places = places;
        this.where = where;
        this.makeRoom = makeRoom;
        this.slot = new GpuSlot("video", where.residency());
    }

    /**
     * Lists the video models, named by folder name; the first place with a name wins.
     *
     * @return the models
     */
    List<ModelInfo> models() {
        List<ModelInfo> out = new ArrayList<>();
        for (Map.Entry<String, Path> e : find().entrySet()) {
            Found f = seen.get(e.getValue());
            boolean diffusers = Files.isRegularFile(e.getValue().resolve("model_index.json"));
            out.add(new ModelInfo("local", e.getKey(), e.getKey(), List.of("video"), 0,
                    label(f != null && f.large()) + " · " + (diffusers ? "diffusers" : "single files")));
        }
        return out;
    }

    static String label(boolean large) {
        return large ? "Wan 2.2 5B video" : "Wan video";
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

    private Map<String, Path> find() {
        Map<String, Path> found = new LinkedHashMap<>();
        for (Path place : places.get()) {
            for (Path p : candidates(place)) {
                if (check(p).wan()) {
                    found.putIfAbsent(p.getFileName().toString(), p);
                }
            }
        }
        return found;
    }

    // the place itself, or the folders directly inside it
    private static List<Path> candidates(Path place) {
        if (!Files.isDirectory(place)) {
            return List.of();
        }
        List<Path> out = new ArrayList<>(List.of(place));
        try (Stream<Path> s = Files.list(place)) {
            s.sorted().filter(Files::isDirectory).forEach(out::add);
        } catch (IOException e) {
            // an unreadable place has no models
        }
        return out;
    }

    // telling a folder apart reads weight headers: kept until the folder changes
    private Found check(Path p) {
        FileTime t;
        try {
            t = Files.getLastModifiedTime(p);
        } catch (IOException e) {
            return new Found(null, false, false, Wan.Input.TEXT);
        }
        Found f = seen.get(p);
        if (f == null || !f.modified().equals(t)) {
            boolean wan = false;
            boolean large = false;
            Wan.Input input = Wan.Input.TEXT;
            // a diffusers folder says what it is; a folder of single files must hold safetensors
            if (Files.isRegularFile(p.resolve("model_index.json")) || hasWeights(p)) {
                try {
                    Wan.Files files = Wan.Files.find(p);
                    wan = true;
                    large = large(files);
                    input = Wan.inputOf(files);
                } catch (IOException | RuntimeException e) {
                    wan = false;
                }
            }
            f = new Found(t, wan, large, input);
            seen.put(p, f);
        }
        return f;
    }

    // Wan 2.2's 16x autoencoder (residual levels, 48 latent channels) marks the 720p 5B model
    private static boolean large(Wan.Files files) {
        if (files.vaeConfig() != null) {
            return Boolean.TRUE.equals(files.vaeConfig().get("is_residual"));
        }
        try (WeightSource w = WeightSource.open(files.vae())) {
            for (String prefix : new String[] {"", "first_stage_model.", "vae."}) {
                for (String n : new String[] {"decoder.conv_in.weight", "decoder.conv1.weight"}) {
                    if (w.contains(prefix + n)) {
                        return w.shape(prefix + n).dim(1) == 48;
                    }
                }
            }
        }
        return false;
    }

    private static boolean hasWeights(Path p) {
        try (Stream<Path> s = Files.list(p)) {
            return s.anyMatch(f -> f.getFileName().toString().endsWith(".safetensors"));
        } catch (IOException e) {
            return false;
        }
    }

    /**
     * Makes a clip.
     *
     * @param id the model's name
     * @param prompt what the clip shows
     * @param s the settings
     * @param listener hears each step and can stop
     * @return the clip, as an animated PNG
     * @throws IOException if the model cannot be loaded
     * @throws java.util.concurrent.CancellationException if the listener stopped it
     */
    Clip make(String id, String prompt, Settings s, Listener listener) throws IOException {
        // one clip at a time; the scheduler may unload the model while none is made
        slot.lock();
        try {
            return makeHolding(id, prompt, s, listener);
        } finally {
            slot.unlock();
        }
    }

    private Clip makeHolding(String id, String prompt, Settings s, Listener listener) throws IOException {
        Path path = find().get(id);
        if (path == null) {
            throw new IOException("no video model named " + id + " on this PC");
        }
        Wan.Input input = check(path).input();
        if (input == Wan.Input.IMAGE && s.start() == null) {
            throw new IOException(id + " makes videos from a picture: choose a start picture");
        }
        if (input == Wan.Input.TEXT && s.start() != null) {
            throw new IOException(id + " makes videos from text only: remove the start picture or choose an"
                    + " image-to-video model");
        }
        GeneratedImage start = s.start() == null ? null : picture(s.start());
        Wan model = load(path);
        VideoRequest request = s.request(prompt, check(path).large(), model.fps());
        GeneratedVideo video = model.generate(request, start, (step, total, preview) -> listener.step(step, total));
        return new Clip(new Media(video.apng(), "image/png"), video.seed(), video.width(), video.height(),
                video.frames().size(), video.fps());
    }

    /** A PNG or JPEG as 8-bit RGB. */
    static GeneratedImage picture(Media m) throws IOException {
        java.awt.image.BufferedImage b = javax.imageio.ImageIO.read(new java.io.ByteArrayInputStream(m.data()));
        if (b == null) {
            throw new IOException("the start picture is not a PNG, JPEG, GIF or BMP image");
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

    private Wan load(Path path) throws IOException {
        Device device = where.device();
        if (loaded != null && path.equals(loadedPath) && device.name().equals(loadedOn)) {
            return loaded;
        }
        makeRoom.run();
        return slot.load(path.getFileName().toString(), path, device, () -> false, () -> {
            DType dtype = device.kind() != Device.Kind.GPU ? DType.FLOAT32
                    : device.supports(DType.BFLOAT16) ? DType.BFLOAT16
                    : device.supports(DType.FLOAT16) ? DType.FLOAT16 : DType.FLOAT32;
            Wan w = Wan.load(path, device, dtype);
            loaded = w;
            loadedPath = path;
            loadedOn = device.name();
            LOG.log(System.Logger.Level.INFO, "loaded " + path.getFileName() + " (" + w + ", " + dtype + ") on "
                    + device);
            return w;
        }, this::let);
    }

    // lets the loaded model go; the slot is held
    private void let() {
        Wan w = loaded;
        loaded = null;
        loadedPath = null;
        loadedOn = null;
        if (w != null) {
            w.close();
        }
    }

    /**
     * Says what a model makes, for the page.
     *
     * @param id the model's name
     * @return {@code {kind, label, steps, guidance, fps, sizes: {aspect: [w, h]}}}, or empty if unknown
     */
    java.util.Optional<Map<String, Object>> describe(String id) {
        Path p = find().get(id);
        if (p == null) {
            return java.util.Optional.empty();
        }
        boolean large = check(p).large();
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("kind", large ? "WAN22_5B" : "WAN");
        // TEXT, TEXT_OR_IMAGE (a start picture is optional) or IMAGE (one is needed)
        m.put("input", String.valueOf(check(p).input()));
        m.put("label", label(large));
        m.put("steps", large ? 40 : 30);
        m.put("guidance", 5.0);
        m.put("fps", large ? Wan.FPS_5B : Wan.FPS);
        Map<String, Object> sizes = new LinkedHashMap<>();
        for (String a : List.of("16:9", "9:16", "1:1", "4:3", "3:4")) {
            int[] wh = (large ? SIZES_720 : SIZES_480).get(a);
            sizes.put(a, List.of(wh[0], wh[1]));
        }
        m.put("sizes", sizes);
        return java.util.Optional.of(m);
    }

    /** Lets the loaded model's memory go. */
    void close() {
        slot.lock();
        try {
            slot.forget();
            let();
        } finally {
            slot.unlock();
        }
    }
}
