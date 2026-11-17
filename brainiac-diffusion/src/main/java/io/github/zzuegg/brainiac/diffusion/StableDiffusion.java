package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.io.HuggingFaceHub;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.SplittableRandom;
import java.util.concurrent.CancellationException;
import java.util.stream.Stream;

/**
 * Text-to-image generation with a pretrained Stable Diffusion model (1.x, 2.x or XL).
 *
 * <pre>{@code
 * try (StableDiffusion sd = StableDiffusion.load(Path.of("stable-diffusion-xl-base-1.0"),
 *         Device.best(), DType.FLOAT16)) {
 *     List<GeneratedImage> images = sd.generate(ImageRequest.of("a red fox in snow").seed(7).build(),
 *             (step, total, preview) -> true);
 *     Files.write(Path.of("fox.png"), images.get(0).png());
 * }
 * }</pre>
 *
 * <p>A model loads from a Hugging Face diffusers folder ({@code model_index.json} with
 * {@code unet}, {@code vae}, {@code text_encoder} and {@code tokenizer} folders) or from a
 * single {@code .safetensors} checkpoint as image tools share them. The UNet and text
 * encoders run in the type given to {@link #load}; the VAE always runs in float32, as SDXL's
 * overflows in float16. One instance is meant to stay loaded and serve many requests, one
 * at a time.
 */
public final class StableDiffusion implements AutoCloseable {

    /** The model families. */
    public enum Kind {
        /** Stable Diffusion 1.x: one CLIP ViT-L/14 text encoder, 512 x 512. */
        SD1,
        /** Stable Diffusion 2.x: OpenCLIP ViT-H, 512 or 768 square (768 models predict velocity). */
        SD2,
        /** Stable Diffusion XL: CLIP ViT-L and OpenCLIP ViT-bigG, size conditioning, 1024 x 1024. */
        SDXL
    }

    private final Kind kind;
    private final Device device;
    private final List<WeightSource> sources;
    private final List<Tensor> weights;
    private final ClipTokenizer tokenizer;
    private final ClipTokenizer tokenizer2;
    private final ClipTextModel textEncoder;
    private final ClipTextModel textEncoder2;
    private final UNet unet;
    private final Vae vae;
    private final NoiseSchedule schedule;
    private final boolean zerosForEmptyNegative;
    private final int nativeSize;

    StableDiffusion(Kind kind, Device device, List<WeightSource> sources, List<Tensor> weights,
            ClipTokenizer tokenizer, ClipTextModel textEncoder, ClipTokenizer tokenizer2, ClipTextModel textEncoder2,
            UNet unet, Vae vae, NoiseSchedule schedule, boolean zerosForEmptyNegative, int nativeSize) {
        this.kind = kind;
        this.device = device;
        this.sources = sources;
        this.weights = weights;
        this.tokenizer = tokenizer;
        this.textEncoder = textEncoder;
        this.tokenizer2 = tokenizer2;
        this.textEncoder2 = textEncoder2;
        this.unet = unet;
        this.vae = vae;
        this.schedule = schedule;
        this.zerosForEmptyNegative = zerosForEmptyNegative;
        this.nativeSize = nativeSize;
    }

    // ------------------------------------------------------------------------------- loading

    /**
     * Says what kind of Stable Diffusion model a folder or file holds, without loading it.
     *
     * @param path a diffusers folder or a single {@code .safetensors} checkpoint
     * @return the kind, or empty if it is not a Stable Diffusion model this library can run
     */
    public static Optional<Kind> detect(Path path) {
        try {
            if (Files.isDirectory(path)) {
                return Files.isRegularFile(path.resolve("unet/config.json"))
                        && Files.isRegularFile(path.resolve("vae/config.json"))
                        && Files.isDirectory(path.resolve("text_encoder"))
                        ? Optional.of(kindOf(Json.object(Json.read(path.resolve("unet/config.json")))))
                        : Optional.empty();
            }
            if (Files.isRegularFile(path) && path.getFileName().toString().endsWith(".safetensors")) {
                try (WeightSource w = WeightSource.open(path)) {
                    return SingleFile.detect(w);
                }
            }
        } catch (IOException | RuntimeException e) {
            return Optional.empty();
        }
        return Optional.empty();
    }

    private static Kind kindOf(Map<String, Object> unetConfig) {
        UNet.Config c = UNet.Config.fromJson(unetConfig);
        if (c.addTimeDim() > 0) {
            return Kind.SDXL;
        }
        return c.contextDim() == 1024 ? Kind.SD2 : Kind.SD1;
    }

    /**
     * Loads a model.
     *
     * @param path a diffusers folder or a single {@code .safetensors} checkpoint
     * @param device where to run it
     * @param dtype the type of the UNet and text encoders: {@link DType#FLOAT16} on a GPU,
     *     {@link DType#FLOAT32} on the CPU
     * @return the loaded model
     * @throws IOException if the files cannot be read
     * @throws IllegalArgumentException if they are not a Stable Diffusion model this library can run
     */
    public static StableDiffusion load(Path path, Device device, DType dtype) throws IOException {
        if (!device.supports(dtype)) {
            throw new IllegalArgumentException(device + " cannot hold " + dtype + " values");
        }
        if (Files.isDirectory(path)) {
            return loadDiffusers(path, device, dtype);
        }
        if (Files.isRegularFile(path)) {
            return SingleFile.load(path, device, dtype);
        }
        throw new IOException("no such model file or folder: " + path);
    }

    private static StableDiffusion loadDiffusers(Path dir, Device device, DType dtype) throws IOException {
        if (!Files.isRegularFile(dir.resolve("unet/config.json"))) {
            throw new IllegalArgumentException(dir + " is not a diffusers Stable Diffusion folder (no unet/config.json)");
        }
        Map<String, Object> index = Files.isRegularFile(dir.resolve("model_index.json"))
                ? Json.object(Json.read(dir.resolve("model_index.json"))) : Map.of();
        Map<String, Object> unetJson = Json.object(Json.read(dir.resolve("unet/config.json")));
        Kind kind = kindOf(unetJson);
        List<WeightSource> sources = new ArrayList<>();
        List<Tensor> all = new ArrayList<>();
        try {
            ClipTextModel enc1 = null;
            ClipTokenizer tok1 = null;
            if (Files.isDirectory(dir.resolve("text_encoder"))) {
                ClipTextModel.Config c = ClipTextModel.Config.fromJson(
                        Json.object(Json.read(dir.resolve("text_encoder/config.json"))));
                enc1 = ClipTextModel.load(weights(dir.resolve("text_encoder"), "model", sources, device, dtype, all),
                        c, c.layers());
                tok1 = ClipTokenizer.load(dir.resolve("tokenizer"), c.positions());
            }
            ClipTextModel enc2 = null;
            ClipTokenizer tok2 = null;
            if (kind == Kind.SDXL) {
                ClipTextModel.Config c = ClipTextModel.Config.fromJson(
                        Json.object(Json.read(dir.resolve("text_encoder_2/config.json"))));
                enc2 = ClipTextModel.load(weights(dir.resolve("text_encoder_2"), "model", sources, device, dtype, all),
                        c, c.layers());
                tok2 = ClipTokenizer.load(dir.resolve("tokenizer_2"), c.positions());
            }
            if (enc1 == null) {
                throw new IllegalArgumentException(dir + " has no text_encoder (SDXL refiners are not supported yet)");
            }
            UNet unet = UNet.load(weights(dir.resolve("unet"), "diffusion_pytorch_model", sources, device, dtype, all),
                    UNet.Config.fromJson(unetJson));
            Vae.Config vc = Vae.Config.fromJson(Json.object(Json.read(dir.resolve("vae/config.json"))));
            Vae vae = Vae.load(weights(dir.resolve("vae"), "diffusion_pytorch_model", sources, device, DType.FLOAT32,
                    all), vc, false);
            Path schedulerConfig = dir.resolve("scheduler/scheduler_config.json");
            NoiseSchedule schedule = Files.isRegularFile(schedulerConfig)
                    ? NoiseSchedule.fromJson(Json.object(Json.read(schedulerConfig)))
                    : NoiseSchedule.stableDiffusion(NoiseSchedule.Prediction.EPSILON);
            int nativeSize = (int) Json.integer(unetJson, "sample_size", 64) * 8;
            boolean zeros = Boolean.TRUE.equals(index.get("force_zeros_for_empty_prompt"))
                    || kind == Kind.SDXL && !index.containsKey("force_zeros_for_empty_prompt");
            return new StableDiffusion(kind, device, sources, all, tok1, enc1, tok2, enc2, unet, vae, schedule, zeros,
                    nativeSize);
        } catch (IOException | RuntimeException e) {
            release(all, sources);
            throw e;
        }
    }

    /** Opens a diffusers component's weights, preferring full precision files over {@code fp16} variants. */
    private static Weights weights(Path dir, String base, List<WeightSource> sources, Device device, DType dtype,
            List<Tensor> all) throws IOException {
        Path file = null;
        for (String name : new String[] {base + ".safetensors", base + ".safetensors.index.json",
                base + ".fp16.safetensors", base + ".fp16.safetensors.index.json"}) {
            if (Files.isRegularFile(dir.resolve(name))) {
                file = dir.resolve(name);
                break;
            }
        }
        if (file == null) {
            try (Stream<Path> files = Files.list(dir)) {
                file = files.filter(p -> p.getFileName().toString().endsWith(".safetensors")).sorted().findFirst()
                        .orElseThrow(() -> new IOException("no .safetensors weights in " + dir
                                + " (PyTorch .bin files are not supported)"));
            }
        }
        WeightSource source = WeightSource.open(file);
        sources.add(source);
        return new Weights(source, "", device, dtype, all);
    }

    private static void release(List<Tensor> all, List<WeightSource> sources) {
        for (Tensor t : all) {
            t.release();
        }
        all.clear();
        for (WeightSource s : sources) {
            try {
                s.close();
            } catch (Exception ignored) {
                // closing a memory map fails only if it is already closed
            }
        }
        sources.clear();
    }

    /**
     * Returns where single-file checkpoints look for the CLIP vocabulary when no
     * {@code tokenizer} folder sits beside them: {@code ~/.cache/brainiac/clip-tokenizer}.
     *
     * @return the folder
     */
    public static Path tokenizerCache() {
        return Path.of(System.getProperty("user.home"), ".cache", "brainiac", "clip-tokenizer");
    }

    /**
     * Downloads the CLIP vocabulary ({@code vocab.json} and {@code merges.txt} of
     * {@code openai/clip-vit-large-patch14}, about 1.6 MB) into {@link #tokenizerCache()},
     * which single-file checkpoints need.
     *
     * @param hub the hub to download from
     * @return the folder
     * @throws IOException if the download fails
     */
    public static Path downloadTokenizer(HuggingFaceHub hub) throws IOException {
        Path dir = tokenizerCache();
        Files.createDirectories(dir);
        for (String f : new String[] {"vocab.json", "merges.txt"}) {
            hub.downloadFile("openai/clip-vit-large-patch14", "main", f, dir);
        }
        return dir;
    }

    // ------------------------------------------------------------------------------- accessors

    /**
     * Returns the model family.
     *
     * @return the kind
     */
    public Kind kind() {
        return kind;
    }

    /**
     * Returns the device the model runs on.
     *
     * @return the device
     */
    public Device device() {
        return device;
    }

    /**
     * Returns the side of the square the model was trained at.
     *
     * @return 512, 768 or 1024
     */
    public int nativeSize() {
        return nativeSize;
    }

    /**
     * Returns a size of the given shape that suits this model.
     *
     * @param ratio the shape
     * @return {@code {width, height}}
     */
    public int[] size(AspectRatio ratio) {
        return ratio.size(nativeSize);
    }

    /**
     * Returns the noise schedule the model was trained with.
     *
     * @return the schedule
     */
    public NoiseSchedule schedule() {
        return schedule;
    }

    // ------------------------------------------------------------------------------- generation

    /** Text conditioning for a batch of {uncond, cond} or {cond}. */
    record Conditioning(Tensor context, Tensor pooled) {
    }

    /**
     * Generates images.
     *
     * @param request what to generate
     * @param progress hears each step and can stop the run
     * @return the images, {@link ImageRequest#count()} of them
     * @throws CancellationException if {@code progress} asked to stop
     */
    public synchronized List<GeneratedImage> generate(ImageRequest request, Progress progress) {
        int width = request.width() == 0 ? nativeSize : request.width();
        int height = request.height() == 0 ? nativeSize : request.height();
        int latentChannels = unet.config().inChannels();
        boolean guided = request.guidance() > 1;
        Conditioning cond = encode(request.prompt(), request.negativePrompt(), guided);
        try {
            List<GeneratedImage> images = new ArrayList<>();
            int[] done = {0};
            int total = request.steps() * request.count();
            for (int i = 0; i < request.count(); i++) {
                long seed = request.seed() + i;
                SplittableRandom random = new SplittableRandom(seed);
                Tensor noise = gaussian(Shape.of(1, latentChannels, height / 8, width / 8), random);
                Tensor latents = denoise(noise, cond, request, random, (step, steps, x0) -> {
                    done[0]++;
                    GeneratedImage preview = null;
                    int every = progress.previewEvery();
                    if (every > 0 && (step % every == 0 || step == steps) && x0 != null) {
                        preview = preview(x0);
                    }
                    return progress.step(done[0], total, preview);
                });
                noise.release();
                Tensor image = decode(latents);
                latents.release();
                images.add(toImage(image, seed));
                image.release();
            }
            return images;
        } finally {
            cond.context().release();
            if (cond.pooled() != null) {
                cond.pooled().release();
            }
        }
    }

    /** Hears each step of {@link #denoise}, with the current estimate of the clean latents. */
    @FunctionalInterface
    interface StepListener {
        boolean step(int step, int steps, Tensor denoised);
    }

    /**
     * Runs the sampler from unit-variance noise to clean latents (as the UNet sees them).
     *
     * @param noise the starting noise {@code [1, channels, h, w]}, standard normal
     * @param cond the text conditioning, {uncond, cond} if guided
     * @param request the steps, guidance and sampler
     * @param random the source of noise for ancestral samplers
     * @param listener hears each step
     * @return the latents
     */
    Tensor denoise(Tensor noise, Conditioning cond, ImageRequest request, SplittableRandom random,
            StepListener listener) {
        Sampler sampler = request.sampler();
        NoiseSchedule.Plan plan = schedule.plan(request.steps(), sampler.karras());
        boolean guided = cond.context().shape().dim(0) == 2;
        int height = noise.shape().dim(2) * 8;
        int width = noise.shape().dim(3) * 8;
        float[][] timeIds = null;
        if (unet.config().addTimeDim() > 0) {
            float[] ids = {height, width, 0, 0, height, width};
            timeIds = guided ? new float[][] {ids, ids} : new float[][] {ids};
        }
        Tensor x = noise.to(DType.FLOAT32).mul((float) plan.initialSigma());
        Tensor previous = null;   // the last step's denoised latents, for DPM-Solver++ (2M)
        double[] sigmas = plan.sigmas();
        List<Tensor> none = List.of();
        for (int i = 0; i < plan.steps(); i++) {
            double sigma = sigmas[i];
            double next = sigmas[i + 1];
            float t = plan.timesteps()[i];
            Tensor xi = x;
            float[][] ids = timeIds;
            Tensor denoised = Nn.scoped(none, () -> {
                Tensor in = xi.mul((float) (1 / Math.sqrt(sigma * sigma + 1)));
                Tensor batch = guided ? Tensor.concat(0, in, in) : in;
                Tensor out = unet.forward(batch, guided ? new float[] {t, t} : new float[] {t}, cond.context(),
                        cond.pooled(), ids).to(DType.FLOAT32);
                if (guided) {
                    Tensor uncond = out.slice(0, 0, 1);
                    Tensor c = out.slice(0, 1, 2);
                    out = uncond.add(c.sub(uncond).mul(request.guidance()));
                }
                if (schedule.prediction() == NoiseSchedule.Prediction.V_PREDICTION) {
                    return xi.mul((float) (1 / (sigma * sigma + 1)))
                            .sub(out.mul((float) (sigma / Math.sqrt(sigma * sigma + 1))));
                }
                return xi.sub(out.mul((float) sigma));
            }, xi);
            Tensor prev = previous;
            double last = i > 0 ? sigmas[i - 1] : 0;
            Tensor nextX = Nn.scoped(none, () -> switch (sampler) {
                case EULER, EULER_KARRAS -> euler(xi, denoised, sigma, next);
                case EULER_ANCESTRAL -> {
                    double up = Math.min(next, Math.sqrt(next * next * (sigma * sigma - next * next) / (sigma * sigma)));
                    double down = Math.sqrt(next * next - up * up);
                    Tensor y = euler(xi, denoised, sigma, down);
                    yield next > 0 ? y.add(gaussian(xi.shape(), random).mul((float) up)) : y;
                }
                case DPM_PP_2M, DPM_PP_2M_KARRAS -> dpmSolver2m(xi, denoised, prev, sigma, next, last);
            }, xi, denoised, prev);
            if (previous != null && previous != nextX) {
                previous.release();
            }
            previous = denoised;
            x.release();
            x = nextX;
            if (!listener.step(i + 1, plan.steps(), denoised)) {
                x.release();
                if (denoised != x) {
                    denoised.release();
                }
                throw new CancellationException("image generation stopped at step " + (i + 1));
            }
        }
        // DPM-Solver++ ends on the denoised latents themselves
        if (previous != null && previous != x) {
            previous.release();
        }
        return x;
    }

    /** Decodes latents into images {@code [batch, 3, h, w]} with values in {@code [0, 1]}. */
    Tensor decode(Tensor latents) {
        Tensor image = vae.decode(latents);
        Tensor out = Nn.scoped(List.of(), () -> image.mul(0.5f).add(0.5f)
                .maximum(Tensor.full(Shape.of(1), 0f, image.device()))
                .minimum(Tensor.full(Shape.of(1), 1f, image.device())), image);
        image.release();
        return out;
    }

    private static Tensor euler(Tensor x, Tensor denoised, double sigma, double next) {
        // x + (next - sigma) * (x - denoised) / sigma
        return denoised.add(x.sub(denoised).mul((float) (next / sigma)));
    }

    private static Tensor dpmSolver2m(Tensor x, Tensor denoised, Tensor previous, double sigma, double next,
            double last) {
        if (next == 0) {
            return denoised;
        }
        double t = -Math.log(sigma);
        double tNext = -Math.log(next);
        double h = tNext - t;
        Tensor d = denoised;
        if (previous != null) {
            double hLast = t + Math.log(last);
            double r = hLast / h;
            d = denoised.mul((float) (1 + 1 / (2 * r))).sub(previous.mul((float) (1 / (2 * r))));
        }
        return x.mul((float) (next / sigma)).sub(d.mul((float) Math.expm1(-h)));
    }

    /** Encodes the prompts: {uncond, cond} if guided, else {cond}. */
    Conditioning encode(String prompt, String negative, boolean guided) {
        String[] texts = guided ? new String[] {negative, prompt} : new String[] {prompt};
        boolean zeroUncond = guided && zerosForEmptyNegative && negative.isBlank();
        if (kind != Kind.SDXL) {
            ClipTextModel.Output o = textEncoder.encode(ids(tokenizer, texts));
            release(o.penultimate());
            release(o.pooled());
            return new Conditioning(o.last(), null);
        }
        ClipTextModel.Output a = textEncoder.encode(ids(tokenizer, texts));
        ClipTextModel.Output b = textEncoder2.encode(ids(tokenizer2, texts));
        Tensor context = Tensor.concat(2, a.penultimate().to(unet.dtype()), b.penultimate().to(unet.dtype()));
        Tensor pooled = b.pooled().to(DType.FLOAT32);
        if (zeroUncond) {
            Tensor mask = Tensor.of(Shape.of(2, 1, 1), new float[] {0, 1}, device).to(context.dtype());
            context = context.mul(mask);
            pooled = pooled.mul(Tensor.of(Shape.of(2, 1), new float[] {0, 1}, device));
        }
        Tensor.evaluate(List.of(context, pooled));
        for (ClipTextModel.Output o : new ClipTextModel.Output[] {a, b}) {
            for (Tensor t : new Tensor[] {o.last(), o.penultimate(), o.pooled()}) {
                // a conversion to the same type returns the tensor itself
                if (t != null && t != context && t != pooled) {
                    t.release();
                }
            }
        }
        return new Conditioning(context, pooled);
    }

    private static void release(Tensor t) {
        if (t != null) {
            t.release();
        }
    }

    private static int[][] ids(ClipTokenizer tokenizer, String[] texts) {
        int[][] ids = new int[texts.length][];
        for (int i = 0; i < texts.length; i++) {
            ids[i] = tokenizer.encode(texts[i]);
        }
        return ids;
    }

    private Tensor gaussian(Shape shape, SplittableRandom random) {
        float[] v = new float[Math.toIntExact(shape.size())];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) random.nextGaussian();
        }
        return Tensor.of(shape, v, device);
    }

    private static GeneratedImage toImage(Tensor image, long seed) {
        Shape s = image.shape();
        int h = s.dim(2);
        int w = s.dim(3);
        float[] v = image.toArray();
        byte[] rgb = new byte[w * h * 3];
        int plane = w * h;
        for (int p = 0; p < plane; p++) {
            for (int c = 0; c < 3; c++) {
                rgb[p * 3 + c] = (byte) Math.round(v[c * plane + p] * 255);
            }
        }
        return new GeneratedImage(w, h, rgb, seed);
    }

    /** Latent-to-RGB factors for quick previews (approximate linear fits of the VAE decoder). */
    private static final float[][] SD_PREVIEW = {
            {0.3512f, 0.2297f, 0.3227f}, {0.3250f, 0.4974f, 0.2350f},
            {-0.2829f, 0.1762f, 0.2721f}, {-0.2120f, -0.2616f, -0.7177f}};
    private static final float[][] SDXL_PREVIEW = {
            {0.3651f, 0.4232f, 0.4341f}, {-0.2533f, -0.0042f, 0.1068f},
            {0.1076f, 0.1111f, -0.0362f}, {-0.3165f, -0.2492f, -0.2188f}};
    private static final float[] SDXL_PREVIEW_BIAS = {0.1084f, -0.0175f, -0.0011f};

    private GeneratedImage preview(Tensor latents) {
        Shape s = latents.shape();
        int c = s.dim(1);
        int h = s.dim(2);
        int w = s.dim(3);
        float[] v = latents.toArray();
        float[][] factors = kind == Kind.SDXL ? SDXL_PREVIEW : SD_PREVIEW;
        float[] bias = kind == Kind.SDXL ? SDXL_PREVIEW_BIAS : new float[3];
        byte[] rgb = new byte[w * h * 3];
        int plane = w * h;
        for (int p = 0; p < plane; p++) {
            for (int k = 0; k < 3; k++) {
                float f = bias[k];
                for (int ch = 0; ch < Math.min(c, 4); ch++) {
                    f += v[ch * plane + p] * factors[ch][k];
                }
                rgb[p * 3 + k] = (byte) Math.round(Math.max(0, Math.min(1, (f + 1) / 2)) * 255);
            }
        }
        return new GeneratedImage(w, h, rgb, -1);
    }

    /**
     * Frees the model's device memory and closes its files.
     */
    @Override
    public synchronized void close() {
        release(weights, sources);
    }

    @Override
    public String toString() {
        return "StableDiffusion[" + kind + ", " + nativeSize + "px, " + unet.dtype() + " on " + device + "]";
    }
}
