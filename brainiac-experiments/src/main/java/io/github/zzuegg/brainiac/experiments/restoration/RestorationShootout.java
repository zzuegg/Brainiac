package io.github.zzuegg.brainiac.experiments.restoration;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.DegradedImages;
import io.github.zzuegg.brainiac.experimental.data.GlyphDocuments;
import io.github.zzuegg.brainiac.experimental.document.LineRecognizer;
import io.github.zzuegg.brainiac.experimental.nn.GaussianSplats2d;
import io.github.zzuegg.brainiac.experimental.nn.UNet2d;
import io.github.zzuegg.brainiac.experimental.restore.Denoisers;
import io.github.zzuegg.brainiac.experimental.restore.ImageDenoiser;
import io.github.zzuegg.brainiac.experimental.restore.PlugAndPlay;
import io.github.zzuegg.brainiac.experimental.restore.SelfSupervisedDenoising;
import io.github.zzuegg.brainiac.experimental.restore.SuperResolution;
import io.github.zzuegg.brainiac.experiments.generative.NeuralFields;
import io.github.zzuegg.brainiac.experiments.generative.ToyImages;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Image restoration on {@link DegradedImages} (wave 10, items 10.6.1-10.6.4):
 *
 * <ul>
 *   <li>{@link #superResolution} (10.6.1): SRCNN, ESPCN, EDSR and SPAN
 *       ({@link SuperResolution}) against nearest and bicubic upsampling, a U-Net
 *       ({@link UNet2d}) on the bicubic image, and a per-image Fourier-feature neural field
 *       ({@link NeuralFields#coordinateMlp}) fitted to the low-resolution pixels and read at the
 *       high-resolution grid; {@link #glyphOcr} adds the character error rate of a
 *       {@link LineRecognizer} trained on clean lines when it reads upsampled lines;</li>
 *   <li>{@link #denoising} (10.6.2): DnCNN with residual learning against the same network
 *       predicting the clean image, FFDNet with the true noise map (also on spatially varying
 *       noise), the U-Net, a diffusion-style noise-conditional U-Net used in one step, and
 *       Gaussian, median and non-local-means filtering;</li>
 *   <li>{@link #selfSupervised} (10.6.3): Noise2Noise, Noise2Void, Neighbor2Neighbor,
 *       zero-shot Noise2Noise and Median2Median ({@link SelfSupervisedDenoising}) against the
 *       supervised DnCNN ceiling and the filters, on i.i.d. Gaussian, Poisson-Gaussian and
 *       spatially correlated noise;</li>
 *   <li>{@link #deblurring} (10.6.4): plug-and-play HQS and ADMM with the FFDNet denoiser and a
 *       deep-unrolled network ({@link PlugAndPlay}) against Wiener, Richardson-Lucy, TV and a
 *       U-Net trained directly on blurred-to-sharp pairs, on seen and unseen kernels.</li>
 * </ul>
 * Scores are PSNR (dB) and SSIM against the clean image.
 */
public final class RestorationShootout {

    private RestorationShootout() {
    }

    /**
     * One method's score.
     *
     * @param method the method
     * @param psnr mean PSNR in dB
     * @param ssim mean SSIM
     * @param parameters trainable scalars (0 for untrained methods)
     * @param macs multiply-adds per image (0 when not counted)
     */
    public record Row(String method, double psnr, double ssim, long parameters, long macs) {
        @Override
        public String toString() {
            return String.format("%s %.2f dB/%.3f%s", method, psnr, ssim,
                    parameters > 0 ? String.format(" (%d params%s)", parameters,
                            macs > 0 ? String.format(", %.2fM MACs", macs / 1e6) : "") : "");
        }
    }

    /** The clean image sources. */
    public enum Source {
        /** Piecewise-smooth scenes. */
        SCENES,
        /** {@code ShapeImages} shapes. */
        SHAPES,
        /** Glyph-line crops. */
        GLYPHS,
        /** {@link ToyImages} enlarged with nearest-neighbour copies. */
        TOY
    }

    /**
     * Clean images of a source.
     *
     * @param source the source
     * @param n how many
     * @param side their side
     * @param seed the seed
     * @return the images
     */
    public static List<float[]> clean(Source source, int n, int side, long seed) {
        return switch (source) {
            case SCENES -> DegradedImages.scenes(n, side, seed);
            case SHAPES -> DegradedImages.shapes(n, side, seed);
            case GLYPHS -> DegradedImages.glyphCrops(n, side, seed);
            case TOY -> {
                SplittableRandom r = new SplittableRandom(seed);
                List<float[]> out = new ArrayList<>();
                int scale = Math.max(1, side / ToyImages.SIDE);
                for (int i = 0; i < n; i++) {
                    float[] t = ToyImages.sample(r.nextInt(ToyImages.CLASSES), r);
                    float[] img = new float[side * side];
                    for (int y = 0; y < side; y++) {
                        for (int x = 0; x < side; x++) {
                            int ty = Math.min(ToyImages.SIDE - 1, y / scale);
                            int tx = Math.min(ToyImages.SIDE - 1, x / scale);
                            img[y * side + x] = 0.5f + 0.35f * t[ty * ToyImages.SIDE + tx];
                        }
                    }
                    out.add(img);
                }
                yield out;
            }
        };
    }

    // ---------------------------------------------------------------- helpers

    static void train(Function<Var, Var> net, List<Var> params, Supplier<Tensor[]> batches, int steps, float lr) {
        Optimizer opt = Optimizer.adam(lr);
        for (int s = 0; s < steps; s++) {
            Tensor[] b = batches.get();
            Var loss = net.apply(Var.of(b[0])).sub(b[1]).square().mean();
            loss.backward();
            opt.step(params);
        }
    }

    static List<float[]> apply(Function<Var, Var> net, List<float[]> inputs, int side) {
        Tensor out = Var.noGrad(() -> net.apply(Var.of(DegradedImages.stack(inputs, side)))).value();
        return DegradedImages.unstack(out);
    }

    static Row score(String method, List<float[]> est, List<float[]> clean, int side, long params, long macs) {
        return new Row(method, DegradedImages.meanPsnr(est, clean), DegradedImages.meanSsim(est, clean, side),
                params, macs);
    }

    static long count(List<Var> params) {
        long n = 0;
        for (Var v : params) {
            n += v.shape().size();
        }
        return n;
    }

    private static Supplier<Tensor[]> pairs(List<float[]> inputs, int inSide, List<float[]> targets, int outSide,
            int batch, SplittableRandom r) {
        return () -> {
            List<float[]> a = new ArrayList<>();
            List<float[]> b = new ArrayList<>();
            for (int i = 0; i < batch; i++) {
                int k = r.nextInt(inputs.size());
                a.add(inputs.get(k));
                b.add(targets.get(k));
            }
            return new Tensor[] {DegradedImages.stack(a, inSide), DegradedImages.stack(b, outSide)};
        };
    }

    // ---------------------------------------------------------------- 10.6.1 super-resolution

    /** Super-resolution methods. */
    public enum SrMethod {
        /** Nearest-neighbour upsampling. */
        NEAREST,
        /** Bicubic upsampling. */
        BICUBIC,
        /** A per-image Fourier-feature coordinate MLP. */
        NEURAL_FIELD,
        /** U-Net on the bicubic image (residual). */
        UNET,
        /** SRCNN. */
        SRCNN,
        /** ESPCN. */
        ESPCN,
        /** EDSR. */
        EDSR,
        /** SPAN. */
        SPAN
    }

    /**
     * Super-resolution of area-downsampled images.
     *
     * @param source the clean images
     * @param side the high-resolution side
     * @param factor 2 or 4
     * @param steps training steps (batch 8)
     * @param testImages test images
     * @param seed the seed
     * @param methods the methods
     * @return one row per method
     */
    public static List<Row> superResolution(Source source, int side, int factor, int steps, int testImages, long seed,
            SrMethod... methods) {
        int low = side / factor;
        List<float[]> trainHr = clean(source, 256, side, seed);
        List<float[]> testHr = clean(source, testImages, side, seed + 1000);
        List<float[]> trainLr = new ArrayList<>();
        for (float[] im : trainHr) {
            trainLr.add(DegradedImages.downsample(im, side, factor));
        }
        List<float[]> testLr = new ArrayList<>();
        for (float[] im : testHr) {
            testLr.add(DegradedImages.downsample(im, side, factor));
        }
        List<Row> rows = new ArrayList<>();
        for (SrMethod m : methods) {
            SplittableRandom r = new SplittableRandom(seed + m.ordinal());
            switch (m) {
                case NEAREST, BICUBIC -> {
                    Functional.Interpolation in = m == SrMethod.NEAREST ? Functional.Interpolation.NEAREST
                            : Functional.Interpolation.CUBIC;
                    rows.add(score(m.name(), apply(x -> DegradedImages.upsample(x, factor, in), testLr, low), testHr,
                            side, 0, 0));
                }
                case NEURAL_FIELD -> {
                    List<float[]> est = new ArrayList<>();
                    for (int i = 0; i < testLr.size(); i++) {
                        est.add(neuralFieldUpsample(testLr.get(i), low, side, Math.max(50, steps), seed + i));
                    }
                    rows.add(score("neural field", est, testHr, side, 0, 0));
                }
                case UNET -> {
                    UNet2d unet = new UNet2d(1, 1, 8, 2, true, 0, r);
                    Function<Var, Var> net = x -> {
                        Var up = DegradedImages.upsample(x, factor, Functional.Interpolation.CUBIC);
                        return up.add(unet.forward(up));
                    };
                    train(net, unet.parameters(), pairs(trainLr, low, trainHr, side, 8, r), steps, 2e-3f);
                    rows.add(score("U-Net", apply(net, testLr, low), testHr, side, count(unet.parameters()), 0));
                }
                default -> {
                    SuperResolution.Kind kind = SuperResolution.Kind.valueOf(m.name());
                    int blocks = kind == SuperResolution.Kind.SPAN ? 2 : 3;
                    SuperResolution sr = new SuperResolution(kind, factor, 16, blocks, true, r);
                    train(sr::forward, sr.parameters(), pairs(trainLr, low, trainHr, side, 8, r), steps, 2e-3f);
                    rows.add(score(kind.name(), apply(sr::forward, testLr, low), testHr, side, sr.parameterCount(),
                            sr.macs(low, low)));
                }
            }
        }
        return rows;
    }

    /**
     * Fits a Fourier-feature coordinate MLP to the low-resolution pixels and evaluates it on
     * the high-resolution grid (both on pixel-centre coordinates in [0, 1]).
     *
     * @param lowRes the low-resolution image
     * @param low its side
     * @param side the target side
     * @param steps Adam steps
     * @param seed the seed
     * @return the upsampled image
     */
    public static float[] neuralFieldUpsample(float[] lowRes, int low, int side, int steps, long seed) {
        Model net = NeuralFields.coordinateMlp(16, 3f, seed);
        NeuralFields.fit(net, lowRes, low, steps, 5e-3f);
        return Var.noGrad(() -> net.forward(Var.of(GaussianSplats2d.grid(side, side)), false)).value().toArray();
    }

    /**
     * OCR on upsampled text lines (10.6.1): a {@link LineRecognizer} is trained on clean
     * 16-pixel-high lines (glyphs at pixel scale 2), an ESPCN and an EDSR on line pairs, and the
     * recogniser then reads the bicubic, ESPCN and EDSR reconstructions of {@code factor}-times
     * downsampled test lines, and the clean lines.
     *
     * @param factor 2 or 4
     * @param srSteps super-resolution training steps
     * @param ocrSteps recogniser training steps (batch 16)
     * @param seed the seed
     * @return character error rate per input ("clean", "bicubic", "ESPCN", "EDSR")
     */
    public static Map<String, Double> glyphOcr(int factor, int srSteps, int ocrSteps, long seed) {
        int h = 16;
        int w = 64;
        List<DegradedImages.GlyphLine> train = DegradedImages.glyphLines(400, 2, h, w, 6, seed);
        List<DegradedImages.GlyphLine> test = DegradedImages.glyphLines(48, 2, h, w, 6, seed + 1);
        LineRecognizer ocr = new LineRecognizer(h, GlyphDocuments.CHARSET.length() + 1, 32,
                LineRecognizer.Encoder.BIGRU, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < ocrSteps; s++) {
            List<DegradedImages.GlyphLine> b = new ArrayList<>();
            for (int i = 0; i < 16; i++) {
                b.add(train.get(r.nextInt(train.size())));
            }
            Var loss = ocr.loss(lines(b, h, w), labels(b));
            loss.backward();
            opt.step(ocr.parameters());
        }
        Map<String, Double> out = new LinkedHashMap<>();
        Tensor cleanTest = lines(test, h, w);
        out.put("clean", cer(ocr.read(cleanTest, GlyphDocuments.CHARSET), test));
        Tensor lowTest = downLines(cleanTest, factor);
        Tensor bic = Var.noGrad(() -> DegradedImages.upsample(Var.of(lowTest), factor,
                Functional.Interpolation.CUBIC)).value();
        out.put("bicubic", cer(ocr.read(bic, GlyphDocuments.CHARSET), test));
        Tensor lowTrain = downLines(lines(train, h, w), factor);
        Tensor hiTrain = lines(train, h, w);
        for (SuperResolution.Kind kind : new SuperResolution.Kind[] {SuperResolution.Kind.ESPCN,
            SuperResolution.Kind.EDSR}) {
            SuperResolution sr = new SuperResolution(kind, factor, 16, 3, true, new SplittableRandom(seed + 7));
            Optimizer so = Optimizer.adam(2e-3f);
            int n = train.size();
            for (int s = 0; s < srSteps; s++) {
                int i0 = r.nextInt(n - 8);
                Var loss = sr.forward(Var.of(lowTrain.slice(0, i0, i0 + 8))).sub(hiTrain.slice(0, i0, i0 + 8)).square()
                        .mean();
                loss.backward();
                so.step(sr.parameters());
            }
            Tensor up = Var.noGrad(() -> sr.forward(Var.of(lowTest))).value();
            out.put(kind.name(), cer(ocr.read(up, GlyphDocuments.CHARSET), test));
        }
        return out;
    }

    private static Tensor lines(List<DegradedImages.GlyphLine> b, int h, int w) {
        float[] x = new float[b.size() * h * w];
        for (int i = 0; i < b.size(); i++) {
            System.arraycopy(b.get(i).pixels(), 0, x, i * h * w, h * w);
        }
        return Tensor.of(Shape.of(b.size(), 1, h, w), x);
    }

    private static int[][] labels(List<DegradedImages.GlyphLine> b) {
        int[][] l = new int[b.size()][];
        for (int i = 0; i < l.length; i++) {
            l[i] = b.get(i).labels();
        }
        return l;
    }

    private static Tensor downLines(Tensor lines, int factor) {
        return Var.noGrad(() -> Functional.pool(Var.of(lines), Functional.PoolKind.AVG, new int[] {factor, factor},
                new int[] {factor, factor}, new int[] {0, 0}, new int[] {0, 0}, new int[] {1, 1}, false, false, 1f))
                .value();
    }

    private static double cer(String[] read, List<DegradedImages.GlyphLine> truth) {
        double errors = 0;
        double total = 0;
        for (int i = 0; i < read.length; i++) {
            errors += LineRecognizer.editDistance(read[i], truth.get(i).text());
            total += truth.get(i).text().length();
        }
        return errors / total;
    }

    // ---------------------------------------------------------------- 10.6.2 denoising

    /** Supervised denoising methods. */
    public enum DenoiseMethod {
        /** Gaussian filter (sigma tuned per level). */
        GAUSSIAN,
        /** 3x3 median filter. */
        MEDIAN,
        /** Non-local means. */
        NLM,
        /** DnCNN predicting the clean image. */
        DNCNN_DIRECT,
        /** DnCNN predicting the noise (residual learning). */
        DNCNN,
        /** FFDNet with the true noise-level map. */
        FFDNET,
        /** U-Net (residual), blind. */
        UNET,
        /** A noise-conditional (diffusion-trained) U-Net used as a one-step denoiser. */
        DIFFUSION_ONE_STEP
    }

    /**
     * Supervised denoising at noise levels 10, 25 and 50 (of 255) and on spatially varying
     * noise (5 to 50). The networks are trained once, blind, on levels drawn uniformly from
     * 0 to 55 (FFDNet and the diffusion U-Net also see the level).
     *
     * @param side the image side
     * @param steps training steps (batch 8)
     * @param testImages test images per condition
     * @param seed the seed
     * @param methods the methods
     * @return per condition ("10", "25", "50", "varying") the rows in method order
     */
    public static Map<String, List<Row>> denoising(int side, int steps, int testImages, long seed,
            DenoiseMethod... methods) {
        List<float[]> train = clean(Source.SCENES, 256, side, seed);
        List<float[]> test = clean(Source.SCENES, testImages, side, seed + 1000);
        Map<String, List<float[][]>> conditions = new LinkedHashMap<>();
        SplittableRandom tr = new SplittableRandom(seed + 2000);
        for (int level : new int[] {10, 25, 50}) {
            List<float[][]> set = new ArrayList<>();
            for (float[] c : test) {
                float[] map = new float[c.length];
                java.util.Arrays.fill(map, level / 255f);
                set.add(new float[][] {DegradedImages.gaussian(c, level / 255.0, tr), map});
            }
            conditions.put(Integer.toString(level), set);
        }
        List<float[][]> varying = new ArrayList<>();
        for (float[] c : test) {
            DegradedImages.NoisyImage n = DegradedImages.spatiallyVarying(c, side, 5 / 255.0, 50 / 255.0, tr);
            varying.add(new float[][] {n.image(), n.sigmaMap()});
        }
        conditions.put("varying", varying);
        Map<String, List<Row>> out = new LinkedHashMap<>();
        for (String k : conditions.keySet()) {
            out.put(k, new ArrayList<>());
        }
        for (DenoiseMethod m : methods) {
            SplittableRandom r = new SplittableRandom(seed + 31L * m.ordinal());
            NoiseConditioned net = switch (m) {
                case DNCNN, DNCNN_DIRECT -> {
                    Denoisers.DnCnn d = new Denoisers.DnCnn(5, 16, m == DenoiseMethod.DNCNN, r);
                    yield new NoiseConditioned(m.name(), (x, map) -> d.forward(x, false), (x, map) -> d.forward(x, true),
                            d.parameters());
                }
                case FFDNET -> {
                    Denoisers.FfdNet f = new Denoisers.FfdNet(5, 24, true, r);
                    yield new NoiseConditioned("FFDNet", (x, map) -> f.forward(x, map, false),
                            (x, map) -> f.forward(x, map, true), f.parameters());
                }
                case UNET -> {
                    UNet2d u = new UNet2d(1, 1, 8, 2, true, 0, r);
                    yield new NoiseConditioned("U-Net", (x, map) -> x.sub(u.forward(x)), (x, map) -> x.sub(u.forward(x)),
                            u.parameters());
                }
                case DIFFUSION_ONE_STEP -> {
                    UNet2d u = new UNet2d(1, 1, 8, 2, true, 1, r);
                    // epsilon prediction conditioned on sigma; one step x0 = x - sigma * eps
                    yield new NoiseConditioned("diffusion one-step", (x, map) -> oneStep(u, x, map),
                            (x, map) -> oneStep(u, x, map), u.parameters());
                }
                default -> null;
            };
            if (net != null) {
                trainDenoiser(net, train, side, steps, r);
            }
            for (Map.Entry<String, List<float[][]>> e : conditions.entrySet()) {
                List<float[]> est = new ArrayList<>();
                List<float[][]> set = e.getValue();
                if (net != null) {
                    List<float[]> noisy = new ArrayList<>();
                    List<float[]> maps = new ArrayList<>();
                    for (float[][] p : set) {
                        noisy.add(p[0]);
                        maps.add(p[1]);
                    }
                    Tensor x = DegradedImages.stack(noisy, side);
                    Tensor map = DegradedImages.stack(maps, side);
                    est = DegradedImages.unstack(Var.noGrad(() -> net.eval().apply(Var.of(x), Var.of(map))).value());
                } else {
                    for (float[][] p : set) {
                        float sigma = mean(p[1]);
                        est.add(switch (m) {
                            case GAUSSIAN -> Denoisers.gaussianFilter(p[0], side, 0.4 + 12 * sigma);
                            case MEDIAN -> Denoisers.medianFilter(p[0], side, 1);
                            default -> Denoisers.nonLocalMeans(p[0], side, sigma, 3, 0.6 * sigma + 1e-3);
                        });
                    }
                }
                out.get(e.getKey()).add(score(net != null ? net.name() : m.name(), est, test, side,
                        net != null ? count(net.params()) : 0, 0));
            }
        }
        return out;
    }

    private static Var oneStep(UNet2d u, Var x, Var map) {
        int b = x.shape().dim(0);
        Var sigma = map.mean(2, false).mean(2, false).reshape(b, 1);
        Var eps = u.forward(x, sigma.mul(10f));
        return x.sub(eps.mul(map));
    }

    private static float mean(float[] v) {
        double s = 0;
        for (float f : v) {
            s += f;
        }
        return (float) (s / v.length);
    }

    /** A denoiser taking the noisy batch and its noise map, in eval and train mode. */
    record NoiseConditioned(String name, java.util.function.BiFunction<Var, Var, Var> eval,
            java.util.function.BiFunction<Var, Var, Var> trainForward, List<Var> params) {
    }

    private static void trainDenoiser(NoiseConditioned net, List<float[]> train, int side, int steps,
            SplittableRandom r) {
        Optimizer opt = Optimizer.adam(2e-3f);
        for (int s = 0; s < steps; s++) {
            List<float[]> c = new ArrayList<>();
            List<float[]> n = new ArrayList<>();
            List<float[]> maps = new ArrayList<>();
            for (int i = 0; i < 8; i++) {
                float[] img = train.get(r.nextInt(train.size()));
                c.add(img);
                if (r.nextInt(4) == 0) {
                    DegradedImages.NoisyImage v = DegradedImages.spatiallyVarying(img, side, 5 / 255.0, 50 / 255.0, r);
                    n.add(v.image());
                    maps.add(v.sigmaMap());
                } else {
                    double sigma = r.nextDouble() * 55 / 255.0;
                    n.add(DegradedImages.gaussian(img, sigma, r));
                    float[] map = new float[img.length];
                    java.util.Arrays.fill(map, (float) sigma);
                    maps.add(map);
                }
            }
            Var est = net.trainForward().apply(Var.of(DegradedImages.stack(n, side)),
                    Var.of(DegradedImages.stack(maps, side)));
            est.sub(DegradedImages.stack(c, side)).square().mean().backward();
            opt.step(net.params());
        }
    }

    // ---------------------------------------------------------------- 10.6.3 self-supervised

    /** Noise types for the self-supervised comparison. */
    public enum NoiseKind {
        /** i.i.d. Gaussian, sigma 25/255. */
        GAUSSIAN,
        /** Poisson-Gaussian, peak 30, read noise 5/255. */
        POISSON_GAUSSIAN,
        /** Spatially correlated Gaussian, sigma 25/255, correlation length 1 pixel. */
        CORRELATED
    }

    /** Self-supervised denoising methods. */
    public enum SelfMethod {
        /** Gaussian filter. */
        GAUSSIAN_FILTER,
        /** 3x3 median filter. */
        MEDIAN,
        /** Non-local means. */
        NLM,
        /** DnCNN trained on clean targets. */
        SUPERVISED,
        /** Noise2Noise (two noisy copies). */
        N2N,
        /** Noise2Void (blind spot). */
        N2V,
        /** Neighbor2Neighbor. */
        NE2NE,
        /** Zero-shot Noise2Noise (per test image). */
        ZS_N2N,
        /** Median2Median (per test image). */
        M2M
    }

    static float[] corrupt(float[] clean, NoiseKind kind, int side, SplittableRandom r) {
        return switch (kind) {
            case GAUSSIAN -> DegradedImages.gaussian(clean, 25 / 255.0, r);
            case POISSON_GAUSSIAN -> DegradedImages.poissonGaussian(clean, 30, 5 / 255.0, r);
            case CORRELATED -> DegradedImages.correlated(clean, side, 25 / 255.0, 1.0, r);
        };
    }

    /**
     * Denoising without clean targets. The dataset methods train a residual DnCNN (5 layers,
     * 16 channels) on noisy training images; the zero-shot methods train per test image.
     *
     * @param kind the noise
     * @param side the image side
     * @param steps training steps (batch 8) for the dataset methods
     * @param zeroShotSteps per-image steps for ZS-N2N and M2M
     * @param testImages test images
     * @param seed the seed
     * @param methods the methods
     * @return one row per method; per-image methods report their mean milliseconds per image as
     *     {@code macs}
     */
    public static List<Row> selfSupervised(NoiseKind kind, int side, int steps, int zeroShotSteps, int testImages,
            long seed, SelfMethod... methods) {
        List<float[]> train = clean(Source.SCENES, 256, side, seed);
        List<float[]> test = clean(Source.SCENES, testImages, side, seed + 1000);
        SplittableRandom tr = new SplittableRandom(seed + 2000);
        List<float[]> noisyTest = new ArrayList<>();
        for (float[] c : test) {
            noisyTest.add(corrupt(c, kind, side, tr));
        }
        double sigma = 25 / 255.0;
        List<Row> rows = new ArrayList<>();
        for (SelfMethod m : methods) {
            SplittableRandom r = new SplittableRandom(seed + 17L * m.ordinal());
            switch (m) {
                case GAUSSIAN_FILTER, MEDIAN, NLM -> {
                    List<float[]> est = new ArrayList<>();
                    for (float[] y : noisyTest) {
                        est.add(switch (m) {
                            case GAUSSIAN_FILTER -> Denoisers.gaussianFilter(y, side, 0.9);
                            case MEDIAN -> Denoisers.medianFilter(y, side, 1);
                            default -> Denoisers.nonLocalMeans(y, side, sigma, 3, 0.6 * sigma);
                        });
                    }
                    rows.add(score(m.name(), est, test, side, 0, 0));
                }
                case ZS_N2N, M2M -> {
                    List<float[]> est = new ArrayList<>();
                    long t0 = System.nanoTime();
                    for (int i = 0; i < noisyTest.size(); i++) {
                        est.add(m == SelfMethod.ZS_N2N
                                ? SelfSupervisedDenoising.zeroShotN2N(noisyTest.get(i), side, 24, zeroShotSteps, seed + i)
                                : SelfSupervisedDenoising.median2Median(noisyTest.get(i), side, 24, zeroShotSteps,
                                        seed + i));
                    }
                    long ms = (System.nanoTime() - t0) / 1_000_000 / noisyTest.size();
                    rows.add(score(m.name(), est, test, side, 0, ms));
                }
                default -> {
                    Denoisers.DnCnn net = new Denoisers.DnCnn(5, 16, true, r);
                    Function<Var, Var> fwd = x -> net.forward(x, true);
                    Optimizer opt = Optimizer.adam(2e-3f);
                    for (int s = 0; s < steps; s++) {
                        List<float[]> c = new ArrayList<>();
                        List<float[]> a = new ArrayList<>();
                        List<float[]> b = new ArrayList<>();
                        for (int i = 0; i < 8; i++) {
                            float[] img = train.get(r.nextInt(train.size()));
                            c.add(img);
                            a.add(corrupt(img, kind, side, r));
                            b.add(corrupt(img, kind, side, r));
                        }
                        Tensor ta = DegradedImages.stack(a, side);
                        Var loss = switch (m) {
                            case SUPERVISED -> fwd.apply(Var.of(ta)).sub(DegradedImages.stack(c, side)).square().mean();
                            case N2N -> SelfSupervisedDenoising.noise2NoiseLoss(fwd, ta, DegradedImages.stack(b, side));
                            case N2V -> SelfSupervisedDenoising.blindSpotLoss(fwd, ta, 0.05, r);
                            default -> SelfSupervisedDenoising.neighbor2NeighborLoss(fwd, ta, 1f, r);
                        };
                        loss.backward();
                        opt.step(net.parameters());
                    }
                    rows.add(score(m.name(), apply(x -> net.forward(x, false), noisyTest, side), test, side,
                            count(net.parameters()), 0));
                }
            }
        }
        return rows;
    }

    // ---------------------------------------------------------------- 10.6.4 deblurring

    /** Deblurring methods. */
    public enum DeblurMethod {
        /** The blurred input. */
        BLURRED,
        /** Wiener deconvolution. */
        WIENER,
        /** Richardson-Lucy, 30 iterations. */
        RICHARDSON_LUCY,
        /** TV-regularised deconvolution. */
        TV,
        /** U-Net trained on blurred-to-sharp pairs (blind to the kernel). */
        UNET,
        /** Plug-and-play HQS with FFDNet. */
        PNP_HQS,
        /** Plug-and-play ADMM with FFDNet. */
        PNP_ADMM,
        /** Deep-unrolled HQS. */
        UNROLLED
    }

    /** A blur kernel. */
    record Kernel(String name, float[] weights, int size) {
    }

    static List<Kernel> trainingKernels() {
        return List.of(new Kernel("gauss1.0", DegradedImages.gaussianKernel(7, 1.0), 7),
                new Kernel("gauss1.6", DegradedImages.gaussianKernel(7, 1.6), 7),
                new Kernel("motion5-0", DegradedImages.motionKernel(7, 5, 0), 7),
                new Kernel("motion5-90", DegradedImages.motionKernel(7, 5, Math.PI / 2), 7));
    }

    static Kernel unseenKernel() {
        return new Kernel("motion6-45 (unseen)", DegradedImages.motionKernel(7, 6, Math.PI / 4), 7);
    }

    /**
     * Non-blind deblurring with known kernels. FFDNet (the PnP prior) is trained on denoising
     * only; the U-Net and the unrolled network are trained on the four training kernels at
     * 1-5 percent noise. Test conditions: one seen Gaussian kernel, one seen motion kernel and
     * an unseen diagonal motion kernel, each at the given noise.
     *
     * @param side the image side (a power of two)
     * @param noise the noise standard deviation (0.01 or 0.05)
     * @param steps training steps (batch 8) of each learned part
     * @param testImages test images per kernel
     * @param seed the seed
     * @param methods the methods
     * @return per kernel the rows in method order
     */
    public static Map<String, List<Row>> deblurring(int side, float noise, int steps, int testImages, long seed,
            DeblurMethod... methods) {
        List<float[]> train = clean(Source.SCENES, 256, side, seed);
        List<float[]> test = clean(Source.SCENES, testImages, side, seed + 1000);
        List<Kernel> kernels = trainingKernels();
        List<Kernel> testKernels = List.of(kernels.get(1), kernels.get(2), unseenKernel());
        SplittableRandom tr = new SplittableRandom(seed + 2000);
        Map<String, List<Row>> out = new LinkedHashMap<>();
        Map<String, Tensor[]> testSets = new LinkedHashMap<>();
        for (Kernel k : testKernels) {
            List<float[]> blurred = new ArrayList<>();
            for (float[] c : test) {
                blurred.add(DegradedImages.gaussian(DegradedImages.blur(c, side, side, k.weights(), k.size()), noise, tr));
            }
            Tensor t = PlugAndPlay.transfer(k.weights(), k.size(), side, side);
            List<Tensor> ts = new ArrayList<>();
            for (int i = 0; i < test.size(); i++) {
                ts.add(t);
            }
            testSets.put(k.name(), new Tensor[] {DegradedImages.stack(blurred, side), PlugAndPlay.stackTransfers(ts)});
            out.put(k.name(), new ArrayList<>());
        }
        Denoisers.FfdNet prior = null;
        for (DeblurMethod m : methods) {
            SplittableRandom r = new SplittableRandom(seed + 13L * m.ordinal());
            UNet2d unet = null;
            PlugAndPlay.Unrolled unrolled = null;
            if (m == DeblurMethod.PNP_HQS || m == DeblurMethod.PNP_ADMM) {
                if (prior == null) {
                    Denoisers.FfdNet f = new Denoisers.FfdNet(5, 24, true, r);
                    trainDenoiser(new NoiseConditioned("FFDNet", (x, map) -> f.forward(x, map, false),
                            (x, map) -> f.forward(x, map, true), f.parameters()), train, side, steps, r);
                    prior = f;
                }
            } else if (m == DeblurMethod.UNET || m == DeblurMethod.UNROLLED) {
                Optimizer opt = Optimizer.adam(2e-3f);
                if (m == DeblurMethod.UNET) {
                    unet = new UNet2d(1, 1, 8, 2, true, 0, r);
                } else {
                    unrolled = new PlugAndPlay.Unrolled(4, 12, r);
                }
                for (int s = 0; s < steps; s++) {
                    List<float[]> c = new ArrayList<>();
                    List<float[]> y = new ArrayList<>();
                    List<Tensor> ts = new ArrayList<>();
                    for (int i = 0; i < 8; i++) {
                        float[] img = train.get(r.nextInt(train.size()));
                        Kernel k = kernels.get(r.nextInt(kernels.size()));
                        c.add(img);
                        y.add(DegradedImages.gaussian(DegradedImages.blur(img, side, side, k.weights(), k.size()),
                                0.01 + 0.04 * r.nextDouble(), r));
                        ts.add(PlugAndPlay.transfer(k.weights(), k.size(), side, side));
                    }
                    Var in = Var.of(DegradedImages.stack(y, side));
                    Var est = unet != null ? in.add(unet.forward(in))
                            : unrolled.forward(in, PlugAndPlay.stackTransfers(ts));
                    est.sub(DegradedImages.stack(c, side)).square().mean().backward();
                    opt.step(unet != null ? unet.parameters() : unrolled.parameters());
                }
            }
            for (Kernel k : testKernels) {
                Tensor[] set = testSets.get(k.name());
                List<float[]> blurred = DegradedImages.unstack(set[0]);
                List<float[]> est;
                long params = 0;
                switch (m) {
                    case BLURRED -> est = blurred;
                    case WIENER -> est = DegradedImages.unstack(PlugAndPlay.wiener(set[0], set[1],
                            Math.max(1e-3f, 30 * noise * noise)));
                    case RICHARDSON_LUCY -> {
                        est = new ArrayList<>();
                        for (float[] b : blurred) {
                            est.add(PlugAndPlay.richardsonLucy(b, side, k.weights(), k.size(), 30));
                        }
                    }
                    case TV -> {
                        est = new ArrayList<>();
                        for (float[] b : blurred) {
                            est.add(PlugAndPlay.totalVariation(b, side, k.weights(), k.size(), 0.4f * noise + 0.002f,
                                    150, 0.5f));
                        }
                    }
                    case UNET -> {
                        UNet2d u = unet;
                        est = DegradedImages.unstack(Var.noGrad(() -> Var.of(set[0]).add(u.forward(Var.of(set[0]))))
                                .value());
                        params = count(u.parameters());
                    }
                    case UNROLLED -> {
                        PlugAndPlay.Unrolled u = unrolled;
                        est = DegradedImages.unstack(Var.noGrad(() -> u.forward(Var.of(set[0]), set[1])).value());
                        params = count(u.parameters());
                    }
                    default -> {
                        ImageDenoiser d = prior;
                        est = DegradedImages.unstack(m == DeblurMethod.PNP_HQS
                                ? PlugAndPlay.hqs(set[0], set[1], noise, d, 8, 0.23f)
                                : PlugAndPlay.admm(set[0], set[1], noise, d, 8, 0.23f));
                        params = count(prior.parameters());
                    }
                }
                out.get(k.name()).add(score(m.name(), est, test, side, params, 0));
            }
        }
        return out;
    }

    /**
     * Prints the long-budget runs (32x32 images, thousands of steps).
     *
     * @param args unused
     */
    public static void main(String[] args) {
        for (Source src : Source.values()) {
            System.out.println("SR x2 " + src + ": " + superResolution(src, 32, 2, 1500, 32, 1, SrMethod.values()));
        }
        System.out.println("SR x4 scenes: " + superResolution(Source.SCENES, 32, 4, 1500, 32, 1, SrMethod.values()));
        System.out.println("OCR x2: " + glyphOcr(2, 1000, 1500, 1));
        System.out.println("OCR x4: " + glyphOcr(4, 1000, 1500, 1));
        System.out.println("denoising: " + denoising(32, 2000, 32, 1, DenoiseMethod.values()));
        for (NoiseKind kind : NoiseKind.values()) {
            System.out.println("self-supervised " + kind + ": "
                    + selfSupervised(kind, 32, 2000, 500, 8, 1, SelfMethod.values()));
        }
        for (float noise : new float[] {0.01f, 0.05f}) {
            System.out.println("deblurring noise " + noise + ": " + deblurring(32, noise, 2000, 16, 1,
                    DeblurMethod.values()));
        }
    }
}
