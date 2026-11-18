package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Clean images and known degradations for image-restoration experiments: denoising,
 * deblurring and super-resolution need the clean image, the degradation operator and its
 * parameters (noise level, blur kernel, scale factor), which real data sets rarely give.
 *
 * <p>Images are square, single-channel, row-major {@code float[]} with values in about
 * {@code [0, 1]}. The clean sources are:
 * <ul>
 *   <li>{@link #scenes}: piecewise-smooth scenes (a gradient background with rectangles,
 *       discs and thin lines), the closest toy analogue of natural images;</li>
 *   <li>{@link #shapes}: {@link ShapeImages}' 3x3 shapes, noise-free;</li>
 *   <li>{@link #glyphCrops}: windows of {@link GlyphDocuments} text lines rendered at an
 *       integer pixel scale (and {@link #glyphLines} for whole lines with their text, for
 *       OCR on restored images);</li>
 *   <li>any other set of images through {@link #normalise}.</li>
 * </ul>
 * Degradations: additive Gaussian noise, Poisson-Gaussian (signal-dependent) noise,
 * spatially correlated noise (white noise filtered by a Gaussian and rescaled to a target
 * standard deviation), salt-and-pepper noise, spatially varying Gaussian noise with its
 * noise-level map, circular convolution with Gaussian and motion-blur kernels, and
 * area (box-filtered) downsampling by 2 or 4 with bicubic and nearest upsampling as
 * reference upsamplers. Metrics: PSNR and SSIM (7x7 uniform window, the usual constants).
 */
public final class DegradedImages {

    private DegradedImages() {
    }

    // ---------------------------------------------------------------- clean sources

    /**
     * Piecewise-smooth scenes.
     *
     * @param n the number of images
     * @param side their side
     * @param seed the seed
     * @return the images
     */
    public static List<float[]> scenes(int n, int side, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<float[]> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            float[] img = new float[side * side];
            double base = 0.15 + 0.3 * r.nextDouble();
            double gx = (r.nextDouble() - 0.5) * 0.3;
            double gy = (r.nextDouble() - 0.5) * 0.3;
            for (int y = 0; y < side; y++) {
                for (int x = 0; x < side; x++) {
                    img[y * side + x] = (float) (base + gx * x / side + gy * y / side);
                }
            }
            int objects = 2 + r.nextInt(3);
            for (int o = 0; o < objects; o++) {
                double v = 0.2 + 0.75 * r.nextDouble();
                int kind = r.nextInt(3);
                if (kind == 0) {
                    int x0 = r.nextInt(side - 3);
                    int y0 = r.nextInt(side - 3);
                    int x1 = Math.min(side, x0 + 3 + r.nextInt(side / 2));
                    int y1 = Math.min(side, y0 + 3 + r.nextInt(side / 2));
                    for (int y = y0; y < y1; y++) {
                        for (int x = x0; x < x1; x++) {
                            img[y * side + x] = (float) v;
                        }
                    }
                } else if (kind == 1) {
                    double cx = r.nextDouble() * side;
                    double cy = r.nextDouble() * side;
                    double rad = 2 + r.nextDouble() * side / 4.0;
                    for (int y = 0; y < side; y++) {
                        for (int x = 0; x < side; x++) {
                            double d = Math.hypot(x + 0.5 - cx, y + 0.5 - cy);
                            double cover = Math.max(0, Math.min(1, rad - d + 0.5));
                            img[y * side + x] = (float) (img[y * side + x] * (1 - cover) + v * cover);
                        }
                    }
                } else {
                    double angle = r.nextDouble() * Math.PI;
                    double cx = r.nextDouble() * side;
                    double cy = r.nextDouble() * side;
                    double nx = -Math.sin(angle);
                    double ny = Math.cos(angle);
                    for (int y = 0; y < side; y++) {
                        for (int x = 0; x < side; x++) {
                            double d = Math.abs((x + 0.5 - cx) * nx + (y + 0.5 - cy) * ny);
                            double cover = Math.max(0, Math.min(1, 1.0 - d));
                            img[y * side + x] = (float) (img[y * side + x] * (1 - cover) + v * cover);
                        }
                    }
                }
            }
            out.add(img);
        }
        return out;
    }

    /**
     * Noise-free {@link ShapeImages} with brightness scaled into {@code [0, 1]}.
     *
     * @param n the number of images
     * @param side their side
     * @param seed the seed
     * @return the images
     */
    public static List<float[]> shapes(int n, int side, long seed) {
        float[] all = new ShapeImages(side, 0f).generate(n, seed)[0].toArray();
        List<float[]> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            float[] img = new float[side * side];
            for (int p = 0; p < img.length; p++) {
                img[p] = 0.1f + 0.6f * all[i * side * side + p];
            }
            out.add(img);
        }
        return out;
    }

    /**
     * A text line rendered at an integer pixel scale, with its text.
     *
     * @param height the image height ({@code GlyphDocuments.LINE_HEIGHT * scale} padded)
     * @param width the image width
     * @param pixels {@code height x width}, dark background, bright ink
     * @param text the text
     * @param labels its CTC labels ({@link GlyphDocuments.Line#labels()})
     */
    public record GlyphLine(int height, int width, float[] pixels, String text, int[] labels) {
    }

    /**
     * Clean text lines at a pixel scale, padded to a common size.
     *
     * @param n the number of lines
     * @param scale the pixel scale (each glyph pixel becomes {@code scale x scale})
     * @param height the padded height, at least {@code LINE_HEIGHT * scale}
     * @param width the padded width (lines are cut to it)
     * @param maxGlyphs the most characters per line
     * @param seed the seed
     * @return the lines
     */
    public static List<GlyphLine> glyphLines(int n, int scale, int height, int width, int maxGlyphs, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<GlyphLine> out = new ArrayList<>(n);
        GlyphDocuments.Style style = GlyphDocuments.Style.clean();
        while (out.size() < n) {
            String text = GlyphDocuments.randomText(3, maxGlyphs, r);
            GlyphDocuments.Line line = GlyphDocuments.line(text, style, r);
            if (line.width() * scale > width) {
                continue;
            }
            float[] img = new float[height * width];
            int top = (height - GlyphDocuments.LINE_HEIGHT * scale) / 2;
            for (int y = 0; y < GlyphDocuments.LINE_HEIGHT * scale; y++) {
                for (int x = 0; x < line.width() * scale; x++) {
                    img[(top + y) * width + x] = 0.1f + 0.8f * line.pixels()[(y / scale) * line.width() + x / scale];
                }
            }
            for (int y = 0; y < height; y++) {
                for (int x = line.width() * scale; x < width; x++) {
                    img[y * width + x] = 0.1f;
                }
                if (y < top || y >= top + GlyphDocuments.LINE_HEIGHT * scale) {
                    for (int x = 0; x < line.width() * scale; x++) {
                        img[y * width + x] = 0.1f;
                    }
                }
            }
            out.add(new GlyphLine(height, width, img, text, line.labels()));
        }
        return out;
    }

    /**
     * Square windows of text lines at a pixel scale.
     *
     * @param n the number of crops
     * @param side their side (a multiple of 8 works best: the scale is {@code side / 8})
     * @param seed the seed
     * @return the crops
     */
    public static List<float[]> glyphCrops(int n, int side, long seed) {
        int scale = Math.max(1, side / 8);
        int width = 3 * side;
        List<GlyphLine> lines = glyphLines(n, scale, side, width, 12, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        List<float[]> out = new ArrayList<>(n);
        for (GlyphLine l : lines) {
            int x0 = r.nextInt(width - side + 1) / 2;
            float[] img = new float[side * side];
            for (int y = 0; y < side; y++) {
                System.arraycopy(l.pixels(), y * width + x0, img, y * side, side);
            }
            out.add(img);
        }
        return out;
    }

    /**
     * Rescales arbitrary images linearly into {@code [lo, hi]} using their joint range.
     *
     * @param images the images
     * @param lo the new minimum
     * @param hi the new maximum
     * @return rescaled copies
     */
    public static List<float[]> normalise(List<float[]> images, float lo, float hi) {
        float min = Float.MAX_VALUE;
        float max = -Float.MAX_VALUE;
        for (float[] im : images) {
            for (float v : im) {
                min = Math.min(min, v);
                max = Math.max(max, v);
            }
        }
        float scale = (hi - lo) / Math.max(1e-6f, max - min);
        List<float[]> out = new ArrayList<>(images.size());
        for (float[] im : images) {
            float[] c = new float[im.length];
            for (int i = 0; i < c.length; i++) {
                c[i] = lo + (im[i] - min) * scale;
            }
            out.add(c);
        }
        return out;
    }

    // ---------------------------------------------------------------- noise

    /**
     * Additive white Gaussian noise.
     *
     * @param x the clean image
     * @param sigma the standard deviation (on the [0, 1] scale; 25/255 is about 0.098)
     * @param random the source
     * @return the noisy copy
     */
    public static float[] gaussian(float[] x, double sigma, RandomGenerator random) {
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = (float) (x[i] + sigma * random.nextGaussian());
        }
        return out;
    }

    /**
     * Poisson-Gaussian noise: {@code Poisson(peak * x) / peak + N(0, sigma^2)}, the usual
     * model of photon shot noise plus read noise; the Poisson draw uses its Gaussian
     * approximation, accurate for the counts used here.
     *
     * @param x the clean image
     * @param peak photons at intensity 1 (lower is noisier)
     * @param sigma the read-noise standard deviation
     * @param random the source
     * @return the noisy copy
     */
    public static float[] poissonGaussian(float[] x, double peak, double sigma, RandomGenerator random) {
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            double lambda = Math.max(0, x[i]) * peak;
            out[i] = (float) ((lambda + Math.sqrt(lambda) * random.nextGaussian()) / peak + sigma * random.nextGaussian());
        }
        return out;
    }

    /**
     * Spatially correlated Gaussian noise: white noise circularly convolved with a Gaussian of
     * standard deviation {@code radius} pixels, rescaled to a per-pixel standard deviation of
     * {@code sigma}. Blind-spot denoisers assume independent noise and fail on it.
     *
     * @param x the clean image
     * @param side the image side
     * @param sigma the per-pixel standard deviation
     * @param radius the correlation length in pixels
     * @param random the source
     * @return the noisy copy
     */
    public static float[] correlated(float[] x, int side, double sigma, double radius, RandomGenerator random) {
        float[] white = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            white[i] = (float) random.nextGaussian();
        }
        int k = 2 * (int) Math.ceil(2 * radius) + 1;
        float[] kernel = gaussianKernel(k, radius);
        float[] coloured = blur(white, side, side, kernel, k);
        double norm = 0;
        for (float w : kernel) {
            norm += w * w;
        }
        double scale = sigma / Math.sqrt(norm);
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = (float) (x[i] + scale * coloured[i]);
        }
        return out;
    }

    /**
     * Salt-and-pepper noise: a fraction of pixels set to 0 or 1.
     *
     * @param x the clean image
     * @param fraction the fraction of corrupted pixels
     * @param random the source
     * @return the noisy copy
     */
    public static float[] saltAndPepper(float[] x, double fraction, RandomGenerator random) {
        float[] out = x.clone();
        for (int i = 0; i < x.length; i++) {
            if (random.nextDouble() < fraction) {
                out[i] = random.nextBoolean() ? 1f : 0f;
            }
        }
        return out;
    }

    /**
     * A noisy image with its true per-pixel noise level.
     *
     * @param image the noisy image
     * @param sigmaMap the per-pixel standard deviation
     */
    public record NoisyImage(float[] image, float[] sigmaMap) {
    }

    /**
     * Spatially varying Gaussian noise: the standard deviation ramps linearly from
     * {@code low} to {@code high} along a random direction across the image.
     *
     * @param x the clean image
     * @param side the image side
     * @param low the smallest standard deviation
     * @param high the largest
     * @param random the source
     * @return the noisy image and its noise map
     */
    public static NoisyImage spatiallyVarying(float[] x, int side, double low, double high, RandomGenerator random) {
        double angle = random.nextDouble() * 2 * Math.PI;
        double c = Math.cos(angle);
        double s = Math.sin(angle);
        float[] out = new float[x.length];
        float[] map = new float[x.length];
        double half = (side - 1) / 2.0;
        double extent = half * (Math.abs(c) + Math.abs(s));
        for (int y = 0; y < side; y++) {
            for (int xx = 0; xx < side; xx++) {
                double t = ((xx - half) * c + (y - half) * s) / Math.max(1e-9, extent) * 0.5 + 0.5;
                double sigma = low + (high - low) * t;
                map[y * side + xx] = (float) sigma;
                out[y * side + xx] = (float) (x[y * side + xx] + sigma * random.nextGaussian());
            }
        }
        return new NoisyImage(out, map);
    }

    // ---------------------------------------------------------------- blur

    /**
     * A normalised square Gaussian kernel.
     *
     * @param size the side (odd)
     * @param sigma the standard deviation in pixels
     * @return {@code size x size}, row-major, summing to 1
     */
    public static float[] gaussianKernel(int size, double sigma) {
        float[] k = new float[size * size];
        int c = size / 2;
        double sum = 0;
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                double v = Math.exp(-((x - c) * (x - c) + (y - c) * (y - c)) / (2 * sigma * sigma));
                k[y * size + x] = (float) v;
                sum += v;
            }
        }
        for (int i = 0; i < k.length; i++) {
            k[i] /= (float) sum;
        }
        return k;
    }

    /**
     * A normalised linear motion-blur kernel: a line segment through the centre, drawn with
     * sub-pixel sampling.
     *
     * @param size the side (odd)
     * @param length the length of the motion in pixels
     * @param angle its direction in radians
     * @return {@code size x size}, row-major, summing to 1
     */
    public static float[] motionKernel(int size, double length, double angle) {
        float[] k = new float[size * size];
        int c = size / 2;
        int samples = 64;
        for (int i = 0; i < samples; i++) {
            double t = (i / (double) (samples - 1) - 0.5) * length;
            double x = c + t * Math.cos(angle);
            double y = c + t * Math.sin(angle);
            int x0 = (int) Math.floor(x);
            int y0 = (int) Math.floor(y);
            double fx = x - x0;
            double fy = y - y0;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    int xx = x0 + dx;
                    int yy = y0 + dy;
                    if (xx >= 0 && yy >= 0 && xx < size && yy < size) {
                        k[yy * size + xx] += (float) ((dx == 0 ? 1 - fx : fx) * (dy == 0 ? 1 - fy : fy));
                    }
                }
            }
        }
        double sum = 0;
        for (float v : k) {
            sum += v;
        }
        for (int i = 0; i < k.length; i++) {
            k[i] /= (float) sum;
        }
        return k;
    }

    /**
     * Circular (wrap-around) convolution with a centred kernel, the blur model that the
     * FFT-domain solvers invert exactly.
     *
     * @param x the image
     * @param h its height
     * @param w its width
     * @param kernel the kernel, {@code k x k}, centred at {@code (k / 2, k / 2)}
     * @param k the kernel side
     * @return the blurred copy
     */
    public static float[] blur(float[] x, int h, int w, float[] kernel, int k) {
        float[] out = new float[x.length];
        int c = k / 2;
        for (int y = 0; y < h; y++) {
            for (int xx = 0; xx < w; xx++) {
                double s = 0;
                for (int ky = 0; ky < k; ky++) {
                    int yy = Math.floorMod(y + c - ky, h);
                    for (int kx = 0; kx < k; kx++) {
                        float kv = kernel[ky * k + kx];
                        if (kv != 0) {
                            s += kv * x[yy * w + Math.floorMod(xx + c - kx, w)];
                        }
                    }
                }
                out[y * w + xx] = (float) s;
            }
        }
        return out;
    }

    // ---------------------------------------------------------------- resampling

    /**
     * Area downsampling: the mean of each {@code factor x factor} block (a box anti-aliasing
     * filter followed by decimation).
     *
     * @param x the image
     * @param side its side (divisible by {@code factor})
     * @param factor the scale factor
     * @return the {@code side / factor} square image
     */
    public static float[] downsample(float[] x, int side, int factor) {
        int s = side / factor;
        float[] out = new float[s * s];
        float inv = 1f / (factor * factor);
        for (int y = 0; y < side; y++) {
            for (int xx = 0; xx < side; xx++) {
                out[(y / factor) * s + xx / factor] += x[y * side + xx] * inv;
            }
        }
        return out;
    }

    /**
     * Upsamples a batch with core {@link Functional#resize}.
     *
     * @param lowRes {@code [n, c, h, w]}
     * @param factor the scale factor
     * @param interpolation nearest, linear or cubic
     * @return {@code [n, c, h * factor, w * factor]}
     */
    public static Var upsample(Var lowRes, int factor, Functional.Interpolation interpolation) {
        Shape s = lowRes.shape();
        return Functional.resize(lowRes, new int[] {s.dim(2) * factor, s.dim(3) * factor}, interpolation, false);
    }

    /**
     * Stacks square single-channel images into a batch.
     *
     * @param images the images, all {@code side x side}
     * @param side their side
     * @return {@code [n, 1, side, side]}
     */
    public static Tensor stack(List<float[]> images, int side) {
        float[] all = new float[images.size() * side * side];
        for (int i = 0; i < images.size(); i++) {
            System.arraycopy(images.get(i), 0, all, i * side * side, side * side);
        }
        return Tensor.of(Shape.of(images.size(), 1, side, side), all);
    }

    /**
     * Splits a batch {@code [n, 1, side, side]} back into images.
     *
     * @param batch the batch
     * @return its images
     */
    public static List<float[]> unstack(Tensor batch) {
        int n = batch.shape().dim(0);
        int per = Math.toIntExact(batch.shape().size() / n);
        float[] all = batch.toArray();
        List<float[]> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            float[] im = new float[per];
            System.arraycopy(all, i * per, im, 0, per);
            out.add(im);
        }
        return out;
    }

    // ---------------------------------------------------------------- metrics

    /**
     * Peak signal-to-noise ratio for a peak value of 1.
     *
     * @param estimate the restored image
     * @param reference the clean image
     * @return PSNR in dB (capped at 100)
     */
    public static double psnr(float[] estimate, float[] reference) {
        double mse = 0;
        for (int i = 0; i < reference.length; i++) {
            double d = Math.max(0, Math.min(1, estimate[i])) - reference[i];
            mse += d * d;
        }
        mse /= reference.length;
        return mse < 1e-10 ? 100 : 10 * Math.log10(1 / mse);
    }

    /**
     * Structural similarity (Wang et al., 2004) with a 7x7 uniform window over all valid
     * positions and {@code C1 = 0.01^2}, {@code C2 = 0.03^2} for a data range of 1.
     *
     * @param estimate the restored image
     * @param reference the clean image
     * @param side the image side
     * @return the mean SSIM
     */
    public static double ssim(float[] estimate, float[] reference, int side) {
        int win = Math.min(7, side);
        double c1 = 1e-4;
        double c2 = 9e-4;
        double total = 0;
        int count = 0;
        int n = win * win;
        for (int y0 = 0; y0 + win <= side; y0++) {
            for (int x0 = 0; x0 + win <= side; x0++) {
                double ma = 0;
                double mb = 0;
                for (int y = y0; y < y0 + win; y++) {
                    for (int x = x0; x < x0 + win; x++) {
                        ma += Math.max(0, Math.min(1, estimate[y * side + x]));
                        mb += reference[y * side + x];
                    }
                }
                ma /= n;
                mb /= n;
                double va = 0;
                double vb = 0;
                double cov = 0;
                for (int y = y0; y < y0 + win; y++) {
                    for (int x = x0; x < x0 + win; x++) {
                        double a = Math.max(0, Math.min(1, estimate[y * side + x])) - ma;
                        double b = reference[y * side + x] - mb;
                        va += a * a;
                        vb += b * b;
                        cov += a * b;
                    }
                }
                va /= n - 1;
                vb /= n - 1;
                cov /= n - 1;
                total += (2 * ma * mb + c1) * (2 * cov + c2) / ((ma * ma + mb * mb + c1) * (va + vb + c2));
                count++;
            }
        }
        return total / count;
    }

    /**
     * Mean PSNR over pairs of images.
     *
     * @param estimates the restored images
     * @param references the clean images
     * @return the mean PSNR in dB
     */
    public static double meanPsnr(List<float[]> estimates, List<float[]> references) {
        double s = 0;
        for (int i = 0; i < references.size(); i++) {
            s += psnr(estimates.get(i), references.get(i));
        }
        return s / references.size();
    }

    /**
     * Mean SSIM over pairs of images.
     *
     * @param estimates the restored images
     * @param references the clean images
     * @param side the image side
     * @return the mean SSIM
     */
    public static double meanSsim(List<float[]> estimates, List<float[]> references, int side) {
        double s = 0;
        for (int i = 0; i < references.size(); i++) {
            s += ssim(estimates.get(i), references.get(i), side);
        }
        return s / references.size();
    }
}
