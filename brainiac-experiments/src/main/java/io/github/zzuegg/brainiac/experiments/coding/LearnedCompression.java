package io.github.zzuegg.brainiac.experiments.coding;

import io.github.zzuegg.brainiac.experimental.codec.LearnedImageCodec;
import io.github.zzuegg.brainiac.experimental.codec.ModelCodec;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.ml.KMeans;
import io.github.zzuegg.brainiac.experimental.stream.StreamingPca;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Wave 12 item 12.2.15: learned image compression. {@link LearnedImageCodec} with a factorised
 * prior and with a scale hyperprior, trained for rate plus λ times distortion at several λ, is
 * compared on 16×16 {@link ShapeImages} with a JPEG-like codec (8×8 DCT, the JPEG luminance table
 * scaled by quality, zig-zag (run, size) tokens and DC size categories coded by the 12.2.14 adaptive
 * arithmetic coder, plus the raw magnitude bits, as in baseline JPEG),
 * PCA with quantised coefficients (coded the same way) and k-means vector quantisation of whole
 * images at {@code log2 K} bits. Every rate is a real code length in bits per pixel; distortion is
 * PSNR on held-out images.
 */
public final class LearnedCompression {

    private LearnedCompression() {
    }

    /**
     * One rate-distortion point.
     *
     * @param method the codec
     * @param setting its rate setting
     * @param bitsPerPixel the rate
     * @param psnr the distortion, in dB
     */
    public record RdPoint(String method, String setting, double bitsPerPixel, double psnr) {
    }

    private static final int SIDE = 16;

    private static final int[] JPEG_LUMA = {16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16,
        24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104,
        113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99};

    private static final int[] ZIGZAG = zigzag();

    private static int[] zigzag() {
        int[] z = new int[64];
        int i = 0;
        for (int s = 0; s < 15; s++) {
            for (int k = 0; k <= s; k++) {
                int r = s % 2 == 0 ? s - k : k;
                int c = s - r;
                if (r < 8 && c < 8) {
                    z[i++] = r * 8 + c;
                }
            }
        }
        return z;
    }

    private static final double[][] DCT = dct();

    private static double[][] dct() {
        double[][] m = new double[8][8];
        for (int u = 0; u < 8; u++) {
            double a = u == 0 ? Math.sqrt(1.0 / 8) : Math.sqrt(2.0 / 8);
            for (int x = 0; x < 8; x++) {
                m[u][x] = a * Math.cos((2 * x + 1) * u * Math.PI / 16);
            }
        }
        return m;
    }

    private static double psnr(float[] a, float[] b) {
        double mse = 0;
        for (int i = 0; i < a.length; i++) {
            double d = Math.max(0, Math.min(1, b[i])) - a[i];
            mse += d * d / a.length;
        }
        return 10 * Math.log10(1 / Math.max(mse, 1e-12));
    }

    private static int bytesOf(int[] symbols, int alphabet) {
        if (symbols.length == 0) {
            return 0;
        }
        return ModelCodec.of(() -> new ModelCodec.ContextModel(alphabet, 0), ModelCodec.Coder.ARITHMETIC, 16)
                .encode(symbols).length;
    }

    /** The JPEG-like codec at one quality: returns {bits per pixel, PSNR}. */
    static double[] jpegLike(float[] images, int n, int quality) {
        double scale = quality < 50 ? 5000.0 / quality : 200 - 2.0 * quality;
        double[] q = new double[64];
        for (int i = 0; i < 64; i++) {
            q[i] = Math.max(1, Math.floor((JPEG_LUMA[i] * scale + 50) / 100)) / 255.0;
        }
        List<Integer> tokens = new ArrayList<>();
        List<Integer> dcs = new ArrayList<>();
        float[] rec = new float[images.length];
        int blocks = SIDE / 8;
        int prevDc = 0;
        long rawBits = 0;
        for (int img = 0; img < n; img++) {
            int base = img * SIDE * SIDE;
            for (int by = 0; by < blocks; by++) {
                for (int bx = 0; bx < blocks; bx++) {
                    double[] coef = new double[64];
                    for (int u = 0; u < 8; u++) {
                        for (int v = 0; v < 8; v++) {
                            double s = 0;
                            for (int y = 0; y < 8; y++) {
                                for (int x = 0; x < 8; x++) {
                                    double p = images[base + (by * 8 + y) * SIDE + bx * 8 + x] - 0.5;
                                    s += DCT[u][y] * DCT[v][x] * p;
                                }
                            }
                            coef[u * 8 + v] = s;
                        }
                    }
                    int[] level = new int[64];
                    for (int i = 0; i < 64; i++) {
                        level[i] = (int) Math.round(coef[i] / q[i]);
                    }
                    int dc = Math.max(-2047, Math.min(2047, level[0]));
                    int diff = dc - prevDc;
                    int dcSize = diff == 0 ? 0 : 32 - Integer.numberOfLeadingZeros(Math.abs(diff));
                    dcs.add(dcSize);
                    rawBits += dcSize;
                    prevDc = dc;
                    level[0] = dc;
                    int run = 0;
                    for (int k = 1; k < 64; k++) {
                        int pos = ZIGZAG[k];
                        int l = Math.max(-2047, Math.min(2047, level[pos]));
                        level[pos] = l;
                        if (l == 0) {
                            run++;
                            continue;
                        }
                        while (run > 15) {
                            tokens.add(1 + 16 * 12);
                            run -= 16;
                        }
                        int size = 32 - Integer.numberOfLeadingZeros(Math.abs(l));
                        tokens.add(1 + run * 12 + size);
                        rawBits += size;
                        run = 0;
                    }
                    tokens.add(0);
                    for (int y = 0; y < 8; y++) {
                        for (int x = 0; x < 8; x++) {
                            double s = 0;
                            for (int u = 0; u < 8; u++) {
                                for (int v = 0; v < 8; v++) {
                                    s += DCT[u][y] * DCT[v][x] * level[u * 8 + v] * q[u * 8 + v];
                                }
                            }
                            rec[base + (by * 8 + y) * SIDE + bx * 8 + x] = (float) (s + 0.5);
                        }
                    }
                }
            }
        }
        int bytes = bytesOf(tokens.stream().mapToInt(Integer::intValue).toArray(), 2 + 16 * 12)
                + bytesOf(dcs.stream().mapToInt(Integer::intValue).toArray(), 13);
        return new double[] {(8.0 * bytes + rawBits) / images.length, psnr(images, rec)};
    }

    /**
     * Runs the comparison.
     *
     * @param trainImages training images
     * @param testImages held-out images
     * @param lambdas the λ values of the learned codecs
     * @param steps training steps per learned codec
     * @param seed the seed
     * @return the rate-distortion points
     */
    public static List<RdPoint> run(int trainImages, int testImages, double[] lambdas, int steps, long seed) {
        ShapeImages gen = new ShapeImages(SIDE, 0.03f);
        Tensor train = gen.generate(trainImages, seed)[0];
        Tensor test = gen.generate(testImages, seed + 1)[0];
        float[] tr = train.toArray();
        float[] te = test.toArray();
        int px = SIDE * SIDE;
        List<RdPoint> rows = new ArrayList<>();
        for (int quality : new int[] {10, 25, 50, 75, 90}) {
            double[] r = jpegLike(te, testImages, quality);
            rows.add(new RdPoint("JPEG-like DCT", "quality " + quality, r[0], r[1]));
        }
        // PCA with quantised coefficients
        double[] mean = new double[px];
        for (int i = 0; i < trainImages; i++) {
            for (int j = 0; j < px; j++) {
                mean[j] += tr[i * px + j] / trainImages;
            }
        }
        for (int k : new int[] {8, 24}) {
            StreamingPca pca = StreamingPca.incrementalSvd(px, k, 1, 20);
            for (int i = 0; i < trainImages; i++) {
                double[] x = new double[px];
                for (int j = 0; j < px; j++) {
                    x[j] = tr[i * px + j];
                }
                pca.add(x);
            }
            pca.flush();
            double[][] basis = pca.basis();
            for (double step : new double[] {0.1, 0.3}) {
                int[] sym = new int[testImages * k];
                float[] rec = new float[te.length];
                for (int i = 0; i < testImages; i++) {
                    for (int c = 0; c < k; c++) {
                        double s = 0;
                        for (int j = 0; j < px; j++) {
                            s += basis[j][c] * (te[i * px + j] - mean[j]);
                        }
                        int l = (int) Math.max(-127, Math.min(127, Math.round(s / step)));
                        sym[i * k + c] = l + 127;
                        for (int j = 0; j < px; j++) {
                            rec[i * px + j] += (float) (basis[j][c] * l * step);
                        }
                    }
                    for (int j = 0; j < px; j++) {
                        rec[i * px + j] += (float) mean[j];
                    }
                }
                rows.add(new RdPoint("PCA, quantised coefficients", k + " components, step " + step,
                        8.0 * bytesOf(sym, 255) / te.length, psnr(te, rec)));
            }
        }
        // k-means VQ of whole images
        double[][] tx = new double[trainImages][px];
        for (int i = 0; i < trainImages; i++) {
            for (int j = 0; j < px; j++) {
                tx[i][j] = tr[i * px + j];
            }
        }
        for (int kk : new int[] {16, 64}) {
            KMeans km = KMeans.fit(tx, kk, KMeans.Init.PLUS_PLUS, 1, 30, seed);
            float[] rec = new float[te.length];
            for (int i = 0; i < testImages; i++) {
                double[] x = new double[px];
                for (int j = 0; j < px; j++) {
                    x[j] = te[i * px + j];
                }
                double[] c = km.centres()[KMeans.nearest(km.centres(), x)];
                for (int j = 0; j < px; j++) {
                    rec[i * px + j] = (float) c[j];
                }
            }
            rows.add(new RdPoint("k-means VQ (whole image)", "K = " + kk, Math.log(kk) / Math.log(2) / px,
                    psnr(te, rec)));
        }
        for (LearnedImageCodec.Prior prior : LearnedImageCodec.Prior.values()) {
            for (double lambda : lambdas) {
                LearnedImageCodec codec = LearnedImageCodec.of(prior, SIDE, 16, 8, 4, seed);
                codec.train(train, lambda, steps, 16, 3e-3f, seed);
                LearnedImageCodec.Encoded e = codec.encode(test);
                Tensor back = codec.decode(e.code(), testImages);
                float[] rb = back.toArray();
                float[] re = e.reconstruction().toArray();
                if (!Arrays.equals(rb, re)) {
                    throw new IllegalStateException("decoder mismatch");
                }
                rows.add(new RdPoint("learned, " + prior.name().toLowerCase(java.util.Locale.ROOT), "lambda " + lambda,
                        e.bitsPerPixel(), LearnedImageCodec.psnr(test, e.reconstruction())));
            }
        }
        return rows;
    }
}
