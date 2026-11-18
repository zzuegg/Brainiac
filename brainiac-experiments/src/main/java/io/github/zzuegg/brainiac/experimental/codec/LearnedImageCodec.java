package io.github.zzuegg.brainiac.experimental.codec;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * End-to-end learned image compression (Ballé, Laparra and Simoncelli, ICLR 2017,
 * <a href="https://arxiv.org/abs/1611.01704">arXiv:1611.01704</a>; Ballé, Minnen, Singh, Hwang and
 * Johnston, ICLR 2018, <a href="https://arxiv.org/abs/1802.01436">arXiv:1802.01436</a>). An analysis
 * transform (two stride-2 3×3 ReLU convolutions) maps a {@code side × side} image to latents
 * {@code y}, which are quantised by rounding and entropy coded; a synthesis transform (two stride-2
 * transposed convolutions) reconstructs the image. Training minimises {@code R + λ D} with
 * quantisation replaced by additive uniform noise, {@code R} the latents' negative log2-likelihood
 * per pixel and {@code D} the mean squared error.
 *
 * <ul>
 *   <li>{@link Prior#FACTORIZED}: each latent channel has its own learned discretised logistic
 *       density.</li>
 *   <li>{@link Prior#HYPERPRIOR}: a hyper-analysis transform maps {@code |y|} to side
 *       information {@code z} (coded with a factorised prior) from which a hyper-synthesis
 *       transform predicts the scale {@code σ} of every latent; {@code y} is then coded as a
 *       discretised zero-mean Gaussian, which captures spatially varying statistics.</li>
 * </ul>
 *
 * <p>{@link #encode} rounds the latents and writes them with {@link ArithmeticCoder} (per-channel
 * tables, or one table per quantised {@code σ}), so reported rates are real code lengths;
 * {@link #decode} inverts it exactly.
 */
public final class LearnedImageCodec {

    /** The latent density model. */
    public enum Prior {
        /** per-channel factorised densities */
        FACTORIZED,
        /** a scale hyperprior */
        HYPERPRIOR
    }

    private static final int RANGE = 24;
    private static final int PRECISION = 16;
    private static final double LOG2 = Math.log(2);

    private final Prior prior;
    private final int side;
    private final int latent;
    private final int hyper;
    private final Var a1w;
    private final Var a1b;
    private final Var a2w;
    private final Var a2b;
    private final Var s1w;
    private final Var s1b;
    private final Var s2w;
    private final Var s2b;
    private final Var yLoc;
    private final Var yScale;
    private final Var h1w;
    private final Var h1b;
    private final Var hs1w;
    private final Var hs1b;
    private final Var zLoc;
    private final Var zScale;
    private final Map<Integer, FrequencyTable> gaussianTables = new HashMap<>();

    private LearnedImageCodec(Prior prior, int side, int channels, int latent, int hyper, long seed) {
        if (side % 8 != 0) {
            throw new IllegalArgumentException("side must be a multiple of 8");
        }
        this.prior = prior;
        this.side = side;
        this.latent = latent;
        this.hyper = hyper;
        SplittableRandom r = new SplittableRandom(seed);
        a1w = param(r, 2.0 / 9, channels, 1, 3, 3);
        a1b = Var.param(Tensor.zeros(Shape.of(channels)));
        a2w = param(r, 2.0 / (9 * channels), latent, channels, 3, 3);
        a2b = Var.param(Tensor.zeros(Shape.of(latent)));
        s1w = param(r, 2.0 / (16 * latent), latent, channels, 4, 4);
        s1b = Var.param(Tensor.zeros(Shape.of(channels)));
        s2w = param(r, 1.0 / (16 * channels), channels, 1, 4, 4);
        s2b = Var.param(Tensor.zeros(Shape.of(1)));
        yLoc = Var.param(Tensor.zeros(Shape.of(1, latent, 1, 1)));
        yScale = Var.param(Tensor.zeros(Shape.of(1, latent, 1, 1)));
        h1w = param(r, 2.0 / (9 * latent), hyper, latent, 3, 3);
        h1b = Var.param(Tensor.zeros(Shape.of(hyper)));
        hs1w = param(r, 1.0 / (16 * hyper), hyper, latent, 4, 4);
        hs1b = Var.param(Tensor.zeros(Shape.of(latent)));
        zLoc = Var.param(Tensor.zeros(Shape.of(1, hyper, 1, 1)));
        zScale = Var.param(Tensor.zeros(Shape.of(1, hyper, 1, 1)));
    }

    private static Var param(SplittableRandom r, double variance, int... dims) {
        return Var.param(Tensor.normal(Shape.of(dims), 0, (float) Math.sqrt(variance), r));
    }

    /**
     * A codec.
     *
     * @param prior the latent model
     * @param side the image side (a multiple of 8)
     * @param channels the hidden channels
     * @param latent the latent channels
     * @param hyper the hyper-latent channels
     * @param seed the seed
     * @return the untrained codec
     */
    public static LearnedImageCodec of(Prior prior, int side, int channels, int latent, int hyper, long seed) {
        return new LearnedImageCodec(prior, side, channels, latent, hyper, seed);
    }

    /** @return the trainable parameters of the chosen prior */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(a1w, a1b, a2w, a2b, s1w, s1b, s2w, s2b));
        if (prior == Prior.FACTORIZED) {
            p.add(yLoc);
            p.add(yScale);
        } else {
            p.addAll(List.of(h1w, h1b, hs1w, hs1b, zLoc, zScale));
        }
        return p;
    }

    private static Var conv(Var x, Var w, Var b) {
        int[] two = {2, 2};
        int[] one = {1, 1};
        return Functional.conv(x, w, b, two, one, one, one, 1);
    }

    private static Var deconv(Var x, Var w, Var b) {
        int[] two = {2, 2};
        int[] one = {1, 1};
        return Functional.convTranspose(x, w, b, two, one, one, new int[] {0, 0}, one, 1);
    }

    private Var analysis(Var x) {
        return conv(conv(x, a1w, a1b).relu(), a2w, a2b);
    }

    private Var synthesis(Var y) {
        return deconv(deconv(y, s1w, s1b).relu(), s2w, s2b);
    }

    private Var hyperAnalysis(Var y) {
        return conv(y.abs(), h1w, h1b);
    }

    private Var scales(Var z) {
        return deconv(z, hs1w, hs1b).softplus().add(0.11f);
    }

    /** -log2 of a discretised logistic with per-channel location and log-scale. */
    private static Var logisticBits(Var v, Var loc, Var logScale) {
        Var s = logScale.exp();
        Var hi = v.add(0.5f).sub(loc).div(s).sigmoid();
        Var lo = v.sub(0.5f).sub(loc).div(s).sigmoid();
        return hi.sub(lo).maximum(1e-9f).log().mul((float) (-1 / LOG2));
    }

    /** -log2 of a discretised zero-mean Gaussian. */
    private static Var gaussianBits(Var v, Var sigma) {
        float k = (float) (1 / Math.sqrt(2));
        Var hi = v.add(0.5f).div(sigma).mul(k).erf();
        Var lo = v.sub(0.5f).div(sigma).mul(k).erf();
        return hi.sub(lo).mul(0.5f).maximum(1e-9f).log().mul((float) (-1 / LOG2));
    }

    /**
     * Trains on images with Adam on {@code R + λ D}.
     *
     * @param images {@code [n, 1, side, side]} in {@code [0, 1]}
     * @param lambda the distortion weight (per-pixel MSE against bits per pixel)
     * @param steps the steps
     * @param batch the minibatch size
     * @param learningRate the Adam step size
     * @param seed the minibatch and noise seed
     * @return the final training loss
     */
    public double train(Tensor images, double lambda, int steps, int batch, float learningRate, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(learningRate);
        List<Var> params = parameters();
        int n = images.shape().dim(0);
        int px = side * side;
        float[] all = images.toArray();
        double last = 0;
        for (int s = 0; s < steps; s++) {
            float[] bx = new float[batch * px];
            for (int i = 0; i < batch; i++) {
                System.arraycopy(all, r.nextInt(n) * px, bx, i * px, px);
            }
            Var x = Var.of(Tensor.of(Shape.of(batch, 1, side, side), bx));
            Var y = analysis(x);
            Var yt = y.add(noise(y.shape(), r));
            Var bits;
            if (prior == Prior.FACTORIZED) {
                bits = logisticBits(yt, yLoc, yScale).sum();
            } else {
                Var z = hyperAnalysis(y);
                Var zt = z.add(noise(z.shape(), r));
                bits = gaussianBits(yt, scales(zt)).sum().add(logisticBits(zt, zLoc, zScale).sum());
            }
            Var rate = bits.div((float) (batch * px));
            Var mse = synthesis(yt).sub(x).square().mean();
            Var loss = rate.add(mse.mul((float) lambda));
            last = loss.value().item();
            loss.backward();
            opt.step(params);
        }
        return last;
    }

    private static Tensor noise(Shape shape, SplittableRandom r) {
        float[] v = new float[Math.toIntExact(shape.size())];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) (r.nextDouble() - 0.5);
        }
        return Tensor.of(shape, v);
    }

    // ------------------------------------------------------------------ real coding

    /**
     * The result of coding a batch of images.
     *
     * @param code the bitstream
     * @param reconstruction the decoded images, {@code [n, 1, side, side]}
     * @param bitsPerPixel the code length per pixel
     */
    public record Encoded(byte[] code, Tensor reconstruction, double bitsPerPixel) {
    }

    private static int clampSymbol(float v) {
        return Math.max(-RANGE, Math.min(RANGE, Math.round(v)));
    }

    private FrequencyTable logisticTable(double loc, double logScale) {
        double s = Math.exp(logScale);
        double[] p = new double[2 * RANGE + 1];
        for (int k = -RANGE; k <= RANGE; k++) {
            p[k + RANGE] = sigmoid((k + 0.5 - loc) / s) - sigmoid((k - 0.5 - loc) / s);
        }
        return FrequencyTable.of(p, PRECISION);
    }

    private static double sigmoid(double v) {
        return 1 / (1 + Math.exp(-v));
    }

    /** Scales are quantised to 64 log-spaced levels between 0.11 and 64. */
    private static int scaleLevel(float sigma) {
        double t = Math.log(Math.max(0.11, Math.min(64, sigma)) / 0.11) / Math.log(64 / 0.11);
        return (int) Math.round(t * 63);
    }

    private FrequencyTable gaussianTable(int level) {
        return gaussianTables.computeIfAbsent(level, l -> {
            double sigma = 0.11 * Math.pow(64 / 0.11, l / 63.0);
            double[] p = new double[2 * RANGE + 1];
            for (int k = -RANGE; k <= RANGE; k++) {
                p[k + RANGE] = cdf((k + 0.5) / sigma) - cdf((k - 0.5) / sigma);
            }
            return FrequencyTable.of(p, PRECISION);
        });
    }

    private static double cdf(double z) {
        return 0.5 * (1 + erf(z / Math.sqrt(2)));
    }

    private static double erf(double x) {
        double t = 1 / (1 + 0.5 * Math.abs(x));
        double y = 1 - t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
                + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587
                + t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0 ? y : -y;
    }

    /**
     * Compresses images with real entropy coding.
     *
     * @param images {@code [n, 1, side, side]}
     * @return the code, the decoder's reconstruction and the rate
     */
    public Encoded encode(Tensor images) {
        int n = images.shape().dim(0);
        ArithmeticCoder.Encoder enc = new ArithmeticCoder.Encoder();
        Tensor y = Var.noGrad(() -> analysis(Var.of(images))).value();
        float[] yv = y.toArray();
        int[] ys = new int[yv.length];
        float[] yq = new float[yv.length];
        for (int i = 0; i < yv.length; i++) {
            ys[i] = clampSymbol(yv[i]);
            yq[i] = ys[i];
        }
        Tensor yhat = Tensor.of(y.shape(), yq);
        int ls = side / 4;
        if (prior == Prior.FACTORIZED) {
            float[] loc = yLoc.value().toArray();
            float[] sc = yScale.value().toArray();
            FrequencyTable[] tables = new FrequencyTable[latent];
            for (int c = 0; c < latent; c++) {
                tables[c] = logisticTable(loc[c], sc[c]);
            }
            for (int i = 0; i < ys.length; i++) {
                int c = (i / (ls * ls)) % latent;
                enc.encode(tables[c], ys[i] + RANGE);
            }
        } else {
            Tensor z = Var.noGrad(() -> hyperAnalysis(Var.of(yhat))).value();
            float[] zv = z.toArray();
            float[] zq = new float[zv.length];
            float[] loc = zLoc.value().toArray();
            float[] sc = zScale.value().toArray();
            FrequencyTable[] tables = new FrequencyTable[hyper];
            for (int c = 0; c < hyper; c++) {
                tables[c] = logisticTable(loc[c], sc[c]);
            }
            int hs = side / 8;
            for (int i = 0; i < zv.length; i++) {
                int s = clampSymbol(zv[i]);
                zq[i] = s;
                enc.encode(tables[(i / (hs * hs)) % hyper], s + RANGE);
            }
            float[] sigma = Var.noGrad(() -> scales(Var.of(Tensor.of(z.shape(), zq)))).value().toArray();
            for (int i = 0; i < ys.length; i++) {
                enc.encode(gaussianTable(scaleLevel(sigma[i])), ys[i] + RANGE);
            }
        }
        byte[] code = enc.finish();
        Tensor rec = Var.noGrad(() -> synthesis(Var.of(yhat))).value();
        return new Encoded(code, rec, 8.0 * code.length / ((double) n * side * side));
    }

    /**
     * Decodes {@link #encode}.
     *
     * @param code the bitstream
     * @param n the number of images
     * @return the reconstructions {@code [n, 1, side, side]}
     */
    public Tensor decode(byte[] code, int n) {
        ArithmeticCoder.Decoder dec = new ArithmeticCoder.Decoder(code);
        int ls = side / 4;
        float[] yq = new float[n * latent * ls * ls];
        if (prior == Prior.FACTORIZED) {
            float[] loc = yLoc.value().toArray();
            float[] sc = yScale.value().toArray();
            FrequencyTable[] tables = new FrequencyTable[latent];
            for (int c = 0; c < latent; c++) {
                tables[c] = logisticTable(loc[c], sc[c]);
            }
            for (int i = 0; i < yq.length; i++) {
                yq[i] = dec.decode(tables[(i / (ls * ls)) % latent]) - RANGE;
            }
        } else {
            int hs = side / 8;
            float[] zq = new float[n * hyper * hs * hs];
            float[] loc = zLoc.value().toArray();
            float[] sc = zScale.value().toArray();
            FrequencyTable[] tables = new FrequencyTable[hyper];
            for (int c = 0; c < hyper; c++) {
                tables[c] = logisticTable(loc[c], sc[c]);
            }
            for (int i = 0; i < zq.length; i++) {
                zq[i] = dec.decode(tables[(i / (hs * hs)) % hyper]) - RANGE;
            }
            float[] sigma = Var.noGrad(() -> scales(Var.of(Tensor.of(Shape.of(n, hyper, hs, hs), zq)))).value()
                    .toArray();
            for (int i = 0; i < yq.length; i++) {
                yq[i] = dec.decode(gaussianTable(scaleLevel(sigma[i]))) - RANGE;
            }
        }
        Tensor yhat = Tensor.of(Shape.of(n, latent, ls, ls), yq);
        return Var.noGrad(() -> synthesis(Var.of(yhat))).value();
    }

    /**
     * Peak signal-to-noise ratio for images in {@code [0, 1]}.
     *
     * @param a the reference
     * @param b the reconstruction
     * @return PSNR in dB
     */
    public static double psnr(Tensor a, Tensor b) {
        float[] x = a.toArray();
        float[] y = b.toArray();
        double s = 0;
        for (int i = 0; i < x.length; i++) {
            double d = x[i] - Math.max(0, Math.min(1, y[i]));
            s += d * d;
        }
        return 10 * Math.log10(1 / Math.max(1e-12, s / x.length));
    }
}
