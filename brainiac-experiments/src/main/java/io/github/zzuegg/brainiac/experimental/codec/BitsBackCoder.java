package io.github.zzuegg.brainiac.experimental.codec;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Bits-back coding with asymmetric numeral systems (BB-ANS; Townsend, Bird and Barber, ICLR 2019,
 * <a href="https://arxiv.org/abs/1901.04866">arXiv:1901.04866</a>) over a small variational
 * autoencoder for binary images. A latent variable model gives only an ELBO, not a likelihood;
 * bits-back coding nevertheless reaches the negative ELBO by using the {@link Rans} stack:
 *
 * <ol>
 *   <li>pop a latent {@code z ~ q(z | x)} from the stack, which <em>decodes</em> it from bits
 *       already there (gaining {@code -log q(z|x)} bits back),</li>
 *   <li>push {@code x} under the decoder's Bernoulli {@code p(x | z)},</li>
 *   <li>push {@code z} under the prior {@code p(z)}.</li>
 * </ol>
 *
 * <p>The net cost per image is {@code -log p(x|z) - log p(z) + log q(z|x)}, whose expectation is
 * the negative ELBO. Decoding runs the steps backwards and pushes the posterior sample back,
 * restoring the bits it borrowed. Latents are discretised into {@code 2^b} buckets of equal mass
 * under the standard-normal prior (so the prior costs exactly {@code b} bits per dimension), the
 * posterior bucket probabilities come from the Gaussian posterior's CDF at the bucket edges, and
 * the decoder is evaluated at the bucket centres. The very first image needs bits that are not
 * there yet: they are drawn as clean random bits ({@link Rans#borrowedBits()}) and amortised over
 * a chain of images.
 */
public final class BitsBackCoder {

    private static final int PRECISION = 16;
    private static final int PIXEL_PRECISION = 12;

    private final int pixels;
    private final int latent;
    private final int bucketBits;
    private final double[] edges;
    private final float[] centres;
    private final Var e1;
    private final Var e1b;
    private final Var eMu;
    private final Var eMub;
    private final Var eLv;
    private final Var eLvb;
    private final Var d1;
    private final Var d1b;
    private final Var d2;
    private final Var d2b;

    private BitsBackCoder(int pixels, int latent, int hidden, int bucketBits, long seed) {
        this.pixels = pixels;
        this.latent = latent;
        this.bucketBits = bucketBits;
        SplittableRandom r = new SplittableRandom(seed);
        e1 = Var.param(Tensor.normal(Shape.of(pixels, hidden), 0, (float) Math.sqrt(1.0 / pixels), r));
        e1b = Var.param(Tensor.zeros(Shape.of(1, hidden)));
        eMu = Var.param(Tensor.normal(Shape.of(hidden, latent), 0, (float) Math.sqrt(1.0 / hidden), r));
        eMub = Var.param(Tensor.zeros(Shape.of(1, latent)));
        eLv = Var.param(Tensor.normal(Shape.of(hidden, latent), 0, 0.01f, r));
        eLvb = Var.param(Tensor.zeros(Shape.of(1, latent)));
        d1 = Var.param(Tensor.normal(Shape.of(latent, hidden), 0, (float) Math.sqrt(1.0 / latent), r));
        d1b = Var.param(Tensor.zeros(Shape.of(1, hidden)));
        d2 = Var.param(Tensor.normal(Shape.of(hidden, pixels), 0, (float) Math.sqrt(1.0 / hidden), r));
        d2b = Var.param(Tensor.zeros(Shape.of(1, pixels)));
        int buckets = 1 << bucketBits;
        edges = new double[buckets + 1];
        centres = new float[buckets];
        edges[0] = Double.NEGATIVE_INFINITY;
        edges[buckets] = Double.POSITIVE_INFINITY;
        for (int i = 1; i < buckets; i++) {
            edges[i] = inverseNormal(i / (double) buckets);
        }
        for (int i = 0; i < buckets; i++) {
            centres[i] = (float) inverseNormal((i + 0.5) / buckets);
        }
    }

    /**
     * Trains a VAE on binary images and returns its coder.
     *
     * @param images binary images, one row of 0/1 values each
     * @param latent the latent dimension
     * @param hidden the hidden width
     * @param bucketBits the latent precision {@code b} (bits per latent dimension under the prior)
     * @param steps Adam steps
     * @param batch the minibatch size
     * @param learningRate the Adam step size
     * @param seed the seed
     * @return the coder
     */
    public static BitsBackCoder train(float[][] images, int latent, int hidden, int bucketBits, int steps,
            int batch, float learningRate, long seed) {
        BitsBackCoder c = new BitsBackCoder(images[0].length, latent, hidden, bucketBits, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        List<Var> params = c.parameters();
        Optimizer opt = Optimizer.adam(learningRate);
        for (int s = 0; s < steps; s++) {
            float[][] b = new float[batch][];
            for (int i = 0; i < batch; i++) {
                b[i] = images[r.nextInt(images.length)];
            }
            Var x = Var.of(Tensor.of(b));
            Var[] q = c.posterior(x);
            float[] eps = new float[batch * latent];
            for (int i = 0; i < eps.length; i++) {
                eps[i] = (float) r.nextGaussian();
            }
            Var z = q[0].add(q[1].mul(0.5f).exp().mul(Tensor.of(Shape.of(batch, latent), eps)));
            Var loss = c.negElbo(x, q, z).div(batch);
            loss.backward();
            opt.step(params);
        }
        return c;
    }

    private List<Var> parameters() {
        return List.of(e1, e1b, eMu, eMub, eLv, eLvb, d1, d1b, d2, d2b);
    }

    private Var[] posterior(Var x) {
        Var h = x.matmul(e1).add(e1b).tanh();
        return new Var[] {h.matmul(eMu).add(eMub), h.matmul(eLv).add(eLvb).clamp(-12, 4)};
    }

    private Var logits(Var z) {
        return z.matmul(d1).add(d1b).tanh().matmul(d2).add(d2b);
    }

    /** The negative ELBO in nats, summed over the batch (one reparameterised sample). */
    private Var negElbo(Var x, Var[] q, Var z) {
        Var l = logits(z);
        // Bernoulli log-likelihood: x l - softplus(l)
        Var rec = l.softplus().sub(l.mul(x)).sum();
        Var kl = q[1].exp().add(q[0].square()).sub(1f).sub(q[1]).mul(0.5f).sum();
        return rec.add(kl);
    }

    /**
     * The negative ELBO in bits per image, averaged over images with {@code samples} posterior
     * samples each (the coder's target rate).
     *
     * @param images binary images
     * @param samples posterior samples
     * @param seed the seed
     * @return bits per image
     */
    public double negElboBits(float[][] images, int samples, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double total = 0;
        Var x = Var.of(Tensor.of(images));
        Var[] q = Var.noGrad(() -> posterior(x));
        for (int s = 0; s < samples; s++) {
            float[] eps = new float[images.length * latent];
            for (int i = 0; i < eps.length; i++) {
                eps[i] = (float) r.nextGaussian();
            }
            Tensor e = Tensor.of(Shape.of(images.length, latent), eps);
            total += Var.noGrad(() -> negElbo(x, q, q[0].add(q[1].mul(0.5f).exp().mul(e)))).value().item();
        }
        return total / samples / images.length / Math.log(2);
    }

    // ------------------------------------------------------------------ coding

    private FrequencyTable[] posteriorTables(int[] image) {
        float[] row = new float[pixels];
        for (int i = 0; i < pixels; i++) {
            row[i] = image[i];
        }
        Var[] q = Var.noGrad(() -> posterior(Var.of(Tensor.of(Shape.of(1, pixels), row))));
        float[] mu = q[0].value().toArray();
        float[] lv = q[1].value().toArray();
        FrequencyTable[] t = new FrequencyTable[latent];
        int buckets = 1 << bucketBits;
        for (int j = 0; j < latent; j++) {
            double sd = Math.exp(0.5 * lv[j]);
            double[] p = new double[buckets];
            double prev = 0;
            for (int b = 0; b < buckets; b++) {
                double cdf = b == buckets - 1 ? 1 : normalCdf((edges[b + 1] - mu[j]) / sd);
                p[b] = Math.max(0, cdf - prev);
                prev = cdf;
            }
            t[j] = FrequencyTable.of(p, PRECISION);
        }
        return t;
    }

    private float[] pixelProbabilities(int[] buckets) {
        float[] z = new float[latent];
        for (int j = 0; j < latent; j++) {
            z[j] = centres[buckets[j]];
        }
        float[] l = Var.noGrad(() -> logits(Var.of(Tensor.of(Shape.of(1, latent), z)))).value().toArray();
        float[] p = new float[pixels];
        for (int i = 0; i < pixels; i++) {
            p[i] = (float) (1 / (1 + Math.exp(-l[i])));
        }
        return p;
    }

    private static FrequencyTable bernoulli(float p) {
        return FrequencyTable.of(new double[] {1 - p, p}, PIXEL_PRECISION);
    }

    /**
     * The coding result.
     *
     * @param message the rANS message
     * @param bits the message length in bits
     * @param borrowedBits clean random bits drawn while coding the first images
     * @param bitsAfter the message length after each image
     */
    public record Encoded(byte[] message, long bits, long borrowedBits, long[] bitsAfter) {
    }

    /**
     * Codes a chain of images with BB-ANS.
     *
     * @param images binary images (0/1 per pixel)
     * @param seed the seed of the clean bits
     * @return the message and its length
     */
    public Encoded encode(int[][] images, long seed) {
        Rans rans = new Rans(seed);
        long[] after = new long[images.length];
        int prior = PRECISION - bucketBits;
        for (int n = 0; n < images.length; n++) {
            int[] x = images[n];
            FrequencyTable[] q = posteriorTables(x);
            int[] z = new int[latent];
            for (int j = 0; j < latent; j++) {
                z[j] = rans.pop(q[j]);
            }
            float[] p = pixelProbabilities(z);
            for (int i = 0; i < pixels; i++) {
                rans.push(bernoulli(p[i]), x[i]);
            }
            for (int j = latent - 1; j >= 0; j--) {
                rans.push(z[j] << prior, 1 << prior, PRECISION);
            }
            after[n] = rans.bits();
        }
        return new Encoded(rans.toBytes(), rans.bits(), rans.borrowedBits(), after);
    }

    /**
     * Decodes {@link #encode}.
     *
     * @param message the message
     * @param count the number of images
     * @return the images, in their original order
     */
    public int[][] decode(byte[] message, int count) {
        Rans rans = Rans.fromBytes(message);
        int[][] out = new int[count][];
        int prior = PRECISION - bucketBits;
        for (int n = count - 1; n >= 0; n--) {
            int[] z = new int[latent];
            for (int j = 0; j < latent; j++) {
                int b = rans.peek(PRECISION) >>> prior;
                rans.pop(b << prior, 1 << prior, PRECISION);
                z[j] = b;
            }
            float[] p = pixelProbabilities(z);
            int[] x = new int[pixels];
            for (int i = pixels - 1; i >= 0; i--) {
                x[i] = rans.pop(bernoulli(p[i]));
            }
            FrequencyTable[] q = posteriorTables(x);
            for (int j = latent - 1; j >= 0; j--) {
                rans.push(q[j], z[j]);
            }
            out[n] = x;
        }
        return out;
    }

    /**
     * The standard normal quantile (Acklam's rational approximation, refined by one Halley step).
     *
     * @param p a probability in {@code (0, 1)}
     * @return {@code Φ^-1(p)}
     */
    public static double inverseNormal(double p) {
        double[] a = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
            -3.066479806614716e+01, 2.506628277459239e+00};
        double[] b = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
            -1.328068155288572e+01};
        double[] c = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
            4.374664141464968e+00, 2.938163982698783e+00};
        double[] d = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00};
        double x;
        if (p < 0.02425) {
            double q = Math.sqrt(-2 * Math.log(p));
            x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        } else if (p > 1 - 0.02425) {
            double q = Math.sqrt(-2 * Math.log(1 - p));
            x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        } else {
            double q = p - 0.5;
            double r = q * q;
            x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                    / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
        }
        double e = normalCdf(x) - p;
        double u = e * Math.sqrt(2 * Math.PI) * Math.exp(x * x / 2);
        return x - u / (1 + x * u / 2);
    }

    private static double normalCdf(double z) {
        // Φ via the complementary error function (Numerical Recipes erfc, |error| < 1.2e-7)
        double t = 1 / (1 + 0.5 * Math.abs(z / Math.sqrt(2)));
        double v = z / Math.sqrt(2);
        double erfc = t * Math.exp(-v * v - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
                + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587
                + t * (-0.82215223 + t * 0.17087277)))))))));
        return v >= 0 ? 1 - erfc / 2 : erfc / 2;
    }
}
