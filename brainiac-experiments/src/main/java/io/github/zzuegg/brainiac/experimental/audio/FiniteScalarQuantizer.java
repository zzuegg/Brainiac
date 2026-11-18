package io.github.zzuegg.brainiac.experimental.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Finite scalar quantization (FSQ, Mentzer et al., ICLR 2024,
 * <a href="https://arxiv.org/abs/2309.15505">arXiv:2309.15505</a>): a drop-in replacement for
 * a VQ-VAE's vector quantizer that has no codebook at all. Each of the {@code d} latent
 * dimensions is bounded with {@code tanh} to {@code L_i} levels and rounded to the nearest
 * level, with the straight-through estimator for the gradient. The implicit codebook is the
 * product grid of {@code prod L_i} points, every one of which is reachable, so there is no
 * codebook loss, no commitment term, no dead-code restarts and no collapse.
 *
 * <p>The bound follows the paper's reference code: with {@code h = (L - 1)(1 - eps) / 2} and
 * an offset of one half for even {@code L}, {@code z = tanh(x + atanh(offset / h)) h - offset},
 * so that rounding {@code z} gives exactly {@code L} integer levels; the output is divided by
 * {@code floor(L / 2)} to lie in {@code [-1, 1]}. A code id is the mixed-radix number of the
 * per-dimension level indices, so {@link #codes} and {@link #lookup} turn quantized vectors
 * into token ids for a MaskGIT or language model on top and back again.
 *
 * <p>As a {@link Layer} it maps {@code [..., d]} to {@code [..., d]}; {@link #usage} counts how
 * often each code was produced since the last {@link #resetUsage}, from which
 * {@link #perplexity} and {@link #utilisation} follow.
 */
public final class FiniteScalarQuantizer implements Layer {

    private static final double EPS = 1e-3;
    private final int[] levels;
    private final int size;
    private final double[] half;
    private final double[] offset;
    private final double[] shift;
    private final long[] usage;

    /**
     * Creates a quantizer.
     *
     * @param levels the levels per dimension, for example {@code {8, 5, 5, 5}} for about 2^10 codes
     */
    public FiniteScalarQuantizer(int... levels) {
        if (levels.length == 0) {
            throw new IllegalArgumentException("FSQ needs at least one dimension");
        }
        this.levels = levels.clone();
        long n = 1;
        half = new double[levels.length];
        offset = new double[levels.length];
        shift = new double[levels.length];
        for (int i = 0; i < levels.length; i++) {
            if (levels[i] < 2) {
                throw new IllegalArgumentException("every dimension needs at least 2 levels");
            }
            n *= levels[i];
            half[i] = (levels[i] - 1) * (1 - EPS) / 2;
            offset[i] = levels[i] % 2 == 0 ? 0.5 : 0;
            shift[i] = atanh(offset[i] / half[i]);
        }
        if (n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("too many codes: " + n);
        }
        this.size = (int) n;
        this.usage = new long[size];
    }

    private static double atanh(double x) {
        return 0.5 * Math.log((1 + x) / (1 - x));
    }

    /**
     * A spec for {@code Model.sequential()} graphs; the input's last dimension must equal the
     * number of levels.
     *
     * @param levels the levels per dimension
     * @return the spec
     */
    public static LayerSpec spec(int... levels) {
        int[] l = levels.clone();
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                if (input.dim(-1) != l.length) {
                    throw new IllegalArgumentException("FSQ with " + l.length + " levels needs " + l.length
                            + " input features, got " + input);
                }
                return new FiniteScalarQuantizer(l);
            }
        };
    }

    /** @return the levels per dimension */
    public int[] levels() {
        return levels.clone();
    }

    /** @return the number of dimensions */
    public int dimension() {
        return levels.length;
    }

    /** @return the size of the implicit codebook, {@code prod L_i} */
    public int codebookSize() {
        return size;
    }

    /**
     * The outcome of quantizing.
     *
     * @param output the quantized values in {@code [-1, 1]}, with straight-through gradients
     * @param codes the code id of every row
     */
    public record Result(Var output, int[] codes) {
    }

    /**
     * Quantizes the rows of {@code x}.
     *
     * @param x {@code [..., d]}
     * @return the quantized values and the code ids
     */
    public Result quantize(Var x) {
        int d = levels.length;
        Shape shape = x.shape();
        if (shape.dim(-1) != d) {
            throw new IllegalArgumentException("FSQ with " + d + " dimensions got " + shape);
        }
        float[] sh = new float[d];
        float[] hf = new float[d];
        float[] of = new float[d];
        float[] scale = new float[d];
        for (int i = 0; i < d; i++) {
            sh[i] = (float) shift[i];
            hf[i] = (float) half[i];
            of[i] = (float) offset[i];
            scale[i] = 1f / (levels[i] / 2);
        }
        Var z = x.add(Tensor.of(Shape.of(d), sh)).tanh().mul(Tensor.of(Shape.of(d), hf)).sub(Tensor.of(Shape.of(d), of));
        float[] zv = z.value().toArray();
        float[] delta = new float[zv.length];
        int rows = zv.length / d;
        int[] codes = new int[rows];
        for (int r = 0; r < rows; r++) {
            int code = 0;
            for (int i = 0; i < d; i++) {
                float q = Math.round(zv[r * d + i]);
                delta[r * d + i] = q - zv[r * d + i];
                int digit = Math.max(0, Math.min(levels[i] - 1, (int) q + levels[i] / 2));
                code = code * levels[i] + digit;
            }
            codes[r] = code;
            usage[code]++;
        }
        Var q = z.add(Tensor.of(shape, delta)).mul(Tensor.of(Shape.of(d), scale));
        return new Result(q, codes);
    }

    @Override
    public Var forward(Var input) {
        return quantize(input).output();
    }

    /**
     * The code ids of quantized rows, without counting usage.
     *
     * @param x {@code [..., d]} raw (unbounded) latents
     * @return the code id per row
     */
    public int[] codes(Tensor x) {
        long[] before = usage.clone();
        int[] c = Var.noGrad(() -> quantize(Var.of(x))).codes();
        System.arraycopy(before, 0, usage, 0, usage.length);
        return c;
    }

    /**
     * The quantized vectors of code ids: the inverse of {@link #codes}.
     *
     * @param codes code ids
     * @return {@code [codes.length, d]} values in {@code [-1, 1]}
     */
    public Tensor lookup(int[] codes) {
        int d = levels.length;
        float[] out = new float[codes.length * d];
        for (int r = 0; r < codes.length; r++) {
            int c = codes[r];
            for (int i = d - 1; i >= 0; i--) {
                int digit = c % levels[i];
                c /= levels[i];
                out[r * d + i] = (float) (digit - levels[i] / 2) / (levels[i] / 2);
            }
        }
        return Tensor.of(Shape.of(codes.length, d), out);
    }

    /** @return how often each code was produced since the last reset */
    public long[] usage() {
        return usage.clone();
    }

    /** Clears the usage counts. */
    public void resetUsage() {
        java.util.Arrays.fill(usage, 0);
    }

    /**
     * The perplexity {@code exp(H)} of a usage histogram: the effective number of codes used.
     *
     * @param usage counts per code
     * @return the perplexity
     */
    public static double perplexity(long[] usage) {
        double total = 0;
        for (long u : usage) {
            total += u;
        }
        double h = 0;
        for (long u : usage) {
            if (u > 0) {
                double p = u / total;
                h -= p * Math.log(p);
            }
        }
        return Math.exp(h);
    }

    /**
     * The share of codes used at least once.
     *
     * @param usage counts per code
     * @return the share in {@code [0, 1]}
     */
    public static double utilisation(long[] usage) {
        int used = 0;
        for (long u : usage) {
            used += u > 0 ? 1 : 0;
        }
        return used / (double) usage.length;
    }

    /** @return the perplexity of this quantizer's usage */
    public double perplexity() {
        return perplexity(usage);
    }

    /** @return the share of this quantizer's codes used */
    public double utilisation() {
        return utilisation(usage);
    }

    @Override
    public List<Var> parameters() {
        return List.of();
    }

    @Override
    public String toString() {
        return "FiniteScalarQuantizer" + java.util.Arrays.toString(levels);
    }
}
