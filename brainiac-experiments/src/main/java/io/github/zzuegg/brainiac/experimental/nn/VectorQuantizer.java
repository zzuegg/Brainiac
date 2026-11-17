package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Vector quantization as in VQ-VAE (van den Oord et al.,
 * <a href="https://arxiv.org/abs/1711.00937">1711.00937</a>): every input row snaps to its
 * nearest entry of a learned codebook. Snapping has no useful gradient, so the forward
 * value is the code but the gradient passes <em>straight through</em> to the input:
 * {@code out = x + stopgrad(code - x)}.
 *
 * <p>The codebook learns from {@link Result#loss()}: the codebook term
 * {@code |stopgrad(x) - code|^2} pulls codes towards the inputs that chose them, and the
 * commitment term {@code beta |x - stopgrad(code)|^2} keeps the encoder near its codes.
 * {@link #restartDeadCodes} re-seeds codes that no input used from the current inputs, the
 * usual cure for codebook collapse, and {@link #initialize} seeds the codebook from data
 * with k-means++.
 *
 * <p>As a {@link Layer} it maps {@code [batch, dim]} to {@code [batch, dim]}; call
 * {@link #quantize} to also get the codes and the loss.
 */
public final class VectorQuantizer implements Layer {
    private final Var codebook;
    private final int size;
    private final int dim;
    private final float beta;
    private final SplittableRandom random;
    private int[] usage;

    /**
     * What one quantization produced.
     *
     * @param output the quantized rows with straight-through gradients
     * @param codes the chosen code per row
     * @param loss the codebook plus commitment loss, a scalar
     */
    public record Result(Var output, int[] codes, Var loss) {
    }

    private VectorQuantizer(int size, int dim, float beta, long seed) {
        if (size < 1 || dim < 1 || beta < 0) {
            throw new IllegalArgumentException("bad VectorQuantizer settings");
        }
        this.size = size;
        this.dim = dim;
        this.beta = beta;
        this.random = new SplittableRandom(seed);
        this.codebook = Var.param(Tensor.uniform(Shape.of(size, dim), -1f / size, 1f / size, random));
        this.usage = new int[size];
    }

    /**
     * Creates a quantizer.
     *
     * @param size the number of codes
     * @param dim the vector size
     * @param beta the commitment weight, typically 0.25
     * @param seed the seed for the codebook and restarts
     * @return the quantizer
     */
    public static VectorQuantizer of(int size, int dim, float beta, long seed) {
        return new VectorQuantizer(size, dim, beta, seed);
    }

    /**
     * Describes the layer for model builders.
     *
     * @param size the number of codes
     * @param beta the commitment weight
     * @return the spec
     */
    public static LayerSpec spec(int size, float beta) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom r) {
                return of(size, input.dim(-1), beta, r.nextLong());
            }
        };
    }

    /** @return the codebook, {@code [size, dim]} */
    public Var codebook() {
        return codebook;
    }

    /** @return how often each code was chosen since the last restart */
    public int[] usage() {
        return usage.clone();
    }

    /**
     * Finds the nearest code of every row.
     *
     * @param x {@code [batch, dim]}
     * @return the code indices
     */
    public int[] nearest(Tensor x) {
        float[] v = x.toArray();
        float[] c = codebook.value().toArray();
        int n = v.length / dim;
        int[] codes = new int[n];
        for (int i = 0; i < n; i++) {
            double best = Double.MAX_VALUE;
            for (int k = 0; k < size; k++) {
                double d = 0;
                for (int j = 0; j < dim; j++) {
                    double e = v[i * dim + j] - c[k * dim + j];
                    d += e * e;
                }
                if (d < best) {
                    best = d;
                    codes[i] = k;
                }
            }
        }
        return codes;
    }

    /**
     * Looks up codes.
     *
     * @param codes code indices
     * @return {@code [codes.length, dim]} code vectors (differentiable in the codebook)
     */
    public Var lookup(int[] codes) {
        float[] idx = new float[codes.length];
        for (int i = 0; i < codes.length; i++) {
            idx[i] = codes[i];
        }
        return codebook.gather(0, Tensor.of(Shape.of(codes.length), idx).to(codebook.value().device()));
    }

    /**
     * Quantizes with straight-through gradients.
     *
     * @param x {@code [batch, dim]}
     * @return the output, codes and loss
     */
    public Result quantize(Var x) {
        if (x.shape().rank() != 2 || x.shape().dim(1) != dim) {
            throw new IllegalArgumentException("VectorQuantizer expects [batch, " + dim + "], got " + x.shape());
        }
        int[] codes = nearest(x.value());
        for (int c : codes) {
            usage[c]++;
        }
        Var q = lookup(codes);
        Var codebookLoss = Var.of(x.value()).sub(q).square().mean();
        Var commitment = x.sub(Var.of(q.value())).square().mean().mul(beta);
        Var out = x.add(Var.of(q.value().sub(x.value())));   // straight-through
        return new Result(out, codes, codebookLoss.add(commitment));
    }

    @Override
    public Var forward(Var input) {
        return quantize(input).output();
    }

    @Override
    public List<Var> parameters() {
        return List.of(codebook);
    }

    /**
     * Seeds the codebook from data with k-means++ (each next code is a data row picked with
     * probability proportional to its squared distance from the codes so far), which
     * spreads the codes over the data and avoids most dead or doubled-up codes.
     *
     * @param x encoder outputs {@code [n, dim]}, ideally many more rows than codes
     */
    public void initialize(Tensor x) {
        float[] v = x.toArray();
        int n = v.length / dim;
        float[] c = new float[size * dim];
        double[] d2 = new double[n];
        java.util.Arrays.fill(d2, Double.MAX_VALUE);
        int pick = random.nextInt(n);
        for (int k = 0; k < size; k++) {
            System.arraycopy(v, pick * dim, c, k * dim, dim);
            double total = 0;
            for (int i = 0; i < n; i++) {
                double d = 0;
                for (int j = 0; j < dim; j++) {
                    double e = v[i * dim + j] - c[k * dim + j];
                    d += e * e;
                }
                d2[i] = Math.min(d2[i], d);
                total += d2[i];
            }
            double u = random.nextDouble() * total;
            pick = n - 1;
            for (int i = 0; i < n; i++) {
                u -= d2[i];
                if (u <= 0) {
                    pick = i;
                    break;
                }
            }
        }
        codebook.assign(Tensor.of(Shape.of(size, dim), c).to(codebook.value().device()));
        usage = new int[size];
    }

    /**
     * Moves every code that was not used since the last call onto a random row of
     * {@code x}, then resets the usage counts.
     *
     * @param x recent encoder outputs {@code [n, dim]}
     * @return the number of codes restarted
     */
    public int restartDeadCodes(Tensor x) {
        float[] v = x.toArray();
        int n = v.length / dim;
        float[] c = codebook.value().toArray();
        int restarted = 0;
        for (int k = 0; k < size; k++) {
            if (usage[k] == 0) {
                int row = random.nextInt(n);
                System.arraycopy(v, row * dim, c, k * dim, dim);
                restarted++;
            }
        }
        if (restarted > 0) {
            codebook.assign(Tensor.of(Shape.of(size, dim), c).to(codebook.value().device()));
        }
        usage = new int[size];
        return restarted;
    }
}
