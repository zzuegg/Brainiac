package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Residual vector quantization (RVQ) as in SoundStream (Zeghidour et al.,
 * <a href="https://arxiv.org/abs/2107.03312">2107.03312</a>): a cascade of
 * {@link VectorQuantizer}s where stage {@code i} quantizes what the stages before it left
 * over. {@code n} codebooks of {@code K} codes describe {@code K^n} points for the cost of
 * {@code n K} vectors, and using only the first stages gives a coarser, cheaper code: one
 * model, many bitrates. <em>Quantizer dropout</em> (training with a random number of
 * stages) is what makes those prefixes good.
 *
 * <p>The output carries straight-through gradients to the input as a whole.
 */
public final class ResidualVectorQuantizer implements Layer {
    private final List<VectorQuantizer> stages = new ArrayList<>();
    private final int dim;

    /**
     * What one quantization produced.
     *
     * @param output the sum of the stage codes, with straight-through gradients
     * @param codes {@code codes[stage][row]}
     * @param loss the summed codebook and commitment losses of the stages used
     */
    public record Result(Var output, int[][] codes, Var loss) {
    }

    private ResidualVectorQuantizer(int stages, int size, int dim, float beta, long seed) {
        if (stages < 1) {
            throw new IllegalArgumentException("at least one stage");
        }
        SplittableRandom r = new SplittableRandom(seed);
        this.dim = dim;
        for (int i = 0; i < stages; i++) {
            this.stages.add(VectorQuantizer.of(size, dim, beta, r.nextLong()));
        }
    }

    /**
     * Creates a residual quantizer.
     *
     * @param stages the number of codebooks
     * @param size the codes per codebook
     * @param dim the vector size
     * @param beta the commitment weight
     * @param seed the seed
     * @return the quantizer
     */
    public static ResidualVectorQuantizer of(int stages, int size, int dim, float beta, long seed) {
        return new ResidualVectorQuantizer(stages, size, dim, beta, seed);
    }

    /**
     * Describes the layer for model builders.
     *
     * @param stages the number of codebooks
     * @param size the codes per codebook
     * @param beta the commitment weight
     * @return the spec
     */
    public static LayerSpec spec(int stages, int size, float beta) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom r) {
                return of(stages, size, input.dim(-1), beta, r.nextLong());
            }
        };
    }

    /** @return the stages, coarse to fine */
    public List<VectorQuantizer> stages() {
        return List.copyOf(stages);
    }

    /**
     * Quantizes with the first {@code used} stages.
     *
     * @param x {@code [batch, dim]}
     * @param used how many stages to use, 1..stages
     * @return the output, codes and loss
     */
    public Result quantize(Var x, int used) {
        if (used < 1 || used > stages.size()) {
            throw new IllegalArgumentException("use 1.." + stages.size() + " stages, got " + used);
        }
        Tensor residual = x.value();
        Tensor sum = Tensor.zeros(x.shape()).to(residual.device());
        Var loss = null;
        int[][] codes = new int[used][];
        for (int s = 0; s < used; s++) {
            // each stage sees its residual as a constant; the encoder is trained through the total
            VectorQuantizer.Result r = stages.get(s).quantize(Var.of(residual));
            Tensor q = stages.get(s).lookup(r.codes()).value();
            codes[s] = r.codes();
            sum = sum.add(q);
            residual = residual.sub(q);
            loss = loss == null ? r.loss() : loss.add(r.loss());
        }
        // the commitment of the encoder to the total code
        float beta = 0.25f;
        Var commitment = x.sub(Var.of(sum)).square().mean().mul(beta);
        Var out = Functional.straightThrough(Var.of(sum), x);
        return new Result(out, codes, loss.add(commitment));
    }

    /**
     * Quantizes with a random number of stages (quantizer dropout), for training.
     *
     * @param x {@code [batch, dim]}
     * @param random picks the number of stages
     * @return the output, codes and loss
     */
    public Result quantizeWithDropout(Var x, SplittableRandom random) {
        return quantize(x, 1 + random.nextInt(stages.size()));
    }

    /**
     * Rebuilds vectors from codes.
     *
     * @param codes {@code codes[stage][row]}, any prefix of the stages
     * @return {@code [rows, dim]}
     */
    public Tensor decode(int[][] codes) {
        Tensor sum = null;
        for (int s = 0; s < codes.length; s++) {
            Tensor q = stages.get(s).lookup(codes[s]).value();
            sum = sum == null ? q : sum.add(q);
        }
        return sum;
    }

    @Override
    public Var forward(Var input) {
        return quantize(input, stages.size()).output();
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (VectorQuantizer q : stages) {
            p.addAll(q.parameters());
        }
        return p;
    }

    /** @return the vector size */
    public int dim() {
        return dim;
    }

    /**
     * Restarts dead codes in every stage, re-seeding each stage from its own residuals.
     *
     * @param x recent encoder outputs {@code [n, dim]}
     * @return the number of codes restarted
     */
    public int restartDeadCodes(Tensor x) {
        Tensor residual = x;
        int total = 0;
        for (VectorQuantizer q : stages) {
            int[] codes = q.nearest(residual);
            Tensor next = residual.sub(q.lookup(codes).value());
            total += q.restartDeadCodes(residual);
            residual = next;
        }
        return total;
    }
}
