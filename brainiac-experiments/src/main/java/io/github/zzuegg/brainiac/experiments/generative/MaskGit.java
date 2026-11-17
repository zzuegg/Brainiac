package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.PatchVqTokenizer;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MaskGitDecoder;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * MaskGIT against autoregressive decoding on the 8x8 {@link ToyImages}. Both work on the
 * same VQ tokens ({@link PatchVqTokenizer}: 2x2 tiles, 16 codes, a 4x4 grid of tokens per
 * image) with the <em>same network</em>: a token MLP that sees every position (masked ones
 * as a mask id) and predicts every position, trained for the same number of steps. Only the
 * training masks and the decoding order differ:
 *
 * <ul>
 *   <li>MaskGIT ({@link MaskGitDecoder}): cosine-schedule random masks; decoding fills all
 *       16 tokens in a few parallel, confidence-ordered steps;</li>
 *   <li>autoregressive: every position after a random cut is masked and the model predicts
 *       the token at the cut; decoding takes one network call per token in raster order.</li>
 * </ul>
 *
 * <p>Decoded samples are scored against the enumerable data set (an FID stand-in): distance
 * to the nearest real image, the share that are real images, and how many distinct images
 * and classes appear.
 */
@SuppressWarnings("exports")
public final class MaskGit {
    private final PatchVqTokenizer tokenizer;
    private final int length;
    private final int vocab;
    private final int mask;
    private final Model parallel;
    private final Model sequential;
    private final MaskGitDecoder maskGit;
    private final SplittableRandom random;
    private final Optimizer maskGitOptimizer = Optimizer.adam(2e-3f);
    private final Optimizer arOptimizer = Optimizer.adam(2e-3f);

    private MaskGit(int hidden, long seed) {
        tokenizer = TokenModels.toyImageTokenizer(seed);
        length = tokenizer.tokens();
        vocab = tokenizer.vocabulary() + 1;
        mask = vocab - 1;
        parallel = tokenMlp(length, vocab, hidden, seed);
        sequential = tokenMlp(length, vocab, hidden, seed);
        maskGit = MaskGitDecoder.of(parallel, vocab, mask);
        random = new SplittableRandom(seed + 1);
    }

    /**
     * Creates both models.
     *
     * @param hidden the hidden width of the token MLP
     * @param seed the seed
     * @return the experiment
     */
    public static MaskGit of(int hidden, long seed) {
        return new MaskGit(hidden, seed);
    }

    /**
     * A bidirectional token network: {@code [batch, length]} ids to
     * {@code [batch, length, vocab]} logits through an MLP over all embedded positions.
     *
     * @param length the sequence length
     * @param vocab the vocabulary
     * @param hidden the hidden width
     * @param seed the seed
     * @return the model
     */
    static Model tokenMlp(int length, int vocab, int hidden, long seed) {
        return Model.sequential().input(length).embedding(vocab, 16).flatten().dense(hidden, Activation.SILU)
                .dense(hidden, Activation.SILU).dense(length * vocab).reshape(length, vocab).seed(seed).build();
    }

    /** @return the image tokenizer */
    public PatchVqTokenizer tokenizer() {
        return tokenizer;
    }

    private int[][] batch(int n) {
        int[][] out = new int[n][];
        for (int i = 0; i < n; i++) {
            out[i] = tokenizer.encode(ToyImages.sample(random.nextInt(ToyImages.CLASSES), random));
        }
        return out;
    }

    /**
     * Trains the MaskGIT model.
     *
     * @param steps optimizer steps
     * @param batch the batch size
     * @return the final loss
     */
    public float trainMaskGit(int steps, int batch) {
        float loss = 0;
        for (int s = 0; s < steps; s++) {
            Var l = maskGit.loss(batch(batch), random);
            loss = l.value().item();
            l.backward();
            maskGitOptimizer.step(parallel.parameters());
        }
        return loss;
    }

    /**
     * Trains the autoregressive model: per sequence a random cut, everything from the cut on
     * masked, and the cross-entropy of the token at the cut.
     *
     * @param steps optimizer steps
     * @param batch the batch size
     * @return the final loss
     */
    public float trainAutoregressive(int steps, int batch) {
        float loss = 0;
        for (int s = 0; s < steps; s++) {
            int[][] seqs = batch(batch);
            int[][] in = new int[batch][length];
            float[] weight = new float[batch * length * vocab];
            for (int i = 0; i < batch; i++) {
                int cut = random.nextInt(length);
                for (int j = 0; j < length; j++) {
                    in[i][j] = j < cut ? seqs[i][j] : mask;
                }
                weight[(i * length + cut) * vocab + seqs[i][cut]] = 1f / batch;
            }
            Var logits = sequential.forward(Var.of(TokenModels.ids(in)), true).reshape(batch * length, vocab);
            Var l = Activation.LOG_SOFTMAX.forward(logits).mul(Tensor.of(Shape.of(batch * length, vocab), weight))
                    .sum().neg();
            loss = l.value().item();
            l.backward();
            arOptimizer.step(sequential.parameters());
        }
        return loss;
    }

    private List<float[]> decode(int[][] tokens) {
        List<float[]> out = new ArrayList<>();
        for (int[] t : tokens) {
            out.add(tokenizer.decode(t));
        }
        return out;
    }

    /**
     * Samples with MaskGIT.
     *
     * @param count the number of images
     * @param steps parallel decoding steps
     * @param choiceTemperature the Gumbel noise scale on confidences
     * @param seed the seed
     * @return the decoded images
     */
    public List<float[]> sampleMaskGit(int count, int steps, float choiceTemperature, long seed) {
        return decode(maskGit.sample(count, length, steps, 1f, choiceTemperature, new SplittableRandom(seed), null));
    }

    /**
     * Samples with the autoregressive model, one token per network call in raster order.
     *
     * @param count the number of images
     * @param seed the seed
     * @return the decoded images
     */
    public List<float[]> sampleAutoregressive(int count, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[][] x = new int[count][length];
        for (int[] row : x) {
            java.util.Arrays.fill(row, mask);
        }
        for (int pos = 0; pos < length; pos++) {
            float[] logits = sequential.forward(Var.of(TokenModels.ids(x)), false).value().toArray();
            for (int i = 0; i < count; i++) {
                x[i][pos] = TokenModels.draw(logits, (i * length + pos) * vocab, mask, 1f, r);
            }
        }
        return decode(x);
    }

    /**
     * Scores decoded images (see {@link MaskGit}).
     *
     * @param images generated images
     * @return {@code {distance per pixel, real share, distinct real images, classes}}
     */
    public static double[] score(List<float[]> images) {
        return TokenModels.score(images);
    }
}
