package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * ELECTRA pre-training, replaced-token detection (Clark et al., <a href="https://arxiv.org/abs/2003.10555">2003.10555</a>).
 * A small <em>generator</em> is trained as a masked language model; its samples replace the
 * masked tokens, and the <em>discriminator</em> (the encoder that is kept) learns to tell,
 * for every position, whether the token is original or replaced. The discriminator gets a
 * learning signal from all positions instead of the ~15% that a masked language model
 * scores, which is where ELECTRA's compute efficiency comes from. Like the paper, a
 * replaced token that happens to equal the original counts as original.
 *
 * <p>Both networks are ordinary {@link Model}s over {@code [batch, length]} token ids: the
 * generator outputs {@code [batch, length, vocabulary]} logits, the discriminator one logit
 * per position, {@code [batch, length, 1]}. {@link #maskedLmLoss} is also the plain
 * masked-LM objective, the usual baseline.
 */
public final class ElectraPretraining {
    private final Model generator;
    private final Model discriminator;
    private final int vocabulary;
    private final int maskId;
    private float maskRate = 0.15f;
    private float discriminatorWeight = 50f;
    private final Optimizer optimizer;

    private ElectraPretraining(Model generator, Model discriminator, int vocabulary, int maskId, float learningRate) {
        this.generator = generator;
        this.discriminator = discriminator;
        this.vocabulary = vocabulary;
        this.maskId = maskId;
        this.optimizer = Optimizer.adam(learningRate);
    }

    /**
     * Creates the trainer.
     *
     * @param generator ids to {@code [batch, length, vocabulary]} logits, usually small
     * @param discriminator ids to {@code [batch, length, 1]} replaced-token logits
     * @param vocabulary the vocabulary size, including the mask token
     * @param maskId the mask token id
     * @param learningRate the Adam learning rate for both networks
     * @return the trainer
     */
    public static ElectraPretraining of(Model generator, Model discriminator, int vocabulary, int maskId,
            float learningRate) {
        if (maskId < 0 || maskId >= vocabulary) {
            throw new IllegalArgumentException("mask id outside the vocabulary");
        }
        return new ElectraPretraining(generator, discriminator, vocabulary, maskId, learningRate);
    }

    /**
     * Sets the share of positions masked, 0.15 by default.
     *
     * @param rate the rate
     * @return this trainer
     */
    public ElectraPretraining maskRate(float rate) {
        this.maskRate = rate;
        return this;
    }

    /**
     * Sets the weight of the discriminator loss, 50 by default as in the paper.
     *
     * @param weight the weight
     * @return this trainer
     */
    public ElectraPretraining discriminatorWeight(float weight) {
        this.discriminatorWeight = weight;
        return this;
    }

    /** @return the discriminator, the encoder to keep */
    public Model discriminator() {
        return discriminator;
    }

    /**
     * Picks masked positions: each with probability {@code rate}, at least one per sequence.
     *
     * @param batch the batch size
     * @param length the sequence length
     * @param rate the masking probability
     * @param random the random source
     * @return {@code masked[batch][length]}
     */
    public static boolean[][] maskPositions(int batch, int length, float rate, RandomGenerator random) {
        boolean[][] m = new boolean[batch][length];
        for (int i = 0; i < batch; i++) {
            boolean any = false;
            for (int j = 0; j < length; j++) {
                m[i][j] = random.nextDouble() < rate;
                any |= m[i][j];
            }
            if (!any) {
                m[i][random.nextInt(length)] = true;
            }
        }
        return m;
    }

    private static Tensor ids(int[][] rows) {
        int len = rows[0].length;
        float[] f = new float[rows.length * len];
        for (int i = 0; i < rows.length; i++) {
            for (int j = 0; j < len; j++) {
                f[i * len + j] = rows[i][j];
            }
        }
        return Tensor.of(Shape.of(rows.length, len), f);
    }

    /**
     * The masked-LM loss: cross-entropy of the original tokens at masked positions.
     *
     * @param model ids to {@code [batch, length, vocabulary]} logits
     * @param tokens clean sequences
     * @param masked which positions are masked
     * @param maskId the mask token id
     * @return the loss and, as a side product, the logits
     */
    public static Var[] maskedLmLoss(Model model, int[][] tokens, boolean[][] masked, int maskId) {
        int b = tokens.length;
        int len = tokens[0].length;
        int[][] in = new int[b][len];
        int count = 0;
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < len; j++) {
                in[i][j] = masked[i][j] ? maskId : tokens[i][j];
                count += masked[i][j] ? 1 : 0;
            }
        }
        Var logits = model.forward(Var.of(ids(in)), true);
        int vocab = logits.shape().dim(-1);
        float[] w = new float[b * len * vocab];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < len; j++) {
                if (masked[i][j]) {
                    w[(i * len + j) * vocab + tokens[i][j]] = 1f / count;
                }
            }
        }
        Var loss = Activation.LOG_SOFTMAX.forward(logits.reshape(b * len, vocab))
                .mul(Tensor.of(Shape.of(b * len, vocab), w)).sum().neg();
        return new Var[] {loss, logits};
    }

    /**
     * One ELECTRA step on a batch: generator MLM loss plus the weighted replaced-token
     * detection loss of the discriminator, one optimizer step on both networks.
     *
     * @param tokens clean sequences
     * @param random the source of masks and generator samples
     * @return {@code {generator loss, discriminator loss, share of positions replaced}}
     */
    public float[] step(int[][] tokens, RandomGenerator random) {
        int b = tokens.length;
        int len = tokens[0].length;
        boolean[][] masked = maskPositions(b, len, maskRate, random);
        Var[] mlm = maskedLmLoss(generator, tokens, masked, maskId);
        float[] logits = mlm[1].value().toArray();
        int[][] corrupted = new int[b][len];
        float[] replaced = new float[b * len];
        int changed = 0;
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < len; j++) {
                int t = tokens[i][j];
                if (masked[i][j]) {
                    t = sample(logits, (i * len + j) * vocabulary, random);
                }
                corrupted[i][j] = t;
                replaced[i * len + j] = t != tokens[i][j] ? 1 : 0;
                changed += t != tokens[i][j] ? 1 : 0;
            }
        }
        Var d = discriminator.forward(Var.of(ids(corrupted)), true).reshape(b * len, 1);
        Tensor y = Tensor.of(Shape.of(b * len, 1), replaced);
        // binary cross-entropy with logits: softplus(d) - y d
        Var bce = d.softplus().sub(d.mul(y)).mean();
        Var loss = mlm[0].add(bce.mul(discriminatorWeight));
        float g = mlm[0].value().item();
        float dl = bce.value().item();
        loss.backward();
        List<Var> params = new ArrayList<>(generator.parameters());
        params.addAll(discriminator.parameters());
        optimizer.step(params);
        return new float[] {g, dl, (float) changed / (b * len)};
    }

    private int sample(float[] logits, int offset, RandomGenerator random) {
        double max = Double.NEGATIVE_INFINITY;
        for (int v = 0; v < vocabulary; v++) {
            if (v != maskId) {
                max = Math.max(max, logits[offset + v]);
            }
        }
        double[] p = new double[vocabulary];
        double sum = 0;
        for (int v = 0; v < vocabulary; v++) {
            p[v] = v == maskId ? 0 : Math.exp(logits[offset + v] - max);
            sum += p[v];
        }
        double u = random.nextDouble() * sum;
        for (int v = 0; v < vocabulary; v++) {
            u -= p[v];
            if (u <= 0 && v != maskId) {
                return v;
            }
        }
        return maskId == 0 ? 1 : 0;
    }
}
