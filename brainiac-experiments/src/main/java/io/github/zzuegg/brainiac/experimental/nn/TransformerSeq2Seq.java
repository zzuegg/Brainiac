package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A small transformer encoder-decoder built from core {@link TransformerBlock}s: token
 * embeddings plus sinusoidal positions, {@code layers} bidirectional encoder blocks, and
 * {@code layers} causal decoder blocks with cross-attention to the encoder output
 * ({@code TransformerBlock.Spec.crossAttention}). It is the "Attention Is All You Need"
 * architecture at toy scale, the baseline for the recurrent seq2seq models and the
 * encoder-decoder of T5-style span corruption. Target id {@code targetVocab} is the
 * begin-of-sequence token.
 *
 * <p>Paper: Vaswani et al., "Attention Is All You Need" (2017),
 * <a href="https://arxiv.org/abs/1706.03762">arXiv:1706.03762</a>.
 */
public final class TransformerSeq2Seq {

    private final int width;
    private final int targetVocab;
    private final Var sourceEmbedding;
    private final Var targetEmbedding;
    private final List<Layer> encoder = new ArrayList<>();
    private final List<MultiInputLayer> decoder = new ArrayList<>();
    private final Var output;
    private final Var outputBias;

    private TransformerSeq2Seq(int sourceVocab, int targetVocab, int width, int heads, int layers, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.width = width;
        this.targetVocab = targetVocab;
        this.sourceEmbedding = Var.param(Tensor.normal(Shape.of(sourceVocab, width), 0, 0.3f, r));
        this.targetEmbedding = Var.param(Tensor.normal(Shape.of(targetVocab + 1, width), 0, 0.3f, r));
        Shape seq = Shape.of(8, width);
        for (int i = 0; i < layers; i++) {
            encoder.add(TransformerBlock.of(heads).ffnSize(2 * width).create(seq, r));
            decoder.add(TransformerBlock.of(heads).ffnSize(2 * width).causal(true).crossAttention(true)
                    .create(List.of(seq, seq), r));
        }
        this.output = ClassicOps.matrix(r, width, targetVocab);
        this.outputBias = ClassicOps.zeros(targetVocab);
    }

    /**
     * Creates the model.
     *
     * @param sourceVocab the source vocabulary
     * @param targetVocab the target vocabulary (one more id is begin-of-sequence)
     * @param width the model width
     * @param heads the attention heads
     * @param layers the encoder blocks, and as many decoder blocks
     * @param seed the initialization seed
     * @return the model
     */
    public static TransformerSeq2Seq of(int sourceVocab, int targetVocab, int width, int heads, int layers,
            long seed) {
        return new TransformerSeq2Seq(sourceVocab, targetVocab, width, heads, layers, seed);
    }

    private Var embed(Var table, int[][] ids) {
        int steps = ids[0].length;
        Tensor pos = Functional.sinusoidal(steps, width, 10000f, table.value().device());
        return table.gather(0, ClassicOps.ids(ids)).add(pos);
    }

    /**
     * Encodes sources.
     *
     * @param source {@code [batch][steps]} ids
     * @param training whether dropout-style layers train
     * @return {@code [batch, steps, width]}
     */
    public Var encode(int[][] source, boolean training) {
        Var x = embed(sourceEmbedding, source);
        for (Layer l : encoder) {
            x = l.forward(x, training);
        }
        return x;
    }

    /**
     * Decoder logits for given decoder inputs.
     *
     * @param memory the encoder output
     * @param inputs {@code [batch][length]} decoder input ids (begin-of-sequence first)
     * @param training whether dropout-style layers train
     * @return {@code [batch, length, targetVocab]}
     */
    public Var decode(Var memory, int[][] inputs, boolean training) {
        Var y = embed(targetEmbedding, inputs);
        for (MultiInputLayer l : decoder) {
            y = l.forward(List.of(y, memory), training);
        }
        return y.matmul(output).add(outputBias);
    }

    /**
     * Teacher-forced cross-entropy.
     *
     * @param source {@code [batch][steps]} ids
     * @param target {@code [batch][length]} ids; negative entries are not scored
     * @return the mean loss
     */
    public Var loss(int[][] source, int[][] target) {
        int b = target.length;
        int len = target[0].length;
        int[][] in = new int[b][len];
        float[] ids = new float[b * len];
        for (int i = 0; i < b; i++) {
            in[i][0] = targetVocab;
            for (int t = 0; t < len; t++) {
                if (t > 0) {
                    in[i][t] = Math.max(0, target[i][t - 1]);
                }
                ids[i * len + t] = target[i][t];
            }
        }
        Var logits = decode(encode(source, true), in, true).reshape(b * len, targetVocab);
        return Functional.crossEntropy(logits, Tensor.of(Shape.of(b * len), ids));
    }

    /**
     * Greedy decoding.
     *
     * @param source {@code [batch][steps]} ids
     * @param length the tokens to emit
     * @return {@code [batch][length]} ids
     */
    public int[][] greedy(int[][] source, int length) {
        int b = source.length;
        Var memory = encode(source, false);
        int[][] out = new int[b][length];
        for (int t = 0; t < length; t++) {
            int[][] in = new int[b][t + 1];
            for (int i = 0; i < b; i++) {
                in[i][0] = targetVocab;
                System.arraycopy(out[i], 0, in[i], 1, t);
            }
            Var logits = decode(memory, in, false);
            int[] next = ClassicOps.argmax(logits.slice(1, t, t + 1).reshape(b, targetVocab).value());
            for (int i = 0; i < b; i++) {
                out[i][t] = next[i];
            }
        }
        return out;
    }

    /** @return the encoder's trainable values (embedding and blocks) */
    public List<Var> encoderParameters() {
        List<Var> p = new ArrayList<>(List.of(sourceEmbedding));
        for (Layer l : encoder) {
            p.addAll(l.parameters());
        }
        return p;
    }

    /** @return all trainable values */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(encoderParameters());
        p.add(targetEmbedding);
        for (MultiInputLayer l : decoder) {
            p.addAll(l.parameters());
        }
        p.add(output);
        p.add(outputBias);
        return p;
    }

    @Override
    public String toString() {
        return "TransformerSeq2Seq(width " + width + ", " + encoder.size() + " + " + decoder.size() + " blocks)";
    }
}
