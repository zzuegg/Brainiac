package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * A Large Concept Model in its base form (LCM team, Meta, <a href="https://arxiv.org/abs/2412.08821">2412.08821</a>):
 * autoregression over <em>sentence embeddings</em> instead of tokens. A frozen sentence
 * encoder (SONAR in the paper) turns each sentence into a vector; a causal transformer reads
 * the vectors of the sentences so far and regresses the next one with a squared error
 * ("Base-LCM"); a decoder (here: the nearest known sentence) turns the predicted vector back
 * into text. One position per sentence makes the sequence several times shorter than its
 * tokens, so the same network costs that many times fewer FLOPs per document.
 *
 * <p>Input {@code [batch, sentences, dim]}; the model is a core {@link Model} (a linear
 * projection, causal {@link TransformerBlock}s with rotary positions, a linear read-out).
 */
public final class ConceptLm {
    private final int sentences;
    private final int dim;
    private final Model model;

    private ConceptLm(int sentences, int dim, int width, int layers, long seed) {
        this.sentences = sentences;
        this.dim = dim;
        Model.Builder b = Model.sequential().input(sentences, dim).dense(width);
        for (int i = 0; i < layers; i++) {
            b.add(TransformerBlock.of(2).causal(true).rotary(100f));
        }
        this.model = b.dense(dim).seed(seed).build();
    }

    /**
     * Creates a model.
     *
     * @param sentences the number of sentences per document
     * @param dim the sentence-embedding size
     * @param width the transformer width
     * @param layers the transformer blocks
     * @param seed the seed
     * @return the model
     */
    public static ConceptLm of(int sentences, int dim, int width, int layers, long seed) {
        return new ConceptLm(sentences, dim, width, layers, seed);
    }

    /** @return the underlying network */
    public Model model() {
        return model;
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        return model.parameters();
    }

    /**
     * Base-LCM's loss: the squared distance of each position's prediction to the next
     * sentence's embedding, averaged over positions.
     *
     * @param documents {@code [batch, sentences, dim]} sentence embeddings
     * @return the loss, a scalar
     */
    public Var loss(Tensor documents) {
        Var out = model.forward(Var.of(documents), true);
        Var prediction = out.slice(1, 0, sentences - 1);
        Tensor target = documents.slice(1, 1, sentences);
        return prediction.sub(Var.of(target)).square().mean().mul(dim);   // squared distance per sentence
    }

    /**
     * Predicts the embedding following each position.
     *
     * @param documents {@code [batch, sentences, dim]}; positions after the ones of interest
     *     may hold anything, the model is causal
     * @return {@code [batch, sentences, dim]}: row {@code t} predicts sentence {@code t + 1}
     */
    public Tensor predict(Tensor documents) {
        return model.forward(Var.of(documents), false).value();
    }

    /**
     * Turns sentences of token ids into documents for the model with a fixed encoder.
     *
     * @param encoder the sentence encoder
     * @param docs {@code [batch][sentences][tokens]}
     * @return {@code [batch, sentences, dim]}
     */
    public static Tensor embed(SentenceEncoder encoder, int[][][] docs) {
        int s = docs[0].length;
        int d = encoder.dim();
        float[] f = new float[docs.length * s * d];
        for (int i = 0; i < docs.length; i++) {
            for (int j = 0; j < s; j++) {
                System.arraycopy(encoder.encode(docs[i][j]), 0, f, (i * s + j) * d, d);
            }
        }
        return Tensor.of(Shape.of(docs.length, s, d), f);
    }

    /** A frozen map from a sentence's tokens to a vector. */
    public interface SentenceEncoder {

        /**
         * Embeds a sentence.
         *
         * @param tokens the sentence
         * @return its vector
         */
        float[] encode(int[] tokens);

        /** @return the vector size */
        int dim();

        /**
         * A fixed random-feature encoder: the sum over positions of a Gaussian vector per
         * (position, token), scaled by {@code 1 / sqrt(length)}. Distinct sentences get nearly
         * orthogonal codes, and nearest-neighbour decoding over a candidate list is exact.
         *
         * @param vocabulary the token vocabulary
         * @param length the longest sentence
         * @param dim the vector size
         * @param seed the seed
         * @return the encoder
         */
        static SentenceEncoder randomFeatures(int vocabulary, int length, int dim, long seed) {
            java.util.SplittableRandom r = new java.util.SplittableRandom(seed);
            float[][] table = new float[vocabulary * length][dim];
            for (float[] row : table) {
                for (int k = 0; k < dim; k++) {
                    row[k] = (float) (r.nextGaussian() / Math.sqrt(dim));
                }
            }
            return new SentenceEncoder() {
                @Override
                public float[] encode(int[] tokens) {
                    float[] v = new float[dim];
                    for (int p = 0; p < tokens.length; p++) {
                        float[] row = table[p * vocabulary + tokens[p]];
                        for (int k = 0; k < dim; k++) {
                            v[k] += row[k] / (float) Math.sqrt(tokens.length);
                        }
                    }
                    return v;
                }

                @Override
                public int dim() {
                    return dim;
                }
            };
        }
    }

    /**
     * Decodes a vector to the nearest candidate sentence.
     *
     * @param encoder the encoder
     * @param vector the vector
     * @param candidates the sentences to choose from
     * @return the index of the nearest candidate
     */
    public static int nearest(SentenceEncoder encoder, float[] vector, List<int[]> candidates) {
        int best = -1;
        double bestD = Double.MAX_VALUE;
        for (int c = 0; c < candidates.size(); c++) {
            float[] e = encoder.encode(candidates.get(c));
            double d = 0;
            for (int k = 0; k < e.length; k++) {
                d += (e[k] - vector[k]) * (e[k] - vector[k]);
            }
            if (d < bestD) {
                bestD = d;
                best = c;
            }
        }
        return best;
    }
}
