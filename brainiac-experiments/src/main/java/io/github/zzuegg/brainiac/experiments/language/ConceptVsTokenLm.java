package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.nn.ConceptLm;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A Large Concept Model ({@link ConceptLm}) against a token language model on documents of
 * arithmetic sentences, at an equal training budget in transformer positions. A document is
 * four sentences {@code x+b=y} that chain: the next sentence starts with the last result and
 * keeps the step, {@code y+b=z} (all mod 10). The concept model sees one fixed random-feature
 * embedding per sentence and regresses the next one; the token model
 * ({@link HookedDecoder}, same width and depth) sees the 24 characters. Each training step
 * covers the same number of positions (the concept model's batches are six times larger),
 * so both spend about the same FLOPs. Scored on predicting sentences 2 to 4 from the ones
 * before: exact-sentence accuracy (the concept model decodes to the nearest of the 100
 * possible sentences) and network calls per sentence.
 */
@SuppressWarnings("exports")
public final class ConceptVsTokenLm {
    /** Sentences per document. */
    public static final int SENTENCES = 4;
    private static final int PLUS = 10;
    private static final int EQUALS = 11;
    private static final int STOP = 12;
    private static final int VOCAB = 13;
    private static final int WORDS = 5;
    private final ConceptLm.SentenceEncoder encoder = ConceptLm.SentenceEncoder.randomFeatures(VOCAB, WORDS, 32, 3);
    private final List<int[]> allSentences = new ArrayList<>();

    private ConceptVsTokenLm() {
        for (int x = 0; x < 10; x++) {
            for (int b = 0; b < 10; b++) {
                allSentences.add(sentence(x, b));
            }
        }
    }

    /**
     * Creates the experiment.
     *
     * @return the experiment
     */
    public static ConceptVsTokenLm create() {
        return new ConceptVsTokenLm();
    }

    private static int[] sentence(int x, int b) {
        return new int[] {x, PLUS, b, EQUALS, (x + b) % 10};
    }

    private static int[][][] documents(int n, SplittableRandom r) {
        int[][][] docs = new int[n][SENTENCES][];
        for (int i = 0; i < n; i++) {
            int x = r.nextInt(10);
            int b = 1 + r.nextInt(9);
            for (int s = 0; s < SENTENCES; s++) {
                docs[i][s] = sentence(x, b);
                x = (x + b) % 10;
            }
        }
        return docs;
    }

    private static int[] tokens(int[][] doc, int sentences) {
        int[] t = new int[sentences * (WORDS + 1)];
        for (int s = 0; s < sentences; s++) {
            System.arraycopy(doc[s], 0, t, s * (WORDS + 1), WORDS);
            t[s * (WORDS + 1) + WORDS] = STOP;
        }
        return t;
    }

    /**
     * Trains the concept model.
     *
     * @param steps optimizer steps
     * @param batch documents per step
     * @param seed the seed
     * @return the model
     */
    public ConceptLm trainConcept(int steps, int batch, long seed) {
        ConceptLm lcm = ConceptLm.of(SENTENCES, encoder.dim(), 32, 2, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            Var loss = lcm.loss(ConceptLm.embed(encoder, documents(batch, r)));
            loss.backward();
            opt.step(lcm.parameters());
        }
        return lcm;
    }

    /**
     * Trains the token model.
     *
     * @param steps optimizer steps
     * @param batch documents per step
     * @param seed the seed
     * @return the model
     */
    public HookedDecoder trainTokens(int steps, int batch, long seed) {
        HookedDecoder lm = HookedDecoder.create(HookedDecoder.Settings.tiny(VOCAB), seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            int[][][] docs = documents(batch, r);
            int[][] t = new int[batch][];
            for (int i = 0; i < batch; i++) {
                t[i] = tokens(docs[i], SENTENCES);
            }
            Var loss = lm.loss(t, true);
            loss.backward();
            opt.step(lm.parameters());
            lm.zeroGrad();
        }
        return lm;
    }

    /**
     * Next-sentence accuracy of the concept model.
     *
     * @param lcm the model
     * @param seed the evaluation seed
     * @return the share of sentences 2..4 decoded exactly from the prediction
     */
    public double conceptAccuracy(ConceptLm lcm, long seed) {
        int[][][] docs = documents(128, new SplittableRandom(seed));
        Tensor pred = lcm.predict(ConceptLm.embed(encoder, docs));
        float[] p = pred.toArray();
        int d = encoder.dim();
        int hit = 0;
        for (int i = 0; i < docs.length; i++) {
            for (int s = 0; s < SENTENCES - 1; s++) {
                float[] v = java.util.Arrays.copyOfRange(p, (i * SENTENCES + s) * d, (i * SENTENCES + s + 1) * d);
                int k = ConceptLm.nearest(encoder, v, allSentences);
                hit += java.util.Arrays.equals(allSentences.get(k), docs[i][s + 1]) ? 1 : 0;
            }
        }
        return (double) hit / (docs.length * (SENTENCES - 1));
    }

    /**
     * Next-sentence accuracy of the token model, writing each sentence greedily.
     *
     * @param lm the model
     * @param seed the evaluation seed
     * @return the share of sentences 2..4 written exactly
     */
    public double tokenAccuracy(HookedDecoder lm, long seed) {
        int[][][] docs = documents(128, new SplittableRandom(seed));
        int hit = 0;
        for (int s = 1; s < SENTENCES; s++) {
            int[][] seq = new int[docs.length][];
            for (int i = 0; i < docs.length; i++) {
                seq[i] = tokens(docs[i], s);
            }
            int[][] written = new int[docs.length][WORDS];
            for (int w = 0; w < WORDS; w++) {
                float[] logits = lm.forward(seq).logits().value().toArray();
                int len = seq[0].length;
                for (int i = 0; i < docs.length; i++) {
                    int off = ((i + 1) * len - 1) * VOCAB;
                    int best = 0;
                    for (int v = 1; v < VOCAB; v++) {
                        if (logits[off + v] > logits[off + best]) {
                            best = v;
                        }
                    }
                    written[i][w] = best;
                    seq[i] = java.util.Arrays.copyOf(seq[i], len + 1);
                    seq[i][len] = best;
                }
            }
            for (int i = 0; i < docs.length; i++) {
                hit += java.util.Arrays.equals(written[i], docs[i][s]) ? 1 : 0;
            }
        }
        return (double) hit / (docs.length * (SENTENCES - 1));
    }

    /** @return transformer positions per document for the token model over the concept model */
    public static int positionRatio() {
        return WORDS + 1;
    }
}
