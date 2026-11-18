package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.ElectraPretraining;
import io.github.zzuegg.brainiac.experimental.llm.StaticWordEmbeddings;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Embedding;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 6 item 2.15: ELMo-style contextual features. A forward and a backward two-layer LSTM
 * language model (core {@link Recurrent} LSTMs) are trained on the {@link SyntheticGrammar}
 * corpus and frozen. Each token's representation is a learned scalar mix
 * {@code gamma * sum_j softmax(s)_j h_j} of three layers (the token embeddings, and the first
 * and second LSTM layers, each the concatenation of both directions), trained together with a
 * linear tagger.
 *
 * <p>The downstream task tags every token as determiner, noun or verb from 200 labelled
 * sentences. Three words ({@code nim, vash, tral}) are verbs in one position and nouns in
 * another, so a context-free representation cannot tag them. Compared on the same linear
 * tagger: the frozen SGNS vectors of 2.14 and the frozen features of a masked-LM transformer
 * encoder trained with {@link ElectraPretraining#maskedLmLoss}. Reported: macro-F1 over the tags
 * and accuracy on the ambiguous tokens.
 */
public final class ContextualFeatures {

    /** The representation fed to the tagger. */
    public enum Features {
        /** frozen biLM with a learned scalar mix */
        ELMO,
        /** frozen SGNS word vectors */
        STATIC,
        /** frozen masked-LM encoder, last block */
        MASKED_LM
    }

    private static final int L = SyntheticGrammar.LENGTH;
    private static final int WIDTH = 32;

    private ContextualFeatures() {
    }

    /** One direction of the biLM. */
    private static final class LanguageModel {
        private final Layer embedding;
        private final Layer lstm1;
        private final Layer lstm2;
        private final Var out;
        private final Var outBias;

        LanguageModel(int vocab, SplittableRandom r) {
            embedding = Embedding.of(vocab, WIDTH).create(Shape.of(L), r);
            lstm1 = Recurrent.lstm(WIDTH).returnSequences(true).create(Shape.of(L, WIDTH), r);
            lstm2 = Recurrent.lstm(WIDTH).returnSequences(true).create(Shape.of(L, WIDTH), r);
            out = Var.param(Tensor.normal(Shape.of(WIDTH, vocab), 0, (float) Math.sqrt(1.0 / WIDTH), r));
            outBias = Var.param(Tensor.zeros(Shape.of(vocab)));
        }

        /** @return the three layers {@code [b, L, WIDTH]} for ids {@code [b, L]} */
        Var[] layers(Tensor ids, boolean training) {
            Var e = embedding.forward(Var.of(ids), training);
            Var h1 = lstm1.forward(e, training);
            Var h2 = lstm2.forward(h1, training);
            return new Var[] {e, h1, h2};
        }

        /** Next-token loss: position {@code t} predicts {@code t + 1}. */
        Var loss(Tensor ids, int b) {
            Var h = layers(ids, true)[2];
            Var logits = h.reshape(b * L, WIDTH).matmul(out).add(outBias);
            float[] x = ids.toArray();
            float[] y = new float[b * L];
            for (int i = 0; i < b; i++) {
                for (int t = 0; t < L; t++) {
                    y[i * L + t] = t + 1 < L ? x[i * L + t + 1] : -1;
                }
            }
            return Functional.crossEntropy(logits, Tensor.of(Shape.of(b * L), y));
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(embedding.parameters());
            p.addAll(lstm1.parameters());
            p.addAll(lstm2.parameters());
            p.add(out);
            p.add(outBias);
            return p;
        }
    }

    private static Tensor ids(int[][] tokens, boolean reversed) {
        float[] f = new float[tokens.length * L];
        for (int i = 0; i < tokens.length; i++) {
            for (int t = 0; t < L; t++) {
                f[i * L + t] = tokens[i][reversed ? L - 1 - t : t];
            }
        }
        return Tensor.of(Shape.of(tokens.length, L), f);
    }

    /**
     * Trains the forward and backward language models.
     *
     * @param g the grammar
     * @param steps Adam steps of 32 sentences for each direction
     * @param seed the seed
     * @return a feature extractor giving the three biLM layers {@code [n * L, 3, 2 * WIDTH]}
     */
    private static java.util.function.Function<int[][], Tensor> elmo(SyntheticGrammar g, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        LanguageModel fwd = new LanguageModel(g.size(), r);
        LanguageModel bwd = new LanguageModel(g.size(), r);
        Optimizer opt = Optimizer.adam(5e-3f);
        for (int s = 0; s < steps; s++) {
            int[][] t = g.sentences(32, r)[0];
            fwd.loss(ids(t, false), 32).add(bwd.loss(ids(t, true), 32)).backward();
            List<Var> p = new ArrayList<>(fwd.parameters());
            p.addAll(bwd.parameters());
            opt.step(p);
        }
        return tokens -> {
            int n = tokens.length;
            Var[] f = fwd.layers(ids(tokens, false), false);
            Var[] b = bwd.layers(ids(tokens, true), false);
            Var[] layers = new Var[3];
            for (int j = 0; j < 3; j++) {
                layers[j] = Var.concat(2, f[j], b[j].flip(1)).reshape(n * L, 1, 2 * WIDTH);
            }
            return Var.concat(1, layers).value();
        };
    }

    private static java.util.function.Function<int[][], Tensor> maskedLm(SyntheticGrammar g, int steps, long seed) {
        int mask = g.size();
        Model m = Model.sequential().input(L).embedding(g.size() + 1, WIDTH).positionalEncoding()
                .add(TransformerBlock.of(2).rotary(100f)).add(TransformerBlock.of(2).rotary(100f)).dense(g.size() + 1)
                .seed(seed).build();
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            int[][] t = g.sentences(32, r)[0];
            ElectraPretraining.maskedLmLoss(m, t, ElectraPretraining.maskPositions(32, L, 0.15f, r), mask)[0]
                    .backward();
            opt.step(m.parameters());
        }
        int last = m.graph().size() - 2;
        return tokens -> {
            Var[] captured = new Var[1];
            m.forward(Var.of(ids(tokens, false)), false, (step, out) -> {
                if (step.index() == last) {
                    captured[0] = out;
                }
                return out;
            });
            return captured[0].value().reshape(tokens.length * L, 1, WIDTH);
        };
    }

    private static java.util.function.Function<int[][], Tensor> staticVectors(SyntheticGrammar g, long seed) {
        StaticWordEmbeddings e = WordVectors.train(StaticWordEmbeddings.Kind.SGNS, g, seed);
        Tensor table = e.table();
        return tokens -> {
            float[] f = new float[tokens.length * L * e.dimension()];
            float[] t = table.toArray();
            for (int i = 0; i < tokens.length; i++) {
                for (int j = 0; j < L; j++) {
                    System.arraycopy(t, tokens[i][j] * e.dimension(), f, (i * L + j) * e.dimension(), e.dimension());
                }
            }
            return Tensor.of(Shape.of(tokens.length * L, 1, e.dimension()), f);
        };
    }

    /**
     * Pre-trains a representation, then trains a linear tagger (with a scalar mix over the
     * layers) on its frozen features.
     *
     * @param kind the representation
     * @param pretrainSteps the language-model steps (ELMo and masked LM)
     * @param seed the seed
     * @return {@code {macro-F1, accuracy on ambiguous tokens}} on 1000 test sentences
     */
    public static double[] run(Features kind, int pretrainSteps, long seed) {
        SyntheticGrammar g = new SyntheticGrammar();
        java.util.function.Function<int[][], Tensor> features = switch (kind) {
            case ELMO -> elmo(g, pretrainSteps, seed);
            case STATIC -> staticVectors(g, seed);
            case MASKED_LM -> maskedLm(g, pretrainSteps, seed);
        };
        SplittableRandom r = new SplittableRandom(seed + 7);
        int[][][] train = g.sentences(200, r);
        Tensor x = features.apply(train[0]);
        int layers = x.shape().dim(1);
        int width = x.shape().dim(2);
        Var mix = Var.param(Tensor.zeros(Shape.of(1, layers, 1)));
        Var gamma = Var.param(Tensor.full(Shape.of(1), 1f));
        Var w = Var.param(Tensor.zeros(Shape.of(width, SyntheticGrammar.TAGS)));
        Var b = Var.param(Tensor.zeros(Shape.of(SyntheticGrammar.TAGS)));
        Optimizer opt = Optimizer.adam(1e-2f);
        Tensor y = tags(train[1]);
        for (int s = 0; s < 300; s++) {
            Functional.crossEntropy(tag(Var.of(x), mix, gamma, w, b), y).backward();
            opt.step(List.of(mix, gamma, w, b));
        }
        int[][][] test = g.sentences(1000, r);
        float[] pred = tag(Var.of(features.apply(test[0])), mix, gamma, w, b).value().argmax(1).toArray();
        long[][] confusion = new long[SyntheticGrammar.TAGS][SyntheticGrammar.TAGS];
        int ambiguous = 0;
        int ambiguousHit = 0;
        List<Integer> ambiguousIds = new ArrayList<>();
        for (String a : SyntheticGrammar.ambiguous()) {
            ambiguousIds.add(g.id(a));
        }
        for (int i = 0; i < test[0].length; i++) {
            for (int t = 0; t < L; t++) {
                int truth = test[1][i][t];
                int p = (int) pred[i * L + t];
                confusion[truth][p]++;
                if (ambiguousIds.contains(test[0][i][t])) {
                    ambiguous++;
                    ambiguousHit += truth == p ? 1 : 0;
                }
            }
        }
        double f1 = 0;
        for (int c = 0; c < SyntheticGrammar.TAGS; c++) {
            long tp = confusion[c][c];
            long fp = 0;
            long fn = 0;
            for (int o = 0; o < SyntheticGrammar.TAGS; o++) {
                if (o != c) {
                    fp += confusion[o][c];
                    fn += confusion[c][o];
                }
            }
            f1 += tp == 0 ? 0 : 2.0 * tp / (2.0 * tp + fp + fn);
        }
        return new double[] {f1 / SyntheticGrammar.TAGS, (double) ambiguousHit / ambiguous};
    }

    private static Var tag(Var x, Var mix, Var gamma, Var w, Var b) {
        int n = x.shape().dim(0);
        Var weights = Activation.SOFTMAX.forward(mix.reshape(1, mix.shape().dim(1))).reshape(1, mix.shape().dim(1), 1);
        Var mixed = x.mul(weights.expand(x.shape())).sum(1, false).mul(gamma);
        return mixed.reshape(n, x.shape().dim(2)).matmul(w).add(b);
    }

    private static Tensor tags(int[][] tags) {
        float[] f = new float[tags.length * L];
        for (int i = 0; i < tags.length; i++) {
            for (int t = 0; t < L; t++) {
                f[i * L + t] = tags[i][t];
            }
        }
        return Tensor.of(Shape.of(tags.length * L), f);
    }
}
