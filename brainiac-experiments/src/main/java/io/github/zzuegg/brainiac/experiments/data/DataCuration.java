package io.github.zzuegg.brainiac.experiments.data;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.DataMixture;
import io.github.zzuegg.brainiac.experimental.data.DataPruner;
import io.github.zzuegg.brainiac.experimental.data.EntityGraphAugmenter;
import io.github.zzuegg.brainiac.experimental.data.QualityClassifierFilter;
import io.github.zzuegg.brainiac.experimental.data.RegressionMixer;
import io.github.zzuegg.brainiac.experimental.train.SelectiveTokenLoss;
import io.github.zzuegg.brainiac.experiments.language.CountingTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.IntFunction;
import java.util.random.RandomGenerator;

/**
 * Data curation for language-model pretraining on a toy corpus, every method against its
 * usual baseline, all with the {@link ContextLm} toy model at a fixed training budget:
 *
 * <ul>
 *   <li>{@link #selectiveTokens}: Rho-1 ({@link SelectiveTokenLoss}) on counting sequences
 *       with 30% noise tokens, against full-token cross-entropy and token pruning by EL2N
 *       ({@link DataPruner}); clean validation loss.</li>
 *   <li>{@link #qualityFilter}: a {@link QualityClassifierFilter} keeping half of a pool of
 *       grammatical and corrupted documents, against no filter and a perplexity filter;
 *       clean validation loss.</li>
 *   <li>{@link #mixture}: {@link RegressionMixer} (RegMix) on four domains for a target
 *       that mixes two of them, against uniform and DoReMi
 *       ({@link DataMixture#updateWithExcessLoss}) mixtures; target loss.</li>
 *   <li>{@link #factLearning}: EntiGraph ({@link EntityGraphAugmenter}) on a few documents
 *       of facts, against the raw corpus repeated and simple paraphrases; fact-question
 *       accuracy, forward, reversed and two-hop.</li>
 * </ul>
 * The corpus: documents of a second-order Markov {@link Grammar} over a small vocabulary,
 * clean or corrupted by random token replacement, from one or several domains.
 */
@SuppressWarnings("exports")   // experimental.* gets exported when the lead integrates it
public final class DataCuration {

    private DataCuration() {
    }

    /**
     * A second-order Markov "grammar": each pair of previous tokens has two or three likely
     * next tokens.
     *
     * @param vocab the vocabulary size
     * @param table {@code [vocab * vocab][vocab]} next-token probabilities
     */
    public record Grammar(int vocab, double[][] table) {

        /**
         * @param vocab the vocabulary size
         * @param seed the seed
         * @return a random grammar
         */
        public static Grammar random(int vocab, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            double[][] t = new double[vocab * vocab][vocab];
            for (double[] row : t) {
                double sum = 0;
                for (int k = 0; k < vocab; k++) {
                    row[k] = 0.01;
                    sum += row[k];
                }
                int likely = 2 + r.nextInt(2);
                for (int k = 0; k < likely; k++) {
                    double w = 1 + 3 * r.nextDouble();
                    row[r.nextInt(vocab)] += w;
                    sum += w;
                }
                for (int k = 0; k < vocab; k++) {
                    row[k] /= sum;
                }
            }
            return new Grammar(vocab, t);
        }

        /**
         * @param length the tokens
         * @param r the source of randomness
         * @return one document
         */
        public int[] document(int length, SplittableRandom r) {
            int[] d = new int[length];
            d[0] = r.nextInt(vocab);
            d[1] = r.nextInt(vocab);
            for (int t = 2; t < length; t++) {
                double[] row = table[d[t - 2] * vocab + d[t - 1]];
                double u = r.nextDouble();
                int k = 0;
                while (k < vocab - 1 && (u -= row[k]) > 0) {
                    k++;
                }
                d[t] = k;
            }
            return d;
        }

        /**
         * @param n the documents
         * @param length tokens per document
         * @param r the source of randomness
         * @return the documents
         */
        public List<int[]> documents(int n, int length, SplittableRandom r) {
            List<int[]> out = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                out.add(document(length, r));
            }
            return out;
        }
    }

    /**
     * Replaces each token with a uniformly random one with probability {@code rate}.
     *
     * @param doc the document
     * @param rate the corruption rate
     * @param vocab the vocabulary size
     * @param r the source of randomness
     * @return a corrupted copy
     */
    public static int[] corrupt(int[] doc, double rate, int vocab, SplittableRandom r) {
        int[] d = doc.clone();
        for (int t = 0; t < d.length; t++) {
            if (r.nextDouble() < rate) {
                d[t] = r.nextInt(vocab);
            }
        }
        return d;
    }

    // ---------------------------------------------------------------- Rho-1

    /** How the noisy counting corpus is trained on in {@link #selectiveTokens}. */
    public enum TokenSelection {
        /** Cross-entropy on every token. */
        FULL,
        /** Rho-1: the 60% of each batch's tokens with the largest excess loss. */
        RHO1,
        /** The 30% hardest tokens by EL2N dropped before training ({@link DataPruner}). */
        EL2N_PRUNED
    }

    private static final int COUNT_VOCAB = 16;
    private static final int[] STRIDES = {1, 3, 5};

    private static List<int[]> counting(int n, long seed) {
        return Arrays.asList(CountingTasks.sequences(new SplittableRandom(seed), n, 12, COUNT_VOCAB, STRIDES));
    }

    /**
     * Trains on counting sequences whose tokens are replaced by noise with probability
     * {@code noise}, then measures the loss on clean sequences. Rho-1's reference model is
     * trained on 16 clean sequences.
     *
     * @param method how tokens are selected
     * @param noise the share of noise tokens, for example 0.3
     * @param steps training steps of 64 tokens
     * @param seed the seed
     * @return the clean validation loss
     */
    public static float selectiveTokens(TokenSelection method, double noise, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<int[]> train = new ArrayList<>();
        for (int[] d : counting(300, seed)) {
            train.add(corrupt(d, noise, COUNT_VOCAB, r));
        }
        List<int[]> valid = counting(100, seed + 1);
        Tensor[] data = ContextLm.examples(COUNT_VOCAB, train);
        ContextLm lm = new ContextLm(COUNT_VOCAB, 48, seed);
        switch (method) {
            case FULL -> lm.train(data, steps, 64, 1e-2f, seed + 2);
            case EL2N_PRUNED -> {
                DataPruner pruner = new DataPruner(data[1].shape().dim(0));
                for (int k = 0; k < 2; k++) {
                    ContextLm scout = new ContextLm(COUNT_VOCAB, 48, seed + 10 + k);
                    for (int e = 0; e < 3; e++) {
                        scout.train(data, steps / 8, 64, 1e-2f, seed + 20 + 3 * k + e);
                        pruner.record(scout.logits(data[0]), data[1]);
                    }
                }
                int[] kept = DataPruner.keep(pruner.el2n(), 0.7, 0.3);
                lm.train(new Tensor[] {ContextLm.rows(data[0], kept), ContextLm.rows(data[1], kept)}, steps, 64,
                        1e-2f, seed + 2);
            }
            case RHO1 -> {
                ContextLm reference = new ContextLm(COUNT_VOCAB, 48, seed + 5)
                        .train(counting(16, seed + 6), steps, 64, 1e-2f, seed + 7);
                Optimizer opt = Optimizer.adam(1e-2f);
                SplittableRandom b = new SplittableRandom(seed + 2);
                int n = data[1].shape().dim(0);
                for (int s = 0; s < steps; s++) {
                    int[] rows = new int[64];
                    for (int i = 0; i < rows.length; i++) {
                        rows[i] = b.nextInt(n);
                    }
                    Tensor x = ContextLm.rows(data[0], rows);
                    Tensor y = ContextLm.rows(data[1], rows);
                    float[] ref = SelectiveTokenLoss.perToken(reference.logits(x), y);
                    SelectiveTokenLoss.loss(lm.net().forward(Var.of(x)), y, ref, 0.6).backward();
                    opt.step(lm.net().parameters());
                }
            }
        }
        return lm.loss(valid);
    }

    // ---------------------------------------------------------------- quality filtering

    /** How half of the pool is kept in {@link #qualityFilter}. */
    public enum Filter {
        /** A random half. */
        NONE,
        /** The half with the lowest perplexity under a model trained on the seed documents. */
        PERPLEXITY,
        /** The half a {@link QualityClassifierFilter} scores highest. */
        CLASSIFIER
    }

    /**
     * A pool of 400 documents, 40% grammatical and 60% corrupted (half of their tokens
     * replaced), is filtered to half its size; a model trained on the kept documents for a
     * fixed number of steps is scored on fresh grammatical documents. The filters see 150
     * grammatical seed documents (the "high-quality reference").
     *
     * @param filter the filter
     * @param steps training steps
     * @param seed the seed
     * @return the validation loss on clean documents and the share of clean documents kept
     */
    public static QualityRun qualityFilter(Filter filter, int steps, long seed) {
        Grammar g = Grammar.random(16, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        List<int[]> pool = new ArrayList<>();
        java.util.Set<int[]> clean = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());
        for (int i = 0; i < 400; i++) {
            int[] d = g.document(16, r);
            if (r.nextDouble() < 0.4) {
                clean.add(d);
            } else {
                d = corrupt(d, 0.5, 16, r);
            }
            pool.add(d);
        }
        List<int[]> seedDocs = g.documents(150, 16, r);
        List<int[]> valid = g.documents(200, 16, new SplittableRandom(seed + 2));
        List<int[]> kept = switch (filter) {
            case NONE -> {
                double[] random = new double[pool.size()];
                for (int i = 0; i < random.length; i++) {
                    random[i] = r.nextDouble();
                }
                yield QualityClassifierFilter.keepTop(pool, random, 0.5);
            }
            case PERPLEXITY -> {
                ContextLm ref = new ContextLm(16, 32, seed + 3).train(seedDocs, 300, 32, 1e-2f, seed + 4);
                double[] score = new double[pool.size()];
                for (int i = 0; i < score.length; i++) {
                    score[i] = -ref.loss(List.of(pool.get(i)));
                }
                yield QualityClassifierFilter.keepTop(pool, score, 0.5);
            }
            case CLASSIFIER -> {
                List<int[]> negatives = new ArrayList<>();
                for (int i = 0; i < seedDocs.size(); i++) {
                    negatives.add(pool.get(r.nextInt(pool.size())));
                }
                yield QualityClassifierFilter.train(seedDocs, negatives, 4096, 3, 200, seed + 5).filter(pool, 0.5);
            }
        };
        double share = kept.stream().filter(clean::contains).count() / (double) kept.size();
        return new QualityRun(new ContextLm(16, 64, seed + 6).train(kept, steps, 32, 1e-2f, seed + 7).loss(valid),
                share);
    }

    /**
     * The outcome of {@link #qualityFilter}.
     *
     * @param loss the validation loss on clean documents
     * @param cleanShare the share of clean documents among those kept (0.4 in the pool)
     */
    public record QualityRun(float loss, double cleanShare) {
    }

    // ---------------------------------------------------------------- mixtures

    /** How the domain mixture is chosen in {@link #mixture}. */
    public enum Mixing {
        /** Equal weights. */
        UNIFORM,
        /** DoReMi: excess-loss reweighting against a uniform reference, averaged over a proxy run. */
        DOREMI,
        /** RegMix: regression over 24 proxy runs. */
        REGMIX
    }

    /**
     * The outcome of {@link #mixture}.
     *
     * @param weights the mixture used
     * @param targetLoss the main model's loss on the target domain
     */
    public record MixRun(double[] weights, float targetLoss) {
    }

    private static final int DOMAINS = 4;

    private static ContextLm trainMixture(List<Tensor[]> domains, IntFunction<double[]> weights, int hidden,
            int steps, long seed, java.util.function.ObjIntConsumer<ContextLm> every) {
        ContextLm lm = new ContextLm(16, hidden, seed);
        Optimizer opt = Optimizer.adam(1e-2f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        int batch = 32;
        for (int s = 0; s < steps; s++) {
            int[] counts = DataMixture.constant(weights.apply(s)).allocate(s, batch);
            List<Tensor> xs = new ArrayList<>();
            List<Tensor> ys = new ArrayList<>();
            for (int d = 0; d < counts.length; d++) {
                if (counts[d] == 0) {
                    continue;
                }
                int n = domains.get(d)[1].shape().dim(0);
                int[] rows = new int[counts[d]];
                for (int i = 0; i < rows.length; i++) {
                    rows[i] = r.nextInt(n);
                }
                xs.add(ContextLm.rows(domains.get(d)[0], rows));
                ys.add(ContextLm.rows(domains.get(d)[1], rows));
            }
            Tensor x = Tensor.concat(0, xs.toArray(new Tensor[0]));
            Tensor y = Tensor.concat(0, ys.toArray(new Tensor[0]));
            Functional.crossEntropy(lm.net().forward(Var.of(x)), y).backward();
            opt.step(lm.net().parameters());
            if (every != null) {
                every.accept(lm, s);
            }
        }
        return lm;
    }

    /**
     * Four domains (grammars); the target is an even mix of documents from domains 1 and 3
     * that the training data never shows as such. Each method picks a mixture, then a model
     * is trained on it for {@code steps} and scored on target documents.
     *
     * @param mixing the method
     * @param steps steps of the main run
     * @param seed the seed
     * @return the run
     */
    public static MixRun mixture(Mixing mixing, int steps, long seed) {
        List<Grammar> grammars = new ArrayList<>();
        List<Tensor[]> train = new ArrayList<>();
        List<List<int[]>> held = new ArrayList<>();
        SplittableRandom r = new SplittableRandom(seed);
        for (int d = 0; d < DOMAINS; d++) {
            Grammar g = Grammar.random(16, seed * 31 + d);
            grammars.add(g);
            train.add(ContextLm.examples(16, g.documents(200, 16, r)));
            held.add(g.documents(30, 16, r));
        }
        List<int[]> target = new ArrayList<>(grammars.get(1).documents(40, 16, r));
        target.addAll(grammars.get(3).documents(40, 16, r));
        double[] uniform = new double[DOMAINS];
        Arrays.fill(uniform, 1.0 / DOMAINS);
        double[] w = switch (mixing) {
            case UNIFORM -> uniform;
            case DOREMI -> {
                int proxySteps = 150;
                ContextLm reference = trainMixture(train, s -> uniform, 16, proxySteps, seed + 5, null);
                double[] refLoss = new double[DOMAINS];
                for (int d = 0; d < DOMAINS; d++) {
                    refLoss[d] = reference.loss(held.get(d));
                }
                DataMixture online = DataMixture.constant(uniform);
                double[] avg = new double[DOMAINS];
                double[][] current = {uniform.clone()};
                trainMixture(train, s -> current[0], 16, proxySteps, seed + 6, (lm, s) -> {
                    if (s % 10 == 9) {
                        double[] loss = new double[DOMAINS];
                        for (int d = 0; d < DOMAINS; d++) {
                            loss[d] = lm.loss(held.get(d));
                        }
                        current[0] = online.updateWithExcessLoss(s, loss, refLoss, 1, 1e-3);
                    }
                    for (int d = 0; d < DOMAINS; d++) {
                        avg[d] += current[0][d] / proxySteps;
                    }
                });
                yield avg;
            }
            case REGMIX -> {
                RegressionMixer mixer = new RegressionMixer(DOMAINS, 1e-3);
                long k = 0;
                for (double[] m : mixer.proposals(24, 1, seed + 7)) {
                    ContextLm proxy = trainMixture(train, s -> m, 16, 150, seed + 100 + k++, null);
                    mixer.observe(m, proxy.loss(target));
                }
                yield mixer.best(2000, 20, seed + 8);
            }
        };
        ContextLm main = trainMixture(train, s -> w, 64, steps, seed + 9, null);
        List<int[]> fresh = new ArrayList<>(grammars.get(1).documents(100, 16, new SplittableRandom(seed + 10)));
        fresh.addAll(grammars.get(3).documents(100, 16, new SplittableRandom(seed + 11)));
        return new MixRun(w, main.loss(fresh));
    }

    // ---------------------------------------------------------------- EntiGraph

    /** The corpus a model learns facts from in {@link #factLearning}. */
    public enum FactCorpus {
        /** The source documents, repeated. */
        RAW,
        /** The source documents and paraphrases of them (facts reordered, fillers changed). */
        PARAPHRASE,
        /** The source documents and EntiGraph's synthetic documents about entity pairs and triples. */
        ENTIGRAPH
    }

    /** The token layout of the fact world. */
    static final int ENTITIES = 12;
    static final int RELATIONS = 2;
    static final int INVERSE = ENTITIES + RELATIONS;
    static final int COMPOSED = INVERSE + RELATIONS;
    static final int FILLER = COMPOSED + RELATIONS * RELATIONS;
    static final int FILLERS = 4;
    static final int FACT_VOCAB = FILLER + FILLERS;

    /**
     * The outcome of {@link #factLearning}.
     *
     * @param forward accuracy on {@code (head, relation) -> tail}, as stated in the documents
     * @param reversed accuracy on {@code (tail, inverse relation) -> head}
     * @param twoHop accuracy on {@code (head, composed relation) -> tail of the next fact}
     * @param trainingDocuments the documents trained on
     */
    public record FactRun(double forward, double reversed, double twoHop, int trainingDocuments) {

        /** @return the accuracy over all questions */
        public double overall() {
            return (forward + reversed + twoHop / 2) / 2.5;   // there are half as many two-hop questions
        }
    }

    private static int filler(RandomGenerator r) {
        return FILLER + r.nextInt(FILLERS);
    }

    /** A document: a chain of two facts {@code a r b, b r' c} with fillers around them. */
    private static int[] chainDocument(int a, int r1, int b, int r2, int c, RandomGenerator r) {
        return new int[] {filler(r), a, r1 + ENTITIES, b, filler(r), filler(r), b, r2 + ENTITIES, c, filler(r)};
    }

    /** The facts {@code {head, relation, tail}} stated in a document (contiguous triples). */
    private static List<int[]> facts(int[] doc) {
        List<int[]> out = new ArrayList<>();
        for (int t = 0; t + 2 < doc.length; t++) {
            if (doc[t] < ENTITIES && doc[t + 1] >= ENTITIES && doc[t + 1] < INVERSE && doc[t + 2] < ENTITIES) {
                out.add(new int[] {doc[t], doc[t + 1] - ENTITIES, doc[t + 2]});
            }
        }
        return out;
    }

    /**
     * The stand-in for EntiGraph's LLM: given a document and two or three of its entities,
     * it writes what the document says about how they relate, in a form chosen at random:
     * the fact itself, its inverse ({@code b inv(r) a}), or, for entities two facts apart, the
     * composed relation ({@code a r.r' c}); for a triple, the chain.
     */
    static int[] writePair(int[] doc, int[] e, RandomGenerator r) {
        List<int[]> f = facts(doc);
        List<int[]> statements = new ArrayList<>();
        for (int[] x : f) {
            if (e.length == 2 && x[0] == e[0] && x[2] == e[1]) {
                statements.add(new int[] {e[0], x[1] + ENTITIES, e[1]});
            }
            if (e.length == 2 && x[2] == e[0] && x[0] == e[1]) {
                statements.add(new int[] {e[0], x[1] + INVERSE, e[1]});
            }
            for (int[] y : f) {
                if (x[2] == y[0] && e[0] == x[0] && e[e.length - 1] == y[2] && (e.length == 2 || e[1] == x[2])) {
                    statements.add(e.length == 2 ? new int[] {e[0], COMPOSED + x[1] * RELATIONS + y[1], e[1]}
                            : new int[] {x[0], x[1] + ENTITIES, x[2], y[1] + ENTITIES, y[2]});
                }
            }
        }
        if (statements.isEmpty()) {
            return new int[0];
        }
        int[] s = statements.get(r.nextInt(statements.size()));
        int[] out = new int[s.length + 2];
        out[0] = filler(r);
        System.arraycopy(s, 0, out, 1, s.length);
        out[out.length - 1] = filler(r);
        return out;
    }

    /** A paraphrase: the document's facts in another order, with new fillers. */
    private static int[] paraphrase(int[] doc, RandomGenerator r) {
        List<int[]> f = new ArrayList<>(facts(doc));
        if (r.nextBoolean()) {
            java.util.Collections.reverse(f);
        }
        List<Integer> out = new ArrayList<>();
        for (int[] x : f) {
            for (int k = r.nextInt(3); k > 0; k--) {
                out.add(filler(r));
            }
            out.add(x[0]);
            out.add(x[1] + ENTITIES);
            out.add(x[2]);
        }
        out.add(filler(r));
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Four short documents each state a chain of two facts about three entities (12
     * entities, 2 relations). A model is trained on a corpus built from them for
     * {@code steps}, then asked every fact forward, reversed (with the inverse relation
     * token) and two hops at once (with the composed relation token); the answer is the
     * entity with the highest probability.
     *
     * @param corpus the training corpus
     * @param steps training steps
     * @param seed the seed
     * @return the accuracies
     */
    public static FactRun factLearning(FactCorpus corpus, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] perm = new int[ENTITIES];
        for (int i = 0; i < ENTITIES; i++) {
            perm[i] = i;
        }
        for (int i = ENTITIES - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = perm[i];
            perm[i] = perm[j];
            perm[j] = t;
        }
        List<int[]> source = new ArrayList<>();
        List<int[]> chains = new ArrayList<>();
        for (int d = 0; d < 4; d++) {
            int r1 = r.nextInt(RELATIONS);
            int r2 = r.nextInt(RELATIONS);
            int[] c = {perm[3 * d], r1, perm[3 * d + 1], r2, perm[3 * d + 2]};
            chains.add(c);
            source.add(chainDocument(c[0], c[1], c[2], c[3], c[4], r));
        }
        List<int[]> docs = new ArrayList<>();
        int synthetic = 96;
        switch (corpus) {
            case RAW -> {
                for (int k = 0; k < (synthetic + source.size()) / source.size(); k++) {
                    docs.addAll(source);
                }
            }
            case PARAPHRASE -> {
                docs.addAll(source);
                for (int k = 0; k < synthetic; k++) {
                    docs.add(paraphrase(source.get(k % source.size()), r));
                }
            }
            case ENTIGRAPH -> {
                docs.addAll(source);
                EntityGraphAugmenter aug = new EntityGraphAugmenter(t -> t < ENTITIES, DataCuration::writePair);
                docs.addAll(aug.augment(source, 4, true, r));
            }
        }
        ContextLm lm = new ContextLm(FACT_VOCAB, 64, seed + 1).train(docs, steps, 32, 1e-2f, seed + 2);
        int[] forward = new int[2];
        int[] reversed = new int[2];
        int[] twoHop = new int[2];
        for (int[] c : chains) {
            ask(lm, c[0], c[1] + ENTITIES, c[2], forward);
            ask(lm, c[2], c[3] + ENTITIES, c[4], forward);
            ask(lm, c[2], c[1] + INVERSE, c[0], reversed);
            ask(lm, c[4], c[3] + INVERSE, c[2], reversed);
            ask(lm, c[0], COMPOSED + c[1] * RELATIONS + c[3], c[4], twoHop);
        }
        return new FactRun(forward[0] / (double) forward[1], reversed[0] / (double) reversed[1],
                twoHop[0] / (double) twoHop[1], docs.size());
    }

    private static void ask(ContextLm lm, int head, int relation, int answer, int[] tally) {
        float[] x = new float[2 * FACT_VOCAB];
        x[head] = 1;
        x[FACT_VOCAB + relation] = 1;
        float[] z = lm.logits(Tensor.of(Shape.of(1, 2 * FACT_VOCAB), x)).toArray();
        int best = 0;
        for (int e = 1; e < ENTITIES; e++) {
            best = z[e] > z[best] ? e : best;
        }
        tally[0] += best == answer ? 1 : 0;
        tally[1]++;
    }
}
