package io.github.zzuegg.brainiac.experiments.data;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.AioliMixer;
import io.github.zzuegg.brainiac.experimental.data.BanditMixer;
import io.github.zzuegg.brainiac.experimental.data.Curriculum;
import io.github.zzuegg.brainiac.experimental.data.DataMixture;
import io.github.zzuegg.brainiac.experimental.data.DataPruner;
import io.github.zzuegg.brainiac.experimental.data.EntityGraphAugmenter;
import io.github.zzuegg.brainiac.experimental.data.QualityClassifierFilter;
import io.github.zzuegg.brainiac.experimental.data.ProgressCurriculum;
import io.github.zzuegg.brainiac.experimental.data.ReducibleLossSelector;
import io.github.zzuegg.brainiac.experimental.data.RegressionMixer;
import io.github.zzuegg.brainiac.experimental.data.SkillGraphSampler;
import io.github.zzuegg.brainiac.experimental.data.SkillTasks;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.rl.LevelReplay;
import io.github.zzuegg.brainiac.experimental.train.SelfPacedLearning;
import io.github.zzuegg.brainiac.experimental.train.SelectiveTokenLoss;
import io.github.zzuegg.brainiac.experiments.language.CountingTasks;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
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
 *   <li>Wave 9: {@link #onlineMixture} (online data mixing with {@link BanditMixer}, Aioli with
 *       {@link AioliMixer}, against uniform, temperature, DoReMi and RegMix at equal steps) and
 *       {@link #twoDomain} (against a grid-searched static mixture); {@link #skillGraph} and
 *       {@link #skillSelection} (Skill-It with {@link SkillGraphSampler} on {@link SkillTasks});
 *       {@link #lengthCurriculum} (learning-progress teachers, {@link ProgressCurriculum});
 *       {@link #selection} (RHO-LOSS with {@link ReducibleLossSelector}).</li>
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

    // ---------------------------------------------------------------- online mixing (wave 9)

    /** How the mixture is set in {@link #onlineMixture}. */
    public enum OnlineMixing {
        /** Equal weights. */
        UNIFORM,
        /** Temperature sampling by domain size ({@code T = 2}). */
        TEMPERATURE,
        /** DoReMi: a reference and a proxy run (each a quarter of the budget), then the main run. */
        DOREMI,
        /** RegMix: six proxy runs (half the budget in all), then the main run. */
        REGMIX,
        /** Online data mixing: an EXP3 bandit with loss rewards ({@link BanditMixer}). */
        ODM,
        /** Online data mixing with the loss reward capped at the median domain loss. */
        ODM_CLIPPED,
        /** Aioli: mixing law fitted online, exponentiated-gradient proportions ({@link AioliMixer}). */
        AIOLI
    }

    /**
     * The outcome of {@link #onlineMixture}.
     *
     * @param method the method
     * @param weights the mixture averaged over the main run
     * @param targetLoss loss on fresh target-domain documents
     * @param averageLoss mean held-out loss over the four domains
     * @param steps training steps spent, proxy runs included
     */
    public record OnlineMixRun(OnlineMixing method, double[] weights, double targetLoss, double averageLoss,
            int steps) {
        @Override
        public String toString() {
            return String.format("%s: target %.4f, average %.4f, weights %s, %d steps", method, targetLoss,
                    averageLoss, Arrays.toString(Arrays.stream(weights).map(w -> Math.round(w * 100) / 100.0)
                            .toArray()), steps);
        }
    }

    /** Domains of the online mixing runs: training examples, held-out documents, target documents. */
    private record Domains(List<Tensor[]> train, List<List<int[]>> held, List<int[]> target, long[] sizes) {
    }

    private static Grammar peaked(int vocab, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] t = new double[vocab * vocab][vocab];
        for (double[] row : t) {
            java.util.Arrays.fill(row, 0.02 / vocab);
            row[r.nextInt(vocab)] += 0.98;
        }
        return new Grammar(vocab, t);
    }

    /**
     * Four domains: easy (a nearly deterministic grammar), hard (a random grammar), noise-heavy
     * (a random grammar with 70% of tokens replaced) and target-like (the grammar of the target
     * documents), with 400, 300, 400 and 100 training documents.
     */
    private static Domains mixingDomains(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Grammar easy = peaked(16, seed * 31 + 1);
        Grammar hard = Grammar.random(16, seed * 31 + 2);
        Grammar noisy = Grammar.random(16, seed * 31 + 3);
        Grammar target = Grammar.random(16, seed * 31 + 4);
        Grammar[] g = {easy, hard, noisy, target};
        int[] docs = {400, 300, 400, 100};
        List<Tensor[]> train = new ArrayList<>();
        List<List<int[]>> held = new ArrayList<>();
        long[] sizes = new long[4];
        for (int d = 0; d < 4; d++) {
            List<int[]> t = g[d].documents(docs[d], 16, r);
            List<int[]> h = g[d].documents(40, 16, r);
            if (d == 2) {
                t.replaceAll(x -> corrupt(x, 0.7, 16, r));
                h.replaceAll(x -> corrupt(x, 0.7, 16, r));
            }
            train.add(ContextLm.examples(16, t));
            held.add(h);
            sizes[d] = docs[d];
        }
        return new Domains(train, held, target.documents(150, 16, new SplittableRandom(seed + 77)), sizes);
    }

    private static double[] heldLosses(ContextLm lm, List<List<int[]>> held) {
        double[] l = new double[held.size()];
        for (int d = 0; d < l.length; d++) {
            l[d] = lm.loss(held.get(d));
        }
        return l;
    }

    /** A mixture that can react to per-domain batch losses. */
    private interface MixPolicy {
        double[] weights(int step);

        default void observe(int step, double[] losses, boolean[] pulled, ContextLm lm) {
        }
    }

    private static ContextLm trainPolicy(List<Tensor[]> domains, MixPolicy policy, int hidden, int steps, long seed,
            double[] meanWeights) {
        ContextLm lm = new ContextLm(16, hidden, seed);
        Optimizer opt = Optimizer.adam(1e-2f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        int batch = 32;
        int k = domains.size();
        for (int s = 0; s < steps; s++) {
            double[] w = policy.weights(s);
            if (meanWeights != null) {
                for (int d = 0; d < k; d++) {
                    meanWeights[d] += w[d] / steps;
                }
            }
            int[] counts = DataMixture.constant(w).allocate(s, batch);
            Var total = null;
            Var[] parts = new Var[k];
            boolean[] pulled = new boolean[k];
            for (int d = 0; d < k; d++) {
                if (counts[d] == 0) {
                    continue;
                }
                int n = domains.get(d)[1].shape().dim(0);
                int[] rows = new int[counts[d]];
                for (int i = 0; i < rows.length; i++) {
                    rows[i] = r.nextInt(n);
                }
                parts[d] = Functional.crossEntropy(lm.net().forward(Var.of(ContextLm.rows(domains.get(d)[0], rows))),
                        ContextLm.rows(domains.get(d)[1], rows));
                pulled[d] = true;
                Var term = parts[d].mul(counts[d] / (float) batch);
                total = total == null ? term : total.add(term);
            }
            total.backward();
            opt.step(lm.net().parameters());
            double[] losses = new double[k];
            for (int d = 0; d < k; d++) {
                losses[d] = parts[d] == null ? 0 : parts[d].value().item();
            }
            policy.observe(s, losses, pulled, lm);
        }
        return lm;
    }

    /**
     * Mixing four domains during training (9.3.1, 9.3.2): every method gets the same number of
     * training steps in all, so DoReMi's reference and proxy runs and RegMix's proxy runs come
     * out of the main run's budget.
     *
     * @param method the method
     * @param budget training steps in all
     * @param seed the seed
     * @return the run
     */
    public static OnlineMixRun onlineMixture(OnlineMixing method, int budget, long seed) {
        Domains dom = mixingDomains(seed);
        int k = 4;
        double[] uniform = new double[k];
        Arrays.fill(uniform, 1.0 / k);
        double[] mean = new double[k];
        int main = budget;
        MixPolicy policy;
        switch (method) {
            case UNIFORM -> policy = s -> uniform;
            case TEMPERATURE -> {
                double[] w = DataMixture.temperature(dom.sizes(), 2);
                policy = s -> w;
            }
            case DOREMI -> {
                int proxy = budget / 4;
                main = budget - 2 * proxy;
                ContextLm reference = trainPolicy(dom.train(), s -> uniform, 16, proxy, seed + 5, null);
                double[] refLoss = heldLosses(reference, dom.held());
                DataMixture online = DataMixture.constant(uniform);
                double[][] current = {uniform.clone()};
                double[] avg = new double[k];
                trainPolicy(dom.train(), new MixPolicy() {
                    @Override
                    public double[] weights(int step) {
                        return current[0];
                    }

                    @Override
                    public void observe(int step, double[] losses, boolean[] pulled, ContextLm lm) {
                        if (step % 10 == 9) {
                            current[0] = online.updateWithExcessLoss(step, heldLosses(lm, dom.held()), refLoss, 1,
                                    1e-3);
                        }
                        for (int d = 0; d < k; d++) {
                            avg[d] += current[0][d] / proxy;
                        }
                    }
                }, 16, proxy, seed + 6, null);
                policy = s -> avg;
            }
            case REGMIX -> {
                int proxies = 6;
                int each = budget / 2 / proxies;
                main = budget - proxies * each;
                RegressionMixer mixer = new RegressionMixer(k, 1e-3);
                long n = 0;
                for (double[] m : mixer.proposals(proxies, 1, seed + 7)) {
                    ContextLm proxy = trainPolicy(dom.train(), s -> m, 16, each, seed + 100 + n++, null);
                    mixer.observe(m, proxy.loss(dom.target()));
                }
                double[] w = mixer.best(2000, 20, seed + 8);
                policy = s -> w;
            }
            case ODM, ODM_CLIPPED -> {
                BanditMixer bandit = method == OnlineMixing.ODM ? BanditMixer.of(k, 0.9)
                        : BanditMixer.clipToMedian(k, 0.9);
                policy = new MixPolicy() {
                    @Override
                    public double[] weights(int step) {
                        return bandit.weights();
                    }

                    @Override
                    public void observe(int step, double[] losses, boolean[] pulled, ContextLm lm) {
                        bandit.update(losses, pulled);
                    }
                };
            }
            case AIOLI -> {
                AioliMixer aioli = new AioliMixer(k, 0.3, 1e-4, 0.5, 3 * (k + 1));
                int chunk = Math.max(5, budget / 60);
                policy = new MixPolicy() {
                    @Override
                    public double[] weights(int step) {
                        return aioli.proportions();
                    }

                    @Override
                    public void observe(int step, double[] losses, boolean[] pulled, ContextLm lm) {
                        if (step % chunk == chunk - 1) {
                            aioli.observe(heldLosses(lm, dom.held()));
                        }
                    }
                };
            }
            default -> throw new IllegalArgumentException(method.name());
        }
        ContextLm lm = trainPolicy(dom.train(), policy, 64, main, seed + 9, mean);
        double avg = Arrays.stream(heldLosses(lm, dom.held())).average().orElse(0);
        return new OnlineMixRun(method, mean, lm.loss(dom.target()), avg, budget);
    }

    /**
     * The two-domain case of 9.3.2 (hard and target-like domains, average held-out loss), where
     * the best static mixture is found by grid search.
     *
     * @param gridOptimum the share of the first domain with the lowest average loss on a 0.1 grid
     * @param gridLoss that average loss
     * @param uniformLoss the average loss of the 50/50 mixture
     * @param aioliShare Aioli's mean share of the first domain
     * @param aioliLoss Aioli's average loss
     * @param odmShare ODM's mean share of the first domain
     * @param odmLoss ODM's average loss
     */
    public record TwoDomain(double gridOptimum, double gridLoss, double uniformLoss, double aioliShare,
            double aioliLoss, double odmShare, double odmLoss) {
    }

    /**
     * Grid search against Aioli and ODM on two domains.
     *
     * @param steps steps per run
     * @param seed the seed
     * @return the comparison
     */
    public static TwoDomain twoDomain(int steps, long seed) {
        Domains all = mixingDomains(seed);
        List<Tensor[]> train = List.of(all.train().get(0), all.train().get(1));
        List<List<int[]>> held = List.of(all.held().get(0), all.held().get(1));
        double best = 0;
        double bestLoss = Double.MAX_VALUE;
        double uniformLoss = 0;
        for (int g = 0; g <= 10; g++) {
            double[] w = {g / 10.0, 1 - g / 10.0};
            ContextLm lm = trainPolicy(train, s -> w, 64, steps, seed + 9, null);
            double loss = Arrays.stream(heldLosses(lm, held)).average().orElse(0);
            if (loss < bestLoss) {
                bestLoss = loss;
                best = w[0];
            }
            if (g == 5) {
                uniformLoss = loss;
            }
        }
        AioliMixer aioli = new AioliMixer(2, 0.3, 1e-4, 0.5, 9);
        int chunk = Math.max(5, steps / 60);
        double[] aioliMean = new double[2];
        ContextLm a = trainPolicy(train, new MixPolicy() {
            @Override
            public double[] weights(int step) {
                return aioli.proportions();
            }

            @Override
            public void observe(int step, double[] losses, boolean[] pulled, ContextLm lm) {
                if (step % chunk == chunk - 1) {
                    aioli.observe(heldLosses(lm, held));
                }
            }
        }, 64, steps, seed + 9, aioliMean);
        BanditMixer bandit = BanditMixer.of(2, 0.9);
        double[] odmMean = new double[2];
        ContextLm o = trainPolicy(train, new MixPolicy() {
            @Override
            public double[] weights(int step) {
                return bandit.weights();
            }

            @Override
            public void observe(int step, double[] losses, boolean[] pulled, ContextLm lm) {
                bandit.update(losses, pulled);
            }
        }, 64, steps, seed + 9, odmMean);
        return new TwoDomain(best, bestLoss, uniformLoss, aioliMean[0],
                Arrays.stream(heldLosses(a, held)).average().orElse(0), odmMean[0],
                Arrays.stream(heldLosses(o, held)).average().orElse(0));
    }

    // ---------------------------------------------------------------- skills (wave 9)

    private static final int SKILLS = SkillTasks.Skill.values().length;

    private static SimpleMlp skillModel(long seed) {
        return SimpleMlp.relu(seed, SkillTasks.INPUT, 128, SkillTasks.OUT * SkillTasks.CLASSES);
    }

    private static void trainSkill(SimpleMlp net, Optimizer opt, SkillTasks.Batch b) {
        SkillTasks.loss(net.forward(Var.of(b.x())), b).backward();
        opt.step(net.parameters());
    }

    private static double[] skillLosses(SimpleMlp net, SkillTasks.Batch[] val) {
        double[] l = new double[val.length];
        for (int i = 0; i < val.length; i++) {
            SkillTasks.Batch b = val[i];
            l[i] = Var.noGrad(() -> SkillTasks.loss(Var.of(net.predict(b.x())), b)).value().item();
        }
        return l;
    }

    private static SkillTasks.Batch[] skillValidation(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        SkillTasks.Batch[] v = new SkillTasks.Batch[SKILLS];
        for (int i = 0; i < SKILLS; i++) {
            v[i] = SkillTasks.sample(SkillTasks.Skill.values()[i], 200, r);
        }
        return v;
    }

    /**
     * The estimated skill graph (9.3.3).
     *
     * @param graph {@code A_ij} from probe runs
     * @param recovered planted edges found (threshold 0.3)
     * @param planted planted edges
     * @param spurious edges found that were not planted
     */
    public record SkillGraph(double[][] graph, int recovered, int planted, int spurious) {
        @Override
        public String toString() {
            return String.format("recovered %d of %d planted edges, %d spurious; A = %s", recovered, planted, spurious,
                    Arrays.deepToString(Arrays.stream(graph).map(row -> Arrays.stream(row)
                            .map(v -> Math.round(v * 100) / 100.0).toArray()).toArray()));
        }
    }

    /**
     * Estimates the skill graph by leave-one-out probe runs ({@link SkillGraphSampler#estimateByAblation})
     * from a briefly pretrained common start: each probe trains on the uniform mixture of every
     * skill but one, and a reference probe on all of them.
     *
     * @param warmup steps of uniform training before the probes
     * @param probeSteps steps of each probe
     * @param seed the seed
     * @return the graph against the planted one
     */
    public static SkillGraph skillGraph(int warmup, int probeSteps, long seed) {
        SimpleMlp start = skillModel(seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < warmup; s++) {
            trainSkill(start, opt, SkillTasks.sample(SkillTasks.Skill.values()[s % SKILLS], 32, r));
        }
        SkillTasks.Batch[] val = skillValidation(seed + 2);
        List<Tensor> snapshot = new ArrayList<>();
        for (Var p : start.parameters()) {
            snapshot.add(p.value().add(0f));
        }
        IntFunction<double[]> probeWithout = i -> {
            SimpleMlp probe = skillModel(seed);
            for (int k = 0; k < snapshot.size(); k++) {
                probe.parameters().get(k).assign(snapshot.get(k));
            }
            Optimizer po = Optimizer.adam(3e-3f);
            SplittableRandom pr = new SplittableRandom(seed + 10);
            int s = 0;
            for (int step = 0; s < probeSteps; step++) {
                if (step % SKILLS != i) {
                    trainSkill(probe, po, SkillTasks.sample(SkillTasks.Skill.values()[step % SKILLS], 32, pr));
                    s++;
                }
            }
            return skillLosses(probe, val);
        };
        double[][] a = SkillGraphSampler.estimateByAblation(SKILLS, probeWithout.apply(-1), probeWithout);
        boolean[][] found = SkillGraphSampler.edges(a, 0.3);
        boolean[][] truth = SkillTasks.prerequisites();
        int rec = 0;
        int pl = 0;
        int sp = 0;
        for (int i = 0; i < SKILLS; i++) {
            for (int j = 0; j < SKILLS; j++) {
                pl += truth[i][j] ? 1 : 0;
                rec += truth[i][j] && found[i][j] ? 1 : 0;
                sp += !truth[i][j] && found[i][j] ? 1 : 0;
            }
        }
        return new SkillGraph(a, rec, pl, sp);
    }

    /** How skills are sampled in {@link #skillSelection}. */
    public enum SkillSampling {
        /** Uniform over the five skills. */
        UNIFORM,
        /** The target skill only. */
        TARGET_ONLY,
        /** Fixed pacing in prerequisite order ({@link Curriculum}). */
        CURRICULUM,
        /** The EXP3.S progress teacher ({@link ProgressCurriculum}). */
        PROGRESS,
        /** Skill-It on the planted graph ({@link SkillGraphSampler}). */
        SKILL_IT,
        /** Skill-It on a graph estimated by probe runs (whose cost is not counted here). */
        SKILL_IT_ESTIMATED
    }

    /**
     * Target-skill outcome of one sampling method.
     *
     * @param method the method
     * @param targetLoss the target skill's validation loss at the end
     * @param stepsToTarget steps until the target loss first falls below the threshold (-1 never)
     * @param skillShares the share of steps on each skill
     */
    public record SkillRun(SkillSampling method, double targetLoss, int stepsToTarget, double[] skillShares) {
        @Override
        public String toString() {
            return String.format("%s: target loss %.4f, steps to threshold %d, shares %s", method, targetLoss,
                    stepsToTarget, Arrays.toString(Arrays.stream(skillShares).map(v -> Math.round(v * 100) / 100.0)
                            .toArray()));
        }
    }

    /**
     * Trains on skills with a sampling method; the target is multi-digit addition.
     *
     * @param method the method
     * @param steps training steps
     * @param threshold the target loss counted as reached
     * @param graph the graph for {@link SkillSampling#SKILL_IT_ESTIMATED} (or null)
     * @param seed the seed
     * @return the run
     */
    public static SkillRun skillSelection(SkillSampling method, int steps, double threshold, double[][] graph,
            long seed) {
        SimpleMlp net = skillModel(seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        SkillTasks.Batch[] val = skillValidation(seed + 2);
        int target = SkillTasks.Skill.ADD_MULTI.ordinal();
        boolean[] targets = new boolean[SKILLS];
        targets[target] = true;
        double[][] planted = new double[SKILLS][SKILLS];
        boolean[][] pre = SkillTasks.prerequisites();
        for (int i = 0; i < SKILLS; i++) {
            for (int j = 0; j < SKILLS; j++) {
                planted[i][j] = i == j ? 1 : pre[i][j] ? 1 : 0;
            }
        }
        SkillGraphSampler skillIt = method == SkillSampling.SKILL_IT ? new SkillGraphSampler(planted, targets, 0.5)
                : method == SkillSampling.SKILL_IT_ESTIMATED ? new SkillGraphSampler(graph, targets, 0.5) : null;
        ProgressCurriculum teacher = method == SkillSampling.PROGRESS ? ProgressCurriculum.exp3s(SKILLS, 0.1, 0.05,
                0.01, seed + 3) : null;
        Curriculum curriculum = new Curriculum(new float[] {0, 0.25f, 0.5f, 0.75f, 1f}, 0.2, steps / 2,
                Curriculum.Pacing.LINEAR);
        int[] counts = new int[SKILLS];
        int reached = -1;
        double last = Double.NaN;
        for (int s = 0; s < steps; s++) {
            int skill = switch (method) {
                case UNIFORM -> r.nextInt(SKILLS);
                case TARGET_ONLY -> target;
                case CURRICULUM -> {
                    int[] avail = curriculum.batch(s, 1, r);
                    yield Math.min(avail[0], target);
                }
                case PROGRESS -> teacher.next();
                case SKILL_IT, SKILL_IT_ESTIMATED -> skillIt.sample(r);
            };
            counts[skill]++;
            trainSkill(net, opt, SkillTasks.sample(SkillTasks.Skill.values()[skill], 32, r));
            if (teacher != null) {
                SkillTasks.Batch b = val[skill];
                teacher.observe(skill, -Var.noGrad(() -> SkillTasks.loss(Var.of(net.predict(b.x())), b)).value()
                        .item());
            }
            if (s % 10 == 9) {
                double[] l = skillLosses(net, val);
                last = l[target];
                if (reached < 0 && last < threshold) {
                    reached = s + 1;
                }
                if (skillIt != null) {
                    skillIt.observe(l);
                }
            }
        }
        double[] shares = new double[SKILLS];
        for (int i = 0; i < SKILLS; i++) {
            shares[i] = counts[i] / (double) steps;
        }
        return new SkillRun(method, last, reached, shares);
    }

    /** How the length curriculum of {@link #lengthCurriculum} picks tasks. */
    public enum LengthTeacher {
        /** Uniform over the five tasks. */
        UNIFORM,
        /** Fixed pacing by length ({@link Curriculum}). */
        FIXED_PACING,
        /** Uniform tasks, self-paced per-sample weights ({@link SelfPacedLearning}). */
        SELF_PACED,
        /** Prioritised level replay on the task loss ({@link LevelReplay}). */
        LEVEL_REPLAY,
        /** EXP3.S on progress. */
        EXP3S,
        /** ε-greedy on the learning-curve slope. */
        SLOPE_GREEDY,
        /** Thompson sampling of the learning-curve slope. */
        SLOPE_THOMPSON
    }

    /**
     * The outcome of {@link #lengthCurriculum}.
     *
     * @param teacher the teacher
     * @param stepsTo09 steps until slot accuracy on 4-digit addition first reaches 0.9 (-1 never)
     * @param finalAccuracy the final slot accuracy on 4-digit addition
     * @param wasted the share of steps spent on the random-label task
     */
    public record LengthRun(LengthTeacher teacher, int stepsTo09, double finalAccuracy, double wasted) {
        @Override
        public String toString() {
            return String.format("%s: steps to 0.9 %d, final slot accuracy %.3f, wasted %.2f", teacher, stepsTo09,
                    finalAccuracy, wasted);
        }
    }

    private static SkillTasks.Batch lengthTask(int task, int n, SplittableRandom r) {
        return task == 4 ? SkillTasks.randomLabels(n, r) : SkillTasks.addition(task + 1, n, r);
    }

    /**
     * A length curriculum (9.3.4): addition of 1- to 4-digit numbers plus an unlearnable task
     * (random answers), one task per step.
     *
     * @param teacher the teacher
     * @param steps training steps
     * @param seed the seed
     * @return the run
     */
    public static LengthRun lengthCurriculum(LengthTeacher teacher, int steps, long seed) {
        SimpleMlp net = skillModel(seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        int tasks = 5;
        SkillTasks.Batch[] val = new SkillTasks.Batch[tasks];
        SplittableRandom vr = new SplittableRandom(seed + 2);
        for (int t = 0; t < tasks; t++) {
            val[t] = lengthTask(t, 100, vr);
        }
        SkillTasks.Batch test = SkillTasks.addition(4, 300, new SplittableRandom(seed + 3));
        ProgressCurriculum pc = switch (teacher) {
            case EXP3S -> ProgressCurriculum.exp3s(tasks, 0.1, 0.05, 0.01, seed + 4);
            case SLOPE_GREEDY -> ProgressCurriculum.slopeGreedy(tasks, 5, 0.1, seed + 4);
            case SLOPE_THOMPSON -> ProgressCurriculum.slopeThompson(tasks, 5, seed + 4);
            default -> null;
        };
        LevelReplay plr = teacher == LengthTeacher.LEVEL_REPLAY ? LevelReplay.of(tasks, seed + 5) : null;
        SelfPacedLearning spl = new SelfPacedLearning(1.5f, 1.05f, false);
        Curriculum curriculum = new Curriculum(new float[] {0, 0.25f, 0.5f, 0.75f, 1f}, 0.2, steps / 2,
                Curriculum.Pacing.LINEAR);
        int wasted = 0;
        int reached = -1;
        double acc = 0;
        for (int s = 0; s < steps; s++) {
            int task = switch (teacher) {
                case UNIFORM, SELF_PACED -> r.nextInt(tasks);
                case FIXED_PACING -> curriculum.batch(s, 1, r)[0];
                case LEVEL_REPLAY -> plr.next();
                default -> pc.next();
            };
            wasted += task == 4 ? 1 : 0;
            SkillTasks.Batch b = lengthTask(task, 32, r);
            Var logits = net.forward(Var.of(b.x()));
            if (teacher == LengthTeacher.SELF_PACED) {
                Var per = SelfPacedLearning.perSampleCrossEntropy(logits.reshape(b.n() * SkillTasks.OUT,
                        SkillTasks.CLASSES), b.y());
                spl.weightedLoss(per).backward();
                if (s % 50 == 49) {
                    spl.nextEpoch();
                }
            } else {
                SkillTasks.loss(logits, b).backward();
            }
            opt.step(net.parameters());
            if (pc != null || plr != null) {
                SkillTasks.Batch v = val[task];
                double loss = Var.noGrad(() -> SkillTasks.loss(Var.of(net.predict(v.x())), v)).value().item();
                if (pc != null) {
                    pc.observe(task, -loss);
                } else {
                    plr.update(task, loss);
                }
            }
            if (s % 20 == 19 || s == steps - 1) {
                acc = SkillTasks.accuracy(net.predict(test.x()), test)[0];
                if (reached < 0 && acc >= 0.9) {
                    reached = s + 1;
                }
            }
        }
        return new LengthRun(teacher, reached, acc, wasted / (double) steps);
    }

    // ---------------------------------------------------------------- RHO-LOSS (wave 9)

    /** How training points are selected in {@link #selection}. */
    public enum Selection {
        /** The first 32 of each candidate batch. */
        UNIFORM,
        /** The 32 with the highest current loss. */
        HIGHEST_LOSS,
        /** The 32 with the highest reducible holdout loss ({@link ReducibleLossSelector}). */
        RHO_LOSS,
        /** Uniform over an EL2N-pruned set ({@link DataPruner}, 70% kept, 10% hardest dropped). */
        EL2N
    }

    /**
     * The outcome of {@link #selection}.
     *
     * @param method the method
     * @param stepsToTarget steps until test accuracy first reaches the target (-1 never)
     * @param finalAccuracy the final test accuracy
     * @param noisyShare the share of selected points whose label is wrong (20% in the data)
     */
    public record SelectionRun(Selection method, int stepsToTarget, double finalAccuracy, double noisyShare) {
        @Override
        public String toString() {
            return String.format("%s: steps to target %d, final accuracy %.3f, noisy share %.2f", method,
                    stepsToTarget, finalAccuracy, noisyShare);
        }
    }

    private static double[] perSampleLoss(SimpleMlp net, Tensor x, Tensor y) {
        float[] v = Var.noGrad(() -> SelfPacedLearning.perSampleCrossEntropy(Var.of(net.predict(x)), y)).value()
                .toArray();
        double[] out = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i];
        }
        return out;
    }

    /**
     * Online batch selection (9.3.5) on {@code ToyTasks}-style digits with 20% label noise
     * and 100 points repeated four times: from each candidate batch of 128 a method keeps 32.
     *
     * @param method the method
     * @param steps training steps
     * @param target the test accuracy counted as reached
     * @param data the noisy digits of {@link #noisyDigits}
     * @param seed the seed
     * @return the run
     */
    public static SelectionRun selection(Selection method, int steps, double target, NoisyDigits data, long seed) {
        int n = data.labels().length;
        Tensor x = data.x();
        float[] yAll = new float[n];
        for (int i = 0; i < n; i++) {
            yAll[i] = data.labels()[i];
        }
        Tensor y = Tensor.of(Shape.of(n), yAll);
        int features = x.shape().dim(1);
        SimpleMlp net = SimpleMlp.relu(seed, features, 64, 10);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        ReducibleLossSelector rho = null;
        int[] pool = null;
        if (method == Selection.RHO_LOSS) {
            SimpleMlp il = SimpleMlp.relu(seed + 7, features, 64, 10);
            Optimizer io = Optimizer.adam(3e-3f);
            int hn = data.holdoutY().shape().dim(0);
            for (int s = 0; s < 300; s++) {
                int[] rows = new int[32];
                for (int i = 0; i < 32; i++) {
                    rows[i] = r.nextInt(hn);
                }
                Functional.crossEntropy(il.forward(Var.of(ContextLm.rows(data.holdoutX(), rows))),
                        ContextLm.rows(data.holdoutY(), rows)).backward();
                io.step(il.parameters());
            }
            rho = new ReducibleLossSelector(perSampleLoss(il, x, y));
        } else if (method == Selection.EL2N) {
            SimpleMlp probe = SimpleMlp.relu(seed + 8, features, 64, 10);
            Optimizer po = Optimizer.adam(3e-3f);
            DataPruner pruner = new DataPruner(n);
            for (int s = 0; s < 60; s++) {
                int[] rows = new int[32];
                for (int i = 0; i < 32; i++) {
                    rows[i] = r.nextInt(n);
                }
                Functional.crossEntropy(probe.forward(Var.of(ContextLm.rows(x, rows))), ContextLm.rows(y, rows))
                        .backward();
                po.step(probe.parameters());
                if (s % 20 == 19) {
                    pruner.record(probe.predict(x), y);
                }
            }
            pool = DataPruner.keep(pruner.el2n(), 0.7, 0.1);
        }
        int selectedNoisy = 0;
        int selected = 0;
        int reached = -1;
        double acc = 0;
        for (int s = 0; s < steps; s++) {
            int[] cand = new int[128];
            for (int i = 0; i < cand.length; i++) {
                cand[i] = pool == null ? r.nextInt(n) : pool[r.nextInt(pool.length)];
            }
            int[] pick;
            if (method == Selection.UNIFORM || method == Selection.EL2N) {
                pick = Arrays.copyOf(cand, 32);
            } else {
                double[] loss = perSampleLoss(net, ContextLm.rows(x, cand), ContextLm.rows(y, cand));
                pick = method == Selection.RHO_LOSS ? rho.select(cand, loss, 32)
                        : ReducibleLossSelector.highestLoss(cand, loss, 32);
            }
            for (int i : pick) {
                selectedNoisy += data.noisy()[i] ? 1 : 0;
                selected++;
            }
            Functional.crossEntropy(net.forward(Var.of(ContextLm.rows(x, pick))), ContextLm.rows(y, pick)).backward();
            opt.step(net.parameters());
            if (s % 10 == 9 || s == steps - 1) {
                acc = accuracy(net.predict(data.testX()), data.testLabels());
                if (reached < 0 && acc >= target) {
                    reached = s + 1;
                }
            }
        }
        return new SelectionRun(method, reached, acc, selectedNoisy / (double) selected);
    }

    private static double accuracy(Tensor logits, int[] labels) {
        float[] l = logits.toArray();
        int c = logits.shape().dim(1);
        int ok = 0;
        for (int i = 0; i < labels.length; i++) {
            int best = 0;
            for (int k = 1; k < c; k++) {
                best = l[i * c + k] > l[i * c + best] ? k : best;
            }
            ok += best == labels[i] ? 1 : 0;
        }
        return ok / (double) labels.length;
    }

    /**
     * Digits with label noise and duplicates, a clean holdout and a clean test set.
     *
     * @param x training inputs
     * @param labels training labels (some wrong)
     * @param noisy which training labels are wrong
     * @param holdoutX holdout inputs (clean labels)
     * @param holdoutY holdout labels as ids
     * @param testX test inputs
     * @param testLabels test labels
     */
    public record NoisyDigits(Tensor x, int[] labels, boolean[] noisy, Tensor holdoutX, Tensor holdoutY, Tensor testX,
            int[] testLabels) {
    }

    /**
     * Builds the selection task: 10 classes of 8x8 digit prototypes, 600 training points with
     * 20% of labels replaced at random, plus 100 of them repeated three more times (redundant
     * points), a 300-point holdout and a 500-point test set.
     *
     * @param seed the seed (prototypes and samples)
     * @return the data
     */
    public static NoisyDigits noisyDigits(long seed) {
        float[][][] sets = new float[3][][];
        int[][] labels = new int[3][];
        int[] perClass = {60, 30, 50};
        for (int s = 0; s < 3; s++) {
            ToyTasks.Data d = ToyTasks.digits(10, perClass[s], 8, 0.1, seed, seed * 13 + s);
            sets[s] = d.x();
            labels[s] = d.y();
        }
        SplittableRandom r = new SplittableRandom(seed + 5);
        int base = sets[0].length;
        int n = base + 300;
        float[][] x = new float[n][];
        int[] y = new int[n];
        boolean[] noisy = new boolean[n];
        for (int i = 0; i < base; i++) {
            x[i] = sets[0][i];
            y[i] = labels[0][i];
            if (r.nextDouble() < 0.2) {
                int wrong = (y[i] + 1 + r.nextInt(9)) % 10;
                y[i] = wrong;
                noisy[i] = true;
            }
        }
        for (int i = base; i < n; i++) {
            int src = (i - base) % 100;
            x[i] = x[src];
            y[i] = y[src];
            noisy[i] = noisy[src];
        }
        float[] hy = new float[labels[1].length];
        for (int i = 0; i < hy.length; i++) {
            hy[i] = labels[1][i];
        }
        return new NoisyDigits(Tensor.of(x), y, noisy, Tensor.of(sets[1]), Tensor.of(Shape.of(hy.length), hy),
                Tensor.of(sets[2]), labels[2]);
    }
}
