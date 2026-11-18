package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.BytePairTokenizer;
import io.github.zzuegg.brainiac.data.Tokenizer;
import io.github.zzuegg.brainiac.data.UnigramTokenizer;
import io.github.zzuegg.brainiac.experimental.data.BpeDropoutTokenizer;
import io.github.zzuegg.brainiac.experimental.data.LengthMaxTokenizer;
import io.github.zzuegg.brainiac.experimental.data.SegmentationStrategies;
import io.github.zzuegg.brainiac.experimental.data.StochasticTokenSplitter;
import io.github.zzuegg.brainiac.experimental.data.SuperBpeTokenizer;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * The wave 8 tokenizer shootout (8.4.12 to 8.4.14), next to {@link DynamicChunking}.
 *
 * <p>{@link #vocabularies}: on two corpora, the byte-level toy language of
 * {@link DynamicChunking} and a morphology corpus (words built from a prefix, a stem and a
 * suffix, so morpheme boundaries are known), each tokenizer learns the same number of tokens
 * beyond the characters: the core BPE and Unigram, {@link SuperBpeTokenizer},
 * {@link LengthMaxTokenizer}, and {@link SegmentationStrategies} applied to the BPE vocabulary
 * (longest prefix, fewest tokens, maximum likelihood) and to the Length-MAX vocabulary (fewest
 * tokens). Reported on held-out text: tokens per character, bits per byte of an interpolated
 * token-bigram LM (absolute discounting) counted on the training text, and the F1 of
 * word-internal token boundaries against morpheme boundaries.
 *
 * <p>{@link #subwordUnderstanding}: two character-level questions about tokenized inputs, a
 * letter count ({@code e}s in a morphology word) and a digit sum (of a four-digit number with a
 * BPE over digits), learned by a bag-of-tokens MLP from few examples and tested on held-out
 * words or numbers tokenized normally. Training inputs: plain BPE, BPE-dropout
 * ({@link BpeDropoutTokenizer}, p = 0.1), {@link StochasticTokenSplitter} (StochasTok,
 * p = 0.1, re-drawn every step) and characters (the upper bound). Reported: test accuracy and
 * test cross-entropy.
 */
public final class TokenizerShootout {

    private static final String[] PREFIXES = {"un", "re", "pre", "dis"};
    private static final String[] SUFFIXES = {"ing", "ed", "er", "ness", "ly", "s"};

    private TokenizerShootout() {
    }

    /**
     * A morphology word.
     *
     * @param text the word
     * @param boundaries the character offsets of morpheme boundaries inside it
     */
    public record Word(String text, Set<Integer> boundaries) {
    }

    /**
     * A morphology lexicon: 40 stems over {@code a-p} with optional prefixes and suffixes.
     *
     * @param random the random source
     * @param count words drawn (Zipf over stems)
     * @return the words
     */
    public static List<Word> morphologyWords(SplittableRandom random, int count) {
        SplittableRandom lex = new SplittableRandom(23);
        String[] stems = new String[40];
        for (int s = 0; s < stems.length; s++) {
            StringBuilder sb = new StringBuilder();
            int len = 3 + lex.nextInt(3);
            for (int i = 0; i < len; i++) {
                sb.append((char) ('a' + lex.nextInt(16)));
            }
            stems[s] = sb.toString();
        }
        double[] cdf = new double[stems.length];
        double z = 0;
        for (int s = 0; s < stems.length; s++) {
            z += 1.0 / (s + 1);
            cdf[s] = z;
        }
        List<Word> out = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            double u = random.nextDouble() * z;
            int s = 0;
            while (cdf[s] < u) {
                s++;
            }
            StringBuilder w = new StringBuilder();
            Set<Integer> b = new LinkedHashSet<>();
            if (random.nextDouble() < 0.4) {
                w.append(PREFIXES[random.nextInt(PREFIXES.length)]);
                b.add(w.length());
            }
            w.append(stems[s]);
            if (random.nextDouble() < 0.6) {
                b.add(w.length());
                w.append(SUFFIXES[random.nextInt(SUFFIXES.length)]);
            }
            out.add(new Word(w.toString(), b));
        }
        return out;
    }

    private static String join(List<Word> words) {
        StringBuilder sb = new StringBuilder();
        for (Word w : words) {
            sb.append(' ').append(w.text());
        }
        return sb.toString();
    }

    /**
     * One tokenizer on one corpus.
     *
     * @param tokensPerChar held-out tokens per character
     * @param bitsPerByte held-out bits per byte of the token-bigram LM
     * @param boundaryF1 morpheme-boundary F1 (morphology corpus; NaN on the other)
     */
    public record Row(double tokensPerChar, double bitsPerByte, double boundaryF1) {
    }

    /**
     * Runs the vocabulary and segmentation rows.
     *
     * @param learned tokens learned beyond the characters
     * @param seed the seed
     * @return "corpus" to tokenizer to row
     */
    public static Map<String, Map<String, Row>> vocabularies(int learned, long seed) {
        Map<String, Map<String, Row>> out = new LinkedHashMap<>();
        for (String corpus : List.of("toy language", "morphology")) {
            SplittableRandom r = new SplittableRandom(seed);
            String train;
            String test;
            List<Word> testWords = null;
            if (corpus.equals("toy language")) {
                train = DynamicChunking.text(r, 20000);
                test = DynamicChunking.text(r, 5000);
            } else {
                train = join(morphologyWords(r, 3000));
                testWords = morphologyWords(r, 800);
                test = join(testWords);
            }
            BytePairTokenizer bpe = BytePairTokenizer.train(train, 256 + learned);
            LengthMaxTokenizer lengthMax = LengthMaxTokenizer.train(train, learned, 12);
            Map<String, Tokenizer> toks = new LinkedHashMap<>();
            toks.put("BPE (merge order)", bpe);
            toks.put("Unigram", UnigramTokenizer.train(train, 256 + learned));
            toks.put("SuperBPE", SuperBpeTokenizer.train(train, 256 + learned / 2, 256 + learned));
            toks.put("Length-MAX (longest match)", lengthMax);
            SegmentationStrategies bpeVocab = SegmentationStrategies.of(bpe, SegmentationStrategies.Strategy.MERGE_ORDER)
                    .withUnigramCounts(train);
            toks.put("BPE vocab, longest prefix", bpeVocab.withStrategy(SegmentationStrategies.Strategy.LONGEST_PREFIX));
            toks.put("BPE vocab, least tokens", bpeVocab.withStrategy(SegmentationStrategies.Strategy.LEAST_TOKENS));
            toks.put("BPE vocab, max likelihood", bpeVocab.withStrategy(SegmentationStrategies.Strategy.MAX_LIKELIHOOD));
            toks.put("Length-MAX vocab, least tokens", SegmentationStrategies.of(lengthMax.pieces().toArray(String[]::new),
                    SegmentationStrategies.Strategy.LEAST_TOKENS));
            Map<String, Row> rows = new LinkedHashMap<>();
            for (Map.Entry<String, Tokenizer> e : toks.entrySet()) {
                Tokenizer t = e.getValue();
                int[] testIds = t.encode(test);
                double f1 = testWords == null ? Double.NaN : boundaryF1(t, testWords);
                rows.put(e.getKey(), new Row(testIds.length / (double) test.length(),
                        bitsPerByte(t.encode(train), testIds, t.size(), test.getBytes(StandardCharsets.UTF_8).length),
                        f1));
            }
            out.put(corpus, rows);
        }
        return out;
    }

    /** Interpolated absolute-discounting bigram LM: held-out bits per byte. */
    static double bitsPerByte(int[] train, int[] test, int vocab, int bytes) {
        Map<Long, Integer> pair = new HashMap<>();
        int[] context = new int[vocab];
        int[] distinct = new int[vocab];
        double[] uni = new double[vocab];
        for (int i = 0; i < train.length; i++) {
            uni[train[i]]++;
            if (i > 0) {
                long key = ((long) train[i - 1] << 32) | train[i];
                int c = pair.merge(key, 1, Integer::sum);
                context[train[i - 1]]++;
                if (c == 1) {
                    distinct[train[i - 1]]++;
                }
            }
        }
        double total = train.length + 0.01 * vocab;
        double bits = 0;
        double d = 0.75;
        for (int i = 0; i < test.length; i++) {
            double pu = (uni[test[i]] + 0.01) / total;
            double p = pu;
            if (i > 0 && context[test[i - 1]] > 0) {
                int v = test[i - 1];
                int c = pair.getOrDefault(((long) v << 32) | test[i], 0);
                p = Math.max(c - d, 0) / context[v] + d * distinct[v] / context[v] * pu;
            }
            bits -= Math.log(p) / Math.log(2);
        }
        return bits / bytes;
    }

    /** F1 of word-internal token boundaries against morpheme boundaries. */
    static double boundaryF1(Tokenizer t, List<Word> words) {
        int tp = 0;
        int predicted = 0;
        int gold = 0;
        for (Word w : words) {
            int[] ids = t.encode(" " + w.text());
            Set<Integer> cuts = new LinkedHashSet<>();
            int at = -1;   // offsets inside the word, after the leading space
            for (int id : ids) {
                at += t.decode(id).length();
                if (at > 0 && at < w.text().length()) {
                    cuts.add(at);
                }
            }
            predicted += cuts.size();
            gold += w.boundaries().size();
            for (int c : cuts) {
                tp += w.boundaries().contains(c) ? 1 : 0;
            }
        }
        double p = predicted == 0 ? 0 : tp / (double) predicted;
        double rc = gold == 0 ? 0 : tp / (double) gold;
        return p + rc == 0 ? 0 : 2 * p * rc / (p + rc);
    }

    // ------------------------------------------------------------------ 8.4.14

    /**
     * One input scheme on one task.
     *
     * @param accuracy held-out accuracy
     * @param loss held-out cross-entropy
     */
    public record TaskRow(double accuracy, double loss) {
    }

    private record Example(String text, int label) {
    }

    /**
     * Runs the subword-understanding rows.
     *
     * @param trainSize training examples
     * @param steps training steps
     * @param seeds the seeds
     * @return task to scheme to row, averaged over seeds
     */
    public static Map<String, Map<String, TaskRow>> subwordUnderstanding(int trainSize, int steps, long[] seeds) {
        Map<String, Map<String, double[]>> sums = new LinkedHashMap<>();
        for (long seed : seeds) {
            for (String task : List.of("letter count", "digit sum")) {
                SplittableRandom r = new SplittableRandom(seed * 7 + task.length());
                List<Example> all = new ArrayList<>();
                String corpus;
                if (task.equals("letter count")) {
                    Set<String> seen = new LinkedHashSet<>();
                    for (Word w : morphologyWords(r, 20000)) {
                        seen.add(w.text());
                    }
                    for (String w : seen) {
                        all.add(new Example(" " + w, (int) Math.min(3, w.chars().filter(c -> c == 'e').count())));
                    }
                    corpus = join(morphologyWords(r, 3000));
                } else {
                    StringBuilder sb = new StringBuilder();
                    for (int i = 0; i < 3000; i++) {
                        sb.append(' ').append(1000 + r.nextInt(9000));
                    }
                    corpus = sb.toString();
                    for (int i = 0; i < 4000; i++) {
                        int n = 1000 + r.nextInt(9000);
                        int s = 0;
                        for (char c : Integer.toString(n).toCharArray()) {
                            s += c - '0';
                        }
                        all.add(new Example(" " + n, s));
                    }
                }
                shuffle(all, r);
                List<Example> train = all.subList(0, Math.min(trainSize, all.size() / 2));
                List<Example> test = all.subList(all.size() / 2, all.size());
                int classes = task.equals("letter count") ? 4 : 37;
                BytePairTokenizer bpe = BytePairTokenizer.train(corpus, 256 + 64);
                BpeDropoutTokenizer dropout = new BpeDropoutTokenizer(bpe, 0.1, seed);
                StochasticTokenSplitter stok = StochasticTokenSplitter.of(bpe, 0.1, 1);
                Map<String, Function<String, int[]>> trainInputs = new LinkedHashMap<>();
                SplittableRandom noise = new SplittableRandom(seed + 5);
                trainInputs.put("BPE", bpe::encode);
                trainInputs.put("BPE-dropout", s -> dropout.encode(s, 0.1, noise));
                trainInputs.put("StochasTok", s -> stok.split(bpe.encode(s), noise));
                trainInputs.put("characters", TokenizerShootout::bytes);
                for (Map.Entry<String, Function<String, int[]>> e : trainInputs.entrySet()) {
                    Function<String, int[]> testInput = e.getKey().equals("characters") ? TokenizerShootout::bytes
                            : bpe::encode;
                    TaskRow row = trainBag(train, test, e.getValue(), testInput, bpe.size(), classes, steps, seed);
                    double[] s = sums.computeIfAbsent(task, k -> new LinkedHashMap<>()).computeIfAbsent(e.getKey(),
                            k -> new double[2]);
                    s[0] += row.accuracy() / seeds.length;
                    s[1] += row.loss() / seeds.length;
                }
            }
        }
        Map<String, Map<String, TaskRow>> out = new LinkedHashMap<>();
        sums.forEach((task, m) -> {
            Map<String, TaskRow> rows = new LinkedHashMap<>();
            m.forEach((k, s) -> rows.put(k, new TaskRow(s[0], s[1])));
            out.put(task, rows);
        });
        return out;
    }

    private static int[] bytes(String s) {
        byte[] b = s.getBytes(StandardCharsets.UTF_8);
        int[] ids = new int[b.length];
        for (int i = 0; i < b.length; i++) {
            ids[i] = b[i] & 0xff;
        }
        return ids;
    }

    private static <T> void shuffle(List<T> list, SplittableRandom r) {
        for (int i = list.size() - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            T t = list.get(i);
            list.set(i, list.get(j));
            list.set(j, t);
        }
    }

    /** A bag-of-tokens MLP: summed token embeddings, one tanh layer, a classifier. */
    private static TaskRow trainBag(List<Example> train, List<Example> test, Function<String, int[]> trainInput,
            Function<String, int[]> testInput, int vocab, int classes, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed + 11);
        Var embed = param(r, 0.3f, vocab, 32);
        Var w1 = param(r, 0.18f, 32, 64);
        Var w2 = param(r, 0.125f, 64, classes);
        Optimizer opt = Optimizer.adam(1e-2f);
        SplittableRandom pick = new SplittableRandom(seed + 12);
        for (int s = 0; s < steps; s++) {
            List<Example> batch = new ArrayList<>();
            for (int i = 0; i < 64; i++) {
                batch.add(train.get(pick.nextInt(train.size())));
            }
            Var loss = Functional.crossEntropy(forward(batch, trainInput, vocab, embed, w1, w2), labels(batch));
            loss.backward();
            opt.step(List.of(embed, w1, w2));
        }
        Var logits = Var.noGrad(() -> forward(test, testInput, vocab, embed, w1, w2));
        float loss = Var.noGrad(() -> Functional.crossEntropy(logits, labels(test))).value().item();
        float[] l = logits.value().toArray();
        int ok = 0;
        for (int i = 0; i < test.size(); i++) {
            int best = 0;
            for (int c = 1; c < classes; c++) {
                if (l[i * classes + c] > l[i * classes + best]) {
                    best = c;
                }
            }
            ok += best == test.get(i).label() ? 1 : 0;
        }
        return new TaskRow(ok / (double) test.size(), loss);
    }

    private static Var forward(List<Example> batch, Function<String, int[]> input, int vocab, Var embed, Var w1,
            Var w2) {
        float[] bag = new float[batch.size() * vocab];
        for (int i = 0; i < batch.size(); i++) {
            for (int id : input.apply(batch.get(i).text())) {
                bag[i * vocab + id]++;
            }
        }
        return Var.of(Tensor.of(Shape.of(batch.size(), vocab), bag)).matmul(embed).matmul(w1).tanh().matmul(w2);
    }

    private static Tensor labels(List<Example> batch) {
        float[] f = new float[batch.size()];
        for (int i = 0; i < f.length; i++) {
            f[i] = batch.get(i).label();
        }
        return Tensor.of(Shape.of(f.length), f);
    }

    private static Var param(SplittableRandom r, float scale, int... dims) {
        int n = dims[0] * dims[1];
        float[] f = new float[n];
        for (int i = 0; i < n; i++) {
            f[i] = (float) (r.nextGaussian() * scale);
        }
        return Var.param(Tensor.of(Shape.of(dims), f));
    }
}
