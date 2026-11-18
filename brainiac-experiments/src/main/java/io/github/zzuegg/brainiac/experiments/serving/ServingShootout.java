package io.github.zzuegg.brainiac.experiments.serving;

import io.github.zzuegg.brainiac.experimental.llm.QuestKvStore;
import io.github.zzuegg.brainiac.experimental.nn.SnapKv;
import io.github.zzuegg.brainiac.experimental.serving.DraftGuidedKvSelector;
import io.github.zzuegg.brainiac.experimental.serving.PerplexityPromptCompressor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * The wave 8 serving shootout, approximate-inference and prompt-compression rows (8.4.1 and
 * 8.4.2): every method keeps a share of the prompt and answers the three-hop
 * {@link ChainRetrievalTask}. The other row groups live in {@code SpeculativeShootout},
 * {@code ThroughputShootout}, {@code KvMemoryShootout} and {@code StructuredOutputShootout}.
 * <ul>
 * <li>KV eviction (the target reads the whole prompt once, then keeps a budget of KV entries):
 *     the full cache, {@link SnapKv} (window of the question), heavy hitters (H2O: cumulative
 *     attention of every prompt query plus a recent window, as
 *     {@code EvictingKvCache.heavyHitters}), Quest ({@link QuestKvStore#selectPages}: nothing is
 *     evicted, each decode step reads its best pages up to the budget) and
 *     {@link DraftGuidedKvSelector#specKv SpecKV};</li>
 * <li>prompt compression (the target reads only the kept tokens): truncation (keep the end),
 *     {@link PerplexityPromptCompressor LLMLingua and LongLLMLingua} with a cache-bigram small
 *     LM, and {@link DraftGuidedKvSelector#specPc SpecPC}.</li>
 * </ul>
 * Reported per budget: exact-answer accuracy, the target's prefill tokens and the draft's extra
 * tokens (at a quarter of the target's width).
 */
public final class ServingShootout {

    private static final int HOPS = 3;
    private static final int WINDOW = 4;
    private static final int DOC = 16;

    private ServingShootout() {
    }

    /**
     * One method at one budget.
     *
     * @param accuracy share of exactly answered problems
     * @param prefillTokens target tokens read per problem
     * @param draftTokens draft tokens read and written per problem (0 without a draft)
     */
    public record Cell(double accuracy, double prefillTokens, double draftTokens) {
    }

    /**
     * Runs the approximate-inference and prompt-compression rows.
     *
     * @param length the prompt length
     * @param distractors distractor pairs per context
     * @param budgets the kept shares of the prompt, such as 0.1 to 0.5
     * @param problems problems per seed
     * @param seeds the seeds (at least 3 for the shootout)
     * @return method to one cell per budget, averaged over seeds
     */
    public static Map<String, Cell[]> approximateInference(int length, int distractors, double[] budgets, int problems,
            long[] seeds) {
        String[] names = {"full cache", "SnapKV", "heavy hitters (H2O)", "Quest (reads budget)", "SpecKV",
            "truncation", "LLMLingua", "LongLLMLingua", "SpecPC"};
        Map<String, double[][]> sums = new LinkedHashMap<>();
        for (String n : names) {
            sums.put(n, new double[budgets.length][3]);
        }
        for (long seed : seeds) {
            ChainRetrievalTask.Reader target = new ChainRetrievalTask.Reader(64, 12, seed);
            ChainRetrievalTask.Reader draft = new ChainRetrievalTask.Reader(16, 12, seed + 7);
            SplittableRandom random = new SplittableRandom(seed + 1);
            List<ChainRetrievalTask.Problem> train = new ArrayList<>();
            for (int i = 0; i < 200; i++) {
                train.add(ChainRetrievalTask.problem(length, HOPS, distractors, random));
            }
            ChainRetrievalTask.CacheBigramLm small = new ChainRetrievalTask.CacheBigramLm(train);
            PerplexityPromptCompressor lingua = PerplexityPromptCompressor.llmLingua(small, DOC);
            PerplexityPromptCompressor longLingua = PerplexityPromptCompressor.longLlmLingua(small, DOC, 1.0);
            DraftGuidedKvSelector specKv = DraftGuidedKvSelector.specKv(WINDOW, 1);
            DraftGuidedKvSelector specPc = DraftGuidedKvSelector.specPc(WINDOW, 1, 3);
            SnapKv snap = SnapKv.of(WINDOW, 3);
            for (int p = 0; p < problems; p++) {
                ChainRetrievalTask.Problem pr = ChainRetrievalTask.problem(length, HOPS, distractors, random);
                int[] prompt = pr.prompt();
                int n = prompt.length;
                float[][] keys = target.keys(prompt);
                float[][] values = target.values(prompt);
                // the draft reads the whole prompt and writes its answer
                int[] draftAnswer = draft.read(prompt, pr.start(), HOPS);
                for (int b = 0; b < budgets.length; b++) {
                    int keep = Math.max(WINDOW + 1, (int) Math.round(budgets[b] * n));
                    Map<String, int[]> answers = new LinkedHashMap<>();
                    Map<String, double[]> costs = new LinkedHashMap<>();
                    answers.put("full cache", target.decode(keys, values, pr.start(), HOPS));
                    costs.put("full cache", new double[] {n, 0});
                    // SnapKV: the window's queries vote once
                    float[][] votes = new float[WINDOW][];
                    for (int i = 0; i < WINDOW; i++) {
                        int at = n - WINDOW + i;
                        votes[i] = target.attend(target.query(prompt[at]), keys, n - WINDOW);
                    }
                    answers.put("SnapKV", decodeKept(target, keys, values, snap.select(votes, keep), pr));
                    costs.put("SnapKV", new double[] {n, 0});
                    answers.put("heavy hitters (H2O)", decodeKept(target, keys, values, heavyHitters(target, prompt,
                            keys, keep), pr));
                    costs.put("heavy hitters (H2O)", new double[] {n, 0});
                    answers.put("Quest (reads budget)", quest(target, keys, values, pr, keep));
                    costs.put("Quest (reads budget)", new double[] {n, 0});
                    // SpecKV: the target's queries of the window and of the draft's answer
                    float[][] queries = new float[WINDOW + HOPS][];
                    for (int i = 0; i < WINDOW; i++) {
                        queries[i] = target.query(prompt[n - WINDOW + i]);
                    }
                    for (int h = 0; h < HOPS; h++) {
                        queries[WINDOW + h] = target.query(draftAnswer[h]);
                    }
                    answers.put("SpecKV", decodeKept(target, keys, values, specKv.select(queries, keys, 1, keep), pr));
                    costs.put("SpecKV", new double[] {n + HOPS, n + HOPS});
                    // prompt compression: the target reads only what is kept
                    int[] truncated = Arrays.copyOfRange(prompt, n - keep, n);
                    answers.put("truncation", target.read(truncated, pr.start(), HOPS));
                    costs.put("truncation", new double[] {keep, 0});
                    int[] question = Arrays.copyOfRange(prompt, n - 3, n);
                    List<int[]> docs = ChainRetrievalTask.documents(prompt, DOC);
                    for (Map.Entry<String, PerplexityPromptCompressor> e : List.of(Map.entry("LLMLingua", lingua),
                            Map.entry("LongLLMLingua", longLingua))) {
                        PerplexityPromptCompressor.Compressed c = e.getValue().compress(docs, question, keep - 3);
                        answers.put(e.getKey(), target.read(c.tokens(), pr.start(), HOPS));
                        costs.put(e.getKey(), new double[] {c.tokens().length, c.lmTokens()});
                    }
                    float[][] draftKeys = draft.keys(prompt);
                    float[][] rows = new float[WINDOW + HOPS][];
                    for (int i = 0; i < WINDOW; i++) {
                        rows[i] = draft.attend(draft.query(prompt[n - WINDOW + i]), draftKeys, n - WINDOW);
                    }
                    for (int h = 0; h < HOPS; h++) {
                        rows[WINDOW + h] = draft.attend(draft.query(draftAnswer[h]), draftKeys, n - WINDOW);
                    }
                    boolean[] kept = specPc.selectFromAttention(rows, n, keep);
                    int[] compressed = keptTokens(prompt, kept);
                    answers.put("SpecPC", target.read(compressed, pr.start(), HOPS));
                    costs.put("SpecPC", new double[] {compressed.length, n + HOPS});
                    for (String name : names) {
                        double[] s = sums.get(name)[b];
                        s[0] += Arrays.equals(answers.get(name), pr.answer()) ? 1 : 0;
                        s[1] += costs.get(name)[0];
                        s[2] += costs.get(name)[1];
                    }
                }
            }
        }
        Map<String, Cell[]> out = new LinkedHashMap<>();
        double total = (double) problems * seeds.length;
        for (String n : names) {
            Cell[] cells = new Cell[budgets.length];
            for (int b = 0; b < budgets.length; b++) {
                double[] s = sums.get(n)[b];
                cells[b] = new Cell(s[0] / total, s[1] / total, s[2] / total);
            }
            out.put(n, cells);
        }
        return out;
    }

    private static int[] decodeKept(ChainRetrievalTask.Reader target, float[][] keys, float[][] values, boolean[] kept,
            ChainRetrievalTask.Problem pr) {
        List<float[]> k = new ArrayList<>();
        List<float[]> v = new ArrayList<>();
        for (int i = 0; i < kept.length; i++) {
            if (kept[i]) {
                k.add(keys[i]);
                v.add(values[i]);
            }
        }
        return target.decode(k.toArray(float[][]::new), v.toArray(float[][]::new), pr.start(), HOPS);
    }

    static int[] keptTokens(int[] prompt, boolean[] kept) {
        int c = 0;
        for (boolean k : kept) {
            c += k ? 1 : 0;
        }
        int[] out = new int[c];
        int j = 0;
        for (int i = 0; i < prompt.length; i++) {
            if (kept[i]) {
                out[j++] = prompt[i];
            }
        }
        return out;
    }

    /** H2O: cumulative causal attention of every prompt query, plus the most recent tokens. */
    private static boolean[] heavyHitters(ChainRetrievalTask.Reader target, int[] prompt, float[][] keys, int keep) {
        int n = prompt.length;
        double[] score = new double[n];
        for (int t = 1; t < n; t++) {
            float[] w = target.attend(target.query(prompt[t]), keys, t + 1);
            for (int j = 0; j <= t; j++) {
                score[j] += w[j];
            }
        }
        boolean[] kept = new boolean[n];
        int recent = Math.min(keep / 2, WINDOW);
        for (int i = n - recent; i < n; i++) {
            kept[i] = true;
        }
        Integer[] order = new Integer[n - recent];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(score[b], score[a]));
        for (int i = 0; i < keep - recent && i < order.length; i++) {
            kept[order[i]] = true;
        }
        return kept;
    }

    /** Quest: every decode step reads the pages with the highest key bound, up to the budget. */
    private static int[] quest(ChainRetrievalTask.Reader target, float[][] keys, float[][] values,
            ChainRetrievalTask.Problem pr, int keep) {
        int page = 4;
        int pages = Math.max(1, keep / page);
        int[] out = new int[HOPS];
        int token = pr.start();
        for (int h = 0; h < HOPS; h++) {
            boolean[] kept = QuestKvStore.selectPages(keys, target.query(token), page, pages);
            List<float[]> k = new ArrayList<>();
            List<float[]> v = new ArrayList<>();
            for (int i = 0; i < keys.length; i++) {
                if (kept[i]) {
                    k.add(keys[i]);
                    v.add(values[i]);
                }
            }
            out[h] = target.decode(k.toArray(float[][]::new), v.toArray(float[][]::new), token, 1)[0];
            token = out[h];
        }
        return out;
    }
}
