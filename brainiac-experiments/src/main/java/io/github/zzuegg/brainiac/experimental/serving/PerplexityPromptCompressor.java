package io.github.zzuegg.brainiac.experimental.serving;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * Budgeted prompt compression by a small language model: LLMLingua (Jiang et al., EMNLP 2023,
 * <a href="https://arxiv.org/abs/2310.05736">arXiv:2310.05736</a>) and its question-aware
 * successor LongLLMLingua (Jiang et al., ACL 2024,
 * <a href="https://arxiv.org/abs/2310.06839">arXiv:2310.06839</a>). Tokens a small LM finds
 * predictable carry little information and are dropped. The compressed prompt is plain tokens,
 * so it works with any target model, unlike KV eviction.
 *
 * <p>The pipeline follows the papers:
 * <ol>
 * <li><b>budget controller</b>: the question (and any instruction) is kept whole; only the
 *     documents (demonstrations) are compressed;</li>
 * <li><b>coarse level</b>: whole documents are ranked and the best kept until
 *     {@code coarseSlack} times the token budget. LLMLingua ranks by the small LM's perplexity
 *     of the document (more surprising is more informative); LongLLMLingua ranks by the
 *     question's log-likelihood given the document ({@code log p(question | document)}), and
 *     reorders kept documents most relevant first;</li>
 * <li><b>fine level, iterative</b>: the kept documents are compressed segment by segment;
 *     every token's surprisal is computed by the small LM conditioned on the
 *     <em>compressed</em> text so far, and the most surprising tokens of the segment are kept
 *     under the segment's share of the budget. LongLLMLingua adds a contrastive term, the gain
 *     in log-probability when the question is put in front ({@code log p(x | q, x<) -
 *     log p(x | x<)}), so tokens related to the question rank higher.</li>
 * </ol>
 */
public final class PerplexityPromptCompressor {

    /** A small causal LM that scores token sequences. */
    @FunctionalInterface
    public interface ScoringLm {
        /**
         * @param tokens a sequence
         * @return {@code log p(tokens[i] | tokens[<i])} for every position
         */
        double[] logProbs(int[] tokens);
    }

    /**
     * A compressed prompt.
     *
     * @param tokens the kept document tokens followed by the question
     * @param documentTokens how many document tokens were kept
     * @param lmTokens the tokens the small LM scored (its cost)
     */
    public record Compressed(int[] tokens, int documentTokens, long lmTokens) {
    }

    private final ScoringLm lm;
    private final boolean questionAware;
    private final int segment;
    private final double contrast;
    private final double coarseSlack;

    private PerplexityPromptCompressor(ScoringLm lm, boolean questionAware, int segment, double contrast,
            double coarseSlack) {
        if (segment < 1 || coarseSlack < 1) {
            throw new IllegalArgumentException("segment must be positive and coarseSlack at least 1");
        }
        this.lm = Objects.requireNonNull(lm, "lm");
        this.questionAware = questionAware;
        this.segment = segment;
        this.contrast = contrast;
        this.coarseSlack = coarseSlack;
    }

    /**
     * LLMLingua: perplexity-ranked documents, then iterative token-level compression.
     *
     * @param lm the small LM
     * @param segment the tokens compressed per iteration
     * @return the compressor
     */
    public static PerplexityPromptCompressor llmLingua(ScoringLm lm, int segment) {
        return new PerplexityPromptCompressor(lm, false, segment, 0, 2);
    }

    /**
     * LongLLMLingua: question-aware document ranking and contrastive token scores.
     *
     * @param lm the small LM
     * @param segment the tokens compressed per iteration
     * @param contrast the weight of the contrastive (question-conditioned) term
     * @return the compressor
     */
    public static PerplexityPromptCompressor longLlmLingua(ScoringLm lm, int segment, double contrast) {
        return new PerplexityPromptCompressor(lm, true, segment, contrast, 2);
    }

    /**
     * @param slack the coarse level keeps documents up to this many times the budget
     * @return a copy with that slack
     */
    public PerplexityPromptCompressor coarseSlack(double slack) {
        return new PerplexityPromptCompressor(lm, questionAware, segment, contrast, slack);
    }

    /**
     * Compresses documents to a token budget and appends the question.
     *
     * @param documents the documents, in prompt order
     * @param question the question, kept whole
     * @param budget the document tokens to keep
     * @return the compressed prompt
     */
    public Compressed compress(List<int[]> documents, int[] question, int budget) {
        long[] cost = new long[1];
        // coarse: rank documents
        int d = documents.size();
        double[] rank = new double[d];
        for (int i = 0; i < d; i++) {
            int[] doc = documents.get(i);
            if (questionAware) {
                double[] lp = score(concat(doc, question), cost);
                double s = 0;
                for (int t = doc.length; t < lp.length; t++) {
                    s += lp[t];
                }
                rank[i] = s / Math.max(1, question.length);
            } else {
                double[] lp = score(doc, cost);
                double s = 0;
                for (double v : lp) {
                    s -= v;
                }
                rank[i] = s / Math.max(1, doc.length);
            }
        }
        Integer[] order = new Integer[d];
        for (int i = 0; i < d; i++) {
            order[i] = i;
        }
        Arrays.sort(order, Comparator.comparingDouble((Integer i) -> -rank[i]));
        List<Integer> chosen = new ArrayList<>();
        int total = 0;
        for (int i : order) {
            if (total >= coarseSlack * budget) {
                break;
            }
            chosen.add(i);
            total += documents.get(i).length;
        }
        if (!questionAware) {
            chosen.sort(Integer::compare);   // LLMLingua keeps prompt order; LongLLMLingua reorders by relevance
        }
        int[] text = new int[0];
        for (int i : chosen) {
            text = concat(text, documents.get(i));
        }
        // fine: iterative token-level compression
        int[] kept = new int[0];
        int remainingBudget = Math.min(budget, text.length);
        for (int start = 0; start < text.length; start += segment) {
            int end = Math.min(text.length, start + segment);
            int[] seg = Arrays.copyOfRange(text, start, end);
            int left = text.length - start;
            int take = (int) Math.round(remainingBudget * (double) seg.length / left);
            take = Math.max(0, Math.min(seg.length, take));
            double[] s = surprisal(kept, seg, question, cost);
            Integer[] idx = new Integer[seg.length];
            for (int i = 0; i < seg.length; i++) {
                idx[i] = i;
            }
            Arrays.sort(idx, Comparator.comparingDouble((Integer i) -> -s[i]));
            boolean[] keep = new boolean[seg.length];
            for (int i = 0; i < take; i++) {
                keep[idx[i]] = true;
            }
            for (int i = 0; i < seg.length; i++) {
                if (keep[i]) {
                    kept = concat(kept, new int[] {seg[i]});
                }
            }
            remainingBudget -= take;
        }
        return new Compressed(concat(kept, question), kept.length, cost[0]);
    }

    /** The token scores of a segment given the compressed text before it. */
    private double[] surprisal(int[] prefix, int[] seg, int[] question, long[] cost) {
        double[] plain = score(concat(prefix, seg), cost);
        double[] s = new double[seg.length];
        for (int i = 0; i < seg.length; i++) {
            s[i] = -plain[prefix.length + i];
        }
        if (questionAware && contrast != 0) {
            int[] withQ = concat(concat(question, prefix), seg);
            double[] cond = score(withQ, cost);
            int off = question.length + prefix.length;
            for (int i = 0; i < seg.length; i++) {
                s[i] += contrast * (cond[off + i] - plain[prefix.length + i]);
            }
        }
        return s;
    }

    private double[] score(int[] tokens, long[] cost) {
        cost[0] += tokens.length;
        return lm.logProbs(tokens);
    }

    private static int[] concat(int[] a, int[] b) {
        int[] c = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, c, a.length, b.length);
        return c;
    }

    @Override
    public String toString() {
        return (questionAware ? "LongLLMLingua" : "LLMLingua") + "(segment " + segment + ")";
    }
}
