package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.tensor.DType;

import java.util.Arrays;

/**
 * Evaluating language models: perplexity, log-likelihoods for multiple-choice benchmarks,
 * and a needle-in-a-haystack test of long-context recall.
 */
public final class Evaluation {

    private Evaluation() {
    }

    /**
     * The perplexity of a text, {@code exp(mean -log p(token | earlier tokens))}, read in
     * windows of {@code window} tokens, each from a fresh state; every window's first token is
     * conditioned on nothing earlier and not scored.
     *
     * @param model the model
     * @param tokens the text's token ids, at least two
     * @param window the tokens per window, at least two
     * @return the perplexity
     */
    public static double perplexity(LanguageModel model, int[] tokens, int window) {
        if (tokens.length < 2 || window < 2) {
            throw new IllegalArgumentException("perplexity needs at least two tokens and windows of two");
        }
        double nll = 0;
        long count = 0;
        for (int from = 0; from + 1 < tokens.length; from += window) {
            int to = Math.min(tokens.length, from + window);
            int[] part = Arrays.copyOfRange(tokens, from, to);
            double[] lp = tokenLogProbs(model, part, 1);
            for (double l : lp) {
                nll -= l;
            }
            count += lp.length;
        }
        return Math.exp(nll / count);
    }

    /**
     * The log-probability of a continuation after a context, summed over its tokens.
     *
     * @param model the model
     * @param context the context's tokens, at least one
     * @param continuation the continuation's tokens, at least one
     * @return {@code log p(continuation | context)}
     */
    public static double logLikelihood(LanguageModel model, int[] context, int[] continuation) {
        int[] all = Arrays.copyOf(context, context.length + continuation.length);
        System.arraycopy(continuation, 0, all, context.length, continuation.length);
        double sum = 0;
        for (double l : tokenLogProbs(model, all, context.length)) {
            sum += l;
        }
        return sum;
    }

    /**
     * Picks the most likely of several answers, as multiple-choice benchmarks (HellaSwag,
     * ARC, MMLU in cloze form) are scored.
     *
     * @param model the model
     * @param context the question's tokens
     * @param perToken whether to compare the mean log-probability per token rather than the
     *     sum, so longer answers are not penalized
     * @param choices each answer's tokens
     * @return the index of the chosen answer
     */
    public static int multipleChoice(LanguageModel model, int[] context, boolean perToken, int[]... choices) {
        int best = -1;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < choices.length; i++) {
            double s = logLikelihood(model, context, choices[i]);
            if (perToken) {
                s /= choices[i].length;
            }
            if (s > bestScore) {
                bestScore = s;
                best = i;
            }
        }
        return best;
    }

    /**
     * Needle in a haystack: a fact hidden at several depths of filler text of several lengths,
     * followed by a question about it; a cell passes when the greedy answer contains the
     * expected text.
     *
     * @param generator the model with its tokenizer
     * @param filler text repeated to make the haystack
     * @param needle the sentence holding the fact
     * @param question asked after the haystack
     * @param answer what a correct answer contains (ignoring case)
     * @param lengths haystack lengths in characters
     * @param depths where the needle goes, as fractions of the haystack in {@code [0, 1]}
     * @param maxNewTokens tokens to generate per answer
     * @return {@code [length][depth]} whether the answer was found
     */
    public static boolean[][] needleInHaystack(Generator generator, String filler, String needle, String question,
            String answer, int[] lengths, double[] depths, int maxNewTokens) {
        boolean[][] found = new boolean[lengths.length][depths.length];
        for (int l = 0; l < lengths.length; l++) {
            StringBuilder hay = new StringBuilder();
            while (hay.length() < lengths[l]) {
                hay.append(filler).append(' ');
            }
            hay.setLength(lengths[l]);
            for (int d = 0; d < depths.length; d++) {
                int at = (int) Math.round(Math.clamp(depths[d], 0, 1) * hay.length());
                while (at > 0 && at < hay.length() && hay.charAt(at - 1) != ' ') {
                    at--;   // between words
                }
                String prompt = hay.substring(0, at) + " " + needle + " " + hay.substring(at) + "\n" + question;
                String out = generator.generate(prompt, Generator.Options.greedy(maxNewTokens));
                found[l][d] = out.toLowerCase(java.util.Locale.ROOT)
                        .contains(answer.toLowerCase(java.util.Locale.ROOT));
            }
        }
        return found;
    }

    /** {@code log p(tokens[t] | tokens[..t])} for every {@code t >= from}. */
    private static double[] tokenLogProbs(LanguageModel model, int[] tokens, int from) {
        float[] logits = model.nextAll(model.newState(1), new int[][] {tokens}).to(DType.FLOAT32).toArray();
        int vocab = model.vocabSize();
        double[] out = new double[tokens.length - from];
        for (int t = from; t < tokens.length; t++) {
            int row = (t - 1) * vocab;   // the logits after token t - 1 predict token t
            double max = Double.NEGATIVE_INFINITY;
            for (int v = 0; v < vocab; v++) {
                max = Math.max(max, logits[row + v]);
            }
            double sum = 0;
            for (int v = 0; v < vocab; v++) {
                sum += Math.exp(logits[row + v] - max);
            }
            out[t - from] = logits[row + tokens[t]] - max - Math.log(sum);
        }
        return out;
    }
}
