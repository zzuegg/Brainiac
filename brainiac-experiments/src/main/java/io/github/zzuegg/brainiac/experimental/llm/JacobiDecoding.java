package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.llm.LanguageModel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Parallel greedy decoding without a draft model. Each step feeds the model the last
 * accepted token followed by a guess of the next {@code window} tokens and checks the guess
 * in one pass ({@code nextAll}): the longest prefix of the guess that matches the model's own
 * greedy predictions is accepted, plus one token of the model's own. The output is exactly
 * greedy decoding's, in fewer model passes whenever guesses hit.
 *
 * <p>Where the guess comes from:
 * <ul>
 * <li>{@link Guess#JACOBI}: Jacobi decoding (Santilli et al. 2023,
 *     <a href="https://arxiv.org/abs/2305.10427">arXiv:2305.10427</a>): the guess is the
 *     previous step's predictions for those positions, iterated to a fixed point.</li>
 * <li>{@link Guess#LOOKAHEAD}: lookahead decoding (Fu et al. 2024,
 *     <a href="https://arxiv.org/abs/2402.02057">arXiv:2402.02057</a>), simplified to one
 *     candidate: n-grams seen along the Jacobi trajectories and in the verified text are kept in a
 *     pool keyed by their first token, and a pool hit replaces the Jacobi guess.</li>
 * <li>{@link Guess#PROMPT_LOOKUP}: prompt lookup decoding (Saxena 2023): the pool is also
 *     seeded with the prompt's n-grams, which helps when the output copies from the input
 *     (summaries, code edits, retrieval).</li>
 * </ul>
 * The model's state must support {@code truncate}.
 */
public final class JacobiDecoding {

    /** Where guesses come from. */
    public enum Guess {
        /** the previous step's predictions */
        JACOBI,
        /** an n-gram pool from the trajectories and output, else Jacobi */
        LOOKAHEAD,
        /** as {@link #LOOKAHEAD}, with the prompt's n-grams added to the pool */
        PROMPT_LOOKUP
    }

    /**
     * @param tokens the generated tokens
     * @param modelCalls the model passes used
     */
    public record Result(int[] tokens, int modelCalls) {
    }

    private JacobiDecoding() {
    }

    /**
     * Greedy decoding with guessed tokens.
     *
     * @param model the model
     * @param prompt the prompt
     * @param maxNew the tokens to generate
     * @param window the tokens guessed per pass
     * @param guess where guesses come from
     * @return the tokens (equal to greedy decoding's) and the passes used
     */
    public static Result generate(LanguageModel model, int[] prompt, int maxNew, int window, Guess guess) {
        LanguageModel.State state = model.newState(1);
        float[] first = model.next(state, new int[][] {prompt}, null).toArray();
        int calls = 1;
        List<Integer> out = new ArrayList<>();
        int last = LogitMath.argmax(first);
        out.add(last);
        Map<Integer, int[]> pool = new HashMap<>();
        if (guess == Guess.PROMPT_LOOKUP) {
            remember(pool, prompt, window);
        }
        int[] jacobi = new int[window];
        Arrays.fill(jacobi, last);
        while (out.size() < maxNew) {
            int[] g = guess != Guess.JACOBI && pool.containsKey(last) ? pool.get(last) : jacobi;
            int[] input = new int[window + 1];
            input[0] = last;
            System.arraycopy(g, 0, input, 1, window);
            int before = state.length();
            float[] logits = model.nextAll(state, new int[][] {input}).toArray();
            calls++;
            int vocab = model.vocabSize();
            int[] pred = new int[window + 1];
            for (int i = 0; i <= window; i++) {
                pred[i] = LogitMath.argmax(Arrays.copyOfRange(logits, i * vocab, (i + 1) * vocab));
            }
            int accepted = 0;
            while (accepted < window && g[accepted] == pred[accepted]) {
                accepted++;
            }
            for (int i = 0; i <= accepted && out.size() < maxNew; i++) {
                out.add(pred[i]);
            }
            state = state.truncate(before + 1 + accepted);
            last = pred[accepted];
            if (guess != Guess.JACOBI) {
                remember(pool, pred, window);
                int[] verified = new int[prompt.length + out.size()];
                System.arraycopy(prompt, 0, verified, 0, prompt.length);
                for (int i = 0; i < out.size(); i++) {
                    verified[prompt.length + i] = out.get(i);
                }
                remember(pool, verified, window);
            }
            jacobi = new int[window];
            for (int i = 0; i < window; i++) {
                int src = accepted + 1 + i;
                jacobi[i] = src <= window ? pred[src] : pred[window];
            }
        }
        return new Result(out.stream().mapToInt(Integer::intValue).toArray(), calls);
    }

    /** Stores every n-gram of {@code seq}: the {@code window} tokens after each token. */
    private static void remember(Map<Integer, int[]> pool, int[] seq, int window) {
        for (int i = 0; i + window < seq.length; i++) {
            pool.put(seq[i], Arrays.copyOfRange(seq, i + 1, i + 1 + window));
        }
    }
}
