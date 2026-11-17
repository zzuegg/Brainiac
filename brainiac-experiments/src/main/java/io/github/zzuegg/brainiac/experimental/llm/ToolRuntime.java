package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.data.Tokenizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Runs a language model that can call a tool in the middle of its output. The model writes
 * {@code [expression>}; when it writes the result marker the runtime evaluates the
 * expression with the {@link Tool}, splices {@code result]} into the sequence, and lets the
 * model continue, now conditioned on the result. This is the inference half of Toolformer
 * (<a href="https://arxiv.org/abs/2302.04761">2302.04761</a>); {@link ToolformerFilter} is the
 * data half.
 *
 * <pre>{@code
 * ToolRuntime rt = ToolRuntime.of(tokenizer, ToolRuntime.Tool.CALCULATOR, '[', '>', ']');
 * ToolRuntime.Trace t = rt.generate(lm, tokenizer.encode("7*8="), 16, eos, 0, random);
 * // t.text() == "7*8=[7*8>56]56#", where "56]" came from the calculator
 * }</pre>
 */
public final class ToolRuntime {

    /** A tool: text arguments in, text result out. */
    @FunctionalInterface
    public interface Tool {
        /**
         * @param arguments what the model wrote between the brackets
         * @return the result, or an error marker
         */
        String call(String arguments);

        /** Integer arithmetic over {@code + - *} with the usual precedence; {@code ?} on errors. */
        Tool CALCULATOR = ToolRuntime::calculate;
    }

    /**
     * A finished generation.
     *
     * @param tokens the prompt and everything after it, spliced results included
     * @param text the decoded tokens
     * @param modelWritten {@code true} for every token the model wrote, {@code false} for
     *     prompt and spliced tool output
     * @param calls how many tool calls ran
     */
    public record Trace(int[] tokens, String text, boolean[] modelWritten, int calls) {
    }

    private final Tokenizer tokenizer;
    private final Tool tool;
    private final int open;
    private final int resultMarker;
    private final int close;

    private ToolRuntime(Tokenizer tokenizer, Tool tool, char open, char resultMarker, char close) {
        this.tokenizer = Objects.requireNonNull(tokenizer);
        this.tool = Objects.requireNonNull(tool);
        this.open = tokenizer.encode(String.valueOf(open))[0];
        this.resultMarker = tokenizer.encode(String.valueOf(resultMarker))[0];
        this.close = tokenizer.encode(String.valueOf(close))[0];
    }

    /**
     * @param tokenizer the model's tokenizer (one token per bracket character)
     * @param tool the tool
     * @param open the character that opens a call
     * @param resultMarker the character after which the result is spliced in
     * @param close the character that closes a call (spliced by the runtime)
     * @return the runtime
     */
    public static ToolRuntime of(Tokenizer tokenizer, Tool tool, char open, char resultMarker, char close) {
        return new ToolRuntime(tokenizer, tool, open, resultMarker, close);
    }

    /**
     * Generates with tool calls.
     *
     * @param lm the model
     * @param prompt the prompt
     * @param maxNew the most tokens the model may write
     * @param end the end token
     * @param temperature the sampling temperature, 0 for greedy
     * @param random the random source
     * @return the trace
     */
    public Trace generate(WindowLm lm, int[] prompt, int maxNew, int end, double temperature,
            SplittableRandom random) {
        List<Integer> seq = new ArrayList<>();
        List<Boolean> written = new ArrayList<>();
        for (int t : prompt) {
            seq.add(t);
            written.add(false);
        }
        int callStart = -1;
        int calls = 0;
        for (int i = 0; i < maxNew; i++) {
            int[] cur = seq.stream().mapToInt(Integer::intValue).toArray();
            int tok = LlmTensors.sample(lm.logits(cur), temperature, random);
            seq.add(tok);
            written.add(true);
            if (tok == open) {
                callStart = seq.size();
            } else if (tok == resultMarker && callStart >= 0) {
                int[] args = seq.subList(callStart, seq.size() - 1).stream().mapToInt(Integer::intValue).toArray();
                String result = tool.call(tokenizer.decode(args));
                for (int r : tokenizer.encode(result)) {
                    seq.add(r);
                    written.add(false);
                }
                seq.add(close);
                written.add(false);
                callStart = -1;
                calls++;
            } else if (tok == end) {
                break;
            }
        }
        int[] tokens = seq.stream().mapToInt(Integer::intValue).toArray();
        boolean[] w = new boolean[written.size()];
        for (int i = 0; i < w.length; i++) {
            w[i] = written.get(i);
        }
        return new Trace(tokens, tokenizer.decode(tokens), w, calls);
    }

    /**
     * Splits a tool-augmented training sequence into (context, model-written chunk) pairs so
     * that supervised training never learns to predict tool output, only to use it.
     *
     * @param prompt the prompt
     * @param completion the completion with calls and results spliced in
     * @return pairs {context, chunk}
     */
    public List<int[][]> trainingChunks(int[] prompt, int[] completion) {
        List<int[][]> out = new ArrayList<>();
        int[] all = Arrays.copyOf(prompt, prompt.length + completion.length);
        System.arraycopy(completion, 0, all, prompt.length, completion.length);
        int start = prompt.length;
        int i = prompt.length;
        while (i < all.length) {
            if (all[i] == resultMarker) {
                out.add(new int[][] {Arrays.copyOf(all, start), Arrays.copyOfRange(all, start, i + 1)});
                int j = i + 1;
                while (j < all.length && all[j] != close) {
                    j++;
                }
                start = Math.min(all.length, j + 1);
                i = start;
            } else {
                i++;
            }
        }
        if (start < all.length) {
            out.add(new int[][] {Arrays.copyOf(all, start), Arrays.copyOfRange(all, start, all.length)});
        }
        return out;
    }

    /**
     * Evaluates integer arithmetic with {@code + - *}.
     *
     * @param expression the expression
     * @return the value, or {@code ?} if it does not parse
     */
    public static String calculate(String expression) {
        try {
            long total = 0;
            long term = 1;
            int sign = 1;
            int i = 0;
            String s = expression.trim();
            if (s.isEmpty()) {
                return "?";
            }
            while (i < s.length()) {
                int j = i;
                while (j < s.length() && Character.isDigit(s.charAt(j))) {
                    j++;
                }
                if (j == i) {
                    return "?";
                }
                term *= Long.parseLong(s.substring(i, j));
                if (j == s.length()) {
                    total += sign * term;
                    break;
                }
                char op = s.charAt(j);
                switch (op) {
                    case '*' -> { }
                    case '+', '-' -> {
                        total += sign * term;
                        term = 1;
                        sign = op == '+' ? 1 : -1;
                    }
                    default -> {
                        return "?";
                    }
                }
                i = j + 1;
                if (i == s.length()) {
                    return "?";
                }
            }
            return Long.toString(total);
        } catch (NumberFormatException e) {
            return "?";
        }
    }
}
