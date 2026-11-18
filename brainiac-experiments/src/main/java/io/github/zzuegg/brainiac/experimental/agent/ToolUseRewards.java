package io.github.zzuegg.brainiac.experimental.agent;

import java.util.ArrayList;
import java.util.List;

/**
 * Rewards for reinforcement learning of tool use. ToolRL (Qian et al. 2025,
 * <a href="https://arxiv.org/abs/2504.13958">arXiv:2504.13958</a>) decomposes the reward into a
 * format term (calls well formed, an answer at the end) and a correctness term that compares
 * each call's tool and arguments with the ground truth; ReTool (Feng et al. 2025,
 * <a href="https://arxiv.org/abs/2504.11536">arXiv:2504.11536</a>) uses the final outcome only and
 * lets the model discover when a call pays. Both are weighted sums of the same parts here, plus
 * an optional cost per call against needless calls.
 *
 * <p>A completion is text in the {@code ToolRuntime} format: calls {@code [arguments>result]}
 * anywhere, then the answer, then the end character.
 */
public final class ToolUseRewards {

    /**
     * A parsed completion.
     *
     * @param calls the arguments of every call, in order
     * @param answer the text after the last call and before the end character
     * @param wellFormed whether every call is closed and the completion ends with the end character
     */
    public record Parsed(List<String> calls, String answer, boolean wellFormed) {
    }

    private final char open;
    private final char marker;
    private final char close;
    private final char end;
    private final float format;
    private final float call;
    private final float outcome;
    private final float callCost;

    private ToolUseRewards(char open, char marker, char close, char end, float format, float call, float outcome,
            float callCost) {
        this.open = open;
        this.marker = marker;
        this.close = close;
        this.end = end;
        this.format = format;
        this.call = call;
        this.outcome = outcome;
        this.callCost = callCost;
    }

    /**
     * ToolRL's structured reward.
     *
     * @param open the call's opening character
     * @param marker the result marker
     * @param close the call's closing character
     * @param end the end character
     * @param callCost the cost per call
     * @return format 0.2, call correctness 0.4, outcome 1, minus {@code callCost} per call
     */
    public static ToolUseRewards toolRl(char open, char marker, char close, char end, float callCost) {
        return new ToolUseRewards(open, marker, close, end, 0.2f, 0.4f, 1f, callCost);
    }

    /**
     * ReTool's outcome-only reward.
     *
     * @param open the call's opening character
     * @param marker the result marker
     * @param close the call's closing character
     * @param end the end character
     * @param callCost the cost per call
     * @return outcome only, minus {@code callCost} per call
     */
    public static ToolUseRewards outcomeOnly(char open, char marker, char close, char end, float callCost) {
        return new ToolUseRewards(open, marker, close, end, 0, 0, 1f, callCost);
    }

    /**
     * @param completion the completion's text
     * @return its calls and answer
     */
    public Parsed parse(String completion) {
        List<String> calls = new ArrayList<>();
        boolean ok = true;
        int i = 0;
        int answerStart = 0;
        while (i < completion.length()) {
            char c = completion.charAt(i);
            if (c == open) {
                int m = completion.indexOf(marker, i + 1);
                int e = m < 0 ? -1 : completion.indexOf(close, m + 1);
                if (e < 0) {
                    ok = false;
                    calls.add(completion.substring(i + 1, m < 0 ? completion.length() : m));
                    answerStart = completion.length();
                    break;
                }
                calls.add(completion.substring(i + 1, m));
                i = e + 1;
                answerStart = i;
            } else if (c == end) {
                break;
            } else {
                i++;
            }
        }
        int stop = completion.indexOf(end, answerStart);
        ok &= stop >= 0 && stop == completion.length() - 1;
        String answer = completion.substring(Math.min(answerStart, completion.length()),
                stop < 0 ? completion.length() : stop);
        return new Parsed(calls, answer, ok);
    }

    /**
     * Scores a completion.
     *
     * @param completion the completion's text
     * @param expectedAnswer the right answer
     * @param expectedCall the right call's arguments, or {@code null} when no call is needed
     * @return the reward
     */
    public float score(String completion, String expectedAnswer, String expectedCall) {
        Parsed p = parse(completion);
        float r = p.wellFormed() ? format : 0;
        if (expectedCall == null) {
            r += p.calls().isEmpty() ? call : 0;
        } else if (!p.calls().isEmpty()) {
            r += p.calls().get(0).equals(expectedCall) ? call : call / 4;
        }
        r += p.wellFormed() && p.answer().equals(expectedAnswer) ? outcome : 0;
        return r - callCost * p.calls().size();
    }

    @Override
    public String toString() {
        return "ToolUseRewards(format " + format + ", call " + call + ", outcome " + outcome + ", cost " + callCost
                + ")";
    }
}
