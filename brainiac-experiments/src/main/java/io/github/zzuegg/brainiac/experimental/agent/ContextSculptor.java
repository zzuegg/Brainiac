package io.github.zzuegg.brainiac.experimental.agent;

import io.github.zzuegg.brainiac.data.Tokenizer;
import io.github.zzuegg.brainiac.experimental.llm.ToolRuntime;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.UnaryOperator;

/**
 * Sculptor: active context management by tool calls (Li et al. 2025,
 * <a href="https://arxiv.org/abs/2508.04664">arXiv:2508.04664</a>). Instead of paging the context
 * (MemGPT) or overwriting a memory (MemAgent), the model is given tools over its own working
 * context: split it into labelled fragments, hide and restore fragments, summarise a fragment in
 * place (restorable), and search all fragments, hidden ones included. Hiding material that is
 * irrelevant or superseded early counters proactive interference from earlier content.
 *
 * <p>The context is a token sequence; what the model reads is {@link #visible()}, the visible
 * fragments in order. Every operation is also a text tool ({@link #tool}) for
 * {@link ToolRuntime}: {@code fragment n}, {@code hide l}, {@code restore l},
 * {@code summarise l}, {@code search text}. {@link #focus} is a scripted policy that issues such
 * calls, standing in for a model trained to use them.
 */
public final class ContextSculptor {

    private final List<int[]> original = new ArrayList<>();
    private final List<int[]> current = new ArrayList<>();
    private final List<Boolean> hidden = new ArrayList<>();
    private long calls;

    private ContextSculptor(int[] context) {
        original.add(context.clone());
        current.add(context.clone());
        hidden.add(false);
    }

    /**
     * @param context the whole context, one fragment to start with
     * @return the sculptor
     */
    public static ContextSculptor of(int[] context) {
        return new ContextSculptor(Objects.requireNonNull(context));
    }

    /**
     * Re-splits the original context into fragments of at most {@code tokens} tokens, all
     * visible and unsummarised, labelled 0, 1, ... in order.
     *
     * @param tokens the fragment size
     * @return the number of fragments
     */
    public int fragment(int tokens) {
        if (tokens < 1) {
            throw new IllegalArgumentException("tokens must be positive");
        }
        calls++;
        int[] all = original.stream().flatMapToInt(Arrays::stream).toArray();
        original.clear();
        current.clear();
        hidden.clear();
        for (int s = 0; s < all.length; s += tokens) {
            int[] f = Arrays.copyOfRange(all, s, Math.min(all.length, s + tokens));
            original.add(f);
            current.add(f.clone());
            hidden.add(false);
        }
        return original.size();
    }

    /** @return the number of fragments */
    public int fragments() {
        return original.size();
    }

    /**
     * @param label a fragment
     * @return whether it is hidden
     */
    public boolean isHidden(int label) {
        return hidden.get(label);
    }

    /**
     * Hides a fragment from the visible context.
     *
     * @param label the fragment
     */
    public void hide(int label) {
        calls++;
        hidden.set(label, true);
    }

    /**
     * Shows a fragment again with its original text (undoing a summary).
     *
     * @param label the fragment
     */
    public void restore(int label) {
        calls++;
        hidden.set(label, false);
        current.set(label, original.get(label).clone());
    }

    /**
     * Replaces a fragment's visible text by a summary of its original text.
     *
     * @param label the fragment
     * @param summariser original tokens to summary tokens
     */
    public void summarise(int label, UnaryOperator<int[]> summariser) {
        calls++;
        current.set(label, summariser.apply(original.get(label).clone()).clone());
    }

    /**
     * Searches every fragment, hidden or not, for a token pattern in its original text.
     *
     * @param pattern the tokens to find
     * @return the labels of the fragments that contain it, in order
     */
    public int[] search(int[] pattern) {
        calls++;
        List<Integer> hits = new ArrayList<>();
        for (int l = 0; l < original.size(); l++) {
            if (contains(original.get(l), pattern)) {
                hits.add(l);
            }
        }
        return hits.stream().mapToInt(Integer::intValue).toArray();
    }

    private static boolean contains(int[] text, int[] pattern) {
        outer:
        for (int i = 0; i + pattern.length <= text.length; i++) {
            for (int j = 0; j < pattern.length; j++) {
                if (text[i + j] != pattern[j]) {
                    continue outer;
                }
            }
            return true;
        }
        return pattern.length == 0;
    }

    /** @return the visible fragments' current text, concatenated in order */
    public int[] visible() {
        List<Integer> out = new ArrayList<>();
        for (int l = 0; l < current.size(); l++) {
            if (!hidden.get(l)) {
                for (int t : current.get(l)) {
                    out.add(t);
                }
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    /** @return the tool calls made so far */
    public long calls() {
        return calls;
    }

    /**
     * The operations as one text tool, for {@link ToolRuntime}: {@code fragment n},
     * {@code hide l}, {@code restore l}, {@code summarise l}, {@code search text}. Results are
     * {@code ok}, a fragment count, or space-separated labels; {@code ?} on errors.
     *
     * @param tokenizer decodes search patterns
     * @param summariser what {@code summarise} applies
     * @return the tool
     */
    public ToolRuntime.Tool tool(Tokenizer tokenizer, UnaryOperator<int[]> summariser) {
        return arguments -> {
            String[] parts = arguments.trim().split(" ", 2);
            try {
                return switch (parts[0]) {
                    case "fragment" -> Integer.toString(fragment(Integer.parseInt(parts[1].trim())));
                    case "hide" -> {
                        hide(Integer.parseInt(parts[1].trim()));
                        yield "ok";
                    }
                    case "restore" -> {
                        restore(Integer.parseInt(parts[1].trim()));
                        yield "ok";
                    }
                    case "summarise" -> {
                        summarise(Integer.parseInt(parts[1].trim()), summariser);
                        yield "ok";
                    }
                    case "search" -> {
                        StringBuilder sb = new StringBuilder();
                        for (int l : search(tokenizer.encode(parts[1]))) {
                            sb.append(sb.isEmpty() ? "" : " ").append(l);
                        }
                        yield sb.toString();
                    }
                    default -> "?";
                };
            } catch (RuntimeException e) {
                return "?";
            }
        };
    }

    /**
     * A scripted focusing policy, issued as tool calls: fragment the context, search for the
     * query's key, hide every fragment, restore the {@code keepLatest} most recent matches
     * (older matches are superseded and stay hidden), and optionally summarise them.
     *
     * @param tool this sculptor's tool ({@link #tool})
     * @param key the key's text
     * @param fragmentTokens the fragment size
     * @param keepLatest how many of the latest matching fragments to show
     * @param summarise whether to summarise the shown fragments
     * @return the tool calls issued, with their results
     */
    public List<String> focus(ToolRuntime.Tool tool, String key, int fragmentTokens, int keepLatest,
            boolean summarise) {
        List<String> transcript = new ArrayList<>();
        int n = Integer.parseInt(run(tool, "fragment " + fragmentTokens, transcript));
        String found = run(tool, "search " + key, transcript);
        for (int l = 0; l < n; l++) {
            run(tool, "hide " + l, transcript);
        }
        String[] hits = found.isEmpty() ? new String[0] : found.split(" ");
        for (int i = Math.max(0, hits.length - keepLatest); i < hits.length; i++) {
            run(tool, "restore " + hits[i], transcript);
            if (summarise) {
                run(tool, "summarise " + hits[i], transcript);
            }
        }
        return transcript;
    }

    private static String run(ToolRuntime.Tool tool, String call, List<String> transcript) {
        String result = tool.call(call);
        transcript.add(call + " > " + result);
        return result;
    }

    @Override
    public String toString() {
        return "ContextSculptor(" + original.size() + " fragments, " + visible().length + " visible tokens)";
    }
}
