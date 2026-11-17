package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.io.PretrainedTokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Data for instruction tuning (supervised fine-tuning): conversations rendered with the
 * model's chat template, with only the assistant's replies counting toward the loss, as
 * TRL's {@code assistant_only_loss} does. Examples are then {@linkplain #pad padded} into
 * rows of equal length or {@linkplain #pack packed} several to a row, and trained with
 * {@link Llama#loss(Batch)}:
 *
 * <pre>
 * List&lt;InstructionData.Example&gt; examples = conversations.stream()
 *         .map(c -&gt; InstructionData.encode(tokenizer, c)).toList();
 * for (InstructionData.Batch b : InstructionData.pack(examples, 4, 2048)) {
 *     Var loss = model.loss(b);
 *     ...
 * }
 * </pre>
 */
public final class InstructionData {

    private InstructionData() {
    }

    /**
     * One tokenized example.
     *
     * @param tokens the token ids
     * @param train for each token, whether the model learns to predict it
     */
    public record Example(int[] tokens, boolean[] train) {

        /** Checks that both have the same length. */
        public Example {
            if (tokens.length != train.length) {
                throw new IllegalArgumentException(tokens.length + " tokens but " + train.length + " flags");
            }
        }

        /** @return how many tokens count toward the loss */
        public int trained() {
            int n = 0;
            for (boolean t : train) {
                n += t ? 1 : 0;
            }
            return n;
        }

        /** The first {@code n} tokens. */
        Example cut(int n) {
            return n >= tokens.length ? this : new Example(Arrays.copyOf(tokens, n), Arrays.copyOf(train, n));
        }
    }

    /**
     * Rows ready for {@link Llama#loss(Batch)}.
     *
     * @param tokens {@code [rows][steps]} token ids, padding 0
     * @param targets {@code [rows][steps]} the token to predict after each step, or -1 where
     *     nothing is learned
     * @param segments {@code [rows][steps]} which example of its row each step belongs to,
     *     from 1, and 0 for padding
     */
    public record Batch(int[][] tokens, int[][] targets, int[][] segments) {
    }

    /**
     * Renders a conversation with the tokenizer's chat template and marks the assistant's
     * replies, each with the end of its turn, for training. Each part is tokenized on its own,
     * so the prompt before a reply has exactly the tokens it has when generating.
     *
     * @param tokenizer the model's tokenizer, with a chat template
     * @param messages the conversation: maps with {@code role} and {@code content}
     * @return the example
     */
    public static Example encode(PretrainedTokenizer tokenizer, List<? extends Map<String, ?>> messages) {
        return encode(tokenizer, messages, Map.of());
    }

    /**
     * As {@link #encode(PretrainedTokenizer, List)}, with more template variables, such as
     * {@code tools}.
     *
     * @param tokenizer the model's tokenizer, with a chat template
     * @param messages the conversation
     * @param variables more variables for the template
     * @return the example
     */
    public static Example encode(PretrainedTokenizer tokenizer, List<? extends Map<String, ?>> messages,
            Map<String, ?> variables) {
        Objects.requireNonNull(tokenizer, "tokenizer");
        List<int[]> parts = new ArrayList<>();
        List<Boolean> trained = new ArrayList<>();
        String done = "";
        for (int i = 0; i < messages.size(); i++) {
            if (!"assistant".equals(messages.get(i).get("role"))) {
                continue;
            }
            String prompt = tokenizer.applyChatTemplate(messages.subList(0, i), true, variables);
            String full = tokenizer.applyChatTemplate(messages.subList(0, i + 1), false, variables);
            if (!prompt.startsWith(done) || !full.startsWith(prompt)) {
                throw new IllegalArgumentException("the chat template does not render message " + i
                        + " after what came before it, so the reply cannot be marked");
            }
            add(tokenizer, parts, trained, prompt.substring(done.length()), false);
            add(tokenizer, parts, trained, full.substring(prompt.length()), true);
            done = full;
        }
        String all = tokenizer.applyChatTemplate(messages, false, variables);
        if (all.startsWith(done)) {
            add(tokenizer, parts, trained, all.substring(done.length()), false);
        }
        return join(parts, trained);
    }

    /**
     * A plain prompt and answer, without a template: only the answer is learned, followed by
     * the end-of-sequence token when the tokenizer has one.
     *
     * @param tokenizer the tokenizer
     * @param prompt the prompt
     * @param answer the answer
     * @return the example
     */
    public static Example completion(PretrainedTokenizer tokenizer, String prompt, String answer) {
        List<int[]> parts = new ArrayList<>();
        List<Boolean> trained = new ArrayList<>();
        add(tokenizer, parts, trained, prompt, false);
        add(tokenizer, parts, trained, answer, true);
        if (tokenizer.eos().isPresent()) {
            parts.add(new int[] {tokenizer.eos().getAsInt()});
            trained.add(true);
        }
        return join(parts, trained);
    }

    private static void add(PretrainedTokenizer tok, List<int[]> parts, List<Boolean> trained, String text,
            boolean train) {
        if (text.isEmpty()) {
            return;
        }
        // the special tokens a template writes are in the text; only the first part may get a BOS
        int[] ids = tok.encode(text, parts.isEmpty());
        if (parts.isEmpty() && tok.bos().isPresent() && ids.length > 1 && ids[0] == tok.bos().getAsInt()
                && ids[1] == ids[0]) {
            ids = Arrays.copyOfRange(ids, 1, ids.length);   // the template wrote one already
        }
        if (parts.isEmpty() && tok.eos().isPresent() && ids.length > 1 && ids[ids.length - 1] == tok.eos().getAsInt()
                && !text.endsWith(tok.token(tok.eos().getAsInt()))) {
            ids = Arrays.copyOf(ids, ids.length - 1);   // added, not written
        }
        parts.add(ids);
        trained.add(train);
    }

    private static Example join(List<int[]> parts, List<Boolean> trained) {
        int n = 0;
        for (int[] p : parts) {
            n += p.length;
        }
        int[] tokens = new int[n];
        boolean[] train = new boolean[n];
        int at = 0;
        for (int i = 0; i < parts.size(); i++) {
            int[] p = parts.get(i);
            System.arraycopy(p, 0, tokens, at, p.length);
            Arrays.fill(train, at, at + p.length, trained.get(i));
            at += p.length;
        }
        return new Example(tokens, train);
    }

    /**
     * One example per row, in order, padded at the end to the longest of each batch. Longer
     * examples are cut to {@code maxLength}.
     *
     * @param examples the examples
     * @param rows the examples per batch
     * @param maxLength the most tokens per row
     * @return the batches
     */
    public static List<Batch> pad(List<Example> examples, int rows, int maxLength) {
        check(rows, maxLength);
        List<Batch> out = new ArrayList<>();
        for (int from = 0; from < examples.size(); from += rows) {
            List<List<Example>> batch = new ArrayList<>();
            int longest = 1;
            for (Example e : examples.subList(from, Math.min(examples.size(), from + rows))) {
                Example cut = e.cut(maxLength);
                batch.add(List.of(cut));
                longest = Math.max(longest, cut.tokens.length);
            }
            out.add(batch(batch, longest));
        }
        return out;
    }

    /**
     * Several examples per row, filled in order until the next does not fit, so rows of
     * {@code length} tokens carry little padding. Each example sees only itself (see
     * {@link Llama#loss(Batch)}). Examples longer than a row are cut.
     *
     * @param examples the examples
     * @param rows the rows per batch
     * @param length the tokens per row
     * @return the batches
     */
    public static List<Batch> pack(List<Example> examples, int rows, int length) {
        check(rows, length);
        List<List<Example>> filled = new ArrayList<>();
        List<Example> row = new ArrayList<>();
        int used = 0;
        for (Example e : examples) {
            Example cut = e.cut(length);
            if (used + cut.tokens.length > length && !row.isEmpty()) {
                filled.add(row);
                row = new ArrayList<>();
                used = 0;
            }
            row.add(cut);
            used += cut.tokens.length;
        }
        if (!row.isEmpty()) {
            filled.add(row);
        }
        List<Batch> out = new ArrayList<>();
        for (int from = 0; from < filled.size(); from += rows) {
            out.add(batch(filled.subList(from, Math.min(filled.size(), from + rows)), length));
        }
        return out;
    }

    private static void check(int rows, int length) {
        if (rows < 1 || length < 2) {
            throw new IllegalArgumentException("need at least 1 row of 2 tokens, got " + rows + " of " + length);
        }
    }

    private static Batch batch(List<List<Example>> rows, int steps) {
        int[][] tokens = new int[rows.size()][steps];
        int[][] targets = new int[rows.size()][steps];
        int[][] segments = new int[rows.size()][steps];
        for (int b = 0; b < rows.size(); b++) {
            Arrays.fill(targets[b], -1);
            int at = 0;
            int segment = 0;
            for (Example e : rows.get(b)) {
                segment++;
                int n = e.tokens.length;
                System.arraycopy(e.tokens, 0, tokens[b], at, n);
                Arrays.fill(segments[b], at, at + n, segment);
                for (int t = 0; t + 1 < n; t++) {
                    if (e.train[t + 1]) {
                        targets[b][at + t] = e.tokens[t + 1];
                    }
                }
                at += n;
            }
        }
        return new Batch(tokens, targets, segments);
    }
}
