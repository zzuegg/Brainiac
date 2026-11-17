package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.data.Tokenizer;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.TreeSet;

/**
 * Writes text with a {@link LanguageModel}: sampling, stop tokens and strings, streaming,
 * batches of prompts and beam search.
 *
 * <pre>{@code
 * Llama model = Llama.load(Path.of("models/qwen2.5-0.5b"), Device.best(), DType.FLOAT16);
 * PretrainedTokenizer tok = PretrainedTokenizer.load(Path.of("models/qwen2.5-0.5b"));
 * Generator gen = new Generator(model, tok);
 * String prompt = tok.applyChatTemplate(List.of(Map.of("role", "user", "content", "Hi!")), true);
 * gen.generate(prompt, Generator.Options.sampled(200, Sampling.temperature(0.7f).withTopP(0.9f), 42),
 *         (row, text) -> System.out.print(text));
 * }</pre>
 */
public final class Generator {

    private final LanguageModel model;
    private final Tokenizer tokenizer;

    /**
     * @param model the model
     * @param tokenizer its tokenizer; {@code null} when only token ids are used
     */
    public Generator(LanguageModel model, Tokenizer tokenizer) {
        this.model = Objects.requireNonNull(model, "model");
        this.tokenizer = tokenizer;
    }

    /**
     * What to generate.
     *
     * @param maxNewTokens the most tokens to add per sequence
     * @param sampling how to pick tokens
     * @param seed seeds the random draws, so the same call gives the same text
     * @param stopTokens tokens that end a sequence (not part of the output); {@code null} for
     *     the tokenizer's end-of-sequence token
     * @param stopStrings strings that end a sequence (cut from the output)
     * @param constraint a language the new text must belong to, or {@code null}: only tokens
     *     that keep it a prefix of an accepted text are sampled, a stop token only once it is
     *     accepted, and a sequence that cannot go on ends there (needs a tokenizer)
     */
    public record Options(int maxNewTokens, Sampling sampling, long seed, Set<Integer> stopTokens,
            List<String> stopStrings, Constraint constraint) {

        /**
         * Options without a constraint.
         *
         * @param maxNewTokens the most tokens to add
         * @param sampling how to pick tokens
         * @param seed seeds the draws
         * @param stopTokens stop tokens
         * @param stopStrings stop strings
         */
        public Options(int maxNewTokens, Sampling sampling, long seed, Set<Integer> stopTokens,
                List<String> stopStrings) {
            this(maxNewTokens, sampling, seed, stopTokens, stopStrings, null);
        }

        /**
         * Checks the options.
         *
         * @param maxNewTokens the most tokens to add
         * @param sampling how to pick tokens
         * @param seed seeds the draws
         * @param stopTokens stop tokens
         * @param stopStrings stop strings
         * @param constraint the constraint, or {@code null}
         */
        public Options {
            if (maxNewTokens < 0) {
                throw new IllegalArgumentException("maxNewTokens < 0");
            }
            Objects.requireNonNull(sampling, "sampling");
            stopTokens = stopTokens == null ? null : Set.copyOf(stopTokens);
            stopStrings = stopStrings == null ? List.of() : List.copyOf(stopStrings);
        }

        /**
         * @param maxNewTokens the most tokens to add
         * @return greedy decoding up to the end-of-sequence token
         */
        public static Options greedy(int maxNewTokens) {
            return new Options(maxNewTokens, Sampling.GREEDY, 0, null, List.of());
        }

        /**
         * @param maxNewTokens the most tokens to add
         * @param sampling how to pick tokens
         * @param seed seeds the draws
         * @return sampling up to the end-of-sequence token
         */
        public static Options sampled(int maxNewTokens, Sampling sampling, long seed) {
            return new Options(maxNewTokens, sampling, seed, null, List.of());
        }

        /**
         * @param tokens the stop tokens
         * @return a copy that stops on them
         */
        public Options withStopTokens(Set<Integer> tokens) {
            return new Options(maxNewTokens, sampling, seed, tokens, stopStrings, constraint);
        }

        /**
         * @param strings the stop strings
         * @return a copy that stops on them
         */
        public Options withStopStrings(String... strings) {
            return new Options(maxNewTokens, sampling, seed, stopTokens, List.of(strings), constraint);
        }

        /**
         * @param c the language the new text must belong to
         * @return a copy constrained to it
         */
        public Options withConstraint(Constraint c) {
            return new Options(maxNewTokens, sampling, seed, stopTokens, stopStrings, c);
        }
    }

    /** Receives text as it is written. */
    @FunctionalInterface
    public interface Listener {

        /**
         * Called with each new piece of a sequence's text: never half a character, and never
         * the start of a stop string that might still complete.
         *
         * @param row which prompt the text continues
         * @param text the new text
         */
        void text(int row, String text);
    }

    /** Why a sequence ended. */
    public enum Finish {
        /** a stop token or stop string */
        STOP,
        /** {@link Options#maxNewTokens} */
        LENGTH
    }

    /**
     * One generated sequence.
     *
     * @param tokens the new tokens, without the prompt and without a stop token
     * @param text their text (empty without a tokenizer), cut before a stop string
     * @param finish why it ended
     */
    public record Output(int[] tokens, String text, Finish finish) {
    }

    /** End-of-turn tokens of common chat models. */
    private static final List<String> END_OF_TURN = List.of("<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>",
            "<|endoftext|>", "<|end_of_text|>", "</s>");

    /**
     * The tokens that end a pretrained model's answer: its {@code generation_config.json}'s
     * {@code eos_token_id} (one or a list), the tokenizer's end-of-sequence token, and the
     * end-of-turn tokens of common chat formats that its vocabulary has.
     *
     * @param model the model directory or {@code .gguf} file
     * @param tokenizer its tokenizer
     * @return the stop tokens
     */
    public static Set<Integer> stopTokens(Path model, PretrainedTokenizer tokenizer) {
        Set<Integer> stops = new TreeSet<>();
        Path config = model.resolve("generation_config.json");
        if (Files.isDirectory(model) && Files.isRegularFile(config)) {
            try {
                Object eos = Json.object(Json.read(config)).get("eos_token_id");
                if (eos instanceof Number n) {
                    stops.add(n.intValue());
                } else if (eos instanceof List<?> list) {
                    for (Object o : list) {
                        stops.add(((Number) o).intValue());
                    }
                }
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
        tokenizer.eos().ifPresent(stops::add);
        for (String t : END_OF_TURN) {
            tokenizer.id(t).ifPresent(stops::add);
        }
        return stops;
    }

    /**
     * Continues a prompt.
     *
     * @param prompt the text to continue (special tokens such as BOS are added as the tokenizer
     *     says)
     * @param options what to generate
     * @return the new text
     */
    public String generate(String prompt, Options options) {
        return generate(prompt, options, null);
    }

    /**
     * Continues a prompt, streaming the text.
     *
     * @param prompt the text to continue
     * @param options what to generate
     * @param listener receives the text as it is written; may be {@code null}
     * @return the new text
     */
    public String generate(String prompt, Options options, Listener listener) {
        return generate(List.of(prompt), options, listener).getFirst().text();
    }

    /**
     * Continues several prompts side by side.
     *
     * @param prompts the texts to continue
     * @param options what to generate
     * @param listener receives the text as it is written; may be {@code null}
     * @return one output per prompt
     */
    public List<Output> generate(List<String> prompts, Options options, Listener listener) {
        int[][] ids = new int[prompts.size()][];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = encode(prompts.get(i));
        }
        return generate(ids, options, listener);
    }

    /**
     * Continues prompts given as token ids. Prompts of different lengths are padded on the
     * left.
     *
     * @param prompts the token ids to continue, at least one each
     * @param options what to generate
     * @param listener receives the text as it is written; may be {@code null}
     * @return one output per prompt
     */
    public List<Output> generate(int[][] prompts, Options options, Listener listener) {
        int batch = prompts.length;
        if (batch == 0) {
            return List.of();
        }
        Set<Integer> stops = stopTokens(options);
        int longest = 0;
        for (int[] p : prompts) {
            if (p.length == 0) {
                throw new IllegalArgumentException("empty prompt");
            }
            longest = Math.max(longest, p.length);
        }
        int[][] first = new int[batch][longest];
        boolean[][] pad = new boolean[batch][longest];
        boolean padded = false;
        for (int b = 0; b < batch; b++) {
            int off = longest - prompts[b].length;
            System.arraycopy(prompts[b], 0, first[b], off, prompts[b].length);
            for (int t = 0; t < off; t++) {
                pad[b][t] = true;
                padded = true;
            }
        }
        Row[] rows = new Row[batch];
        for (int b = 0; b < batch; b++) {
            rows[b] = new Row(b, prompts[b], options, listener);
        }
        SplittableRandom random = new SplittableRandom(options.seed());
        LanguageModel.State state = model.newState(batch);
        int active = batch;
        if (options.maxNewTokens() > 0) {
            Tensor logits = model.next(state, first, padded ? pad : null);
            // plain greedy decoding reads back only the chosen tokens, which the device finds
            Sampling sampling = options.sampling();
            boolean onDevice = options.constraint() == null && sampling.temperature() == 0
                    && sampling.repetitionPenalty() == 1;
            for (int step = 0; active > 0; step++) {
                float[] all = onDevice ? logits.argmax(-1).toArray() : logits.toArray();
                int vocab = all.length / batch;
                int[][] next = new int[batch][1];
                boolean[][] skip = new boolean[batch][1];
                for (int b = 0; b < batch; b++) {
                    Row r = rows[b];
                    if (r.finish != null) {
                        skip[b][0] = true;
                        continue;
                    }
                    if (onDevice) {
                        next[b][0] = (int) all[b];
                        if (r.add(next[b][0], stops)) {
                            active--;
                            skip[b][0] = true;
                        }
                        continue;
                    }
                    float[] row = Arrays.copyOfRange(all, b * vocab, (b + 1) * vocab);
                    if (r.matcher != null && !constrain(row, r.matcher, stops)) {
                        r.end();   // accepted and nothing can follow, but no stop token to say so
                        active--;
                        skip[b][0] = true;
                        continue;
                    }
                    int token = options.sampling().sample(row, r.seen(), random);
                    next[b][0] = token;
                    if (r.matcher != null && !stops.contains(token)) {
                        r.matcher = vocabulary(vocab).advance(r.matcher, token);
                    }
                    if (r.add(token, stops)) {
                        active--;
                        skip[b][0] = true;
                    }
                }
                if (active == 0 || step + 1 >= options.maxNewTokens()) {
                    break;
                }
                logits = model.next(state, next, active < batch ? skip : null);
            }
        }
        state.close();
        List<Output> out = new ArrayList<>(batch);
        for (Row r : rows) {
            out.add(r.finish());
        }
        return out;
    }

    /**
     * Turns a prompt into token ids as the model expects them: with the special tokens its
     * tokenizer adds at the start (such as Llama's begin token), but not twice when a chat
     * template already wrote it, and without an end-of-sequence token at the end.
     *
     * @param prompt the text
     * @return the token ids
     */
    public int[] encode(String prompt) {
        if (!(tokenizer() instanceof PretrainedTokenizer p)) {
            return tokenizer.encode(prompt);
        }
        int[] ids = p.encode(prompt, true);
        int from = 0;
        int to = ids.length;
        if (p.bos().isPresent() && to - from > 1 && ids[0] == p.bos().getAsInt() && ids[1] == ids[0]) {
            from = 1;
        }
        if (p.eos().isPresent() && to - from > 1 && ids[to - 1] == p.eos().getAsInt()
                && !prompt.endsWith(p.token(p.eos().getAsInt()))) {
            to--;
        }
        return from == 0 && to == ids.length ? ids : Arrays.copyOfRange(ids, from, to);
    }

    private Tokenizer tokenizer() {
        if (tokenizer == null) {
            throw new IllegalStateException("this generator has no tokenizer");
        }
        return tokenizer;
    }

    private Set<Integer> stopTokens(Options options) {
        if (options.stopTokens() != null) {
            return options.stopTokens();
        }
        if (tokenizer instanceof PretrainedTokenizer p && p.eos().isPresent()) {
            return Set.of(p.eos().getAsInt());
        }
        return Set.of();
    }

    private String decode(int[] ids) {
        if (tokenizer == null) {
            return "";
        }
        return tokenizer instanceof PretrainedTokenizer p ? p.decode(ids, true) : tokenizer.decode(ids);
    }

    /** One sequence being written. */
    private final class Row {
        final int index;
        final int[] prompt;
        final Options options;
        final Listener listener;
        int[] tokens = new int[16];
        int count;
        String text = "";
        int emitted;
        Finish finish;

        /** Where the text stands in the options' constraint, or {@code null} without one. */
        Constraint.Matcher matcher;

        Row(int index, int[] prompt, Options options, Listener listener) {
            this.index = index;
            this.prompt = prompt;
            this.options = options;
            this.listener = listener;
            if (options.constraint() != null) {
                if (tokenizer == null) {
                    throw new IllegalStateException("constrained decoding needs a tokenizer");
                }
                matcher = options.constraint().start();
            }
        }

        /** Ends the sequence as if a stop token came. */
        void end() {
            finish = Finish.STOP;
            text = decode(Arrays.copyOf(tokens, count));
            flush(text.length());
        }

        int[] seen() {
            if (options.sampling().repetitionPenalty() == 1) {
                return prompt;
            }
            int[] s = Arrays.copyOf(prompt, prompt.length + count);
            System.arraycopy(tokens, 0, s, prompt.length, count);
            return s;
        }

        /** @return whether the sequence ended */
        boolean add(int token, Set<Integer> stops) {
            if (stops.contains(token)) {
                finish = Finish.STOP;
                flush(text.length());
                return true;
            }
            if (count == tokens.length) {
                tokens = Arrays.copyOf(tokens, count * 2);
            }
            tokens[count++] = token;
            if (tokenizer != null && (listener != null || !options.stopStrings().isEmpty())) {
                text = decode(Arrays.copyOf(tokens, count));
                int cut = firstStop();
                if (cut >= 0) {
                    text = text.substring(0, cut);
                    finish = Finish.STOP;
                    flush(text.length());
                    return true;
                }
                flush(safeEnd());
            }
            if (count >= options.maxNewTokens()) {
                finish = Finish.LENGTH;
                if (tokenizer != null) {
                    text = decode(Arrays.copyOf(tokens, count));
                }
                flush(text.length());
                return true;
            }
            return false;
        }

        int firstStop() {
            int cut = -1;
            for (String s : options.stopStrings()) {
                int i = text.indexOf(s, Math.max(0, emitted - s.length()));
                if (i >= 0 && (cut < 0 || i < cut)) {
                    cut = i;
                }
            }
            return cut;
        }

        /** Where the text can be shown up to: no half characters, no possible stop string. */
        int safeEnd() {
            int end = text.length();
            while (end > emitted && text.charAt(end - 1) == '�') {
                end--;
            }
            for (String s : options.stopStrings()) {
                for (int k = Math.min(s.length() - 1, end); k > 0; k--) {
                    if (text.startsWith(s.substring(0, k), end - k)) {
                        end = Math.min(end, end - k);
                        break;
                    }
                }
            }
            if (end > emitted && Character.isHighSurrogate(text.charAt(end - 1))) {
                end--;
            }
            return Math.max(end, emitted);
        }

        void flush(int end) {
            if (end > emitted && emitted <= text.length()) {
                if (listener != null) {
                    listener.text(index, text.substring(emitted, end));
                }
                emitted = end;
            }
        }

        Output finish() {
            int[] t = Arrays.copyOf(tokens, count);
            if (finish == null) {
                finish = Finish.LENGTH;
            }
            if (finish == Finish.LENGTH || options.stopStrings().isEmpty() && listener == null) {
                text = decode(t);
            }
            return new Output(t, text, finish);
        }
    }

    private Constraint.Vocabulary vocabulary;

    /** The token texts, made once per vocabulary size. */
    private synchronized Constraint.Vocabulary vocabulary(int size) {
        if (vocabulary == null || vocabulary.size() != size) {
            vocabulary = Constraint.Vocabulary.of(tokenizer, size);
        }
        return vocabulary;
    }

    /**
     * Masks the tokens a constraint does not allow.
     *
     * @return whether any token is left
     */
    private boolean constrain(float[] logits, Constraint.Matcher m, Set<Integer> stops) {
        boolean[] ok = vocabulary(logits.length).allowed(m);
        boolean accepted = m.accepting();
        boolean any = false;
        for (int t = 0; t < logits.length; t++) {
            if (ok[t] && !stops.contains(t) || accepted && stops.contains(t)) {
                any = true;
            } else {
                logits[t] = Float.NEGATIVE_INFINITY;
            }
        }
        return any;
    }

    // ---------------------------------------------------------------- continuous batching

    /**
     * Starts a background decoder that batches requests as they come: a new request is read
     * on its own and then {@linkplain LanguageModel.State#join joins} the sequences being
     * decoded, and a finished one leaves at once, so short answers never wait for long ones.
     * Each request keeps its own options and draws, so alone it writes what {@link #generate}
     * writes. The model's states must support joining.
     *
     * @param maxBatch the most sequences decoded side by side
     * @return the batcher; close it to stop
     */
    public Batcher batcher(int maxBatch) {
        return new Batcher(maxBatch);
    }

    /** Decodes submitted requests together; see {@link #batcher(int)}. */
    public final class Batcher implements AutoCloseable {

        private final int maxBatch;
        private final java.util.concurrent.LinkedBlockingQueue<Job> queue = new java.util.concurrent.LinkedBlockingQueue<>();
        private final Thread thread;
        private volatile boolean closed;
        private volatile int largest;

        private final class Job {
            final Row row;
            final Set<Integer> stops;
            final SplittableRandom random;
            final java.util.concurrent.CompletableFuture<Output> future = new java.util.concurrent.CompletableFuture<>();
            float[] logits;
            /** The next token when only it was read back (greedy rows), else -1. */
            int chosen = -1;

            Job(int[] prompt, Options options, Listener listener) {
                row = new Row(0, prompt, options, listener);
                stops = stopTokens(options);
                random = new SplittableRandom(options.seed());
            }
        }

        private Batcher(int maxBatch) {
            if (maxBatch < 1) {
                throw new IllegalArgumentException("maxBatch must be at least 1, got " + maxBatch);
            }
            this.maxBatch = maxBatch;
            thread = new Thread(this::loop, "brainiac-batcher");
            thread.setDaemon(true);
            thread.start();
        }

        /**
         * @param prompt the text to continue
         * @param options what to generate
         * @param listener receives the text as it is written (row 0), or {@code null}
         * @return the output once it is done
         */
        public java.util.concurrent.CompletableFuture<Output> submit(String prompt, Options options,
                Listener listener) {
            return submit(encode(prompt), options, listener);
        }

        /**
         * @param prompt the token ids to continue, at least one
         * @param options what to generate
         * @param listener receives the text as it is written (row 0), or {@code null}
         * @return the output once it is done
         */
        public java.util.concurrent.CompletableFuture<Output> submit(int[] prompt, Options options,
                Listener listener) {
            if (prompt.length == 0) {
                throw new IllegalArgumentException("empty prompt");
            }
            if (closed) {
                throw new IllegalStateException("the batcher is closed");
            }
            Job j = new Job(prompt.clone(), Objects.requireNonNull(options, "options"), listener);
            queue.add(j);
            return j.future;
        }

        /**
         * Decodes {@code maxBatch} copies of a prompt together and waits, so the first real
         * burst of requests does not pay for preparing the batch shapes (on the GPU, compiling
         * them) or for the JVM compiling the decoding loop. Copy {@code i} writes
         * {@code tokens * (i + 1) / maxBatch} tokens, so the copies finish one after another
         * and every batch size from {@code maxBatch} down to 1 is decoded. About 64 tokens
         * warm the loop well; the JVM keeps getting faster over the first few bursts.
         *
         * @param prompt a short prompt
         * @param tokens the tokens to write for the longest copy
         * @return this
         */
        public Batcher warmUp(int[] prompt, int tokens) {
            List<java.util.concurrent.CompletableFuture<Output>> all = new ArrayList<>();
            for (int i = 0; i < maxBatch; i++) {
                int n = Math.max(1, (int) ((long) tokens * (i + 1) / maxBatch));
                all.add(submit(prompt, Options.greedy(n).withStopTokens(Set.of()), null));
            }
            java.util.concurrent.CompletableFuture.allOf(all.toArray(java.util.concurrent.CompletableFuture[]::new))
                    .join();
            return this;
        }

        /** @return the most sequences decoded side by side so far */
        public int largestBatch() {
            return largest;
        }

        /** Stops decoding; unfinished requests fail. */
        @Override
        public void close() {
            closed = true;
            thread.interrupt();
            try {
                thread.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            for (Job j; (j = queue.poll()) != null;) {
                j.future.completeExceptionally(new java.util.concurrent.CancellationException("batcher closed"));
            }
        }

        private void loop() {
            List<Job> active = new ArrayList<>();
            LanguageModel.State state = null;
            try {
                while (!closed) {
                    List<Job> joining = new ArrayList<>();
                    if (active.isEmpty()) {
                        joining.add(queue.take());
                    }
                    queue.drainTo(joining, maxBatch - active.size() - joining.size());
                    joining.removeIf(j -> {
                        if (j.row.options.maxNewTokens() == 0) {
                            j.future.complete(j.row.finish());
                            return true;
                        }
                        return false;
                    });
                    try {
                        if (!joining.isEmpty()) {
                            LanguageModel.State fresh = prefill(joining);
                            state = state == null ? fresh : state.join(fresh);
                            active.addAll(joining);
                            largest = Math.max(largest, active.size());
                        }
                        if (active.isEmpty()) {
                            state = null;
                            continue;
                        }
                        int[] next = new int[active.size()];
                        boolean[] done = new boolean[active.size()];
                        for (int i = 0; i < active.size(); i++) {
                            Job j = active.get(i);
                            Row r = j.row;
                            if (j.logits == null) {
                                next[i] = j.chosen;
                                done[i] = r.add(j.chosen, j.stops);
                                continue;
                            }
                            float[] row = j.logits;
                            if (r.matcher != null && !constrain(row, r.matcher, j.stops)) {
                                r.end();
                                done[i] = true;
                                continue;
                            }
                            int token = r.options.sampling().sample(row, r.seen(), j.random);
                            next[i] = token;
                            if (r.matcher != null && !j.stops.contains(token)) {
                                r.matcher = vocabulary(row.length).advance(r.matcher, token);
                            }
                            done[i] = r.add(token, j.stops);
                        }
                        List<Job> kept = new ArrayList<>();
                        List<Integer> rows = new ArrayList<>();
                        for (int i = 0; i < active.size(); i++) {
                            if (done[i]) {
                                active.get(i).future.complete(active.get(i).row.finish());
                            } else {
                                kept.add(active.get(i));
                                rows.add(i);
                            }
                        }
                        if (kept.isEmpty()) {
                            active.clear();
                            state.close();
                            state = null;
                            continue;
                        }
                        if (kept.size() < active.size()) {
                            state = state.select(rows.stream().mapToInt(Integer::intValue).toArray());
                        }
                        active = kept;
                        int[][] tokens = new int[active.size()][1];
                        for (int i = 0; i < active.size(); i++) {
                            tokens[i][0] = next[rows.get(i)];
                        }
                        spread(model.next(state, tokens, null), active);
                    } catch (RuntimeException e) {
                        for (Job j : active) {
                            j.future.completeExceptionally(e);
                        }
                        for (Job j : joining) {
                            j.future.completeExceptionally(e);
                        }
                        active.clear();
                        state = null;
                    }
                }
            } catch (InterruptedException e) {
                // closed
            }
            for (Job j : active) {
                j.future.completeExceptionally(new java.util.concurrent.CancellationException("batcher closed"));
            }
        }

        /** Reads new prompts side by side, padded on the left, and keeps each one's logits. */
        private LanguageModel.State prefill(List<Job> jobs) {
            int n = jobs.size();
            int longest = 0;
            for (Job j : jobs) {
                longest = Math.max(longest, j.row.prompt.length);
            }
            int[][] first = new int[n][longest];
            boolean[][] pad = new boolean[n][longest];
            boolean padded = false;
            for (int b = 0; b < n; b++) {
                int[] p = jobs.get(b).row.prompt;
                int off = longest - p.length;
                System.arraycopy(p, 0, first[b], off, p.length);
                for (int t = 0; t < off; t++) {
                    pad[b][t] = true;
                    padded = true;
                }
            }
            LanguageModel.State s = model.newState(n);
            spread(model.next(s, first, padded ? pad : null), jobs);
            return s;
        }

        /**
         * Hands each job its logits. When every job decodes greedily without constraints or
         * a repetition penalty, only the chosen tokens are read back: the device finds them,
         * so the whole vocabulary per row is neither downloaded nor copied.
         */
        private void spread(Tensor logits, List<Job> jobs) {
            boolean greedy = true;
            for (Job j : jobs) {
                Sampling s = j.row.options.sampling();
                greedy &= j.row.matcher == null && s.temperature() == 0 && s.repetitionPenalty() == 1;
            }
            if (greedy) {
                float[] best = logits.argmax(-1).toArray();
                for (int i = 0; i < jobs.size(); i++) {
                    jobs.get(i).logits = null;
                    jobs.get(i).chosen = (int) best[i];
                }
                return;
            }
            float[] all = logits.toArray();
            int vocab = all.length / jobs.size();
            for (int i = 0; i < jobs.size(); i++) {
                jobs.get(i).logits = Arrays.copyOfRange(all, i * vocab, (i + 1) * vocab);
            }
        }
    }

    // ---------------------------------------------------------------- speculative decoding

    /**
     * Speculative decoding (Leviathan et al., 2023; Chen et al., 2023): a small draft model
     * guesses {@code lookahead} tokens one by one, then this generator's model checks them all
     * in one step and keeps the longest run it agrees with, plus one token of its own. Each
     * token is accepted with probability {@code min(1, p / q)} and a rejected one is replaced
     * from {@code max(0, p - q)}, so the text follows the large model's distribution exactly:
     * greedy decoding gives the same tokens as {@link #generate}, only in fewer steps of the
     * large model. Both models must share the vocabulary and be able to
     * {@linkplain LanguageModel.State#truncate go back}.
     *
     * @param prompt the token ids to continue, one sequence
     * @param draft the small model
     * @param lookahead the tokens guessed per step
     * @param options as for {@link #generate}; the repetition penalty is not supported
     * @param listener receives the text as it is written, or {@code null}
     * @return the result, with {@link Output#tokens()} and the number of tokens accepted
     */
    public Speculation speculative(int[] prompt, LanguageModel draft, int lookahead, Options options,
            Listener listener) {
        if (lookahead < 1) {
            throw new IllegalArgumentException("lookahead must be at least 1, got " + lookahead);
        }
        if (prompt.length == 0) {
            throw new IllegalArgumentException("empty prompt");
        }
        if (draft.vocabSize() != model.vocabSize()) {
            throw new IllegalArgumentException("the draft model's vocabulary (" + draft.vocabSize()
                    + ") differs from the model's (" + model.vocabSize() + ")");
        }
        Sampling sampling = options.sampling();
        if (sampling.repetitionPenalty() != 1) {
            throw new IllegalArgumentException("speculative decoding does not support a repetition penalty");
        }
        Set<Integer> stops = stopTokens(options);
        Row row = new Row(0, prompt, options, listener);
        SplittableRandom random = new SplittableRandom(options.seed());
        LanguageModel.State target = model.newState(1);
        LanguageModel.State small = draft.newState(1);
        int proposed = 0;
        int accepted = 0;
        int targetSteps = 0;
        if (options.maxNewTokens() > 0) {
            float[] p = model.next(target, new int[][] {prompt}, null).toArray();
            float[] q = draft.next(small, new int[][] {prompt}, null).toArray();
            targetSteps++;
            int vocab = model.vocabSize();
            int pending = -1;   // the model's last token, which the model has not read yet
            boolean done = false;
            while (!done) {
                int base = target.length();
                // the draft's guesses, and the distributions it drew them from
                int[] guess = new int[lookahead];
                double[][] qs = new double[lookahead][];
                for (int i = 0; i < lookahead; i++) {
                    qs[i] = sampling.probabilities(q);
                    guess[i] = draw(qs[i], random);
                    if (i + 1 < lookahead) {
                        q = draft.next(small, new int[][] {{guess[i]}}, null).toArray();
                    }
                }
                // one step of the model reads its pending token and every guess, and gives
                // its distribution before each guess and after the last one
                int skip = pending < 0 ? 0 : 1;
                int[] read = new int[skip + lookahead];
                if (skip == 1) {
                    read[0] = pending;
                }
                System.arraycopy(guess, 0, read, skip, lookahead);
                float[] all = model.nextAll(target, new int[][] {read}).toArray();
                targetSteps++;
                int keep = 0;
                int extra = -1;
                for (int i = 0; i < lookahead; i++) {
                    int at = skip + i - 1;
                    float[] logits = at < 0 ? p : Arrays.copyOfRange(all, at * vocab, (at + 1) * vocab);
                    double[] ps = sampling.probabilities(logits);
                    int x = guess[i];
                    if (random.nextDouble() * qs[i][x] < ps[x]) {
                        keep++;
                        continue;
                    }
                    // rejected: from what the model wants more than the draft offered
                    double[] rest = new double[vocab];
                    for (int t = 0; t < vocab; t++) {
                        rest[t] = Math.max(0, ps[t] - qs[i][t]);
                    }
                    extra = draw(rest, random);
                    break;
                }
                proposed += lookahead;
                accepted += keep;
                if (extra < 0) {
                    // every guess kept: one more token from the model's last distribution
                    int at = skip + lookahead - 1;
                    extra = draw(sampling.probabilities(Arrays.copyOfRange(all, at * vocab, (at + 1) * vocab)),
                            random);
                }
                for (int i = 0; i < keep && !done; i++) {
                    done = row.add(guess[i], stops);
                }
                if (!done) {
                    done = row.add(extra, stops);
                }
                if (done) {
                    break;
                }
                // the model keeps its pending token and the accepted guesses; the new token
                // waits for the next step
                target = target.truncate(base + skip + keep);
                pending = extra;
                // the draft keeps the accepted guesses, then reads the new token
                int draftBase = base + skip;
                int drafted = Math.min(keep, lookahead - 1);   // the draft never read its last guess
                small = small.truncate(draftBase + drafted);
                int[] missing = new int[keep - drafted + 1];
                for (int i = drafted; i < keep; i++) {
                    missing[i - drafted] = guess[i];
                }
                missing[missing.length - 1] = extra;
                q = draft.next(small, new int[][] {missing}, null).toArray();
            }
        }
        target.close();
        small.close();
        return new Speculation(row.finish(), proposed, accepted, targetSteps);
    }

    /**
     * What {@link #speculative} wrote, and how well the draft guessed.
     *
     * @param output the text and tokens
     * @param proposed the tokens the draft guessed
     * @param accepted the guesses the model kept
     * @param modelSteps the steps the large model took
     */
    public record Speculation(Output output, int proposed, int accepted, int modelSteps) {

        /** @return the share of guesses kept */
        public double acceptance() {
            return proposed == 0 ? 0 : (double) accepted / proposed;
        }
    }

    /** A token drawn from weights that need not add up to 1 (all zero: the first largest). */
    private static int draw(double[] weights, SplittableRandom random) {
        double sum = 0;
        int best = 0;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i];
            if (weights[i] > weights[best]) {
                best = i;
            }
        }
        if (sum <= 0) {
            return best;
        }
        double r = random.nextDouble() * sum;
        for (int i = 0; i < weights.length; i++) {
            r -= weights[i];
            if (r < 0 && weights[i] > 0) {
                return i;
            }
        }
        return best;
    }

    // ---------------------------------------------------------------- beam search

    /**
     * Beam search: keeps the {@code beams} most likely sequences at every step and returns
     * the most likely one, scored by the sum of its log-probabilities divided by its length to
     * the power {@code lengthPenalty}.
     *
     * @param prompt the token ids to continue
     * @param beams how many sequences to keep
     * @param maxNewTokens the most tokens to add
     * @param lengthPenalty 1 to compare average log-probabilities, 0 to compare sums
     * @param stopTokens tokens that end a sequence; {@code null} for the tokenizer's
     *     end-of-sequence token
     * @return the new tokens, without a stop token
     */
    public int[] beamSearch(int[] prompt, int beams, int maxNewTokens, double lengthPenalty,
            Set<Integer> stopTokens) {
        if (beams < 1) {
            throw new IllegalArgumentException("beams < 1");
        }
        Set<Integer> stops = stopTokens != null ? stopTokens
                : stopTokens(new Options(0, Sampling.GREEDY, 0, null, null));
        record Hyp(int[] tokens, double score, double normalized) {
        }
        List<Hyp> finished = new ArrayList<>();
        LanguageModel.State state = model.newState(1);
        Tensor logits = model.next(state, new int[][] {prompt}, null);
        int[][] alive = {new int[0]};
        double[] scores = {0};
        for (int step = 0; step < maxNewTokens && alive.length > 0; step++) {
            float[] all = logits.toArray();
            int n = alive.length;
            int vocab = all.length / n;
            int want = 2 * beams;
            // candidates: (score, parent, token), kept in a small sorted list
            List<double[]> cands = new ArrayList<>();
            for (int r = 0; r < n; r++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int v = 0; v < vocab; v++) {
                    max = Math.max(max, all[r * vocab + v]);
                }
                double sum = 0;
                for (int v = 0; v < vocab; v++) {
                    sum += Math.exp(all[r * vocab + v] - max);
                }
                double lse = max + Math.log(sum);
                for (int v = 0; v < vocab; v++) {
                    cands.add(new double[] {scores[r] + all[r * vocab + v] - lse, r, v});
                }
            }
            cands.sort(Comparator.comparingDouble((double[] c) -> c[0]).reversed());
            List<int[]> nextTokens = new ArrayList<>();
            List<Integer> parents = new ArrayList<>();
            List<Double> nextScores = new ArrayList<>();
            int length = step + 1;
            for (int i = 0; i < Math.min(want, cands.size()) && nextTokens.size() < beams; i++) {
                double[] c = cands.get(i);
                int parent = (int) c[1];
                int token = (int) c[2];
                if (stops.contains(token)) {
                    if (i < beams) {
                        int[] t = alive[parent];
                        finished.add(new Hyp(t, c[0], c[0] / Math.pow(Math.max(1, t.length), lengthPenalty)));
                    }
                    continue;
                }
                int[] t = Arrays.copyOf(alive[parent], length);
                t[length - 1] = token;
                nextTokens.add(t);
                parents.add(parent);
                nextScores.add(c[0]);
            }
            finished.sort(Comparator.comparingDouble((Hyp h) -> h.normalized).reversed());
            if (finished.size() > beams) {
                finished.subList(beams, finished.size()).clear();
            }
            alive = nextTokens.toArray(int[][]::new);
            scores = nextScores.stream().mapToDouble(Double::doubleValue).toArray();
            if (finished.size() == beams && alive.length > 0) {
                double best = scores[0] / Math.pow(length, lengthPenalty);
                if (best <= finished.getLast().normalized) {
                    break;
                }
            }
            if (step + 1 < maxNewTokens && alive.length > 0) {
                state = state.select(parents.stream().mapToInt(Integer::intValue).toArray());
                int[][] feed = new int[alive.length][];
                for (int r = 0; r < alive.length; r++) {
                    feed[r] = new int[] {alive[r][length - 1]};
                }
                logits = model.next(state, feed, null);
            }
        }
        state.close();
        for (int r = 0; r < alive.length; r++) {
            int[] t = alive[r];
            finished.add(new Hyp(t, scores[r], scores[r] / Math.pow(Math.max(1, t.length), lengthPenalty)));
        }
        return finished.stream().max(Comparator.comparingDouble(Hyp::normalized)).orElseThrow().tokens();
    }
}
