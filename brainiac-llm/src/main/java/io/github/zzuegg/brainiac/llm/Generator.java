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
     */
    public record Options(int maxNewTokens, Sampling sampling, long seed, Set<Integer> stopTokens,
            List<String> stopStrings) {

        /**
         * Checks the options.
         *
         * @param maxNewTokens the most tokens to add
         * @param sampling how to pick tokens
         * @param seed seeds the draws
         * @param stopTokens stop tokens
         * @param stopStrings stop strings
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
            return new Options(maxNewTokens, sampling, seed, tokens, stopStrings);
        }

        /**
         * @param strings the stop strings
         * @return a copy that stops on them
         */
        public Options withStopStrings(String... strings) {
            return new Options(maxNewTokens, sampling, seed, stopTokens, List.of(strings));
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
            for (int step = 0; active > 0; step++) {
                float[] all = logits.toArray();
                int vocab = all.length / batch;
                int[][] next = new int[batch][1];
                boolean[][] skip = new boolean[batch][1];
                for (int b = 0; b < batch; b++) {
                    Row r = rows[b];
                    if (r.finish != null) {
                        skip[b][0] = true;
                        continue;
                    }
                    float[] row = Arrays.copyOfRange(all, b * vocab, (b + 1) * vocab);
                    int token = options.sampling().sample(row, r.seen(), random);
                    next[b][0] = token;
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

        Row(int index, int[] prompt, Options options, Listener listener) {
            this.index = index;
            this.prompt = prompt;
            this.options = options;
            this.listener = listener;
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
        for (int r = 0; r < alive.length; r++) {
            int[] t = alive[r];
            finished.add(new Hyp(t, scores[r], scores[r] / Math.pow(Math.max(1, t.length), lengthPenalty)));
        }
        return finished.stream().max(Comparator.comparingDouble(Hyp::normalized)).orElseThrow().tokens();
    }
}
