package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.CancellationException;
import java.util.function.Consumer;
import java.util.zip.Deflater;

/**
 * Whisper speech-to-text: audio of any length, decoded 30 seconds at a time as OpenAI's
 * {@code whisper.transcribe} does. Each window is decoded greedily with Whisper's rules
 * (suppressed tokens, timestamps that only move forward, a first timestamp within one
 * second), split into segments at timestamp pairs, and the next window starts where the
 * last complete segment ended. A window whose text repeats itself or whose tokens are
 * unlikely is decoded again at a higher temperature; a window the model thinks is silent
 * is skipped.
 *
 * <pre>{@code
 * try (WhisperTranscriber stt = WhisperTranscriber.load(Path.of("whisper-base"), Device.cpu(), DType.FLOAT32)) {
 *     Transcription t = stt.transcribe(Wav.read(Path.of("talk.wav")));
 *     System.out.println(t.language() + ": " + t.text());
 * }
 * }</pre>
 */
public final class WhisperTranscriber implements SpeechToText {

    private static final float[] FALLBACK_TEMPERATURES = {0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f};
    private static final double COMPRESSION_RATIO_THRESHOLD = 2.4;
    private static final double LOGPROB_THRESHOLD = -1.0;
    private static final double NO_SPEECH_THRESHOLD = 0.6;
    /** Seconds per timestamp token. */
    private static final double TIME_PRECISION = 0.02;
    /** Mel frames per encoder position. */
    private static final int INPUT_STRIDE = 2;

    private final Whisper model;
    private final WhisperTokenizer tokenizer;
    private final int[] suppress;
    private final int[] beginSuppress;

    /**
     * Combines a model and its tokenizer.
     *
     * @param model the network
     * @param tokenizer its tokenizer
     * @param suppressTokens tokens never to produce, or {@code null} for Whisper's non-speech symbols
     */
    public WhisperTranscriber(Whisper model, WhisperTokenizer tokenizer, int[] suppressTokens) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.suppress = suppressTokens != null ? suppressTokens.clone() : tokenizer.nonSpeechTokens();
        int[] blank = tokenizer.encode(" ");
        this.beginSuppress = new int[] {blank.length > 0 ? blank[0] : tokenizer.eot(), tokenizer.eot()};
    }

    /**
     * Loads a Hugging Face Whisper folder: the model, its tokenizer and the
     * {@code suppress_tokens} of {@code generation_config.json}.
     *
     * @param dir the folder
     * @param device where the model runs
     * @param dtype the type of its weights
     * @return the transcriber
     */
    public static WhisperTranscriber load(Path dir, Device device, DType dtype) {
        Whisper model = Whisper.load(dir, device, dtype);
        WhisperTokenizer tok = WhisperTokenizer.load(dir, model.config().vocabSize());
        int[] suppress = null;
        Path gen = dir.resolve("generation_config.json");
        if (Files.isRegularFile(gen)) {
            try {
                Map<String, Object> g = Json.object(Json.read(gen));
                if (g.get("suppress_tokens") instanceof List<?> list) {
                    suppress = list.stream().mapToInt(o -> ((Number) o).intValue()).toArray();
                }
            } catch (IOException e) {
                throw new UncheckedIOException("cannot read " + gen + ": " + e.getMessage(), e);
            }
        }
        return new WhisperTranscriber(model, tok, suppress);
    }

    /**
     * Returns the network.
     *
     * @return the model
     */
    public Whisper model() {
        return model;
    }

    /**
     * Returns the tokenizer.
     *
     * @return the tokenizer
     */
    public WhisperTokenizer tokenizer() {
        return tokenizer;
    }

    @Override
    public List<String> languages() {
        return tokenizer.multilingual() ? tokenizer.languageCodes() : List.of("en");
    }

    @Override
    public boolean canTranslate() {
        return tokenizer.multilingual();
    }

    /**
     * Detects the spoken language of the first 30 seconds of audio.
     *
     * @param samples 16 kHz mono samples
     * @return the language code
     */
    public String detectLanguage(float[] samples) {
        if (!tokenizer.multilingual()) {
            return "en";
        }
        float[][] mel = WhisperFeatures.padded(samples, model.config().mels());
        Tensor audio = model.encode(WhisperFeatures.window(mel, 0));
        try {
            return detect(audio);
        } finally {
            audio.release();
        }
    }

    private String detect(Tensor audio) {
        try (Whisper.Decoder d = model.decoder(audio)) {
            float[] logits = d.next(tokenizer.sot());
            int best = -1;
            for (int t : tokenizer.languageTokens()) {
                if (best < 0 || logits[t] > logits[best]) {
                    best = t;
                }
            }
            return tokenizer.languageOf(best);
        }
    }

    /** One decoded window. */
    private record Result(int[] tokens, double avgLogProb, double noSpeechProb, double compressionRatio,
            float temperature) {
    }

    @Override
    public Transcription transcribe(float[] samples, TranscriptionOptions options,
            Consumer<Transcription.Segment> onSegment) {
        WhisperConfig c = model.config();
        float[][] mel = WhisperFeatures.padded(samples, c.mels());
        int contentFrames = mel[0].length - WhisperFeatures.WINDOW_FRAMES;
        String language = options.language() == null ? null : WhisperTokenizer.languageCode(options.language());
        if (!tokenizer.multilingual()) {
            if (language != null && !language.equals("en")) {
                throw new IllegalArgumentException("this Whisper model is English-only; it cannot take language '"
                        + options.language() + "'");
            }
            if (options.task() == TranscriptionOptions.Task.TRANSLATE) {
                throw new IllegalArgumentException("this Whisper model is English-only and cannot translate;"
                        + " use a multilingual model such as whisper-base");
            }
            language = "en";
        } else if (language != null) {
            tokenizer.languageToken(language);   // fails early on an unknown language
        }
        List<Transcription.Segment> segments = new ArrayList<>();
        List<Integer> allTokens = new ArrayList<>();
        if (options.prompt() != null && !options.prompt().isBlank()) {
            for (int t : tokenizer.encode(" " + options.prompt().strip())) {
                allTokens.add(t);
            }
        }
        int promptResetSince = 0;
        int seek = 0;
        while (seek < contentFrames) {
            check(options);
            int segmentSize = Math.min(WhisperFeatures.WINDOW_FRAMES, contentFrames - seek);
            double timeOffset = (double) seek * WhisperFeatures.HOP / WhisperFeatures.SAMPLE_RATE;
            double segmentDuration = (double) segmentSize * WhisperFeatures.HOP / WhisperFeatures.SAMPLE_RATE;
            float[][] window = WhisperFeatures.window(mel, seek);
            if (segmentSize < WhisperFeatures.WINDOW_FRAMES) {
                for (float[] row : window) {
                    Arrays.fill(row, segmentSize, row.length, 0f);   // as Whisper's pad_or_trim
                }
            }
            Tensor audio = model.encode(window);
            Result r;
            try {
                if (language == null) {
                    language = detect(audio);
                }
                List<Integer> context = allTokens.subList(promptResetSince, allTokens.size());
                r = decodeWithFallback(audio, language, options, context);
            } finally {
                audio.release();
            }
            if (r.noSpeechProb > NO_SPEECH_THRESHOLD && r.avgLogProb < LOGPROB_THRESHOLD) {
                seek += segmentSize;   // silence
                continue;
            }
            int[] tokens = r.tokens;
            int tb = tokenizer.timestampBegin();
            int n = tokens.length;
            boolean singleTimestampEnding = n >= 2 && tokens[n - 2] < tb && tokens[n - 1] >= tb;
            List<Integer> consecutive = new ArrayList<>();
            for (int i = 1; i < n; i++) {
                if (tokens[i] >= tb && tokens[i - 1] >= tb) {
                    consecutive.add(i);
                }
            }
            List<Transcription.Segment> added = new ArrayList<>();
            List<Integer> spoken = new ArrayList<>();
            if (!consecutive.isEmpty()) {
                if (singleTimestampEnding) {
                    consecutive.add(n);
                }
                int last = 0;
                for (int current : consecutive) {
                    int[] sliced = Arrays.copyOfRange(tokens, last, current);
                    double start = timeOffset + (sliced[0] - tb) * TIME_PRECISION;
                    double end = timeOffset + (sliced[sliced.length - 1] - tb) * TIME_PRECISION;
                    added.add(segment(start, end, sliced, r));
                    for (int t : sliced) {
                        spoken.add(t);
                    }
                    last = current;
                }
                if (singleTimestampEnding) {
                    seek += segmentSize;
                } else {
                    int lastTimestamp = tokens[last - 1] - tb;
                    seek += lastTimestamp * INPUT_STRIDE;
                }
            } else {
                double duration = segmentDuration;
                for (int i = n - 1; i >= 0; i--) {
                    if (tokens[i] >= tb) {
                        if (tokens[i] != tb) {
                            duration = (tokens[i] - tb) * TIME_PRECISION;
                        }
                        break;
                    }
                }
                added.add(segment(timeOffset, timeOffset + duration, tokens, r));
                for (int t : tokens) {
                    spoken.add(t);
                }
                seek += segmentSize;
            }
            for (Transcription.Segment s : added) {
                if (s.text().isBlank() || s.start() == s.end()) {
                    continue;
                }
                segments.add(s);
                if (onSegment != null) {
                    onSegment.accept(s);
                }
            }
            allTokens.addAll(spoken);
            if (!options.conditionOnPreviousText() || r.temperature > 0.5f) {
                promptResetSince = allTokens.size();
            }
        }
        StringBuilder text = new StringBuilder();
        for (Transcription.Segment s : segments) {
            text.append(s.text());
        }
        return new Transcription(text.toString().strip(), language, segments,
                (double) samples.length / WhisperFeatures.SAMPLE_RATE);
    }

    private Transcription.Segment segment(double start, double end, int[] tokens, Result r) {
        return new Transcription.Segment(start, end, tokenizer.decode(tokens), r.avgLogProb, r.noSpeechProb);
    }

    private static void check(TranscriptionOptions options) {
        if (options.cancelled().getAsBoolean()) {
            throw new CancellationException("transcription cancelled");
        }
    }

    private Result decodeWithFallback(Tensor audio, String language, TranscriptionOptions options,
            List<Integer> context) {
        Result r = null;
        for (float t : FALLBACK_TEMPERATURES) {
            if (t < options.temperature()) {
                continue;
            }
            r = decode(audio, language, options, context, t);
            boolean tooRepetitive = r.compressionRatio > COMPRESSION_RATIO_THRESHOLD;
            boolean unsure = r.avgLogProb < LOGPROB_THRESHOLD;
            boolean silent = r.noSpeechProb > NO_SPEECH_THRESHOLD;
            if (silent || !tooRepetitive && !unsure) {
                return r;
            }
        }
        if (r == null) {
            r = decode(audio, language, options, context, options.temperature());
        }
        return r;
    }

    /** Decodes one window at one temperature. */
    private Result decode(Tensor audio, String language, TranscriptionOptions options, List<Integer> context,
            float temperature) {
        WhisperConfig c = model.config();
        List<Integer> prompt = new ArrayList<>();
        int maxContext = c.textPositions() / 2 - 1;
        if (!context.isEmpty()) {
            prompt.add(tokenizer.startOfPrevious());
            prompt.addAll(context.subList(Math.max(0, context.size() - maxContext), context.size()));
        }
        int sotIndex = prompt.size();
        prompt.add(tokenizer.sot());
        if (tokenizer.multilingual()) {
            prompt.add(tokenizer.languageToken(language));
            prompt.add(options.task() == TranscriptionOptions.Task.TRANSLATE ? tokenizer.translate()
                    : tokenizer.transcribe());
        }
        if (!options.timestamps()) {
            prompt.add(tokenizer.noTimestamps());
        }
        int[] promptIds = prompt.stream().mapToInt(Integer::intValue).toArray();
        int sampleLen = c.textPositions() / 2;
        SplittableRandom random = new SplittableRandom(42);
        List<Integer> out = new ArrayList<>();
        double sumLogProb = 0;
        double noSpeech;
        try (Whisper.Decoder d = model.decoder(audio)) {
            float[][] first = d.nextAll(promptIds);
            noSpeech = softmax(first[sotIndex])[tokenizer.noSpeech()];
            float[] logits = first[first.length - 1];
            for (int step = 0; step < sampleLen && promptIds.length + step < c.textPositions(); step++) {
                check(options);
                filter(logits, out, options.timestamps());
                double[] logProbs = logSoftmax(logits);
                int next = temperature == 0 ? argmax(logits) : sample(logits, temperature, random);
                sumLogProb += logProbs[next];
                out.add(next);
                if (next == tokenizer.eot()) {
                    break;
                }
                logits = d.next(next);
            }
        }
        if (!out.isEmpty() && out.getLast() == tokenizer.eot()) {
            out.removeLast();
        }
        int[] tokens = out.stream().mapToInt(Integer::intValue).toArray();
        String text = tokenizer.decode(tokens);
        double avg = sumLogProb / (tokens.length + 1);
        return new Result(tokens, avg, noSpeech, compressionRatio(text), temperature);
    }

    /** Whisper's logit filters: blank start, suppressed tokens, timestamp rules. */
    private void filter(float[] logits, List<Integer> sampled, boolean timestamps) {
        float ninf = Float.NEGATIVE_INFINITY;
        int vocab = logits.length;
        if (sampled.isEmpty()) {
            for (int t : beginSuppress) {
                logits[t] = ninf;
            }
        }
        for (int t : suppress) {
            if (t >= 0 && t < vocab) {
                logits[t] = ninf;
            }
        }
        int tb = tokenizer.timestampBegin();
        // the special tokens between end-of-text and the timestamps are never sampled
        for (int t = tokenizer.eot() + 1; t < Math.min(tb, vocab); t++) {
            logits[t] = ninf;
        }
        if (!timestamps) {
            for (int t = tb; t < vocab; t++) {
                logits[t] = ninf;
            }
            return;
        }
        int n = sampled.size();
        boolean lastWasTimestamp = n >= 1 && sampled.get(n - 1) >= tb;
        boolean penultimateWasTimestamp = n < 2 || sampled.get(n - 2) >= tb;
        if (lastWasTimestamp) {
            if (penultimateWasTimestamp) {
                Arrays.fill(logits, tb, vocab, ninf);   // a pair: text must follow
            } else {
                Arrays.fill(logits, 0, tokenizer.eot(), ninf);   // an end timestamp: another timestamp or the end
            }
        }
        int lastTs = -1;
        for (int i = n - 1; i >= 0; i--) {
            if (sampled.get(i) >= tb) {
                lastTs = sampled.get(i);
                break;
            }
        }
        if (lastTs >= 0) {
            // timestamps never decrease, and do not repeat unless closing a segment
            int bound = lastWasTimestamp && !penultimateWasTimestamp ? lastTs : lastTs + 1;
            Arrays.fill(logits, tb, Math.min(bound, vocab), ninf);
        }
        if (n == 0) {
            Arrays.fill(logits, 0, tb, ninf);   // start with a timestamp
            int maxInitial = tb + (int) Math.round(1.0 / TIME_PRECISION);
            if (maxInitial + 1 < vocab) {
                Arrays.fill(logits, maxInitial + 1, vocab, ninf);
            }
        }
        // a timestamp when all timestamps together are likelier than any text token
        double[] lp = logSoftmax(logits);
        double maxText = Double.NEGATIVE_INFINITY;
        for (int t = 0; t < tb; t++) {
            maxText = Math.max(maxText, lp[t]);
        }
        double tsMax = Double.NEGATIVE_INFINITY;
        for (int t = tb; t < vocab; t++) {
            tsMax = Math.max(tsMax, lp[t]);
        }
        if (tsMax > Double.NEGATIVE_INFINITY) {
            double sum = 0;
            for (int t = tb; t < vocab; t++) {
                sum += Math.exp(lp[t] - tsMax);
            }
            if (tsMax + Math.log(sum) > maxText) {
                Arrays.fill(logits, 0, tb, ninf);
            }
        }
    }

    private static int argmax(float[] x) {
        int best = 0;
        for (int i = 1; i < x.length; i++) {
            if (x[i] > x[best]) {
                best = i;
            }
        }
        return best;
    }

    private static int sample(float[] logits, float temperature, SplittableRandom random) {
        double max = Double.NEGATIVE_INFINITY;
        for (float v : logits) {
            max = Math.max(max, v);
        }
        double[] p = new double[logits.length];
        double sum = 0;
        for (int i = 0; i < p.length; i++) {
            p[i] = Math.exp((logits[i] - max) / temperature);
            sum += p[i];
        }
        double u = random.nextDouble() * sum;
        for (int i = 0; i < p.length; i++) {
            u -= p[i];
            if (u <= 0) {
                return i;
            }
        }
        return argmax(logits);
    }

    private static double[] logSoftmax(float[] x) {
        double max = Double.NEGATIVE_INFINITY;
        for (float v : x) {
            max = Math.max(max, v);
        }
        double sum = 0;
        for (float v : x) {
            sum += Math.exp(v - max);
        }
        double lse = max + Math.log(sum);
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i] - lse;
        }
        return out;
    }

    private static double[] softmax(float[] x) {
        double[] lp = logSoftmax(x);
        for (int i = 0; i < lp.length; i++) {
            lp[i] = Math.exp(lp[i]);
        }
        return lp;
    }

    /** UTF-8 length over zlib-compressed length: high for text that repeats itself. */
    static double compressionRatio(String text) {
        byte[] b = text.getBytes(StandardCharsets.UTF_8);
        if (b.length == 0) {
            return 0;
        }
        Deflater d = new Deflater();
        d.setInput(b);
        d.finish();
        byte[] buf = new byte[b.length + 64];
        int len = 0;
        while (!d.finished()) {
            len += d.deflate(buf, len, buf.length - len);
            if (len == buf.length) {
                buf = Arrays.copyOf(buf, buf.length * 2);
            }
        }
        d.end();
        return (double) b.length / len;
    }

    @Override
    public void close() {
        model.close();
    }
}
