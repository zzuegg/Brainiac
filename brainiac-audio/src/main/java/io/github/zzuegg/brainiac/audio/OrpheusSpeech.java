package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.llm.Generator;
import io.github.zzuegg.brainiac.llm.Llama;
import io.github.zzuegg.brainiac.llm.Sampling;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.concurrent.CancellationException;
import java.util.function.Consumer;
import java.util.regex.Pattern;

/**
 * Orpheus TTS (Canopy Labs, 2025): a Llama 3.2 3B language model that writes speech as SNAC
 * codec tokens, seven per 85 ms frame, which {@link Snac} turns into 24 kHz audio. Because it
 * is a language model it reads emotion tags in the text, such as {@code <laugh>},
 * {@code <sigh>} or {@code <gasp>}, and makes the sound. Loads the fine-tuned
 * {@code orpheus-3b-0.1-ft} (Hugging Face folder) with {@code hubertsiuzdak/snac_24khz} in or
 * next to it; its eight voices are named in the prompt.
 *
 * <pre>{@code
 * try (OrpheusSpeech tts = OrpheusSpeech.load(Path.of("models/orpheus-3b"), Device.cpu(), DType.BFLOAT16)) {
 *     Audio a = tts.synthesize("I can't believe it <laugh> that actually worked!",
 *             SpeechOptions.defaults().voice("leo"), null);
 * }
 * }</pre>
 */
public final class OrpheusSpeech implements TextToSpeech {

    /** The fine-tuned model's voices, the most natural first. */
    public static final List<String> VOICES = List.of("tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe");
    /** The emotion tags the fine-tuned model was trained on. */
    public static final List<String> TAGS = List.of("<laugh>", "<chuckle>", "<sigh>", "<cough>", "<sniffle>",
            "<groan>", "<yawn>", "<gasp>");

    static final int START_OF_HUMAN = 128259;
    static final int END_OF_TEXT = 128009;
    static final int END_OF_HUMAN = 128260;
    static final int START_OF_AI = 128261;
    static final int START_OF_SPEECH = 128257;
    static final int END_OF_SPEECH = 128258;
    static final int END_OF_AI = 128262;
    /** The first audio token: {@code <custom_token_10>}. */
    static final int AUDIO_BASE = 128266;
    static final int CODEBOOK = 4096;
    /** Text characters per generation, so the speech stays within the model's trained lengths. */
    static final int MAX_CHARS = 300;

    private final Llama model;
    private final PretrainedTokenizer tokenizer;
    private final Generator generator;
    private final Snac snac;
    private volatile Sampling sampling = new Sampling(0.6f, 0, 0.8f, 0, 1.3f);
    private volatile int maxTokens = 2000;

    private OrpheusSpeech(Llama model, PretrainedTokenizer tokenizer, Snac snac) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.generator = new Generator(model, tokenizer);
        this.snac = snac;
    }

    /**
     * Loads Orpheus and finds SNAC: a SNAC folder inside {@code dir}, next to it, or next to its
     * parent.
     *
     * @param dir the Orpheus folder (Hugging Face Llama with the tokenizer)
     * @param device where it runs
     * @param dtype the type of the language model's weights
     * @return the text-to-speech model
     */
    public static OrpheusSpeech load(Path dir, Device device, DType dtype) {
        Path snacDir = findSnac(dir);
        if (snacDir == null) {
            throw new IllegalArgumentException("no SNAC codec (hubertsiuzdak/snac_24khz: config.json and"
                    + " pytorch_model.bin) in or next to " + dir);
        }
        Snac snac = Snac.load(snacDir, device, DType.FLOAT32);
        try {
            return new OrpheusSpeech(Llama.load(dir, device, dtype), PretrainedTokenizer.load(dir), snac);
        } catch (RuntimeException e) {
            snac.close();
            throw e;
        }
    }

    /**
     * Whether a folder holds Orpheus: a Llama {@code config.json} whose vocabulary has the
     * audio tokens, with SNAC in reach.
     *
     * @param dir a folder
     * @return whether {@link #load} can load it
     */
    public static boolean isOrpheus(Path dir) {
        Path config = dir.resolve("config.json");
        if (!Files.isRegularFile(config)) {
            return false;
        }
        try {
            Map<String, Object> c = Json.object(Json.read(config));
            return "llama".equals(c.get("model_type")) && c.get("vocab_size") instanceof Number n
                    && n.intValue() >= AUDIO_BASE + 7 * CODEBOOK && findSnac(dir) != null;
        } catch (IOException | RuntimeException e) {
            return false;
        }
    }

    static Path findSnac(Path dir) {
        Path p = dir.toAbsolutePath().normalize();
        for (int up = 0; up < 3 && p != null; up++, p = p.getParent()) {
            if (Snac.isSnac(p)) {
                return p;
            }
            try (var subs = Files.list(p)) {
                Path found = subs.filter(Files::isDirectory).sorted().filter(Snac::isSnac).findFirst().orElse(null);
                if (found != null) {
                    return found;
                }
            } catch (IOException e) {
                // look further up
            }
        }
        return null;
    }

    /** @return the language model */
    public Llama model() {
        return model;
    }

    /** @return the codec */
    public Snac codec() {
        return snac;
    }

    /**
     * Sets how audio tokens are sampled (by default temperature 0.6, top-p 0.8, repetition
     * penalty 1.3, as the orpheus-speech package).
     *
     * @param s the sampling
     * @param maxTokens the most audio tokens per piece of text (7 per 85 ms)
     */
    public void sampling(Sampling s, int maxTokens) {
        this.sampling = java.util.Objects.requireNonNull(s, "sampling");
        this.maxTokens = maxTokens;
    }

    @Override
    public int sampleRate() {
        return snac.config().sampleRate();
    }

    @Override
    public List<Voice> voices() {
        return VOICES.stream().map(v -> new Voice(v, v.substring(0, 1).toUpperCase() + v.substring(1), "en")).toList();
    }

    @Override
    public List<String> tags() {
        return TAGS;
    }

    /**
     * The prompt for one piece of text: the voice's name, the text, and the markers that
     * start the speech.
     *
     * @param text the token ids of {@code "voice: text"}, with the begin-of-text token
     * @return the prompt
     */
    static int[] prompt(int[] text) {
        int[] p = new int[text.length + 5];
        p[0] = START_OF_HUMAN;
        System.arraycopy(text, 0, p, 1, text.length);
        p[text.length + 1] = END_OF_TEXT;
        p[text.length + 2] = END_OF_HUMAN;
        p[text.length + 3] = START_OF_AI;
        p[text.length + 4] = START_OF_SPEECH;
        return p;
    }

    /**
     * SNAC codes from generated tokens: the tokens after the last start-of-speech marker,
     * seven per frame (one coarse, two middle, four fine codes, each frame position offset by
     * its own 4096 block). Frames with a code out of range are dropped.
     *
     * @param tokens the generated tokens
     * @return {@code [3][]}: {@code frames}, {@code 2 frames} and {@code 4 frames} codes
     */
    static int[][] codes(int[] tokens) {
        int from = 0;
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i] == START_OF_SPEECH) {
                from = i + 1;
            }
        }
        List<Integer> audio = new ArrayList<>();
        for (int i = from; i < tokens.length; i++) {
            int t = tokens[i];
            if (t == END_OF_SPEECH || t == END_OF_AI) {
                break;
            }
            if (t >= AUDIO_BASE) {
                audio.add(t - AUDIO_BASE);
            }
        }
        int frames = audio.size() / 7;
        int[] l1 = new int[frames];
        int[] l2 = new int[2 * frames];
        int[] l3 = new int[4 * frames];
        int kept = 0;
        for (int f = 0; f < frames; f++) {
            int[] c = new int[7];
            boolean ok = true;
            for (int j = 0; j < 7; j++) {
                c[j] = audio.get(7 * f + j) - j * CODEBOOK;
                ok &= c[j] >= 0 && c[j] < CODEBOOK;
            }
            if (!ok) {
                continue;
            }
            l1[kept] = c[0];
            l2[2 * kept] = c[1];
            l3[4 * kept] = c[2];
            l3[4 * kept + 1] = c[3];
            l2[2 * kept + 1] = c[4];
            l3[4 * kept + 2] = c[5];
            l3[4 * kept + 3] = c[6];
            kept++;
        }
        return new int[][] {java.util.Arrays.copyOf(l1, kept), java.util.Arrays.copyOf(l2, 2 * kept),
                java.util.Arrays.copyOf(l3, 4 * kept)};
    }

    private static final Pattern SENTENCE_END = Pattern.compile("(?<=[.!?…])\\s+");

    /** Sentences grouped into pieces of at most {@link #MAX_CHARS} characters (longer sentences stay whole). */
    static List<String> pieces(String text) {
        return pieces(text, MAX_CHARS);
    }

    /** Sentences grouped into pieces of at most {@code maxChars} characters (longer sentences stay whole). */
    static List<String> pieces(String text, int maxChars) {
        List<String> out = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        for (String line : text.split("\\n+")) {
            for (String s : SENTENCE_END.split(line.strip())) {
                if (s.isBlank()) {
                    continue;
                }
                if (!cur.isEmpty() && cur.length() + 1 + s.length() > maxChars) {
                    out.add(cur.toString());
                    cur.setLength(0);
                }
                if (!cur.isEmpty()) {
                    cur.append(' ');
                }
                cur.append(s.strip());
            }
        }
        if (!cur.isEmpty()) {
            out.add(cur.toString());
        }
        return out;
    }

    @Override
    public Audio synthesize(String text, SpeechOptions options, Consumer<Audio> onChunk) {
        String voice = options.voice() == null ? VOICES.getFirst() : options.voice();
        if (voice.contains("+") || voice.contains("*")) {
            throw new IllegalArgumentException("Orpheus cannot blend voices; pick one of " + VOICES);
        }
        List<String> pieces = pieces(text);
        if (pieces.isEmpty()) {
            throw new IllegalArgumentException("nothing to say: the text is empty");
        }
        List<float[]> parts = new ArrayList<>();
        int total = 0;
        for (int i = 0; i < pieces.size(); i++) {
            if (options.cancelled().getAsBoolean()) {
                throw new CancellationException("speech synthesis cancelled");
            }
            int[] prompt = prompt(tokenizer.encode(voice + ": " + pieces.get(i), true));
            Generator.Options o = new Generator.Options(maxTokens, sampling, options.seed() + i,
                    Set.of(END_OF_SPEECH, END_OF_AI, END_OF_TEXT), List.of());
            int[] tokens = generator.generate(new int[][] {prompt}, o, (row, t) -> {
                if (options.cancelled().getAsBoolean()) {
                    throw new CancellationException("speech synthesis cancelled");
                }
            }).getFirst().tokens();
            int[][] codes = codes(tokens);
            if (codes[0].length == 0) {
                continue;
            }
            long seed = options.seed() + i;
            float[] wave = snac.decode(codes, b -> new SplittableRandom(seed * 31 + b));
            parts.add(wave);
            total += wave.length;
            if (onChunk != null) {
                onChunk.accept(new Audio(wave, sampleRate()));
            }
        }
        if (parts.isEmpty()) {
            throw new IllegalStateException("the model wrote no speech");
        }
        float[] all = new float[total];
        int at = 0;
        for (float[] p : parts) {
            System.arraycopy(p, 0, all, at, p.length);
            at += p.length;
        }
        return new Audio(all, sampleRate());
    }

    @Override
    public void close() {
        snac.close();
    }
}
