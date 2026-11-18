package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;

/**
 * Kokoro text-to-speech: English text through {@link EnglishPhonemizer}, a sentence (or up
 * to 510 phonemes) at a time through {@link Kokoro}, with the voice packs of the model's
 * {@code voices} folder. Voices blend: {@code "af_heart*0.7+am_adam*0.3"} mixes two styles,
 * and {@link #addVoice} adds a custom one (a saved blend, or a style vector from elsewhere).
 *
 * <pre>{@code
 * try (KokoroSpeech tts = KokoroSpeech.load(Path.of("Kokoro-82M"), Device.cpu(), DType.FLOAT32)) {
 *     Audio a = tts.synthesize("Hello there!", SpeechOptions.defaults().voice("af_heart*0.5+bf_emma*0.5"), null);
 * }
 * }</pre>
 */
public final class KokoroSpeech implements TextToSpeech {

    /** The most phonemes Kokoro reads at once. */
    public static final int MAX_PHONEMES = 510;

    private static final Map<Character, String> ACCENTS = Map.of('a', "American", 'b', "British");

    private final Kokoro model;
    private final Map<String, Path> voiceFiles = new LinkedHashMap<>();
    private final Map<String, float[][]> packs = new ConcurrentHashMap<>();
    private final List<Voice> voices = new ArrayList<>();

    private KokoroSpeech(Kokoro model, Path voiceDir) {
        this.model = model;
        if (voiceDir != null && Files.isDirectory(voiceDir)) {
            try (var files = Files.list(voiceDir)) {
                files.filter(f -> f.getFileName().toString().endsWith(".pt")).sorted().forEach(f -> {
                    String id = f.getFileName().toString().replaceFirst("\\.pt$", "");
                    voiceFiles.put(id, f);
                });
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
        // English voices first (the phonemizer reads English), the default first of all
        List<String> ids = new ArrayList<>(voiceFiles.keySet());
        ids.removeIf(id -> !isEnglish(id));
        ids.sort((x, y) -> Boolean.compare(!x.equals("af_heart"), !y.equals("af_heart")));
        for (String id : ids) {
            voices.add(new Voice(id, name(id), "en"));
        }
        if (voices.isEmpty()) {
            throw new IllegalArgumentException("no English voices (voices/a*.pt or voices/b*.pt) in " + voiceDir);
        }
    }

    /**
     * Loads a Kokoro folder with its {@code voices} folder.
     *
     * @param dir the folder
     * @param device where the model runs
     * @param dtype the type of its weights
     * @return the text-to-speech model
     */
    public static KokoroSpeech load(Path dir, Device device, DType dtype) {
        Kokoro model = Kokoro.load(dir, device, dtype);
        try {
            return new KokoroSpeech(model, dir.resolve("voices"));
        } catch (RuntimeException e) {
            model.close();
            throw e;
        }
    }

    private static boolean isEnglish(String id) {
        return id.length() > 3 && id.charAt(2) == '_' && ACCENTS.containsKey(id.charAt(0));
    }

    private static String name(String id) {
        String given = id.substring(3);
        given = given.isEmpty() ? id : given.substring(0, 1).toUpperCase(Locale.ROOT) + given.substring(1);
        String sex = switch (id.charAt(1)) {
            case 'f' -> "female";
            case 'm' -> "male";
            default -> "voice";
        };
        return given + " (" + ACCENTS.get(id.charAt(0)) + " " + sex + ")";
    }

    /** @return the network */
    public Kokoro model() {
        return model;
    }

    @Override
    public int sampleRate() {
        return Kokoro.SAMPLE_RATE;
    }

    @Override
    public List<Voice> voices() {
        return List.copyOf(voices);
    }

    @Override
    public boolean canMixVoices() {
        return true;
    }

    /**
     * A voice's style vectors: one {@code [256]} row per phoneme count (the model picks the
     * row for the length of what it reads).
     *
     * @param voice a voice id or blend
     * @return {@code [510][256]}
     */
    public float[][] voicePack(String voice) {
        Map<String, Double> parts = parseMix(voice);
        float[][] out = null;
        for (Map.Entry<String, Double> e : parts.entrySet()) {
            float[][] p = pack(e.getKey());
            if (out == null) {
                out = new float[p.length][p[0].length];
            }
            for (int r = 0; r < Math.min(p.length, out.length); r++) {
                for (int c = 0; c < out[r].length; c++) {
                    out[r][c] += (float) (e.getValue() * p[r][c]);
                }
            }
        }
        return out;
    }

    /**
     * Adds a voice, such as a saved blend or a style found elsewhere.
     *
     * @param id its id
     * @param name its display name
     * @param pack style vectors {@code [rows][256]}, one per phoneme count (a single row works for all lengths)
     */
    public void addVoice(String id, String name, float[][] pack) {
        if (id.contains("+") || id.contains(",") || id.contains("*")) {
            throw new IllegalArgumentException("a voice id cannot contain + , or *: " + id);
        }
        int width = 2 * model.config().style();
        for (float[] row : pack) {
            if (row.length != width) {
                throw new IllegalArgumentException("Kokoro style vectors have " + width + " values, got " + row.length);
            }
        }
        packs.put(id, pack.clone());
        voices.removeIf(v -> v.id().equals(id));
        voices.add(new Voice(id, name, "en"));
    }

    /** Splits {@code "a*0.7+b*0.3"} or {@code "a,b"} into normalized weights. */
    static Map<String, Double> parseMix(String voice) {
        Map<String, Double> parts = new LinkedHashMap<>();
        double total = 0;
        for (String part : voice.split("[+,]")) {
            String p = part.strip();
            if (p.isEmpty()) {
                continue;
            }
            int star = Math.max(p.indexOf('*'), p.indexOf(':'));
            String id = star < 0 ? p : p.substring(0, star).strip();
            double w;
            try {
                w = star < 0 ? 1 : Double.parseDouble(p.substring(star + 1).strip());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("bad voice weight in '" + part + "'; write it as voice*0.5");
            }
            if (!(w >= 0)) {
                throw new IllegalArgumentException("voice weights cannot be negative: " + part);
            }
            parts.merge(id, w, Double::sum);
            total += w;
        }
        if (parts.isEmpty() || !(total > 0)) {
            throw new IllegalArgumentException("no voice in '" + voice + "'");
        }
        for (Map.Entry<String, Double> e : parts.entrySet()) {
            e.setValue(e.getValue() / total);
        }
        return parts;
    }

    private float[][] pack(String id) {
        float[][] cached = packs.get(id);
        if (cached != null) {
            return cached;
        }
        Path f = voiceFiles.get(id);
        if (f == null) {
            throw new IllegalArgumentException("unknown voice '" + id + "'; this model has "
                    + voices.stream().map(Voice::id).toList());
        }
        try (WeightSource w = WeightSource.open(f)) {
            String name = w.names().getFirst();
            int width = 2 * model.config().style();
            float[] flat = w.tensor(name, Device.cpu(), DType.FLOAT32).toArray();
            float[][] rows = new float[flat.length / width][width];
            for (int r = 0; r < rows.length; r++) {
                System.arraycopy(flat, r * width, rows[r], 0, width);
            }
            packs.put(id, rows);
            return rows;
        }
    }

    @Override
    public Audio synthesize(String text, SpeechOptions options, Consumer<Audio> onChunk) {
        String voice = options.voice() == null ? voices.getFirst().id() : options.voice();
        float[][] pack = voicePack(voice);
        String first = parseMix(voice).keySet().iterator().next();
        EnglishPhonemizer g2p = first.startsWith("b") ? EnglishPhonemizer.british() : EnglishPhonemizer.american();
        List<String> pieces = chunks(g2p, text);
        List<float[]> parts = new ArrayList<>();
        int total = 0;
        for (int i = 0; i < pieces.size(); i++) {
            if (options.cancelled().getAsBoolean()) {
                throw new CancellationException("speech synthesis cancelled");
            }
            String ps = pieces.get(i);
            int[] ids = model.tokenize(ps);
            if (ids.length == 0) {
                continue;
            }
            float[] style = pack[Math.min(ps.length(), pack.length) - 1];
            float[] wave = model.synthesize(ids, style, options.speed(), new SplittableRandom(options.seed() + i))
                    .audio();
            parts.add(wave);
            total += wave.length;
            if (onChunk != null) {
                onChunk.accept(new Audio(wave, sampleRate()));
            }
        }
        if (parts.isEmpty()) {
            throw new IllegalArgumentException("nothing to say: the text has no words");
        }
        float[] all = new float[total];
        int at = 0;
        for (float[] p : parts) {
            System.arraycopy(p, 0, all, at, p.length);
            at += p.length;
        }
        return new Audio(all, sampleRate());
    }

    /**
     * Phonemes in pieces Kokoro reads at once: a sentence each, longer sentences cut at the
     * last clause break (then comma, then space) before {@link #MAX_PHONEMES}.
     */
    static List<String> chunks(EnglishPhonemizer g2p, String text) {
        List<String> out = new ArrayList<>();
        for (String line : text.split("\\n+")) {
            if (line.isBlank()) {
                continue;
            }
            List<EnglishPhonemizer.Word> words = g2p.words(line);
            StringBuilder cur = new StringBuilder();
            for (EnglishPhonemizer.Word w : words) {
                cur.append(w.phonemes()).append(w.whitespace());
                String ps = w.phonemes();
                boolean end = !ps.isEmpty() && "!.?…".indexOf(ps.charAt(ps.length() - 1)) >= 0;
                if (end) {
                    flush(out, cur);
                }
            }
            flush(out, cur);
        }
        return out;
    }

    private static void flush(List<String> out, StringBuilder cur) {
        String s = cur.toString().strip();
        cur.setLength(0);
        while (s.length() > MAX_PHONEMES) {
            int cut = -1;
            for (String marks : new String[] {":;", ",—", " "}) {
                for (int i = MAX_PHONEMES - 1; i > 0 && cut < 0; i--) {
                    if (marks.indexOf(s.charAt(i)) >= 0) {
                        cut = i + 1;
                    }
                }
                if (cut > 0) {
                    break;
                }
            }
            if (cut <= 0) {
                cut = MAX_PHONEMES;
            }
            out.add(s.substring(0, cut).strip());
            s = s.substring(cut).strip();
        }
        if (!s.isEmpty()) {
            out.add(s);
        }
    }

    @Override
    public void close() {
        model.close();
    }
}
