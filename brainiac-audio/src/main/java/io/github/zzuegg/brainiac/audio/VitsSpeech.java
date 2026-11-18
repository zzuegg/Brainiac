package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.function.Consumer;

/**
 * VITS text-to-speech (MMS-TTS and others): the text is split into sentences, each
 * spoken by {@link Vits} and handed out as soon as it is ready, with a short pause
 * between sentences.
 *
 * <pre>{@code
 * try (VitsSpeech tts = VitsSpeech.load(Path.of("mms-tts-eng"), Device.cpu(), DType.FLOAT32)) {
 *     Wav.write(tts.synthesize("Hello. How are you?"), Path.of("hello.wav"));
 * }
 * }</pre>
 */
public final class VitsSpeech implements TextToSpeech {

    /** Silence between sentences, in seconds. */
    private static final double PAUSE = 0.2;
    /** Longest sentence piece in characters; longer ones are split at spaces. */
    private static final int MAX_CHARS = 400;

    private final Vits model;
    private final VitsTokenizer tokenizer;
    private final List<Voice> voices;

    /**
     * Combines a model and its tokenizer.
     *
     * @param model the network
     * @param tokenizer its tokenizer
     * @param name the model's name, for the voice names
     */
    public VitsSpeech(Vits model, VitsTokenizer tokenizer, String name) {
        this.model = model;
        this.tokenizer = tokenizer;
        String lang = languageCode(tokenizer.language());
        List<Voice> v = new ArrayList<>();
        int speakers = model.config().speakers();
        if (speakers > 1 && model.config().speakerEmbedding() > 0) {
            for (int s = 0; s < speakers; s++) {
                v.add(new Voice(String.valueOf(s), name + " speaker " + s, lang));
            }
        } else {
            v.add(new Voice("default", name, lang));
        }
        this.voices = List.copyOf(v);
    }

    /**
     * Loads a Hugging Face VITS folder with its tokenizer.
     *
     * @param dir the folder
     * @param device where the model runs
     * @param dtype the type of its weights
     * @return the text-to-speech model
     */
    public static VitsSpeech load(Path dir, Device device, DType dtype) {
        Vits model = Vits.load(dir, device, dtype);
        VitsTokenizer tok = VitsTokenizer.load(dir);
        Path name = dir.toAbsolutePath().normalize().getFileName();
        return new VitsSpeech(model, tok, name == null ? "vits" : name.toString());
    }

    /**
     * Returns the network.
     *
     * @return the model
     */
    public Vits model() {
        return model;
    }

    @Override
    public int sampleRate() {
        return model.config().sampleRate();
    }

    @Override
    public List<Voice> voices() {
        return voices;
    }

    @Override
    public Audio synthesize(String text, SpeechOptions options, Consumer<Audio> onChunk) {
        int speaker = 0;
        if (options.voice() != null) {
            Voice voice = voices.stream().filter(v -> v.id().equals(options.voice())).findFirst()
                    .orElseThrow(() -> new IllegalArgumentException("unknown voice '" + options.voice()
                            + "'; this model has " + voices.stream().map(Voice::id).toList()));
            speaker = voices.size() > 1 ? Integer.parseInt(voice.id()) : 0;
        }
        Vits.Sampling base = Vits.Sampling.of(model.config());
        List<String> sentences = sentences(text);
        List<float[]> parts = new ArrayList<>();
        int total = 0;
        int pause = (int) Math.round(PAUSE * sampleRate());
        boolean spoke = false;
        for (int i = 0; i < sentences.size(); i++) {
            if (options.cancelled().getAsBoolean()) {
                throw new CancellationException("speech synthesis cancelled");
            }
            int[] ids = tokenizer.encode(sentences.get(i));
            if (ids.length == 0) {
                continue;
            }
            Vits.Sampling s = new Vits.Sampling(base.noiseScale(), base.noiseScaleDuration(),
                    base.speakingRate() * options.speed(), speaker, options.seed() + i);
            float[] wave = model.synthesize(ids, s);
            if (spoke) {
                float[] withPause = new float[pause + wave.length];
                System.arraycopy(wave, 0, withPause, pause, wave.length);
                wave = withPause;
            }
            spoke = true;
            parts.add(wave);
            total += wave.length;
            if (onChunk != null) {
                onChunk.accept(new Audio(wave, sampleRate()));
            }
        }
        if (!spoke) {
            throw new IllegalArgumentException("nothing to say: the text has no characters this voice can speak");
        }
        float[] all = new float[total];
        int at = 0;
        for (float[] p : parts) {
            System.arraycopy(p, 0, all, at, p.length);
            at += p.length;
        }
        return new Audio(all, sampleRate());
    }

    /** Splits text after sentence ends and line breaks; long sentences at spaces. */
    static List<String> sentences(String text) {
        List<String> out = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            cur.append(c);
            boolean end = c == '\n' || (c == '.' || c == '!' || c == '?' || c == ';' || c == '。')
                    && (i + 1 == text.length() || Character.isWhitespace(text.charAt(i + 1)));
            if (end || cur.length() >= MAX_CHARS && Character.isWhitespace(c)) {
                add(out, cur);
            }
        }
        add(out, cur);
        return out;
    }

    private static void add(List<String> out, StringBuilder cur) {
        String s = cur.toString().strip();
        if (!s.isEmpty()) {
            out.add(s);
        }
        cur.setLength(0);
    }

    /** MMS's ISO 639-3 codes to the two-letter codes Whisper and the studio use, where common. */
    private static String languageCode(String code) {
        if (code == null) {
            return null;
        }
        return switch (code) {
            case "eng" -> "en";
            case "deu" -> "de";
            case "fra" -> "fr";
            case "spa" -> "es";
            case "ita" -> "it";
            case "por" -> "pt";
            case "nld" -> "nl";
            case "pol" -> "pl";
            case "rus" -> "ru";
            case "ukr" -> "uk";
            case "ron" -> "ro";
            case "hun" -> "hu";
            case "fin" -> "fi";
            case "swe" -> "sv";
            case "tur" -> "tr";
            case "vie" -> "vi";
            default -> code;
        };
    }

    @Override
    public void close() {
        model.close();
    }
}
