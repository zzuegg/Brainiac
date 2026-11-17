package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.audio.Audio;
import io.github.zzuegg.brainiac.audio.SpeechModels;
import io.github.zzuegg.brainiac.audio.SpeechToText;
import io.github.zzuegg.brainiac.audio.TextToSpeech;
import io.github.zzuegg.brainiac.audio.Transcription;
import io.github.zzuegg.brainiac.audio.TranscriptionOptions;
import io.github.zzuegg.brainiac.audio.Voice;
import io.github.zzuegg.brainiac.audio.Wav;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.Stream;

/**
 * The speech models on this PC, found in the same places as the language models: Whisper, which
 * writes down what is said, and VITS (MMS-TTS), which speaks. One of each is kept loaded, on the best device, with 32-bit weights.
 */
public final class LocalSpeech {

    private static final System.Logger LOG = System.getLogger(LocalSpeech.class.getName());

    /**
     * What to write down, beyond the audio.
     *
     * @param language the spoken language's code, such as {@code de}, or null to detect it
     * @param translate true to write it in English
     * @param timestamps true for each segment's start and end
     * @param prompt words that help, such as names and terms, or an empty string
     */
    public record Listening(String language, boolean translate, boolean timestamps, String prompt) {
    }

    private record Found(FileTime modified, Optional<SpeechModels.Kind> kind) {
    }

    private final Supplier<List<Path>> places;
    private final Map<Path, Found> seen = new ConcurrentHashMap<>();
    private final Object speaking = new Object();
    private final Object listening = new Object();
    private TextToSpeech voice;
    private Path voicePath;
    private SpeechToText ears;
    private Path earsPath;

    LocalSpeech(Supplier<List<Path>> places) {
        this.places = places;
    }

    /**
     * Lists the models: speaking ones as {@code speech}, writing-down ones as {@code transcribe}.
     *
     * @return the models
     */
    List<ModelInfo> models() {
        List<ModelInfo> out = new ArrayList<>();
        for (Map.Entry<String, Path> e : find().entrySet()) {
            boolean tts = kind(e.getValue()).orElseThrow() == SpeechModels.Kind.TEXT_TO_SPEECH;
            out.add(new ModelInfo("local", e.getKey(), e.getKey(), List.of(tts ? "speech" : "transcribe"), 0,
                    tts ? "VITS text to speech" : "Whisper speech to text"));
        }
        return out;
    }

    /**
     * Says whether a name is one of these models.
     *
     * @param id the name
     * @return true if it is
     */
    boolean has(String id) {
        return find().containsKey(id);
    }

    private Map<String, Path> find() {
        Map<String, Path> found = new LinkedHashMap<>();
        for (Path place : places.get()) {
            List<Path> candidates;
            if (Files.isRegularFile(place.resolve("config.json"))) {
                candidates = List.of(place);
            } else if (Files.isDirectory(place)) {
                try (Stream<Path> s = Files.list(place)) {
                    candidates = s.sorted().filter(p -> Files.isRegularFile(p.resolve("config.json"))).toList();
                } catch (IOException e) {
                    candidates = List.of();
                }
            } else {
                candidates = List.of();
            }
            for (Path p : candidates) {
                if (kind(p).isPresent()) {
                    found.putIfAbsent(p.getFileName().toString(), p);
                }
            }
        }
        return found;
    }

    // read from config.json once, until it changes
    private Optional<SpeechModels.Kind> kind(Path p) {
        FileTime t;
        try {
            t = Files.getLastModifiedTime(p.resolve("config.json"));
        } catch (IOException e) {
            return Optional.empty();
        }
        Found f = seen.get(p);
        if (f == null || !f.modified().equals(t)) {
            Optional<SpeechModels.Kind> k;
            try {
                k = SpeechModels.kind(p);
            } catch (RuntimeException e) {
                k = Optional.empty();
            }
            f = new Found(t, k);
            seen.put(p, f);
        }
        return f.kind();
    }

    private Path path(String id, SpeechModels.Kind want) throws IOException {
        Path p = find().get(id);
        if (p == null || kind(p).orElseThrow() != want) {
            throw new IOException("no " + (want == SpeechModels.Kind.TEXT_TO_SPEECH ? "speaking" : "transcribing")
                    + " model named " + id + " on this PC");
        }
        return p;
    }

    private static Device device() {
        return Device.best();
    }

    // speech models are small, and 16-bit weights are not checked for them yet, so full precision everywhere
    private static DType dtype(Device d) {
        return DType.FLOAT32;
    }

    /**
     * Describes a speaking model's voices, for the page.
     *
     * @param id the model's name
     * @return its options
     */
    SpeechOptions options(String id) {
        // listing the models must not load one: a model's own voices show once it has spoken
        TextToSpeech v = voice;
        Path p = find().get(id);
        List<String> voices = v != null && p != null && p.equals(voicePath)
                ? v.voices().stream().map(Voice::id).toList() : List.of();
        return new SpeechOptions(voices.isEmpty() ? List.of("default") : voices, true, false, List.of("wav"));
    }

    /**
     * Speaks text.
     *
     * @param id the model's name
     * @param text the words
     * @param voiceId the voice
     * @param speed 1 for normal pace
     * @param stop says when to stop
     * @return the audio, as WAV
     * @throws IOException if the model cannot be loaded
     */
    Media speak(String id, String text, String voiceId, double speed, BooleanSupplier stop) throws IOException {
        synchronized (speaking) {
            TextToSpeech tts = loadVoice(id);
            io.github.zzuegg.brainiac.audio.SpeechOptions o = io.github.zzuegg.brainiac.audio.SpeechOptions.defaults()
                    .speed((float) speed).cancelled(stop);
            if (voiceId != null && !voiceId.isBlank() && tts.voices().stream().anyMatch(v -> v.id().equals(voiceId))) {
                o = o.voice(voiceId);
            }
            Audio a = tts.synthesize(text, o, null);
            return new Media(Wav.encode(a), "audio/wav");
        }
    }

    private TextToSpeech loadVoice(String id) throws IOException {
        Path p = path(id, SpeechModels.Kind.TEXT_TO_SPEECH);
        if (voice != null && p.equals(voicePath)) {
            return voice;
        }
        if (voice != null) {
            voice.close();
            voice = null;
        }
        Device d = device();
        voice = SpeechModels.textToSpeech(p, d, dtype(d));
        voicePath = p;
        LOG.log(System.Logger.Level.INFO, "loaded " + p.getFileName() + " to speak on " + d);
        return voice;
    }

    /**
     * Describes a transcribing model, for the page.
     *
     * @param id the model's name
     * @return {@code {languages, translate}}
     */
    Map<String, Object> listeningOptions(String id) {
        Map<String, Object> m = new LinkedHashMap<>();
        try {
            synchronized (listening) {
                SpeechToText stt = loadEars(id);
                m.put("languages", stt.languages());
                m.put("translate", stt.canTranslate());
            }
        } catch (IOException | RuntimeException e) {
            m.put("languages", List.of());
            m.put("translate", false);
            m.put("error", String.valueOf(e.getMessage()));
        }
        return m;
    }

    /**
     * Writes down what is said in a recording.
     *
     * @param id the model's name
     * @param wav the recording, as WAV
     * @param l how to listen
     * @param onSegment hears each piece as it is written down
     * @param stop says when to stop
     * @return what was said
     * @throws IOException if the model cannot be loaded
     * @throws IllegalArgumentException if the recording is not WAV
     */
    Transcription transcribe(String id, byte[] wav, Listening l, Consumer<Transcription.Segment> onSegment,
            BooleanSupplier stop) throws IOException {
        Audio audio = Wav.decode(wav);
        synchronized (listening) {
            SpeechToText stt = loadEars(id);
            TranscriptionOptions o = TranscriptionOptions.defaults()
                    .language(l.language() == null || l.language().isBlank() ? null : l.language().toLowerCase(Locale.ROOT))
                    .task(l.translate() ? TranscriptionOptions.Task.TRANSLATE : TranscriptionOptions.Task.TRANSCRIBE)
                    .timestamps(l.timestamps()).cancelled(stop);
            if (l.prompt() != null && !l.prompt().isBlank()) {
                o = o.prompt(l.prompt());
            }
            return stt.transcribe(audio, o, onSegment);
        }
    }

    private SpeechToText loadEars(String id) throws IOException {
        Path p = path(id, SpeechModels.Kind.SPEECH_TO_TEXT);
        if (ears != null && p.equals(earsPath)) {
            return ears;
        }
        if (ears != null) {
            ears.close();
            ears = null;
        }
        Device d = device();
        ears = SpeechModels.speechToText(p, d, dtype(d));
        earsPath = p;
        LOG.log(System.Logger.Level.INFO, "loaded " + p.getFileName() + " to transcribe on " + d);
        return ears;
    }
}
