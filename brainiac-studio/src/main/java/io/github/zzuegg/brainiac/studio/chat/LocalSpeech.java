package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.audio.Audio;
import io.github.zzuegg.brainiac.audio.F5Speech;
import io.github.zzuegg.brainiac.audio.F5Tts;
import io.github.zzuegg.brainiac.audio.Kokoro;
import io.github.zzuegg.brainiac.audio.OrpheusSpeech;
import io.github.zzuegg.brainiac.audio.SpeechModels;
import io.github.zzuegg.brainiac.audio.SpeechToText;
import io.github.zzuegg.brainiac.audio.TextToSpeech;
import io.github.zzuegg.brainiac.audio.Transcription;
import io.github.zzuegg.brainiac.audio.TranscriptionOptions;
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
 * writes down what is said, and Kokoro, F5-TTS, Orpheus or VITS (MMS-TTS), which speak. F5-TTS speaks
 * in voices made from recordings, which are kept in the studio's data folder. One of each is kept loaded, on the best device, with 32-bit weights.
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

    // family: whisper, kokoro, f5, orpheus or vits
    private record Found(FileTime modified, Optional<SpeechModels.Kind> kind, String family) {
    }

    private static final java.util.regex.Pattern VOICE_NAME = java.util.regex.Pattern.compile("[A-Za-z0-9][A-Za-z0-9 _-]{0,39}");

    private static final Map<Character, String> ACCENTS = Map.of('a', "American", 'b', "British");

    private final Supplier<List<Path>> places;
    private final Path saved;
    private final Map<Path, Found> seen = new ConcurrentHashMap<>();
    private final GpuSlot speaking;
    private final GpuSlot listening;
    private volatile TextToSpeech voice;
    private volatile Path voicePath;
    private volatile SpeechToText ears;
    private volatile Path earsPath;
    private volatile String voiceOn;
    private volatile String earsOn;
    private final LocalDevice where;

    LocalSpeech(Supplier<List<Path>> places) {
        this(places, null);
    }

    /**
     * Finds speech models in some places, keeping voices made from recordings in a folder.
     *
     * @param places the model folders
     * @param saved where voices made from recordings go ({@code <saved>/<model>/<name>.wav} and
     *     {@code .txt}), or null for none
     */
    LocalSpeech(Supplier<List<Path>> places, Path saved) {
        this(places, saved, new LocalDevice(null));
    }

    /**
     * Finds speech models in some places and runs them on the device chosen.
     *
     * @param places the model folders
     * @param saved where voices made from recordings go, or null for none
     * @param where the device this PC's models run on
     */
    LocalSpeech(Supplier<List<Path>> places, Path saved, LocalDevice where) {
        this.places = places;
        this.saved = saved;
        this.where = where;
        this.speaking = new GpuSlot("speech", where.residency());
        this.listening = new GpuSlot("transcribe", where.residency());
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
                    switch (family(e.getValue())) {
                        case "whisper" -> "Whisper speech to text";
                        case "kokoro" -> "Kokoro text to speech";
                        case "f5" -> "F5-TTS voice cloning";
                        case "orpheus" -> "Orpheus text to speech";
                        default -> "VITS text to speech";
                    }));
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
            // the models inside a place, else the place itself: F5-TTS finds its checkpoint and vocoder in
            // the folders around it, so a folder holding one would otherwise pass for one
            candidates = List.of();
            if (Files.isDirectory(place)) {
                try (Stream<Path> s = Files.list(place)) {
                    candidates = s.sorted().filter(Files::isDirectory).filter(p -> kind(p).isPresent()).toList();
                } catch (IOException e) {
                    candidates = List.of();
                }
                if (candidates.isEmpty() && kind(place).isPresent()) {
                    candidates = List.of(place);
                }
            }
            for (Path p : candidates) {
                if (kind(p).isPresent()) {
                    found.putIfAbsent(p.getFileName().toString(), p);
                }
            }
        }
        return found;
    }

    // worked out once, until config.json (or, without one, the folder) changes: F5-TTS has no config.json
    private Optional<SpeechModels.Kind> kind(Path p) {
        FileTime t;
        try {
            Path config = p.resolve("config.json");
            t = Files.getLastModifiedTime(Files.isRegularFile(config) ? config : p);
        } catch (IOException e) {
            return Optional.empty();
        }
        Found f = seen.get(p);
        if (f == null || !f.modified().equals(t)) {
            Optional<SpeechModels.Kind> k;
            String family = "";
            try {
                k = SpeechModels.kind(p);
                if (k.isPresent()) {
                    family = k.get() == SpeechModels.Kind.SPEECH_TO_TEXT ? "whisper"
                            : Kokoro.isKokoro(p) ? "kokoro"
                            : F5Speech.isF5(p) ? "f5"
                            : OrpheusSpeech.isOrpheus(p) ? "orpheus" : "vits";
                }
            } catch (RuntimeException e) {
                k = Optional.empty();
            }
            f = new Found(t, k, family);
            seen.put(p, f);
        }
        return f.kind();
    }

    private String family(Path p) {
        kind(p);
        Found f = seen.get(p);
        return f == null ? "" : f.family();
    }

    // an F5-TTS folder's own voices: name.wav (with name.txt) in voices next to its checkpoint
    private static Map<String, String> f5Voices(Path dir) {
        Map<String, String> out = new LinkedHashMap<>();
        Path ckpt = F5Tts.find(dir);
        if (ckpt == null) {
            return out;
        }
        Path voices = Files.isDirectory(ckpt.resolve("voices")) ? ckpt.resolve("voices")
                : ckpt.getParent() == null ? null : ckpt.getParent().resolve("voices");
        if (voices == null || !Files.isDirectory(voices)) {
            return out;
        }
        try (Stream<Path> s = Files.list(voices)) {
            s.map(f -> f.getFileName().toString()).filter(n -> n.toLowerCase(Locale.ROOT).endsWith(".wav")).sorted()
                    .forEach(n -> out.put(n.substring(0, n.length() - 4), n.substring(0, n.length() - 4)));
        } catch (IOException e) {
            // unreadable: no voices of its own
        }
        return out;
    }

    private Path savedDir(String id) {
        return saved == null ? null : saved.resolve(id);
    }

    /**
     * The voices made from recordings for a model, with what each recording says.
     *
     * @param id the model's name
     * @return name to transcript
     */
    Map<String, String> savedVoices(String id) {
        Map<String, String> out = new LinkedHashMap<>();
        Path dir = savedDir(id);
        if (dir == null || !Files.isDirectory(dir)) {
            return out;
        }
        try (Stream<Path> s = Files.list(dir)) {
            for (Path w : s.filter(f -> f.getFileName().toString().endsWith(".wav")).sorted().toList()) {
                String name = w.getFileName().toString().replaceFirst("\\.wav$", "");
                Path txt = dir.resolve(name + ".txt");
                out.put(name, Files.isRegularFile(txt) ? Files.readString(txt).strip() : "");
            }
        } catch (IOException e) {
            LOG.log(System.Logger.Level.WARNING, "cannot read the saved voices in " + dir, e);
        }
        return out;
    }

    /**
     * Keeps a voice made from a recording, for a model that clones voices.
     *
     * @param id the model's name
     * @param name the voice's name: letters, digits, spaces, {@code _} and {@code -}
     * @param wav the recording, as WAV
     * @param transcript what the recording says
     * @throws IOException if it cannot be saved
     * @throws IllegalArgumentException if the model does not clone, or the name or recording is wrong
     */
    void saveVoice(String id, String name, byte[] wav, String transcript) throws IOException {
        Path p = path(id, SpeechModels.Kind.TEXT_TO_SPEECH);
        if (!"f5".equals(family(p)) || saved == null) {
            throw new IllegalArgumentException(id + " does not speak in voices made from recordings");
        }
        if (name == null || !VOICE_NAME.matcher(name).matches()) {
            throw new IllegalArgumentException("name the voice with up to 40 letters, digits, spaces, _ or -");
        }
        if (transcript == null || transcript.isBlank()) {
            throw new IllegalArgumentException("say what is spoken in the recording");
        }
        Audio a = Wav.decode(wav);   // refuses what is not WAV
        if (a.seconds() < 1) {
            throw new IllegalArgumentException("the recording is too short: use 3 to 12 seconds of speech");
        }
        Path dir = Files.createDirectories(savedDir(id));
        Files.write(dir.resolve(name + ".wav"), wav);
        Files.writeString(dir.resolve(name + ".txt"), transcript.strip());
        speaking.lock();
        try {
            if (voice instanceof F5Speech f5 && p.equals(voicePath)) {
                f5.addVoice(name, a, transcript.strip());
            }
        } finally {
            speaking.unlock();
        }
    }

    /**
     * Forgets a voice made from a recording.
     *
     * @param id the model's name
     * @param name the voice's name
     * @throws IOException if it cannot be deleted
     */
    void removeVoice(String id, String name) throws IOException {
        Path dir = savedDir(id);
        if (dir == null || name == null || !VOICE_NAME.matcher(name).matches()) {
            throw new IllegalArgumentException("no saved voice named " + name);
        }
        boolean gone = Files.deleteIfExists(dir.resolve(name + ".wav"));
        Files.deleteIfExists(dir.resolve(name + ".txt"));
        if (!gone) {
            throw new IllegalArgumentException("no saved voice named " + name);
        }
        speaking.lock();
        try {
            // a loaded model keeps the voice it was given: load it again without it
            if (voice != null && voicePath != null && voicePath.equals(find().get(id))) {
                speaking.forget();
                letVoice();
            }
        } finally {
            speaking.unlock();
        }
    }

    /**
     * Names the first transcribing model, to write down what a recording says.
     *
     * @return its name, or empty if there is none
     */
    Optional<String> firstTranscriber() {
        return find().entrySet().stream().filter(e -> "whisper".equals(family(e.getValue()))).map(Map.Entry::getKey)
                .findFirst();
    }

    // Kokoro's English voices, as KokoroSpeech names them, read from the voices folder without loading the model
    static Map<String, String> kokoroVoices(Path dir) {
        List<String> ids = new ArrayList<>();
        try (Stream<Path> s = Files.list(dir.resolve("voices"))) {
            s.map(f -> f.getFileName().toString()).filter(n -> n.endsWith(".pt"))
                    .map(n -> n.substring(0, n.length() - 3))
                    .filter(id -> id.length() > 3 && id.charAt(2) == '_' && ACCENTS.containsKey(id.charAt(0)))
                    .sorted().forEach(ids::add);
        } catch (IOException e) {
            return Map.of();
        }
        ids.sort((x, y) -> Boolean.compare(!x.equals("af_heart"), !y.equals("af_heart")));
        Map<String, String> out = new LinkedHashMap<>();
        for (String id : ids) {
            String given = id.substring(3, 4).toUpperCase(Locale.ROOT) + id.substring(4);
            String sex = switch (id.charAt(1)) {
                case 'f' -> "female";
                case 'm' -> "male";
                default -> "voice";
            };
            out.put(id, given + " (" + ACCENTS.get(id.charAt(0)) + " " + sex + ")");
        }
        return out;
    }

    private Path path(String id, SpeechModels.Kind want) throws IOException {
        Path p = find().get(id);
        if (p == null || kind(p).orElseThrow() != want) {
            throw new IOException("no " + (want == SpeechModels.Kind.TEXT_TO_SPEECH ? "speaking" : "transcribing")
                    + " model named " + id + " on this PC");
        }
        return p;
    }

    private Device device() {
        return where.device();
    }

    // most speech models are small and checked only in full precision; Orpheus (3B, released in
    // bfloat16) runs in bfloat16 on a GPU that has it, since it would take about 12 GB in float32
    private static DType dtype(Device d) {
        return DType.FLOAT32;
    }

    private static DType dtype(Device d, String family) {
        return "orpheus".equals(family) && d.kind() == Device.Kind.GPU && d.supports(DType.BFLOAT16) ? DType.BFLOAT16
                : dtype(d);
    }

    /**
     * Describes a speaking model's voices, for the page.
     *
     * @param id the model's name
     * @return its options
     */
    SpeechOptions options(String id) {
        // listing the models must not load one: a VITS model's own voices show once it has spoken
        TextToSpeech v = voice;
        Path p = find().get(id);
        if (v != null && p != null && p.equals(voicePath)) {
            Map<String, String> names = new LinkedHashMap<>();
            v.voices().forEach(x -> names.put(x.id(), x.name()));
            return options(names, v.canMixVoices(), v.tags(), v.canClone());
        }
        String family = p == null ? "" : family(p);
        return switch (family) {
            case "kokoro" -> options(kokoroVoices(p), true, List.of(), false);
            case "orpheus" -> {
                Map<String, String> names = new LinkedHashMap<>();
                OrpheusSpeech.VOICES.forEach(x -> names.put(x, x.substring(0, 1).toUpperCase(Locale.ROOT) + x.substring(1)));
                yield options(names, false, OrpheusSpeech.TAGS, false);
            }
            case "f5" -> {
                Map<String, String> names = f5Voices(p);
                savedVoices(id).keySet().forEach(n -> names.put(n, n));
                yield options(names, false, List.of(), true);
            }
            default -> options(Map.of(), false, List.of(), false);
        };
    }

    // tags go to the page without their brackets: <laugh> is offered as laugh
    private static SpeechOptions options(Map<String, String> names, boolean mix, List<String> tags, boolean clone) {
        List<String> voices = names.isEmpty() && !clone ? List.of("default") : List.copyOf(names.keySet());
        List<String> plain = tags.stream().map(t -> t.replaceAll("^<|>$", "")).toList();
        return new SpeechOptions(voices, true, false, List.of("wav"), names, mix, plain, clone);
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
        speaking.lock();
        try {
            TextToSpeech tts = loadVoice(id);
            io.github.zzuegg.brainiac.audio.SpeechOptions o = io.github.zzuegg.brainiac.audio.SpeechOptions.defaults()
                    .speed((float) speed).cancelled(stop);
            if (voiceId != null && !voiceId.isBlank() && (tts.voices().stream().anyMatch(v -> v.id().equals(voiceId))
                    || (tts.canMixVoices() && voiceId.matches("[A-Za-z0-9_]+(\\*[0-9.]+)?([+,][A-Za-z0-9_]+(\\*[0-9.]+)?)+")))) {
                o = o.voice(voiceId);
            }
            Audio a = tts.synthesize(text, o, null);
            return new Media(Wav.encode(a), "audio/wav");
        } finally {
            speaking.unlock();
        }
    }

    private TextToSpeech loadVoice(String id) throws IOException {
        Path p = path(id, SpeechModels.Kind.TEXT_TO_SPEECH);
        Device d = device();
        if (voice != null && p.equals(voicePath) && d.name().equals(voiceOn)) {
            return voice;
        }
        return speaking.load(id, p, d, () -> false, () -> {
            TextToSpeech t = SpeechModels.textToSpeech(p, d, dtype(d, family(p)));
            voice = t;
            voiceOn = d.name();
            voicePath = p;
            if (t instanceof F5Speech f5) {
                for (Map.Entry<String, String> e : savedVoices(id).entrySet()) {
                    f5.addVoice(e.getKey(), Wav.read(savedDir(id).resolve(e.getKey() + ".wav")), e.getValue());
                }
            }
            LOG.log(System.Logger.Level.INFO, "loaded " + p.getFileName() + " to speak on " + d);
            return t;
        }, this::letVoice);
    }

    // lets the speaking model go; its slot is held
    private void letVoice() {
        TextToSpeech t = voice;
        voice = null;
        voicePath = null;
        voiceOn = null;
        if (t != null) {
            t.close();
        }
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
            listening.lock();
            try {
                SpeechToText stt = loadEars(id);
                m.put("languages", stt.languages());
                m.put("translate", stt.canTranslate());
            } finally {
                listening.unlock();
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
        listening.lock();
        try {
            SpeechToText stt = loadEars(id);
            TranscriptionOptions o = TranscriptionOptions.defaults()
                    .language(l.language() == null || l.language().isBlank() ? null : l.language().toLowerCase(Locale.ROOT))
                    .task(l.translate() ? TranscriptionOptions.Task.TRANSLATE : TranscriptionOptions.Task.TRANSCRIBE)
                    .timestamps(l.timestamps()).cancelled(stop);
            if (l.prompt() != null && !l.prompt().isBlank()) {
                o = o.prompt(l.prompt());
            }
            return stt.transcribe(audio, o, onSegment);
        } finally {
            listening.unlock();
        }
    }

    private SpeechToText loadEars(String id) throws IOException {
        Path p = path(id, SpeechModels.Kind.SPEECH_TO_TEXT);
        Device d = device();
        if (ears != null && p.equals(earsPath) && d.name().equals(earsOn)) {
            return ears;
        }
        return listening.load(id, p, d, () -> false, () -> {
            SpeechToText t = SpeechModels.speechToText(p, d, dtype(d));
            ears = t;
            earsOn = d.name();
            earsPath = p;
            LOG.log(System.Logger.Level.INFO, "loaded " + p.getFileName() + " to transcribe on " + d);
            return t;
        }, this::letEars);
    }

    // lets the transcribing model go; its slot is held
    private void letEars() {
        SpeechToText t = ears;
        ears = null;
        earsPath = null;
        earsOn = null;
        if (t != null) {
            t.close();
        }
    }
}
