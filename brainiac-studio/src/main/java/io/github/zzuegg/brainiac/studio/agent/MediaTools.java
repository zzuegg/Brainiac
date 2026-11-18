package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.llm.Tools;
import io.github.zzuegg.brainiac.studio.chat.ChatMessage;
import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.chat.LocalSpeech;
import io.github.zzuegg.brainiac.studio.chat.LocalVideos;
import io.github.zzuegg.brainiac.studio.chat.Media;
import io.github.zzuegg.brainiac.studio.project.Project;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Tools that make and read media: clips, transcripts, descriptions of pictures, and voices. A
 * file is named by its place in the project's folder, or by the full path a tool that made it
 * returned.
 */
public final class MediaTools {

    private static final long MAX_MEDIA = 512L << 20;
    private static final int DESCRIBE_TOKENS = 1024;

    private final ChatService chat;

    /**
     * Creates the tools.
     *
     * @param chat the models
     */
    public MediaTools(ChatService chat) {
        this.chat = chat;
    }

    /**
     * Makes a clip.
     *
     * @param prompt what it shows
     * @param model the model
     * @param seconds how long
     * @param aspect the shape
     * @param start a picture to start from
     * @param saveAs where to copy it in the project
     * @return the clip
     */
    @Tools.Tool(name = "make_video", value = "Makes a short video clip from a description, and saves it. 'auto' uses "
            + "the project's video route. A start picture makes the clip begin from that picture.")
    @Tier(value = Risk.COMPUTE, what = "makes a video clip, which uses the GPU or a paid provider", medium = "video",
            media = true)
    public Map<String, Object> makeVideo(@Tools.Param("what the clip shows and how things move") String prompt,
            @Tools.Param("a model key, or 'auto' for the project's route") Optional<String> model,
            @Tools.Param("how long, 0.5 to 10 seconds; 2 if left out") Optional<Double> seconds,
            @Tools.Param("16:9, 9:16 or 1:1; 16:9 if left out") Optional<String> aspect,
            @Tools.Param("a picture to start from: a file in the project's folder or one a tool made") Optional<String> start,
            @Tools.Param("a place in the project's folder to copy the clip to, such as 'clips/intro.mp4'") Optional<String> saveAs) {
        Media first = null;
        if (start.isPresent() && !start.get().isBlank()) {
            first = picture(media(start.get()));
        }
        LocalVideos.Settings settings = new LocalVideos.Settings("", aspect.orElse("16:9"), false, seconds.orElse(2.0),
                30, 5, new java.util.Random().nextLong(1L << 32), 0, 0, first);
        try {
            ChatService.Filmed f = chat.video(model.orElse(ChatService.AUTO), prompt, settings, (step, total) -> true,
                    m -> { });
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("model", f.model().key());
            out.put("files", List.of(f.file().toString()));
            saveAs.filter(s -> !s.isBlank()).ifPresent(s -> out.put("saved", copy(f.file(), s)));
            return out;
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AgentRuns.Stopped();
        }
    }

    /**
     * Transcribes a recording.
     *
     * @param path the recording
     * @param model the model
     * @param language its language
     * @param timestamps whether to give times
     * @param saveAs where to write the transcript
     * @return the transcript
     */
    @Tools.Tool(name = "transcribe_audio", value = "Writes down what is said in a recording. WAV works always; MP3, "
            + "M4A, video files and others work when FFmpeg is installed. 'auto' uses the project's transcription "
            + "route.")
    @Tier(value = Risk.COMPUTE, what = "transcribes a recording on this PC", medium = "transcribe", resultChars = 16_000)
    public Map<String, Object> transcribeAudio(@Tools.Param("the recording: a file in the project's folder or one a "
            + "tool made") String path,
            @Tools.Param("a transcribing model's key, or 'auto'") Optional<String> model,
            @Tools.Param("the spoken language's code, such as 'de'; found by itself if left out") Optional<String> language,
            @Tools.Param("true for a time on each piece") Optional<Boolean> timestamps,
            @Tools.Param("a text file in the project's folder to write the transcript to") Optional<String> saveAs) {
        Path f = media(path);
        byte[] wav;
        try {
            byte[] head = new byte[12];
            try (var in = Files.newInputStream(f)) {
                int n = in.readNBytes(head, 0, 12);
                boolean riff = n == 12 && new String(head, 0, 4, StandardCharsets.US_ASCII).equals("RIFF")
                        && new String(head, 8, 4, StandardCharsets.US_ASCII).equals("WAVE");
                wav = riff ? Files.readAllBytes(f) : Ffmpeg.wav(f);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
        boolean times = timestamps.orElse(false);
        LocalSpeech.Listening listening = new LocalSpeech.Listening(language.filter(l -> !l.isBlank()).orElse(null),
                false, times, "");
        try {
            ChatService.Transcribed t = chat.transcribe(model.orElse(ChatService.AUTO), wav, listening, seg -> { },
                    m -> { }, () -> false);
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("model", t.model().key());
            out.put("language", t.transcription().language());
            out.put("seconds", t.transcription().seconds());
            String text = t.transcription().text();
            if (times) {
                StringBuilder s = new StringBuilder();
                for (var seg : t.transcription().segments()) {
                    s.append(String.format(Locale.ROOT, "[%s - %s] %s%n", clock(seg.start()), clock(seg.end()),
                            seg.text().strip()));
                }
                text = s.toString();
            }
            if (saveAs.isPresent() && !saveAs.get().isBlank()) {
                Path to = FileTools.project().inFolder(saveAs.get());
                if (Files.exists(to)) {
                    throw new IllegalArgumentException(saveAs.get() + " exists already; pick another name");
                }
                Files.createDirectories(to.getParent());
                Files.writeString(to, text, StandardCharsets.UTF_8);
                out.put("saved", saveAs.get());
            }
            out.put("text", text);
            return out;
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AgentRuns.Stopped();
        }
    }

    private static String clock(double seconds) {
        long s = (long) seconds;
        return String.format(Locale.ROOT, "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60);
    }

    /**
     * Looks at a picture.
     *
     * @param path the picture
     * @param question what to say about it
     * @param model the model
     * @return the answer
     */
    @Tools.Tool(name = "describe_image", value = "Looks at a picture and describes it, or answers a question about "
            + "it, such as whether it matches a brief. Uses a model that sees: the one given, or the first one there is.")
    @Tier(value = Risk.COMPUTE, what = "shows a picture to a model that sees", medium = "text")
    public Map<String, Object> describeImage(@Tools.Param("the picture: a file in the project's folder or one a tool "
            + "made") String path,
            @Tools.Param("what to say about it; a plain description if left out") Optional<String> question,
            @Tools.Param("a model that sees; the first one there is if left out") Optional<String> model) {
        Media picture = picture(media(path));
        String key = model.filter(m -> !m.isBlank() && !m.equals(ChatService.AUTO)).orElseGet(this::seeing);
        StringBuilder answer = new StringBuilder();
        try {
            chat.chat(key, "", List.of(new ChatMessage("user", question.filter(q -> !q.isBlank())
                    .orElse("Describe this picture: what it shows, its style, and any text in it."), List.of(picture))),
                    DESCRIBE_TOKENS, null, m -> { }, answer::append);
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AgentRuns.Stopped();
        }
        return Map.of("model", key, "answer", io.github.zzuegg.brainiac.llm.Agent.content(answer.toString()));
    }

    // the first model that sees, one on this PC first
    private String seeing() {
        Map<String, Object> d = chat.describe(false);
        Map<String, Boolean> local = new LinkedHashMap<>();
        for (Object p : Json.array(d.get("providers"))) {
            Map<String, Object> m = Json.object(p);
            local.put(String.valueOf(m.get("id")), Boolean.TRUE.equals(m.get("local")));
        }
        String remote = null;
        for (Object o : Json.array(d.get("models"))) {
            Map<String, Object> m = Json.object(o);
            if (Boolean.FALSE.equals(m.get("shown")) || !(m.get("media") instanceof List<?> media) || !media.contains("vision")) {
                continue;
            }
            String key = m.get("provider") + "/" + m.get("id");
            Project p = ChatService.SCOPE.isBound() && ChatService.SCOPE.get() instanceof Project pr ? pr : null;
            if (p != null && !p.allows(key)) {
                continue;
            }
            if (local.getOrDefault(String.valueOf(m.get("provider")), false)) {
                return key;
            }
            if (remote == null) {
                remote = key;
            }
        }
        if (remote == null) {
            throw new IllegalArgumentException("no model that sees pictures is available: add a vision model (such as "
                    + "Qwen2-VL) or a provider key");
        }
        return remote;
    }

    /**
     * Keeps a voice.
     *
     * @param model the speech model
     * @param name the voice's name
     * @param path the recording
     * @param transcript what is said in it
     * @param consent whether the speaker agreed
     * @param who whose voice it is and how they agreed
     * @return the voice
     */
    @Tools.Tool(name = "clone_voice", value = "Keeps a voice from a recording of 5 to 15 seconds, for a speech model on "
            + "this PC that clones voices. Only with the speaker's consent: say whose voice it is and how they agreed. "
            + "Never for a voice of someone who did not agree.")
    @Tier(value = Risk.WRITE, what = "keeps a person's voice for speech")
    public Map<String, Object> cloneVoice(@Tools.Param("a local speech model that clones voices, such as 'local/f5-tts'") String model,
            @Tools.Param("a name for the voice") String name,
            @Tools.Param("the recording, WAV, in the project's folder or made by a tool") String path,
            @Tools.Param("what is said in it; written down by itself if left out") Optional<String> transcript,
            @Tools.Param("true only if the speaker agreed to have their voice kept") boolean consent,
            @Tools.Param("whose voice it is and how they agreed, such as 'my own voice' or 'Anna, by email on 3 May'") String who) {
        if (!consent || who == null || who.isBlank()) {
            throw new IllegalArgumentException("a voice is only kept with the speaker's consent: say whose it is and "
                    + "how they agreed");
        }
        Path f = media(path);
        try {
            String saved = chat.saveVoice(model, name, Files.readAllBytes(f), transcript.orElse(""));
            Map<String, Object> record = new LinkedHashMap<>();
            record.put("at", System.currentTimeMillis());
            record.put("model", model);
            record.put("voice", saved);
            record.put("who", who.strip());
            record.put("recording", f.toString());
            Path log = chat.folder().resolve("voice-consent.jsonl");
            Files.writeString(log, Json.write(record) + "\n", StandardCharsets.UTF_8, StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND);
            return Map.of("voice", saved, "consent", "kept in voice-consent.jsonl");
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AgentRuns.Stopped();
        }
    }

    // ---------------------------------------------------------------- files

    /**
     * Finds a media file: a place in the project's folder, or a file the studio made.
     *
     * @param path the place
     * @return the file
     */
    Path media(String path) {
        if (path == null || path.isBlank()) {
            throw new IllegalArgumentException("name the file");
        }
        Path given = Path.of(path);
        Path f;
        if (given.isAbsolute()) {
            Path outputs = chat.outputs().normalize();
            f = given.toAbsolutePath().normalize();
            if (!f.startsWith(outputs)) {
                // the project's folder by its full path is fine too
                f = FileTools.project().inFolder(FileTools.project().folder() == null ? path
                        : FileTools.project().folder().toAbsolutePath().normalize().relativize(f).toString());
            }
        } else {
            f = FileTools.project().inFolder(path);
        }
        if (!Files.isRegularFile(f)) {
            throw new IllegalArgumentException("there is no file " + path);
        }
        try {
            if (Files.size(f) > MAX_MEDIA) {
                throw new IllegalArgumentException(path + " is over 512 MB");
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
        return f;
    }

    static Media picture(Path f) {
        String name = f.getFileName().toString().toLowerCase(Locale.ROOT);
        String mime = name.endsWith(".png") ? "image/png" : name.endsWith(".jpg") || name.endsWith(".jpeg") ? "image/jpeg"
                : name.endsWith(".webp") ? "image/webp" : name.endsWith(".gif") ? "image/gif" : null;
        if (mime == null) {
            throw new IllegalArgumentException(f.getFileName() + " is not a PNG, JPEG, WebP or GIF picture");
        }
        try {
            return new Media(Files.readAllBytes(f), mime);
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
    }

    // copies a made file into the project's folder
    static String copy(Path from, String to) {
        Path t = FileTools.project().inFolder(to);
        if (Files.exists(t)) {
            throw new IllegalArgumentException(to + " exists already; pick another name");
        }
        try {
            Files.createDirectories(t.getParent());
            Files.copy(from, t, StandardCopyOption.COPY_ATTRIBUTES);
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
        return to;
    }

    // keeps ArrayList imported for helpers that gather several files
    static List<String> names(List<Path> files) {
        List<String> out = new ArrayList<>();
        for (Path p : files) {
            out.add(p.toString());
        }
        return out;
    }
}
