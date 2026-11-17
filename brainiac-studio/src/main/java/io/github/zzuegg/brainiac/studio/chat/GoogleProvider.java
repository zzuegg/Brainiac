package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** Google models through the Gemini API: Gemini for chat; Imagen, Veo and speech models are listed. */
final class GoogleProvider extends RemoteProvider {

    static final URI BASE = URI.create("https://generativelanguage.googleapis.com/");

    GoogleProvider(Keys keys, URI base) {
        super(keys, base);
    }

    @Override
    public String id() {
        return "google";
    }

    @Override
    String note() {
        return "Gemini, Imagen pictures and Gemini speech";
    }

    @Override
    public String name() {
        return "Google";
    }

    @Override
    List<String> auth(String key) {
        return List.of("x-goog-api-key", key);
    }

    @Override
    List<ModelInfo> models(String key) throws IOException, InterruptedException {
        List<ModelInfo> out = new ArrayList<>();
        String token = null;
        do {
            Map<String, Object> page = getJson(uri("v1beta/models?pageSize=1000"
                    + (token == null ? "" : "&pageToken=" + URLEncoder.encode(token, StandardCharsets.UTF_8))), key);
            for (Object o : arr(page.get("models"))) {
                Map<String, Object> m = obj(o);
                String name = Json.string(m, "name", "");
                String id = name.startsWith("models/") ? name.substring(7) : name;
                List<String> media = media(id, arr(m.get("supportedGenerationMethods")));
                if (!id.isEmpty() && !media.isEmpty()) {
                    long context = num(m.get("inputTokenLimit"));
                    out.add(new ModelInfo(id(), id, Json.string(m, "displayName", id), media, 0,
                            media.contains("text") ? ModelInfo.context(context) : ""));
                }
            }
            token = Json.string(page, "nextPageToken", null);
        } while (token != null && !token.isEmpty());
        return out;
    }

    static List<String> media(String id, List<Object> methods) {
        String s = id.toLowerCase(Locale.ROOT);
        if (s.contains("embedding") || s.contains("aqa") || s.contains("live")) {
            return List.of();
        }
        if (s.startsWith("imagen") || s.contains("image")) {
            return List.of("image");
        }
        if (s.startsWith("veo")) {
            return List.of("video");
        }
        if (s.contains("tts")) {
            return List.of("speech");
        }
        if (methods.contains("generateContent")) {
            return s.startsWith("gemini") ? List.of("text", "vision") : List.of("text");
        }
        return List.of();
    }

    @Override
    public ModelInfo preferred(List<ModelInfo> models) {
        // the newest Gemini Pro by version number, else any Gemini
        return models.stream().filter(ModelInfo::chats).filter(m -> m.id().startsWith("gemini-"))
                .filter(m -> m.id().contains("-pro") && !m.id().contains("preview") && !m.id().contains("exp"))
                .max((a, b) -> compareVersions(a.id(), b.id()))
                .orElseGet(() -> super.preferred(models));
    }

    private static int compareVersions(String a, String b) {
        return Double.compare(version(a), version(b));
    }

    private static double version(String id) {
        String[] p = id.split("-");
        try {
            return p.length > 1 ? Double.parseDouble(p[1]) : 0;
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    @Override
    ChatResult chat(ChatRequest request, ChatListener listener, String key) throws IOException, InterruptedException {
        List<Object> contents = new ArrayList<>();
        for (ChatMessage m : request.messages()) {
            contents.add(Map.of("role", m.role().equals("assistant") ? "model" : "user",
                    "parts", List.of(Map.of("text", m.text()))));
        }
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("contents", contents);
        if (!request.system().isBlank()) {
            body.put("systemInstruction", Map.of("parts", List.of(Map.of("text", request.system()))));
        }
        Map<String, Object> config = new LinkedHashMap<>();
        config.put("maxOutputTokens", request.maxTokens());
        if (request.model().startsWith("gemini-") && version(request.model()) >= 2.5) {
            config.put("thinkingConfig", Map.of("includeThoughts", true));   // summaries of its reasoning
        }
        body.put("generationConfig", config);
        long[] usage = {-1, -1};
        String[] stop = {"end"};
        String model = URLEncoder.encode(request.model(), StandardCharsets.UTF_8);
        stream(uri("v1beta/models/" + model + ":streamGenerateContent?alt=sse"), key, body, (event, data) -> {
            Map<String, Object> e = Json.object(Json.parse(data));
            if (e.get("error") != null) {
                throw new IOException("Google stopped the answer: " + Json.string(obj(e.get("error")), "message", data));
            }
            String blocked = Json.string(obj(e.get("promptFeedback")), "blockReason", null);
            if (blocked != null) {
                stop[0] = "refusal";
            }
            for (Object c : arr(e.get("candidates"))) {
                Map<String, Object> cand = obj(c);
                for (Object p : arr(obj(cand.get("content")).get("parts"))) {
                    Map<String, Object> part = obj(p);
                    if (part.get("text") instanceof String s && !s.isEmpty()) {
                        if (Boolean.TRUE.equals(part.get("thought"))) {
                            listener.thinking(s);
                        } else {
                            listener.text(s);
                        }
                    }
                }
                if (cand.get("finishReason") instanceof String f) {
                    stop[0] = switch (f) {
                        case "STOP" -> "end";
                        case "MAX_TOKENS" -> "length";
                        case "SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII" -> "refusal";
                        default -> f.toLowerCase(Locale.ROOT);
                    };
                }
            }
            Map<String, Object> u = obj(e.get("usageMetadata"));
            if (!u.isEmpty()) {
                usage[0] = num(u.get("promptTokenCount"));
                long out = num(u.get("candidatesTokenCount"));
                long thoughts = num(u.get("thoughtsTokenCount"));
                usage[1] = out < 0 ? -1 : out + Math.max(0, thoughts);
            }
            return true;
        });
        return new ChatResult(stop[0], usage[0], usage[1]);
    }

    // ---------------------------------------------------------------- images and speech

    private static final List<String> VOICES = List.of("Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus",
            "Aoede", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome",
            "Algenib", "Rasalgethi", "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird",
            "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat");

    private URI method(String model, String method) {
        return uri("v1beta/models/" + URLEncoder.encode(model, StandardCharsets.UTF_8) + ":" + method);
    }

    @Override
    List<Media> images(ImageRequest request, String key) throws IOException, InterruptedException {
        List<Media> out = new ArrayList<>();
        if (request.model().startsWith("imagen")) {
            Map<String, Object> body = Map.of("instances", List.of(Map.of("prompt", request.prompt())),
                    "parameters", Map.of("sampleCount", request.count(), "aspectRatio", request.aspect()));
            for (Object o : arr(postJson(method(request.model(), "predict"), key, body).get("predictions"))) {
                Map<String, Object> p = obj(o);
                String b64 = Json.string(p, "bytesBase64Encoded", null);
                if (b64 != null) {
                    out.add(new Media(Base64.getDecoder().decode(b64), Json.string(p, "mimeType", "image/png")));
                }
            }
        } else {
            // Gemini image models draw one picture per call
            for (int c = 0; c < request.count(); c++) {
                Map<String, Object> body = Map.of(
                        "contents", List.of(Map.of("role", "user", "parts", List.of(Map.of("text", request.prompt())))),
                        "generationConfig", Map.of("responseModalities", List.of("IMAGE"),
                                "imageConfig", Map.of("aspectRatio", request.aspect())));
                out.addAll(inline(postJson(method(request.model(), "generateContent"), key, body), "image/"));
            }
        }
        if (out.isEmpty()) {
            throw new IOException("Google sent no picture back (the prompt may have been blocked)");
        }
        return out;
    }

    @Override
    Media speech(SpeechRequest request, String key) throws IOException, InterruptedException {
        // Gemini speech takes its directions in the text itself
        String text = request.instructions().isBlank() ? request.text()
                : request.instructions().strip() + ": " + request.text();
        Map<String, Object> body = Map.of(
                "contents", List.of(Map.of("role", "user", "parts", List.of(Map.of("text", text)))),
                "generationConfig", Map.of("responseModalities", List.of("AUDIO"),
                        "speechConfig", Map.of("voiceConfig",
                                Map.of("prebuiltVoiceConfig", Map.of("voiceName", request.voice())))));
        List<Media> audio = inline(postJson(method(request.model(), "generateContent"), key, body), "audio/");
        if (audio.isEmpty()) {
            throw new IOException("Google sent no audio back");
        }
        Media pcm = audio.getFirst();
        return pcm.mime().toLowerCase(Locale.ROOT).startsWith("audio/l16") ? wav(pcm) : pcm;
    }

    private static List<Media> inline(Map<String, Object> answer, String prefix) {
        List<Media> out = new ArrayList<>();
        for (Object c : arr(answer.get("candidates"))) {
            for (Object p : arr(obj(obj(c).get("content")).get("parts"))) {
                Map<String, Object> data = obj(obj(p).get("inlineData"));
                String mime = Json.string(data, "mimeType", "");
                String b64 = Json.string(data, "data", null);
                if (b64 != null && mime.startsWith(prefix)) {
                    out.add(new Media(Base64.getDecoder().decode(b64), mime));
                }
            }
        }
        return out;
    }

    /** Wraps 16-bit mono PCM ({@code audio/L16;rate=24000}) in a WAV header. */
    static Media wav(Media pcm) {
        int rate = 24000;
        for (String part : pcm.mime().split(";")) {
            String p = part.trim().toLowerCase(Locale.ROOT);
            if (p.startsWith("rate=")) {
                rate = Integer.parseInt(p.substring(5));
            }
        }
        byte[] d = pcm.data();
        ByteBuffer b = ByteBuffer.allocate(44 + d.length).order(ByteOrder.LITTLE_ENDIAN);
        b.put("RIFF".getBytes(StandardCharsets.US_ASCII)).putInt(36 + d.length).put("WAVE".getBytes(StandardCharsets.US_ASCII));
        b.put("fmt ".getBytes(StandardCharsets.US_ASCII)).putInt(16).putShort((short) 1).putShort((short) 1)
                .putInt(rate).putInt(rate * 2).putShort((short) 2).putShort((short) 16);
        b.put("data".getBytes(StandardCharsets.US_ASCII)).putInt(d.length).put(d);
        return new Media(b.array(), "audio/wav");
    }

    @Override
    SpeechOptions speechOptions(String model) {
        return new SpeechOptions(VOICES, false, true, List.of("wav"));
    }

    @Override
    public ModelInfo preferredFor(String medium, List<ModelInfo> models) {
        String want = switch (medium) {
            case "image" -> "imagen-4";
            case "speech" -> "flash";
            default -> null;
        };
        if (want != null) {
            for (ModelInfo m : models) {
                if (m.id().contains(want) && m.media().contains(medium)) {
                    return m;
                }
            }
        }
        return super.preferredFor(medium, models);
    }
}
