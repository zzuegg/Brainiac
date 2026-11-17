package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** OpenAI models: chat through Chat Completions; image, speech and transcription models are listed. */
final class OpenAiProvider extends RemoteProvider {

    static final URI BASE = URI.create("https://api.openai.com/");

    OpenAiProvider(Keys keys, URI base) {
        super(keys, base);
    }

    @Override
    public String id() {
        return "openai";
    }

    @Override
    public String name() {
        return "OpenAI";
    }

    @Override
    List<String> auth(String key) {
        return List.of("Authorization", "Bearer " + key);
    }

    @Override
    List<ModelInfo> models(String key) throws IOException, InterruptedException {
        List<ModelInfo> out = new ArrayList<>();
        for (Object o : arr(getJson(uri("v1/models"), key).get("data"))) {
            Map<String, Object> m = obj(o);
            String id = Json.string(m, "id", null);
            List<String> media = id == null ? List.of() : media(id);
            if (!media.isEmpty()) {
                out.add(new ModelInfo(id(), id, id, media, Math.max(0, num(m.get("created"))), ""));
            }
        }
        out.sort(Comparator.comparingLong(ModelInfo::created).reversed());
        return out;
    }

    // The list has no capability fields, so the medium comes from the name. Models that need
    // other APIs (realtime, audio chat, embeddings, moderation) are left out.
    static List<String> media(String id) {
        String s = id.toLowerCase(Locale.ROOT);
        if (s.contains("embedding") || s.contains("moderation") || s.contains("realtime") || s.contains("audio")
                || s.contains("search") || s.contains("computer-use") || s.contains("deep-research")
                || s.startsWith("babbage") || s.startsWith("davinci") || s.contains("instruct")) {
            return List.of();
        }
        if (s.contains("tts")) {
            return List.of("speech");
        }
        if (s.contains("transcribe") || s.startsWith("whisper")) {
            return List.of("transcribe");
        }
        if (s.startsWith("dall-e") || s.contains("image")) {
            return List.of("image");
        }
        if (s.startsWith("sora")) {
            return List.of("video");
        }
        if (s.startsWith("gpt-4o") || s.startsWith("gpt-4.1") || s.startsWith("gpt-5") || s.startsWith("o3")
                || s.startsWith("o4") || s.startsWith("chatgpt")) {
            return List.of("text", "vision");
        }
        if (s.startsWith("gpt-") || s.startsWith("o1")) {
            return List.of("text");
        }
        return List.of();
    }

    @Override
    public ModelInfo preferred(List<ModelInfo> models) {
        // newest full-size GPT model; the list is sorted newest first
        return models.stream().filter(ModelInfo::chats)
                .filter(m -> m.id().startsWith("gpt-") && !m.id().contains("mini") && !m.id().contains("nano"))
                .findFirst().orElseGet(() -> super.preferred(models));
    }

    @Override
    ChatResult chat(ChatRequest request, ChatListener listener, String key) throws IOException, InterruptedException {
        List<Object> messages = new ArrayList<>();
        if (!request.system().isBlank()) {
            messages.add(Map.of("role", "system", "content", request.system()));
        }
        for (ChatMessage m : request.messages()) {
            messages.add(Map.of("role", m.role(), "content", m.text()));
        }
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", request.model());
        body.put("messages", messages);
        body.put("max_completion_tokens", request.maxTokens());
        body.put("stream", true);
        body.put("stream_options", Map.of("include_usage", true));
        long[] usage = {-1, -1};
        String[] stop = {"end"};
        stream(uri("v1/chat/completions"), key, body, (event, data) -> {
            if (data.equals("[DONE]")) {
                return false;
            }
            Map<String, Object> e = Json.object(Json.parse(data));
            if (e.get("error") != null) {
                throw new IOException("OpenAI stopped the answer: " + Json.string(obj(e.get("error")), "message", data));
            }
            for (Object c : arr(e.get("choices"))) {
                Map<String, Object> choice = obj(c);
                Map<String, Object> delta = obj(choice.get("delta"));
                if (delta.get("content") instanceof String s && !s.isEmpty()) {
                    listener.text(s);
                }
                if (delta.get("refusal") instanceof String s && !s.isEmpty()) {
                    listener.text(s);
                    stop[0] = "refusal";
                }
                if (choice.get("finish_reason") instanceof String f) {
                    stop[0] = switch (f) {
                        case "stop" -> stop[0];
                        case "length" -> "length";
                        case "content_filter" -> "refusal";
                        default -> f;
                    };
                }
            }
            Map<String, Object> u = obj(e.get("usage"));
            if (!u.isEmpty()) {
                usage[0] = num(u.get("prompt_tokens"));
                usage[1] = num(u.get("completion_tokens"));
            }
            return true;
        });
        return new ChatResult(stop[0], usage[0], usage[1]);
    }

    // ---------------------------------------------------------------- images and speech

    private static final List<String> VOICES = List.of("alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage",
            "shimmer");
    private static final List<String> MORE_VOICES = List.of("ballad", "verse");

    @Override
    List<Media> images(ImageRequest request, String key) throws IOException, InterruptedException {
        String model = request.model();
        boolean dalle2 = model.equals("dall-e-2");
        boolean dalle3 = model.equals("dall-e-3");
        String size = dalle2 ? "1024x1024"
                : request.wide() ? (dalle3 ? "1792x1024" : "1536x1024")
                : request.tall() ? (dalle3 ? "1024x1792" : "1024x1536") : "1024x1024";
        List<Media> out = new ArrayList<>();
        // DALL-E 3 draws one picture per call
        int calls = dalle3 ? request.count() : 1;
        for (int c = 0; c < calls; c++) {
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("model", model);
            body.put("prompt", request.prompt());
            body.put("n", dalle3 ? 1 : request.count());
            body.put("size", size);
            if (model.startsWith("dall-e")) {
                body.put("response_format", "b64_json");   // GPT image models always answer with base64
            }
            for (Object o : arr(postJson(uri("v1/images/generations"), key, body).get("data"))) {
                String b64 = Json.string(obj(o), "b64_json", null);
                if (b64 != null) {
                    out.add(new Media(Base64.getDecoder().decode(b64), "image/png"));
                }
            }
        }
        if (out.isEmpty()) {
            throw new IOException("OpenAI sent no picture back");
        }
        return out;
    }

    @Override
    Media speech(SpeechRequest request, String key) throws IOException, InterruptedException {
        SpeechOptions options = speechOptions(request.model());
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", request.model());
        body.put("input", request.text());
        body.put("voice", request.voice());
        body.put("response_format", request.format());
        if (request.speed() != 1) {
            body.put("speed", request.speed());
        }
        if (options.instructions() && !request.instructions().isBlank()) {
            body.put("instructions", request.instructions());
        }
        Media m = postForBytes(uri("v1/audio/speech"), key, body);
        String mime = switch (request.format()) {
            case "wav" -> "audio/wav";
            case "opus" -> "audio/ogg";
            default -> "audio/mpeg";
        };
        return m.mime().startsWith("audio/") ? m : new Media(m.data(), mime);
    }

    @Override
    SpeechOptions speechOptions(String model) {
        // the older tts-1 models have fewer voices and no instructions
        boolean old = model.startsWith("tts-1");
        List<String> voices = new ArrayList<>(VOICES);
        if (!old) {
            voices.addAll(MORE_VOICES);
        }
        return new SpeechOptions(voices, true, !old, List.of("mp3", "wav", "opus"));
    }

    @Override
    public ModelInfo preferredFor(String medium, List<ModelInfo> models) {
        String want = switch (medium) {
            case "image" -> "gpt-image";
            case "speech" -> "gpt-4o-mini-tts";
            default -> null;
        };
        if (want != null) {
            for (ModelInfo m : models) {
                if (m.id().startsWith(want) && m.media().contains(medium)) {
                    return m;
                }
            }
        }
        return super.preferredFor(medium, models);
    }
}
