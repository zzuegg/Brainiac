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

/** OpenAI models: chat through Chat Completions, pictures, and speech. */
final class OpenAiProvider extends CompatibleProvider {

    OpenAiProvider(Keys keys, URI base) {
        super(keys, Preset.OPENAI.at(base));
    }

    // The list has no capability fields, so the medium comes from the name. Models that need
    // other APIs (realtime, audio chat, embeddings, moderation) are left out.
    @Override
    List<String> media(String id) {
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
            for (Object o : arr(postJson(uri("images/generations"), key, body).get("data"))) {
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
        Media m = postForBytes(uri("audio/speech"), key, body);
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
