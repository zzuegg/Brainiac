package io.github.zzuegg.brainiac.studio.chat;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Consumer;

/**
 * All model providers behind one door: lists their models, keeps the user's keys, and sends a
 * conversation, a picture to draw or text to speak to the model asked for, or with
 * {@code auto} to the first one that can take it, this PC first. Pictures and audio are also
 * saved to the outputs folder.
 */
public final class ChatService {

    /** The model key that lets the studio choose. */
    public static final String AUTO = "auto";

    private static final long REMOTE_LIST_MILLIS = 10 * 60 * 1000;

    private final Keys keys;
    private final List<ChatProvider> providers;
    private final Path outputs;
    private final Map<String, Listing> listings = new ConcurrentHashMap<>();

    private record Listing(List<ModelInfo> models, String error, long at) {
    }

    /**
     * Creates the service.
     *
     * @param keys the key store
     * @param providers the providers, in the order {@code auto} tries them
     * @param outputs where generated pictures and audio are saved
     */
    public ChatService(Keys keys, List<ChatProvider> providers, Path outputs) {
        this.keys = keys;
        this.providers = List.copyOf(providers);
        this.outputs = outputs;
    }

    /**
     * What a generation made.
     *
     * @param model the model that made it
     * @param media the pictures or audio
     * @param files where each one was saved
     */
    public record Generated(ModelInfo model, List<Media> media, List<Path> files) {
    }

    /**
     * Creates the service with this PC's models and the Anthropic, OpenAI and Google APIs at the
     * given addresses (the public ones, or a proxy or gateway in front of them).
     *
     * @param keys the key store
     * @param modelFolders where to look for local models
     * @param anthropic the Anthropic API's base address
     * @param openai the OpenAI API's base address
     * @param google the Gemini API's base address
     * @param outputs where generated pictures and audio are saved
     * @return the service
     */
    public static ChatService at(Keys keys, List<Path> modelFolders, URI anthropic, URI openai, URI google,
            Path outputs) {
        return new ChatService(keys, List.of(new LocalProvider(modelFolders), new AnthropicProvider(keys, anthropic),
                new OpenAiProvider(keys, openai), new GoogleProvider(keys, google)), outputs);
    }

    /**
     * Saves a key under a name (replacing one of the same name), and forgets the provider's
     * model list so the next one includes what the key can see.
     *
     * @param provider the provider id
     * @param label the key's name, such as {@code work}
     * @param key the key
     * @throws IOException if the key cannot be saved
     */
    public void addKey(String provider, String label, String key) throws IOException {
        keys.add(provider, label, key);
        listings.remove(provider);
    }

    /**
     * Removes a saved key.
     *
     * @param provider the provider id
     * @param label the key's name
     * @return true if there was one
     * @throws IOException if the change cannot be saved
     */
    public boolean removeKey(String provider, String label) throws IOException {
        listings.remove(provider);
        return keys.remove(provider, label);
    }

    /**
     * Makes a saved key the one tried first.
     *
     * @param provider the provider id
     * @param label the key's name
     * @throws IOException if the change cannot be saved
     */
    public void makeKeyFirst(String provider, String label) throws IOException {
        keys.makeFirst(provider, label);
    }

    /**
     * Describes the providers and their models, and what {@code auto} picks for chat.
     *
     * @param refresh true to ask remote providers again instead of using lists up to ten minutes old
     * @return the description, ready to write as JSON
     */
    public Map<String, Object> describe(boolean refresh) {
        Map<String, Listing> lists = listAll(refresh);
        List<Object> ps = new ArrayList<>();
        List<Object> models = new ArrayList<>();
        for (ChatProvider p : providers) {
            Listing l = lists.get(p.id());
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", p.id());
            m.put("name", p.name());
            m.put("local", p.local());
            m.put("ready", p.ready());
            if (p instanceof RemoteProvider r) {
                m.put("keys", r.keyStatus());
            }
            m.put("models", l == null ? 0 : l.models().size());
            if (l != null && l.error() != null) {
                m.put("error", l.error());
            }
            if (p instanceof LocalProvider local) {
                m.put("folders", local.folders().stream().map(Path::toString).toList());
            }
            ps.add(m);
            if (l != null) {
                for (ModelInfo x : l.models()) {
                    Map<String, Object> j = x.toJson();
                    if (x.media().contains("speech") && p instanceof RemoteProvider r) {
                        j.put("speech", r.speechOptions(x.id()).toJson());
                    }
                    models.add(j);
                }
            }
        }
        Map<String, Object> auto = new LinkedHashMap<>();
        for (String medium : List.of("text", "image", "speech")) {
            ModelInfo m = pickAuto(lists, medium);
            if (m != null) {
                auto.put(medium, m.key());
            }
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("providers", ps);
        out.put("models", models);
        out.put("auto", auto);
        return out;
    }

    /**
     * Answers a conversation.
     *
     * @param model a model key ({@code provider/id}) or {@value #AUTO}
     * @param system instructions for the whole conversation, or an empty string
     * @param messages the conversation, ending with the user's turn
     * @param maxTokens the most tokens the answer may have
     * @param key the name of the API key to use, or null to try the provider's keys in order
     * @param route told which model answers, before the answer starts
     * @param listener receives the answer
     * @return how the answer ended
     * @throws IOException if no model can answer or the provider fails
     * @throws InterruptedException if interrupted while waiting
     */
    public ChatResult chat(String model, String system, List<ChatMessage> messages, int maxTokens, String key,
            Consumer<ModelInfo> route, ChatListener listener) throws IOException, InterruptedException {
        ModelInfo target = resolve(model, "text");
        ChatProvider p = provider(target.provider());
        route.accept(target);
        return p.chat(new ChatRequest(target.id(), system, messages, maxTokens, key), listener);
    }

    /**
     * Draws pictures and saves them to the outputs folder.
     *
     * @param model a model key ({@code provider/id}) or {@value #AUTO}
     * @param prompt what the picture shows
     * @param aspect {@code 1:1}, {@code 4:3}, {@code 3:4}, {@code 16:9} or {@code 9:16}
     * @param count how many, 1 to 4
     * @param key the name of the API key to use, or null to try the provider's keys in order
     * @return the pictures
     * @throws IOException if no model can draw or the provider fails
     * @throws InterruptedException if interrupted while waiting
     */
    public Generated images(String model, String prompt, String aspect, int count, String key)
            throws IOException, InterruptedException {
        ModelInfo target = resolve(model, "image");
        ImageRequest request = new ImageRequest(target.id(), prompt, aspect, count, key);
        List<Media> media = remote(target).images(request);
        return new Generated(target, media, save(media, "image"));
    }

    /**
     * Speaks text and saves the audio to the outputs folder.
     *
     * @param model a model key ({@code provider/id}) or {@value #AUTO}
     * @param text the words
     * @param voice the voice's name
     * @param speed 1 for normal pace
     * @param instructions how to say it, or an empty string
     * @param format {@code mp3}, {@code wav} or {@code opus}
     * @param key the name of the API key to use, or null to try the provider's keys in order
     * @return the audio
     * @throws IOException if no model can speak or the provider fails
     * @throws InterruptedException if interrupted while waiting
     */
    public Generated speech(String model, String text, String voice, double speed, String instructions, String format,
            String key) throws IOException, InterruptedException {
        ModelInfo target = resolve(model, "speech");
        Media audio = remote(target).speech(new SpeechRequest(target.id(), text, voice, speed, instructions, format, key));
        return new Generated(target, List.of(audio), save(List.of(audio), "speech"));
    }

    private RemoteProvider remote(ModelInfo m) {
        if (provider(m.provider()) instanceof RemoteProvider r) {
            return r;
        }
        throw new IllegalArgumentException(m.key() + " cannot do that on this PC yet");
    }

    private List<Path> save(List<Media> media, String kind) throws IOException {
        Files.createDirectories(outputs);
        String stamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
        List<Path> files = new ArrayList<>();
        for (int i = 0; i < media.size(); i++) {
            Media m = media.get(i);
            Path f = outputs.resolve(kind + "-" + stamp + "-" + (i + 1) + "." + m.extension());
            for (int n = 2; Files.exists(f); n++) {
                f = outputs.resolve(kind + "-" + stamp + "-" + (i + 1) + "-" + n + "." + m.extension());
            }
            Files.write(f, m.data());
            files.add(f);
        }
        return files;
    }

    private ModelInfo resolve(String model, String medium) throws IOException {
        ModelInfo target;
        if (AUTO.equals(model)) {
            target = pickAuto(listAll(false), medium);
            if (target == null) {
                throw new IOException(switch (medium) {
                    case "text" -> "no chat model is available: put a model in a model folder or add an API key";
                    case "image" -> "no image model is available: add an OpenAI or Google API key";
                    default -> "no speech model is available: add an OpenAI or Google API key";
                });
            }
        } else {
            int slash = model.indexOf('/');
            if (slash <= 0) {
                throw new IllegalArgumentException("a model is named provider/id, such as anthropic/claude-opus-5-5");
            }
            String pid = model.substring(0, slash);
            String id = model.substring(slash + 1);
            Listing l = listings.get(pid);
            target = l == null ? null : l.models().stream().filter(m -> m.id().equals(id)).findFirst().orElse(null);
            if (target == null) {
                provider(pid);   // fails for an unknown provider
                target = new ModelInfo(pid, id, id, List.of(medium), 0, "");
            }
        }
        return target;
    }

    private ChatProvider provider(String id) {
        return providers.stream().filter(p -> p.id().equals(id)).findFirst()
                .orElseThrow(() -> new IllegalArgumentException("unknown provider: " + id));
    }

    private ModelInfo pickAuto(Map<String, Listing> lists, String medium) {
        for (ChatProvider p : providers) {
            Listing l = lists.get(p.id());
            if (p.ready() && l != null) {
                ModelInfo m = p.preferredFor(medium, l.models());
                if (m != null) {
                    return m;
                }
            }
        }
        return null;
    }

    // Asks every ready provider for its models at once; a failing provider reports its error.
    private Map<String, Listing> listAll(boolean refresh) {
        long now = System.currentTimeMillis();
        Map<String, Future<Listing>> pending = new LinkedHashMap<>();
        try (ExecutorService pool = Executors.newVirtualThreadPerTaskExecutor()) {
            for (ChatProvider p : providers) {
                Listing cached = listings.get(p.id());
                boolean fresh = cached != null && !p.local() && cached.error() == null
                        && now - cached.at() < REMOTE_LIST_MILLIS;
                if (!p.ready()) {
                    listings.remove(p.id());
                } else if (!fresh || refresh) {
                    pending.put(p.id(), pool.submit(() -> list(p)));
                }
            }
        }
        for (Map.Entry<String, Future<Listing>> e : pending.entrySet()) {
            try {
                listings.put(e.getKey(), e.getValue().get());
            } catch (InterruptedException ex) {
                Thread.currentThread().interrupt();
                break;
            } catch (ExecutionException ex) {
                listings.put(e.getKey(), new Listing(List.of(), String.valueOf(ex.getCause().getMessage()), now));
            }
        }
        return Map.copyOf(listings);
    }

    private static Listing list(ChatProvider p) {
        try {
            return new Listing(p.models(), null, System.currentTimeMillis());
        } catch (IOException | RuntimeException e) {
            return new Listing(List.of(), String.valueOf(e.getMessage()), System.currentTimeMillis());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return new Listing(List.of(), "interrupted", System.currentTimeMillis());
        }
    }
}
