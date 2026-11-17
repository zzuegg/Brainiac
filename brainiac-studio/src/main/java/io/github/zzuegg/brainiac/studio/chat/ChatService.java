package io.github.zzuegg.brainiac.studio.chat;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;
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
 * All chat providers behind one door: lists their models, keeps the user's keys, and sends a
 * conversation to the model asked for, or with {@code auto} to the first one that can take
 * it, this PC first.
 */
public final class ChatService {

    /** The model key that lets the studio choose. */
    public static final String AUTO = "auto";

    private static final long REMOTE_LIST_MILLIS = 10 * 60 * 1000;

    private final Keys keys;
    private final List<ChatProvider> providers;
    private final Map<String, Listing> listings = new ConcurrentHashMap<>();

    private record Listing(List<ModelInfo> models, String error, long at) {
    }

    /**
     * Creates the service.
     *
     * @param keys the key store
     * @param providers the providers, in the order {@code auto} tries them
     */
    public ChatService(Keys keys, List<ChatProvider> providers) {
        this.keys = keys;
        this.providers = List.copyOf(providers);
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
     * @return the service
     */
    public static ChatService at(Keys keys, List<Path> modelFolders, URI anthropic, URI openai, URI google) {
        return new ChatService(keys, List.of(new LocalProvider(modelFolders), new AnthropicProvider(keys, anthropic),
                new OpenAiProvider(keys, openai), new GoogleProvider(keys, google)));
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
                l.models().forEach(x -> models.add(x.toJson()));
            }
        }
        Map<String, Object> auto = new LinkedHashMap<>();
        ModelInfo text = pickAuto(lists);
        if (text != null) {
            auto.put("text", text.key());
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
        ModelInfo target;
        if (AUTO.equals(model)) {
            target = pickAuto(listAll(false));
            if (target == null) {
                throw new IOException("no chat model is available: put a model in a model folder or add an API key");
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
                target = new ModelInfo(pid, id, id, List.of("text"), 0, "");
            }
        }
        ChatProvider p = provider(target.provider());
        route.accept(target);
        return p.chat(new ChatRequest(target.id(), system, messages, maxTokens, key), listener);
    }

    private ChatProvider provider(String id) {
        return providers.stream().filter(p -> p.id().equals(id)).findFirst()
                .orElseThrow(() -> new IllegalArgumentException("unknown provider: " + id));
    }

    private ModelInfo pickAuto(Map<String, Listing> lists) {
        for (ChatProvider p : providers) {
            Listing l = lists.get(p.id());
            if (p.ready() && l != null) {
                ModelInfo m = p.preferred(l.models());
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
