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
    private java.util.function.Supplier<List<ChatProvider>> hosts = List::of;
    private final Path outputs;
    private final UsageLog usage;
    private final ProviderSettings settings;
    private Hub hub;
    private VastAi vast;
    private final Conversations conversations;
    private final PageState page;
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
        this.usage = new UsageLog(outputs.resolveSibling("usage.jsonl"));
        this.settings = new ProviderSettings(outputs.resolveSibling("providers.properties"));
        this.conversations = new Conversations(outputs.resolveSibling("chats"));
        this.page = new PageState(outputs.resolveSibling("page.json"));
        for (ChatProvider p : this.providers) {
            if (p instanceof RemoteProvider r) {
                apply(r);
            }
        }
    }

    private void apply(RemoteProvider r) {
        r.base(settings.url(r.id()).orElse(null));
        Map<String, String> o = new LinkedHashMap<>();
        for (String name : r.optionNames()) {
            String v = settings.option(r.id(), name);
            if (!v.isEmpty()) {
                o.put(name, v);
            }
        }
        r.options = Map.copyOf(o);
    }

    /**
     * Changes how a provider is used. Each argument left null stays as it is.
     *
     * @param provider the provider id
     * @param auto whether Auto may pick its models
     * @param models the model ids to show, an empty optional to show all, or null to leave as is
     * @param url another address, an empty string for its own, or null to leave as is
     * @param options options by name, such as {@link ProviderSettings#BILL_TO}; an empty value clears one
     * @throws IOException if the settings cannot be saved
     */
    public void configure(String provider, Boolean auto, java.util.Optional<List<String>> models, String url,
            Map<String, String> options) throws IOException {
        if (!(provider(provider) instanceof RemoteProvider r)) {
            throw new IllegalArgumentException(provider + " has no settings");
        }
        if (options != null) {
            for (String name : options.keySet()) {
                if (!r.optionNames().contains(name)) {
                    throw new IllegalArgumentException(r.name() + " has no option " + name);
                }
            }
        }
        if (auto != null) {
            settings.auto(provider, auto);
        }
        if (models != null) {
            settings.models(provider, models.orElse(null));
        }
        if (url != null) {
            settings.url(provider, url);
            listings.remove(provider);
        }
        if (options != null) {
            for (Map.Entry<String, String> e : options.entrySet()) {
                settings.option(provider, e.getKey(), e.getValue());
            }
        }
        apply(r);
    }

    private boolean shown(ModelInfo m) {
        return settings.models(m.provider()).map(ids -> ids.contains(m.id())).orElse(true);
    }

    /**
     * Returns the chat conversations, kept next to the outputs folder in {@code chats}.
     *
     * @return the conversations
     */
    public Conversations conversations() {
        return conversations;
    }

    /**
     * Returns what the page remembers between visits, kept next to the outputs folder in
     * {@code page.json}.
     *
     * @return the page state
     */
    public PageState page() {
        return page;
    }

    /**
     * Returns the folder everything is kept in: the one that holds the outputs folder.
     *
     * @return the folder
     */
    public Path folder() {
        return outputs.toAbsolutePath().getParent();
    }

    /**
     * Returns the log of every call, kept next to the outputs folder in {@code usage.jsonl}.
     *
     * @return the log
     */
    public UsageLog usage() {
        return usage;
    }

    /**
     * Reads the remaining balance of every key at the providers that report one (DeepSeek,
     * Kimi, OpenRouter).
     *
     * @return per provider id, one entry per key with its balance or error
     * @throws InterruptedException if interrupted while waiting
     */
    public Map<String, Object> balances() throws InterruptedException {
        Map<String, Future<List<Map<String, Object>>>> pending = new LinkedHashMap<>();
        try (ExecutorService pool = Executors.newVirtualThreadPerTaskExecutor()) {
            for (ChatProvider p : providers()) {
                if (p instanceof RemoteProvider r && r.ready()) {
                    pending.put(p.id(), pool.submit(r::balances));
                }
            }
        }
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<String, Future<List<Map<String, Object>>>> e : pending.entrySet()) {
            try {
                List<Map<String, Object>> b = e.getValue().get();
                if (!b.isEmpty()) {
                    out.put(e.getKey(), b);
                }
            } catch (ExecutionException ex) {
                out.put(e.getKey(), List.of(Map.of("error", String.valueOf(ex.getCause().getMessage()))));
            }
        }
        return out;
    }

    /** Runs a call and logs it, whether it works or not. */
    private <T> T logged(ModelInfo m, String kind, Call<T> call, java.util.function.Function<T, long[]> counts)
            throws IOException, InterruptedException {
        long start = System.nanoTime();
        RemoteProvider.LAST_KEY.remove();
        boolean ok = false;
        long[] c = {-1, -1, 0};
        try {
            T result = call.run();
            c = counts.apply(result);
            ok = true;
            return result;
        } finally {
            String key = RemoteProvider.LAST_KEY.get();
            usage.add(new UsageLog.Entry(System.currentTimeMillis(), m.provider(), key == null ? "" : key, m.id(), kind,
                    c[0], c[1], c[2], (System.nanoTime() - start) / 1e9, ok));
        }
    }

    @FunctionalInterface
    private interface Call<T> {
        T run() throws IOException, InterruptedException;
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
     * Adds a place to find models on this PC: a {@code .gguf} file, a Hugging Face model folder, or
     * a folder holding them. It is kept in {@code local-models.txt} in the data folder.
     *
     * @param place the file or folder
     * @return how many models it holds
     * @throws IOException if it holds no kind of model the studio reads, or cannot be saved
     */
    public int addLocal(Path place) throws IOException {
        return local().add(place);
    }

    /**
     * Forgets a place added with {@link #addLocal}; the files stay where they are.
     *
     * @param place the file or folder
     * @return whether it had been added
     * @throws IOException if the list cannot be saved
     */
    public boolean removeLocal(Path place) throws IOException {
        return local().remove(place);
    }

    /**
     * Lists a folder for picking a model place: its subfolders, {@code .gguf} files and model
     * folders. With no folder, lists the home folder, the model folders and the file system roots.
     *
     * @param folder the folder, or null
     * @return {@code {path, parent, entries: [{name, path, kind}]}}, kind being "folder", "gguf" or "model"
     * @throws IOException if the folder cannot be read
     */
    public Map<String, Object> browse(Path folder) throws IOException {
        Map<String, Object> out = new LinkedHashMap<>();
        List<Map<String, Object>> entries = new ArrayList<>();
        if (folder == null) {
            out.put("path", "");
            out.put("parent", null);
            List<Path> starts = new ArrayList<>();
            starts.add(Path.of(System.getProperty("user.home")));
            starts.addAll(local().folders());
            for (Path r : java.nio.file.FileSystems.getDefault().getRootDirectories()) {
                starts.add(r);
            }
            for (Path p : starts) {
                if (Files.isDirectory(p) && entries.stream().noneMatch(e -> e.get("path").equals(p.toString()))) {
                    entries.add(Map.of("name", p.toString(), "path", p.toString(), "kind", LocalProvider.kind(p)));
                }
            }
        } else {
            Path dir = folder.toAbsolutePath().normalize();
            if (!Files.isDirectory(dir)) {
                throw new IOException("no folder at " + dir);
            }
            out.put("path", dir.toString());
            out.put("parent", dir.getParent() == null ? "" : dir.getParent().toString());
            out.put("kind", LocalProvider.kind(dir));
            try (java.util.stream.Stream<Path> s = Files.list(dir)) {
                for (Path p : s.sorted().limit(2000).toList()) {
                    String name = p.getFileName().toString();
                    if (name.startsWith(".") || !Files.isReadable(p)) {
                        continue;
                    }
                    String kind = LocalProvider.kind(p);
                    if (!kind.equals("other")) {
                        entries.add(Map.of("name", name, "path", p.toString(), "kind", kind));
                    }
                }
            }
        }
        out.put("entries", entries);
        return out;
    }

    /**
     * Returns where this PC's models run, which the user can change.
     *
     * @return the choice
     */
    public LocalDevice localDevice() {
        return local().device();
    }

    private LocalProvider local() {
        return providers().stream().filter(LocalProvider.class::isInstance).map(LocalProvider.class::cast).findFirst()
                .orElseThrow(() -> new IllegalStateException("no local provider"));
    }

    /**
     * Creates the service with this PC's models and every remote provider: Anthropic, OpenAI,
     * Google, and the OpenAI-compatible ones in {@link Preset#COMPATIBLE}.
     *
     * @param keys the key store
     * @param modelFolders where to look for local models
     * @param bases other base addresses by provider id (a gateway, a proxy, or a stand-in for
     *     tests); providers not named use their public address
     * @param outputs where generated pictures and audio are saved
     * @return the service
     */
    public static ChatService at(Keys keys, List<Path> modelFolders, Map<String, URI> bases, Path outputs) {
        List<ChatProvider> ps = new ArrayList<>();
        Path downloads = outputs.resolveSibling("downloads");
        List<Path> folders = new ArrayList<>(modelFolders);
        if (!folders.contains(downloads)) {
            folders.add(downloads);
        }
        LocalProvider local = new LocalProvider(folders, outputs.resolveSibling("local-models.txt"));
        local.speakToo(outputs.resolveSibling("voices"));
        ps.add(local);
        ps.add(new AnthropicProvider(keys, bases.getOrDefault("anthropic", AnthropicProvider.BASE)));
        ps.add(new OpenAiProvider(keys, bases.getOrDefault("openai", Preset.OPENAI.base())));
        ps.add(new GoogleProvider(keys, bases.getOrDefault("google", GoogleProvider.BASE)));
        for (Preset p : Preset.COMPATIBLE) {
            ps.add(new CompatibleProvider(keys, p.at(bases.getOrDefault(p.id(), p.base()))));
        }
        ChatService service = new ChatService(keys, ps, outputs);
        service.hub = new Hub(keys, downloads, bases.getOrDefault("hub", Hub.BASE));
        local.drawToo(() -> service.hub().core());
        service.vast = new VastAi(keys, bases.getOrDefault(VastAi.PROVIDER, VastAi.BASE));
        return service;
    }

    /**
     * Returns the Vast.ai client, which reads the user's account and the GPU machines to rent.
     *
     * @return the client
     */
    public VastAi vast() {
        return vast;
    }

    /**
     * Adds the chat providers of the user's other machines, asked for each time the providers are
     * listed so hosts added or removed show at once. They come right after this PC.
     *
     * @param hosts gives the host providers
     */
    public void hosts(java.util.function.Supplier<List<ChatProvider>> hosts) {
        this.hosts = hosts;
    }

    private List<ChatProvider> providers() {
        List<ChatProvider> host = hosts.get();
        if (host.isEmpty()) {
            return providers;
        }
        List<ChatProvider> all = new ArrayList<>();
        providers.stream().filter(ChatProvider::local).forEach(all::add);
        all.addAll(host);
        providers.stream().filter(p -> !p.local()).forEach(all::add);
        return all;
    }

    /**
     * Returns a provider for the language models in some folders, as a worker serves them.
     *
     * @param folders where to look
     * @return the provider
     */
    public static ChatProvider localModels(List<Path> folders) {
        return new LocalProvider(folders);
    }


    /**
     * Returns the Hugging Face Hub client that finds and downloads models into this PC's
     * {@code downloads} folder.
     *
     * @return the client
     */
    public Hub hub() {
        if (hub == null) {
            hub = new Hub(keys, outputs.resolveSibling("downloads"), Hub.BASE);
        }
        return hub;
    }

    /**
     * Returns the ids of the providers that take API keys.
     *
     * @return the ids, in the order Auto tries them
     */
    public List<String> remoteIds() {
        return providers.stream().filter(p -> !p.local()).map(ChatProvider::id).toList();
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
     * Describes the Hugging Face token the downloader sends, without its value.
     *
     * @return {@code {label, source, hint}} of the token used, or null if there is none
     */
    public Map<String, Object> hubToken() {
        List<Keys.Entry> list = keys.list("huggingface");
        if (list.isEmpty()) {
            return null;
        }
        Keys.Entry e = list.getFirst();
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("label", e.label());
        m.put("source", e.source());
        m.put("hint", e.hint());
        m.put("managed", e.label().endsWith("/" + Keys.HF_TOKEN_FILE));
        return m;
    }

    /**
     * Saves a Hugging Face token to {@code ~/.brainiac/hf_token}, which the core's downloader
     * reads too, or deletes that file.
     *
     * @param token the token, or null to delete the file
     * @return whether a file was written or deleted
     * @throws IOException if the file cannot be changed
     */
    public boolean saveHubToken(String token) throws IOException {
        listings.remove("huggingface");
        if (token == null) {
            return keys.removeFile("huggingface");
        }
        keys.saveFile("huggingface", token);
        return true;
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
        for (ChatProvider p : providers()) {
            Listing l = lists.get(p.id());
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", p.id());
            m.put("name", p.name());
            m.put("local", p.local());
            m.put("kind", p.local() ? "local" : p instanceof RemoteProvider ? "api" : "host");
            m.put("ready", p.ready());
            if (p instanceof RemoteProvider r) {
                m.put("keys", r.keyStatus());
                m.put("note", r.note());
                m.put("env", Keys.ENV.getOrDefault(p.id(), List.of()));
                m.put("balance", r.reportsBalance());
                Map<String, Object> st = new LinkedHashMap<>();
                st.put("auto", settings.auto(p.id()));
                st.put("models", settings.models(p.id()).map(List::copyOf).orElse(null));
                st.put("url", settings.url(p.id()).map(URI::toString).orElse(""));
                st.put("home", r.home().toString());
                Map<String, Object> opts = new LinkedHashMap<>();
                for (String name : r.optionNames()) {
                    opts.put(name, settings.option(p.id(), name));
                }
                st.put("options", opts);
                m.put("settings", st);
            }
            m.put("models", l == null ? 0 : l.models().size());
            if (l != null && l.error() != null) {
                m.put("error", l.error());
            }
            if (p instanceof LocalProvider local) {
                m.put("folders", local.folders().stream().map(Path::toString).toList());
                m.put("added", local.added().stream().map(Path::toString).toList());
                m.put("families", List.copyOf(new java.util.TreeSet<>(io.github.zzuegg.brainiac.llm.LlamaConfig.MODEL_TYPES)));
            }
            ps.add(m);
            if (l != null) {
                for (ModelInfo x : l.models()) {
                    Map<String, Object> j = x.toJson();
                    j.put("shown", shown(x));
                    if (x.media().contains("speech") && p instanceof RemoteProvider r) {
                        j.put("speech", r.speechOptions(x.id()).toJson());
                    } else if (x.media().contains("speech") && p instanceof LocalProvider lp && lp.speech() != null) {
                        j.put("speech", lp.speech().options(x.id()).toJson());
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
        return chat(model, system, messages, maxTokens, key, null, route, listener);
    }

    /**
     * Answers a conversation at a chosen temperature.
     *
     * @param model a model key ({@code provider/id}) or {@value #AUTO}
     * @param system instructions for the whole conversation, or an empty string
     * @param messages the conversation, ending with the user's turn
     * @param maxTokens the most tokens the answer may have
     * @param key the name of the API key to use, or null to try the provider's keys in order
     * @param temperature from 0 (always the likeliest word) to 2, or null for the model's default
     * @param route told which model answers, before the answer starts
     * @param listener receives the answer
     * @return how the answer ended
     * @throws IOException if no model can answer or the provider fails
     * @throws InterruptedException if interrupted while waiting
     */
    public ChatResult chat(String model, String system, List<ChatMessage> messages, int maxTokens, String key,
            Float temperature, Consumer<ModelInfo> route, ChatListener listener)
            throws IOException, InterruptedException {
        ModelInfo target = resolve(model, "text");
        ChatProvider p = provider(target.provider());
        route.accept(target);
        // reasoning models that think in plain text: their <think> block goes to the thinking
        ThinkSplitter split = new ThinkSplitter(listener, false);
        return logged(target, "chat", () -> {
            ChatResult r = p.chat(new ChatRequest(target.id(), system, messages, maxTokens, key, temperature), split);
            split.finish();
            return r;
        }, r -> new long[] {r.inputTokens(), r.outputTokens(), 0});
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
        if (provider(target.provider()) instanceof LocalProvider) {
            return drawHere(target, prompt, LocalImages.Settings.defaults(aspect, count), (step, total, preview) -> true)
                    .generated();
        }
        ImageRequest request = new ImageRequest(target.id(), prompt, aspect, count, key);
        List<Media> media = logged(target, "image", () -> remote(target).images(request),
                r -> new long[] {-1, -1, r.size()});
        return new Generated(target, media, save(media, "image"));
    }

    /**
     * What a drawing on this PC made.
     *
     * @param generated the pictures and where they were saved
     * @param seeds each picture's seed
     */
    public record Drawn(Generated generated, List<Long> seeds) {
    }

    /**
     * Draws pictures with a Stable Diffusion model on this PC, step by step, and saves them to the
     * outputs folder.
     *
     * @param model a local image model's key ({@code local/name}) or {@value #AUTO}
     * @param prompt what the pictures show
     * @param settings the settings
     * @param listener hears each step, with rough previews, and can stop the drawing
     * @param route told which model draws, before it starts
     * @return the pictures
     * @throws IOException if there is no such model or it cannot be loaded
     * @throws InterruptedException if interrupted
     * @throws java.util.concurrent.CancellationException if the listener stopped it
     */
    public Drawn draw(String model, String prompt, LocalImages.Settings settings, LocalImages.Listener listener,
            Consumer<ModelInfo> route) throws IOException, InterruptedException {
        if (prompt == null || prompt.isBlank()) {
            throw new IllegalArgumentException("describe the picture first");
        }
        ModelInfo target = resolve(model, "image");
        if (!(provider(target.provider()) instanceof LocalProvider)) {
            throw new IllegalArgumentException(target.key() + " does not run on this PC");
        }
        route.accept(target);
        return drawHere(target, prompt, settings, listener);
    }

    /**
     * Describes the samplers and a local image model's usual settings, for the page.
     *
     * @param model a local image model's key
     * @return {@code {samplers, kind?, steps?, guidance?}}
     */
    public Map<String, Object> drawOptions(String model) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("samplers", LocalImages.SAMPLERS);
        LocalImages images = local().images();
        if (images != null && model != null && model.startsWith("local/")) {
            images.describe(model.substring("local/".length())).ifPresent(out::putAll);
        }
        return out;
    }

    private Drawn drawHere(ModelInfo target, String prompt, LocalImages.Settings settings, LocalImages.Listener listener)
            throws IOException, InterruptedException {
        LocalImages images = local().images();
        if (images == null) {
            throw new IOException("this studio does not draw on this PC");
        }
        List<LocalImages.Picture> pictures = logged(target, "image", () -> images.draw(target.id(), prompt, settings, listener),
                r -> new long[] {-1, -1, r.size()});
        List<Media> media = pictures.stream().map(LocalImages.Picture::media).toList();
        return new Drawn(new Generated(target, media, save(media, "image")),
                pictures.stream().map(LocalImages.Picture::seed).toList());
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
        if (provider(target.provider()) instanceof LocalProvider lp && lp.speech() != null) {
            Media wav = logged(target, "speech", () -> lp.speech().speak(target.id(), text, voice, speed, () -> false),
                    r -> new long[] {-1, -1, text.length()});
            return new Generated(target, List.of(wav), save(List.of(wav), "speech"));
        }
        SpeechRequest request = new SpeechRequest(target.id(), text, voice, speed, instructions, format, key);
        Media audio = logged(target, "speech", () -> remote(target).speech(request),
                r -> new long[] {-1, -1, text.length()});
        return new Generated(target, List.of(audio), save(List.of(audio), "speech"));
    }

    /**
     * What was written down from a recording.
     *
     * @param model the model that listened
     * @param transcription the text, its language and its segments
     */
    public record Transcribed(ModelInfo model, io.github.zzuegg.brainiac.audio.Transcription transcription) {
    }

    /**
     * Writes down what is said in a recording, with a speech model on this PC.
     *
     * @param model a local transcribing model's key ({@code local/whisper-base}) or {@value #AUTO}
     * @param wav the recording, as WAV (any rate; it is resampled)
     * @param listening the language, whether to translate to English, and timestamps
     * @param onSegment hears each piece as it is written down
     * @param route told which model listens, before it starts
     * @param stop says when to stop
     * @return what was said
     * @throws IOException if there is no such model or it cannot be loaded
     * @throws InterruptedException if interrupted
     * @throws java.util.concurrent.CancellationException if stopped
     */
    public Transcribed transcribe(String model, byte[] wav, LocalSpeech.Listening listening,
            Consumer<io.github.zzuegg.brainiac.audio.Transcription.Segment> onSegment, Consumer<ModelInfo> route,
            java.util.function.BooleanSupplier stop) throws IOException, InterruptedException {
        LocalSpeech speech = local().speech();
        // only this PC transcribes here, so Auto means its first transcribing model
        ModelInfo target = AUTO.equals(model) && speech != null
                ? speech.models().stream().filter(m -> m.media().contains("transcribe")).findFirst()
                        .orElseThrow(() -> new IOException("no transcribing model is available: put a Whisper model in a model folder"))
                : resolve(model, "transcribe");
        if (!(provider(target.provider()) instanceof LocalProvider) || speech == null) {
            throw new IllegalArgumentException(target.key() + " does not transcribe on this PC");
        }
        route.accept(target);
        return logged(target, "transcribe",
                () -> new Transcribed(target, speech.transcribe(target.id(), wav, listening, onSegment, stop)),
                r -> new long[] {-1, -1, Math.round(r.transcription().seconds())});
    }

    /**
     * Keeps a voice made from a recording, for a local model that clones voices (F5-TTS). Without a
     * transcript, the first transcribing model on this PC writes it down.
     *
     * @param model the model's key ({@code local/f5-tts})
     * @param name the voice's name
     * @param wav the recording, as WAV
     * @param transcript what it says, or blank to have it written down
     * @return the transcript used
     * @throws IOException if it cannot be saved or written down
     * @throws InterruptedException if interrupted
     */
    public String saveVoice(String model, String name, byte[] wav, String transcript) throws IOException, InterruptedException {
        LocalSpeech speech = local().speech();
        if (speech == null || model == null || !model.startsWith("local/")) {
            throw new IllegalArgumentException("only a speech model on this PC takes a voice from a recording");
        }
        String text = transcript == null ? "" : transcript.strip();
        if (text.isEmpty()) {
            String ears = speech.firstTranscriber().orElseThrow(() -> new IllegalArgumentException(
                    "say what is spoken in the recording, or put a Whisper model in a model folder to write it down"));
            text = speech.transcribe(ears, wav, new LocalSpeech.Listening(null, false, false, ""), s -> { }, () -> false)
                    .text().strip();
            if (text.isEmpty()) {
                throw new IllegalArgumentException("no speech was heard in the recording");
            }
        }
        speech.saveVoice(model.substring("local/".length()), name, wav, text);
        return text;
    }

    /**
     * Forgets a voice made from a recording.
     *
     * @param model the model's key
     * @param name the voice's name
     * @throws IOException if it cannot be deleted
     */
    public void removeVoice(String model, String name) throws IOException {
        LocalSpeech speech = local().speech();
        if (speech == null || model == null || !model.startsWith("local/")) {
            throw new IllegalArgumentException("no saved voice named " + name);
        }
        speech.removeVoice(model.substring("local/".length()), name);
    }

    /**
     * Describes a local transcribing model for the page: the languages it knows and whether it
     * translates. This loads the model.
     *
     * @param model the model's key
     * @return {@code {languages, translate}}
     */
    public Map<String, Object> transcribeOptions(String model) {
        LocalSpeech speech = local().speech();
        if (speech == null || model == null || !model.startsWith("local/")) {
            return Map.of("languages", List.of(), "translate", false);
        }
        return speech.listeningOptions(model.substring("local/".length()));
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
                    case "image" -> "no image model is available: put a Stable Diffusion or FLUX.1 model in a model folder"
                            + " or add an OpenAI or Google API key";
                    case "transcribe" -> "no transcribing model is available: put a Whisper model in a model folder";
                    default -> "no speech model is available: put a Kokoro, F5-TTS, Orpheus or VITS (MMS-TTS) model"
                            + " in a model folder"
                            + " or add an OpenAI or Google API key";
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
        return providers().stream().filter(p -> p.id().equals(id)).findFirst()
                .orElseThrow(() -> new IllegalArgumentException("unknown provider: " + id));
    }

    private ModelInfo pickAuto(Map<String, Listing> lists, String medium) {
        for (ChatProvider p : providers()) {
            Listing l = lists.get(p.id());
            if (p.ready() && l != null && settings.auto(p.id())) {
                ModelInfo m = p.preferredFor(medium, l.models().stream().filter(this::shown).toList());
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
            for (ChatProvider p : providers()) {
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
