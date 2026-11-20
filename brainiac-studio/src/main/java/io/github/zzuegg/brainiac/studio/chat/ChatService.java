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

    private static final System.Logger LOG = System.getLogger(ChatService.class.getName());

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
     * Returns the folder generated pictures, audio and clips are saved in.
     *
     * @return the folder
     */
    public Path outputs() {
        return outputs.toAbsolutePath();
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
     * How a project runs one kind of task.
     *
     * @param model the model Auto means, as {@code provider/id}, or null to pick as usual
     * @param policy what to do when a model on this PC does not fit in GPU memory, or null for the
     *     studio's own choice
     * @param fallback the model ({@code provider/id}, such as one on a worker host or a rented
     *     machine) a task goes to when the policy sends it elsewhere, or null for none
     */
    public record Route(String model, GpuResidency.Policy policy, String fallback) {
    }

    /** What a project allows and how it routes its tasks, for the requests made in it. */
    public interface Scope {

        /** @return the project's name, for messages */
        String name();

        /**
         * Returns the route for a kind of task.
         *
         * @param medium {@code text}, {@code image}, {@code speech}, {@code transcribe} or {@code video}
         * @return the route, or null to run it as the studio does
         */
        Route route(String medium);

        /**
         * Says whether a model may be used.
         *
         * @param key the model, {@code provider/id}
         * @return whether the project allows it
         */
        boolean allows(String key);

        /**
         * Returns a runner a route names with {@code vast:<name>}.
         *
         * @param name the runner's name
         * @return what to rent and fetch, as JSON, or null if there is none
         */
        default Map<String, Object> runner(String name) {
            return null;
        }
    }

    /** Sets a runner up and names the model on it, as {@code provider/id}. */
    @FunctionalInterface
    public interface Runners {
        /**
         * Sets it up.
         *
         * @param scope the project
         * @param name the runner's name
         * @param runner what to rent and fetch
         * @return the model's key on it
         * @throws IOException if it cannot be set up
         * @throws InterruptedException if interrupted while waiting
         */
        String ensure(Scope scope, String name, Map<String, Object> runner) throws IOException, InterruptedException;
    }

    private volatile Runners runners;

    /**
     * Sets up the runners routes name with {@code vast:<name>}.
     *
     * @param runners sets them up
     */
    public void runners(Runners runners) {
        this.runners = runners;
    }

    // a route's fallback: a model key, or a runner that is set up first
    private ModelInfo fallback(String fallback, String medium) throws IOException, InterruptedException {
        if (fallback.startsWith("vast:")) {
            String name = fallback.substring(5);
            Map<String, Object> r = SCOPE.isBound() ? SCOPE.get().runner(name) : null;
            if (r == null || runners == null) {
                throw new IOException("there is no runner " + name + " in this project");
            }
            return resolveAny(runners.ensure(SCOPE.get(), name, r), medium);
        }
        return resolve(fallback, medium);
    }

    /** The project the requests made now belong to; unbound, nothing is limited or routed. */
    public static final ScopedValue<Scope> SCOPE = ScopedValue.newInstance();

    /**
     * Hears the steps of a drawing made through {@link #images}, when a caller such as an agent
     * run wants to show its progress; returning false stops the drawing.
     */
    public static final ScopedValue<LocalImages.Listener> STEPS = ScopedValue.newInstance();

    @FunctionalInterface
    private interface Body<T> {
        T run(ModelInfo target) throws IOException, InterruptedException;
    }

    // runs a task on its model with the project's scheduling policy; when the policy sends it
    // elsewhere and the route names where, it goes there
    private <T> T routed(String medium, ModelInfo target, Body<T> body) throws IOException, InterruptedException {
        Route r = SCOPE.isBound() ? SCOPE.get().route(medium) : null;
        try {
            // a policy chosen for this call (the user's answer to a full GPU) wins over the route's
            if (r == null || r.policy() == null || GpuResidency.POLICY.isBound()) {
                return body.run(target);
            }
            try {
                return ScopedValue.where(GpuResidency.POLICY, r.policy()).call(() -> body.run(target));
            } catch (IOException | InterruptedException | RuntimeException e) {
                throw e;
            } catch (Exception e) {
                throw new IOException(e);
            }
        } catch (GpuResidency.NoRoom e) {
            if (e.policy() == GpuResidency.Policy.REMOTE && r != null && r.fallback() != null && !r.fallback().isBlank()) {
                LOG.log(System.Logger.Level.INFO, "sending the " + medium + " task to " + r.fallback() + ": " + e.getMessage());
                return body.run(fallback(r.fallback(), medium));
            }
            throw e;
        }
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

    private LocalServing serving;
    private LocalQuantizer quantizer;
    private LocalMerger merger;

    /**
     * Returns what merges this PC's chat models into new ones.
     *
     * @return the merger
     */
    public synchronized LocalMerger merger() {
        if (merger == null) {
            merger = new LocalMerger(local());
        }
        return merger;
    }
    private LocalFineTune fineTune;
    private LocalEvaluation evaluation;

    /**
     * Returns what serves one of this PC's chat models to other programs with the OpenAI API.
     *
     * @return it
     */
    public synchronized LocalServing serving() {
        if (serving == null) {
            serving = new LocalServing(local());
        }
        return serving;
    }

    /**
     * Returns what makes smaller copies of this PC's chat models.
     *
     * @return it
     */
    public synchronized LocalQuantizer quantizer() {
        if (quantizer == null) {
            quantizer = new LocalQuantizer(local());
        }
        return quantizer;
    }

    /**
     * Returns what teaches this PC's chat models from examples.
     *
     * @return it
     */
    public synchronized LocalFineTune fineTune() {
        if (fineTune == null) {
            fineTune = new LocalFineTune(local());
        }
        return fineTune;
    }

    /**
     * Returns what measures this PC's chat models, its results kept next to the outputs folder
     * in {@code evaluations.json}.
     *
     * @return it
     */
    public synchronized LocalEvaluation evaluation() {
        if (evaluation == null) {
            evaluation = new LocalEvaluation(local(), outputs.resolveSibling("evaluations.json"));
        }
        return evaluation;
    }

    /**
     * Returns the folders local models are found in, the downloads folder among them.
     *
     * @return the folders
     */
    public List<Path> modelFolders() {
        return local().folders();
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
        {
            Preset o = OllamaProvider.PRESET;
            ps.add(new OllamaProvider(keys, o.at(bases.getOrDefault(o.id(), o.base()))));
        }
        ChatService service = new ChatService(keys, ps, outputs);
        service.hub = new Hub(keys, downloads, bases.getOrDefault("hub", Hub.BASE));
        local.drawToo(() -> service.hub().core());
        local.filmToo();
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
     * Makes a provider of the chat, image and speech models in some folders, as a worker host
     * serves them.
     *
     * @param folders where the models are
     * @param voices where voices made from recordings are kept
     * @return the provider
     */
    public static ChatProvider localMedia(List<Path> folders, Path voices) {
        LocalProvider local = new LocalProvider(folders);
        local.speakToo(voices);
        local.drawToo(io.github.zzuegg.brainiac.io.HuggingFaceHub::standard);
        local.filmToo();
        return local;
    }


    /**
     * Returns the key this studio would use for a provider, to lend it to a worker host.
     *
     * @param provider the provider id, such as {@code huggingface}
     * @return the first of its keys, if it has one
     */
    public java.util.Optional<String> secret(String provider) {
        List<Keys.Entry> list = keys.list(provider);
        return list.isEmpty() ? java.util.Optional.empty() : keys.get(provider, list.getFirst().label());
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
                    if (x.media().contains("video")) {
                        j.put("makesVideo", p.makesVideo());
                    }
                    if (x.media().contains("speech") && p instanceof RemoteProvider r) {
                        j.put("speech", r.speechOptions(x.id()).toJson());
                    } else if (x.media().contains("speech") && p.voiceOptions(x.id()) != null) {
                        j.put("speech", p.voiceOptions(x.id()));
                    }
                    models.add(j);
                }
            }
        }
        Map<String, Object> auto = new LinkedHashMap<>();
        for (String medium : List.of("text", "image", "speech", "video")) {
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
        return chat(model, system, messages, maxTokens, key, temperature, Sampler.NONE, route, listener);
    }

    /**
     * Sends a conversation to a model with its sampling chosen.
     *
     * @param model "auto" or a model's key
     * @param system instructions for the whole conversation, or an empty string
     * @param messages the conversation, ending with the user's turn
     * @param maxTokens the most tokens the answer may have
     * @param key the name of the API key to use, or null to try the provider's keys in order
     * @param temperature from 0 (always the likeliest word) to 2, or null for the model's default
     * @param sampler top-p, top-k and the repeat penalty, each left to the model when not set
     * @param route told which model answers, before the answer starts
     * @param listener receives the answer
     * @return how the answer ended
     * @throws IOException if no model can answer or the provider fails
     * @throws InterruptedException if interrupted while waiting
     */
    public ChatResult chat(String model, String system, List<ChatMessage> messages, int maxTokens, String key,
            Float temperature, Sampler sampler, Consumer<ModelInfo> route, ChatListener listener)
            throws IOException, InterruptedException {
        return chat(model, system, messages, maxTokens, key, temperature, sampler, Format.NONE, route, listener);
    }

    /**
     * Answers a conversation in a given form.
     *
     * @param model a model key ({@code provider/id}) or {@value #AUTO}
     * @param system instructions for the whole conversation, or an empty string
     * @param messages the conversation, ending with the user's turn
     * @param maxTokens the most tokens the answer may have
     * @param key the name of the API key to use, or null to try the provider's keys in order
     * @param temperature from 0 to 2, or null for the model's own default
     * @param sampler top-p, top-k and the repeat penalty, each left to the model when not set
     * @param format the form the answer must have
     * @param route told which model answers, before the answer starts
     * @param listener receives the answer
     * @return how the answer ended
     * @throws IOException if no model can answer, the provider fails or it cannot keep to the format
     * @throws InterruptedException if interrupted while waiting
     */
    public ChatResult chat(String model, String system, List<ChatMessage> messages, int maxTokens, String key,
            Float temperature, Sampler sampler, Format format, Consumer<ModelInfo> route, ChatListener listener)
            throws IOException, InterruptedException {
        return chat(model, system, messages, maxTokens, key, temperature, sampler, format, List.of(), route, listener);
    }

    /**
     * Sends a conversation to a model that may call tools. Providers that take tools natively get
     * them in the request and write their calls back as Hermes {@code <tool_call>} text; every
     * other model is told about the tools in its instructions, in the same format, so the caller
     * reads calls the same way whichever model answers.
     *
     * @param model a model key or {@value #AUTO}
     * @param system instructions
     * @param messages the conversation, tool calls and answers kept as Hermes text
     * @param maxTokens the longest answer
     * @param key the API key's name, or null
     * @param temperature the temperature, or null
     * @param sampler how to pick words
     * @param format the answer's form
     * @param tools {@code {name, description, parameters}} each; empty for none
     * @param route told which model answers
     * @param listener hears the answer
     * @return how it ended
     * @throws IOException if the model fails
     * @throws InterruptedException if interrupted
     */
    public ChatResult chat(String model, String system, List<ChatMessage> messages, int maxTokens, String key,
            Float temperature, Sampler sampler, Format format, List<Map<String, Object>> tools,
            Consumer<ModelInfo> route, ChatListener listener) throws IOException, InterruptedException {
        return routed("text", resolve(model, "text"), target -> {
            ChatProvider p = provider(target.provider());
            if ("grammar".equals(format.kind()) && !p.takesGrammar()) {
                throw new IOException(p.name() + " takes no grammar: use a JSON schema with it, or a model on this PC"
                        + " or a worker host");
            }
            // a JSON answer is asked for in words too: some providers can only be asked
            String instructions = format.kind().equals("json") && !system.toLowerCase(java.util.Locale.ROOT).contains("json")
                    ? (system.isBlank() ? "" : system.strip() + "\n\n") + Format.JSON_HINT : system;
            boolean nativeTools = !tools.isEmpty() && p.nativeTools(target.id());
            if (!tools.isEmpty() && !nativeTools) {
                instructions = (instructions.isBlank() ? "" : instructions.strip() + "\n\n") + ToolCalls.instructions(tools);
            }
            String told = instructions;
            route.accept(target);
            // reasoning models that think in plain text: their <think> block goes to the thinking
            ThinkSplitter split = new ThinkSplitter(listener, false);
            String plainTold = (instructions.isBlank() ? "" : instructions.strip() + "\n\n") + ToolCalls.instructions(tools);
            return logged(target, "chat", () -> {
                ChatResult r;
                try {
                    r = p.chat(new ChatRequest(target.id(), told, messages, maxTokens, key, temperature, sampler,
                            format, nativeTools ? tools : List.of()), split);
                } catch (IOException e) {
                    // a model whose template takes no tools (Ollama says so): tell it about them in words
                    if (!nativeTools || e.getMessage() == null
                            || !e.getMessage().toLowerCase(java.util.Locale.ROOT).contains("support tools")) {
                        throw e;
                    }
                    r = p.chat(new ChatRequest(target.id(), plainTold, messages, maxTokens, key, temperature, sampler,
                            format, List.of()), split);
                }
                split.finish();
                return r;
            }, r -> new long[] {r.inputTokens(), r.outputTokens(), 0});
        });
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
        return pictures(model, prompt, aspect, count, key).generated();
    }

    /**
     * Draws pictures like {@link #images}, and tells the seed of each one drawn on this PC, so it
     * can be drawn again with changes.
     *
     * @param model a model key or {@value #AUTO}
     * @param prompt what the picture shows
     * @param aspect the shape
     * @param count how many, 1 to 4
     * @param key the name of the API key to use, or null
     * @return the pictures; the seeds are empty for a provider that does not tell them
     * @throws IOException if no model can draw or the provider fails
     * @throws InterruptedException if interrupted while waiting
     */
    public Drawn pictures(String model, String prompt, String aspect, int count, String key)
            throws IOException, InterruptedException {
        LocalImages.Listener steps = STEPS.isBound() ? STEPS.get() : (step, total, preview) -> true;
        return routed("image", resolve(model, "image"), target -> {
            if (provider(target.provider()).runsMedia()) {
                return drawHere(target, prompt, LocalImages.Settings.defaults(aspect, count), steps);
            }
            ImageRequest request = new ImageRequest(target.id(), prompt, aspect, count, key);
            List<Media> media = logged(target, "image", () -> remote(target).images(request),
                    r -> new long[] {-1, -1, r.size()});
            return new Drawn(new Generated(target, media, save(media, "image")), List.of());
        });
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
        return routed("image", resolve(model, "image"), target -> {
            if (!provider(target.provider()).runsMedia()) {
                throw new IllegalArgumentException(target.key() + " does not run on this PC or on a worker host");
            }
            route.accept(target);
            return drawHere(target, prompt, settings, listener);
        });
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
        ChatProvider where = provider(target.provider());
        if (where instanceof LocalProvider here && here.videos() != null) {
            here.videos().close();   // a video model and an image model do not fit together
        }
        List<LocalImages.Picture> pictures = logged(target, "image", () -> where.draw(target.id(), prompt, settings, listener),
                r -> new long[] {-1, -1, r.size()});
        List<Media> media = pictures.stream().map(LocalImages.Picture::media).toList();
        return new Drawn(new Generated(target, media, save(media, "image")),
                pictures.stream().map(LocalImages.Picture::seed).toList());
    }

    /**
     * What a video clip run made.
     *
     * @param model the model that made it
     * @param file where the clip was saved
     * @param clip the clip
     */
    public record Filmed(ModelInfo model, Path file, LocalVideos.Clip clip) {
    }

    /**
     * Makes a video clip, step by step, with a video model on this PC (or on a worker host
     * whose provider makes videos), and saves it to the outputs folder as an animated PNG.
     *
     * @param model a video model's key ({@code local/name}) or {@value #AUTO}
     * @param prompt what the clip shows
     * @param settings the clip's shape, length and sampling
     * @param listener hears each step and can stop the clip
     * @param route told which model makes it, before it starts
     * @return the clip
     * @throws IOException if there is no such model or it cannot be loaded
     * @throws InterruptedException if interrupted
     * @throws java.util.concurrent.CancellationException if the listener stopped it
     */
    public Filmed video(String model, String prompt, LocalVideos.Settings settings, LocalVideos.Listener listener,
            Consumer<ModelInfo> route) throws IOException, InterruptedException {
        if (prompt == null || prompt.isBlank()) {
            throw new IllegalArgumentException("describe the video first");
        }
        return routed("video", resolve(model, "video"), target -> {
            ChatProvider p = provider(target.provider());
            if (!p.makesVideo()) {
                throw new IllegalArgumentException(target.key() + " cannot make clips through the studio yet");
            }
            route.accept(target);
            LocalVideos.Clip clip = logged(target, "video", () -> p.video(target.id(), prompt, settings, listener),
                    r -> new long[] {-1, -1, 1});
            return new Filmed(target, save(List.of(clip.media()), "video").get(0), clip);
        });
    }

    /**
     * Describes a video model's usual settings and sizes, for the page.
     *
     * @param model a video model's key
     * @return {@code {kind?, label?, steps?, guidance?, fps?, sizes?}}
     */
    public Map<String, Object> videoOptions(String model) {
        Map<String, Object> out = new LinkedHashMap<>();
        int slash = model == null ? -1 : model.indexOf('/');
        if (slash > 0) {
            // this PC's model, or a worker host's, which is asked
            providers().stream().filter(p -> p.id().equals(model.substring(0, slash)) && p.makesVideo()).findFirst()
                    .map(p -> p.videoOptions(model.substring(slash + 1))).ifPresent(out::putAll);
        }
        return out;
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
        return routed("speech", resolve(model, "speech"), target -> {
            ChatProvider where = provider(target.provider());
            if (where.runsMedia()) {
                Media wav = logged(target, "speech", () -> where.speak(target.id(), text, voice, speed, () -> false),
                        r -> new long[] {-1, -1, text.length()});
                return new Generated(target, List.of(wav), save(List.of(wav), "speech"));
            }
            SpeechRequest request = new SpeechRequest(target.id(), text, voice, speed, instructions, format, key);
            Media audio = logged(target, "speech", () -> remote(target).speech(request),
                    r -> new long[] {-1, -1, text.length()});
            return new Generated(target, List.of(audio), save(List.of(audio), "speech"));
        });
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
        return routed("transcribe", AUTO.equals(model) && routeFor("transcribe") == null ? firstTranscriber() : resolve(model, "transcribe"), target -> {
            ChatProvider where = provider(target.provider());
            if (!where.runsMedia()) {
                throw new IllegalArgumentException(target.key() + " does not transcribe on this PC or on a worker host");
            }
            route.accept(target);
            return logged(target, "transcribe",
                    () -> new Transcribed(target, where.transcribe(target.id(), wav, listening, onSegment, stop)),
                    r -> new long[] {-1, -1, Math.round(r.transcription().seconds())});
        });
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
     * Turns a text into another with a text-to-text model on this PC (BART, Marian, T5): a
     * summary, a translation, or what one of the model's tasks asks for.
     *
     * @param model the model's key ({@code local/bart-large-cnn}), or {@code auto} for the first
     * @param text the input
     * @param task one of the model's tasks, or blank for none
     * @param maxTokens the most tokens to write, or 0 for the task's own limit
     * @return the answer and the model that wrote it
     * @throws IOException if there is no such model or it cannot be loaded
     */
    public Rewritten rewrite(String model, String text, String task, int maxTokens) throws IOException {
        LocalTexts texts = local().texts();
        String id;
        if (model == null || model.isBlank() || AUTO.equals(model)) {
            id = texts.models().stream().map(ModelInfo::id).findFirst().orElseThrow(() -> new IOException(
                    "no text-to-text model is available: put a BART, Marian or T5 model in a model folder"));
        } else if (model.startsWith("local/")) {
            id = model.substring("local/".length());
        } else {
            throw new IllegalArgumentException("only BART, Marian and T5 models on this PC rewrite texts");
        }
        LocalTexts.Rewritten r = texts.rewrite(id, text, task, maxTokens);
        return new Rewritten("local/" + id, r.text(), r.task());
    }

    /**
     * Lists the tasks of a local text-to-text model.
     *
     * @param model the model's key
     * @return the tasks, maybe none
     * @throws IOException if there is no such model
     */
    public List<LocalTexts.Task> rewriteTasks(String model) throws IOException {
        if (model == null || !model.startsWith("local/")) {
            throw new IllegalArgumentException("only BART, Marian and T5 models on this PC rewrite texts");
        }
        return local().texts().tasks(model.substring("local/".length()));
    }

    /**
     * What a text-to-text model wrote.
     *
     * @param model the model's key
     * @param text the answer
     * @param task the task it did, or an empty string
     */
    public record Rewritten(String model, String text, String task) {
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

    /**
     * Whether a task would leave this PC: a provider's API or another machine, which may cost
     * money. Nothing is started to find out; a model that cannot be found counts as paid.
     *
     * @param model a model key or {@value #AUTO}
     * @param medium {@code text}, {@code image}, {@code speech}, {@code transcribe} or {@code video}
     * @return true unless a model on this PC would do it
     */
    public boolean paid(String model, String medium) {
        try {
            Route r = AUTO.equals(model) ? routeFor(medium) : null;
            if (r != null && r.fallback() != null && r.fallback().startsWith("vast:")) {
                return true;   // a full GPU sends it to a rented machine
            }
            ModelInfo t = resolveAny(model == null || model.isBlank() ? AUTO : model, medium);
            return !provider(t.provider()).local();
        } catch (IOException | RuntimeException e) {
            return true;
        }
    }

    /**
     * The context window of a model, as its provider or its config says.
     *
     * @param key the model's key
     * @return tokens, or 0 if unknown
     */
    public long contextOf(String key) {
        if (key == null) {
            return 0;
        }
        for (Listing l : listAll(false).values()) {
            for (ModelInfo m : l.models()) {
                if (m.key().equals(key)) {
                    return m.contextTokens();
                }
            }
        }
        return 0;
    }

    /**
     * Returns the model Auto picks for a kind of task now, in the project of the call: the
     * project's route, or the studio's Auto order, within the models the project allows. For
     * {@code vision} that is Auto's chat model when it sees pictures.
     *
     * @param medium {@code text}, {@code vision}, {@code image}, {@code speech}, …
     * @return the model's key
     * @throws IOException if no model can do it
     */
    public String autoKey(String medium) throws IOException {
        return resolve(AUTO, medium).key();
    }

    private ModelInfo resolve(String model, String medium) throws IOException {
        ModelInfo target = resolveAny(model, medium);
        if (SCOPE.isBound() && !SCOPE.get().allows(target.key())) {
            throw new IllegalArgumentException("the project " + SCOPE.get().name() + " does not allow " + target.key()
                    + ": allow it in the project's settings or pick another model");
        }
        return target;
    }

    // the route's model when Auto is asked for in a project that routes this kind of task
    private Route routeFor(String medium) {
        Route r = SCOPE.isBound() ? SCOPE.get().route(medium) : null;
        return r == null || r.model() == null || r.model().isBlank() ? null : r;
    }

    private ModelInfo resolveAny(String model, String medium) throws IOException {
        ModelInfo target;
        Route routed = AUTO.equals(model) ? routeFor(medium) : null;
        if (routed != null && !AUTO.equals(routed.model())) {
            return resolveAny(routed.model(), medium);
        }
        if (AUTO.equals(model) && medium.equals("vision")) {
            // Auto for a picture is Auto for chat, when that model sees
            try {
                ModelInfo chat = resolveAny(AUTO, "text");
                if (chat.media().contains("vision")) {
                    return chat;
                }
            } catch (IOException | IllegalArgumentException none) {
                // no chat model: a model that only sees may still do
            }
        }
        if (AUTO.equals(model)) {
            target = pickAuto(listAll(false), medium);
            if (target == null) {
                throw new IOException(switch (medium) {
                    case "text" -> "no chat model is available: put a model in a model folder or add an API key";
                    case "image" -> "no image model is available: put a Stable Diffusion or FLUX.1 model in a model folder"
                            + " or add an OpenAI or Google API key";
                    case "transcribe" -> "no transcribing model is available: put a Whisper model in a model folder";
                    case "video" -> "no video model is available: put a Wan 2.1 or 2.2 model in a model folder";
                    case "vision" -> "no model that sees pictures is available: add a vision model (such as Qwen2-VL)"
                            + " or a provider key";
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

    // Auto for a recording: the first transcribing model on this PC, then on the worker hosts
    private ModelInfo firstTranscriber() throws IOException {
        Map<String, Listing> lists = listAll(false);
        for (ChatProvider p : providers()) {
            Listing l = lists.get(p.id());
            if (p.ready() && p.runsMedia() && l != null) {
                for (ModelInfo m : l.models()) {
                    if (m.media().contains("transcribe") && shown(m)) {
                        return m;
                    }
                }
            }
        }
        throw new IOException("no transcribing model is available: put a Whisper model in a model folder");
    }

    private ModelInfo pickAuto(Map<String, Listing> lists, String medium) {
        for (ChatProvider p : providers()) {
            Listing l = lists.get(p.id());
            if (p.ready() && l != null && settings.auto(p.id()) && (!medium.equals("video") || p.makesVideo())) {
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
