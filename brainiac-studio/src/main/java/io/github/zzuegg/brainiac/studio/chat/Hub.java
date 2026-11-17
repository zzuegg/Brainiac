package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.io.Gguf;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.llm.LlamaConfig;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ConnectException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpConnectTimeoutException;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Finds language models on the Hugging Face Hub, says which of their files the studio can run,
 * and downloads them into the data folder's {@code downloads} folder, where the studio finds
 * them like any other model on this PC.
 *
 * <p>A model can come as a whole Hugging Face folder ({@code config.json}, the tokenizer and
 * {@code .safetensors} weights) of a family the core loads, or as one {@code .gguf} file of an
 * architecture and quantization it reads. Those come from the core ({@link LlamaConfig#MODEL_TYPES},
 * {@link LlamaConfig#GGUF_ARCHITECTURES} and {@link Gguf#formats()}), never from a list here.
 * Downloads resume where they stopped, are checked against the Hub's SHA-256, and use the
 * Hugging Face key, when there is one, for gated models.
 */
public final class Hub {

    /** The Hub's address. */
    public static final URI BASE = URI.create("https://huggingface.co/");

    // What runs is asked from the core, so a family or quantization it learns shows up here unchanged.
    /** The Hugging Face model families the core loads from a model folder. */
    static final Set<String> FAMILIES = LlamaConfig.MODEL_TYPES;
    /** The GGUF architectures the core loads. */
    static final Set<String> GGUF_ARCHITECTURES = LlamaConfig.GGUF_ARCHITECTURES;
    /** The GGUF tensor types the core reads. */
    static final Set<String> QUANTS = Gguf.formats();

    private static final Pattern QUANT = Pattern.compile(
            "(?i)(?<![a-z0-9])(I?Q\\d+_[0-9A-Z]+(?:_[A-Z0-9]+)*|TQ\\d_\\d|MXFP4|BF16|F16|F32)(?![a-z0-9])");
    private static final Pattern SPLIT = Pattern.compile("(?i)-(\\d{5})-of-(\\d{5})\\.gguf$");
    private static final Set<String> FOLDER_FILES = Set.of("config.json", "generation_config.json",
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
            "model.safetensors.index.json");

    private final Keys keys;
    private final Path folder;
    private final HttpClient http;
    private volatile URI base;
    private final Map<String, Download> downloads = new ConcurrentHashMap<>();
    private final AtomicInteger ids = new AtomicInteger();

    /**
     * Creates the hub client.
     *
     * @param keys where the Hugging Face key is, for gated models
     * @param folder where downloads go
     * @param base the Hub's address, or a stand-in
     */
    public Hub(Keys keys, Path folder, URI base) {
        this.keys = keys;
        this.folder = folder;
        this.base = base.toString().endsWith("/") ? base : URI.create(base + "/");
        this.http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(15))
                .followRedirects(HttpClient.Redirect.NEVER).build();
    }

    /**
     * Returns where downloads go.
     *
     * @return the folder
     */
    public Path folder() {
        return folder;
    }

    // ---------- finding models ----------

    /**
     * Searches the Hub for text generation models the studio runs, most downloaded first. Each
     * model's family (its {@code config.json} model type, or its GGUF architecture) comes with the
     * search, and only the families the core loads are kept.
     *
     * @param query words in the name, such as "qwen2.5 0.5b"
     * @return {@code {models: [{id, downloads, likes, gguf, safetensors, family, gated, parameters,
     *     context, updated}], hidden}}: hidden counts the models found that do not run
     * @throws IOException if the Hub cannot be reached
     */
    public Map<String, Object> search(String query) throws IOException {
        StringBuilder q = new StringBuilder("api/models?limit=100&sort=downloads&direction=-1&pipeline_tag=text-generation");
        if (!query.isBlank()) {
            q.append("&search=").append(URLEncoder.encode(query.strip(), StandardCharsets.UTF_8));
        }
        for (String e : List.of("downloads", "likes", "tags", "lastModified", "gated", "config", "gguf")) {
            q.append("&expand%5B%5D=").append(e);
        }
        List<Map<String, Object>> out = new ArrayList<>();
        int hidden = 0;
        for (Object o : Json.array(Json.parse(text(base.resolve(q.toString()))))) {
            Map<String, Object> m = Json.object(o);
            List<String> tags = Json.array(m.getOrDefault("tags", List.of())).stream().map(String::valueOf).toList();
            Map<String, Object> gguf = m.get("gguf") instanceof Map<?, ?> g ? Json.object(g) : Map.of();
            Map<String, Object> config = m.get("config") instanceof Map<?, ?> c ? Json.object(c) : Map.of();
            String arch = Json.string(gguf, "architecture", "");
            String type = Json.string(config, "model_type", "");
            boolean hasGguf = !gguf.isEmpty() || tags.contains("gguf");
            boolean runs = (!arch.isEmpty() && GGUF_ARCHITECTURES.contains(arch))
                    || (!type.isEmpty() && FAMILIES.contains(type) && tags.contains("safetensors"));
            if (!runs) {
                hidden++;
                continue;
            }
            if (out.size() >= 40) {
                continue;
            }
            Map<String, Object> r = new LinkedHashMap<>();
            r.put("id", Json.string(m, "id", Json.string(m, "modelId", "")));
            r.put("downloads", m.getOrDefault("downloads", 0));
            r.put("likes", m.getOrDefault("likes", 0));
            r.put("gguf", !arch.isEmpty() && GGUF_ARCHITECTURES.contains(arch));
            r.put("safetensors", FAMILIES.contains(type) && tags.contains("safetensors"));
            r.put("family", arch.isEmpty() ? type : arch);
            Object gated = m.get("gated");
            r.put("gated", gated != null && !Boolean.FALSE.equals(gated));
            r.put("parameters", gguf.get("total") instanceof Number n ? n.longValue() : 0L);
            r.put("context", gguf.get("context_length") instanceof Number n ? n.longValue() : 0L);
            r.put("updated", Json.string(m, "lastModified", Json.string(m, "createdAt", "")));
            out.add(r);
        }
        Map<String, Object> res = new LinkedHashMap<>();
        res.put("models", out);
        res.put("hidden", hidden);
        return res;
    }

    /**
     * Says what the core runs, for the page: model folder families, GGUF architectures and
     * GGUF quantizations.
     *
     * @return {@code {families, architectures, quantizations}}, each sorted
     */
    public Map<String, Object> runs() {
        return Map.of("families", List.copyOf(new java.util.TreeSet<>(FAMILIES)),
                "architectures", List.copyOf(new java.util.TreeSet<>(GGUF_ARCHITECTURES)),
                "quantizations", List.copyOf(QUANTS));
    }

    /**
     * Lists what can be downloaded from a repository: each {@code .gguf} file, and the whole
     * model folder when the repository has one, each with its size and whether the studio can
     * run it, and why not.
     *
     * @param repo the repository, such as "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
     * @return {@code {repo, family, options: [{id, kind, name, size, quant, supported, reason, have}]}}
     * @throws IOException if the Hub cannot be reached or has no such repository
     */
    public Map<String, Object> files(String repo) throws IOException {
        checkRepo(repo);
        List<Map<String, Object>> tree = tree(repo);
        Map<String, Map<String, Object>> byPath = new LinkedHashMap<>();
        for (Map<String, Object> f : tree) {
            if ("file".equals(f.get("type"))) {
                byPath.put(Json.string(f, "path", ""), f);
            }
        }
        List<Map<String, Object>> options = new ArrayList<>();
        // the whole folder, when it is a transformers model
        String family = "";
        if (byPath.containsKey("config.json")) {
            try {
                family = Json.string(Json.object(Json.parse(text(fileUri(repo, "config.json")))), "model_type", "");
            } catch (IOException | RuntimeException e) {
                family = "";
            }
            List<String> weights = byPath.keySet().stream().filter(p -> !p.contains("/") && p.endsWith(".safetensors"))
                    .toList();
            if (!weights.isEmpty()) {
                List<String> files = new ArrayList<>(weights);
                byPath.keySet().stream().filter(FOLDER_FILES::contains).forEach(files::add);
                long size = files.stream().mapToLong(p -> size(byPath.get(p))).sum();
                Map<String, Object> o = option("folder", "folder", repoName(repo), size);
                o.put("files", files);
                o.put("family", family);
                if (!FAMILIES.contains(family)) {
                    unsupported(o, family.isEmpty() ? "no model_type in config.json"
                            : "the " + family + " family does not run in the studio yet; "
                                    + String.join(", ", new java.util.TreeSet<>(FAMILIES)) + " do");
                } else if (!byPath.containsKey("tokenizer.json")) {
                    unsupported(o, "it has no tokenizer.json");
                }
                o.put("have", Files.isRegularFile(folder.resolve(repoName(repo)).resolve("config.json")));
                options.add(o);
            }
        }
        // each GGUF file; the architecture is read from the start of the first one
        List<String> ggufs = byPath.keySet().stream().filter(p -> p.toLowerCase(Locale.ROOT).endsWith(".gguf")).toList();
        String arch = null;
        String archError = null;
        Optional<String> probe = ggufs.stream().filter(p -> !p.toLowerCase(Locale.ROOT).contains("mmproj")).findFirst();
        if (probe.isPresent()) {
            try {
                arch = ggufArchitecture(head(fileUri(repo, probe.get()), 1 << 20));
            } catch (IOException e) {
                archError = e.getMessage();
            }
        }
        for (String p : ggufs) {
            String name = p.substring(p.lastIndexOf('/') + 1);
            Map<String, Object> o = option(p, "gguf", name, size(byPath.get(p)));
            String quant = quant(name);
            o.put("quant", quant);
            o.put("architecture", arch == null ? "" : arch);
            Matcher split = SPLIT.matcher(name);
            if (name.toLowerCase(Locale.ROOT).contains("mmproj")) {
                unsupported(o, "an image projector for a vision model, not a language model");
            } else if (split.find()) {
                unsupported(o, "split into " + Integer.parseInt(split.group(2)) + " parts; the studio reads one-file GGUFs");
            } else if (!quant.isEmpty() && !QUANTS.contains(quantFamily(quant))) {
                unsupported(o, quant + " is a quantization the studio cannot read yet; it reads "
                        + String.join(", ", QUANTS.stream().filter(f -> f.startsWith("Q") || f.startsWith("F")
                                || f.startsWith("BF")).toList()));
            } else if (arch != null && !GGUF_ARCHITECTURES.contains(arch)) {
                unsupported(o, "the " + arch + " architecture does not run in the studio yet; "
                        + String.join(" and ", new java.util.TreeSet<>(GGUF_ARCHITECTURES)) + " do");
            } else if (arch == null && archError != null) {
                o.put("reason", "could not read its architecture (" + archError + "); it may not run");
            }
            o.put("have", Files.isRegularFile(folder.resolve(name)));
            options.add(o);
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("repo", repo);
        out.put("family", family.isEmpty() ? arch == null ? "" : arch : family);
        out.put("options", options);
        return out;
    }

    private static Map<String, Object> option(String id, String kind, String name, long size) {
        Map<String, Object> o = new LinkedHashMap<>();
        o.put("id", id);
        o.put("kind", kind);
        o.put("name", name);
        o.put("size", size);
        o.put("supported", true);
        o.put("reason", "");
        return o;
    }

    private static void unsupported(Map<String, Object> o, String why) {
        o.put("supported", false);
        o.put("reason", why);
    }

    /**
     * Reads the quantization from a GGUF file name, such as "Q4_K_M" from "qwen2.5-7b-instruct-q4_k_m.gguf".
     *
     * @param name the file name
     * @return the quantization in capitals, or "" if the name does not say
     */
    static String quant(String name) {
        Matcher m = QUANT.matcher(name.replaceAll("(?i)\\.gguf$", ""));
        String last = "";
        while (m.find()) {
            last = m.group(1).toUpperCase(Locale.ROOT);
        }
        return last;
    }

    // Q4_K_M and Q4_K_S are both Q4_K; Q4_0_4_4 is a repacked layout the studio does not read
    private static String quantFamily(String quant) {
        Matcher k = Pattern.compile("^(Q\\d_K)(_[SML]|_XL)?$").matcher(quant);
        return k.matches() ? k.group(1) : quant;
    }

    /**
     * Reads {@code general.architecture} from the start of a GGUF file.
     *
     * @param head the file's first bytes
     * @return the architecture, such as "qwen2"
     * @throws IOException if it is not a GGUF file or the key is not in those bytes
     */
    static String ggufArchitecture(byte[] head) throws IOException {
        ByteBuffer b = ByteBuffer.wrap(head).order(ByteOrder.LITTLE_ENDIAN);
        try {
            if (b.getInt() != 0x46554747) {
                throw new IOException("not a GGUF file");
            }
            int version = b.getInt();
            if (version < 2) {
                throw new IOException("GGUF version " + version + " is too old");
            }
            b.getLong();   // tensors
            long kv = b.getLong();
            for (long i = 0; i < kv; i++) {
                String key = string(b);
                int type = b.getInt();
                if (key.equals("general.architecture") && type == 8) {
                    return string(b);
                }
                skip(b, type);
            }
        } catch (java.nio.BufferUnderflowException e) {
            throw new IOException("its architecture is not near the start of the file");
        }
        throw new IOException("it does not say its architecture");
    }

    private static String string(ByteBuffer b) {
        long n = b.getLong();
        if (n < 0 || n > b.remaining()) {
            throw new java.nio.BufferUnderflowException();
        }
        byte[] s = new byte[(int) n];
        b.get(s);
        return new String(s, StandardCharsets.UTF_8);
    }

    private static void skip(ByteBuffer b, int type) {
        switch (type) {
            case 0, 1, 7 -> b.position(b.position() + 1);
            case 2, 3 -> b.position(b.position() + 2);
            case 4, 5, 6 -> b.position(b.position() + 4);
            case 10, 11, 12 -> b.position(b.position() + 8);
            case 8 -> string(b);
            case 9 -> {
                int t = b.getInt();
                long n = b.getLong();
                for (long i = 0; i < n; i++) {
                    skip(b, t);
                }
            }
            default -> throw new java.nio.BufferUnderflowException();
        }
    }

    // ---------- downloading ----------

    /** One download: a GGUF file or a model folder. */
    public static final class Download {
        private final String id;
        private final String repo;
        private final String name;
        private final String kind;
        private final String option;
        private final long total;
        private final long started = System.currentTimeMillis();
        private volatile long done;
        private volatile double speed;
        private volatile String state = "queued";
        private volatile String error = "";
        private volatile boolean cancelled;
        private volatile Thread thread;

        Download(String id, String repo, String name, String kind, String option, long total) {
            this.id = id;
            this.repo = repo;
            this.name = name;
            this.kind = kind;
            this.option = option;
            this.total = total;
        }

        /**
         * Returns the download's state, such as "downloading" or "done".
         *
         * @return the state
         */
        public String state() {
            return state;
        }

        Map<String, Object> toJson() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", id);
            m.put("repo", repo);
            m.put("name", name);
            m.put("kind", kind);
            m.put("option", option);
            m.put("total", total);
            m.put("done", done);
            m.put("speed", Math.round(speed));
            m.put("state", state);
            m.put("error", error);
            m.put("started", started);
            m.put("model", "local/" + name);
            return m;
        }
    }

    /**
     * Lists the downloads since the studio started, newest first.
     *
     * @return them
     */
    public List<Map<String, Object>> downloads() {
        return downloads.values().stream().sorted((a, b) -> Long.compare(b.started, a.started)).map(Download::toJson)
                .toList();
    }

    /**
     * Starts downloading one option from {@link #files}.
     *
     * @param repo the repository
     * @param option the option's id: a GGUF file's path, or "folder"
     * @return the download
     * @throws IOException if the option is not there, does not run in the studio, or does not fit on the disk
     */
    public synchronized Map<String, Object> download(String repo, String option) throws IOException {
        checkRepo(repo);
        for (Download d : downloads.values()) {
            if (d.repo.equals(repo) && d.kind.equals(option.equals("folder") ? "folder" : "gguf")
                    && (d.state.equals("queued") || d.state.equals("downloading") || d.state.equals("checking"))
                    && (option.equals("folder") || d.name.equals(option.substring(option.lastIndexOf('/') + 1)))) {
                return d.toJson();
            }
        }
        // a stopped or failed try of the same file makes way for this one, which resumes it
        downloads.values().removeIf(d -> d.repo.equals(repo) && d.option.equals(option)
                && (d.state.equals("stopped") || d.state.equals("failed")));
        Map<String, Object> listing = files(repo);
        Map<String, Object> o = Json.array(listing.get("options")).stream().map(Json::object)
                .filter(x -> option.equals(x.get("id"))).findFirst()
                .orElseThrow(() -> new IOException(repo + " has no " + option));
        if (!Boolean.TRUE.equals(o.get("supported"))) {
            throw new IOException("the studio cannot run " + o.get("name") + ": " + o.get("reason"));
        }
        long size = ((Number) o.get("size")).longValue();
        Files.createDirectories(folder);
        long free = Files.getFileStore(folder).getUsableSpace();
        if (size > free) {
            throw new IOException(String.format(Locale.ROOT, "it needs %.1f GB and %s has %.1f GB free", size / 1e9,
                    folder, free / 1e9));
        }
        List<Map<String, Object>> tree = tree(repo);
        String name = (String) o.get("name");
        Download d = new Download(String.valueOf(ids.incrementAndGet()), repo, name, (String) o.get("kind"), option, size);
        downloads.put(d.id, d);
        List<String> files = o.get("files") instanceof List<?> l ? l.stream().map(String::valueOf).toList()
                : List.of(option);
        d.thread = Thread.ofVirtual().name("hub-" + d.id).start(() -> run(d, files, tree));
        return d.toJson();
    }

    /**
     * Stops a download; what it fetched so far is kept, so starting it again resumes.
     *
     * @param id the download's id
     * @return whether it was running
     */
    public boolean cancel(String id) {
        Download d = downloads.get(id);
        if (d == null || !(d.state.equals("queued") || d.state.equals("downloading"))) {
            return false;
        }
        d.cancelled = true;
        Thread t = d.thread;
        if (t != null) {
            t.interrupt();
        }
        return true;
    }

    /**
     * Deletes a downloaded model from the downloads folder.
     *
     * @param name the file or folder name, as the model is called on this PC
     * @return whether it was there
     * @throws IOException if it cannot be deleted
     */
    public boolean delete(String name) throws IOException {
        if (name.isBlank() || name.contains("/") || name.contains("\\") || name.startsWith(".")) {
            return false;
        }
        Path p = folder.resolve(name).normalize();
        if (!p.getParent().equals(folder.normalize()) || !Files.exists(p)) {
            return false;
        }
        if (Files.isDirectory(p)) {
            try (var s = Files.walk(p)) {
                for (Path x : s.sorted(java.util.Comparator.reverseOrder()).toList()) {
                    Files.delete(x);
                }
            }
        } else {
            Files.delete(p);
        }
        return true;
    }

    private void run(Download d, List<String> files, List<Map<String, Object>> tree) {
        Map<String, Map<String, Object>> byPath = new LinkedHashMap<>();
        tree.forEach(f -> byPath.put(Json.string(f, "path", ""), f));
        boolean folderKind = d.kind.equals("folder");
        Path target = folder.resolve(d.name);
        Path work = folderKind ? folder.resolve("." + d.name + ".partial") : folder;
        try {
            Files.createDirectories(work);
            d.state = "downloading";
            long before = 0;
            for (String file : files) {
                Map<String, Object> info = byPath.getOrDefault(file, Map.of());
                Path out = folderKind ? work.resolve(file) : work.resolve(d.name);
                if (Files.isRegularFile(out) && Files.size(out) == size(info)) {
                    before += size(info);
                    d.done = before;
                    continue;
                }
                fetch(d, file, out, info, before);
                before += size(info);
            }
            if (folderKind) {
                if (Files.exists(target)) {
                    delete(d.name);
                }
                Files.move(work, target, StandardCopyOption.ATOMIC_MOVE);
            }
            d.done = d.total;
            d.state = "done";
        } catch (InterruptedException | java.io.InterruptedIOException | java.nio.channels.ClosedByInterruptException e) {
            d.state = d.cancelled ? "stopped" : "failed";
            d.error = d.cancelled ? "" : "interrupted";
        } catch (IOException | RuntimeException e) {
            if (d.cancelled) {
                d.state = "stopped";
            } else {
                d.state = "failed";
                d.error = String.valueOf(e.getMessage());
            }
        }
    }

    // fetches one file into out, resuming from out.part, and checks its SHA-256 when the Hub gives one
    private void fetch(Download d, String file, Path out, Map<String, Object> info, long before)
            throws IOException, InterruptedException {
        Files.createDirectories(out.toAbsolutePath().getParent());
        Path part = out.resolveSibling(out.getFileName() + ".part");
        long have = Files.isRegularFile(part) ? Files.size(part) : 0;
        long size = size(info);
        if (size > 0 && have > size) {
            Files.delete(part);
            have = 0;
        }
        if (size == 0 || have < size) {
            HttpResponse<InputStream> r = open(fileUri(d.repo, file), have);
            if (r.statusCode() == 200 && have > 0) {
                have = 0;   // the server sent the whole file again
            }
            long last = System.nanoTime();
            long lastBytes = have;
            try (InputStream in = r.body();
                 OutputStream o = Files.newOutputStream(part, StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                         have > 0 ? StandardOpenOption.APPEND : StandardOpenOption.TRUNCATE_EXISTING)) {
                byte[] buf = new byte[1 << 16];
                int n;
                long got = have;
                while ((n = in.read(buf)) > 0) {
                    if (d.cancelled || Thread.currentThread().isInterrupted()) {
                        throw new InterruptedException();
                    }
                    o.write(buf, 0, n);
                    got += n;
                    d.done = before + got;
                    long now = System.nanoTime();
                    if (now - last > 500_000_000L) {
                        double s = (got - lastBytes) / ((now - last) / 1e9);
                        d.speed = d.speed == 0 ? s : d.speed * 0.6 + s * 0.4;
                        last = now;
                        lastBytes = got;
                    }
                }
            } catch (java.io.InterruptedIOException e) {
                throw e;
            } catch (IOException e) {
                if (d.cancelled) {
                    throw new InterruptedException();
                }
                throw new IOException("the connection dropped at " + mb(Files.size(part)) + " of " + mb(size) + " of "
                        + file + " (" + e.getMessage() + "); start it again to resume", e);
            }
        }
        if (size > 0 && Files.size(part) != size) {
            throw new IOException(file + " stopped at " + Files.size(part) + " of " + size + " bytes; start it again to"
                    + " resume");
        }
        String sha = info.get("lfs") instanceof Map<?, ?> lfs ? Json.string(Json.object(lfs), "oid", "") : "";
        if (sha.length() == 64) {
            d.state = "checking";
            String got = sha256(part);
            d.state = "downloading";
            if (!got.equalsIgnoreCase(sha)) {
                Files.delete(part);
                throw new IOException(file + " arrived damaged (its SHA-256 does not match the Hub's); start it again");
            }
        }
        Files.move(part, out, StandardCopyOption.REPLACE_EXISTING);
    }

    private static String mb(long bytes) {
        return bytes >= 1e9 ? String.format(Locale.ROOT, "%.2f GB", bytes / 1e9)
                : String.format(Locale.ROOT, "%.1f MB", bytes / 1e6);
    }

    static String sha256(Path p) throws IOException {
        try (InputStream in = Files.newInputStream(p)) {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] buf = new byte[1 << 20];
            int n;
            while ((n = in.read(buf)) > 0) {
                md.update(buf, 0, n);
            }
            return HexFormat.of().formatHex(md.digest());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    // ---------- talking to the Hub ----------

    private static void checkRepo(String repo) throws IOException {
        if (!repo.matches("[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*") || repo.contains("..")) {
            throw new IOException("not a Hugging Face repository: " + repo + " (it looks like owner/name)");
        }
    }

    private static String repoName(String repo) {
        return repo.substring(repo.indexOf('/') + 1);
    }

    private static long size(Map<String, Object> f) {
        if (f == null) {
            return 0;
        }
        if (f.get("lfs") instanceof Map<?, ?> lfs && Json.object(lfs).get("size") instanceof Number n) {
            return n.longValue();
        }
        return f.get("size") instanceof Number n ? n.longValue() : 0;
    }

    private URI fileUri(String repo, String file) {
        StringBuilder p = new StringBuilder(repo).append("/resolve/main/");
        String[] parts = file.split("/");
        for (int i = 0; i < parts.length; i++) {
            p.append(i == 0 ? "" : "/").append(URLEncoder.encode(parts[i], StandardCharsets.UTF_8).replace("+", "%20"));
        }
        return base.resolve(p.toString());
    }

    private List<Map<String, Object>> tree(String repo) throws IOException {
        List<Map<String, Object>> out = new ArrayList<>();
        URI next = base.resolve("api/models/" + repo + "/tree/main?recursive=true");
        for (int page = 0; next != null && page < 50; page++) {
            HttpResponse<String> r = get(next, HttpResponse.BodyHandlers.ofString(), -1);
            check(r.statusCode(), repo, String.valueOf(r.body()));
            Json.array(Json.parse(r.body())).forEach(o -> out.add(Json.object(o)));
            next = r.headers().firstValue("Link").flatMap(Hub::nextLink).map(u -> base.resolve(u)).orElse(null);
        }
        return out;
    }

    private static Optional<String> nextLink(String link) {
        Matcher m = Pattern.compile("<([^>]+)>;\\s*rel=\"next\"").matcher(link);
        return m.find() ? Optional.of(m.group(1)) : Optional.empty();
    }

    private String text(URI uri) throws IOException {
        HttpResponse<String> r = get(uri, HttpResponse.BodyHandlers.ofString(), -1);
        check(r.statusCode(), uri.getPath(), String.valueOf(r.body()));
        return r.body();
    }

    // the first bytes of a file; a server that ignores the range is cut off after them
    private byte[] head(URI uri, int bytes) throws IOException {
        HttpResponse<InputStream> r = get(uri, HttpResponse.BodyHandlers.ofInputStream(), 0, bytes - 1);
        try (InputStream in = r.body()) {
            if (r.statusCode() >= 300) {
                check(r.statusCode(), uri.getPath(), new String(in.readNBytes(2000), StandardCharsets.UTF_8));
            }
            return in.readNBytes(bytes);
        }
    }

    private HttpResponse<InputStream> open(URI uri, long from) throws IOException {
        HttpResponse<InputStream> r = get(uri, HttpResponse.BodyHandlers.ofInputStream(), from, -1);
        if (r.statusCode() >= 300) {
            String body;
            try (InputStream in = r.body()) {
                body = new String(in.readNBytes(2000), StandardCharsets.UTF_8);
            }
            check(r.statusCode(), uri.getPath(), body);
        }
        return r;
    }

    private <T> HttpResponse<T> get(URI uri, HttpResponse.BodyHandler<T> handler, long from) throws IOException {
        return get(uri, handler, from, -1);
    }

    // follows redirects by hand: the key goes only to the Hub itself, never to the file servers it redirects to
    private <T> HttpResponse<T> get(URI uri, HttpResponse.BodyHandler<T> handler, long from, long to)
            throws IOException {
        URI at = uri;
        for (int hops = 0; hops < 8; hops++) {
            HttpRequest.Builder b = HttpRequest.newBuilder(at).timeout(Duration.ofMinutes(5))
                    .header("User-Agent", "brainiac-studio");
            if (from > 0 || to >= 0) {
                b.header("Range", "bytes=" + Math.max(0, from) + "-" + (to >= 0 ? String.valueOf(to) : ""));
            }
            if (sameHost(at, base)) {
                token().ifPresent(t -> b.header("Authorization", "Bearer " + t));
            }
            HttpResponse<T> r;
            try {
                r = http.send(b.build(), handler);
            } catch (ConnectException | HttpConnectTimeoutException e) {
                throw new IOException("cannot reach " + base.getHost() + "; is this PC online?", e);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new java.io.InterruptedIOException("stopped");
            }
            int s = r.statusCode();
            Optional<String> loc = r.headers().firstValue("Location");
            if (s >= 300 && s < 400 && loc.isPresent()) {
                if (r.body() instanceof InputStream in) {
                    in.close();
                }
                at = at.resolve(loc.get().replace(" ", "%20"));
                continue;
            }
            return r;
        }
        throw new IOException("too many redirects for " + uri);
    }

    private static boolean sameHost(URI a, URI b) {
        return a.getHost() != null && a.getHost().equalsIgnoreCase(b.getHost()) && a.getPort() == b.getPort();
    }

    private Optional<String> token() {
        List<Keys.Entry> list = keys.list("huggingface");
        return list.isEmpty() ? Optional.empty() : keys.get("huggingface", list.getFirst().label());
    }

    private void check(int status, String what, String body) throws IOException {
        if (status < 300) {
            return;
        }
        if (status == 401 || status == 403) {
            boolean hasKey = token().isPresent();
            throw new IOException(hasKey
                    ? "Hugging Face refused it. If the model is gated, accept its license on huggingface.co with the account"
                            + " your Hugging Face key belongs to, then try again."
                    : "this model is gated or private. Accept its license on huggingface.co, then add a Hugging Face key"
                            + " (a read token) under Providers and try again.");
        }
        if (status == 404) {
            throw new IOException("Hugging Face has nothing at " + what);
        }
        if (status == 429) {
            throw new IOException("Hugging Face is limiting requests; wait a minute and try again");
        }
        String msg = "";
        try {
            msg = Json.string(Json.object(Json.parse(body)), "error", "");
        } catch (RuntimeException e) {
            msg = "";
        }
        throw new IOException("Hugging Face answered " + status + (msg.isEmpty() ? "" : ": " + msg));
    }

    /**
     * Points the client at another address, such as a mirror.
     *
     * @param address the address, or null for the Hub
     */
    void base(URI address) {
        URI a = address == null ? BASE : address;
        this.base = a.toString().endsWith("/") ? a : URI.create(a + "/");
    }
}
