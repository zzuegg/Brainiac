package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.llm.Tools;
import io.github.zzuegg.brainiac.studio.project.Project;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.MalformedInputException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.PathMatcher;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;

/**
 * A search index over the project's text files: files are cut into passages, and passages are
 * ranked by BM25 against the words asked for. The index is kept per project in the studio's
 * folder and brought up to date for files that changed.
 */
public final class KnowledgeTools {

    private static final int PASSAGE_LINES = 40;
    private static final long FILE_BYTES = 2L << 20;
    private static final int MAX_FILES = 20_000;
    private static final Set<String> TEXT = Set.of("txt", "md", "markdown", "rst", "adoc", "org", "csv", "tsv", "json",
            "jsonl", "yaml", "yml", "toml", "xml", "html", "htm", "java", "kt", "kts", "py", "js", "ts", "tsx", "jsx",
            "go", "rs", "c", "h", "cpp", "hpp", "cs", "rb", "php", "swift", "sh", "sql", "tex", "ini", "cfg", "properties",
            "gradle", "srt", "vtt", "eml");
    private static final Set<String> SKIP = Set.of(".git", "node_modules", "build", "target", ".gradle", "dist", "out",
            ".venv", "__pycache__");

    private static final int EMBED_MAX = 5000;

    private final Path folder;
    private final Embedder embedder;

    /**
     * Creates the tools, finding passages by their words only.
     *
     * @param folder where indexes are kept, one file per project
     */
    public KnowledgeTools(Path folder) {
        this(folder, null);
    }

    /**
     * Creates the tools.
     *
     * @param folder where indexes are kept, one file per project
     * @param embedder what finds passages by meaning too, or null for words only
     */
    public KnowledgeTools(Path folder, Embedder embedder) {
        this.folder = folder;
        this.embedder = embedder;
    }

    @Tools.Tool(name = "index_files", value = "Builds or refreshes the search index over the project's text files, "
            + "so search_knowledge can find passages by their words and, with an embedding model, by meaning. Only changed files are read again.")
    @Tier(value = Risk.COMPUTE, what = "indexes the project's files")
    public Map<String, Object> indexFiles(@Tools.Param("which files, such as '**/*.md'; every text file if left out")
            Optional<String> glob) {
        Project p = FileTools.project();
        Path root = p.inFolder("");
        PathMatcher match = glob.filter(g -> !g.isBlank())
                .map(g -> FileSystems.getDefault().getPathMatcher("glob:" + g)).orElse(null);
        synchronized (this) {
            Map<String, Object> old = load(p.id());
            Map<String, Object> oldFiles = old.get("files") instanceof Map<?, ?> m ? Json.object(m) : Map.of();
            String model = embedder == null ? "" : embedder.name().orElse("");
            boolean sameModel = model.equals(Json.string(old, "embedder", ""));
            List<Map<String, Object>> toEmbed = new ArrayList<>();
            List<String> texts = new ArrayList<>();
            Map<String, Object> files = new LinkedHashMap<>();
            int read = 0;
            int kept = 0;
            try (Stream<Path> s = Files.walk(root)) {
                for (Path f : (Iterable<Path>) s.filter(Files::isRegularFile)::iterator) {
                    Path rel = root.relativize(f);
                    if (skipped(rel) || match != null && !match.matches(rel) || !text(f) || Files.size(f) > FILE_BYTES) {
                        continue;
                    }
                    if (files.size() >= MAX_FILES) {
                        break;
                    }
                    String key = rel.toString().replace('\\', '/');
                    long stamp = Files.getLastModifiedTime(f).toMillis() ^ Files.size(f);
                    Object was = oldFiles.get(key);
                    if (was instanceof Map<?, ?> w && Json.integer(Json.object(w), "stamp", -1) == stamp && sameModel) {
                        files.put(key, was);
                        kept++;
                        continue;
                    }
                    List<String> lines;
                    try {
                        lines = Files.readAllLines(f, StandardCharsets.UTF_8);
                    } catch (MalformedInputException e) {
                        continue;
                    }
                    List<Map<String, Object>> passages = new ArrayList<>();
                    for (int i = 0; i < lines.size(); i += PASSAGE_LINES - 5) {
                        int end = Math.min(lines.size(), i + PASSAGE_LINES);
                        String text = String.join("\n", lines.subList(i, end));
                        Map<String, Integer> tf = terms(key + "\n" + text);
                        if (!tf.isEmpty()) {
                            Map<String, Object> ps = new LinkedHashMap<>();
                            ps.put("line", i + 1);
                            ps.put("terms", tf);
                            ps.put("length", tf.values().stream().mapToInt(Integer::intValue).sum());
                            passages.add(ps);
                            if (!model.isEmpty() && toEmbed.size() < EMBED_MAX) {
                                toEmbed.add(ps);
                                texts.add(key + "\n" + text);
                            }
                        }
                        if (end == lines.size()) {
                            break;
                        }
                    }
                    files.put(key, Map.of("stamp", stamp, "passages", passages));
                    read++;
                }
            } catch (IOException e) {
                throw new UncheckedIOException(e.getMessage(), e);
            }
            String note = null;
            if (!toEmbed.isEmpty()) {
                try {
                    float[][] v = embedder.embed(texts, false);
                    for (int i = 0; i < v.length; i++) {
                        toEmbed.get(i).put("vec", Embedder.pack(v[i]));
                    }
                } catch (RuntimeException | LinkageError e) {
                    note = "found by words only: the embedding model failed (" + e.getMessage() + ")";
                }
            }
            Map<String, Object> index = new LinkedHashMap<>();
            index.put("project", p.id());
            index.put("embedder", note == null ? model : "");
            index.put("built", System.currentTimeMillis());
            index.put("files", files);
            save(p.id(), index);
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("files", files.size());
            out.put("readAgain", read);
            out.put("unchanged", kept);
            out.put("meaning", note == null && !model.isEmpty() ? model : "off: words only");
            if (note != null) {
                out.put("note", note);
            } else if (model.isEmpty()) {
                out.put("note", "found by words only; download a sentence-embedding model such as BAAI/bge-small-en-v1.5, "
                        + "or set an embeddings API, to find passages by meaning too");
            }
            return out;
        }
    }

    @Tools.Tool(name = "search_knowledge", value = "Finds the passages of the project's files that best match some "
            + "words, best first, with file and line; then read_project_file reads around them. Run index_files first "
            + "if the files changed.")
    @Tier(value = Risk.READ, resultChars = 12_000)
    public Map<String, Object> searchKnowledge(@Tools.Param("the words to look for") String query,
            @Tools.Param("how many passages, 8 if left out, at most 30") Optional<Integer> count) {
        Project p = FileTools.project();
        Map<String, Object> index;
        synchronized (this) {
            index = load(p.id());
        }
        if (!(index.get("files") instanceof Map<?, ?> fm) || fm.isEmpty()) {
            throw new IllegalArgumentException("the project has no index yet; run index_files first");
        }
        Map<String, Object> files = Json.object(fm);
        Set<String> q = terms(query).keySet();
        if (q.isEmpty()) {
            throw new IllegalArgumentException("say what to look for");
        }
        record Hit(String file, int line, double score) {
        }
        List<Hit> all = new ArrayList<>();
        Map<String, Integer> df = new HashMap<>();
        long n = 0;
        double totalLength = 0;
        List<Object[]> passages = new ArrayList<>();
        for (Map.Entry<String, Object> f : files.entrySet()) {
            for (Object o : Json.array(Json.object(f.getValue()).getOrDefault("passages", List.of()))) {
                Map<String, Object> ps = Json.object(o);
                Map<String, Object> tf = Json.object(ps.get("terms"));
                n++;
                totalLength += Json.integer(ps, "length", 1);
                for (String t : q) {
                    if (tf.containsKey(t)) {
                        df.merge(t, 1, Integer::sum);
                    }
                }
                passages.add(new Object[] {f.getKey(), ps, tf});
            }
        }
        double avg = n == 0 ? 1 : totalLength / n;
        String model = Json.string(index, "embedder", "");
        float[] qv = null;
        if (embedder != null && !model.isEmpty() && model.equals(embedder.name().orElse(""))) {
            try {
                qv = embedder.embed(List.of(query), true)[0];
            } catch (RuntimeException | LinkageError e) {
                qv = null;
            }
        }
        double best = 0;
        List<double[]> raw = new ArrayList<>();
        for (Object[] x : passages) {
            @SuppressWarnings("unchecked")
            Map<String, Object> ps = (Map<String, Object>) x[1];
            @SuppressWarnings("unchecked")
            Map<String, Object> tf = (Map<String, Object>) x[2];
            double len = Json.integer(ps, "length", 1);
            double score = 0;
            for (String t : q) {
                Object c = tf.get(t);
                if (c instanceof Number num) {
                    double f = num.doubleValue();
                    double idf = Math.log(1 + (n - df.get(t) + 0.5) / (df.get(t) + 0.5));
                    score += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len / avg));
                }
            }
            double meaning = qv != null && ps.get("vec") instanceof String vs ? Math.max(0, Embedder.cosine(qv, Embedder.unpack(vs))) : 0;
            best = Math.max(best, score);
            raw.add(new double[] {score, meaning});
        }
        for (int i = 0; i < passages.size(); i++) {
            double[] r = raw.get(i);
            // words and meaning, each scaled to 0..1, count the same once vectors are there
            double s = qv == null ? r[0] : 0.5 * (best > 0 ? r[0] / best : 0) + 0.5 * r[1];
            if (s > 0 && (qv == null || r[0] > 0 || r[1] > 0.3)) {
                @SuppressWarnings("unchecked")
                Map<String, Object> ps = (Map<String, Object>) passages.get(i)[1];
                all.add(new Hit((String) passages.get(i)[0], (int) Json.integer(ps, "line", 1), s));
            }
        }
        all.sort((a, b) -> Double.compare(b.score(), a.score()));
        Path root = p.inFolder("");
        List<Map<String, Object>> out = new ArrayList<>();
        for (Hit h : all.subList(0, Math.min(all.size(), Math.clamp(count.orElse(8), 1, 30)))) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("file", h.file());
            m.put("line", h.line());
            m.put("score", Math.round(h.score() * 100) / 100.0);
            m.put("text", excerpt(root.resolve(h.file()), h.line(), q));
            out.add(m);
        }
        return Map.of("query", query, "by", qv == null ? "words" : "words and meaning", "passages", out);
    }

    private static String excerpt(Path f, int line, Set<String> q) {
        try {
            List<String> lines = Files.readAllLines(f, StandardCharsets.UTF_8);
            int from = Math.max(0, line - 1);
            int to = Math.min(lines.size(), from + PASSAGE_LINES);
            // the lines that hold the words, with one around each
            StringBuilder b = new StringBuilder();
            int shown = 0;
            for (int i = from; i < to && shown < 12; i++) {
                String l = lines.get(i).toLowerCase(Locale.ROOT);
                if (q.stream().anyMatch(l::contains)) {
                    for (int k = Math.max(from, i - 1); k <= Math.min(to - 1, i + 1); k++) {
                        b.append(k + 1).append(": ").append(cut(lines.get(k))).append('\n');
                        shown++;
                    }
                    i++;
                }
            }
            if (b.isEmpty()) {
                for (int i = from; i < Math.min(to, from + 6); i++) {
                    b.append(i + 1).append(": ").append(cut(lines.get(i))).append('\n');
                }
            }
            return b.toString().strip();
        } catch (IOException e) {
            return "(the file changed or is gone; run index_files)";
        }
    }

    private static String cut(String s) {
        return s.length() > 200 ? s.substring(0, 200) + "…" : s;
    }

    static Map<String, Integer> terms(String text) {
        Map<String, Integer> tf = new HashMap<>();
        for (String w : text.toLowerCase(Locale.ROOT).split("[^\\p{L}\\p{N}]+")) {
            if (w.length() < 2 || w.length() > 40) {
                continue;
            }
            tf.merge(stem(w), 1, Integer::sum);
        }
        return tf;
    }

    // a light stem, so plurals and the like meet
    private static String stem(String w) {
        if (w.length() > 5 && w.endsWith("ing")) {
            return w.substring(0, w.length() - 3);
        }
        if (w.length() > 4 && w.endsWith("es")) {
            return w.substring(0, w.length() - 2);
        }
        if (w.length() > 3 && w.endsWith("s") && !w.endsWith("ss")) {
            return w.substring(0, w.length() - 1);
        }
        return w;
    }

    private static boolean skipped(Path rel) {
        for (Path part : rel) {
            if (SKIP.contains(part.toString())) {
                return true;
            }
        }
        return false;
    }

    private static boolean text(Path f) {
        String n = f.getFileName().toString().toLowerCase(Locale.ROOT);
        int dot = n.lastIndexOf('.');
        return dot < 0 ? !n.startsWith(".") : TEXT.contains(n.substring(dot + 1));
    }

    private Map<String, Object> load(String project) {
        Path f = folder.resolve(project + ".json");
        if (!Files.isRegularFile(f)) {
            return Map.of();
        }
        try {
            return Json.object(Json.parse(Files.readString(f, StandardCharsets.UTF_8)));
        } catch (IOException | RuntimeException e) {
            return Map.of();
        }
    }

    private void save(String project, Map<String, Object> index) {
        try {
            Files.createDirectories(folder);
            Path tmp = folder.resolve(project + ".json.tmp");
            Files.writeString(tmp, Json.write(index), StandardCharsets.UTF_8);
            Files.move(tmp, folder.resolve(project + ".json"), StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
    }
}
