package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.llm.Tools;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Tools that see code as declarations rather than lines: a file's outline, where a class or
 * method is declared and used, and a ranked map of a whole repository, so an agent can find its
 * way in a large codebase without reading it all.
 */
@ToolGroup(id = "symbols", summary = "outlines, declarations, uses and a map of the code")
public final class SymbolTools implements RiskRefiner {

    private static final int MAX_FILES = 20_000;

    /** Creates the tools. */
    public SymbolTools() {
    }

    @Override
    public Risk refine(String tool, Map<String, Object> arguments, Risk declared) {
        return declared;
    }

    @Override
    public List<Reach.Access> reach(String tool, Map<String, Object> arguments) {
        return Reach.of(arguments, false, "path");
    }

    /**
     * Outlines a file.
     *
     * @param path the file
     * @return its declarations
     */
    @Tools.Tool(name = "outline", value = "Lists a source file's classes, methods, functions and fields with the "
            + "lines each spans, nested as in the file, so you can read only the part you need with read_file. Works "
            + "for Java, Kotlin, JavaScript, TypeScript, Python, C-like languages and more, also on code that does not "
            + "build.")
    @Tier(value = Risk.READ, resultChars = 16_000)
    @Returns("text: one line per declaration, 'start-end kind name: signature', indented by nesting")
    public String outline(@Tools.Param("the file") String path) {
        Path f = Reach.file(path);
        if (!Files.isRegularFile(f)) {
            throw new IllegalArgumentException("there is no file " + path + FileTools.nearby(path));
        }
        if (!Symbols.source(f.getFileName().toString())) {
            throw new IllegalArgumentException(path + " is not source code this reads; read_file reads any text");
        }
        List<Symbols.Symbol> all = Symbols.of(f).symbols();
        if (all.isEmpty()) {
            return path + " declares nothing found at the top; read_file reads it";
        }
        StringBuilder b = new StringBuilder(path).append(" (").append(all.size()).append(" declarations)\n");
        for (Symbols.Symbol s : all) {
            if (b.length() > 15_000) {
                b.append("[cut; read_file reads the rest]\n");
                break;
            }
            b.append("  ".repeat(s.depth())).append(s.line()).append('-').append(s.end()).append(' ').append(s.kind())
                    .append(' ').append(s.name());
            if (!s.kind().equals("field")) {
                b.append(": ").append(cut(s.signature(), 140));
            }
            b.append('\n');
        }
        return b.toString();
    }

    /**
     * Finds declarations.
     *
     * @param name the name
     * @param kind which kind
     * @param path where to look
     * @return the declarations
     */
    @Tools.Tool(name = "find_symbol", value = "Finds where a class, method, function or field is declared across "
            + "the project, by its name or a pattern such as 'read*' or '*Tools'. Answers with file, lines and "
            + "signature. Faster and more exact than grep for 'where is X defined'.")
    @Tier(value = Risk.READ, resultChars = 16_000)
    @Returns("{symbols: [{file, line, end, kind, name, container, signature}], total}")
    public Map<String, Object> findSymbol(@Tools.Param("the name, or a pattern with *; case matters only if it has capitals") String name,
            @Tools.Param("only this kind: class, interface, enum, record, method, function, field or constructor") Optional<String> kind,
            @Tools.Param("the folder to look in; the project's folder if left out") Optional<String> path) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("give a name, such as 'FileTools' or 'read*'");
        }
        Pattern want = pattern(name.strip());
        String k = kind.map(x -> x.strip().toLowerCase(Locale.ROOT)).filter(x -> !x.isEmpty()).orElse(null);
        Path root = folder(path);
        List<Map<String, Object>> found = new ArrayList<>();
        int[] total = {0};
        each(root, (rel, fs) -> {
            for (Symbols.Symbol s : fs.symbols()) {
                if (want.matcher(s.name()).matches() && (k == null || kindOf(s.kind(), k))) {
                    total[0]++;
                    if (found.size() < 100) {
                        found.add(row(rel, s));
                    }
                }
            }
        });
        // declarations of types first, then shallow ones
        found.sort(Comparator.comparingInt((Map<String, Object> m) -> isType(String.valueOf(m.get("kind"))) ? 0 : 1));
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("symbols", found);
        out.put("total", total[0]);
        if (total[0] == 0) {
            out.put("note", "nothing declared by that name; grep finds it in text, search_knowledge by meaning");
        }
        return out;
    }

    /**
     * Finds the uses of a name.
     *
     * @param name the name
     * @param path where to look
     * @return the uses
     */
    @Tools.Tool(name = "find_references", value = "Finds every use of a class, method, function or field name across "
            + "the project's source files, as a whole word, with the declaration it sits in. Use it before renaming "
            + "something or changing its parameters, to see what else must change.")
    @Tier(value = Risk.READ, resultChars = 16_000)
    @Returns("{uses: [{file, line, text, in}], declared: [{file, line, kind}], files, total, next?}")
    public Map<String, Object> findReferences(@Tools.Param("the name, exactly, such as 'readFile'") String name,
            @Tools.Param("the folder to look in; the project's folder if left out") Optional<String> path,
            @Tools.Param("the use to start at, from 'next' of the last answer; 0 if left out") Optional<Integer> offset) {
        if (name == null || !name.strip().matches("[A-Za-z_$][\\w$]*")) {
            throw new IllegalArgumentException("give one name, such as 'readFile'");
        }
        String n = name.strip();
        Pattern word = Pattern.compile("(?<![\\w$])" + Pattern.quote(n) + "(?![\\w$])");
        Path root = folder(path);
        int from = Math.max(0, offset.orElse(0));
        List<Map<String, Object>> uses = new ArrayList<>();
        List<Map<String, Object>> declared = new ArrayList<>();
        int[] total = {0};
        int[] files = {0};
        each(root, (rel, fs) -> {
            if (!fs.words().contains(n)) {
                return;
            }
            List<String> lines;
            try {
                lines = Files.readAllLines(Reach.file(FileWalk.name(rel)), java.nio.charset.StandardCharsets.UTF_8);
            } catch (java.io.IOException | RuntimeException e) {
                return;
            }
            boolean any = false;
            for (int i = 0; i < lines.size(); i++) {
                if (!word.matcher(lines.get(i)).find()) {
                    continue;
                }
                int line = i + 1;
                Symbols.Symbol decl = fs.symbols().stream().filter(s -> s.line() == line && s.name().equals(n))
                        .findFirst().orElse(null);
                if (decl != null) {
                    Map<String, Object> d = new LinkedHashMap<>();
                    d.put("file", FileWalk.name(rel));
                    d.put("line", line);
                    d.put("kind", decl.kind());
                    declared.add(d);
                    continue;
                }
                any = true;
                if (total[0]++ >= from && uses.size() < 120) {
                    Map<String, Object> u = new LinkedHashMap<>();
                    u.put("file", FileWalk.name(rel));
                    u.put("line", line);
                    u.put("text", cut(lines.get(i).strip(), 200));
                    String in = enclosing(fs.symbols(), line);
                    if (!in.isEmpty()) {
                        u.put("in", in);
                    }
                    uses.add(u);
                }
            }
            if (any) {
                files[0]++;
            }
        });
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("uses", uses);
        out.put("declared", declared);
        out.put("files", files[0]);
        out.put("total", total[0]);
        if (from + uses.size() < total[0]) {
            out.put("next", from + uses.size());
        }
        return out;
    }

    /**
     * Maps the repository.
     *
     * @param path which folder
     * @param budget how many characters
     * @return the map
     */
    @Tools.Tool(name = "repo_map", value = "A compact map of the code: its most used files first, each with its "
            + "classes and the signatures of their methods, within a size budget. Files are ranked by how many other "
            + "files use what they declare. Call it first in an unfamiliar codebase, or for one module, to see where "
            + "things are before you search or read.")
    @Tier(value = Risk.READ, resultChars = 40_000)
    @Returns("text: a header, then per file 'path (used by n files)' and its declarations with line numbers")
    public String repoMap(@Tools.Param("a folder, such as one module; the project's folder if left out") Optional<String> path,
            @Tools.Param("about how many characters the map may take, 12000 if left out, at most 36000") Optional<Integer> budget) {
        Path root = folder(path);
        int max = Math.clamp(budget.orElse(12_000), 2000, 36_000);
        Map<Path, Symbols.FileSymbols> all = new LinkedHashMap<>();
        each(root, all::put);
        if (all.isEmpty()) {
            return "no source files found under " + path.orElse("the project's folder") + "; tree shows what is there";
        }
        // which files declare each top-level name
        Map<String, List<Path>> declaredIn = new HashMap<>();
        for (Map.Entry<Path, Symbols.FileSymbols> e : all.entrySet()) {
            for (Symbols.Symbol s : e.getValue().symbols()) {
                if (s.depth() == 0 && s.name().length() > 2) {
                    declaredIn.computeIfAbsent(s.name(), x -> new ArrayList<>()).add(e.getKey());
                }
            }
        }
        // a file's rank: how many other files use a name it declares at the top
        Map<Path, Integer> usedBy = new HashMap<>();
        for (Map.Entry<Path, Symbols.FileSymbols> e : all.entrySet()) {
            Set<String> words = e.getValue().words();
            java.util.Set<Path> uses = new java.util.HashSet<>();
            for (String w : words) {
                List<Path> d = declaredIn.get(w);
                if (d != null && d.size() <= 3) {   // names declared everywhere say little
                    for (Path p : d) {
                        if (!p.equals(e.getKey())) {
                            uses.add(p);
                        }
                    }
                }
            }
            for (Path p : uses) {
                usedBy.merge(p, 1, Integer::sum);
            }
        }
        List<Path> ranked = new ArrayList<>(all.keySet());
        ranked.sort(Comparator.comparingInt((Path p) -> -usedBy.getOrDefault(p, 0)).thenComparing(Path::toString));
        StringBuilder b = new StringBuilder();
        b.append(all.size()).append(" source files under ").append(path.filter(s -> !s.isBlank()).orElse("the project"))
                .append(", most used first. Lines are 'start-end kind name'; outline shows a whole file.\n");
        int shown = 0;
        for (Path rel : ranked) {
            StringBuilder part = new StringBuilder();
            part.append('\n').append(FileWalk.name(rel)).append("  (used by ").append(usedBy.getOrDefault(rel, 0))
                    .append(" files)\n");
            int members = 0;
            for (Symbols.Symbol s : all.get(rel).symbols()) {
                if (s.kind().equals("field") || s.depth() > 1 || s.signature().strip().startsWith("private ")) {
                    continue;
                }
                if (members++ == 25) {
                    part.append("  …\n");
                    break;
                }
                part.append("  ".repeat(s.depth() + 1)).append(s.line()).append('-').append(s.end()).append(' ')
                        .append(s.kind()).append(' ').append(isType(s.kind()) ? s.name() : cut(signature(s), 120))
                        .append('\n');
            }
            if (b.length() + part.length() > max) {
                if (b.length() + 200 < max) {
                    continue;   // a smaller file may still fit
                }
                break;
            }
            b.append(part);
            shown++;
        }
        if (shown < ranked.size()) {
            b.append("\n[").append(ranked.size() - shown).append(" more files left out; pass a folder or a bigger budget]\n");
        }
        return b.toString();
    }

    // ---------------------------------------------------------------- helpers

    private interface Visit {
        void file(Path rel, Symbols.FileSymbols symbols);
    }

    // each source file under a folder, with its path relative to the project's folder
    private static void each(Path root, Visit visit) {
        int[] n = {0};
        new FileWalk(root).files((f, rel) -> {
            if (Symbols.source(f.getFileName().toString())) {
                visit.file(FileWalk.shown(root, rel), Symbols.of(f));
            }
            return ++n[0] < MAX_FILES;
        });
    }

    private static Path folder(Optional<String> path) {
        Path root = Reach.file(path.orElse(""));
        if (!Files.isDirectory(root)) {
            throw new IllegalArgumentException(path.orElse("") + " is not a folder" + FileTools.nearby(path.orElse("")));
        }
        return root;
    }

    private static Pattern pattern(String name) {
        boolean exact = !name.equals(name.toLowerCase(Locale.ROOT));
        String regex = Pattern.quote(name).replace("*", "\\E.*\\Q");
        return Pattern.compile(regex, exact ? 0 : Pattern.CASE_INSENSITIVE);
    }

    private static boolean kindOf(String kind, String want) {
        return kind.equals(want) || (want.equals("class") && isType(kind))
                || (want.equals("method") && (kind.equals("function") || kind.equals("constructor")))
                || (want.equals("function") && kind.equals("method"));
    }

    private static boolean isType(String kind) {
        return switch (kind) {
            case "class", "interface", "enum", "record", "annotation", "object", "trait", "struct", "namespace" -> true;
            default -> false;
        };
    }

    private static Map<String, Object> row(Path rel, Symbols.Symbol s) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("file", FileWalk.name(rel));
        m.put("line", s.line());
        m.put("end", s.end());
        m.put("kind", s.kind());
        m.put("name", s.name());
        if (!s.container().isEmpty()) {
            m.put("container", s.container());
        }
        m.put("signature", cut(s.signature(), 200));
        return m;
    }

    // the innermost declaration around a line, such as 'FileTools.readFile'
    private static String enclosing(List<Symbols.Symbol> symbols, int line) {
        Symbols.Symbol best = null;
        for (Symbols.Symbol s : symbols) {
            if (!s.kind().equals("field") && s.line() <= line && line <= s.end()
                    && (best == null || s.depth() >= best.depth())) {
                best = s;
            }
        }
        if (best == null) {
            return "";
        }
        return best.container().isEmpty() ? best.name() : best.container() + "." + best.name();
    }

    // a signature without annotations, modifiers that say little, or the opening brace
    private static String signature(Symbols.Symbol s) {
        String t = s.signature().replaceAll("^(@[\\w.]+(\\([^)]*\\))?\\s+)+", "").replaceAll("\\s*[{;]\\s*$", "")
                .replaceAll("^(public|protected|final|synchronized|override)\\s+", "");
        return t.isEmpty() ? s.name() : t;
    }

    private static String cut(String s, int n) {
        return s.length() > n ? s.substring(0, n) + "…" : s;
    }
}
