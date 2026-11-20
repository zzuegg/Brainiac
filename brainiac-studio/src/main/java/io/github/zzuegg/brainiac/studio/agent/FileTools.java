package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.llm.Tools;
import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.project.Project;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.MalformedInputException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.PathMatcher;
import java.nio.file.StandardOpenOption;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.TreeMap;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Tools that read, write, edit and find files in the project's folder. Every change is
 * checkpointed first, so a run can be undone as a whole. Locations are given as a file and a line
 * number throughout, so what grep or a test finds can be read and edited directly.
 */
@ToolGroup(id = "files", summary = "read, write, edit, move and find files in the project's folder")
public final class FileTools implements RiskRefiner, Checkpointed {

    /** The most characters a file written by an agent may have. */
    static final int MAX_FILE = 2_000_000;

    private static final long READ_BYTES = 32L << 20;
    private static final long SEARCH_BYTES = 4L << 20;
    private static final int SEARCH_FILES = 50_000;
    private static final int READ_LINES = 400;
    private static final int READ_CHARS = 30_000;
    private static final int LINE_CHARS = 2000;
    private static final int PAGE = 200;

    private volatile Checkpoints checkpoints;

    /** Creates the tools; checkpoints are set by the runs that use them. */
    public FileTools() {
    }

    @Override
    public void checkpoints(Checkpoints c) {
        this.checkpoints = c;
    }

    @Override
    public Risk refine(String tool, Map<String, Object> arguments, Risk declared) {
        if (!tool.equals("write_file")) {
            return declared;
        }
        String mode = String.valueOf(arguments.getOrDefault("mode", "create"));
        Object path = arguments.get("path");
        try {
            Path f = path == null ? null : Reach.locate(String.valueOf(path));
            if (mode.equals("create") || (f != null && !Files.exists(f))) {
                return Risk.CREATE;
            }
        } catch (RuntimeException e) {
            return declared;
        }
        return Risk.WRITE;
    }

    @Override
    public List<Reach.Access> reach(String tool, Map<String, Object> arguments) {
        return switch (tool) {
            case "write_file", "edit_file", "multi_edit", "delete_file" -> Reach.of(arguments, true, "path");
            case "move_file" -> Reach.of(arguments, true, "from", "to");
            case "read_file", "list_files", "glob", "tree" -> Reach.of(arguments, false, "path");
            case "grep" -> Reach.of(arguments, false, "folder");
            default -> List.of();
        };
    }

    // ---------------------------------------------------------------- reading

    /**
     * Reads a text file by lines.
     *
     * @param path the file
     * @param startLine the first line, from 1
     * @param lines how many lines
     * @return the lines, numbered
     */
    @Tools.Tool(name = "read_file", value = "Reads a text file by lines, each with its number, as in '12\\tcode'. "
            + "Reads 400 lines from the top if left out; the header says how many lines there are and where to read "
            + "on. Pass startLine from grep, outline or a test failure to read around it. Don't copy the line "
            + "numbers into edit_file.")
    @Tier(value = Risk.READ, resultChars = 32_000)
    @Returns("text: a header '[path: lines a-b of n]' then numbered lines")
    public String readFile(@Tools.Param("the file, relative to the project's folder, or a full path elsewhere") String path,
            @Tools.Param("the first line to read, from 1; 1 if left out") Optional<Integer> startLine,
            @Tools.Param("how many lines, 400 if left out, at most 2000") Optional<Integer> lines) {
        Path f = Reach.file(path);
        if (Files.isDirectory(f)) {
            throw new IllegalArgumentException(path + " is a folder; list it with list_files or tree");
        }
        if (!Files.isRegularFile(f)) {
            throw new IllegalArgumentException("there is no file " + path + nearby(path));
        }
        List<String> all = lines(f, path);
        int total = all.size();
        int from = Math.clamp(startLine.orElse(1), 1, Math.max(1, total));
        int count = Math.clamp(lines.orElse(READ_LINES), 1, 2000);
        int to = Math.min(total, from + count - 1);
        StringBuilder b = new StringBuilder();
        b.append('[').append(path).append(": ");
        if (total == 0) {
            return b.append("the file is empty]").toString();
        }
        int width = String.valueOf(to).length();
        StringBuilder body = new StringBuilder();
        int last = from - 1;
        for (int i = from; i <= to; i++) {
            String l = all.get(i - 1);
            if (l.length() > LINE_CHARS) {
                l = l.substring(0, LINE_CHARS) + "… [line cut at " + LINE_CHARS + " of " + l.length() + " characters]";
            }
            if (body.length() + l.length() > READ_CHARS && i > from) {
                break;   // keep within what the model is shown
            }
            body.append(" ".repeat(width - String.valueOf(i).length())).append(i).append('\t').append(l).append('\n');
            last = i;
        }
        b.append("lines ").append(from).append('-').append(last).append(" of ").append(total);
        if (last < total) {
            b.append("; read on with startLine ").append(last + 1);
        }
        b.append("]\n").append(body);
        return b.toString();
    }

    /**
     * Lists one folder.
     *
     * @param path the folder
     * @param offset where to start
     * @return the entries
     */
    @Tools.Tool(name = "list_files", value = "Lists the files and folders directly in one folder, with sizes. For a "
            + "whole tree use tree; to find files by name use glob.")
    @Tier(value = Risk.READ)
    @Returns("{folder, entries: [{name, folder, bytes}], total, next?}")
    public Map<String, Object> listFiles(@Tools.Param("a folder relative to the project's folder, or a full path elsewhere; the top if left out")
            Optional<String> path,
            @Tools.Param("the entry to start at, from 'next' of the last answer; 0 if left out") Optional<Integer> offset) {
        Path dir = Reach.file(path.orElse(""));
        if (!Files.isDirectory(dir)) {
            throw new IllegalArgumentException(path.orElse("") + " is not a folder" + nearby(path.orElse("")));
        }
        List<Path> all;
        try (Stream<Path> s = Files.list(dir)) {
            all = s.sorted(Comparator.comparing((Path p) -> !Files.isDirectory(p)).thenComparing(Path::getFileName))
                    .toList();
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
        int from = Math.clamp(offset.orElse(0), 0, all.size());
        List<Map<String, Object>> out = new ArrayList<>();
        for (Path p : all.subList(from, Math.min(all.size(), from + 500))) {
            Map<String, Object> e = new LinkedHashMap<>();
            e.put("name", p.getFileName().toString());
            e.put("folder", Files.isDirectory(p));
            if (Files.isRegularFile(p)) {
                try {
                    e.put("bytes", Files.size(p));
                } catch (IOException x) {
                    // gone meanwhile
                }
            }
            out.add(e);
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("folder", dir.toString());
        m.put("entries", out);
        m.put("total", all.size());
        if (from + out.size() < all.size()) {
            m.put("next", from + out.size());
        }
        return m;
    }

    /**
     * Finds files by a name pattern.
     *
     * @param pattern the glob
     * @param path the folder
     * @param offset where to start
     * @return the files
     */
    @Tools.Tool(name = "glob", value = "Finds files whose path matches a pattern, such as '**/*Test.java' or "
            + "'src/**/*.kt'; a pattern without '/' matches names in any folder, as '*.md'. Newest first. Build "
            + "output, caches and what .gitignore names are skipped.")
    @Tier(value = Risk.READ, resultChars = 16_000)
    @Returns("{files: [path], total, next?}")
    public Map<String, Object> glob(@Tools.Param("the pattern; '{a,b}' for either, '**' for any folders") String pattern,
            @Tools.Param("the folder to look in; the project's folder if left out") Optional<String> path,
            @Tools.Param("the file to start at, from 'next' of the last answer; 0 if left out") Optional<Integer> offset) {
        if (pattern == null || pattern.isBlank()) {
            throw new IllegalArgumentException("give a pattern, such as '**/*.java'");
        }
        Path root = folder(path);
        PathMatcher m = FileWalk.glob(pattern);
        record Hit(String path, long modified) {
        }
        List<Hit> hits = new ArrayList<>();
        int[] seen = {0};
        new FileWalk(root).files((f, rel) -> {
            if (m.matches(rel)) {
                long t;
                try {
                    t = Files.getLastModifiedTime(f).toMillis();
                } catch (IOException e) {
                    t = 0;
                }
                hits.add(new Hit(FileWalk.name(FileWalk.shown(root, rel)), t));
            }
            return ++seen[0] < SEARCH_FILES;
        });
        hits.sort(Comparator.comparingLong(Hit::modified).reversed().thenComparing(Hit::path));
        int from = Math.clamp(offset.orElse(0), 0, hits.size());
        int to = Math.min(hits.size(), from + PAGE);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("files", hits.subList(from, to).stream().map(Hit::path).toList());
        out.put("total", hits.size());
        if (to < hits.size()) {
            out.put("next", to);
        }
        if (seen[0] >= SEARCH_FILES) {
            out.put("note", "stopped after " + SEARCH_FILES + " files; narrow the folder");
        }
        return out;
    }

    /**
     * Shows a folder tree.
     *
     * @param path the folder
     * @param depth how deep
     * @return the tree as text
     */
    @Tools.Tool(name = "tree", value = "Shows the folder tree to a depth, with how many files and bytes each "
            + "folder holds, so you can see how a project is laid out before reading. Build output, caches and what "
            + ".gitignore names are skipped.")
    @Tier(value = Risk.READ, resultChars = 16_000)
    @Returns("text: one line per folder or file, indented, folders with their file count and size")
    public String tree(@Tools.Param("the folder; the project's folder if left out") Optional<String> path,
            @Tools.Param("how many folder levels to show, 3 if left out, at most 8") Optional<Integer> depth) {
        Path root = folder(path);
        int deep = Math.clamp(depth.orElse(3), 1, 8);
        // counts per folder, every level
        Map<String, long[]> sizes = new TreeMap<>();
        Map<String, List<String>> filesIn = new TreeMap<>();
        new FileWalk(root).files((f, rel) -> {
            long bytes;
            try {
                bytes = Files.size(f);
            } catch (IOException e) {
                bytes = 0;
            }
            Path parent = rel.getParent();
            String dir = parent == null ? "" : FileWalk.name(parent);
            filesIn.computeIfAbsent(dir, k -> new ArrayList<>()).add(rel.getFileName().toString());
            for (Path p = parent; ; p = p.getParent()) {
                long[] s = sizes.computeIfAbsent(p == null ? "" : FileWalk.name(p), k -> new long[2]);
                s[0]++;
                s[1] += bytes;
                if (p == null) {
                    break;
                }
            }
            return sizes.size() < SEARCH_FILES;
        });
        StringBuilder b = new StringBuilder();
        long[] top = sizes.getOrDefault("", new long[2]);
        b.append(path.filter(s -> !s.isBlank()).orElse(".")).append("/  (").append(top[0]).append(" files, ")
                .append(bytes(top[1])).append(")\n");
        int[] lines = {0};
        walkTree(b, "", 1, deep, sizes, filesIn, lines);
        if (lines[0] >= 400) {
            b.append("[cut at 400 lines; pass a smaller depth or a folder]\n");
        }
        return b.toString();
    }

    private static void walkTree(StringBuilder b, String dir, int level, int deep, Map<String, long[]> sizes,
            Map<String, List<String>> filesIn, int[] lines) {
        String prefix = dir.isEmpty() ? "" : dir + "/";
        List<String> subs = sizes.keySet().stream()
                .filter(k -> !k.isEmpty() && k.startsWith(prefix) && k.indexOf('/', prefix.length()) < 0)
                .sorted().toList();
        String indent = "  ".repeat(level);
        for (String s : subs) {
            if (lines[0]++ >= 400) {
                return;
            }
            long[] z = sizes.get(s);
            b.append(indent).append(s.substring(prefix.length())).append("/  (").append(z[0]).append(" files, ")
                    .append(bytes(z[1])).append(")\n");
            if (level < deep) {
                walkTree(b, s, level + 1, deep, sizes, filesIn, lines);
            }
        }
        List<String> files = filesIn.getOrDefault(dir, List.of());
        if (level <= deep) {
            List<String> sorted = files.stream().sorted().toList();
            int shown = 0;
            for (String f : sorted) {
                if (lines[0] >= 400) {
                    return;
                }
                if (shown == 30) {
                    b.append(indent).append("… ").append(sorted.size() - shown).append(" more files\n");
                    lines[0]++;
                    break;
                }
                b.append(indent).append(f).append('\n');
                lines[0]++;
                shown++;
            }
        }
    }

    private static String bytes(long n) {
        if (n < 1024) {
            return n + " B";
        }
        if (n < 1024 * 1024) {
            return (n / 1024) + " KB";
        }
        return String.format(Locale.ROOT, "%.1f MB", n / (1024.0 * 1024.0));
    }

    // ---------------------------------------------------------------- writing

    /**
     * Writes a file.
     *
     * @param path the file
     * @param text its text
     * @param mode create, overwrite or append
     * @return what was written
     */
    @Tools.Tool(name = "write_file", value = "Writes a text file in the project's folder. 'create' makes a new file and "
            + "fails if it exists; 'overwrite' replaces one; 'append' adds to the end, for writing a long file a part "
            + "at a time. Folders are made as needed. To change part of a file, use edit_file instead.")
    @Tier(value = Risk.WRITE, what = "writes a file in the project's folder")
    @Returns("{file, characters, mode}")
    public Map<String, Object> writeFile(@Tools.Param("the file; a path relative to the project's folder, or a full path outside it, which the user's rules may make ask first") String path,
            @Tools.Param("the text") String text,
            @Tools.Param("create, overwrite or append; create if left out") Optional<String> mode) {
        String how = mode.orElse("create").strip().toLowerCase(Locale.ROOT);
        if (!List.of("create", "overwrite", "append").contains(how)) {
            throw new IllegalArgumentException("the mode is create, overwrite or append, not " + how);
        }
        Path f = writable(Reach.file(path));
        if (Files.isDirectory(f)) {
            throw new IllegalArgumentException(path + " is a folder");
        }
        if (how.equals("create") && Files.exists(f)) {
            throw new IllegalArgumentException(path + " exists already: overwrite it, append to it, or pick another name");
        }
        try {
            long have = Files.exists(f) && how.equals("append") ? Files.size(f) : 0;
            if (have + text.length() > MAX_FILE) {
                throw new IllegalArgumentException("files written by agents are kept under " + MAX_FILE + " characters");
            }
            checkpoint(f);
            Files.createDirectories(f.getParent());
            if (how.equals("append")) {
                Files.writeString(f, text, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            } else {
                Files.writeString(f, text, StandardCharsets.UTF_8);
            }
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("file", path);
            out.put("characters", Files.size(f));
            out.put("mode", how);
            return out;
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
    }

    /**
     * One replacement in a file.
     *
     * @param find the passage as it is now
     * @param replace what replaces it
     * @param replaceAll whether every place it appears is replaced
     */
    public record Change(@Tools.Param("the passage as it is now, exactly, without line numbers") String find,
            @Tools.Param("what replaces it; empty deletes it") String replace,
            @Tools.Param(value = "true to replace every place it appears; it must appear once if left out",
                    optional = true) Boolean replaceAll) {
    }

    /**
     * Edits a file.
     *
     * @param path the file
     * @param find the passage
     * @param replace its replacement
     * @param replaceAll whether every place is replaced
     * @return what changed, with the lines around it
     */
    @Tools.Tool(name = "edit_file", value = "Replaces a passage of a text file in the project's folder. The passage "
            + "must appear exactly once, unless replaceAll is true; quote enough of it to make it unique. An empty "
            + "replacement deletes it. The answer shows the changed lines, numbered, so there is no need to read the "
            + "file again. For several changes to one file use multi_edit.")
    @Tier(value = Risk.WRITE, what = "changes a file in the project's folder")
    @Returns("text: where it changed, then the changed lines with 3 lines around them, numbered")
    public String editFile(@Tools.Param("the file; a path relative to the project's folder, or a full path outside it, which the user's rules may make ask first") String path,
            @Tools.Param("the passage as it is now, exactly, without line numbers") String find,
            @Tools.Param("what replaces it") String replace,
            @Tools.Param("true to replace every place it appears") Optional<Boolean> replaceAll) {
        return edit(path, List.of(new Change(find, replace, replaceAll.orElse(false))));
    }

    /**
     * Makes several edits to one file.
     *
     * @param path the file
     * @param edits the changes, in order
     * @return what changed
     */
    @Tools.Tool(name = "multi_edit", value = "Makes several replacements in one file, in order, all or none: if one "
            + "passage is missing or not unique, nothing changes. Each later edit sees the file as the earlier ones "
            + "left it.")
    @Tier(value = Risk.WRITE, what = "changes a file in the project's folder")
    @Returns("text: where each edit changed, then the changed lines, numbered")
    public String multiEdit(@Tools.Param("the file; a path relative to the project's folder, or a full path outside it, which the user's rules may make ask first") String path,
            @Tools.Param("the changes, in order") List<Change> edits) {
        if (edits == null || edits.isEmpty()) {
            throw new IllegalArgumentException("give at least one edit");
        }
        return edit(path, edits);
    }

    private String edit(String path, List<Change> edits) {
        Path f = writable(Reach.file(path));
        if (!Files.isRegularFile(f)) {
            throw new IllegalArgumentException("there is no file " + path + " in the project's folder" + nearby(path));
        }
        try {
            String text = Files.readString(f, StandardCharsets.UTF_8);
            String now = text;
            // where each edit landed, as [start, end) of the new text after all edits
            List<int[]> spans = new ArrayList<>();
            List<String> said = new ArrayList<>();
            for (int n = 0; n < edits.size(); n++) {
                Change c = edits.get(n);
                String which = edits.size() > 1 ? "edit " + (n + 1) + ": " : "";
                if (c.find() == null || c.find().isEmpty()) {
                    throw new IllegalArgumentException(which + "quote the passage to replace");
                }
                String rep = c.replace() == null ? "" : c.replace();
                int at = now.indexOf(c.find());
                if (at < 0) {
                    throw new IllegalArgumentException(which + "the passage is not in " + path + closest(now, c.find())
                            + (edits.size() > 1 ? "; nothing was changed" : ""));
                }
                boolean all = Boolean.TRUE.equals(c.replaceAll());
                if (!all && now.indexOf(c.find(), at + 1) >= 0) {
                    int count = count(now, c.find());
                    throw new IllegalArgumentException(which + "the passage appears " + count + " times in " + path
                            + " (first at line " + lineOf(now, at) + "); quote more of it, or pass replaceAll"
                            + (edits.size() > 1 ? "; nothing was changed" : ""));
                }
                StringBuilder b = new StringBuilder();
                int from = 0;
                int times = 0;
                int shift = 0;
                List<int[]> mine = new ArrayList<>();
                for (int i = at; i >= 0; i = all ? now.indexOf(c.find(), i + c.find().length()) : -1) {
                    b.append(now, from, i).append(rep);
                    mine.add(new int[] {i + shift, i + shift + rep.length()});
                    shift += rep.length() - c.find().length();
                    from = i + c.find().length();
                    times++;
                }
                b.append(now.substring(from));
                // earlier spans move with this edit's changes before them
                for (int[] s : spans) {
                    int delta = 0;
                    for (int[] m : mine) {
                        if (m[0] < s[0]) {
                            delta += (m[1] - m[0]) - c.find().length();
                        }
                    }
                    s[0] += delta;
                    s[1] += delta;
                }
                spans.addAll(mine);
                now = b.toString();
                said.add(which + "line " + lineOf(now, mine.getFirst()[0]) + (times > 1 ? ", " + times + " places" : ""));
            }
            if (now.length() > MAX_FILE) {
                throw new IllegalArgumentException("files written by agents are kept under " + MAX_FILE + " characters");
            }
            checkpoint(f);
            Files.writeString(f, now, StandardCharsets.UTF_8);
            return "changed " + path + " at " + String.join("; ", said) + "\n" + around(now, spans);
        } catch (MalformedInputException e) {
            throw new IllegalArgumentException(path + " is not a text file");
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
    }

    // the changed lines with 3 lines of context, numbered, merged where they touch; at most 80 lines
    private static String around(String text, List<int[]> spans) {
        String[] lines = text.split("\n", -1);
        List<int[]> ranges = new ArrayList<>();
        for (int[] s : spans) {
            int a = Math.max(1, lineOf(text, s[0]) - 3);
            int b = Math.min(lines.length, lineOf(text, Math.max(s[0], s[1] - 1)) + 3);
            ranges.add(new int[] {a, b});
        }
        ranges.sort(Comparator.comparingInt(r -> r[0]));
        Deque<int[]> merged = new ArrayDeque<>();
        for (int[] r : ranges) {
            if (!merged.isEmpty() && r[0] <= merged.peekLast()[1] + 1) {
                merged.peekLast()[1] = Math.max(merged.peekLast()[1], r[1]);
            } else {
                merged.add(r);
            }
        }
        StringBuilder b = new StringBuilder();
        int shown = 0;
        for (int[] r : merged) {
            if (shown > 0) {
                b.append("…\n");
            }
            for (int i = r[0]; i <= r[1]; i++) {
                if (shown++ >= 80) {
                    return b.append("[more changes; read the file to see them]\n").toString();
                }
                String l = lines[i - 1];
                b.append(i).append('\t').append(l.length() > 300 ? l.substring(0, 300) + "…" : l).append('\n');
            }
        }
        return b.toString();
    }

    private static int lineOf(String text, int at) {
        int line = 1;
        for (int i = 0; i < at && i < text.length(); i++) {
            if (text.charAt(i) == '\n') {
                line++;
            }
        }
        return line;
    }

    private static int count(String text, String find) {
        int n = 0;
        for (int i = text.indexOf(find); i >= 0; i = text.indexOf(find, i + find.length())) {
            n++;
        }
        return n;
    }

    // the line most like the passage's first line, to show the agent what is really there
    private static String closest(String text, String find) {
        String first = find.strip().lines().findFirst().orElse("").strip();
        if (first.length() < 4) {
            return "; read the file and quote it exactly";
        }
        String[] lines = text.split("\n", -1);
        int best = -1;
        int bestScore = Integer.MAX_VALUE;
        for (int i = 0; i < lines.length; i++) {
            String l = lines[i].strip();
            if (l.isEmpty() || Math.abs(l.length() - first.length()) > first.length()) {
                continue;
            }
            int d = distance(l, first, bestScore);
            if (d < bestScore) {
                bestScore = d;
                best = i;
            }
        }
        if (best < 0 || bestScore > first.length() / 2) {
            return "; read the file and quote it exactly";
        }
        String l = lines[best];
        return "; the closest line is " + (best + 1) + ": " + (l.length() > 200 ? l.substring(0, 200) + "…" : l)
                + " (whitespace and case count)";
    }

    // edit distance, giving up above a limit
    private static int distance(String a, String b, int limit) {
        if (a.length() > 400 || b.length() > 400) {
            return a.equals(b) ? 0 : Integer.MAX_VALUE;
        }
        int[] prev = new int[b.length() + 1];
        int[] cur = new int[b.length() + 1];
        for (int j = 0; j <= b.length(); j++) {
            prev[j] = j;
        }
        for (int i = 1; i <= a.length(); i++) {
            cur[0] = i;
            int min = cur[0];
            for (int j = 1; j <= b.length(); j++) {
                int cost = a.charAt(i - 1) == b.charAt(j - 1) ? 0 : 1;
                cur[j] = Math.min(Math.min(cur[j - 1] + 1, prev[j] + 1), prev[j - 1] + cost);
                min = Math.min(min, cur[j]);
            }
            if (min > limit) {
                return Integer.MAX_VALUE;
            }
            int[] t = prev;
            prev = cur;
            cur = t;
        }
        return prev[b.length()];
    }

    /**
     * Moves or renames a file or folder.
     *
     * @param from where it is
     * @param to where it goes
     * @return what moved
     */
    @Tools.Tool(name = "move_file", value = "Moves or renames a file in the project's folder. Folders on the way are "
            + "made; an existing file at the target is not replaced. The run can be undone, which moves it back.")
    @Tier(value = Risk.WRITE, what = "moves a file in the project's folder")
    @Returns("text: what moved where")
    public String moveFile(@Tools.Param("the file as it is now") String from,
            @Tools.Param("where it goes, with its new name") String to) {
        Path a = writable(Reach.file(from));
        Path b = writable(Reach.file(to));
        if (!Files.isRegularFile(a)) {
            throw new IllegalArgumentException("there is no file " + from + nearby(from)
                    + (Files.isDirectory(a) ? "; move the files of a folder one at a time" : ""));
        }
        if (Files.exists(b)) {
            throw new IllegalArgumentException(to + " exists already; delete it first or pick another name");
        }
        try {
            checkpoint(a);
            checkpoint(b);
            Files.createDirectories(b.getParent());
            Files.move(a, b);
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
        return "moved " + from + " to " + to;
    }

    /**
     * Deletes a file.
     *
     * @param path the file
     * @return what happened
     */
    @Tools.Tool(name = "delete_file", value = "Deletes a file in the project's folder. The run can be undone, which "
            + "brings it back.")
    @Tier(value = Risk.WRITE, what = "deletes a file in the project's folder")
    @Returns("text: the file it deleted")
    public String deleteFile(@Tools.Param("the file; a path relative to the project's folder, or a full path outside it, which the user's rules may make ask first") String path) {
        Path f = writable(Reach.file(path));
        if (!Files.isRegularFile(f)) {
            throw new IllegalArgumentException("there is no file " + path + " in the project's folder" + nearby(path));
        }
        try {
            checkpoint(f);
            Files.delete(f);
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
        return "deleted " + path;
    }

    // ---------------------------------------------------------------- searching

    /**
     * Searches files for text.
     *
     * @param query the text or pattern
     * @param glob which files
     * @param type which kind of files
     * @param regex whether the query is a pattern
     * @param caseSensitive whether case matters
     * @param context lines around each match
     * @param mode lines, files or count
     * @param folder where to search
     * @param offset where to start
     * @return the matches
     */
    @Tools.Tool(name = "grep", value = "Finds text in the project's files, like ripgrep: each match with its file "
            + "and line, and context lines around it if asked. mode 'files' lists only the files, 'count' the matches "
            + "per file. Build output, caches, binary files and what .gitignore names are skipped. Then read_file "
            + "with startLine reads around a match. To find passages by meaning, use search_knowledge.")
    @Tier(value = Risk.READ, resultChars = 16_000)
    @Returns("{matches: [{file, line, text, before?, after?}] | files: [path] | counts: [{file, count}], total, next?, note?}")
    public Map<String, Object> grep(@Tools.Param("the text to find, or a regular expression if regex is true") String query,
            @Tools.Param("which files, such as '*.java' or 'src/**/*.kt'; every file if left out") Optional<String> glob,
            @Tools.Param("a kind of file, such as java, kotlin, js, ts, py, md, gradle; instead of glob") Optional<String> type,
            @Tools.Param("true if the query is a regular expression") Optional<Boolean> regex,
            @Tools.Param("true if case matters; it does not if left out") Optional<Boolean> caseSensitive,
            @Tools.Param("how many lines to show before and after each match, at most 10; none if left out") Optional<Integer> context,
            @Tools.Param("lines, files or count; lines if left out") Optional<String> mode,
            @Tools.Param("the folder to search; the project's folder if left out") Optional<String> folder,
            @Tools.Param("the result to start at, from 'next' of the last answer; 0 if left out") Optional<Integer> offset) {
        if (query == null || query.isEmpty()) {
            throw new IllegalArgumentException("say what to find");
        }
        Path root = folder(folder);
        boolean cs = caseSensitive.orElse(false);
        Predicate<String> hit;
        if (regex.orElse(false)) {
            Pattern p;
            try {
                p = Pattern.compile(query, cs ? 0 : Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE);
            } catch (java.util.regex.PatternSyntaxException e) {
                throw new IllegalArgumentException("the pattern does not read: " + e.getDescription()
                        + "; or pass regex false to find the text as it is");
            }
            hit = l -> p.matcher(l).find();
        } else if (cs) {
            hit = l -> l.contains(query);
        } else {
            String q = query.toLowerCase(Locale.ROOT);
            hit = l -> l.toLowerCase(Locale.ROOT).contains(q);
        }
        String how = mode.orElse("lines").strip().toLowerCase(Locale.ROOT);
        if (!List.of("lines", "files", "count").contains(how)) {
            throw new IllegalArgumentException("the mode is lines, files or count, not " + how);
        }
        if (glob.isPresent() && !glob.get().isBlank() && type.isPresent() && !type.get().isBlank()) {
            throw new IllegalArgumentException("give glob or type, not both");
        }
        String g = type.filter(t -> !t.isBlank()).map(FileWalk::typeGlob).orElse(glob.orElse(""));
        PathMatcher match = g.isBlank() ? null : FileWalk.glob(g);
        int around = Math.clamp(context.orElse(0), 0, 10);
        int from = Math.max(0, offset.orElse(0));
        int limit = how.equals("lines") ? (around > 0 ? 50 : 150) : 300;
        List<Map<String, Object>> found = new ArrayList<>();
        int[] total = {0};
        int[] searched = {0};
        new FileWalk(root).files((f, rel) -> {
            if (match != null && !match.matches(rel)) {
                return true;
            }
            if (++searched[0] > SEARCH_FILES) {
                return false;
            }
            try {
                if (Files.size(f) > SEARCH_BYTES || !FileWalk.text(f)) {
                    return true;
                }
            } catch (IOException e) {
                return true;
            }
            List<String> lines;
            try {
                lines = Files.readAllLines(f, StandardCharsets.UTF_8);
            } catch (IOException | UncheckedIOException e) {
                return true;   // not UTF-8 text
            }
            String name = FileWalk.name(FileWalk.shown(root, rel));
            int inFile = 0;
            for (int i = 0; i < lines.size(); i++) {
                if (!hit.test(lines.get(i))) {
                    continue;
                }
                inFile++;
                if (how.equals("lines")) {
                    if (total[0]++ >= from && found.size() < limit) {
                        Map<String, Object> m = new LinkedHashMap<>();
                        m.put("file", name);
                        m.put("line", i + 1);
                        m.put("text", cut(lines.get(i)));
                        if (around > 0) {
                            List<String> before = new ArrayList<>();
                            for (int j = Math.max(0, i - around); j < i; j++) {
                                before.add(cut(lines.get(j)));
                            }
                            List<String> after = new ArrayList<>();
                            for (int j = i + 1; j <= Math.min(lines.size() - 1, i + around); j++) {
                                after.add(cut(lines.get(j)));
                            }
                            m.put("before", before);
                            m.put("after", after);
                        }
                        found.add(m);
                    }
                }
            }
            if (inFile > 0 && !how.equals("lines")) {
                if (total[0]++ >= from && found.size() < limit) {
                    found.add(how.equals("files") ? Map.of("file", name) : Map.of("file", name, "count", inFile));
                }
            }
            return true;
        });
        Map<String, Object> out = new LinkedHashMap<>();
        switch (how) {
            case "files" -> out.put("files", found.stream().map(m -> m.get("file")).toList());
            case "count" -> out.put("counts", found);
            default -> out.put("matches", found);
        }
        out.put("total", total[0]);
        if (from + found.size() < total[0]) {
            out.put("next", from + found.size());
        }
        if (searched[0] > SEARCH_FILES) {
            out.put("note", "stopped after " + SEARCH_FILES + " files; narrow the folder, glob or type");
        } else if (total[0] == 0) {
            out.put("note", "no match in " + searched[0] + " files" + (regex.orElse(false) ? "" : "; try fewer words, "
                    + "a regular expression, or search_knowledge to find by meaning"));
        }
        return out;
    }

    private static String cut(String line) {
        String l = line.strip();
        return l.length() > 240 ? l.substring(0, 240) + "…" : l;
    }

    // ---------------------------------------------------------------- helpers

    private static Path folder(Optional<String> path) {
        Path root = Reach.file(path.orElse(""));
        if (!Files.isDirectory(root)) {
            throw new IllegalArgumentException(path.orElse("") + " is not a folder" + nearby(path.orElse("")));
        }
        return root;
    }

    private static List<String> lines(Path f, String path) {
        try {
            if (Files.size(f) > READ_BYTES) {
                throw new IllegalArgumentException(path + " is over 32 MB; search it with grep instead");
            }
            if (!FileWalk.text(f)) {
                throw new IllegalArgumentException(path + " is not a text file"
                        + (path.matches("(?i).*\\.(png|jpe?g|gif|webp|bmp)$") ? "; describe_image looks at pictures" : ""));
            }
            String text = Files.readString(f, StandardCharsets.UTF_8);
            List<String> out = new ArrayList<>(text.lines().toList());
            return out;
        } catch (MalformedInputException e) {
            throw new IllegalArgumentException(path + " is not a UTF-8 text file");
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
    }

    // names close to a missing path, so the agent can correct itself
    static String nearby(String path) {
        try {
            Path p = Reach.locate(path);
            if (p == null) {
                return "";
            }
            Path dir = p.getParent();
            while (dir != null && !Files.isDirectory(dir)) {
                dir = dir.getParent();
            }
            if (dir == null || !Reach.inside(dir) && !dir.equals(Reach.root())) {
                return "";
            }
            String want = p.getFileName() == null ? "" : p.getFileName().toString().toLowerCase(Locale.ROOT);
            List<String> names;
            try (Stream<Path> s = Files.list(dir)) {
                names = s.map(x -> x.getFileName().toString()).toList();
            }
            List<String> close = names.stream()
                    .sorted(Comparator.comparingInt(n -> distance(n.toLowerCase(Locale.ROOT), want, 1000)))
                    .limit(5).toList();
            if (close.isEmpty()) {
                return "";
            }
            Path root = Reach.root();
            String in = root != null && dir.startsWith(root) ? FileWalk.name(root.relativize(dir)) : dir.toString();
            return "; in " + (in.isEmpty() ? "the top folder" : in) + " there are: " + String.join(", ", close);
        } catch (IOException | RuntimeException e) {
            return "";
        }
    }

    // git's own folder holds settings that run programs: agents change it only through the git tools
    static Path writable(Path f) {
        for (Path part : f) {
            if (part.toString().equalsIgnoreCase(".git")) {
                throw new IllegalArgumentException("files inside .git are changed only through the git tools");
            }
        }
        return f;
    }

    private void checkpoint(Path f) throws IOException {
        Checkpointed.before(checkpoints, f);
    }

    // the project the call belongs to
    static Project project() {
        if (ChatService.SCOPE.isBound() && ChatService.SCOPE.get() instanceof Project p) {
            return p;
        }
        throw new IllegalArgumentException("this call belongs to no project");
    }
}
