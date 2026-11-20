package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.llm.Tools;
import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.project.Project;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.MalformedInputException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.PathMatcher;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * Tools that write in the project's folder. Every change is checkpointed first, so a run can be
 * undone as a whole; nothing outside the folder can be reached.
 */
public final class FileTools implements RiskRefiner, Checkpointed {

    /** The most characters a file written by an agent may have. */
    static final int MAX_FILE = 2_000_000;

    private static final long SEARCH_BYTES = 4L << 20;
    private static final int SEARCH_FILES = 5000;

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
            case "write_file", "edit_file", "delete_file" -> Reach.of(arguments, true, "path");
            case "search_files" -> Reach.of(arguments, false, "folder");
            default -> List.of();
        };
    }

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
            + "at a time. Folders are made as needed.")
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
     * Edits a file.
     *
     * @param path the file
     * @param find the passage
     * @param replace its replacement
     * @return what changed
     */
    @Tools.Tool(name = "edit_file", value = "Replaces one passage of a text file in the project's folder. The passage "
            + "must appear exactly once; quote enough of it to make it unique. An empty replacement deletes it.")
    @Tier(value = Risk.WRITE, what = "changes a file in the project's folder")
    @Returns("text: the file and line it changed")
    public String editFile(@Tools.Param("the file; a path relative to the project's folder, or a full path outside it, which the user's rules may make ask first") String path,
            @Tools.Param("the passage as it is now, exactly") String find,
            @Tools.Param("what replaces it") String replace) {
        Path f = writable(Reach.file(path));
        if (!Files.isRegularFile(f)) {
            throw new IllegalArgumentException("there is no file " + path + " in the project's folder");
        }
        if (find == null || find.isEmpty()) {
            throw new IllegalArgumentException("quote the passage to replace");
        }
        try {
            String text = Files.readString(f, StandardCharsets.UTF_8);
            int at = text.indexOf(find);
            if (at < 0) {
                throw new IllegalArgumentException("the passage is not in " + path + "; read the file and quote it exactly");
            }
            if (text.indexOf(find, at + 1) >= 0) {
                throw new IllegalArgumentException("the passage appears more than once in " + path + "; quote more of it");
            }
            String changed = text.substring(0, at) + replace + text.substring(at + find.length());
            if (changed.length() > MAX_FILE) {
                throw new IllegalArgumentException("files written by agents are kept under " + MAX_FILE + " characters");
            }
            checkpoint(f);
            Files.writeString(f, changed, StandardCharsets.UTF_8);
            int line = (int) text.substring(0, at).chars().filter(c -> c == '\n').count() + 1;
            return "changed " + path + " at line " + line;
        } catch (MalformedInputException e) {
            throw new IllegalArgumentException(path + " is not a text file");
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
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
            throw new IllegalArgumentException("there is no file " + path + " in the project's folder");
        }
        try {
            checkpoint(f);
            Files.delete(f);
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
        return "deleted " + path;
    }

    /**
     * Searches files.
     *
     * @param query the words
     * @param glob which files
     * @param regex whether the query is a pattern
     * @return the matching lines
     */
    @Tools.Tool(name = "search_files", value = "Finds lines holding some text in the project's folder, with the file "
            + "and line number of each; then read_project_file reads around them.")
    @Tier(value = Risk.READ, resultChars = 12_000)
    @Returns("{matches: [...], files, note?}")
    public Map<String, Object> searchFiles(@Tools.Param("the text to find; case does not matter") String query,
            @Tools.Param("which files, such as '**/*.md'; every file if left out") Optional<String> glob,
            @Tools.Param("true if the query is a regular expression") Optional<Boolean> regex,
            @Tools.Param("the folder to search; the project's folder if left out") Optional<String> folder) {
        if (query == null || query.isBlank()) {
            throw new IllegalArgumentException("say what to find");
        }
        Path root = Reach.file(folder.orElse(""));
        if (!Files.isDirectory(root)) {
            throw new IllegalArgumentException(folder.orElse("") + " is not a folder");
        }
        java.util.function.Predicate<String> hit;
        if (regex.orElse(false)) {
            java.util.regex.Pattern p = java.util.regex.Pattern.compile(query, java.util.regex.Pattern.CASE_INSENSITIVE);
            hit = l -> p.matcher(l).find();
        } else {
            String q = query.toLowerCase(Locale.ROOT);
            hit = l -> l.toLowerCase(Locale.ROOT).contains(q);
        }
        PathMatcher match = glob.isPresent() && !glob.get().isBlank()
                ? FileSystems.getDefault().getPathMatcher("glob:" + glob.get()) : null;
        List<Map<String, Object>> found = new ArrayList<>();
        int files = 0;
        int searched = 0;
        try (Stream<Path> s = Files.walk(root)) {
            for (Path p : (Iterable<Path>) s.filter(Files::isRegularFile)::iterator) {
                Path rel = root.relativize(p);
                if (rel.toString().startsWith(".git") || (match != null && !match.matches(rel))) {
                    continue;
                }
                if (++searched > SEARCH_FILES || found.size() >= 200) {
                    break;
                }
                if (Files.size(p) > SEARCH_BYTES) {
                    continue;
                }
                List<String> lines;
                try {
                    lines = Files.readAllLines(p, StandardCharsets.UTF_8);
                } catch (MalformedInputException e) {
                    continue;   // not text
                }
                boolean any = false;
                for (int i = 0; i < lines.size() && found.size() < 200; i++) {
                    if (hit.test(lines.get(i))) {
                        String l = lines.get(i).strip();
                        found.add(Map.of("file", rel.toString().replace('\\', '/'), "line", i + 1,
                                "text", l.length() > 200 ? l.substring(0, 200) + "…" : l));
                        any = true;
                    }
                }
                if (any) {
                    files++;
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("matches", found);
        out.put("files", files);
        if (found.size() >= 200) {
            out.put("note", "the first 200 matches; narrow the query or the files");
        }
        return out;
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
