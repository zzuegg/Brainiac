package io.github.zzuegg.brainiac.studio.agent;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.PathMatcher;
import java.nio.file.LinkOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.function.BiPredicate;
import java.util.stream.Stream;

/**
 * Walks a folder the way a programmer would look through it: build output, caches, dependency
 * folders and whatever the top-level {@code .gitignore} names are skipped, so searches and
 * listings show the code rather than what is made from it.
 */
final class FileWalk {

    /** Folders skipped wherever they are. */
    static final Set<String> SKIPPED = Set.of(".git", ".gradle", ".idea", ".vscode", "build", "out", "target",
            "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", "dist", ".next",
            ".kotlin", ".cache", ".brainiac");

    private final Path root;
    private final List<PathMatcher> ignored = new ArrayList<>();

    /**
     * Prepares a walk.
     *
     * @param root the folder
     */
    FileWalk(Path root) {
        this.root = root;
        Path gi = root.resolve(".gitignore");
        if (Files.isRegularFile(gi)) {
            try {
                for (String line : Files.readAllLines(gi, StandardCharsets.UTF_8)) {
                    String l = line.strip();
                    if (l.isEmpty() || l.startsWith("#") || l.startsWith("!")) {
                        continue;
                    }
                    l = l.startsWith("/") ? l.substring(1) : "**/" + l;
                    if (l.endsWith("/")) {
                        l = l.substring(0, l.length() - 1);
                    }
                    try {
                        ignored.add(FileSystems.getDefault().getPathMatcher("glob:" + l));
                        ignored.add(FileSystems.getDefault().getPathMatcher("glob:" + l.replaceFirst("^\\*\\*/", "")));
                    } catch (RuntimeException e) {
                        // a pattern Java's globs cannot read: not ignored
                    }
                }
            } catch (IOException | UncheckedIOException e) {
                // unreadable: nothing extra is ignored
            }
        }
    }

    /**
     * Whether a path inside the root is skipped.
     *
     * @param rel the path relative to the root
     * @param folder whether it is a folder
     * @return true to skip it
     */
    boolean skipped(Path rel, boolean folder) {
        if (rel.getNameCount() == 0 || rel.toString().isEmpty()) {
            return false;
        }
        if (folder && SKIPPED.contains(rel.getFileName().toString())) {
            return true;
        }
        for (PathMatcher m : ignored) {
            if (m.matches(rel)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Visits every file that is not skipped, depth first and in name order, until the visitor says stop.
     * Links to folders are not followed.
     *
     * @param visit gets each file and its path relative to the root; returns false to stop
     */
    void files(BiPredicate<Path, Path> visit) {
        walk(root, visit);
    }

    // depth first, folders and files each in name order; false once the visitor said stop
    private boolean walk(Path dir, BiPredicate<Path, Path> visit) {
        List<Path> entries;
        try (Stream<Path> s = Files.list(dir)) {
            entries = s.sorted().toList();
        } catch (IOException | UncheckedIOException e) {
            return true;   // unreadable: skipped
        }
        for (Path p : entries) {
            Path rel = root.relativize(p);
            if (Files.isDirectory(p, LinkOption.NOFOLLOW_LINKS)) {
                if (!skipped(rel, true) && !walk(p, visit)) {
                    return false;
                }
            } else if (Files.isRegularFile(p) && !skipped(rel, false) && !visit.test(p, rel)) {
                return false;
            }
        }
        return true;
    }

    /**
     * A path as agents see it: relative, with forward slashes.
     *
     * @param rel the path
     * @return its text
     */
    static String name(Path rel) {
        return rel.toString().replace('\\', '/');
    }

    /**
     * Whether a file looks like text: no zero byte in its first 8 KB.
     *
     * @param f the file
     * @return true for text
     */
    static boolean text(Path f) {
        try (InputStream in = Files.newInputStream(f)) {
            byte[] b = in.readNBytes(8192);
            for (byte x : b) {
                if (x == 0) {
                    return false;
                }
            }
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    /**
     * A glob that matches paths relative to the root, where a pattern without a slash matches the
     * file's name in any folder, as in {@code *.java}.
     *
     * @param glob the pattern
     * @return the matcher
     */
    static PathMatcher glob(String glob) {
        String g = glob.strip();
        if (!g.contains("/")) {
            PathMatcher name = FileSystems.getDefault().getPathMatcher("glob:" + g);
            return p -> p.getFileName() != null && name.matches(p.getFileName());
        }
        PathMatcher full = FileSystems.getDefault().getPathMatcher("glob:" + g);
        // '**/x' should also match 'x' at the top
        PathMatcher top = g.startsWith("**/") ? FileSystems.getDefault().getPathMatcher("glob:" + g.substring(3)) : null;
        return p -> full.matches(p) || (top != null && top.matches(p));
    }

    /**
     * File types by their usual extensions, for the type filter of grep.
     *
     * @param type a type such as java or js
     * @return a glob for its files
     */
    static String typeGlob(String type) {
        return switch (type.strip().toLowerCase(Locale.ROOT)) {
            case "java" -> "*.java";
            case "kotlin", "kt" -> "*.{kt,kts}";
            case "js", "javascript" -> "*.{js,mjs,cjs,jsx}";
            case "ts", "typescript" -> "*.{ts,tsx}";
            case "py", "python" -> "*.py";
            case "md", "markdown" -> "*.{md,markdown}";
            case "json" -> "*.json";
            case "html" -> "*.{html,htm}";
            case "css" -> "*.{css,scss}";
            case "xml" -> "*.xml";
            case "yaml", "yml" -> "*.{yaml,yml}";
            case "gradle" -> "*.{gradle,gradle.kts}";
            case "c" -> "*.{c,h}";
            case "cpp", "c++" -> "*.{cpp,cc,cxx,hpp,hh,h}";
            case "cuda" -> "*.{cu,cuh}";
            case "rust", "rs" -> "*.rs";
            case "go" -> "*.go";
            case "sh", "shell" -> "*.{sh,bash}";
            default -> throw new IllegalArgumentException("unknown type " + type + "; use glob instead, such as '*."
                    + type + "'");
        };
    }
}
