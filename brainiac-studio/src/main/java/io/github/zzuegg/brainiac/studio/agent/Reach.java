package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.project.Project;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;

/**
 * Where a tool call reaches on disk. Paths inside the project's folder need nothing more; paths
 * outside it are reached only when the rules allowed the call, or the user approved it.
 */
public final class Reach {

    /** The paths outside the project's folder the current call was allowed to reach. */
    static final ScopedValue<Set<Path>> ALLOWED = ScopedValue.newInstance();

    /**
     * A path a call reads or writes.
     *
     * @param path the absolute, normalized path
     * @param write whether it is changed
     */
    public record Access(Path path, boolean write) {
    }

    private Reach() {
    }

    /**
     * Turns what an agent wrote into an absolute path: relative paths are inside the project's
     * folder, {@code ~} is the user's home.
     *
     * @param path the path as given
     * @return the absolute path, or null if it is relative and the call has no project folder
     */
    public static Path locate(String path) {
        String p = path == null ? "" : path.strip().replace('\\', '/');
        if (p.equals("~") || p.startsWith("~/")) {
            return Path.of(System.getProperty("user.home")).resolve(p.length() > 2 ? p.substring(2) : "").toAbsolutePath()
                    .normalize();
        }
        Path given = Path.of(p);
        if (given.isAbsolute()) {
            return given.normalize();
        }
        Path root = root();
        return root == null ? null : root.resolve(p).normalize();
    }

    /**
     * The project's folder, if the call belongs to a project that has one.
     *
     * @return the folder, absolute and normalized, or null
     */
    public static Path root() {
        if (ChatService.SCOPE.isBound() && ChatService.SCOPE.get() instanceof Project pr && pr.folder() != null) {
            return pr.folder().toAbsolutePath().normalize();
        }
        return null;
    }

    /**
     * Whether a path is inside the project's folder, links followed.
     *
     * @param p an absolute path
     * @return true if inside
     */
    public static boolean inside(Path p) {
        Path root = root();
        if (root == null || p == null || !p.startsWith(root)) {
            return false;
        }
        try {
            return !Files.exists(p) || p.toRealPath().startsWith(root.toRealPath());
        } catch (IOException e) {
            return false;
        }
    }

    /**
     * Resolves a file a tool was asked to read or write: inside the project's folder always,
     * outside only if this call was allowed to reach it.
     *
     * @param path the path as given
     * @return the file
     * @throws IllegalArgumentException if the path is outside and the call was not allowed there
     */
    public static Path file(String path) {
        Path p = locate(path);
        if (p == null) {
            FileTools.project().inFolder(path);   // says why: no project, or no folder
            throw new IllegalArgumentException("this call has no project folder");
        }
        Path root = root();
        if (root != null && p.startsWith(root)) {
            return ((Project) ChatService.SCOPE.get()).inFolder(root.relativize(p).toString());
        }
        if (ALLOWED.isBound()) {
            for (Path a : ALLOWED.get()) {
                if (p.startsWith(a)) {
                    return p;
                }
            }
        }
        throw new IllegalArgumentException(path + " is outside the project's folder, and this call was not allowed "
                + "to reach it");
    }

    /**
     * Lists what a call reaches, from the arguments named.
     *
     * @param arguments the call's arguments
     * @param write whether the call changes them
     * @param keys the arguments holding paths
     * @return the accesses
     */
    static List<Access> of(java.util.Map<String, Object> arguments, boolean write, String... keys) {
        java.util.List<Access> out = new java.util.ArrayList<>();
        for (String k : keys) {
            Object v = arguments.get(k);
            if (v instanceof String s && !s.isBlank()) {
                Path p = locate(s);
                if (p != null) {
                    out.add(new Access(p, write));
                }
            }
        }
        return out;
    }
}
