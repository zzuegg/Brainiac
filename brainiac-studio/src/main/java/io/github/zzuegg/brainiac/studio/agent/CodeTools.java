package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.llm.Tools;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * Tools for code: git in the project's repository, and commands run in a container on a copy of
 * the project, whose changes are copied back with checkpoints so the run can be undone.
 */
@ToolGroup(id = "code", summary = "git and commands run in the sandbox")
public final class CodeTools implements RiskRefiner, Checkpointed {

    private static final int OUT = 12_000;
    private static final long COPY_BYTES = 2L << 30;
    private static final int COPY_FILES = 50_000;

    private final Services services;
    private volatile Checkpoints checkpoints;

    /**
     * Creates the tools.
     *
     * @param services the sandbox settings
     */
    public CodeTools(Services services) {
        this.services = services;
    }

    @Override
    public void checkpoints(Checkpoints c) {
        this.checkpoints = c;
    }

    @Override
    public Risk refine(String tool, Map<String, Object> arguments, Risk declared) {
        return declared;
    }

    @Override
    public List<Reach.Access> reach(String tool, Map<String, Object> arguments) {
        boolean write = switch (tool) {
            case "git_status", "git_diff", "git_log" -> false;
            default -> true;
        };
        if (!tool.startsWith("git_")) {
            return List.of();
        }
        List<Reach.Access> out = Reach.of(arguments, write, "repo");
        if (out.isEmpty() && Reach.root() != null) {
            out = List.of(new Reach.Access(Reach.root(), write));
        }
        return out;
    }

    // ---- git

    @Tools.Tool(name = "git_status", value = "Says the repository's branch and which files are changed, staged or new.")
    @Tier(value = Risk.READ)
    @Returns("text: git's short status")
    public String gitStatus(@Tools.Param("the repository's folder; the project's folder if left out") Optional<String> repo) {
        return git(repo, OUT, "status", "--short", "--branch");
    }

    @Tools.Tool(name = "git_diff", value = "Shows what changed in the repository: the working files against the last "
            + "commit, the staged changes, or against another commit or branch.")
    @Tier(value = Risk.READ, resultChars = 16_000)
    @Returns("text: the diff")
    public String gitDiff(@Tools.Param("the repository's folder; the project's folder if left out") Optional<String> repo,
            @Tools.Param("true for what is staged for the next commit") Optional<Boolean> staged,
            @Tools.Param("a commit or branch to compare with, such as 'main'") Optional<String> against,
            @Tools.Param("only this file or folder") Optional<String> path) {
        List<String> a = new ArrayList<>(List.of("diff", "--no-ext-diff", "--no-textconv", "--no-color", "--stat", "--patch"));
        if (staged.orElse(false)) {
            a.add("--cached");
        }
        against.filter(s -> !s.isBlank()).ifPresent(s -> a.add(ref(s)));
        a.add("--");
        path.filter(s -> !s.isBlank()).ifPresent(a::add);
        return git(repo, 15_000, a.toArray(String[]::new));
    }

    @Tools.Tool(name = "git_log", value = "Lists the repository's last commits, one line each.")
    @Tier(value = Risk.READ)
    @Returns("text: the log")
    public String gitLog(@Tools.Param("the repository's folder; the project's folder if left out") Optional<String> repo,
            @Tools.Param("how many, 20 if left out, at most 200") Optional<Integer> count,
            @Tools.Param("only commits touching this file or folder") Optional<String> path) {
        List<String> a = new ArrayList<>(List.of("log", "--no-color", "--date=short",
                "--pretty=format:%h %ad %an: %s", "-n", String.valueOf(Math.clamp(count.orElse(20), 1, 200)), "--"));
        path.filter(s -> !s.isBlank()).ifPresent(a::add);
        return git(repo, OUT, a.toArray(String[]::new));
    }

    @Tools.Tool(name = "git_branch", value = "Switches the repository to a branch, making it first if 'create' is "
            + "true. Uncommitted changes go along.")
    @Tier(value = Risk.WRITE, what = "switches the repository's branch")
    @Returns("text: git's answer")
    public String gitBranch(@Tools.Param("the repository's folder; the project's folder if left out") Optional<String> repo,
            @Tools.Param("the branch") String name,
            @Tools.Param("true to make a new branch") Optional<Boolean> create) {
        String n = ref(name);
        return create.orElse(false) ? git(repo, OUT, "switch", "-c", n) : git(repo, OUT, "switch", n);
    }

    @Tools.Tool(name = "git_commit", value = "Commits changes in the repository: the files named, or every change if "
            + "none are named. Hooks do not run. Nothing is pushed.")
    @Tier(value = Risk.CODE, what = "commits to the repository")
    @Returns("text: git's answer and the new commit")
    public String gitCommit(@Tools.Param("the repository's folder; the project's folder if left out") Optional<String> repo,
            @Tools.Param("the commit message: a short first line, then details if needed") String message,
            @Tools.Param("the files to commit; every change if left out") Optional<List<String>> files) {
        if (message == null || message.isBlank()) {
            throw new IllegalArgumentException("write a commit message");
        }
        List<String> add = new ArrayList<>(List.of("add"));
        if (files.isPresent() && !files.get().isEmpty()) {
            add.add("--");
            add.addAll(files.get());
        } else {
            add.add("-A");
        }
        git(repo, OUT, add.toArray(String[]::new));
        String out = git(repo, OUT, "commit", "--no-verify", "-m", message.strip());
        return out + "\n" + git(repo, 200, "log", "-1", "--pretty=format:committed %h on %D");
    }

    @Tools.Tool(name = "git_push", value = "Pushes a branch to a remote, such as GitHub, under the user's name. Never "
            + "forces. Always asks the user first.")
    @Tier(value = Risk.OUTWARD, what = "pushes commits to a remote under your name")
    @Returns("text: git's answer")
    public String gitPush(@Tools.Param("the repository's folder; the project's folder if left out") Optional<String> repo,
            @Tools.Param("the remote, 'origin' if left out") Optional<String> remote,
            @Tools.Param("the branch; the current one if left out") Optional<String> branch) {
        String b = branch.filter(s -> !s.isBlank()).map(CodeTools::ref)
                .orElseGet(() -> git(repo, 200, "rev-parse", "--abbrev-ref", "HEAD").strip());
        if (b.equals("HEAD")) {
            throw new IllegalArgumentException("the repository is on no branch; switch to one first");
        }
        return git(repo, OUT, "push", "-u", ref(remote.orElse("origin")), b);
    }

    private static String ref(String s) {
        String r = s == null ? "" : s.strip();
        if (r.isEmpty() || r.startsWith("-") || r.contains("..") || !r.matches("[A-Za-z0-9._/@^~-]+")) {
            throw new IllegalArgumentException("not a branch, remote or commit name: " + s);
        }
        return r;
    }

    private String git(Optional<String> repo, int cap, String... args) {
        Path dir = Reach.file(repo.filter(s -> !s.isBlank()).orElse(""));
        if (!Files.isDirectory(dir)) {
            throw new IllegalArgumentException(repo.orElse("") + " is not a folder");
        }
        List<String> cmd = new ArrayList<>(List.of("git", "-C", dir.toString(),
                // what an agent cannot have set up: hooks, fsmonitor and diff programs run nothing
                "-c", "core.hooksPath=" + dir.resolve(".git").resolve("no-hooks"), "-c", "core.fsmonitor=false",
                "-c", "protocol.ext.allow=never", "-c", "color.ui=false"));
        cmd.addAll(Arrays.asList(args));
        try {
            Proc.Result r = Proc.run(cmd, dir, Map.of("GIT_TERMINAL_PROMPT", "0", "GIT_PAGER", "cat", "PAGER", "cat"),
                    300, cap);
            if (r.timedOut()) {
                throw new IllegalStateException("git took longer than 5 minutes");
            }
            if (r.code() != 0) {
                throw new IllegalArgumentException("git " + args[0] + " failed: " + r.out().strip());
            }
            return r.out().isBlank() ? "(nothing)" : r.out();
        } catch (IOException e) {
            throw new UncheckedIOException("cannot run git: is it installed? " + e.getMessage(), e);
        }
    }

    // ---- the sandbox

    /**
     * Runs a command in a container.
     *
     * @param command the command
     * @param seconds the time limit
     * @return what it printed and changed
     */
    @Tools.Tool(name = "run_command", value = "Runs a shell command in a container on a copy of the project's folder, "
            + "such as building or testing code, with no network unless the settings allow it. Files it changes are "
            + "copied back and can be undone with the run. Nothing runs on this PC directly.")
    @Tier(value = Risk.CODE, what = "runs a command in a container")
    @Returns("{exitCode, output, changed?, note?}")
    public Map<String, Object> runCommand(@Tools.Param("the command, run by sh in the project's folder") String command,
            @Tools.Param("the time limit in seconds; the settings' limit if left out, at most 3600") Optional<Integer> seconds) {
        if (command == null || command.isBlank()) {
            throw new IllegalArgumentException("say what to run");
        }
        Path root = Reach.file("");
        String engine = engine();
        String image = services.get("sandbox", "image").orElse("debian:stable-slim");
        int limit = Math.clamp(seconds.orElse(Integer.parseInt(services.get("sandbox", "timeout").orElse("600"))), 5, 3600);
        Path copy = null;
        try {
            copy = Files.createTempDirectory("brainiac-sandbox");
            Map<String, Long> before = copy(root, copy);
            List<String> cmd = new ArrayList<>(List.of(engine, "run", "--rm",
                    "--network", services.get("sandbox", "network").filter(n -> n.equals("on") || n.equals("bridge"))
                            .map(n -> "bridge").orElse("none"),
                    "--cpus", services.get("sandbox", "cpus").orElse("2"),
                    "--memory", services.get("sandbox", "memory").orElse("4g"),
                    "--pids-limit", "512", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                    "-v", copy.toAbsolutePath() + ":/work", "-w", "/work"));
            Optional<String> user = owner(copy);
            user.ifPresent(u -> {
                cmd.add("--user");
                cmd.add(u);
            });
            cmd.addAll(List.of(image, "sh", "-c", command));
            Proc.Result r = Proc.run(cmd, null, Map.of(), limit + 30L, OUT);
            List<String> changed = applyBack(copy, root, before);
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("exitCode", r.code());
            if (r.timedOut()) {
                out.put("note", "stopped after " + limit + " seconds");
            }
            out.put("output", r.out());
            out.put("changed", changed);
            return out;
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        } finally {
            if (copy != null) {
                delete(copy);
            }
        }
    }

    private String engine() {
        String kind = services.get("sandbox", "kind").orElse("");
        if (!kind.isBlank()) {
            if (!List.of("docker", "podman").contains(kind)) {
                throw new IllegalArgumentException("the sandbox is docker or podman, not " + kind);
            }
            return kind;
        }
        for (String k : List.of("docker", "podman")) {
            String path = System.getenv("PATH");
            if (path != null) {
                for (String d : path.split(java.io.File.pathSeparator)) {
                    if (Files.isExecutable(Path.of(d, k)) || Files.isExecutable(Path.of(d, k + ".exe"))) {
                        return k;
                    }
                }
            }
        }
        throw new IllegalArgumentException("commands run in a container, and neither Docker nor Podman was found; "
                + "install one, or set the sandbox on the settings page");
    }

    private static Optional<String> owner(Path p) {
        try {
            Object uid = Files.getAttribute(p, "unix:uid");
            Object gid = Files.getAttribute(p, "unix:gid");
            return Optional.of(uid + ":" + gid);
        } catch (IOException | UnsupportedOperationException | IllegalArgumentException e) {
            return Optional.empty();
        }
    }

    private static boolean skipped(Path rel) {
        return rel.getNameCount() > 0 && rel.getName(0).toString().equals(".git");
    }

    // copies the project, without .git, and says each file's size and time
    private static Map<String, Long> copy(Path from, Path to) throws IOException {
        Map<String, Long> seen = new LinkedHashMap<>();
        long[] bytes = {0};
        Files.walkFileTree(from, new SimpleFileVisitor<>() {
            @Override
            public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException {
                Path rel = from.relativize(dir);
                if (skipped(rel)) {
                    return FileVisitResult.SKIP_SUBTREE;
                }
                Files.createDirectories(to.resolve(rel.toString()));
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                if (!attrs.isRegularFile()) {
                    return FileVisitResult.CONTINUE;   // links are not followed out of the folder
                }
                bytes[0] += attrs.size();
                if (seen.size() >= COPY_FILES || bytes[0] > COPY_BYTES) {
                    throw new IllegalArgumentException("the project's folder is too big to copy into the sandbox ("
                            + COPY_FILES + " files or " + (COPY_BYTES >> 30) + " GB at most)");
                }
                Path rel = from.relativize(file);
                Path t = to.resolve(rel.toString());
                Files.copy(file, t, StandardCopyOption.COPY_ATTRIBUTES);
                seen.put(rel.toString().replace('\\', '/'), Files.getLastModifiedTime(t).toMillis() ^ (attrs.size() << 20));
                return FileVisitResult.CONTINUE;
            }
        });
        return seen;
    }

    // copies what the command changed back into the project, checkpointing each file first
    private List<String> applyBack(Path copy, Path root, Map<String, Long> before) throws IOException {
        List<String> changed = new ArrayList<>();
        java.util.Set<String> present = new java.util.HashSet<>();
        try (Stream<Path> s = Files.walk(copy)) {
            for (Path f : (Iterable<Path>) s.filter(p -> Files.isRegularFile(p, LinkOption.NOFOLLOW_LINKS))::iterator) {
                Path rel = copy.relativize(f);
                if (skipped(rel)) {
                    continue;
                }
                String key = rel.toString().replace('\\', '/');
                present.add(key);
                long sig = Files.getLastModifiedTime(f).toMillis() ^ (Files.size(f) << 20);
                Long was = before.get(key);
                if (was != null && was == sig) {
                    continue;
                }
                Path target = root.resolve(rel.toString()).normalize();
                if (!target.startsWith(root)) {
                    continue;
                }
                if (was != null && Files.isRegularFile(target) && Files.mismatch(f, target) == -1) {
                    continue;
                }
                Checkpointed.before(checkpoints, target);
                Files.createDirectories(target.getParent());
                Files.copy(f, target, StandardCopyOption.REPLACE_EXISTING);
                changed.add((was == null ? "new " : "changed ") + key);
            }
        }
        for (String key : before.keySet()) {
            if (!present.contains(key)) {
                Path target = root.resolve(key).normalize();
                if (target.startsWith(root) && Files.isRegularFile(target)) {
                    Checkpointed.before(checkpoints, target);
                    Files.delete(target);
                    changed.add("deleted " + key);
                }
            }
        }
        changed.sort(Comparator.comparing(c -> c.substring(c.indexOf(' ') + 1).toLowerCase(Locale.ROOT)));
        return changed.size() > 300 ? List.copyOf(changed.subList(0, 300)) : changed;
    }

    private static void delete(Path dir) {
        try (Stream<Path> s = Files.walk(dir)) {
            for (Path p : s.sorted(Comparator.reverseOrder()).toList()) {
                try {
                    Files.deleteIfExists(p);
                } catch (IOException ignored) {
                    // left for the system's temp cleanup
                }
            }
        } catch (IOException ignored) {
            // nothing to delete
        }
    }
}
