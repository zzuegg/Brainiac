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
    private final Path sandboxes;
    private volatile Checkpoints checkpoints;

    /**
     * Creates the tools, keeping the sandbox's copies in the system's temporary folder.
     *
     * @param services the sandbox settings
     */
    public CodeTools(Services services) {
        this(services, Path.of(System.getProperty("java.io.tmpdir"), "brainiac-sandbox"));
    }

    /**
     * Creates the tools.
     *
     * @param services the sandbox settings
     * @param sandboxes where the sandbox keeps each project's copy and the download caches
     */
    public CodeTools(Services services, Path sandboxes) {
        this.services = services;
        this.sandboxes = sandboxes;
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
            case "git_status", "git_diff", "git_log", "git_show", "git_blame" -> false;
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

    @Tools.Tool(name = "git_log", value = "Lists the repository's last commits, one line each, optionally only "
            + "those touching a file, by an author, since a date or with words in their message.")
    @Tier(value = Risk.READ)
    @Returns("text: the log")
    public String gitLog(@Tools.Param("the repository's folder; the project's folder if left out") Optional<String> repo,
            @Tools.Param("how many, 20 if left out, at most 200") Optional<Integer> count,
            @Tools.Param("only commits touching this file or folder") Optional<String> path,
            @Tools.Param("only commits whose message holds these words") Optional<String> grep,
            @Tools.Param("only commits by this author, a part of the name or email") Optional<String> author,
            @Tools.Param("only commits since a date, such as '2026-11-01' or '2 weeks ago'") Optional<String> since) {
        List<String> a = new ArrayList<>(List.of("log", "--no-color", "--date=short",
                "--pretty=format:%h %ad %an: %s", "-n", String.valueOf(Math.clamp(count.orElse(20), 1, 200))));
        grep.filter(s -> !s.isBlank()).ifPresent(g -> a.add("--grep=" + g.strip()));
        author.filter(s -> !s.isBlank()).ifPresent(w -> a.add("--author=" + w.strip()));
        since.filter(s -> !s.isBlank()).ifPresent(w -> a.add("--since=" + w.strip()));
        a.add("-i");
        a.add("--");
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

    @Tools.Tool(name = "git_show", value = "Shows one commit: its message, the files it changed and its diff; or, "
            + "given a file, that file as it was at the commit.")
    @Tier(value = Risk.READ, resultChars = 16_000)
    @Returns("text: the commit and its diff, or the file's text at that commit")
    public String gitShow(@Tools.Param("the repository's folder; the project's folder if left out") Optional<String> repo,
            @Tools.Param("a commit, branch or tag, such as 'HEAD~1' or 'a1b2c3d'") String commit,
            @Tools.Param("a file, to see it as it was at the commit") Optional<String> path) {
        String c = ref(commit);
        if (path.isPresent() && !path.get().isBlank()) {
            String p = path.get().strip().replace('\\', '/');
            if (p.startsWith("/") || p.contains("..") || p.startsWith("-")) {
                throw new IllegalArgumentException("give the file relative to the repository, such as 'src/App.java'");
            }
            return git(repo, 15_000, "show", "--no-color", c + ":" + p);
        }
        return git(repo, 15_000, "show", "--no-color", "--no-ext-diff", "--no-textconv", "--stat", "--patch",
                "--format=commit %H%nAuthor: %an%nDate:   %ad%n%n%B", c, "--");
    }

    @Tools.Tool(name = "git_blame", value = "Says who last changed each line of a file, in which commit and when; "
            + "for a range of lines if given. git_show then shows the commit.")
    @Tier(value = Risk.READ, resultChars = 16_000)
    @Returns("text: one line per line of the file: commit, author, date, line number, text")
    public String gitBlame(@Tools.Param("the repository's folder; the project's folder if left out") Optional<String> repo,
            @Tools.Param("the file, relative to the repository") String path,
            @Tools.Param("the first line, 1 if left out") Optional<Integer> startLine,
            @Tools.Param("the last line; 200 lines on if left out") Optional<Integer> endLine) {
        String p = path == null ? "" : path.strip().replace('\\', '/');
        if (p.isEmpty() || p.startsWith("-")) {
            throw new IllegalArgumentException("give the file, such as 'src/App.java'");
        }
        int from = Math.max(1, startLine.orElse(1));
        int to = Math.max(from, endLine.orElse(from + 199));
        return git(repo, 15_000, "blame", "--date=short", "-L", from + "," + to, "--", p);
    }

    @Tools.Tool(name = "git_restore", value = "Puts files back as they were at the last commit, or at another "
            + "commit: drops uncommitted changes to them. Only the files or folder named. The run can be undone, "
            + "which brings the changes back.")
    @Tier(value = Risk.WRITE, what = "puts files back as they were committed")
    @Returns("text: the files put back")
    public String gitRestore(@Tools.Param("the repository's folder; the project's folder if left out") Optional<String> repo,
            @Tools.Param("the files or folders, relative to the repository") List<String> paths,
            @Tools.Param("the commit to take them from; the last commit if left out") Optional<String> commit) {
        if (paths == null || paths.isEmpty()) {
            throw new IllegalArgumentException("name the files to put back; '.' for every file");
        }
        String c = ref(commit.filter(s -> !s.isBlank()).orElse("HEAD"));
        List<String> files = new ArrayList<>();
        for (String p : paths) {
            String q = p.strip().replace('\\', '/');
            if (q.isEmpty() || q.startsWith("-") || q.contains("..")) {
                throw new IllegalArgumentException("not a path in the repository: " + p);
            }
            files.add(q);
        }
        Path dir = Reach.file(repo.filter(s -> !s.isBlank()).orElse(""));
        List<String> diff = new ArrayList<>(List.of("diff", "--name-only", c, "--"));
        diff.addAll(files);
        String changed = git(repo, 50_000, diff.toArray(String[]::new));
        List<String> touched = changed.equals("(nothing)") ? List.of() : changed.lines().filter(l -> !l.isBlank()).toList();
        try {
            for (String f : touched) {
                Checkpointed.before(checkpoints, dir.resolve(f));
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
        List<String> restore = new ArrayList<>(List.of("restore", "--source=" + c, "--worktree", "--staged", "--"));
        restore.addAll(files);
        git(repo, OUT, restore.toArray(String[]::new));
        return touched.isEmpty() ? "nothing differed from " + c
                : "put back " + touched.size() + " file(s) as at " + c + ": " + String.join(", ",
                        touched.subList(0, Math.min(40, touched.size()))) + (touched.size() > 40 ? ", …" : "");
    }

    @Tools.Tool(name = "git_fetch", value = "Fetches new commits and branches from a remote, such as GitHub, without "
            + "changing your files or branch. Then git_log or git_diff against 'origin/main' shows what is new, and "
            + "git_merge brings it in.")
    @Tier(value = Risk.WRITE, what = "fetches from a remote into the repository")
    @Returns("text: git's answer")
    public String gitFetch(@Tools.Param("the repository's folder; the project's folder if left out") Optional<String> repo,
            @Tools.Param("the remote, 'origin' if left out") Optional<String> remote) {
        String out = git(repo, OUT, "fetch", "--prune", ref(remote.orElse("origin")));
        return out.equals("(nothing)") ? "up to date" : out;
    }

    @Tools.Tool(name = "git_merge", value = "Merges a branch into the current one, such as an agent's branch or "
            + "'origin/main'. If files conflict, the merge stops with the conflicts marked in them: fix each with "
            + "edit_file and finish with git_commit, or call again with abort true.")
    @Tier(value = Risk.WRITE, what = "merges a branch into the current one")
    @Returns("text: git's answer, or the files in conflict")
    public String gitMerge(@Tools.Param("the repository's folder; the project's folder if left out") Optional<String> repo,
            @Tools.Param("the branch to merge in") Optional<String> branch,
            @Tools.Param("true to give up a merge that stopped on conflicts") Optional<Boolean> abort) {
        if (abort.orElse(false)) {
            git(repo, OUT, "merge", "--abort");
            return "the merge was given up; the files are as before it";
        }
        String b = ref(branch.orElseThrow(() -> new IllegalArgumentException("name the branch to merge in")));
        Path dir = Reach.file(repo.filter(s -> !s.isBlank()).orElse(""));
        String incoming = git(repo, 50_000, "diff", "--name-only", "HEAD..." + b);
        try {
            for (String f : incoming.equals("(nothing)") ? List.<String>of() : incoming.lines().toList()) {
                if (!f.isBlank()) {
                    Checkpointed.before(checkpoints, dir.resolve(f));
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
        try {
            return git(repo, OUT, "merge", "--no-edit", "--no-verify", b);
        } catch (IllegalArgumentException failed) {
            String conflicts = git(repo, OUT, "diff", "--name-only", "--diff-filter=U");
            if (conflicts.equals("(nothing)")) {
                throw failed;
            }
            return "the merge stopped: these files conflict, with <<<<<<< ======= >>>>>>> marking each place:\n"
                    + conflicts.strip() + "\nfix them with edit_file, then git_commit; or git_merge with abort true";
        }
    }

    @Tools.Tool(name = "git_worktree_remove", value = "Removes the folder of an agent's worktree, made by "
            + "spawn_agent with worktree true, once its branch is merged or not wanted. The branch stays.")
    @Tier(value = Risk.WRITE, what = "removes an agent's worktree folder")
    @Returns("text: what was removed")
    public String gitWorktreeRemove(@Tools.Param("the repository's folder; the project's folder if left out") Optional<String> repo,
            @Tools.Param("the worktree's branch, such as 'agent/3f2a'") String branch,
            @Tools.Param("true to remove it even with uncommitted changes") Optional<Boolean> force) {
        String b = ref(branch);
        String list = git(repo, 50_000, "worktree", "list", "--porcelain");
        String path = null;
        String at = null;
        for (String l : list.lines().toList()) {
            if (l.startsWith("worktree ")) {
                at = l.substring("worktree ".length());
            } else if (l.equals("branch refs/heads/" + b)) {
                path = at;
            }
        }
        if (path == null) {
            throw new IllegalArgumentException("no worktree is on the branch " + b);
        }
        if (force.orElse(false)) {
            git(repo, OUT, "worktree", "remove", "--force", path);
        } else {
            git(repo, OUT, "worktree", "remove", path);
        }
        return "removed the worktree " + path + "; the branch " + b + " stays";
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

    /** Folders of build output, caches and dependencies: they stay in the sandbox's copy. */
    static final java.util.Set<String> OUTPUTS = java.util.Set.of(".gradle", "build", "out", "target", "node_modules",
            "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".next", ".kotlin", ".cache");

    /** Where the sandbox keeps its logs, inside the project's folder. */
    static final String LOGS = ".brainiac/logs";

    private static final int KEEP_LOGS = 20;
    private static final java.util.concurrent.ConcurrentHashMap<String, Object> LOCKS =
            new java.util.concurrent.ConcurrentHashMap<>();

    /**
     * What a command did in the sandbox.
     *
     * @param result its exit code and output
     * @param changed the files it changed in the project
     * @param log the log file, relative to the project's folder
     * @param copy the sandbox's copy of the project, while it is still locked
     */
    record Ran(Proc.Result result, List<String> changed, String log, Path copy) {
    }

    /**
     * Runs a command in a container.
     *
     * @param command the command
     * @param seconds the time limit
     * @param bringBack build output to copy back
     * @return what it printed and changed
     */
    @Tools.Tool(name = "run_command", value = "Runs a shell command in a container on the project's files, such as "
            + "building or testing code, with no network unless the settings allow it. The container's copy of the "
            + "project is kept between calls, with build output (build/, target/, node_modules/) and download caches, "
            + "so builds stay incremental. Changed source files are copied back and can be undone with the run; "
            + "build output stays in the copy unless bringBack names it. The whole output is saved to a log file you "
            + "can read with read_file. To build or test, prefer build and run_tests, which return the errors.")
    @Tier(value = Risk.CODE, what = "runs a command in a container")
    @Returns("{exitCode, output: the start and end, log: path, changed?, note?}")
    public Map<String, Object> runCommand(@Tools.Param("the command, run by sh in the project's folder") String command,
            @Tools.Param("the time limit in seconds; the settings' limit if left out, at most 3600") Optional<Integer> seconds,
            @Tools.Param("build output to copy back into the project, such as 'build/libs/*.jar'") Optional<String> bringBack) {
        if (command == null || command.isBlank()) {
            throw new IllegalArgumentException("say what to run");
        }
        Map<String, Object> out = new LinkedHashMap<>();
        sandbox(command, seconds, bringBack.orElse(""), copy -> { }, ran -> describe(ran, out));
        return out;
    }

    private static void describe(Ran ran, Map<String, Object> out) {
        Proc.Result r = ran.result();
        out.put("exitCode", r.code());
        if (r.timedOut()) {
            out.put("note", "stopped at the time limit");
        }
        out.put("output", r.out());
        out.put("log", ran.log());
        if (!ran.changed().isEmpty()) {
            out.put("changed", ran.changed());
        }
    }

    /**
     * Builds the project.
     *
     * @param module which module
     * @param command a command instead of the usual one
     * @param seconds the time limit
     * @return whether it built, and the errors
     */
    @Tools.Tool(name = "build", value = "Compiles the project, or one module, in the sandbox with its own build tool "
            + "(Gradle, Maven, npm, Cargo or Go, found from its files) and returns the compile errors as file, line "
            + "and message, so you can go straight to them with read_file. Faster than reading build output.")
    @Tier(value = Risk.CODE, what = "builds the project in a container")
    @Returns("{ok, tool, command, errors: [{file, line, column?, message}], warnings, exitCode, log, output?: the end, when "
            + "it failed without errors found}")
    public Map<String, Object> build(@Tools.Param("a module or subproject, such as 'brainiac-studio'; all if left out") Optional<String> module,
            @Tools.Param("a command to run instead of the usual one") Optional<String> command,
            @Tools.Param("the time limit in seconds; the settings' limit if left out, at most 3600") Optional<Integer> seconds) {
        Tool tool = Tool.detect(Reach.file(""));
        String cmd = command.filter(c -> !c.isBlank()).orElseGet(() -> tool.buildCommand(module.orElse("")));
        Map<String, Object> out = new LinkedHashMap<>();
        sandbox(cmd, seconds, "", copy -> { }, ran -> {
            List<Map<String, Object>> errors = BuildOutput.errors(ran.result().out(), ran.copy(), Reach.file(""),
                    ran.log());
            int warnings = BuildOutput.warnings(ran.result().out());
            boolean ok = ran.result().code() == 0;
            out.put("ok", ok);
            out.put("tool", tool.name);
            out.put("command", cmd);
            out.put("errors", errors);
            out.put("warnings", warnings);
            out.put("exitCode", ran.result().code());
            out.put("log", ran.log());
            if (!ok && errors.isEmpty()) {
                out.put("output", tail(ran.result().out(), 4000));
            }
        });
        return out;
    }

    /**
     * Runs tests.
     *
     * @param module which module
     * @param filter which tests
     * @param command a command instead of the usual one
     * @param seconds the time limit
     * @return the results
     */
    @Tools.Tool(name = "run_tests", value = "Runs the project's tests, or one module's, or only those matching a "
            + "filter, in the sandbox, and returns the results: how many passed, and each failure with its message, "
            + "file and line and a short stack. Reads JUnit XML reports (Gradle, Maven) where there are any; otherwise "
            + "gives the end of the output. Compile errors come back as in build.")
    @Tier(value = Risk.CODE, what = "runs the project's tests in a container")
    @Returns("{ok, tool, command, passed, failed: [{test, message, file?, line?, stack}], skipped, errors?, exitCode, "
            + "log, output?: the end, when no report was found}")
    public Map<String, Object> runTests(@Tools.Param("a module or subproject, such as 'brainiac-studio'; all if left out") Optional<String> module,
            @Tools.Param("which tests: a class or method pattern, such as '*FileToolsTest' or 'AppTest.readsFiles'") Optional<String> filter,
            @Tools.Param("a command to run instead of the usual one") Optional<String> command,
            @Tools.Param("the time limit in seconds; the settings' limit if left out, at most 3600") Optional<Integer> seconds) {
        Tool tool = Tool.detect(Reach.file(""));
        String cmd = command.filter(c -> !c.isBlank()).orElseGet(() -> tool.testCommand(module.orElse(""),
                filter.orElse("")));
        Map<String, Object> out = new LinkedHashMap<>();
        // only this run's reports are read
        sandbox(cmd, seconds, "", BuildOutput::clearReports, ran -> {
            BuildOutput.Tests t = BuildOutput.tests(ran.copy(), Reach.file(""));
            out.put("ok", ran.result().code() == 0 && t.failed().isEmpty());
            out.put("tool", tool.name);
            out.put("command", cmd);
            out.put("passed", t.passed());
            out.put("failed", t.failed());
            out.put("skipped", t.skipped());
            List<Map<String, Object>> errors = BuildOutput.errors(ran.result().out(), ran.copy(), Reach.file(""),
                    ran.log());
            if (!errors.isEmpty()) {
                out.put("errors", errors);
            }
            out.put("exitCode", ran.result().code());
            out.put("log", ran.log());
            if (!t.found() || (ran.result().code() != 0 && t.failed().isEmpty() && errors.isEmpty())) {
                out.put("output", tail(ran.result().out(), 4000));
            }
        });
        return out;
    }

    private static String tail(String text, int chars) {
        return text.length() <= chars ? text : "…" + text.substring(text.length() - chars);
    }

    /** A project's build tool and how it builds and tests. */
    static final class Tool {

        final String name;
        private final String run;

        private Tool(String name, String run) {
            this.name = name;
            this.run = run;
        }

        static Tool detect(Path root) {
            if (Files.isRegularFile(root.resolve("gradlew"))) {
                return new Tool("gradle", "sh ./gradlew");
            }
            for (String f : List.of("settings.gradle.kts", "settings.gradle", "build.gradle.kts", "build.gradle")) {
                if (Files.isRegularFile(root.resolve(f))) {
                    return new Tool("gradle", "gradle");
                }
            }
            if (Files.isRegularFile(root.resolve("pom.xml"))) {
                return new Tool("maven", Files.isRegularFile(root.resolve("mvnw")) ? "sh ./mvnw" : "mvn");
            }
            if (Files.isRegularFile(root.resolve("package.json"))) {
                return new Tool("npm", "npm");
            }
            if (Files.isRegularFile(root.resolve("Cargo.toml"))) {
                return new Tool("cargo", "cargo");
            }
            if (Files.isRegularFile(root.resolve("go.mod"))) {
                return new Tool("go", "go");
            }
            throw new IllegalArgumentException("found no Gradle, Maven, npm, Cargo or Go build in the project's folder; "
                    + "give a command, or use run_command");
        }

        String buildCommand(String module) {
            String m = safe(module);
            return switch (name) {
                case "gradle" -> run + " " + (m.isEmpty() ? "classes testClasses" : ":" + m + ":classes :" + m
                        + ":testClasses") + " --console=plain --continue";
                case "maven" -> run + " -B -q test-compile" + (m.isEmpty() ? "" : " -pl " + m + " -am");
                case "npm" -> (m.isEmpty() ? "" : "cd " + m + " && ") + "npm run build --if-present";
                case "cargo" -> "cargo build" + (m.isEmpty() ? "" : " -p " + m);
                default -> "go build ./" + (m.isEmpty() ? "" : m + "/") + "...";
            };
        }

        String testCommand(String module, String filter) {
            String m = safe(module);
            String f = filter.strip();
            if (!f.isEmpty() && !f.matches("[A-Za-z0-9_.*#$/ -]+")) {
                throw new IllegalArgumentException("a test filter holds letters, digits and . * # $ _ / - only");
            }
            return switch (name) {
                case "gradle" -> run + " " + (m.isEmpty() ? "test" : ":" + m + ":test") + " --console=plain --continue"
                        + (f.isEmpty() ? "" : " --tests '" + f + "'");
                case "maven" -> run + " -B test" + (m.isEmpty() ? "" : " -pl " + m + " -am")
                        + (f.isEmpty() ? "" : " -Dtest='" + f + "' -Dsurefire.failIfNoSpecifiedTests=false");
                case "npm" -> (m.isEmpty() ? "" : "cd " + m + " && ") + "npm test" + (f.isEmpty() ? "" : " -- '" + f + "'");
                case "cargo" -> "cargo test" + (m.isEmpty() ? "" : " -p " + m) + (f.isEmpty() ? "" : " '" + f + "'");
                default -> "go test ./" + (m.isEmpty() ? "" : m + "/") + "..." + (f.isEmpty() ? "" : " -run '" + f + "'");
            };
        }

        private static String safe(String module) {
            String m = module == null ? "" : module.strip().replace('\\', '/');
            if (m.startsWith(":")) {
                m = m.substring(1);
            }
            if (!m.isEmpty() && (!m.matches("[A-Za-z0-9._/:-]+") || m.contains(".."))) {
                throw new IllegalArgumentException("not a module name: " + module);
            }
            return m;
        }
    }

    // runs a command on the project's kept copy: syncs the project in, runs, copies changes back
    private void sandbox(String command, Optional<Integer> seconds, String bringBack,
            java.util.function.Consumer<Path> prepare, java.util.function.Consumer<Ran> after) {
        Path root = Reach.file("");
        String engine = engine();
        String image = services.get("sandbox", "image").orElse("debian:stable-slim");
        int limit = Math.clamp(seconds.orElse(Integer.parseInt(services.get("sandbox", "timeout").orElse("600"))), 5, 3600);
        String key = keyOf(root);
        Path copy = sandboxes.resolve("work").resolve(key);
        Path cache = sandboxes.resolve("cache");
        java.nio.file.PathMatcher back = bringBack.isBlank() ? null
                : java.nio.file.FileSystems.getDefault().getPathMatcher("glob:" + bringBack.strip());
        synchronized (LOCKS.computeIfAbsent(key, k -> new Object())) {
            try {
                Files.createDirectories(copy);
                Files.createDirectories(cache);
                Map<String, Long> before = sync(root, copy);
                prepare.accept(copy);
                Path logs = root.resolve(LOGS);
                Files.createDirectories(logs);
                Path ignore = root.resolve(".brainiac").resolve(".gitignore");
                if (!Files.exists(ignore)) {
                    Files.writeString(ignore, "# the studio's own files: not part of the project\n*\n");
                }
                String stamp = java.time.LocalDateTime.now().format(java.time.format.DateTimeFormatter
                        .ofPattern("yyyyMMdd-HHmmss-SSS", Locale.ROOT));
                Path log = logs.resolve(stamp + ".log");
                List<String> cmd = new ArrayList<>(List.of(engine, "run", "--rm",
                        "--network", services.get("sandbox", "network").filter(n -> n.equals("on") || n.equals("bridge"))
                                .map(n -> "bridge").orElse("none"),
                        "--cpus", services.get("sandbox", "cpus").orElse("2"),
                        "--memory", services.get("sandbox", "memory").orElse("4g"),
                        "--pids-limit", "512", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                        "-v", copy.toAbsolutePath() + ":/work", "-v", cache.toAbsolutePath() + ":/cache",
                        "-e", "HOME=/cache/home", "-e", "GRADLE_USER_HOME=/cache/gradle",
                        "-e", "MAVEN_OPTS=-Dmaven.repo.local=/cache/m2", "-e", "npm_config_cache=/cache/npm",
                        "-e", "PIP_CACHE_DIR=/cache/pip", "-e", "CARGO_HOME=/cache/cargo", "-e", "GOPATH=/cache/go",
                        "-w", "/work"));
                Optional<String> user = owner(copy);
                user.ifPresent(u -> {
                    cmd.add("--user");
                    cmd.add(u);
                });
                cmd.addAll(List.of(image, "sh", "-c", "mkdir -p /cache/home 2>/dev/null; " + command));
                Proc.Result r = Proc.run(cmd, null, Map.of(), limit + 30L, OUT, log);
                pruneLogs(logs);
                List<String> changed = applyBack(copy, root, before, back);
                after.accept(new Ran(r, changed, LOGS + "/" + log.getFileName(), copy));
            } catch (IOException e) {
                throw new UncheckedIOException(e.getMessage(), e);
            }
        }
    }

    // one kept copy per project folder
    private static String keyOf(Path root) {
        String name = root.getFileName() == null ? "project" : root.getFileName().toString().replaceAll("[^A-Za-z0-9._-]", "_");
        return name + "-" + Integer.toHexString(root.toAbsolutePath().normalize().toString().hashCode());
    }

    private static void pruneLogs(Path logs) {
        try (Stream<Path> s = Files.list(logs)) {
            List<Path> all = s.filter(p -> p.getFileName().toString().endsWith(".log")).sorted().toList();
            for (Path p : all.subList(0, Math.max(0, all.size() - KEEP_LOGS))) {
                Files.deleteIfExists(p);
            }
        } catch (IOException ignored) {
            // old logs stay
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

    // .git and the sandbox's logs are never copied; build output stays on its side
    static boolean skipped(Path rel) {
        if (rel.getNameCount() == 0) {
            return false;
        }
        String first = rel.getName(0).toString();
        return first.equals(".git") || rel.toString().replace('\\', '/').startsWith(".brainiac");
    }

    static boolean output(Path rel) {
        for (Path part : rel) {
            if (OUTPUTS.contains(part.toString())) {
                return true;
            }
        }
        return false;
    }

    private static long signature(Path f) throws IOException {
        return Files.getLastModifiedTime(f).toMillis() ^ (Files.size(f) << 20);
    }

    /**
     * Brings the kept copy up to date with the project: changed files copied in, removed ones
     * removed; build output in the copy is left alone.
     *
     * @return each source file's signature in the copy afterwards
     */
    static Map<String, Long> sync(Path root, Path copy) throws IOException {
        Map<String, Long> seen = new LinkedHashMap<>();
        long[] bytes = {0};
        Files.walkFileTree(root, new SimpleFileVisitor<>() {
            @Override
            public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) {
                Path rel = root.relativize(dir);
                return skipped(rel) || output(rel) ? FileVisitResult.SKIP_SUBTREE : FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                if (!attrs.isRegularFile()) {
                    return FileVisitResult.CONTINUE;   // links are not followed out of the folder
                }
                bytes[0] += attrs.size();
                if (seen.size() >= COPY_FILES || bytes[0] > COPY_BYTES) {
                    throw new IllegalArgumentException("the project's folder is too big to copy into the sandbox ("
                            + COPY_FILES + " files or " + (COPY_BYTES >> 30) + " GB at most, build output not counted)");
                }
                Path rel = root.relativize(file);
                Path t = copy.resolve(rel.toString());
                if (!Files.isRegularFile(t) || signature(t) != signature(file)) {
                    Files.createDirectories(t.getParent());
                    Files.copy(file, t, StandardCopyOption.COPY_ATTRIBUTES, StandardCopyOption.REPLACE_EXISTING);
                }
                seen.put(rel.toString().replace('\\', '/'), signature(t));
                return FileVisitResult.CONTINUE;
            }
        });
        // what was removed from the project goes from the copy too
        try (Stream<Path> s = Files.walk(copy)) {
            for (Path f : (Iterable<Path>) s.filter(p -> Files.isRegularFile(p, LinkOption.NOFOLLOW_LINKS))::iterator) {
                Path rel = copy.relativize(f);
                if (!skipped(rel) && !output(rel) && !seen.containsKey(rel.toString().replace('\\', '/'))) {
                    Files.deleteIfExists(f);
                }
            }
        }
        return seen;
    }

    // copies what the command changed back into the project, checkpointing each file first
    private List<String> applyBack(Path copy, Path root, Map<String, Long> before, java.nio.file.PathMatcher back)
            throws IOException {
        List<String> changed = new ArrayList<>();
        java.util.Set<String> present = new java.util.HashSet<>();
        try (Stream<Path> s = Files.walk(copy)) {
            for (Path f : (Iterable<Path>) s.filter(p -> Files.isRegularFile(p, LinkOption.NOFOLLOW_LINKS))::iterator) {
                Path rel = copy.relativize(f);
                if (skipped(rel) || (output(rel) && (back == null || !back.matches(rel)))) {
                    continue;
                }
                String key = rel.toString().replace('\\', '/');
                present.add(key);
                long sig = signature(f);
                Long was = before.get(key);
                if (was != null && was == sig) {
                    continue;
                }
                Path target = root.resolve(rel.toString()).normalize();
                if (!target.startsWith(root)) {
                    continue;
                }
                if (Files.isRegularFile(target) && Files.mismatch(f, target) == -1) {
                    continue;
                }
                Checkpointed.before(checkpoints, target);
                Files.createDirectories(target.getParent());
                Files.copy(f, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.COPY_ATTRIBUTES);
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
}
