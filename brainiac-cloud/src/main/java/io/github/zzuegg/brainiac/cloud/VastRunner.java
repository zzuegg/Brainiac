package io.github.zzuegg.brainiac.cloud;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Runs Brainiac on a rented instance over SSH: ships a checkout there, runs
 * {@code cloud/vast/setup.sh} jobs and starts a studio worker behind an SSH tunnel.
 *
 * <p>Uses the {@code ssh} and {@code git} commands (both come with Windows 10 and later, and
 * with Git for Windows). The SSH key is the one added to the Vast account; {@code ssh} finds it
 * in {@code ~/.ssh} unless {@link #identity} names another.
 */
public final class VastRunner {

    /** Where the checkout goes on the instance. */
    public static final String REMOTE_DIR = "/root/brainiac";

    private final Path repo;
    private final Path identity;
    private final Path knownHosts;

    /**
     * Creates a runner.
     *
     * @param repo the Brainiac checkout to ship (its {@code HEAD} commit is sent)
     * @param identity the SSH private key, or null for ssh's default keys
     * @throws IOException if the temporary known-hosts file cannot be made
     */
    public VastRunner(Path repo, Path identity) throws IOException {
        this.repo = Objects.requireNonNull(repo);
        this.identity = identity;
        // every instance has a new host key on a reused proxy address: a throwaway known-hosts
        // file keeps them out of ~/.ssh/known_hosts
        this.knownHosts = Files.createTempFile("brainiac-vast-known-hosts", "");
        this.knownHosts.toFile().deleteOnExit();
    }

    /**
     * Builds an {@code ssh} command line for an instance.
     *
     * @param i the instance
     * @param options more ssh options, such as {@code -L 7870:127.0.0.1:7870}
     * @param remote the command to run there
     * @return the command line
     */
    public List<String> ssh(VastInstance i, List<String> options, String remote) {
        List<String> c = new ArrayList<>(List.of("ssh", "-p", String.valueOf(i.sshPort()),
                "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                "-o", "UserKnownHostsFile=" + knownHosts, "-o", "ServerAliveInterval=30",
                "-o", "ConnectTimeout=20", "-o", "LogLevel=ERROR"));
        if (identity != null) {
            c.addAll(List.of("-i", identity.toString(), "-o", "IdentitiesOnly=yes"));
        }
        c.addAll(options);
        c.add("root@" + i.sshHost());
        if (remote != null) {
            c.add(remote);
        }
        return c;
    }

    /**
     * Waits until SSH logins work; Vast's SSH proxy takes a little while after the container starts.
     *
     * @param i the running instance
     * @param timeout how long to keep trying
     * @throws IOException if no login worked in time
     * @throws InterruptedException if the wait is interrupted
     */
    public void awaitSsh(VastInstance i, Duration timeout) throws IOException, InterruptedException {
        long end = System.nanoTime() + timeout.toNanos();
        while (true) {
            Process p = new ProcessBuilder(ssh(i, List.of(), "true")).redirectErrorStream(true).start();
            String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8).strip();
            if (p.waitFor() == 0) {
                return;
            }
            if (System.nanoTime() > end) {
                throw new IOException("cannot log in to instance " + i.id() + " over SSH: " + out
                        + "\nIs the SSH key added at https://cloud.vast.ai/manage-keys/ the one ssh uses here?");
            }
            Thread.sleep(5000);
        }
    }

    /**
     * Copies the checkout's {@code HEAD} commit to {@link #REMOTE_DIR}, replacing what was there.
     * Uncommitted changes are not sent. The archive is made locally first, uploaded with
     * {@code cat} (checked by size, up to three tries: Vast's SSH proxy sometimes drops a
     * connection in the first minutes) and unpacked in a second step.
     *
     * @param i the instance
     * @throws IOException if git fails or every upload failed
     * @throws InterruptedException if interrupted
     */
    public void ship(VastInstance i) throws IOException, InterruptedException {
        Path tar = Files.createTempFile("brainiac-src", ".tar");
        try {
            Process git = new ProcessBuilder("git", "archive", "--format=tar", "-o", tar.toString(), "HEAD")
                    .directory(repo.toFile()).redirectErrorStream(true).start();
            String gitOut = new String(git.getInputStream().readAllBytes(), StandardCharsets.UTF_8).strip();
            if (git.waitFor() != 0) {
                throw new IOException("git archive failed: " + gitOut);
            }
            long size = Files.size(tar);
            String last = "";
            for (int attempt = 1; attempt <= 3; attempt++) {
                long t = System.nanoTime();
                Process up = new ProcessBuilder(ssh(i, List.of(), "cat > /root/brainiac-src.tar && stat -c %s /root/brainiac-src.tar"))
                        .redirectInput(tar.toFile()).redirectErrorStream(true).start();
                String out = new String(up.getInputStream().readAllBytes(), StandardCharsets.UTF_8).strip();
                int code = up.waitFor();
                if (code == 0 && out.endsWith(String.valueOf(size))) {
                    System.out.printf(java.util.Locale.ROOT, "  uploaded %.1f MB in %.0f s%n", size / 1e6,
                            (System.nanoTime() - t) / 1e9);
                    String unpack = "rm -rf " + REMOTE_DIR + " && mkdir -p " + REMOTE_DIR + " && tar -xf /root/brainiac-src.tar -C "
                            + REMOTE_DIR + " && rm /root/brainiac-src.tar";
                    Process x = new ProcessBuilder(ssh(i, List.of(), unpack)).redirectErrorStream(true).start();
                    String xOut = new String(x.getInputStream().readAllBytes(), StandardCharsets.UTF_8).strip();
                    if (x.waitFor() == 0) {
                        return;
                    }
                    last = "unpacking failed: " + xOut;
                } else {
                    last = "upload attempt " + attempt + " failed (ssh exit " + code + ", " + size + " bytes sent): " + out;
                }
                System.err.println("  " + last);
                Thread.sleep(15000);
            }
            throw new IOException("copying the checkout failed: " + last);
        } finally {
            Files.deleteIfExists(tar);
        }
    }

    /**
     * Runs a {@code setup.sh} job and copies its output to this process's output and a log.
     *
     * @param i the instance
     * @param job {@code check}, {@code smoke}, {@code tests}, {@code bench} or {@code run}
     * @param args the job's arguments (for {@code run}, the command)
     * @param log a file that gets a copy of the output, or null
     * @return the job's exit code: 0 when it worked, 2 when the machine cannot run Brainiac
     * @throws IOException if ssh cannot be started
     * @throws InterruptedException if interrupted
     */
    public int run(VastInstance i, String job, List<String> args, Path log) throws IOException, InterruptedException {
        // the command goes in as a script on ssh's input, so no shell (cmd.exe included) re-quotes it
        Process p = new ProcessBuilder(ssh(i, List.of(), "bash -s")).redirectErrorStream(true).start();
        try (OutputStream in = p.getOutputStream()) {
            in.write((remote(job, args) + "\n").getBytes(StandardCharsets.UTF_8));
        }
        copy(p.getInputStream(), log);
        return p.waitFor();
    }

    /**
     * Starts a studio worker on the instance behind an SSH tunnel: the worker listens on the
     * instance's 127.0.0.1 only, and {@code http://127.0.0.1:<localPort>} here reaches it. The
     * token goes over the SSH connection's input, so it is in no command line or log.
     *
     * @param i the instance
     * @param token the worker's bearer token, at least 32 characters
     * @param localPort the port on this machine
     * @param args more {@code setup.sh worker} arguments, such as {@code --models /workspace/models}
     * @return the ssh process; the worker and the tunnel stop when it is destroyed
     * @throws IOException if ssh cannot be started
     */
    public Process startWorker(VastInstance i, String token, int localPort, List<String> args) throws IOException {
        if (token == null || token.length() < 32) {
            throw new IllegalArgumentException("a worker token has at least 32 characters");
        }
        // bash reads its script from the pipe a line at a time, so read takes the next line: the token
        String script = "IFS= read -r BRAINIAC_WORKER_TOKEN && export BRAINIAC_WORKER_TOKEN && " + remote("worker", args)
                + "\n" + token + "\n";
        Process p = new ProcessBuilder(ssh(i, List.of("-L", localPort + ":127.0.0.1:7870", "-o", "ExitOnForwardFailure=yes"),
                "bash -s")).redirectErrorStream(true).start();
        try (OutputStream in = p.getOutputStream()) {
            in.write(script.getBytes(StandardCharsets.UTF_8));
        }
        return p;
    }

    /**
     * Stops a studio worker on the instance. Ending the local ssh process alone can leave the
     * worker running there, holding GPU memory.
     *
     * @param i the instance
     * @throws IOException if ssh cannot be started
     * @throws InterruptedException if interrupted
     */
    public void stopWorker(VastInstance i) throws IOException, InterruptedException {
        Process p = new ProcessBuilder(ssh(i, List.of(), "pkill -f brainiac.studio; true"))
                .redirectErrorStream(true).start();
        p.getInputStream().readAllBytes();
        p.waitFor();
    }

    /**
     * Copies a process's output to this process's output and, if given, a log file.
     *
     * @param in the output to copy
     * @param log the log file, or null
     * @throws IOException if the log cannot be written
     */
    public static void copy(InputStream in, Path log) throws IOException {
        OutputStream file = log == null ? OutputStream.nullOutputStream() : Files.newOutputStream(log);
        try (in; file) {
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) > 0) {
                System.out.write(buf, 0, n);
                System.out.flush();
                file.write(buf, 0, n);
                file.flush();
            }
        }
    }

    private static String remote(String job, List<String> args) {
        StringBuilder c = new StringBuilder("cd " + REMOTE_DIR + " && bash cloud/vast/setup.sh " + quote(job));
        for (String a : args) {
            c.append(' ').append(quote(a));
        }
        return c.toString();
    }

    /** Quotes a word for the remote shell. */
    static String quote(String s) {
        return s.matches("[A-Za-z0-9_./:=,+@%-]+") ? s : "'" + s.replace("'", "'\\''") + "'";
    }
}
