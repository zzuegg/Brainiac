package io.github.zzuegg.brainiac.studio.agent;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/** Runs an outside program, never through a shell, with a time limit and a cap on what it prints. */
final class Proc {

    /**
     * What a program did.
     *
     * @param code its exit code, or -1 if it was stopped
     * @param out what it printed, cut to the cap
     * @param cut whether the output was cut
     * @param timedOut whether it ran out of time
     */
    record Result(int code, String out, boolean cut, boolean timedOut) {
    }

    private Proc() {
    }

    /**
     * Runs a program.
     *
     * @param cmd the program and its arguments
     * @param work the folder it runs in
     * @param env extra environment
     * @param seconds the time limit
     * @param cap the most characters kept, the start and end of the output
     * @return what it did
     * @throws IOException if it cannot be started
     */
    static Result run(List<String> cmd, Path work, Map<String, String> env, long seconds, int cap) throws IOException {
        ProcessBuilder b = new ProcessBuilder(cmd).redirectErrorStream(true);
        if (work != null) {
            b.directory(work.toFile());
        }
        b.environment().putAll(env);
        b.redirectInput(ProcessBuilder.Redirect.from(new java.io.File(System.getProperty("os.name").startsWith("Windows")
                ? "NUL" : "/dev/null")));
        Process p = b.start();
        ByteArrayOutputStream buf = new ByteArrayOutputStream();
        long[] total = {0};
        Thread reader = Thread.ofVirtual().start(() -> {
            byte[] chunk = new byte[8192];
            try (InputStream in = p.getInputStream()) {
                int k;
                while ((k = in.read(chunk)) > 0) {
                    synchronized (buf) {
                        total[0] += k;
                        if (buf.size() < 4 * cap) {
                            buf.write(chunk, 0, Math.min(k, 4 * cap - buf.size()));
                        }
                    }
                }
            } catch (IOException ignored) {
                // the process ended
            }
        });
        boolean done;
        try {
            done = p.waitFor(seconds, TimeUnit.SECONDS);
            if (!done) {
                p.descendants().forEach(ProcessHandle::destroyForcibly);
                p.destroyForcibly();
                p.waitFor(5, TimeUnit.SECONDS);
            }
            reader.join(2000);
        } catch (InterruptedException e) {
            p.descendants().forEach(ProcessHandle::destroyForcibly);
            p.destroyForcibly();
            Thread.currentThread().interrupt();
            throw new AgentRuns.Stopped();
        }
        String text;
        long all;
        synchronized (buf) {
            text = buf.toString(StandardCharsets.UTF_8);
            all = total[0];
        }
        boolean cut = all > text.length() || text.length() > cap;
        if (text.length() > cap) {
            text = text.substring(0, cap / 3) + "\n…[" + (text.length() - cap) + " characters left out]…\n"
                    + text.substring(text.length() - 2 * cap / 3);
        }
        return new Result(done ? p.exitValue() : -1, text, cut, !done);
    }
}
