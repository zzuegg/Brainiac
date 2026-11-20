package io.github.zzuegg.brainiac.studio.agent;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
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
        return run(cmd, work, env, seconds, cap, null);
    }

    /**
     * Runs a program, writing everything it prints to a log file.
     *
     * @param cmd the program and its arguments
     * @param work the folder it runs in
     * @param env extra environment
     * @param seconds the time limit
     * @param cap the most characters kept, the start and end of the output
     * @param log where the whole output goes, or null
     * @return what it did
     * @throws IOException if it cannot be started
     */
    static Result run(List<String> cmd, Path work, Map<String, String> env, long seconds, int cap, Path log)
            throws IOException {
        ProcessBuilder b = new ProcessBuilder(cmd).redirectErrorStream(true);
        if (work != null) {
            b.directory(work.toFile());
        }
        b.environment().putAll(env);
        b.redirectInput(ProcessBuilder.Redirect.from(new java.io.File(System.getProperty("os.name").startsWith("Windows")
                ? "NUL" : "/dev/null")));
        Process p = b.start();
        // the start and the end of the output; what lies between goes only to the log
        int headCap = Math.max(1, cap / 3);
        int tailCap = Math.max(1, 2 * cap);
        ByteArrayOutputStream head = new ByteArrayOutputStream();
        byte[] tail = new byte[tailCap];
        long[] total = {0};
        Thread reader = Thread.ofVirtual().start(() -> {
            byte[] chunk = new byte[8192];
            try (InputStream in = p.getInputStream();
                    OutputStream out = log == null ? OutputStream.nullOutputStream() : Files.newOutputStream(log)) {
                int k;
                while ((k = in.read(chunk)) > 0) {
                    out.write(chunk, 0, k);
                    synchronized (head) {
                        for (int i = 0; i < k; i++) {
                            if (head.size() < headCap) {
                                head.write(chunk[i]);
                            } else {
                                tail[(int) ((total[0] - headCap) % tailCap)] = chunk[i];
                            }
                            total[0]++;
                        }
                    }
                }
            } catch (IOException ignored) {
                // the process ended, or the log could not be written
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
        boolean cut;
        synchronized (head) {
            long all = total[0];
            byte[] h = head.toByteArray();
            long rest = all - h.length;
            if (rest <= 0) {
                text = new String(h, StandardCharsets.UTF_8);
                cut = false;
            } else {
                int kept = (int) Math.min(rest, tailCap);
                byte[] t = new byte[kept];
                long start = rest - kept;
                for (int i = 0; i < kept; i++) {
                    t[i] = tail[(int) ((start + i) % tailCap)];
                }
                String end = new String(t, StandardCharsets.UTF_8);
                cut = rest > kept;
                text = new String(h, StandardCharsets.UTF_8) + (cut ? "\n…[" + (rest - kept) + " bytes left out"
                        + (log == null ? "" : "; the whole output is in the log") + "]…\n" : "") + end;
            }
        }
        if (text.length() > cap) {
            int keepHead = Math.min(headCap, text.length());
            text = text.substring(0, keepHead / 2) + "\n…[" + (text.length() - cap) + " characters left out]…\n"
                    + text.substring(text.length() - (cap - keepHead / 2));
            cut = true;
        }
        return new Result(done ? p.exitValue() : -1, text, cut, !done);
    }
}
