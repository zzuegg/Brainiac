package io.github.zzuegg.brainiac.studio.agent;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * FFmpeg, when the user has it: an optional outside program the studio calls to turn audio into
 * WAV and to put clips, sound and captions together. It is found on the {@code PATH}, or at
 * {@code BRAINIAC_FFMPEG}. Nothing else in the studio needs it.
 */
final class Ffmpeg {

    private static final long TIMEOUT_MINUTES = 30;

    private Ffmpeg() {
    }

    /**
     * Finds FFmpeg.
     *
     * @return its path, or null if it is not installed
     */
    static Path find() {
        String set = System.getenv("BRAINIAC_FFMPEG");
        if (set != null && !set.isBlank() && Files.isExecutable(Path.of(set))) {
            return Path.of(set);
        }
        String path = System.getenv("PATH");
        if (path == null) {
            return null;
        }
        boolean windows = System.getProperty("os.name", "").toLowerCase(java.util.Locale.ROOT).contains("win");
        for (String dir : path.split(java.io.File.pathSeparator)) {
            if (dir.isBlank()) {
                continue;
            }
            Path p = Path.of(dir.strip(), windows ? "ffmpeg.exe" : "ffmpeg");
            if (Files.isExecutable(p)) {
                return p;
            }
        }
        return null;
    }

    /**
     * Finds FFmpeg or says how to get it.
     *
     * @return its path
     */
    static Path need() {
        Path p = find();
        if (p == null) {
            throw new IllegalArgumentException("this needs FFmpeg, which is not installed: install it from ffmpeg.org "
                    + "and put it on the PATH, or set BRAINIAC_FFMPEG to where it is");
        }
        return p;
    }

    /**
     * Runs FFmpeg with arguments, never through a shell.
     *
     * @param args the arguments after the program
     * @param work the folder it runs in
     * @return the end of what it printed
     * @throws IOException if it fails
     */
    static String run(List<String> args, Path work) throws IOException {
        List<String> cmd = new ArrayList<>();
        cmd.add(need().toString());
        cmd.add("-hide_banner");
        cmd.add("-nostdin");
        cmd.add("-y");
        cmd.addAll(args);
        Process p = new ProcessBuilder(cmd).directory(work.toFile()).redirectErrorStream(true).start();
        byte[] out;
        try {
            out = p.getInputStream().readAllBytes();
            if (!p.waitFor(TIMEOUT_MINUTES, TimeUnit.MINUTES)) {
                p.destroyForcibly();
                throw new IOException("FFmpeg took longer than " + TIMEOUT_MINUTES + " minutes");
            }
        } catch (InterruptedException e) {
            p.destroyForcibly();
            Thread.currentThread().interrupt();
            throw new IOException("stopped");
        }
        String text = new String(out, StandardCharsets.UTF_8);
        String tail = text.length() > 2000 ? text.substring(text.length() - 2000) : text;
        if (p.exitValue() != 0) {
            throw new IOException("FFmpeg failed: " + tail.strip());
        }
        return tail;
    }

    /**
     * Turns any audio or video file into 16 kHz mono WAV.
     *
     * @param in the file
     * @return the WAV's bytes
     * @throws IOException if it fails
     */
    static byte[] wav(Path in) throws IOException {
        Path tmp = Files.createTempFile("brainiac-audio", ".wav");
        try {
            run(List.of("-i", in.toAbsolutePath().toString(), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav",
                    tmp.toString()), tmp.getParent());
            return Files.readAllBytes(tmp);
        } finally {
            Files.deleteIfExists(tmp);
        }
    }
}
