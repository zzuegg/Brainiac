package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.io.GgufWriter;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.llm.Llama;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Makes a smaller copy of one of this PC's chat models: a quantized GGUF file next to the
 * original, which the studio, llama.cpp, Ollama and LM Studio can all run. One copy is made at
 * a time, on the CPU, so a chat on the GPU keeps running; the page asks how it is going.
 */
public final class LocalQuantizer {

    /**
     * A size a copy can have.
     *
     * @param id the name it goes by, as llama.cpp's files carry it
     * @param type how the matrices are stored
     * @param bits about how many bits each weight takes
     * @param note what it is good for
     */
    public record Size(String id, GgufWriter.Type type, double bits, String note) {
    }

    /** The sizes offered, the usual choice first. */
    public static final List<Size> SIZES = List.of(
            new Size("Q4_K_M", GgufWriter.Type.Q4_K, 4.9, "the usual choice: small, with little lost"),
            new Size("Q6_K", GgufWriter.Type.Q6_K, 6.6, "larger, closer to the original"),
            new Size("Q8_0", GgufWriter.Type.Q8_0, 8.5, "nearly lossless, about twice the size of Q4_K_M"),
            new Size("Q4_0", GgufWriter.Type.Q4_0, 4.6, "the older 4-bit format, for programs that need it"),
            new Size("F16", GgufWriter.Type.F16, 16, "no quantizing: the model as a GGUF file with 16-bit floats"));

    private record Job(String model, String size, Path file, long started, String state, String error, long ended) {

        Job to(String next, String why) {
            boolean over = next.equals("done") || next.equals("failed");
            return new Job(model, size, file, started, next, why, over ? System.currentTimeMillis() : 0);
        }
    }

    private final LocalProvider local;
    private volatile Job job;

    /**
     * A quantizer for a provider's models.
     *
     * @param local this PC's models
     */
    public LocalQuantizer(LocalProvider local) {
        this.local = local;
    }

    /**
     * Describes the sizes and the copy being made or last made, for the page.
     *
     * @return {@code {sizes: [{id, bits, note}], job?: {model, size, file, state, error?, seconds, bytes?}}};
     *     a job's state is {@code loading}, {@code writing}, {@code done} or {@code failed}
     */
    public Map<String, Object> state() {
        Map<String, Object> out = new LinkedHashMap<>();
        List<Object> sizes = new ArrayList<>();
        for (Size s : SIZES) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", s.id());
            m.put("bits", s.bits());
            m.put("note", s.note());
            sizes.add(m);
        }
        out.put("sizes", sizes);
        Job j = job;
        if (j != null) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("model", j.model());
            m.put("size", j.size());
            m.put("file", j.file().toString());
            m.put("name", j.file().getFileName().toString());
            m.put("state", j.state());
            if (j.error() != null) {
                m.put("error", j.error());
            }
            m.put("seconds", ((j.ended() > 0 ? j.ended() : System.currentTimeMillis()) - j.started()) / 1000.0);
            if (j.state().equals("done")) {
                try {
                    m.put("bytes", Files.size(j.file()));
                } catch (IOException e) {
                    // moved or deleted since: the page shows it without a size
                }
            }
            out.put("job", m);
        }
        return out;
    }

    /**
     * Returns whether a copy is being made.
     *
     * @return whether one is
     */
    public boolean running() {
        Job j = job;
        return j != null && (j.state().equals("loading") || j.state().equals("writing"));
    }

    /**
     * Starts making a copy of a chat model. It is written next to the original as
     * {@code <model>-<size>.gguf} and shows up in the model list when it is done.
     *
     * @param name the model as the provider lists it
     * @param size one of {@link #SIZES}, by its id
     * @return the state, as {@link #state} gives it
     * @throws IOException if there is no such model, it looks at pictures, the copy is there
     *     already, or another copy is being made
     */
    public synchronized Map<String, Object> start(String name, String size) throws IOException {
        Size s = SIZES.stream().filter(x -> x.id().equals(size)).findFirst()
                .orElseThrow(() -> new IllegalArgumentException("choose one of " + SIZES.stream().map(Size::id).toList()));
        if (running()) {
            throw new IOException("a copy of " + job.model() + " is being made: wait for it to finish");
        }
        Path source = local.location(name);
        if (LocalProvider.looksAtPictures(source)) {
            throw new IOException(name + " looks at pictures: only its text part would be copied, so the studio leaves it");
        }
        String stem = source.getFileName().toString().replaceFirst("(?i)\\.gguf$", "");
        Path target = source.toAbsolutePath().getParent().resolve(stem + "-" + s.id() + ".gguf");
        if (Files.exists(target)) {
            throw new IOException(target.getFileName() + " is there already");
        }
        Job started = new Job(name, s.id(), target, System.currentTimeMillis(), "loading", null, 0);
        job = started;
        Thread.ofVirtual().name("quantize-" + name).start(() -> run(started, source, s));
        return state();
    }

    private void run(Job mine, Path source, Size size) {
        Path part = mine.file().resolveSibling(mine.file().getFileName() + ".part");
        try {
            Llama model = Llama.load(source, Device.cpu(), DType.FLOAT32);
            PretrainedTokenizer tokenizer;
            try {
                tokenizer = PretrainedTokenizer.load(source);
            } catch (RuntimeException e) {
                throw new IOException(mine.model() + " has no tokenizer the studio can read, and a GGUF file needs one: "
                        + e.getMessage(), e);
            }
            job = mine.to("writing", null);
            if (size.type() == GgufWriter.Type.F16) {
                model.saveGguf(part, tokenizer, true);
            } else {
                model.saveGguf(part, tokenizer, size.type());
            }
            Files.move(part, mine.file(), StandardCopyOption.ATOMIC_MOVE);
            job = mine.to("done", null);
        } catch (IOException | RuntimeException | OutOfMemoryError e) {
            try {
                Files.deleteIfExists(part);
            } catch (IOException ignored) {
                // the half-written file stays; its name says what it is
            }
            String why = e instanceof OutOfMemoryError
                    ? "not enough memory: the whole model is held in memory at full size while its copy is written"
                    : String.valueOf(e.getMessage());
            job = mine.to("failed", why);
        }
    }
}
