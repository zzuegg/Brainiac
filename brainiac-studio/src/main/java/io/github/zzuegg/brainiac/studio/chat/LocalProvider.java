package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.io.ChatTemplate;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.llm.Generator;
import io.github.zzuegg.brainiac.llm.Llama;
import io.github.zzuegg.brainiac.llm.Sampling;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

/**
 * Language models on this PC: Hugging Face model folders (Llama, Qwen2 and Mistral layouts) and
 * {@code .gguf} files found in the model folders. One model stays loaded at a time, and answers
 * are written one at a time.
 */
final class LocalProvider implements ChatProvider {

    private static final System.Logger LOG = System.getLogger(LocalProvider.class.getName());
    private static final Set<String> TYPES = Set.of("llama", "qwen2", "mistral");

    private final List<Path> folders;
    private Loaded loaded;

    private record Loaded(Path path, Generator generator, PretrainedTokenizer tokenizer, Set<Integer> stops) {
    }

    LocalProvider(List<Path> folders) {
        this.folders = List.copyOf(folders);
    }

    @Override
    public String id() {
        return "local";
    }

    @Override
    public String name() {
        return "This PC";
    }

    @Override
    public boolean ready() {
        return !find().isEmpty();
    }

    @Override
    public boolean local() {
        return true;
    }

    List<Path> folders() {
        return folders;
    }

    @Override
    public List<ModelInfo> models() {
        List<ModelInfo> out = new ArrayList<>();
        for (Map.Entry<String, Path> e : find().entrySet()) {
            Path p = e.getValue();
            boolean gguf = Files.isRegularFile(p);
            out.add(new ModelInfo(id(), e.getKey(), e.getKey(), List.of("text"), 0,
                    (gguf ? "GGUF" : "safetensors") + " · " + size(p)));
        }
        return out;
    }

    // name -> path; the first folder that has a name wins
    private Map<String, Path> find() {
        Map<String, Path> found = new LinkedHashMap<>();
        for (Path folder : folders) {
            if (!Files.isDirectory(folder)) {
                continue;
            }
            try (Stream<Path> s = Files.list(folder)) {
                s.sorted().forEach(p -> {
                    String name = p.getFileName().toString();
                    if (!found.containsKey(name) && (isGguf(p) || isModelFolder(p))) {
                        found.put(name, p);
                    }
                });
            } catch (IOException e) {
                LOG.log(System.Logger.Level.WARNING, "cannot list " + folder, e);
            }
        }
        return found;
    }

    private static boolean isGguf(Path p) {
        return Files.isRegularFile(p) && p.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".gguf");
    }

    private static boolean isModelFolder(Path p) {
        Path config = p.resolve("config.json");
        if (!Files.isRegularFile(config) || !Files.isRegularFile(p.resolve("tokenizer.json"))) {
            return false;
        }
        try {
            return TYPES.contains(Json.string(Json.object(Json.read(config)), "model_type", ""));
        } catch (IOException | IllegalArgumentException e) {
            return false;
        }
    }

    private static String size(Path p) {
        long bytes;
        try (Stream<Path> s = Files.isDirectory(p) ? Files.list(p) : Stream.of(p)) {
            bytes = s.filter(Files::isRegularFile).mapToLong(f -> {
                try {
                    return Files.size(f);
                } catch (IOException e) {
                    return 0;
                }
            }).sum();
        } catch (IOException e) {
            return "?";
        }
        return bytes >= 1L << 30 ? String.format(Locale.ROOT, "%.1f GB", bytes / (double) (1L << 30))
                : String.format(Locale.ROOT, "%d MB", bytes >> 20);
    }

    @Override
    public synchronized ChatResult chat(ChatRequest request, ChatListener listener) throws IOException {
        Path path = find().get(request.model());
        if (path == null) {
            throw new IOException("no local model named " + request.model() + " in " + folders);
        }
        if (loaded == null || !loaded.path().equals(path)) {
            loaded = null;   // let the old weights go before loading new ones
            Device device = Device.best();
            DType dtype = device.kind() == Device.Kind.GPU && device.supports(DType.FLOAT16) ? DType.FLOAT16
                    : DType.FLOAT32;
            Llama model = Llama.load(path, device, dtype);
            PretrainedTokenizer tok = PretrainedTokenizer.load(path);
            loaded = new Loaded(path, new Generator(model, tok), tok, Generator.stopTokens(path, tok));
        }
        List<Map<String, Object>> history = new ArrayList<>();
        if (!request.system().isBlank()) {
            history.add(ChatTemplate.message("system", request.system()));
        }
        for (ChatMessage m : request.messages()) {
            history.add(ChatTemplate.message(m.role(), m.text()));
        }
        PretrainedTokenizer tok = loaded.tokenizer();
        String prompt = tok.chatTemplate() != null ? tok.applyChatTemplate(history, true)
                : request.messages().getLast().text();
        Generator.Options options = new Generator.Options(request.maxTokens(),
                Sampling.temperature(0.7f).withTopP(0.9f), System.nanoTime(), loaded.stops(), List.of());
        Generator.Output out;
        try {
            out = loaded.generator().generate(List.of(prompt), options, (row, piece) -> {
                try {
                    listener.text(piece);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }).getFirst();
        } catch (UncheckedIOException e) {
            throw e.getCause();
        }
        return new ChatResult(out.finish() == Generator.Finish.LENGTH ? "length" : "end",
                loaded.generator().encode(prompt).length, out.tokens().length);
    }
}
