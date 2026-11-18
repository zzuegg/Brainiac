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
 * {@code .gguf} files found in the model folders, plus the folders and files added on the Models
 * screen, which are kept one per line in a file in the data folder. One model stays loaded at a
 * time, and answers are written one at a time.
 */
final class LocalProvider implements ChatProvider {

    private static final System.Logger LOG = System.getLogger(LocalProvider.class.getName());
    private static final Set<String> TYPES = io.github.zzuegg.brainiac.llm.LlamaConfig.MODEL_TYPES;

    private final List<Path> folders;
    private final Path placesFile;
    private final List<Path> added = new ArrayList<>();
    private Loaded loaded;
    private volatile LocalImages images;
    private volatile LocalSpeech speech;

    private record Loaded(Path path, Generator generator, PretrainedTokenizer tokenizer, Set<Integer> stops) {
    }

    LocalProvider(List<Path> folders) {
        this(folders, null);
    }

    LocalProvider(List<Path> folders, Path placesFile) {
        this.folders = List.copyOf(folders);
        this.placesFile = placesFile;
        if (placesFile != null && Files.isRegularFile(placesFile)) {
            try {
                for (String line : Files.readAllLines(placesFile)) {
                    if (!line.isBlank()) {
                        added.add(Path.of(line.strip()));
                    }
                }
            } catch (IOException | RuntimeException e) {
                LOG.log(System.Logger.Level.WARNING, "cannot read " + placesFile, e);
            }
        }
    }

    /**
     * Returns the folders and files added on the Models screen.
     *
     * @return them, in the order added
     */
    synchronized List<Path> added() {
        return List.copyOf(added);
    }

    /**
     * Adds a place to find models: a {@code .gguf} file, a Hugging Face model folder, or a folder
     * holding them.
     *
     * @param place the file or folder
     * @return how many models are there now
     * @throws IOException if it is none of those, or the list cannot be saved
     */
    synchronized int add(Path place) throws IOException {
        Path p = place.toAbsolutePath().normalize();
        if (!Files.exists(p)) {
            throw new IOException("nothing at " + p);
        }
        if (Files.isRegularFile(p) && !isGguf(p)) {
            throw new IOException(p.getFileName() + " is not a .gguf file. Pick a .gguf file, a Hugging Face model"
                    + " folder (with config.json and tokenizer.json) or a folder holding them.");
        }
        if (!added.contains(p) && !folders.contains(p)) {
            added.add(p);
            save();
        }
        return models(p).size();
    }

    /**
     * Forgets a place added on the Models screen. The files stay where they are.
     *
     * @param place the file or folder
     * @return whether it was there
     * @throws IOException if the list cannot be saved
     */
    synchronized boolean remove(Path place) throws IOException {
        boolean had = added.remove(place.toAbsolutePath().normalize()) || added.remove(place);
        if (had) {
            save();
        }
        return had;
    }

    private void save() throws IOException {
        if (placesFile == null) {
            return;
        }
        Files.createDirectories(placesFile.toAbsolutePath().getParent());
        Files.write(placesFile, added.stream().map(Path::toString).toList());
    }

    /**
     * Says what a path is to the Models screen: a model, a folder that may hold models, or neither.
     *
     * @param p the path
     * @return "gguf", "model", "folder" or "other"
     */
    static String kind(Path p) {
        return isGguf(p) ? "gguf" : isModelFolder(p) ? "model" : Files.isDirectory(p) ? "folder" : "other";
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
        return !find().isEmpty() || (images != null && !images.models().isEmpty())
                || (speech != null && !speech.models().isEmpty());
    }

    @Override
    public boolean local() {
        return true;
    }

    List<Path> folders() {
        return folders;
    }

    /**
     * Also lists and draws with the Stable Diffusion models in the same places.
     *
     * @param hub where a single-file checkpoint's CLIP vocabulary is fetched from
     */
    void drawToo(java.util.function.Supplier<io.github.zzuegg.brainiac.io.HuggingFaceHub> hub) {
        images = new LocalImages(() -> {
            List<Path> places = new ArrayList<>(folders);
            places.addAll(added());
            return places;
        }, hub);
    }

    /**
     * Returns the image models' side, or null when this provider only chats.
     *
     * @return the image models
     */
    LocalImages images() {
        return images;
    }

    /** Also lists, and speaks and transcribes with, the speech models in the same places. */
    void speakToo() {
        speech = new LocalSpeech(() -> {
            List<Path> places = new ArrayList<>(folders);
            places.addAll(added());
            return places;
        });
    }

    /**
     * Returns the speech models' side, or null when this provider only chats.
     *
     * @return the speech models
     */
    LocalSpeech speech() {
        return speech;
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
        if (images != null) {
            out.addAll(images.models());
        }
        if (speech != null) {
            out.addAll(speech.models());
        }
        return out;
    }

    // name -> path; the first place that has a name wins
    private Map<String, Path> find() {
        List<Path> places = new ArrayList<>(folders);
        places.addAll(added());
        Map<String, Path> found = new LinkedHashMap<>();
        for (Path place : places) {
            for (Path p : models(place)) {
                found.putIfAbsent(p.getFileName().toString(), p);
            }
        }
        return found;
    }

    // the models a place holds: itself if it is one, else the ones directly inside it
    private static List<Path> models(Path place) {
        if (isGguf(place) || isModelFolder(place)) {
            return List.of(place);
        }
        if (!Files.isDirectory(place)) {
            return List.of();
        }
        try (Stream<Path> s = Files.list(place)) {
            return s.sorted().filter(p -> isGguf(p) || isModelFolder(p)).toList();
        } catch (IOException e) {
            LOG.log(System.Logger.Level.WARNING, "cannot list " + place, e);
            return List.of();
        }
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
            if (device.kind() == Device.Kind.GPU) {
                // two products per block instead of five: the same answers, much less GPU overhead
                try {
                    model.fuseProjections();
                } catch (RuntimeException e) {
                    LOG.log(System.Logger.Level.INFO, "running " + path.getFileName() + " unfused: " + e.getMessage());
                }
            }
            PretrainedTokenizer tok = PretrainedTokenizer.load(path);
            // a chat sends its whole history each turn; only the new part is read again
            Generator generator = new Generator(model, tok).cachePrefixes(4);
            loaded = new Loaded(path, generator, tok, Generator.stopTokens(path, tok));
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
                sampling(request.temperature()), System.nanoTime(), loaded.stops(), List.of());
        // a template that opens the think block itself (DeepSeek-R1, Qwen3 with thinking on) starts the answer inside it
        ThinkSplitter split = new ThinkSplitter(listener, prompt.stripTrailing().endsWith("<think>"));
        Generator.Output out;
        try {
            out = loaded.generator().generate(List.of(prompt), options, (row, piece) -> {
                try {
                    split.text(piece);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }).getFirst();
        } catch (UncheckedIOException e) {
            throw e.getCause();
        }
        split.finish();
        return new ChatResult(out.finish() == Generator.Finish.LENGTH ? "length" : "end",
                loaded.generator().encode(prompt).length, out.tokens().length);
    }

    /** The usual sampling, or the temperature asked for; 0 always takes the likeliest token. */
    private static Sampling sampling(Float temperature) {
        if (temperature == null) {
            return Sampling.temperature(0.7f).withTopP(0.9f);
        }
        return temperature <= 0 ? Sampling.GREEDY : Sampling.temperature(temperature).withTopP(0.9f);
    }
}
