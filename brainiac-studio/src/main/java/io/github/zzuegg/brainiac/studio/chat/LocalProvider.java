package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.io.ChatTemplate;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.llm.Generator;
import io.github.zzuegg.brainiac.llm.Llama;
import io.github.zzuegg.brainiac.llm.LanguageModel;
import io.github.zzuegg.brainiac.llm.Sampling;
import io.github.zzuegg.brainiac.llm.VisionLanguageModel;
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
    /** The model types that look at pictures: LLaVA 1.5, Gemma 3 and Qwen2-VL / Qwen2.5-VL. */
    static final Set<String> SEEING = Set.of("llava", "gemma3", "qwen2_vl", "qwen2_5_vl");

    private final List<Path> folders;
    private final Path placesFile;
    private final List<Path> added = new ArrayList<>();
    private final LocalDevice device;
    private Loaded loaded;
    private volatile LocalImages images;
    private volatile LocalSpeech speech;
    private volatile LocalVideos videos;

    record Loaded(Path path, Device device, Generator generator, PretrainedTokenizer tokenizer,
            Set<Integer> stops, VisionLanguageModel vision) {
    }

    LocalProvider(List<Path> folders) {
        this(folders, null);
    }

    LocalProvider(List<Path> folders, Path placesFile) {
        this.folders = List.copyOf(folders);
        this.placesFile = placesFile;
        this.device = new LocalDevice(placesFile == null ? null : placesFile.resolveSibling("local-device.txt"));
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
                || (speech != null && !speech.models().isEmpty()) || (videos != null && !videos.models().isEmpty());
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
        }, hub, device);
    }

    /**
     * Also lists and makes clips with the Wan video models in the same places. A video model
     * and an image model do not stay loaded together: loading one lets the other go.
     */
    void filmToo() {
        videos = new LocalVideos(() -> {
            List<Path> places = new ArrayList<>(folders);
            places.addAll(added());
            return places;
        }, device, () -> {
            if (images != null) {
                images.close();
            }
        });
    }

    /**
     * Returns the video models' side, or null when this provider makes no videos.
     *
     * @return the video models
     */
    LocalVideos videos() {
        return videos;
    }

    @Override
    public boolean makesVideo() {
        return videos != null;
    }

    @Override
    public Map<String, Object> videoOptions(String model) {
        return videos == null ? null : videos.describe(model).orElse(null);
    }

    @Override
    public LocalVideos.Clip video(String model, String prompt, LocalVideos.Settings settings,
            LocalVideos.Listener listener) throws IOException {
        if (videos == null) {
            throw new IOException("this studio makes no videos on this PC");
        }
        return videos.make(model, prompt, settings, listener);
    }

    /**
     * Returns where this PC's models run.
     *
     * @return the choice
     */
    LocalDevice device() {
        return device;
    }

    /**
     * Returns the image models' side, or null when this provider only chats.
     *
     * @return the image models
     */
    LocalImages images() {
        return images;
    }

    /**
     * Also lists, and speaks and transcribes with, the speech models in the same places.
     *
     * @param voices where voices made from recordings are kept
     */
    void speakToo(Path voices) {
        speech = new LocalSpeech(() -> {
            List<Path> places = new ArrayList<>(folders);
            places.addAll(added());
            return places;
        }, voices, device);
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
    public boolean takesGrammar() {
        return true;
    }

    @Override
    public boolean runsMedia() {
        return true;
    }

    @Override
    public List<LocalImages.Picture> draw(String model, String prompt, LocalImages.Settings settings,
            LocalImages.Listener listener) throws IOException {
        if (images == null) {
            throw new IOException("this studio does not draw on this PC");
        }
        return images.draw(model, prompt, settings, listener);
    }

    @Override
    public Media speak(String model, String text, String voice, double speed, java.util.function.BooleanSupplier stop)
            throws IOException {
        if (speech == null) {
            throw new IOException("this studio does not speak on this PC");
        }
        return speech.speak(model, text, voice, speed, stop);
    }

    @Override
    public io.github.zzuegg.brainiac.audio.Transcription transcribe(String model, byte[] wav,
            LocalSpeech.Listening listening,
            java.util.function.Consumer<io.github.zzuegg.brainiac.audio.Transcription.Segment> onSegment,
            java.util.function.BooleanSupplier stop) throws IOException {
        if (speech == null) {
            throw new IOException("this studio does not transcribe on this PC");
        }
        return speech.transcribe(model, wav, listening, onSegment, stop);
    }

    @Override
    public Map<String, Object> voiceOptions(String model) {
        return speech == null ? null : speech.options(model).toJson();
    }

    @Override
    public List<ModelInfo> models() {
        List<ModelInfo> out = new ArrayList<>();
        for (Map.Entry<String, Path> e : find().entrySet()) {
            Path p = e.getValue();
            boolean gguf = Files.isRegularFile(p);
            out.add(new ModelInfo(id(), e.getKey(), e.getKey(), List.of("text"), 0,
                    (gguf ? "GGUF" : "safetensors") + " · " + size(p) + (sees(p) ? " · looks at pictures" : "")));
        }
        if (images != null) {
            out.addAll(images.models());
        }
        if (speech != null) {
            out.addAll(speech.models());
        }
        if (videos != null) {
            out.addAll(videos.models());
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
        if (p.getFileName() == null) {
            return false;   // a drive's root
        }
        String name = p.getFileName().toString().toLowerCase(Locale.ROOT);
        // a projector file belongs to the model next to it and is not one to chat with
        return Files.isRegularFile(p) && name.endsWith(".gguf") && !name.startsWith("mmproj");
    }

    /**
     * Finds llama.cpp's projector file for a GGUF model: an {@code mmproj*.gguf} in the same
     * folder. With several there, the one that carries the model's name is taken, and none if
     * that does not settle it.
     */
    static Path mmproj(Path model) {
        if (!isGguf(model) || model.getParent() == null) {
            return null;
        }
        List<Path> found;
        try (Stream<Path> s = Files.list(model.getParent())) {
            found = s.filter(p -> {
                String n = p.getFileName().toString().toLowerCase(Locale.ROOT);
                return Files.isRegularFile(p) && n.startsWith("mmproj") && n.endsWith(".gguf");
            }).sorted().toList();
        } catch (IOException e) {
            return null;
        }
        // a projector the studio downloaded is named for its model: "mmproj-" and the model's file name
        Path own = model.resolveSibling("mmproj-" + model.getFileName());
        if (Files.isRegularFile(own)) {
            return own;
        }
        found = found.stream().filter(p -> {
            Path other = p.resolveSibling(p.getFileName().toString().replaceFirst("(?i)^mmproj-", ""));
            return other.equals(p) || !Files.isRegularFile(other);
        }).toList();
        if (found.size() <= 1) {
            return found.isEmpty() ? null : found.getFirst();
        }
        // "gemma-3-4b-it-Q4_K_M.gguf" goes with "mmproj-gemma-3-4b-it-f16.gguf": the longest shared start wins
        String name = model.getFileName().toString().toLowerCase(Locale.ROOT);
        Path best = null;
        int bestLength = 3, ties = 0;
        for (Path p : found) {
            String n = p.getFileName().toString().toLowerCase(Locale.ROOT).replaceFirst("^mmproj[-_.]?", "");
            int k = 0;
            while (k < n.length() && k < name.length() && n.charAt(k) == name.charAt(k)) {
                k++;
            }
            if (k > bestLength) {
                best = p;
                bestLength = k;
                ties = 0;
            } else if (k == bestLength && best != null) {
                ties++;
            }
        }
        return ties == 0 ? best : null;
    }

    // whether a model looks at pictures: a vision model's folder, or a GGUF with its projector
    private static boolean sees(Path p) {
        if (Files.isRegularFile(p)) {
            return mmproj(p) != null;
        }
        try {
            return SEEING.contains(Json.string(Json.object(Json.read(p.resolve("config.json"))), "model_type", ""));
        } catch (IOException | IllegalArgumentException e) {
            return false;
        }
    }

    private static boolean isModelFolder(Path p) {
        Path config = p.resolve("config.json");
        if (!Files.isRegularFile(config) || !Files.isRegularFile(p.resolve("tokenizer.json"))) {
            return false;
        }
        try {
            String type = Json.string(Json.object(Json.read(config)), "model_type", "");
            if (SEEING.contains(type)) {
                return Files.isRegularFile(p.resolve("preprocessor_config.json"));
            }
            // Orpheus is a Llama that speaks: it is a speech model, not one to chat with
            return TYPES.contains(type) && !("llama".equals(type)
                    && io.github.zzuegg.brainiac.audio.OrpheusSpeech.isOrpheus(p));
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
        Device device = this.device.device();
        if (loaded == null || !loaded.path().equals(path) || !loaded.device().name().equals(device.name())) {
            loaded = null;   // let the old weights go before loading new ones
            loaded = load(path, device);
        }
        return answer(request, listener);
    }

    /**
     * Loads a chat model for a use of its own, such as serving it to other programs: apart from
     * the one this provider keeps for the studio's chat.
     *
     * @param name the model as {@link #models} lists it
     * @return the loaded model
     * @throws IOException if there is no such chat model
     */
    Loaded open(String name) throws IOException {
        Path path = find().get(name);
        if (path == null) {
            throw new IOException("no local chat model named " + name);
        }
        return load(path, device.device());
    }

    /**
     * Returns where one of the chat models is.
     *
     * @param name the model as {@link #models} lists it
     * @return its folder or GGUF file
     * @throws IOException if there is no such chat model
     */
    Path location(String name) throws IOException {
        Path path = find().get(name);
        if (path == null) {
            throw new IOException("no local chat model named " + name);
        }
        return path;
    }

    /**
     * Returns whether a model looks at pictures.
     *
     * @param path its folder or GGUF file
     * @return whether it does
     */
    static boolean looksAtPictures(Path path) {
        return sees(path);
    }

    private static Loaded load(Path path, Device device) {
        DType dtype = device.kind() == Device.Kind.GPU && device.supports(DType.FLOAT16) ? DType.FLOAT16
                : DType.FLOAT32;
        LanguageModel model;
        VisionLanguageModel vision = null;
        if (sees(path)) {
            Path projector = mmproj(path);
            if (projector == null) {
                vision = VisionLanguageModel.load(path, device, dtype);
            } else {
                try {
                    vision = VisionLanguageModel.load(path, projector, device, dtype);
                } catch (RuntimeException e) {
                    // a projector the core cannot read must not take the model's text chat with it
                    LOG.log(System.Logger.Level.WARNING, path.getFileName() + " runs without pictures: "
                            + projector.getFileName() + " cannot be read (" + e.getMessage() + ")");
                }
            }
        }
        if (vision != null) {
            model = vision;
        } else {
            Llama llama = Llama.load(path, device, dtype);
            if (device.kind() == Device.Kind.GPU) {
                // two products per block instead of five: the same answers, much less GPU overhead
                try {
                    llama.fuseProjections();
                } catch (RuntimeException e) {
                    LOG.log(System.Logger.Level.INFO, "running " + path.getFileName() + " unfused: " + e.getMessage());
                }
            }
            model = llama;
        }
        PretrainedTokenizer tok = PretrainedTokenizer.load(path);
        Generator generator = new Generator(model, tok);
        if (vision == null) {
            // a chat sends its whole history each turn; only the new part is read again. Not for
            // a model that sees: two pictures make the same tokens, so a kept start could be another picture's
            generator.cachePrefixes(4);
        }
        return new Loaded(path, device, generator, tok, Generator.stopTokens(path, tok), vision);
    }

    private ChatResult answer(ChatRequest request, ChatListener listener) throws IOException {
        List<java.awt.image.BufferedImage> pictures = new ArrayList<>();
        for (ChatMessage m : request.messages()) {
            for (Media image : m.images()) {
                java.awt.image.BufferedImage read = javax.imageio.ImageIO.read(new java.io.ByteArrayInputStream(image.data()));
                if (read == null) {
                    throw new IOException("a picture cannot be read here (" + image.mime() + "): send it as PNG or JPEG");
                }
                pictures.add(read);
            }
        }
        VisionLanguageModel vision = loaded.vision();
        if (!pictures.isEmpty() && vision == null) {
            throw new IOException(request.model() + " cannot look at pictures. Pick a model that can: LLaVA 1.5, Gemma 3 or "
                    + "Qwen-VL on this PC (a folder, or a GGUF with its mmproj file beside it), or Claude, GPT or Gemini.");
        }
        List<Map<String, Object>> history = new ArrayList<>();
        if (!request.system().isBlank()) {
            history.add(ChatTemplate.message("system", request.system()));
        }
        for (ChatMessage m : request.messages()) {
            if (m.images().isEmpty()) {
                history.add(ChatTemplate.message(m.role(), m.text()));
                continue;
            }
            // the template marks each picture where the model's processor expects it
            List<Object> parts = new ArrayList<>();
            for (int k = 0; k < m.images().size(); k++) {
                parts.add(Map.of("type", "image"));
            }
            parts.add(Map.of("type", "text", "text", m.text()));
            Map<String, Object> message = new LinkedHashMap<>();
            message.put("role", m.role());
            message.put("content", parts);
            history.add(message);
        }
        PretrainedTokenizer tok = loaded.tokenizer();
        String prompt;
        if (tok.chatTemplate() != null) {
            prompt = tok.applyChatTemplate(history, true);
        } else if (pictures.isEmpty()) {
            prompt = request.messages().getLast().text();
        } else if (vision.kind() == VisionLanguageModel.Kind.LLAVA) {
            // LLaVA 1.5 without a template: its own prompt form, for the pictures and the last message
            prompt = "USER: " + "<image>\n".repeat(pictures.size()) + request.messages().getLast().text() + " ASSISTANT:";
        } else {
            throw new IOException(request.model() + " has no chat template, so the studio cannot place the pictures in its prompt");
        }
        Generator.Options options = new Generator.Options(request.maxTokens(),
                sampling(request.temperature(), request.sampler()), System.nanoTime(), loaded.stops(), List.of());
        boolean formed = !request.format().none();
        if (formed) {
            try {
                // only words that keep the answer in the format can be chosen
                options = options.withConstraint(request.format().constraint());
            } catch (IllegalArgumentException e) {
                throw new IOException(e.getMessage(), e);
            }
        }
        // a template that opens the think block itself (DeepSeek-R1, Qwen3 with thinking on) starts the answer inside it;
        // an answer held to a format has no think block
        ThinkSplitter split = new ThinkSplitter(listener, !formed && prompt.stripTrailing().endsWith("<think>"));
        Generator.Output out;
        Generator.Listener pieces = (row, piece) -> {
            try {
                split.text(piece);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        };
        int promptTokens;
        try {
            if (pictures.isEmpty()) {
                out = loaded.generator().generate(List.of(prompt), options, pieces).getFirst();
                promptTokens = loaded.generator().encode(prompt).length;
            } else {
                // each picture's marker becomes its tokens, and its features wait for the pass that reads them
                int[] ids;
                try {
                    ids = vision.prompt(loaded.generator().encode(vision.markImages(prompt)), pictures);
                } catch (IllegalArgumentException e) {
                    throw new IOException("the pictures do not fit this model's prompt: " + e.getMessage(), e);
                }
                out = loaded.generator().generate(new int[][] {ids}, options, pieces).getFirst();
                promptTokens = ids.length;
            }
        } catch (UncheckedIOException e) {
            throw e.getCause();
        }
        split.finish();
        return new ChatResult(out.finish() == Generator.Finish.LENGTH ? "length" : "end",
                promptTokens, out.tokens().length);
    }

    /** The usual sampling, or the temperature asked for; 0 always takes the likeliest token. */
    static Sampling sampling(Float temperature, Sampler sampler) {
        // at temperature 0 there is nothing to pick among, but a repeat penalty still moves the likeliest word
        Sampling s = temperature != null && temperature <= 0 ? Sampling.GREEDY
                : Sampling.temperature(temperature == null ? 0.7f : temperature)
                        .withTopP(sampler.topP() == null ? 0.9f : sampler.topP());
        if (sampler.topK() != null && s != Sampling.GREEDY) {
            s = s.withTopK(sampler.topK());
        }
        if (sampler.repeatPenalty() != null) {
            s = s.withRepetitionPenalty(sampler.repeatPenalty());
        }
        return s;
    }
}
