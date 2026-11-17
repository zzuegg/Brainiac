package io.github.zzuegg.brainiac.examples;

import io.github.zzuegg.brainiac.io.ChatTemplate;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.llm.Generator;
import io.github.zzuegg.brainiac.llm.Llama;
import io.github.zzuegg.brainiac.llm.Sampling;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Chats with a pretrained model downloaded from Hugging Face: a model directory (such as
 * {@code SmolLM2-135M-Instruct} or {@code Qwen2.5-0.5B-Instruct}) or a {@code .gguf} file.
 *
 * <pre>
 * ./gradlew :brainiac-examples:run --args="chat models/Qwen2.5-0.5B-Instruct --prompt 'Name three rivers.'"
 * ./gradlew :brainiac-examples:run --args="chat models/qwen2.5-0.5b-instruct-q8_0.gguf --device vulkan"
 * </pre>
 *
 * Without {@code --prompt} it reads questions from the console. Options: {@code --device
 * cpu|vulkan|opencl|<part of a name>}, {@code --dtype f32|f16|bf16}, {@code --max N},
 * {@code --temperature T} (0 for greedy), {@code --top-p P}.
 */
public final class ChatExample {

    private ChatExample() {
    }

    /**
     * Runs the example.
     *
     * @param args the model path and options
     */
    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("Usage: chat <model directory or .gguf> [--prompt text] [--device name] [--dtype f32|f16]"
                    + " [--max N] [--temperature T] [--top-p P]");
            return;
        }
        Path path = Path.of(args[0]);
        String prompt = null;
        String deviceName = "best";
        DType dtype = null;
        int max = 256;
        float temperature = 0.7f;
        float topP = 0.9f;
        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--prompt" -> prompt = args[++i];
                case "--device" -> deviceName = args[++i];
                case "--dtype" -> dtype = switch (args[++i]) {
                    case "f16" -> DType.FLOAT16;
                    case "bf16" -> DType.BFLOAT16;
                    default -> DType.FLOAT32;
                };
                case "--max" -> max = Integer.parseInt(args[++i]);
                case "--temperature" -> temperature = Float.parseFloat(args[++i]);
                case "--top-p" -> topP = Float.parseFloat(args[++i]);
                default -> throw new IllegalArgumentException("unknown option " + args[i]);
            }
        }
        Device device = device(deviceName);
        if (dtype == null) {
            dtype = device.kind() == Device.Kind.GPU && device.supports(DType.FLOAT16) ? DType.FLOAT16 : DType.FLOAT32;
        }
        long start = System.nanoTime();
        Llama model = Llama.load(path, device, dtype);
        PretrainedTokenizer tok = PretrainedTokenizer.load(path);
        System.out.printf("%s on %s as %s, loaded in %.1f s%n", model, device, dtype, (System.nanoTime() - start) / 1e9);
        Set<Integer> stops = Generator.stopTokens(path, tok);
        Sampling sampling = temperature == 0 ? Sampling.GREEDY : Sampling.temperature(temperature).withTopP(topP);
        Generator gen = new Generator(model, tok);
        List<Map<String, Object>> history = new ArrayList<>();
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        long seed = 1;
        while (true) {
            String question = prompt;
            if (question == null) {
                System.out.print("\n> ");
                try {
                    question = in.readLine();
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
                if (question == null || question.isBlank()) {
                    return;
                }
            }
            history.add(ChatTemplate.message("user", question));
            String text = tok.chatTemplate() != null ? tok.applyChatTemplate(history, true) : question;
            int promptTokens = tok.encode(text).length;
            Generator.Options options = new Generator.Options(max, sampling, seed++, stops, List.of());
            long t0 = System.nanoTime();
            int[] count = {0};
            String answer = gen.generate(text, options, (row, piece) -> {
                System.out.print(piece);
                System.out.flush();
                count[0]++;
            });
            double seconds = (System.nanoTime() - t0) / 1e9;
            int tokens = tok.encode(answer, false).length;
            System.out.printf(Locale.ROOT, "%n[%d prompt tokens, %d new tokens in %.1f s: %.1f tokens/s]%n",
                    promptTokens, tokens, seconds, tokens / seconds);
            history.add(ChatTemplate.message("assistant", answer));
            if (prompt != null) {
                return;
            }
        }
    }

    private static Device device(String name) {
        if (name.equals("best")) {
            return Device.best();
        }
        if (name.equals("cpu")) {
            return Device.cpu();
        }
        String want = name.toLowerCase(Locale.ROOT);
        return Device.available().stream().filter(d -> d.name().toLowerCase(Locale.ROOT).contains(want)).findFirst()
                .orElseThrow(() -> new IllegalArgumentException("no device matches '" + name + "': "
                        + Device.available()));
    }
}
