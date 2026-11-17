package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A tool-using assistant: the model sees the tools in its chat template, asks for calls in
 * the Hermes format that Qwen and many other models use ({@code <tool_call>{"name": ...,
 * "arguments": {...}}</tool_call>}) or as a bare JSON object, and each call's result goes back
 * as a {@code tool} message until the model answers without calling.
 *
 * <pre>{@code
 * Agent.Tool weather = new Agent.Tool("weather", "Current temperature of a city",
 *         Map.of("type", "object", "properties", Map.of("city", Map.of("type", "string")), "required", List.of("city")),
 *         args -> "21 C in " + args.get("city"));
 * Agent agent = new Agent(generator, tokenizer, List.of(weather), Generator.Options.greedy(256).withStopTokens(stops));
 * Agent.Result r = agent.run(List.of(ChatTemplate.message("user", "How warm is it in Vienna?")), 4);
 * }</pre>
 *
 * <p>{@link Tools} makes tools from annotated methods and records, with the schema derived
 * from their types.
 */
public final class Agent {

    /**
     * A function the model may call.
     *
     * @param name its name
     * @param description what it does, for the model
     * @param parameters a JSON schema of its arguments
     * @param run computes the result from the arguments
     */
    public record Tool(String name, String description, Map<String, Object> parameters,
            Function<Map<String, Object>, String> run) {

        /** Checks the parts. */
        public Tool {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(run, "run");
            description = description == null ? "" : description;
            parameters = parameters == null ? Map.of("type", "object", "properties", Map.of()) : parameters;
        }
    }

    /**
     * A finished conversation.
     *
     * @param messages the whole conversation, the calls and their results included
     * @param answer the model's last answer
     * @param toolCalls how many calls were made
     * @param finished whether the model answered without a call within the step limit
     */
    public record Result(List<Map<String, Object>> messages, String answer, int toolCalls, boolean finished) {
    }

    private static final Pattern HERMES = Pattern.compile("<tool_call>\\s*(\\{.*?\\})\\s*</tool_call>", Pattern.DOTALL);

    private final Generator generator;
    private final PretrainedTokenizer tokenizer;
    private final Map<String, Tool> tools = new LinkedHashMap<>();
    private final Generator.Options options;

    /**
     * @param generator writes the model's turns
     * @param tokenizer renders the conversation with its chat template, which should handle tools
     * @param tools the tools the model may call
     * @param options how each turn is generated; stop at the end-of-turn token
     */
    public Agent(Generator generator, PretrainedTokenizer tokenizer, List<Tool> tools, Generator.Options options) {
        this.generator = Objects.requireNonNull(generator, "generator");
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.options = Objects.requireNonNull(options, "options");
        for (Tool t : tools) {
            if (this.tools.put(t.name(), t) != null) {
                throw new IllegalArgumentException("two tools named " + t.name());
            }
        }
    }

    /**
     * Converses until the model answers without calling a tool.
     *
     * @param messages the conversation so far, ending with the user's message
     * @param maxSteps the most model turns
     * @return the conversation and the answer
     */
    public Result run(List<? extends Map<String, ?>> messages, int maxSteps) {
        List<Map<String, Object>> conversation = new ArrayList<>();
        for (Map<String, ?> m : messages) {
            conversation.add(new LinkedHashMap<>(m));
        }
        List<Map<String, Object>> specs = new ArrayList<>();
        for (Tool t : tools.values()) {
            Map<String, Object> f = new LinkedHashMap<>();
            f.put("name", t.name());
            f.put("description", t.description());
            f.put("parameters", t.parameters());
            specs.add(Map.of("type", "function", "function", f));
        }
        Map<String, Object> variables = specs.isEmpty() ? Map.of() : Map.of("tools", specs);
        int calls = 0;
        String text = "";
        for (int step = 0; step < maxSteps; step++) {
            String prompt = tokenizer.applyChatTemplate(conversation, true, variables);
            text = generator.generate(prompt, options).strip();
            List<Map<String, Object>> requested = calls(text);
            if (requested.isEmpty()) {
                conversation.add(message("assistant", text));
                return new Result(conversation, text, calls, true);
            }
            Map<String, Object> assistant = message("assistant", content(text));
            List<Object> toolCalls = new ArrayList<>();
            for (Map<String, Object> c : requested) {
                toolCalls.add(Map.of("type", "function", "function", c));
            }
            assistant.put("tool_calls", toolCalls);
            conversation.add(assistant);
            for (Map<String, Object> c : requested) {
                String name = String.valueOf(c.get("name"));
                Tool t = tools.get(name);
                String result;
                if (t == null) {
                    result = "error: no tool named " + name;
                } else {
                    try {
                        result = t.run().apply(Json.object(c.get("arguments")));
                    } catch (RuntimeException e) {
                        result = "error: " + e.getMessage();
                    }
                }
                calls++;
                Map<String, Object> reply = message("tool", result);
                reply.put("name", name);
                conversation.add(reply);
            }
        }
        return new Result(conversation, text, calls, false);
    }

    private static Map<String, Object> message(String role, String content) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }

    /** The text outside the tool calls. */
    private static String content(String text) {
        return HERMES.matcher(text).replaceAll("").strip();
    }

    /**
     * The calls a turn asks for: Hermes {@code <tool_call>} blocks, or a turn that is one JSON
     * object with a {@code name} and {@code arguments} (or {@code parameters}).
     *
     * @param text the model's turn
     * @return each call as {@code {"name": ..., "arguments": {...}}}
     */
    static List<Map<String, Object>> calls(String text) {
        List<Map<String, Object>> out = new ArrayList<>();
        Matcher m = HERMES.matcher(text);
        while (m.find()) {
            Map<String, Object> c = parse(m.group(1));
            if (c != null) {
                out.add(c);
            }
        }
        if (out.isEmpty() && text.startsWith("{") && text.endsWith("}")) {
            Map<String, Object> c = parse(text);
            if (c != null) {
                out.add(c);
            }
        }
        return out;
    }

    private static Map<String, Object> parse(String json) {
        try {
            Map<String, Object> o = Json.object(Json.parse(json));
            Object args = o.containsKey("arguments") ? o.get("arguments") : o.get("parameters");
            if (!(o.get("name") instanceof String name) || args == null) {
                return null;
            }
            if (args instanceof String s) {
                args = Json.parse(s);   // some models write the arguments as a JSON string
            }
            Map<String, Object> c = new LinkedHashMap<>();
            c.put("name", name);
            c.put("arguments", Json.object(args));
            return c;
        } catch (RuntimeException e) {
            return null;
        }
    }
}
