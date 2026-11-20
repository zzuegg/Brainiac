package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
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
            Parsed parsed = read(text, tools.keySet());
            List<Map<String, Object>> requested = parsed.calls();
            if (requested.isEmpty()) {
                conversation.add(message("assistant", text));
                return new Result(conversation, text, calls, true);
            }
            Map<String, Object> assistant = message("assistant", parsed.content());
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

    /**
     * The text of a turn outside its calls.
     *
     * @param text the model's turn
     * @return the rest, stripped
     */
    public static String content(String text) {
        return read(text, null).content();
    }

    /**
     * The calls a turn asks for: Hermes {@code <tool_call>} blocks (a missing end tag or a code
     * fence inside is fine), or a turn that is one JSON object with a {@code name} and
     * {@code arguments} (or {@code parameters}), or a list of them.
     *
     * @param text the model's turn
     * @return each call as {@code {"name": ..., "arguments": {...}}}
     */
    public static List<Map<String, Object>> calls(String text) {
        return read(text, null).calls();
    }

    /**
     * The calls a turn asks for, read leniently, as small local models write them. Besides
     * {@link #calls(String)}: JSON calls anywhere in the turn, also in code fences, with the
     * arguments under {@code args} or {@code input} or left out, or wrapped as
     * {@code {"function": {...}}}; and Python-style calls such as {@code [list_files(path="a")]}.
     * Calls found outside {@code <tool_call>} tags count only if they name a known tool.
     *
     * @param text the model's turn
     * @param known the tools' names, or null to read only the strict forms
     * @return the calls and the turn's words without them
     */
    public static Parsed read(String text, Set<String> known) {
        String t = text == null ? "" : text;
        List<Map<String, Object>> out = new ArrayList<>();
        List<int[]> spans = new ArrayList<>();
        // 1. Hermes tags
        int from = 0;
        while (true) {
            int tag = t.indexOf(OPEN, from);
            if (tag < 0) {
                break;
            }
            int i = skipFence(t, tag + OPEN.length());
            int end = i < t.length() && (t.charAt(i) == '{' || t.charAt(i) == '[') ? balanced(t, i) : -1;
            List<Map<String, Object>> found = end < 0 ? List.of() : toCalls(parseJson(t.substring(i, end)), true);
            if (found.isEmpty()) {
                from = tag + OPEN.length();
                continue;
            }
            out.addAll(found);
            int stop = skipFence(t, end);
            int close = t.startsWith(CLOSE, stop) ? stop + CLOSE.length() : end;
            spans.add(new int[] {tag, close});
            from = close;
        }
        if (out.isEmpty()) {
            // 2. JSON in the turn: the whole turn in the strict form, anywhere for known tools
            for (int i = 0; i < t.length(); i++) {
                char c = t.charAt(i);
                if (c != '{' && c != '[') {
                    continue;
                }
                int end = balanced(t, i);
                if (end < 0) {
                    continue;
                }
                Object v = parseJson(t.substring(i, end));
                if (v == null) {
                    continue;
                }
                List<Map<String, Object>> found = toCalls(v, known != null);
                boolean whole = unfenced(t, i, end).isBlank();
                if (!found.isEmpty() && (known == null ? whole : found.stream().allMatch(f -> known.contains(f.get("name"))))) {
                    out.addAll(found);
                    spans.add(fenced(t, i, end));
                }
                i = end - 1;
            }
        }
        if (out.isEmpty() && known != null) {
            // 3. Python-style calls of known tools, in a code fence or as the whole turn
            Matcher m = PY_CALL.matcher(t);
            while (m.find()) {
                if (!known.contains(m.group(1))) {
                    continue;
                }
                int open = m.end() - 1;
                int close = closing(t, open);
                if (close < 0) {
                    continue;
                }
                Map<String, Object> args = kwargs(t.substring(open + 1, close));
                if (args == null) {
                    continue;
                }
                boolean inFence = t.lastIndexOf("```", m.start()) >= 0
                        && (t.substring(0, m.start()).split("```", -1).length % 2 == 0);
                if (!inFence && !PY_WHOLE.matcher(t.strip()).matches()) {
                    continue;
                }
                Map<String, Object> c = new LinkedHashMap<>();
                c.put("name", m.group(1));
                c.put("arguments", args);
                out.add(c);
                spans.add(new int[] {m.start(), close + 1});
                m.region(close + 1, t.length());
            }
            if (!out.isEmpty() && !PY_WHOLE.matcher(t.strip()).matches()) {
                // the fences that held the calls go with them
                spans.clear();
                spans.add(fenceAround(t, t.indexOf("```")));
            } else if (!out.isEmpty()) {
                spans.clear();
                spans.add(new int[] {0, t.length()});
            }
        }
        StringBuilder rest = new StringBuilder();
        int at = 0;
        for (int[] sp : spans) {
            if (sp[0] >= at) {
                rest.append(t, at, sp[0]);
                at = sp[1];
            }
        }
        rest.append(t.substring(Math.min(at, t.length())));
        String words = HERMES.matcher(rest.toString()).replaceAll("").strip();
        return new Parsed(out, words);
    }

    /**
     * A turn read for calls.
     *
     * @param calls each as {@code {"name": ..., "arguments": {...}}}
     * @param content the words outside them, stripped
     */
    public record Parsed(List<Map<String, Object>> calls, String content) {
    }

    private static final String OPEN = "<tool_call>";
    private static final String CLOSE = "</tool_call>";
    private static final Pattern PY_CALL = Pattern.compile("(?:default_api\\.|functions\\.)?\\b([A-Za-z_]\\w*)\\s*\\(");
    private static final Pattern PY_WHOLE = Pattern.compile("(?s)\\[?\\s*(?:print\\()?\\s*(?:default_api\\.|functions\\.)?[A-Za-z_]\\w*\\s*\\(.*\\)\\s*\\)?\\s*\\]?");

    // past whitespace and an opening code fence such as ```json
    private static int skipFence(String t, int i) {
        int j = i;
        while (j < t.length() && Character.isWhitespace(t.charAt(j))) {
            j++;
        }
        if (t.startsWith("```", j)) {
            j += 3;
            while (j < t.length() && (Character.isLetterOrDigit(t.charAt(j)) || t.charAt(j) == '_')) {
                j++;
            }
            while (j < t.length() && Character.isWhitespace(t.charAt(j))) {
                j++;
            }
        }
        return j;
    }

    // the text around a span with its code fence, if one holds only it, taken out
    private static String unfenced(String t, int from, int to) {
        int[] f = fenced(t, from, to);
        return (t.substring(0, f[0]) + t.substring(f[1])).replaceAll("(?s)<tool_call>|</tool_call>", "");
    }

    // a span widened to the code fence that holds only it
    private static int[] fenced(String t, int from, int to) {
        String before = t.substring(0, from);
        Matcher open = Pattern.compile("```[\\w-]*\\s*$").matcher(before);
        if (open.find()) {
            Matcher close = Pattern.compile("^\\s*```").matcher(t.substring(to));
            if (close.find()) {
                return new int[] {open.start(), to + close.end()};
            }
        }
        return new int[] {from, to};
    }

    // the code fence that starts at i, whole
    private static int[] fenceAround(String t, int i) {
        if (i < 0) {
            return new int[] {0, 0};
        }
        int end = t.indexOf("```", i + 3);
        return new int[] {i, end < 0 ? t.length() : end + 3};
    }

    // the end (exclusive) of the JSON object or array that starts at i, or -1
    private static int balanced(String t, int i) {
        int depth = 0;
        boolean str = false;
        for (int j = i; j < t.length(); j++) {
            char c = t.charAt(j);
            if (str) {
                if (c == '\\') {
                    j++;
                } else if (c == '"') {
                    str = false;
                }
                continue;
            }
            switch (c) {
                case '"' -> str = true;
                case '{', '[' -> depth++;
                case '}', ']' -> {
                    if (--depth == 0) {
                        return j + 1;
                    }
                }
                default -> {
                }
            }
        }
        return -1;
    }

    // the closing parenthesis for the one at i, past quoted text, or -1
    private static int closing(String t, int i) {
        int depth = 0;
        char quote = 0;
        for (int j = i; j < t.length(); j++) {
            char c = t.charAt(j);
            if (quote != 0) {
                if (c == '\\') {
                    j++;
                } else if (c == quote) {
                    quote = 0;
                }
                continue;
            }
            if (c == '"' || c == '\'') {
                quote = c;
            } else if (c == '(' || c == '[' || c == '{') {
                depth++;
            } else if (c == ')' || c == ']' || c == '}') {
                if (--depth == 0) {
                    return c == ')' ? j : -1;
                }
            }
        }
        return -1;
    }

    // Python keyword arguments as JSON values, or null if they are something else
    private static Map<String, Object> kwargs(String inner) {
        Map<String, Object> args = new LinkedHashMap<>();
        if (inner.isBlank()) {
            return args;
        }
        String json = pythonToJson(inner);
        if (json == null) {
            return null;
        }
        // name=value pairs become a JSON object
        StringBuilder o = new StringBuilder("{");
        int depth = 0;
        boolean str = false;
        int start = 0;
        List<String> parts = new ArrayList<>();
        for (int j = 0; j < json.length(); j++) {
            char c = json.charAt(j);
            if (str) {
                if (c == '\\') {
                    j++;
                } else if (c == '"') {
                    str = false;
                }
                continue;
            }
            if (c == '"') {
                str = true;
            } else if (c == '{' || c == '[' || c == '(') {
                depth++;
            } else if (c == '}' || c == ']' || c == ')') {
                depth--;
            } else if (c == ',' && depth == 0) {
                parts.add(json.substring(start, j));
                start = j + 1;
            }
        }
        parts.add(json.substring(start));
        for (String part : parts) {
            if (part.isBlank()) {
                continue;
            }
            Matcher kv = Pattern.compile("(?s)\\s*([A-Za-z_]\\w*)\\s*[=:]\\s*(.+?)\\s*").matcher(part);
            if (!kv.matches()) {
                return null;
            }
            if (o.length() > 1) {
                o.append(',');
            }
            o.append('"').append(kv.group(1)).append("\":").append(kv.group(2));
        }
        Object v = parseJson(o.append('}').toString());
        if (!(v instanceof Map<?, ?>)) {
            return null;
        }
        args.putAll(Json.object(v));
        return args;
    }

    // Python literals as JSON: quotes, True, False and None
    private static String pythonToJson(String py) {
        StringBuilder b = new StringBuilder();
        for (int j = 0; j < py.length(); j++) {
            char c = py.charAt(j);
            if (c == '"' || c == '\'') {
                b.append('"');
                int k = j + 1;
                for (; k < py.length() && py.charAt(k) != c; k++) {
                    char d = py.charAt(k);
                    if (d == '\\' && k + 1 < py.length()) {
                        char e = py.charAt(++k);
                        b.append(e == '\'' ? "'" : "\\" + e);
                    } else if (d == '"') {
                        b.append("\\\"");
                    } else if (d == '\n') {
                        b.append("\\n");
                    } else {
                        b.append(d);
                    }
                }
                if (k >= py.length()) {
                    return null;
                }
                b.append('"');
                j = k;
            } else if (Character.isLetter(c)) {
                int k = j;
                while (k < py.length() && (Character.isLetterOrDigit(py.charAt(k)) || py.charAt(k) == '_')) {
                    k++;
                }
                String word = py.substring(j, k);
                boolean key = py.substring(k).stripLeading().startsWith("=") && !py.substring(k).stripLeading().startsWith("==");
                b.append(key ? word : switch (word) {
                    case "True", "true" -> "true";
                    case "False", "false" -> "false";
                    case "None", "null" -> "null";
                    default -> "\"" + word + "\"";
                });
                j = k - 1;
            } else {
                b.append(c);
            }
        }
        return b.toString();
    }

    private static Object parseJson(String json) {
        try {
            return Json.parse(json);
        } catch (RuntimeException e) {
            return null;
        }
    }

    // a JSON value as calls: a call, a list of calls, or {tool_calls: [...]}; empty if it is not one
    private static List<Map<String, Object>> toCalls(Object v, boolean lenient) {
        List<Map<String, Object>> out = new ArrayList<>();
        if (v instanceof List<?> l) {
            for (Object o : l) {
                Map<String, Object> c = toCall(o, lenient);
                if (c == null) {
                    return List.of();
                }
                out.add(c);
            }
            return out;
        }
        if (v instanceof Map<?, ?> m && m.get("tool_calls") instanceof List<?> l) {
            return toCalls(l, lenient);
        }
        Map<String, Object> c = toCall(v, lenient);
        return c == null ? List.of() : List.of(c);
    }

    private static Map<String, Object> toCall(Object v, boolean lenient) {
        if (!(v instanceof Map<?, ?> raw)) {
            return null;
        }
        Map<String, Object> o = Json.object(raw);
        for (String wrap : List.of("function", "tool_call", "call")) {
            if (o.get(wrap) instanceof Map<?, ?> inner) {
                return toCall(inner, lenient);
            }
        }
        Object name = o.get("name");
        if (lenient && !(name instanceof String)) {
            name = o.containsKey("tool") ? o.get("tool") : o.containsKey("tool_name") ? o.get("tool_name") : o.get("function");
        }
        if (!(name instanceof String n) || n.isBlank()) {
            return null;
        }
        List<String> keys = lenient ? List.of("arguments", "parameters", "args", "input", "params")
                : List.of("arguments", "parameters");
        Object args = null;
        for (String k : keys) {
            if (o.containsKey(k)) {
                args = o.get(k);
                break;
            }
        }
        if (args == null) {
            boolean bare = o.keySet().stream().allMatch(k -> List.of("name", "tool", "tool_name", "function", "type", "id")
                    .contains(k));
            if (!lenient || !bare) {
                return null;
            }
            args = Map.of();
        }
        if (args instanceof String s) {
            args = s.isBlank() ? Map.of() : parseJson(s);   // some models write the arguments as a JSON string
        }
        if (!(args instanceof Map<?, ?> a)) {
            return null;
        }
        Map<String, Object> c = new LinkedHashMap<>();
        String plain = n.strip();
        c.put("name", plain.startsWith("functions.") || plain.startsWith("default_api.")
                ? plain.substring(plain.indexOf('.') + 1) : plain);
        c.put("arguments", Json.object(a));
        return c;
    }
}
