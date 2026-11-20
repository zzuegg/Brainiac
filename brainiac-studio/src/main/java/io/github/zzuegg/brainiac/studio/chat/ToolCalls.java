package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.llm.Agent;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Tool calls in the two shapes the studio meets them. Agents keep a conversation as plain text:
 * the model's calls as Hermes {@code <tool_call>} blocks and the answers as
 * {@code <tool_response>} blocks, which any model can read. Providers that take tools natively
 * (OpenAI-format APIs, Anthropic) are sent the same conversation as their own tool-use and
 * tool-result messages, and the calls they make are written back as Hermes text, so the agent
 * never needs to know which kind of model it talks to.
 */
public final class ToolCalls {

    private static final Pattern RESPONSE = Pattern.compile("<tool_response>\\s*(.*?)\\s*</tool_response>", Pattern.DOTALL);

    private ToolCalls() {
    }

    /** One call a model made, with the id it is answered by. */
    public record Call(String id, String name, Map<String, Object> arguments) {
    }

    /** The answer to a call. */
    public record Result(String id, String name, String text) {
    }

    /**
     * One message of a conversation in tool terms.
     *
     * @param role {@code user} or {@code assistant}
     * @param text its words, without call or response blocks
     * @param calls the calls an assistant message makes
     * @param results the answers a user message carries, in the order of the calls before it
     * @param source the message itself
     */
    public record Turn(String role, String text, List<Call> calls, List<Result> results, ChatMessage source) {
    }

    /**
     * Writes a call as Hermes text.
     *
     * @param name the tool
     * @param arguments its arguments
     * @return {@code <tool_call>...</tool_call>}
     */
    public static String hermes(String name, Map<String, Object> arguments) {
        Map<String, Object> c = new LinkedHashMap<>();
        c.put("name", name);
        c.put("arguments", arguments == null ? Map.of() : arguments);
        return "<tool_call>\n" + Json.write(c) + "\n</tool_call>";
    }

    /**
     * Describes tools in the system prompt, in the Hermes format Qwen and many open models are
     * trained on.
     *
     * @param tools {@code {name, description, parameters}} each
     * @return the instructions to add
     */
    public static String instructions(List<Map<String, Object>> tools) {
        StringBuilder s = new StringBuilder("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n")
                .append("The functions, one per line as name(argument: type, optional?: type), are within <tools></tools> ")
                .append("XML tags:\n<tools>\n");
        for (Map<String, Object> t : tools) {
            s.append(signature(t)).append('\n');
        }
        return s.append("</tools>\n\nFor each function call, return a json object with function name and arguments ")
                .append("within <tool_call></tool_call> XML tags:\n<tool_call>\n")
                .append("{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n\n")
                .append("The results come back in the next message, each within <tool_response></tool_response> ")
                .append("tags. Stop after your calls and wait for them.").toString();
    }

    /**
     * A tool as one short line, with a line for the arguments whose names do not say enough:
     * {@code name(path: string, from?: integer) - what it does}. Far shorter than its JSON schema,
     * which matters for small models with small windows.
     *
     * @param tool {@code {name, description, parameters}}
     * @return the lines
     */
    public static String signature(Map<String, Object> tool) {
        Map<String, Object> params = Json.object(tool.getOrDefault("parameters", Map.of()));
        Map<String, Object> props = params.get("properties") instanceof Map<?, ?> m ? Json.object(m) : Map.of();
        List<?> required = params.get("required") instanceof List<?> l ? l : List.of();
        StringBuilder s = new StringBuilder("- ").append(tool.get("name")).append('(');
        StringBuilder notes = new StringBuilder();
        boolean first = true;
        for (Map.Entry<String, Object> p : props.entrySet()) {
            Map<String, Object> schema = p.getValue() instanceof Map<?, ?> ps ? Json.object(ps) : Map.of();
            s.append(first ? "" : ", ").append(p.getKey()).append(required.contains(p.getKey()) ? "" : "?").append(": ")
                    .append(type(schema));
            first = false;
            String d = String.valueOf(schema.getOrDefault("description", "")).strip();
            if (!d.isEmpty() && !d.equalsIgnoreCase("the " + p.getKey().replace('_', ' '))) {
                notes.append(notes.isEmpty() ? "" : "; ").append(p.getKey()).append(": ").append(d);
            }
        }
        s.append(") - ").append(tool.get("description"));
        if (!notes.isEmpty()) {
            s.append("\n  ").append(notes);
        }
        return s.toString();
    }

    // a JSON schema's type in a few characters
    private static String type(Map<String, Object> schema) {
        if (schema.get("enum") instanceof List<?> e) {
            return String.join("|", e.stream().map(String::valueOf).toList());
        }
        String t = String.valueOf(schema.getOrDefault("type", "any"));
        if (t.equals("array")) {
            Map<String, Object> items = schema.get("items") instanceof Map<?, ?> i ? Json.object(i) : Map.of();
            return type(items) + "[]";
        }
        if (t.equals("object") && schema.get("properties") instanceof Map<?, ?> m && !m.isEmpty()) {
            StringBuilder b = new StringBuilder("{");
            List<?> req = schema.get("required") instanceof List<?> l ? l : List.of();
            for (Map.Entry<String, Object> p : Json.object(m).entrySet()) {
                b.append(b.length() > 1 ? ", " : "").append(p.getKey()).append(req.contains(p.getKey()) ? "" : "?")
                        .append(": ").append(type(p.getValue() instanceof Map<?, ?> ps ? Json.object(ps) : Map.of()));
            }
            return b.append('}').toString();
        }
        return t;
    }

    /**
     * Reads a conversation kept as text in tool terms. Ids are made from the message's place, so
     * the same conversation always gets the same ids. A user message only counts as answers when
     * it follows calls and holds a response for each.
     *
     * @param messages the conversation
     * @return a turn per message
     */
    public static List<Turn> turns(List<ChatMessage> messages) {
        List<Turn> out = new ArrayList<>();
        List<Call> open = List.of();
        for (int i = 0; i < messages.size(); i++) {
            ChatMessage m = messages.get(i);
            if (m.role().equals("assistant")) {
                List<Call> calls = new ArrayList<>();
                List<Map<String, Object>> found = Agent.calls(m.text());
                for (int k = 0; k < found.size(); k++) {
                    Map<String, Object> c = found.get(k);
                    calls.add(new Call("call_" + i + "_" + k, String.valueOf(c.get("name")), Json.object(c.get("arguments"))));
                }
                out.add(new Turn("assistant", calls.isEmpty() ? m.text().strip() : Agent.content(m.text()), calls,
                        List.of(), m));
                open = calls;
                continue;
            }
            List<Result> results = new ArrayList<>();
            Matcher r = RESPONSE.matcher(m.text());
            int end = 0;
            while (r.find() && results.size() < open.size()) {
                Call c = open.get(results.size());
                results.add(new Result(c.id(), c.name(), r.group(1)));
                end = r.end();
            }
            if (!open.isEmpty() && results.size() == open.size()) {
                out.add(new Turn("user", m.text().substring(end).strip(), List.of(), results, m));
            } else {
                out.add(new Turn("user", m.text(), List.of(), List.of(), m));
            }
            open = List.of();
        }
        return out;
    }

    /**
     * Turns a tool definition into the OpenAI {@code tools} entry.
     *
     * @param tool {@code {name, description, parameters}}
     * @return {@code {type: function, function: {...}}}
     */
    static Map<String, Object> openAi(Map<String, Object> tool) {
        Map<String, Object> f = new LinkedHashMap<>();
        f.put("name", tool.get("name"));
        f.put("description", tool.get("description"));
        f.put("parameters", tool.get("parameters"));
        return Map.of("type", "function", "function", f);
    }

    /**
     * Turns a tool definition into an Anthropic {@code tools} entry.
     *
     * @param tool {@code {name, description, parameters}}
     * @return {@code {name, description, input_schema}}
     */
    static Map<String, Object> anthropic(Map<String, Object> tool) {
        Map<String, Object> f = new LinkedHashMap<>();
        f.put("name", tool.get("name"));
        f.put("description", tool.get("description"));
        f.put("input_schema", tool.get("parameters"));
        return f;
    }

    /** Gathers a streamed call whose arguments arrive in pieces. */
    static final class Pending {
        String id;
        String name;
        final StringBuilder arguments = new StringBuilder();

        /**
         * Writes the call as Hermes text.
         *
         * @return the call, or an empty string if it has no name
         */
        String hermes() {
            if (name == null || name.isBlank()) {
                return "";
            }
            Map<String, Object> args;
            try {
                String a = arguments.toString().strip();
                args = a.isEmpty() ? Map.of() : Json.object(Json.parse(a));
            } catch (RuntimeException e) {
                args = Map.of("_unreadable", arguments.toString());
            }
            return ToolCalls.hermes(name, args);
        }
    }
}
