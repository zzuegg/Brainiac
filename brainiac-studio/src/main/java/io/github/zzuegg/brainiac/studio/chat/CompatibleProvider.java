package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * A provider that speaks the OpenAI Chat Completions format: Hugging Face, DeepSeek, Qwen,
 * Kimi, GLM, MiniMax, Mistral, xAI, Groq, OpenRouter, and OpenAI itself.
 */
class CompatibleProvider extends RemoteProvider {

    final Preset preset;

    CompatibleProvider(Keys keys, Preset preset) {
        super(keys, preset.base());
        this.preset = preset;
    }

    @Override
    public String id() {
        return preset.id();
    }

    @Override
    public String name() {
        return preset.name();
    }

    @Override
    String note() {
        return preset.note();
    }

    @Override
    boolean reportsBalance() {
        return !preset.balance().equals("none");
    }

    @Override
    List<String> auth(String key) {
        if (key == null || key.isEmpty()) {
            return List.of();
        }
        String org = options.getOrDefault(ProviderSettings.BILL_TO, "");
        return org.isEmpty() ? List.of("Authorization", "Bearer " + key)
                : List.of("Authorization", "Bearer " + key, "X-HF-Bill-To", org);
    }

    @Override
    List<String> optionNames() {
        return preset.id().equals("huggingface") ? List.of(ProviderSettings.BILL_TO) : List.of();
    }

    @Override
    List<ModelInfo> models(String key) throws IOException, InterruptedException {
        List<ModelInfo> out = new ArrayList<>();
        for (Object o : arr(getJson(uri("models"), key).get("data"))) {
            Map<String, Object> m = obj(o);
            String id = Json.string(m, "id", null);
            List<String> media = id == null ? List.of() : media(id);
            if (media.isEmpty()) {
                continue;
            }
            long context = num(m.get("context_length"));
            String detail = ModelInfo.context(context);
            if (m.get("providers") instanceof List<?> hosts) {
                // Hugging Face's router: each model lists the inference providers that serve it
                int live = 0;
                double in = Double.NaN;
                double outPrice = Double.NaN;
                for (Object h : hosts) {
                    Map<String, Object> host = obj(h);
                    String status = Json.string(host, "status", "live");
                    if (!status.equals("live")) {
                        continue;
                    }
                    live++;
                    context = Math.max(context, num(host.get("context_length")));
                    Map<String, Object> price = obj(host.get("pricing"));
                    if (price.get("input") instanceof Number pi && price.get("output") instanceof Number po
                            && (Double.isNaN(in) || pi.doubleValue() + po.doubleValue() < in + outPrice)) {
                        in = pi.doubleValue();
                        outPrice = po.doubleValue();
                    }
                }
                if (live == 0) {
                    continue;   // listed, but no provider serves it now
                }
                List<String> bits = new ArrayList<>();
                if (context > 0) {
                    bits.add(ModelInfo.context(context));
                }
                bits.add(live + (live == 1 ? " host" : " hosts"));
                if (!Double.isNaN(in)) {
                    bits.add(String.format(Locale.ROOT, "from $%s in / $%s out per 1M", price(in), price(outPrice)));
                }
                detail = String.join(" · ", bits);
            }
            out.add(new ModelInfo(id(), id, Json.string(m, "name", id), media, Math.max(0, num(m.get("created"))),
                    detail));
        }
        out.sort(Comparator.comparingLong(ModelInfo::created).reversed());
        return out;
    }

    private static String price(double d) {
        // cents for most prices; two significant digits for fractions of a cent
        return d >= 0.01 || d == 0 ? String.format(Locale.ROOT, "%.2f", d)
                : new java.math.BigDecimal(d).round(new java.math.MathContext(2)).stripTrailingZeros().toPlainString();
    }

    /** What a model does, from its name: the lists have no capability fields. */
    List<String> media(String id) {
        String s = id.toLowerCase(Locale.ROOT);
        if (s.contains("embed") || s.contains("rerank") || s.contains("/bge-") || s.startsWith("bge-") || s.contains("moderation") || s.contains("guard")
                || s.contains("whisper") || s.contains("tts") || s.contains("ocr")) {
            return List.of();
        }
        if (s.contains("image") || s.contains("flux") || s.contains("stable-diffusion")) {
            return List.of();   // pictures from these providers are not wired yet
        }
        return s.contains("-vl") || s.contains("vision") || s.contains("pixtral") ? List.of("text", "vision")
                : List.of("text");
    }

    /**
     * Whether a model accepts a temperature; OpenAI's reasoning models answer only at their own.
     *
     * @param model the model's id
     * @return true to send the temperature asked for
     */
    boolean takesTemperature(String model) {
        return !(preset.id().equals("openai") && model.matches("(gpt-5|o\\d).*"));
    }

    @Override
    ChatResult chat(ChatRequest request, ChatListener listener, String key) throws IOException, InterruptedException {
        List<Object> messages = new ArrayList<>();
        if (!request.system().isBlank()) {
            messages.add(Map.of("role", "system", "content", request.system()));
        }
        boolean tools = !request.tools().isEmpty();
        for (ToolCalls.Turn t : tools ? ToolCalls.turns(request.messages()) : plain(request.messages())) {
            ChatMessage m = t.source();
            if (!t.calls().isEmpty()) {
                List<Object> calls = new ArrayList<>();
                for (ToolCalls.Call c : t.calls()) {
                    calls.add(Map.of("id", c.id(), "type", "function", "function",
                            Map.of("name", c.name(), "arguments", Json.write(c.arguments()))));
                }
                Map<String, Object> a = new LinkedHashMap<>();
                a.put("role", "assistant");
                a.put("content", t.text().isEmpty() ? null : t.text());
                a.put("tool_calls", calls);
                messages.add(a);
                continue;
            }
            for (ToolCalls.Result r : t.results()) {
                messages.add(Map.of("role", "tool", "tool_call_id", r.id(), "content", r.text()));
            }
            if (!t.results().isEmpty() && t.text().isEmpty() && m.images().isEmpty()) {
                continue;
            }
            String text = t.results().isEmpty() ? m.text() : t.text();
            if (m.images().isEmpty()) {
                messages.add(Map.of("role", m.role(), "content", text));
                continue;
            }
            List<Object> parts = new ArrayList<>();
            if (!text.isBlank()) {
                parts.add(Map.of("type", "text", "text", text));
            }
            for (Media image : m.images()) {
                parts.add(Map.of("type", "image_url", "image_url",
                        Map.of("url", "data:" + image.mime() + ";base64," + ChatMessage.base64(image))));
            }
            messages.add(Map.of("role", m.role(), "content", parts));
        }
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", request.model());
        body.put("messages", messages);
        body.put(preset.tokensField(), request.maxTokens());
        body.put("stream", true);
        if (request.temperature() != null && takesTemperature(request.model())) {
            body.put("temperature", request.temperature());
        }
        // top_p is in every OpenAI-format API; top-k and a repeat penalty are not, so they stay out
        if (request.sampler().topP() != null && takesTemperature(request.model())) {
            body.put("top_p", request.sampler().topP());
        }
        if (request.format().kind().equals("json")) {
            body.put("response_format", Map.of("type", "json_object"));
        } else if (request.format().kind().equals("schema")) {
            // strict when the schema is closed already, as strict mode demands; else the schema guides the answer
            Map<String, Object> schema = new LinkedHashMap<>();
            schema.put("name", "answer");
            schema.put("schema", request.format().schema());
            if (Boolean.FALSE.equals(request.format().schema().get("additionalProperties"))) {
                schema.put("strict", true);
            }
            body.put("response_format", Map.of("type", "json_schema", "json_schema", schema));
        }
        if (preset.usageOption()) {
            body.put("stream_options", Map.of("include_usage", true));
        }
        if (tools) {
            body.put("tools", request.tools().stream().map(ToolCalls::openAi).toList());
        }
        Map<Integer, ToolCalls.Pending> pending = new java.util.TreeMap<>();
        long[] usage = {-1, -1};
        String[] stop = {"end"};
        stream(uri("chat/completions"), key, body, (event, data) -> {
            if (data.equals("[DONE]")) {
                return false;
            }
            Map<String, Object> e = Json.object(Json.parse(data));
            if (e.get("error") != null) {
                throw new IOException(name() + " stopped the answer: " + Json.string(obj(e.get("error")), "message", data));
            }
            for (Object c : arr(e.get("choices"))) {
                Map<String, Object> choice = obj(c);
                Map<String, Object> delta = obj(choice.get("delta"));
                if (delta.get("content") instanceof String s && !s.isEmpty()) {
                    listener.text(s);
                }
                for (Object tc : arr(delta.get("tool_calls"))) {
                    Map<String, Object> d = obj(tc);
                    ToolCalls.Pending p = pending.computeIfAbsent((int) Math.max(0, num(d.get("index"))),
                            k -> new ToolCalls.Pending());
                    if (d.get("id") instanceof String id) {
                        p.id = id;
                    }
                    Map<String, Object> f = obj(d.get("function"));
                    if (f.get("name") instanceof String n && !n.isEmpty()) {
                        p.name = n;
                    }
                    if (f.get("arguments") instanceof String a) {
                        p.arguments.append(a);
                    }
                }
                if (delta.get("refusal") instanceof String s && !s.isEmpty()) {
                    listener.text(s);
                    stop[0] = "refusal";
                }
                if (choice.get("finish_reason") instanceof String f) {
                    stop[0] = switch (f) {
                        case "stop" -> stop[0];
                        case "length" -> "length";
                        case "content_filter" -> "refusal";
                        default -> f;
                    };
                }
            }
            // usually at the top; some providers put it in the choice
            Map<String, Object> u = obj(e.get("usage"));
            if (u.isEmpty() && !arr(e.get("choices")).isEmpty()) {
                u = obj(obj(arr(e.get("choices")).getFirst()).get("usage"));
            }
            if (!u.isEmpty()) {
                usage[0] = num(u.get("prompt_tokens"));
                usage[1] = num(u.get("completion_tokens"));
            }
            return true;
        });
        for (ToolCalls.Pending p : pending.values()) {
            String call = p.hermes();
            if (!call.isEmpty()) {
                listener.text("\n" + call);
            }
        }
        return new ChatResult(stop[0].equals("tool_calls") ? "end" : stop[0], usage[0], usage[1]);
    }

    // a conversation read as plain messages, with no tool turns
    static List<ToolCalls.Turn> plain(List<ChatMessage> messages) {
        return messages.stream().map(m -> new ToolCalls.Turn(m.role(), m.text(), List.of(), List.of(), m)).toList();
    }

    @Override
    public boolean nativeTools(String model) {
        // Hugging Face routes to many hosts, and not all of them take tools
        return !preset.id().equals("huggingface");
    }


    @Override
    Balance balance(String key) throws IOException, InterruptedException {
        switch (preset.balance()) {
            case "deepseek" -> {
                Map<String, Object> b = getJson(uri("user/balance"), key);
                for (Object o : arr(b.get("balance_infos"))) {
                    Map<String, Object> info = obj(o);
                    return new Balance(Double.parseDouble(Json.string(info, "total_balance", "0")),
                            Json.string(info, "currency", ""), Boolean.FALSE.equals(b.get("is_available"))
                                    ? "too low to make calls" : "granted " + Json.string(info, "granted_balance", "0")
                                    + ", topped up " + Json.string(info, "topped_up_balance", "0"));
                }
                return null;
            }
            case "moonshot" -> {
                Map<String, Object> d = obj(getJson(uri("users/me/balance"), key).get("data"));
                return new Balance(Json.number(d, "available_balance", 0), "USD", "cash "
                        + Json.number(d, "cash_balance", 0) + ", vouchers " + Json.number(d, "voucher_balance", 0));
            }
            case "openrouter" -> {
                Map<String, Object> d = obj(getJson(uri("key"), key).get("data"));
                double used = Json.number(d, "usage", 0);
                if (d.get("limit_remaining") instanceof Number left) {
                    return new Balance(left.doubleValue(), "USD", String.format(Locale.ROOT, "used %.2f of this key's limit", used));
                }
                return new Balance(Double.NaN, "USD", String.format(Locale.ROOT, "no limit on this key; used %.2f", used));
            }
            default -> {
                return super.balance(key);
            }
        }
    }
}
