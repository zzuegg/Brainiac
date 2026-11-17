package io.github.zzuegg.brainiac.studio.chat;

import java.net.URI;
import java.util.List;

/**
 * A provider that speaks the OpenAI Chat Completions format at its own address.
 *
 * @param id the provider id used in model keys, such as {@code deepseek}
 * @param name the name to show
 * @param base the API's base address; {@code models} and {@code chat/completions} are resolved against it
 * @param env the environment variables that may hold a key, in order
 * @param tokensField the request field for the answer's length: {@code max_tokens} or {@code max_completion_tokens}
 * @param usageOption whether to ask for token counts with {@code stream_options}; strict APIs reject the field
 * @param balance how to read the remaining balance: {@code none}, {@code deepseek}, {@code moonshot} or {@code openrouter}
 * @param note a short line shown with the provider
 */
public record Preset(String id, String name, URI base, List<String> env, String tokensField, boolean usageOption,
        String balance, String note) {

    /** Copies the list. */
    public Preset {
        env = List.copyOf(env);
    }

    /**
     * Returns this preset at another address, such as a gateway or a stand-in for tests.
     *
     * @param other the address
     * @return the preset
     */
    public Preset at(URI other) {
        return new Preset(id, name, other, env, tokensField, usageOption, balance, note);
    }

    private static Preset of(String id, String name, String base, List<String> env, String tokens, boolean usage,
            String balance, String note) {
        return new Preset(id, name, URI.create(base), env, tokens, usage, balance, note);
    }

    /** OpenAI itself. */
    public static final Preset OPENAI = of("openai", "OpenAI", "https://api.openai.com/v1/", List.of("OPENAI_API_KEY"),
            "max_completion_tokens", true, "none", "GPT, image, speech and transcription models");

    /** The other providers that take the OpenAI format, in the order Auto tries them. */
    public static final List<Preset> COMPATIBLE = List.of(
            of("huggingface", "Hugging Face", "https://router.huggingface.co/v1/", List.of("HF_TOKEN",
                    "HUGGINGFACE_HUB_TOKEN"), "max_tokens", false, "none",
                    "open models served by Hugging Face's inference providers"),
            of("deepseek", "DeepSeek", "https://api.deepseek.com/", List.of("DEEPSEEK_API_KEY"), "max_tokens", true,
                    "deepseek", "DeepSeek chat and reasoning models"),
            of("qwen", "Qwen (Alibaba Cloud)", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/",
                    List.of("DASHSCOPE_API_KEY"), "max_tokens", true, "none", "Qwen models through Model Studio"),
            of("moonshot", "Kimi (Moonshot)", "https://api.moonshot.ai/v1/", List.of("MOONSHOT_API_KEY"), "max_tokens",
                    false, "moonshot", "Kimi models"),
            of("zhipu", "GLM (Zhipu)", "https://open.bigmodel.cn/api/paas/v4/", List.of("ZHIPUAI_API_KEY",
                    "BIGMODEL_API_KEY"), "max_tokens", false, "none", "GLM models from BigModel"),
            of("minimax", "MiniMax", "https://api.minimax.io/v1/", List.of("MINIMAX_API_KEY"), "max_tokens", false,
                    "none", "MiniMax text models"),
            of("mistral", "Mistral", "https://api.mistral.ai/v1/", List.of("MISTRAL_API_KEY"), "max_tokens", false,
                    "none", "Mistral and Codestral models"),
            of("xai", "xAI", "https://api.x.ai/v1/", List.of("XAI_API_KEY"), "max_tokens", true, "none", "Grok models"),
            of("groq", "Groq", "https://api.groq.com/openai/v1/", List.of("GROQ_API_KEY"), "max_tokens", true, "none",
                    "open models on fast inference hardware"),
            of("openrouter", "OpenRouter", "https://openrouter.ai/api/v1/", List.of("OPENROUTER_API_KEY"), "max_tokens",
                    true, "openrouter", "hundreds of models from many providers through one key"));
}
