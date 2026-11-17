/**
 * Large language models: Llama-family models that load published weights (safetensors or
 * GGUF), and text generation with sampling, streaming and beam search.
 */
module io.github.zzuegg.brainiac.llm {
    requires transitive io.github.zzuegg.brainiac.nn;
    requires transitive io.github.zzuegg.brainiac.io;
    requires jdk.httpserver;
    exports io.github.zzuegg.brainiac.llm;
}
