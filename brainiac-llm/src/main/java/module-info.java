/**
 * Large language models: Llama-family models that load published weights (safetensors or
 * GGUF), text generation with sampling, streaming and beam search, and encoders for text
 * (BERT, XLM-R) and images (CLIP, SigLIP, ViT, DINOv2).
 */
module io.github.zzuegg.brainiac.llm {
    requires transitive io.github.zzuegg.brainiac.nn;
    requires transitive io.github.zzuegg.brainiac.io;
    requires jdk.httpserver;
    requires transitive java.desktop;
    exports io.github.zzuegg.brainiac.llm;
}
