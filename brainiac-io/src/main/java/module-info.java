/**
 * Reading and writing the file formats other tools use: JSON, safetensors and GGUF weights
 * (memory-mapped), the name mapping that turns another library's weight names into
 * Brainiac's, and pretrained tokenizers (Hugging Face, SentencePiece, GGUF).
 */
module io.github.zzuegg.brainiac.io {
    requires transitive io.github.zzuegg.brainiac.tensor;
    requires transitive io.github.zzuegg.brainiac.data;
    requires java.xml;
    requires java.net.http;
    exports io.github.zzuegg.brainiac.io;
}
