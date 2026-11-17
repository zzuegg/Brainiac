/**
 * Reading and writing the file formats other tools use: JSON, safetensors weights
 * (memory-mapped), and the name mapping that turns another library's weight names into
 * Brainiac's.
 */
module io.github.zzuegg.brainiac.io {
    requires transitive io.github.zzuegg.brainiac.tensor;
    exports io.github.zzuegg.brainiac.io;
}
