package io.github.zzuegg.brainiac.backend.vulkan;

import io.github.zzuegg.brainiac.tensor.DType;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

import static org.lwjgl.util.shaderc.Shaderc.shaderc_compilation_status_success;
import static org.lwjgl.util.shaderc.Shaderc.shaderc_compile_into_spv;
import static org.lwjgl.util.shaderc.Shaderc.shaderc_compile_options_initialize;
import static org.lwjgl.util.shaderc.Shaderc.shaderc_compile_options_set_optimization_level;
import static org.lwjgl.util.shaderc.Shaderc.shaderc_compile_options_set_target_env;
import static org.lwjgl.util.shaderc.Shaderc.shaderc_compiler_initialize;
import static org.lwjgl.util.shaderc.Shaderc.shaderc_compute_shader;
import static org.lwjgl.util.shaderc.Shaderc.shaderc_env_version_vulkan_1_2;
import static org.lwjgl.util.shaderc.Shaderc.shaderc_optimization_level_performance;
import static org.lwjgl.util.shaderc.Shaderc.shaderc_result_get_bytes;
import static org.lwjgl.util.shaderc.Shaderc.shaderc_result_get_compilation_status;
import static org.lwjgl.util.shaderc.Shaderc.shaderc_result_get_error_message;
import static org.lwjgl.util.shaderc.Shaderc.shaderc_result_release;
import static org.lwjgl.util.shaderc.Shaderc.shaderc_target_env_vulkan;

/**
 * Compiles generated GLSL compute shaders to SPIR-V at run time with shaderc, after the same
 * prelude as the precompiled shaders ({@code common.glsl} and {@code math.glsl}).
 */
final class ShaderCompiler {

    private final long compiler;
    private final long options;
    /** Without optimization, as the precompiled shaders are built; for diagnosing drivers. */
    private final long plainOptions;
    private final String prelude;

    /** Loads shaderc; throws (a {@link LinkageError} or runtime exception) where it is not available. */
    ShaderCompiler() {
        prelude = resource("common.glsl") + "\n" + resource("math.glsl") + "\n";
        compiler = shaderc_compiler_initialize();
        if (compiler == 0) {
            throw new IllegalStateException("shaderc could not be initialized");
        }
        options = shaderc_compile_options_initialize();
        shaderc_compile_options_set_target_env(options, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
        shaderc_compile_options_set_optimization_level(options, shaderc_optimization_level_performance);
        plainOptions = shaderc_compile_options_initialize();
        shaderc_compile_options_set_target_env(plainOptions, shaderc_target_env_vulkan,
                shaderc_env_version_vulkan_1_2);
    }

    private static String resource(String name) {
        try (InputStream in = ShaderCompiler.class.getResourceAsStream("shaders/" + name)) {
            if (in == null) {
                throw new IllegalStateException("missing shader source " + name);
            }
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** The full source of a generated shader for values of {@code type}. */
    String source(String body, DType type) {
        String define = switch (type) {
            case FLOAT32 -> "TYPE_F32";
            case FLOAT64 -> "TYPE_F64";
            case FLOAT16 -> "TYPE_F16";
            case BFLOAT16 -> "TYPE_BF16";
        };
        return "#version 460\n#define " + define + "\n" + prelude + body;
    }

    /** Compiles a full source; throws with the compiler's message if it does not compile. */
    byte[] compile(String source) {
        return compile(source, true);
    }

    /** As {@link #compile(String)}, optimized for speed or not optimized at all. */
    synchronized byte[] compile(String source, boolean optimize) {
        long result = shaderc_compile_into_spv(compiler, source, shaderc_compute_shader, "fused.comp", "main",
                optimize ? options : plainOptions);
        try {
            if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) {
                throw new IllegalStateException("a generated shader did not compile: "
                        + shaderc_result_get_error_message(result) + "\n" + source);
            }
            ByteBuffer b = shaderc_result_get_bytes(result);
            byte[] code = new byte[b.remaining()];
            b.get(code);
            return code;
        } finally {
            shaderc_result_release(result);
        }
    }
}
