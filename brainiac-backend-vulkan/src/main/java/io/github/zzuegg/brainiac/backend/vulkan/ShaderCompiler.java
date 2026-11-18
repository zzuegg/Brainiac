package io.github.zzuegg.brainiac.backend.vulkan;

import io.github.zzuegg.brainiac.tensor.DType;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

import org.lwjgl.util.shaderc.Shaderc;

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
            default -> throw new IllegalArgumentException("Vulkan shaders hold floating point values, not " + type);
        };
        return "#version 460\n#define " + define + "\n" + prelude + body;
    }

    /**
     * Where compiled shaders are kept between runs ({@code brainiac.vulkan.cache}, default
     * {@code ~/.cache/brainiac/vulkan}; {@code none} keeps nothing), named by a hash of their source.
     */
    private static final Path CACHE = cacheDirectory();

    private static Path cacheDirectory() {
        String dir = System.getProperty("brainiac.vulkan.cache");
        if ("none".equals(dir)) {
            return null;
        }
        try {
            return dir != null ? Path.of(dir) : Path.of(System.getProperty("user.home"), ".cache", "brainiac", "vulkan");
        } catch (RuntimeException e) {
            return null;
        }
    }

    /**
     * Compiles a full source, or reads it from the cache of earlier runs; throws with the
     * compiler's message if it does not compile.
     */
    byte[] compile(String source) {
        Path file = null;
        if (CACHE != null) {
            try {
                byte[] hash = MessageDigest.getInstance("SHA-256").digest(
                        (Shaderc.class.getPackage().getImplementationVersion() + "\n" + source)
                                .getBytes(StandardCharsets.UTF_8));
                file = CACHE.resolve(HexFormat.of().formatHex(hash, 0, 16) + ".spv");
                if (Files.isRegularFile(file)) {
                    byte[] code = Files.readAllBytes(file);
                    if (code.length >= 20 && code.length % 4 == 0) {
                        return code;
                    }
                }
            } catch (IOException | NoSuchAlgorithmException | RuntimeException e) {
                file = null;   // compiled without the cache
            }
        }
        byte[] code = compile(source, true);
        if (file != null) {
            try {
                Files.createDirectories(file.getParent());
                Path tmp = Files.createTempFile(file.getParent(), "shader", ".tmp");
                Files.write(tmp, code);
                Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException | RuntimeException e) {
                // not cached; compiled again next run
            }
        }
        return code;
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
