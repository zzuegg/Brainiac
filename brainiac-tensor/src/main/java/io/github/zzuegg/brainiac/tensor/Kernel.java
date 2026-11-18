package io.github.zzuegg.brainiac.tensor;

import java.util.Objects;

/**
 * A user-written GPU kernel: a GLSL compute shader body that the Vulkan backend compiles at
 * run time, with a Java version for every other backend (and to test the shader against).
 * One invocation runs per output value unless the shader loops itself; see
 * {@link Tensor#custom}.
 *
 * <p>The shader is compiled once per number format after Brainiac's shared prelude and these
 * declarations, so it only writes {@code main()}:
 *
 * <pre>
 * layout(local_size_x = GROUP) in;   // GROUP = 256
 * layout(push_constant) uniform P { Buf y; In x0; In x1; ...; uint n; int ints[4]; float floats[8]; } p;
 * </pre>
 *
 * {@code y} is the output and {@code x0, x1, ...} the inputs, all in the output's format;
 * {@code LOAD(b, i)} reads value {@code i} as {@code ACC} (float, or double for float64) and
 * {@code STORE(b, i, v)} writes it; {@code gid()} is the invocation's index and {@code p.n}
 * the output's size.
 *
 * <pre>{@code
 * Kernel scaledSquare = Kernel.of("scaled_square", 1, """
 *         void main() {
 *             uint i = gid();
 *             if (i >= p.n) return;
 *             ACC v = LOAD(p.x0, i);
 *             STORE(p.y, i, v * v * p.floats[0]);
 *         }""", (in, out, ints, floats) -> {
 *             for (int i = 0; i < out.length; i++) out[i] = in[0][i] * in[0][i] * floats[0];
 *         });
 * Tensor y = Tensor.custom(scaledSquare, x.shape(), x.dtype(), new int[0], new float[] {3}, x);
 * }</pre>
 */
public final class Kernel {

    /** The most inputs a kernel takes. */
    public static final int MAX_INPUTS = 8;
    /** The most integer parameters ({@code p.ints}). */
    public static final int MAX_INTS = 4;
    /** The most float parameters ({@code p.floats}). */
    public static final int MAX_FLOATS = 8;

    /** The kernel in Java, for backends that do not compile GLSL. */
    @FunctionalInterface
    public interface Reference {

        /**
         * Computes the output.
         *
         * @param inputs the inputs' values, row-major
         * @param out the output's values, to fill
         * @param ints the integer parameters
         * @param floats the float parameters
         */
        void run(double[][] inputs, double[] out, int[] ints, float[] floats);
    }

    private final String name;
    private final int inputs;
    private final String glsl;
    private final Reference reference;

    private Kernel(String name, int inputs, String glsl, Reference reference) {
        this.name = name;
        this.inputs = inputs;
        this.glsl = glsl;
        this.reference = reference;
    }

    /**
     * Defines a kernel.
     *
     * @param name a name for profiles and error messages
     * @param inputs how many input tensors it reads, at most {@link #MAX_INPUTS}
     * @param glsl the shader's {@code main()} (and any helper functions), or {@code null} for a
     *     kernel that only runs in Java
     * @param reference the same computation in Java
     * @return the kernel
     */
    public static Kernel of(String name, int inputs, String glsl, Reference reference) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(reference, "reference");
        if (inputs < 0 || inputs > MAX_INPUTS) {
            throw new IllegalArgumentException("a kernel takes 0 to " + MAX_INPUTS + " inputs, got " + inputs);
        }
        return new Kernel(name, inputs, glsl, reference);
    }

    /** @return the kernel's name */
    public String name() {
        return name;
    }

    /** @return how many inputs it reads */
    public int inputs() {
        return inputs;
    }

    /** @return the shader body, or {@code null} for a Java-only kernel */
    public String glsl() {
        return glsl;
    }

    /** @return the Java version */
    public Reference reference() {
        return reference;
    }

    /**
     * The declarations compiled before the shader body.
     *
     * @return the GLSL text
     */
    public String header() {
        StringBuilder b = new StringBuilder("layout(local_size_x = GROUP) in;\n");
        b.append("layout(push_constant) uniform P { Buf y; ");
        for (int i = 0; i < inputs; i++) {
            b.append("In x").append(i).append("; ");
        }
        b.append("uint n; int ints[").append(MAX_INTS).append("]; float floats[").append(MAX_FLOATS).append("]; } p;\n");
        return b.toString();
    }

    @Override
    public String toString() {
        return "Kernel[" + name + ", " + inputs + " inputs" + (glsl == null ? ", Java only" : "") + "]";
    }
}
