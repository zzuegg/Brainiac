package io.github.zzuegg.brainiac.backend.vulkan;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;

import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

/**
 * Compiles small shaders one at a time and creates their pipelines, printing each before it is
 * tried, to find the code a driver's shader compiler fails on (a crash ends the JVM, so the last
 * line printed names the culprit). Runs only with {@code -Pbrainiac.vulkan.diagnose=<case>} (or
 * {@code all}); cases are the keys of {@link #cases()}, each tried in float64 unoptimized and
 * optimized.
 */
class DriverDiagnosisTest {

    private static Map<String, String> cases() {
        Map<String, String> c = new LinkedHashMap<>();
        c.put("unary", "unary_op(p.op, v)");
        c.put("activation", "activation_op(p.op, ACC(0.1), v)");
        c.put("slope", "activation_slope(p.op, ACC(0.1), v)");
        c.put("binary", "binary_op(p.op, v, v)");
        c.put("neg", "-v");
        c.put("abs", "abs(v)");
        c.put("sqrt", "sqrt(v)");
        c.put("exp", "exp_(v)");
        c.put("log", "log_(v)");
        c.put("tanh", "tanh_(v)");
        c.put("sigmoid", "sigmoid_(v)");
        c.put("erf", "erf_(v)");
        c.put("expm1", "expm1_(v)");
        c.put("log1p", "log1p_(v)");
        c.put("softplus", "softplus_(v)");
        c.put("relu", "v > K(0.0) ? v : K(0.0)");
        c.put("sign", "v > K(0.0) ? K(1.0) : v < K(0.0) ? K(-1.0) : v");
        c.put("clamp", "clamp(v, K(-20.0), K(20.0))");
        c.put("divide", "K(1.0) / (K(1.0) + v)");
        c.put("floor", "floor(v)");
        c.put("mod", "mod(v, K(2.0))");
        c.put("exp+tanh", "exp_(v) + tanh_(v)");
        c.put("exp+log", "exp_(v) + log_(K(1.0) + v * v)");
        c.put("exp+exp", "exp_(v) + exp_(v * K(0.5))");
        return c;
    }

    @Test
    void shadersCompile() {
        String which = System.getProperty("brainiac.vulkan.diagnose");
        assumeTrue(which != null && !which.isEmpty(), "set -Pbrainiac.vulkan.diagnose=<case> to run");
        assumeTrue(Device.available().stream().anyMatch(d -> d.name().startsWith("vulkan")), "no Vulkan device");
        VulkanBackend vk = VulkanBackend.loaded().orElseThrow();
        System.out.println("diagnosing on " + vk.status());
        for (Map.Entry<String, String> e : cases().entrySet()) {
            if (!which.equals("all") && !which.equals(e.getKey())) {
                continue;
            }
            String body = "layout(local_size_x = GROUP) in;\n"
                    + "layout(push_constant) uniform P { In x; Buf y; uint n; uint op; } p;\n"
                    + "void main() {\n    uint i = gid();\n    if (i >= p.n) return;\n"
                    + "    ACC v = LOAD(p.x, i);\n    STORE(p.y, i, " + e.getValue() + ");\n}\n";
            for (boolean optimize : new boolean[] {false, true}) {
                String name = e.getKey() + (optimize ? " optimized" : " plain") + " float64";
                System.out.println("compiling " + name + " ...");
                System.out.flush();
                vk.compileForDiagnosis(body, DType.FLOAT64, optimize);
                System.out.println("compiled " + name);
                System.out.flush();
            }
        }
    }
}
