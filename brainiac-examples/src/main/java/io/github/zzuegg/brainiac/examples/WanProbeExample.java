package io.github.zzuegg.brainiac.examples;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.diffusion.Wan;
import io.github.zzuegg.brainiac.io.SafeTensors;
import io.github.zzuegg.brainiac.nn.diffusion.WanTransformer;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Runs one forward pass of a Wan transformer on stored inputs and writes the patch embedding,
 * every block's output and the result, to compare against diffusers block by block (see
 * {@code brainiac-diffusion/src/test/python/wan_probe.py}).
 *
 * <p>{@code wanprobe <model folder> <inputs.safetensors> <out.safetensors> [--device name]
 * [--dtype f32|bf16]}; the inputs are {@code latents [1, c, f, h, w]}, {@code timestep [1]} and
 * {@code context [1, 512, text]}, or several such cases with names prefixed {@code c0.},
 * {@code c1.}, ..., whose outputs keep the prefix.
 */
public final class WanProbeExample {

    private WanProbeExample() {
    }

    /**
     * Runs the probe.
     *
     * @param args the arguments
     */
    public static void main(String[] args) {
        try {
            run(args);
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
    }

    private static void run(String[] args) throws java.io.IOException {
        if (args.length < 3) {
            System.out.println("Usage: wanprobe <model folder> <inputs.safetensors> <out.safetensors>"
                    + " [--device name] [--dtype f32|bf16]");
            return;
        }
        String deviceName = "best";
        DType dtype = DType.BFLOAT16;
        for (int i = 3; i < args.length; i++) {
            switch (args[i]) {
                case "--device" -> deviceName = args[++i];
                case "--dtype" -> dtype = "f32".equals(args[++i]) ? DType.FLOAT32 : DType.BFLOAT16;
                default -> throw new IllegalArgumentException("unknown option " + args[i]);
            }
        }
        Device device = VideoExample.device(deviceName);
        Map<String, Tensor> out = new LinkedHashMap<>();
        try (SafeTensors in = SafeTensors.open(Path.of(args[1]));
                Wan wan = Wan.load(Path.of(args[0]), device, dtype, new Wan.Options(0, 0, 512))) {
            WanTransformer t = wan.transformer();
            System.out.println("Loaded " + wan + " on " + device);
            // one case, or several named c0., c1., ...
            List<String> cases = new ArrayList<>();
            for (String n : in.names()) {
                if (n.endsWith("latents")) {
                    cases.add(n.substring(0, n.length() - "latents".length()));
                }
            }
            for (String c : cases) {
                Tensor latents = in.tensor(c + "latents", device, dtype);
                Tensor timestep = in.tensor(c + "timestep", device, DType.FLOAT32);
                Tensor context = in.tensor(c + "context", device, dtype);
                WanTransformer.blockProbe = (x, i) -> out.put(c + (i < 0 ? "embedded" : "block." + i),
                        x.to(Device.cpu()).to(DType.FLOAT32));
                Tensor result = Var.noGrad(() -> t.forward(Var.of(latents), timestep, Var.of(context), null, false)
                        .value().to(Device.cpu()).to(DType.FLOAT32));
                WanTransformer.blockProbe = null;
                out.put(c + "out", result);
                System.out.printf("%sout: %s, mean |x| %.4g%n", c, result.shape(), mean(result));
                latents.release();
                context.release();
            }
        }
        SafeTensors.write(Path.of(args[2]), out);
        System.out.println("Wrote " + out.size() + " tensors to " + args[2]);
    }

    private static double mean(Tensor t) {
        float[] v = t.toArray();
        double s = 0;
        for (float f : v) {
            s += Math.abs(f);
        }
        return s / v.length;
    }
}
