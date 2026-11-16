package io.github.zzuegg.brainiac.backend.opencl;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.Optional;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/**
 * A different activation for every neuron, as evolved designs have, must not cost one kernel
 * launch per activation: the masked sum it is computed as fuses into one kernel, the same
 * number of launches as a single shared activation.
 */
class PerUnitLaunchTest {

    private static long launches(OpenClBackend backend) throws ReflectiveOperationException {
        var field = OpenClBackend.class.getDeclaredField("launchIndex");
        field.setAccessible(true);
        return (long) field.get(backend);
    }

    /** Launches for one forward and backward pass. */
    private static long passLaunches(OpenClBackend backend, Device gpu, Activation a, float[][] out)
            throws ReflectiveOperationException {
        Var x = Var.param(Tensor.uniform(Shape.of(32, 64), -2, 2, new SplittableRandom(1)).to(gpu));
        Tensor.evaluate(List.of(x.value()));
        long before = launches(backend);
        Var y = a.forward(x);
        Tensor.evaluate(List.of(y.value()));
        y.sum().backward();
        Tensor.evaluate(List.of(x.grad()));
        long n = launches(backend) - before;
        out[0] = y.value().toArray();
        out[1] = x.grad().toArray();
        return n;
    }

    @Test
    void perNeuronActivationsRunAsOneKernel() throws Exception {
        Optional<Device> found = Device.available().stream().filter(d -> d.name().startsWith("opencl")).findFirst();
        assumeTrue(found.isPresent(), "no OpenCL device");
        Device gpu = found.get();
        var getBackend = Device.class.getDeclaredMethod("backend");
        getBackend.setAccessible(true);
        OpenClBackend backend = (OpenClBackend) getBackend.invoke(gpu);

        Activation[] palette = {Activation.RELU, Activation.TANH, Activation.SIGMOID, Activation.GELU};
        Activation[] units = new Activation[64];
        for (int i = 0; i < units.length; i++) {
            units[i] = palette[(i * 7) % palette.length];
        }
        Activation perUnit = Activation.perUnit(units);
        float[][] shared = new float[2][];
        float[][] mixed = new float[2][];
        // generated kernels compile in the background and chains run unfused meanwhile, so
        // compare once both have settled
        long one = -1;
        long many = -2;
        long deadline = System.nanoTime() + 60_000_000_000L;
        while (one != many && System.nanoTime() < deadline) {
            one = passLaunches(backend, gpu, Activation.TANH, shared);
            many = passLaunches(backend, gpu, perUnit, mixed);
            if (one != many) {
                Thread.sleep(200);
            }
        }
        assertEquals(one, many, "launches for a forward and backward pass");

        // and it computes what the CPU does
        Var x = Var.param(Tensor.uniform(Shape.of(32, 64), -2, 2, new SplittableRandom(1)));
        Var y = perUnit.forward(x);
        y.sum().backward();
        assertArrayEquals(y.value().toArray(), mixed[0], 1e-5f);
        assertArrayEquals(x.grad().toArray(), mixed[1], 1e-4f);
    }
}
