package io.github.zzuegg.brainiac.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Conv;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.PadMode;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;
import java.util.function.Supplier;
import java.util.random.RandomGenerator;

/** Building blocks shared by the diffusion layers. */
final class Parts {

    private Parts() {
    }

    /** A square 2-d convolution with {@code pad} zeros before and {@code padEnd} after each side. */
    static Conv conv(int in, int out, int kernel, int stride, int pad, int padEnd, RandomGenerator random) {
        Conv.Settings s = new Conv.Settings(false, new int[] {stride, stride}, new int[] {1, 1},
                new int[] {pad, pad}, new int[] {padEnd, padEnd}, new int[] {0, 0}, PadMode.CONSTANT, 0, 1);
        int fanIn = in * kernel * kernel;
        Tensor w = Init.HE.create(Shape.of(out, in, kernel, kernel), fanIn, out * kernel * kernel, random);
        return new Conv(s, Activation.IDENTITY, w, Tensor.zeros(Shape.of(out)));
    }

    /** A 3x3 convolution that keeps the size. */
    static Conv conv3(int in, int out, RandomGenerator random) {
        return conv(in, out, 3, 1, 1, 1, random);
    }

    /** A 1x1 convolution. */
    static Conv conv1(int in, int out, RandomGenerator random) {
        return conv(in, out, 1, 1, 0, 0, random);
    }

    static Dense dense(int in, int out, Activation activation, RandomGenerator random) {
        return new Dense(in, out, activation, activation == Activation.IDENTITY ? Init.XAVIER : Init.HE, random);
    }

    static Var silu(Var x) {
        return x.activation(ActivationOp.SILU, 0);
    }

    /** {@code [batch, channels, h, w]} as tokens {@code [batch, h * w, channels]}. */
    static Var tokens(Var x) {
        Shape s = x.shape();
        return x.reshape(s.dim(0), s.dim(1), s.dim(2) * s.dim(3)).transpose();
    }

    /** Tokens {@code [batch, h * w, channels]} back as {@code [batch, channels, h, w]}. */
    static Var image(Var x, int h, int w) {
        Shape s = x.shape();
        return x.transpose().reshape(s.dim(0), s.dim(2), h, w);
    }

    /** Nearest-neighbour upsampling by 2 of {@code [batch, channels, h, w]}. */
    static Var upsample2x(Var x) {
        Shape s = x.shape();
        int b = s.dim(0);
        int c = s.dim(1);
        int h = s.dim(2);
        int w = s.dim(3);
        return x.reshape(b, c, h, 1, w, 1).expand(Shape.of(b, c, h, 2, w, 2)).reshape(b, c, 2 * h, 2 * w);
    }

    /**
     * Runs one step of a forward pass. Without gradients ({@link Var#noGrad}) it then frees
     * every tensor the step made except its result, so that memory stays bounded over a deep
     * network on a GPU; with gradients the values stay for the backward pass.
     */
    static Var step(Supplier<Var> body) {
        if (Var.gradEnabled()) {
            return body.get();
        }
        Var[] out = new Var[1];
        List<Tensor> made = Tensor.created(() -> out[0] = body.get());
        Tensor.evaluate(List.of(out[0].value()));
        Set<Tensor> keep = Collections.newSetFromMap(new IdentityHashMap<>());
        keep.add(out[0].value());
        Tensor.releaseCreated(made, keep);
        return out[0];
    }

    /**
     * Frees a value a forward pass no longer needs, when nothing records gradients; values
     * in {@code inputs} (the caller's) stay.
     */
    static void free(Var v, Var... inputs) {
        if (v == null || Var.gradEnabled()) {
            return;
        }
        for (Var in : inputs) {
            if (in != null && in.value() == v.value()) {
                return;
            }
        }
        v.value().release();
    }
}
