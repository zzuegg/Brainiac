package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.FourierFeatures;
import io.github.zzuegg.brainiac.experimental.nn.GaussianSplats2d;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

/**
 * Fitting one image three ways, the 2-D toy versions of NeRF-style scene representations:
 * a coordinate MLP on raw {@code (x, y)}, the same MLP behind Fourier features (which beats
 * the MLP's spectral bias towards blur), and a set of 2-D Gaussian splats rendered by a
 * differentiable rasterizer. Each returns the PSNR it reached, so they can be compared.
 */
public final class NeuralFields {
    private NeuralFields() {
    }

    /**
     * A test picture with sharp edges and fine detail: a checkerboard with a disc on top.
     *
     * @param side the image side
     * @param squares checker squares per side
     * @return {@code side * side} values in [0, 1]
     */
    public static float[] picture(int side, int squares) {
        float[] p = new float[side * side];
        for (int y = 0; y < side; y++) {
            for (int x = 0; x < side; x++) {
                boolean check = ((x * squares / side) + (y * squares / side)) % 2 == 0;
                double r = Math.hypot(x - side * 0.6, y - side * 0.4);
                p[y * side + x] = r < side * 0.22 ? 0.9f : check ? 0.7f : 0.2f;
            }
        }
        return p;
    }

    /**
     * Peak signal-to-noise ratio for signals in [0, 1].
     *
     * @param a one image
     * @param b another
     * @return the PSNR in decibels
     */
    public static double psnr(float[] a, float[] b) {
        double mse = Nets.distance2(a, b) / a.length;
        return 10 * Math.log10(1 / Math.max(mse, 1e-12));
    }

    /**
     * Fits a coordinate network (any layer from {@code [pixels, 2]} to {@code [pixels, 1]}).
     *
     * @param net the network
     * @param picture the target
     * @param side the image side
     * @param steps full-batch Adam steps
     * @param learningRate the learning rate
     * @return the PSNR of the final fit
     */
    public static double fit(Layer net, float[] picture, int side, int steps, float learningRate) {
        return fit(net::forward, net.parameters(), picture, side, steps, learningRate);
    }

    /**
     * Fits a coordinate model.
     *
     * @param net the model, from {@code [pixels, 2]} to {@code [pixels, 1]}
     * @param picture the target
     * @param side the image side
     * @param steps full-batch Adam steps
     * @param learningRate the learning rate
     * @return the PSNR of the final fit
     */
    public static double fit(Model net, float[] picture, int side, int steps, float learningRate) {
        return fit(x -> net.forward(x, true), net.parameters(), picture, side, steps, learningRate);
    }

    private static double fit(java.util.function.UnaryOperator<Var> net, java.util.List<Var> params, float[] picture,
            int side, int steps, float learningRate) {
        Tensor coords = GaussianSplats2d.grid(side, side);
        Tensor target = Tensor.of(Shape.of(side * side, 1), picture);
        Optimizer opt = Optimizer.adam(learningRate);
        for (int s = 0; s < steps; s++) {
            Var loss = net.apply(Var.of(coords)).sub(target).square().mean();
            loss.backward();
            opt.step(params);
        }
        return psnr(net.apply(Var.of(coords)).value().toArray(), picture);
    }

    /**
     * A coordinate MLP, optionally behind random Fourier features.
     *
     * @param fourierFeatures the number of random frequencies, 0 for raw coordinates
     * @param scale their standard deviation (bandwidth)
     * @param seed the seed
     * @return the network
     */
    public static Model coordinateMlp(int fourierFeatures, float scale, long seed) {
        Model.Builder b = Model.sequential().input(2);
        if (fourierFeatures > 0) {
            b.add(FourierFeatures.spec(fourierFeatures, scale, false));
        }
        return b.dense(64, Activation.RELU).dense(64, Activation.RELU).dense(1).seed(seed).build();
    }
}
