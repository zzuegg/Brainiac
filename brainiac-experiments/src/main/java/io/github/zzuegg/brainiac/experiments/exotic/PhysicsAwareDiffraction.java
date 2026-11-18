package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.autograd.Complex;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.DiffractiveLayer;
import io.github.zzuegg.brainiac.experimental.train.PhysicsAwareTraining;
import io.github.zzuegg.brainiac.experiments.generative.ToyImages;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Physics-aware training of the diffractive shape classifier ({@link DiffractiveShapeClassifier}'s
 * optics: two phase masks, free-space propagation, four detector squares) when the fabricated
 * device differs from the simulation: every mask pixel carries a fixed fabrication phase error,
 * the plates sit at a slightly wrong distance, and the detectors saturate ({@code I^0.6})
 * instead of measuring intensity linearly, each with its own uncalibrated gain. The "physical"
 * device is that perturbed simulation,
 * run as a black box without gradients; the digital twin is the ideal simulation.
 *
 * <ul>
 *   <li>{@link Mode#IN_SILICO}: train the twin, then fabricate it (the sim-to-real gap in full);</li>
 *   <li>{@link Mode#NOISE_AWARE}: train the twin with random phase noise injected every step, as
 *       {@link AnalogNoiseAwareTraining} does for analog weights, then fabricate it;</li>
 *   <li>{@link Mode#PHYSICS_AWARE}: {@link PhysicsAwareTraining}, forward through the device,
 *       backward through the twin.</li>
 * </ul>
 * The score is the accuracy of the fabricated device on every toy image.
 */
public final class PhysicsAwareDiffraction {

    /** How the masks are trained. */
    public enum Mode {
        /** simulation only */
        IN_SILICO,
        /** simulation with injected phase noise */
        NOISE_AWARE,
        /** forward on the device, backward through the twin */
        PHYSICS_AWARE
    }

    private static final int N = 32;
    private static final int UPSAMPLE = 2;
    private static final double WAVELENGTH = 1;
    private static final double DISTANCE = 12;
    private static final float TEMPERATURE = 10;
    /** The unequal, uncalibrated gains of the four fabricated detectors. */
    private static final Tensor DETECTOR_GAIN = Tensor.of(Shape.of(4), new float[] {1.6f, 0.6f, 1.2f, 0.7f});

    private final List<DiffractiveLayer> layers = new ArrayList<>();
    private final Tensor[] fabricationError;
    private final double realDistance;
    private final double detectorExponent;
    private final Tensor detectors;

    private PhysicsAwareDiffraction(int depth, double phaseError, double distanceError, double detectorExponent,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed + 1000);
        fabricationError = new Tensor[depth];
        for (int i = 0; i < depth; i++) {
            layers.add(DiffractiveLayer.of(N, WAVELENGTH, DISTANCE, seed + i));
            float[] e = new float[N * N];
            for (int k = 0; k < e.length; k++) {
                e[k] = (float) (r.nextGaussian() * phaseError);
            }
            fabricationError[i] = Tensor.of(Shape.of(N, N, 1), e);
        }
        this.realDistance = DISTANCE * (1 + distanceError);
        this.detectorExponent = detectorExponent;
        this.detectors = detectors();
    }

    /**
     * A design and the device it will be fabricated as.
     *
     * @param depth phase masks
     * @param phaseError the standard deviation of the fixed per-pixel fabrication phase error, radians
     * @param distanceError the relative error of the plate spacing
     * @param detectorExponent the detector response {@code I^p}
     * @param seed the seed of the initial phases and of the fabrication errors
     * @return the experiment
     */
    public static PhysicsAwareDiffraction of(int depth, double phaseError, double distanceError,
            double detectorExponent, long seed) {
        return new PhysicsAwareDiffraction(depth, phaseError, distanceError, detectorExponent, seed);
    }

    private static Tensor detectors() {
        float[] m = new float[N * N * 4];
        int c = N / 2;
        int[][] corners = {{c - 7, c - 7}, {c - 7, c + 3}, {c + 3, c - 7}, {c + 3, c + 3}};
        for (int k = 0; k < 4; k++) {
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    m[((corners[k][0] + y) * N + corners[k][1] + x) * 4 + k] = 1;
                }
            }
        }
        return Tensor.of(Shape.of(N * N, 4), m);
    }

    private static Tensor encode(List<float[]> images) {
        float[] f = new float[images.size() * N * N * 2];
        int offset = (N - ToyImages.SIDE * UPSAMPLE) / 2;
        for (int b = 0; b < images.size(); b++) {
            float[] img = images.get(b);
            for (int y = 0; y < ToyImages.SIDE * UPSAMPLE; y++) {
                for (int x = 0; x < ToyImages.SIDE * UPSAMPLE; x++) {
                    if (img[(y / UPSAMPLE) * ToyImages.SIDE + x / UPSAMPLE] > 0) {
                        f[((b * N + offset + y) * N + offset + x) * 2] = 1;
                    }
                }
            }
        }
        return Tensor.of(Shape.of(images.size(), N, N, 2), f);
    }

    /** The digital twin: detector shares of the ideal simulation. */
    private Var twin(Var field, Tensor[] noise) {
        Var e = field;
        for (int i = 0; i < layers.size(); i++) {
            DiffractiveLayer l = layers.get(i);
            if (noise == null) {
                e = l.forward(e);
            } else {
                Var phase = l.phase().add(noise[i]);
                Var modulated = Complex.mul(e, Var.concat(2, phase.cos(), phase.sin()));
                e = DiffractiveLayer.propagate(modulated, WAVELENGTH, DISTANCE);
            }
        }
        int b = field.shape().dim(0);
        Var signal = DiffractiveLayer.intensity(e).reshape(b, N * N).matmul(Var.of(detectors));
        return signal.div(signal.sum(1, true).add(1e-6f));
    }

    /** The fabricated device: perturbed phases and spacing, saturating detectors; no gradient. */
    private Tensor device(Tensor field) {
        Var e = Var.of(field);
        for (int i = 0; i < layers.size(); i++) {
            Var phase = Var.of(layers.get(i).phase().value().add(fabricationError[i]));
            Var modulated = Complex.mul(e, Var.concat(2, phase.cos(), phase.sin()));
            e = DiffractiveLayer.propagate(modulated, WAVELENGTH, realDistance);
        }
        int b = field.shape().dim(0);
        Var light = DiffractiveLayer.intensity(e).add(1e-12f).log().mul((float) detectorExponent).exp();
        Var signal = light.reshape(b, N * N).matmul(Var.of(detectors)).mul(DETECTOR_GAIN);
        return signal.div(signal.sum(1, true).add(1e-6f)).value();
    }

    /**
     * Trains the masks.
     *
     * @param mode the training mode
     * @param steps Adam steps on 16 random shapes
     * @param noise the injected phase noise of {@link Mode#NOISE_AWARE}, radians
     * @param seed the data seed
     */
    public void train(Mode mode, int steps, double noise, long seed) {
        Optimizer opt = Optimizer.adam(0.05f);
        SplittableRandom r = new SplittableRandom(seed);
        List<Var> params = new ArrayList<>();
        for (DiffractiveLayer l : layers) {
            params.addAll(l.parameters());
        }
        for (int s = 0; s < steps; s++) {
            List<float[]> images = new ArrayList<>();
            float[] labels = new float[16];
            for (int i = 0; i < 16; i++) {
                labels[i] = i % ToyImages.CLASSES;
                images.add(ToyImages.sample(i % ToyImages.CLASSES, r));
            }
            Var field = Var.of(encode(images));
            Var out = switch (mode) {
                case IN_SILICO -> twin(field, null);
                case NOISE_AWARE -> twin(field, randomNoise(noise, r));
                case PHYSICS_AWARE -> PhysicsAwareTraining.apply(field, this::device, f -> twin(f, null));
            };
            Var loss = Functional.crossEntropy(out.mul(TEMPERATURE), Tensor.of(Shape.of(16), labels));
            loss.backward();
            opt.step(params);
        }
    }

    private Tensor[] randomNoise(double sd, SplittableRandom r) {
        Tensor[] out = new Tensor[layers.size()];
        for (int i = 0; i < out.length; i++) {
            float[] e = new float[N * N];
            for (int k = 0; k < e.length; k++) {
                e[k] = (float) (r.nextGaussian() * sd);
            }
            out[i] = Tensor.of(Shape.of(N, N, 1), e);
        }
        return out;
    }

    /**
     * Accuracy on every toy image.
     *
     * @param onDevice the fabricated device if true, the twin otherwise
     * @return the accuracy
     */
    public double accuracy(boolean onDevice) {
        List<float[]> images = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        for (int c = 0; c < ToyImages.CLASSES; c++) {
            for (float[] img : ToyImages.all(c)) {
                images.add(img);
                labels.add(c);
            }
        }
        Tensor field = encode(images);
        Tensor out = onDevice ? device(field) : twin(Var.of(field), null).value();
        float[] pred = out.argmax(1).toArray();
        int right = 0;
        for (int i = 0; i < pred.length; i++) {
            right += Math.round(pred[i]) == labels.get(i) ? 1 : 0;
        }
        return (double) right / pred.length;
    }
}
