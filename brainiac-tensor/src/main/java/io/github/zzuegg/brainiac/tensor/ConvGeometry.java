package io.github.zzuegg.brainiac.tensor;

import java.util.Arrays;

/**
 * Where a sliding window visits an image, for any number of spatial dimensions: the
 * shared description behind convolution, transposed convolution and pooling.
 *
 * <p>An image has the layout {@code [batch, channels, d1, d2, ...]}. The window has
 * {@code kernel} cells per dimension, spaced {@code dilation} apart, and moves
 * {@code stride} cells between positions. Position {@code p} along a dimension covers
 * image cells {@code p * stride - padBefore + i * dilation}; cells outside the image are
 * padding. {@code positions} is the number of window positions per dimension, so
 * padding after the image is implied by it.
 *
 * <p>{@link Tensor#im2col(ConvGeometry, float)} unrolls every window into a row, and
 * {@link Tensor#col2im(ConvGeometry)} adds rows back into an image.
 *
 * @param batch the number of images
 * @param channels the channels per image
 * @param image the image size per spatial dimension
 * @param kernel the window size per spatial dimension
 * @param stride the step between positions per spatial dimension
 * @param padBefore the padding before the image per spatial dimension
 * @param dilation the spacing of window cells per spatial dimension
 * @param positions the number of window positions per spatial dimension
 */
public record ConvGeometry(int batch, int channels, int[] image, int[] kernel, int[] stride, int[] padBefore,
        int[] dilation, int[] positions) {

    /**
     * Validates and copies the arrays.
     *
     * @param batch the number of images
     * @param channels the channels per image
     * @param image the image size per spatial dimension
     * @param kernel the window size per spatial dimension
     * @param stride the step between positions per spatial dimension
     * @param padBefore the padding before the image per spatial dimension
     * @param dilation the spacing of window cells per spatial dimension
     * @param positions the number of window positions per spatial dimension
     */
    public ConvGeometry {
        int r = image.length;
        if (r == 0 || kernel.length != r || stride.length != r || padBefore.length != r || dilation.length != r
                || positions.length != r) {
            throw new IllegalArgumentException("every geometry array needs one entry per spatial dimension");
        }
        if (batch < 1 || channels < 1) {
            throw new IllegalArgumentException("batch and channels must be at least 1");
        }
        for (int i = 0; i < r; i++) {
            if (image[i] < 1 || kernel[i] < 1 || stride[i] < 1 || dilation[i] < 1 || positions[i] < 1
                    || padBefore[i] < 0) {
                throw new IllegalArgumentException("bad geometry in dimension " + i + ": image " + image[i]
                        + ", kernel " + kernel[i] + ", stride " + stride[i] + ", dilation " + dilation[i]
                        + ", positions " + positions[i] + ", padding " + padBefore[i]);
            }
        }
        image = image.clone();
        kernel = kernel.clone();
        stride = stride.clone();
        padBefore = padBefore.clone();
        dilation = dilation.clone();
        positions = positions.clone();
    }

    /**
     * Creates the geometry of a window sliding over an image, with the number of
     * positions worked out from the padding on both sides.
     *
     * @param batch the number of images
     * @param channels the channels per image
     * @param image the image size per spatial dimension
     * @param kernel the window size per spatial dimension
     * @param stride the step between positions
     * @param padBefore padding before the image
     * @param padAfter padding after the image
     * @param dilation the spacing of window cells
     * @param ceil whether a last, partial window position counts (pooling's ceil mode)
     * @return the geometry
     * @throws IllegalArgumentException if the window does not fit even once
     */
    public static ConvGeometry sliding(int batch, int channels, int[] image, int[] kernel, int[] stride,
            int[] padBefore, int[] padAfter, int[] dilation, boolean ceil) {
        int[] positions = new int[image.length];
        for (int i = 0; i < image.length; i++) {
            int span = dilation[i] * (kernel[i] - 1) + 1;
            int room = image[i] + padBefore[i] + padAfter[i] - span;
            if (room < 0) {
                throw new IllegalArgumentException("window of " + span + " cells does not fit dimension " + i
                        + " of size " + image[i] + " with padding " + padBefore[i] + "+" + padAfter[i]);
            }
            int p = (ceil ? Math.ceilDiv(room, stride[i]) : room / stride[i]) + 1;
            // a ceil-mode window must still start inside the image or its leading padding
            if (ceil && (p - 1) * stride[i] >= image[i] + padBefore[i]) {
                p--;
            }
            positions[i] = p;
        }
        return new ConvGeometry(batch, channels, image, kernel, stride, padBefore, dilation, positions);
    }

    /**
     * Returns the number of spatial dimensions.
     *
     * @return the spatial rank
     */
    public int spatialRank() {
        return image.length;
    }

    /**
     * Returns the number of cells in one window.
     *
     * @return the product of the kernel sizes
     */
    public int kernelSize() {
        return product(kernel);
    }

    /**
     * Returns the number of window positions per image.
     *
     * @return the product of the positions
     */
    public int positionCount() {
        return product(positions);
    }

    /**
     * Returns the number of cells per image and channel.
     *
     * @return the product of the image sizes
     */
    public int imageSize() {
        return product(image);
    }

    /**
     * Returns the shape of the image tensor.
     *
     * @return {@code [batch, channels, image...]}
     */
    public Shape imageShape() {
        int[] dims = new int[image.length + 2];
        dims[0] = batch;
        dims[1] = channels;
        System.arraycopy(image, 0, dims, 2, image.length);
        return Shape.of(dims);
    }

    /**
     * Returns the shape of the unrolled windows.
     *
     * @return {@code [batch * positions, channels * kernel]}
     */
    public Shape columnShape() {
        return Shape.of(Math.multiplyExact(batch, positionCount()), Math.multiplyExact(channels, kernelSize()));
    }

    /**
     * Returns the same geometry for another batch size and channel count.
     *
     * @param batch the number of images
     * @param channels the channels per image
     * @return the geometry
     */
    public ConvGeometry with(int batch, int channels) {
        return new ConvGeometry(batch, channels, image, kernel, stride, padBefore, dilation, positions);
    }

    private static int product(int[] a) {
        int p = 1;
        for (int x : a) {
            p = Math.multiplyExact(p, x);
        }
        return p;
    }

    @Override
    public int[] image() {
        return image.clone();
    }

    @Override
    public int[] kernel() {
        return kernel.clone();
    }

    @Override
    public int[] stride() {
        return stride.clone();
    }

    @Override
    public int[] padBefore() {
        return padBefore.clone();
    }

    @Override
    public int[] dilation() {
        return dilation.clone();
    }

    @Override
    public int[] positions() {
        return positions.clone();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof ConvGeometry g && batch == g.batch && channels == g.channels
                && Arrays.equals(image, g.image) && Arrays.equals(kernel, g.kernel) && Arrays.equals(stride, g.stride)
                && Arrays.equals(padBefore, g.padBefore) && Arrays.equals(dilation, g.dilation)
                && Arrays.equals(positions, g.positions);
    }

    @Override
    public int hashCode() {
        return 31 * Arrays.hashCode(image) + Arrays.hashCode(kernel) + batch * 7 + channels;
    }

    @Override
    public String toString() {
        return "ConvGeometry[batch=" + batch + ", channels=" + channels + ", image=" + Arrays.toString(image)
                + ", kernel=" + Arrays.toString(kernel) + ", stride=" + Arrays.toString(stride) + ", padBefore="
                + Arrays.toString(padBefore) + ", dilation=" + Arrays.toString(dilation) + ", positions="
                + Arrays.toString(positions) + "]";
    }
}
