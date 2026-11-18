package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class AtrousSegmentationTest {

    @Test
    void asppSegmentsWithFewerParameters() {
        DensePrediction.SegScore unet = DensePrediction.segment(DensePrediction.Segmenter.UNET,
                DensePrediction.SegLoss.LOVASZ, false, 120, 1);
        DensePrediction.SegScore aspp = DensePrediction.segment(DensePrediction.Segmenter.DEEPLAB,
                DensePrediction.SegLoss.LOVASZ, false, 120, 1);
        System.out.printf("segmentation with the Lovasz loss, mIoU (params, MACs): U-Net %.3f (%d, %d),"
                + " DeepLab-lite %.3f (%d, %d)%n", unet.miou(), unet.parameters(), unet.macs(), aspp.miou(),
                aspp.parameters(), aspp.macs());
        assertTrue(aspp.parameters() < unet.parameters(), "ASPP is the smaller network");
        assertTrue(aspp.miou() > 0.4, "ASPP segments the shapes (U-Net is still ahead at this budget)");
    }
}
