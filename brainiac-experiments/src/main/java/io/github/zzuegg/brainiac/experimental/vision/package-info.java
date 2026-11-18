/**
 * Computer vision beyond classification: object detection (RetinaNet-style anchors with a
 * feature pyramid and focal loss, CenterNet heatmaps, DETR set prediction with denoising
 * queries) and its metrics, instance segmentation by discriminative embeddings, atrous
 * spatial pyramid pooling, classical and learned optical flow (Horn-Schunck, Lucas-Kanade,
 * RAFT-style correlation volumes and recurrent refinement), stereo cost volumes,
 * keypoint heads, multi-object tracking (SORT, ByteTrack) and its metrics, video prediction
 * (ConvLSTM, SimVP), neural scene fields (NeRF volume rendering, multiresolution hash
 * encoding, SIREN and signed distance fields), point-cloud networks (PointNet, EdgeConv),
 * group-equivariant convolutions and spatial transformers. Layers implement the core
 * {@code Layer} (or {@code MultiInputLayer}) and come with a {@code LayerSpec} where they fit
 * a model builder; losses implement the core {@code Loss}.
 */
package io.github.zzuegg.brainiac.experimental.vision;
