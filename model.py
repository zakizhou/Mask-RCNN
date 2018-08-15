from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


# ---------------------------------------------
# backbone network


class IdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, stage, block, data_format="channels_last"):
        super(IdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1), name=conv_name_base + '2a', data_format=data_format)
        self.bn2a = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same', data_format=data_format,
                                             name=conv_name_base + '2b')
        self.bn2b = tf.keras.layers.BatchNormalization( axis=bn_axis, name=bn_name_base + '2b')

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c', data_format=data_format)
        self.bn2c = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')

    def call(self, input_tensor, training=False, mask=None):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


class ConvBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, stage, block, data_format="channels_last", strides=(2, 2)):
        super(ConvBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a',
                                             data_format=data_format)
        self.bn2a = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b',
                                             data_format=data_format)
        self.bn2b = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c', data_format=data_format)
        self.bn2c = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')

        self.conv_shortcut = tf.keras.layers.Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1',
                                                    data_format=data_format)
        self.bn_shortcut = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1')

    def call(self, input_tensor, training=False, mask=None):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        shortcut = self.conv_shortcut(input_tensor)
        shortcut = self.bn_shortcut(shortcut, training=training)

        x += shortcut
        return tf.nn.relu(x)


# TODO add resnet 100
class ResidualNetwork(tf.keras.Model):
    def __init__(self, data_format="channels_last"):
        super(ResidualNetwork, self).__init__(name='')
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.conv1 = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), data_format=data_format, padding='same',
                                            name='conv1')

        self.bn_conv1 = tf.keras.layers.BatchNormalization(axis=bn_axis, name='bn_conv1')
        self.max_pool = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), data_format=data_format, padding="same")

        self.l2a = ConvBlock(3, [64, 64, 256], stage=2, block='a', strides=(1, 1), data_format=data_format)
        self.l2b = IdentityBlock(3, [64, 64, 256], stage=2, block='b')
        self.l2c = IdentityBlock(3, [64, 64, 256], stage=2, block='c')

        self.l3a = ConvBlock(3, [128, 128, 512], stage=3, block='a')
        self.l3b = IdentityBlock(3, [128, 128, 512], stage=3, block='b')
        self.l3c = IdentityBlock(3, [128, 128, 512], stage=3, block='c')
        self.l3d = IdentityBlock(3, [128, 128, 512], stage=3, block='d')

        self.l4a = ConvBlock(3, [256, 256, 1024], stage=4, block='a')
        self.l4b = IdentityBlock(3, [256, 256, 1024], stage=4, block='b')
        self.l4c = IdentityBlock(3, [256, 256, 1024], stage=4, block='c')
        self.l4d = IdentityBlock(3, [256, 256, 1024], stage=4, block='d')
        self.l4e = IdentityBlock(3, [256, 256, 1024], stage=4, block='e')
        self.l4f = IdentityBlock(3, [256, 256, 1024], stage=4, block='f')

        self.l5a = ConvBlock(3, [512, 512, 2048], stage=5, block='a')
        self.l5b = IdentityBlock(3, [512, 512, 2048], stage=5, block='b')
        self.l5c = IdentityBlock(3, [512, 512, 2048], stage=5, block='c')

    def call(self, input_tensor, training=True, mask=None):
        feature_maps = []
        x = self.conv1(input_tensor)
        x = self.bn_conv1(x, training=training)
        x = tf.nn.relu(x)
        x = self.max_pool(x)
        feature_maps.append(x)

        x = self.l2a(x, training=training)
        x = self.l2b(x, training=training)
        x = self.l2c(x, training=training)
        feature_maps.append(x)

        x = self.l3a(x, training=training)
        x = self.l3b(x, training=training)
        x = self.l3c(x, training=training)
        x = self.l3d(x, training=training)
        feature_maps.append(x)

        x = self.l4a(x, training=training)
        x = self.l4b(x, training=training)
        x = self.l4c(x, training=training)
        x = self.l4d(x, training=training)
        x = self.l4e(x, training=training)
        x = self.l4f(x, training=training)
        feature_maps.append(x)

        x = self.l5a(x, training=training)
        x = self.l5b(x, training=training)
        x = self.l5c(x, training=training)
        feature_maps.append(x)
        return feature_maps

# ---------------------------------------------
# feature pyramid Network


class FeaturePyramidNetwork(tf.keras.Model):
    def __init__(self):
        super(FeaturePyramidNetwork, self).__init__()
        self.c5p5 = tf.keras.layers.Conv2D(256, (1, 1), name='fpn_c5p5')
        self.c4p4 = tf.keras.layers.Conv2D(256, (1, 1), name='fpn_c4p4')
        self.c3p3 = tf.keras.layers.Conv2D(256, (1, 1), name='fpn_c3p3')
        self.c2p2 = tf.keras.layers.Conv2D(256, (1, 1), name='fpn_c2p2')

        self.p2 = tf.keras.layers.Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")
        self.p3 = tf.keras.layers.Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")
        self.p4 = tf.keras.layers.Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")
        self.p5 = tf.keras.layers.Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")

    def call(self, feature_maps, training=True, mask=None):
        """
        accepts feature_maps returned from backbone network and apply feature pyramid network on them and return
        transformed feature_maps
        :param feature_maps:
        :return: rpn_feature_maps and mrcnn_feature_maps
        """
        _, C2, C3, C4, C5 = feature_maps
        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = self.c5p5(C5)
        P4 = tf.add(tf.keras.layers.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                    self.c4p4(C4),
                    name="fpn_p4add")
        P3 = tf.add(tf.keras.layers.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                    self.c3p3(C3),
                    name="fpn_p3add")
        P2 = tf.add(tf.keras.layers.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                    self.c2p2(C2),
                    name="fpn_p2add")
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = self.p2(P2)
        P3 = self.p3(P3)
        P4 = self.p4(P4)
        P5 = self.p5(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]
        return {"rpn_feature_maps": rpn_feature_maps,
                "mrcnn_feature_maps": mrcnn_feature_maps}


# ---------------------------------------------
# region proposal network


class AnchorsGenerator(tf.keras.Model):
    def __init__(self, settings):
        super(AnchorsGenerator, self).__init__()
        self.scales = settings["scales"]
        self.ratios = settings["ratios"]
        self.image_shape = settings["image_shape"]
        self.image_height, self.image_width = self.image_shape
        self.feature_strides = settings["feature_strides"]
        self.anchor_stride = settings["anchor_stride"]

    def call(self, useless_inputs, training=True, mask=None):

        sqrt_ratios = tf.sqrt(tf.convert_to_tensor(self.ratios))

        image_shape = tf.convert_to_tensor(self.image_shape)
        anchors = []
        for scale, feature_stride in zip(self.scales, self.feature_strides):
            shape = image_shape / feature_stride
            height = shape[0]
            width = shape[1]
            heights, widths = tf.meshgrid(tf.range(0, height, self.anchor_stride),
                                          tf.range(0, width, self.anchor_stride))
            # broadcast is used here
            centors_x = tf.multiply(tf.to_float(tf.reshape(heights, [-1, 1])) + 0.5,
                                    feature_stride)
            centors_y = tf.multiply(tf.to_float(tf.reshape(widths, [-1, 1])) + 0.5,
                                    feature_stride)

            anchors_heights = scale / sqrt_ratios
            anchors_widths = scale * sqrt_ratios

            y1 = tf.reshape(centors_x - anchors_heights * 0.5, [-1])
            y2 = tf.reshape(centors_x + anchors_heights * 0.5, [-1])
            x1 = tf.reshape(centors_y - anchors_widths * 0.5, [-1])
            x2 = tf.reshape(centors_y + anchors_widths * 0.5, [-1])

            # normalize
            anchor = tf.stack([y1 / self.image_height, x1 / self.image_width,
                               y2 / self.image_height, x2 / self.image_width], axis=1)
            anchors.append(anchor)

        anchors = tf.concat(anchors, axis=0, name="anchors")
        return anchors


class ApplyBoundingBoxDeltas(tf.keras.layers.Layer):
    def __init__(self):
        super(ApplyBoundingBoxDeltas, self).__init__()

    def call(self, inputs, training=True, mask=None):
        """
        :param inputs: "bboxes": [batch_size, num_anchors, 4]
                        "deltas": [batch_size, num_anchors, 4]
        :return:
        """
        boxes = inputs["bboxes"]
        deltas = inputs["deltas"]
        height = boxes[:, :, 2] - boxes[:, :, 0]
        width = boxes[:, :, 3] - boxes[:, :, 1]
        center_y = boxes[:, :, 0] + 0.5 * height
        center_x = boxes[:, :, 1] + 0.5 * width
        # Apply deltas
        center_y += deltas[:, :, 0] * height
        center_x += deltas[:, :, 1] * width
        height *= tf.exp(deltas[:, :, 2])
        width *= tf.exp(deltas[:, :, 3])
        # Convert back to y1, x1, y2, x2
        y1 = center_y - 0.5 * height
        x1 = center_x - 0.5 * width
        y2 = y1 + height
        x2 = x1 + width
        refined_boxes = tf.stack([y1, x1, y2, x2], axis=2, name="apply_box_deltas_out")
        return refined_boxes


# non maximum supression that supports batch size because of padding
class NonMaxSupression(tf.keras.Model):
    def __init__(self, proposal_count):
        super(NonMaxSupression, self).__init__()
        self.proposal_count = proposal_count

    def call(self, boxes_and_scores, training=True, mask=None):
        # batch_boxes, batch_scores = boxes_and_scores

        def non_maximun_supression(boxes_and_scores):
            boxes, scores = boxes_and_scores
            index = tf.image.non_max_suppression(boxes=boxes, scores=scores,
                                                 max_output_size=self.proposal_count,
                                                 iou_threshold=0.8)
            selected_boxes = tf.gather(boxes, index)
            num_padding = tf.maximum(self.proposal_count - tf.shape(selected_boxes)[0], 0)
            padded_boxes = tf.pad(selected_boxes, [(0, num_padding), (0, 0)])
            # can manually given the shape of boxes, useful in graph mode when shape can't be inferred
            padded_boxes.set_shape([self.proposal_count, 4])
            return padded_boxes
        nms_boxes = tf.map_fn(non_maximun_supression, boxes_and_scores, dtype=tf.float32)
        return nms_boxes


class Clipper(tf.keras.layers.Layer):
    def __init__(self):
        super(Clipper, self).__init__()
        self.window = tf.constant([0., 0., 1., 1.])

    def call(self, boxes, training=True, mask=None):
        """

        :param boxes: [batch_size, num_boxes, 4]
        :return:
        """
        wy1, wx1, wy2, wx2 = tf.unstack(self.window)
        y1, x1, y2, x2 = tf.unstack(boxes, axis=2)
        # Clip
        y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
        x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
        y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
        x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
        clipped = tf.stack([y1, x1, y2, x2], axis=-1, name="clipped_boxes")
        return clipped


class RegionProposalNetwork(tf.keras.Model):
    def __init__(self, anchors_settings, proposal_count):
        super(RegionProposalNetwork, self).__init__()
        # anchors related
        self.anchors_settings = anchors_settings
        self.proposal_count = proposal_count

        # generate anchors
        self.anchors_generator = AnchorsGenerator(self.anchors_settings)
        anchors_per_location = len(self.anchors_settings["ratios"])
        anchor_stride = self.anchors_settings["anchor_stride"]
        # filter anchors with nms
        self.nms = NonMaxSupression(self.proposal_count)

        # anchors to proposals
        self.anchors_to_proposals_refiner = ApplyBoundingBoxDeltas()

        # clipper
        self.clipper = Clipper()

        # layers that contain parameters
        self.shared = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu',
                                             strides=anchor_stride,
                                             name='rpn_conv_shared')

        self.rpn_class_raw = tf.keras.layers.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                                                    activation='linear', name='rpn_class_raw')

        self.rpn_bbox_pred = tf.keras.layers.Conv2D(4 * anchors_per_location, (1, 1), padding="valid",
                                                    activation='linear', name='rpn_bbox_pred')

    def call(self, feature_maps, training=True, mask=None):
        # Shared convolutional base of the RPN
        rpn_class_logitses = []
        rpn_bbox_probs = []
        rpn_bbox_deltas = []
        for feature_map in feature_maps:
            batch_size = tf.shape(feature_map)[0]

            shared = self.shared(feature_map)

            # Anchor Score. [batch, height, width, anchors per location * 2].
            x = self.rpn_class_raw(shared)

            # Reshape to [batch, anchors, 2]
            rpn_class_logits = tf.reshape(x, [batch_size, -1, 2])
            rpn_class_logitses.append(rpn_class_logits)
            # Softmax on last dimension of BG/FG.
            rpn_prob = tf.nn.softmax(rpn_class_logits, name="rpn_class_xxx")
            rpn_bbox_probs.append(rpn_prob)
            # Bounding box refinement. [batch, H, W, anchors per location, depth]
            # where depth is [x, y, log(w), log(h)]
            x = self.rpn_bbox_pred(shared)

            # Reshape to [batch, anchors, 4]
            rpn_bbox_delta = tf.reshape(x, [batch_size, -1, 4])
            rpn_bbox_deltas.append(rpn_bbox_delta)

        # [batch_size, num_anchors, 2]
        rpn_class_logitses = tf.concat(rpn_class_logitses, axis=1, name="rpn_class_logitses")
        rpn_bbox_probs = tf.concat(rpn_bbox_probs, axis=1, name="rpn_probs")
        # [batch_size, num_anchors, 4]
        rpn_bbox_deltas = tf.concat(rpn_bbox_deltas, axis=1, name="rpn_bboxes")

        # [batch_size, num_anchors]
        rpn_bbox_scores = rpn_bbox_probs[:, :, 0]
        # return [rpn_scores, rpn_bboxes]

        # [batch_size, num_anchors, 4]
        anchors = self.anchors_generator(None)
        # return [anchors, rpn_probs, rpn_bboxes]
        # ---
        pre_nms_limit = tf.minimum(6000, tf.shape(anchors)[0])

        # [batch_size, N]
        ix = tf.nn.top_k(rpn_bbox_scores, pre_nms_limit, sorted=True, name="top_anchors").indices

        pre_nms_anchors = tf.gather(anchors, ix)
        rpn_bbox_scores = tf.map_fn(lambda params_and_indices: tf.gather(*params_and_indices), [rpn_bbox_scores, ix],
                                    dtype=tf.float32)
        rpn_bbox_deltas = tf.map_fn(lambda params_and_indices: tf.gather(*params_and_indices), [rpn_bbox_deltas, ix],
                                    dtype=tf.float32)

        # # TODO filter anchors according to scores
        anchors_to_proposals_refiner_inputs = {"deltas": rpn_bbox_deltas,
                                               "bboxes": pre_nms_anchors}

        rpn_proposals = self.anchors_to_proposals_refiner(anchors_to_proposals_refiner_inputs)

        # TODO refer to original paper to find when to do clipping
        rpn_proposals = self.clipper(rpn_proposals)

        rpn_proposals = self.nms([rpn_proposals, rpn_bbox_scores])
        return {"rpn_bbox_scores": rpn_bbox_scores,
                "rpn_proposals": rpn_proposals,
                "rpn_bbox_deltas": rpn_bbox_deltas}

# TODO add unit test for roi align related classes

# ---------------------------------------------
# roi align layer
class Assigner(tf.keras.layers.Layer):
    def __init__(self):
        super(Assigner, self).__init__()

    def call(self, proposals, training=True, mask=None):
        """

        :param proposals: [batch_size, num_proposals, 4]
        :return:
        """
        y1, x1, y2, x2 = tf.unstack(proposals, axis=2)
        h = y2 - y1
        w = x2 - x1
        # TODO, consider where to put the image_area
        image_area = 1024. * 768.
        roi_level = tf.log(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area))) / tf.log(2.)
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))

        ixes = []
        level_proposals = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            ixes.append(tf.to_int32(ix))
            level_proposal = tf.gather_nd(proposals, ix)
            level_proposals.append(level_proposal)
        # TODO figure out where to add tf.stop_gradient
        return {"ixes": ixes,
                "level_proposals": level_proposals}


class ROIAlignment(tf.keras.layers.Layer):
    def __init__(self, pool_size):
        super(ROIAlignment, self).__init__()
        self.pool_size = pool_size

    def call(self, inputs, training=True, mask=None):
        feature_maps = inputs["feature_maps"]
        ixes = inputs["ixes"]
        level_proposals = inputs["level_proposals"]
        roi_alignes = []
        for ix, level_proposal, feature_map in zip(ixes, level_proposals, feature_maps):
            roi_align = tf.image.crop_and_resize(image=feature_map,
                                                 boxes=level_proposal,
                                                 box_ind=ix[:, 0],
                                                 crop_size=[self.pool_size, self.pool_size])
            roi_alignes.append(roi_align)

        # [batch_size * number_boxes_per_image, pool_size, pool_size, channles]
        roi_alignes = tf.concat(roi_alignes, axis=0)
        ixes = tf.concat(ixes, axis=0)

        roi_alignes_shape = tf.shape(roi_alignes)
        return_roi_alignes_shape = tf.concat([tf.TensorShape([2, 2000]), roi_alignes_shape[1:]], axis=0)
        roi_alignes = tf.scatter_nd(ixes, roi_alignes, shape=return_roi_alignes_shape)
        return roi_alignes


# ---------------------------------------------
# three heads: classification, regression, mask
# implement classification and regression head together because they shared a xxx
# tf.keras.layers.Flatten


class ClassificationAndRegressionHead(tf.keras.Model):
    def __init__(self, pool_size, num_classes):
        super(ClassificationAndRegressionHead, self).__init__()
        self.pool_size = pool_size
        self.num_classes = num_classes

        self.mrcnn_class_conv1 = tf.keras.layers.Conv2D(1024, [self.pool_size, self.pool_size],
                                                        padding="valid", name="mrcnn_class_conv1")
        self.mrcnn_class_bn1 = tf.keras.layers.BatchNormalization(name='mrcnn_class_bn1')
        self.mrcnn_class_conv2 = tf.keras.layers.Conv2D(1024, (1, 1), name="mrcnn_class_conv2")
        self.mrcnn_class_bn2 = tf.keras.layers.BatchNormalization(name='mrcnn_class_bn2')

        self.mrcnn_class_logits = tf.keras.layers.Dense(self.num_classes, name='mrcnn_class_logits')
        self.mrcnn_bbox_fc = tf.keras.layers.Dense(self.num_classes * 4, activation='linear', name='mrcnn_bbox_fc')

    def call(self, feature_map, training=True, mask=None):
        # Two 1024 FC layers (implemented with Conv2D for consistency)
        # [batch_size, boxes, 1, 1, channels], need to be squeezed to [batch_size, boxes, channels]

        # first reshape [batch_size, boxes, pool_size, pool_size, channels] to
        # [batch_size * boxes, pool_size, pool_size, channels] so that tf.keras.layers.Conv2D can be used
        origin_shape = tf.shape()
        shape = tf.concat([tf.constant([-1]), tf.shape(feature_map)[2:]], axis=0)
        feature_map = tf.reshape(feature_map, shape)

        x = self.mrcnn_class_conv1(feature_map)

        x = self.mrcnn_class_bn1(x, training=training)
        x = tf.keras.layers.Activation('relu')(x)
        x = self.mrcnn_class_conv2(x)
        x = self.mrcnn_class_bn2(x, training=training)
        x = tf.keras.layers.Activation('relu')(x)

        shared = tf.squeeze(x, name="pool_squeeze")

        # Classifier head
        mrcnn_class_logits = self.mrcnn_class_logits(shared)
        mrcnn_probs = tf.nn.softmax(mrcnn_class_logits, name="mrcnn_class")

        # BBox head
        # [batch, boxes, num_classes * (dy, dx, log(dh), log(dw))]
        x = self.mrcnn_bbox_fc(shared)
        # Reshape to [batch, boxes, num_classes, (dy, dx, log(dh), log(dw))]
        batch_and_boxes_shape = shared[:, :, 0].shape
        mrcnn_bbox_shape = batch_and_boxes_shape.concatenate([self.num_classes, 4])
        print(mrcnn_bbox_shape)
        mrcnn_bbox = tf.reshape(x, mrcnn_bbox_shape, name="mrcnn_bbox")

        return {"mrcnn_class_logits": mrcnn_class_logits,
                "mrcnn_probs": mrcnn_probs,
                "mrcnn_deltas": mrcnn_bbox}


class MaskHead(tf.keras.Model):
    def __init__(self, num_classes):
        super(MaskHead, self).__init__()
        self.num_classes = num_classes
        for i in range(1, 5):
            setattr(self, "mrcnn_mask_conv%d" % i,
                    tf.keras.layers.Conv2D(256, (3, 3), padding="same", name="mrcnn_mask_conv%d" % i))
            setattr(self, "mrcnn_mask_bn%d" % i,
                    tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(name='mrcnn_mask_bn1'),
                                                    ))
            setattr(self, "relu%d" % i, tf.keras.layers.Activation('relu'))

        self.mrcnn_mask_deconv = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=2,
                                                                                                 activation="relu"),
                                                                 name="mrcnn_mask_deconv")
        self.mrcnn_mask = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(num_classes, (1, 1), strides=1,
                                                                                 activation="sigmoid"),
                                                          name="mrcnn_mask")

    def call(self, x, training=True, mask=None):
        for i in range(1, 5):
            x = getattr(self, "mrcnn_mask_conv%d" % i)(x)
            x = getattr(self, "mrcnn_mask_bn%d" % i)(x, training=training)
            x = getattr(self, "relu%i" % i)(x)
        x = self.mrcnn_mask_deconv(x)
        x = self.mrcnn_mask(x)
        return x


# ---------------------------------------------
# mask rcnn


class MaskRCNN(tf.keras.Model):
    def __init__(self, config, mode):
        super(MaskRCNN, self).__init__()
        assert mode in ("training", "inference")
        self.config = config
        n_proposals = self.config.POST_NMS_ROIS_TRAINING if mode == "training" else self.config.POST_NMS_ROIS_INFERENCE

        # submodels for maskrcnn
        self.resnet = ResidualNetwork()
        self.feature_pyramid_network = FeaturePyramidNetwork()
        anchors_settings = {"ratios": self.config.RPN_ANCHOR_RATIOS,
                            "scales": self.config.RPN_ANCHOR_SCALES,
                            "image_shape": self.config.IMAGE_SHAPE,
                            "feature_strides": self.config.BACKBONE_STRIDES,
                            "anchor_stride": self.config.RPN_ANCHOR_STRIDE}
        self.region_proposal_network = RegionProposalNetwork(anchors_settings=anchors_settings,
                                                             proposal_count=n_proposals)
        self.assigner = Assigner()
        self.class_and_regression_align = ROIAlignment(self.config.POOL_SIZE)
        self.mask_align = ROIAlignment(self.config.MASK_POOL_SIZE)
        self.classifiction_and_regression_head = ClassificationAndRegressionHead(self.config.POOL_SIZE,
                                                                                 self.config.NUM_CLASSES)
        self.mask_head = MaskHead(self.config.NUM_CLASSES)
        self.proposal_refiner = ApplyBoundingBoxDeltas()

    def call(self, images, training=True, mask=None):
        """

        :param images:
        :return:
        """
        feature_maps = self.resnet(images)
        fpn_outputs = self.feature_pyramid_network(feature_maps)
        rpn_outputs = self.region_proposal_network(fpn_outputs["rpn_feature_maps"])
        proposals = rpn_outputs["rpn_proposals"]
        assigner_outputs = self.assigner(proposals)
        roi_align_inputs = {"ixes": assigner_outputs["ixes"],
                            "feature_maps": fpn_outputs["rpn_feature_maps"],
                            "level_proposals": assigner_outputs["level_proposals"]}

        cls_and_reg_alignes = self.class_and_regression_align(roi_align_inputs)
        mask_alignes = self.mask_align(roi_align_inputs)

        return_dict = self.classifiction_and_regression_head(cls_and_reg_alignes, False)
        # return return_dict
        masks = self.mask_head(mask_alignes, False)
        return_dict["mrcnn_mask"] = masks
        return return_dict
        proposal_refiner_inputs = {"bboxes": proposals,
                                   "deltas": return_dict["mrcnn_deltas"]}
        predicted_boxes = self.proposal_refiner(proposal_refiner_inputs)
        return_dict["mrcnn_boxes"] = predicted_boxes
        return return_dict


# ---------------------------------------------
# loss related classes


class Matcher(tf.keras.Model):
    def __init__(self):
        super(Matcher, self).__init__()

    def call(self, inputs, training=True, mask=None):
        boxes1, boxes2 = inputs


class Subsampler(tf.keras.Model):
    def __init__(self):
        super(Subsampler, self).__init__()

    def call(self, inputs, training=True, mask=None):
        pass


if __name__ == "__main__":
    import tensorflow as tf
    tf.enable_eager_execution()
    features = tf.random_normal((2, 3, 2, 4))
    bn = tf.keras.layers.BatchNormalization()
    # conv2d = tf.keras.layers.Conv2D(1, (7, 7), padding="valid")
    #
    # shape = tf.concat([tf.constant([-1]), tf.shape(features)[2:]], axis=0)
    # reshape = tf.reshape(features, shape)
    #
    # o1 = conv2d(reshape)
    #
    # o2 = tf.keras.layers.TimeDistributed(conv2d)(features)