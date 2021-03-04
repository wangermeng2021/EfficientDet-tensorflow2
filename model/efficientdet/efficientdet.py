
from model.efficientdet.network import EfficientDetNet
from model.efficientdet import postprocess
from utils import preprocess
import tensorflow as tf
from config import efficientdet_config
def get_model(args, training=True):
    model_args = efficientdet_config.get_struct_args(args)
    if training:
        cur_num_classes = model_args.num_classes
        model_args.num_classes = 90
        model_pretrain = EfficientDetNet(model_args)
        model_inputs_pretrain = tf.keras.layers.Input(shape=(model_args.image_size, model_args.image_size, 3))
        model_outputs_pretrain = model_pretrain(model_inputs_pretrain,training=True)
        model_pretrain = tf.keras.Model(inputs=model_inputs_pretrain, outputs=model_outputs_pretrain)
        model_args.num_classes = cur_num_classes
        if args.use_pretrain:
            try:
                model_pretrained_weights = "./pretrain/efficientdet-{}/model".format(args.model_type)
                model_pretrain.load_weights(model_pretrained_weights).expect_partial()
            except:
                raise ValueError('weight file {} is invalid!'.format(model_pretrained_weights))

        model = EfficientDetNet(model_args)
        model_inputs = tf.keras.layers.Input(shape=(model_args.image_size, model_args.image_size, 3))
        model_outputs = model(model_inputs)
        num_level = model_args.max_level-model_args.min_level+1
        level_cls_outputs = [tf.keras.layers.Lambda(lambda x: x, name='level_{}_cls'.format(level))(model_outputs[0][level]) for level in range(num_level)]
        level_box_outputs = [tf.keras.layers.Lambda(lambda x: x, name='level_{}_box'.format(level))(model_outputs[1][level]) for level in range(num_level)]
        model = tf.keras.Model(inputs=model_inputs, outputs=(level_cls_outputs,level_box_outputs))

        for layer in model_pretrain.layers[-1].layers:
            if layer.name!='class_net':
                model.layers[-11].get_layer(layer.name).set_weights(model_pretrain.layers[-1].get_layer(layer.name).get_weights())
        return model
    else:
        model = EfficientDetNet(model_args)
        image_size = model_args.image_size
        model_inputs = tf.keras.layers.Input(shape = (None,None, 3),dtype= tf.dtypes.uint8)
        resized_inputs = tf.keras.layers.Lambda(lambda x: preprocess.resize_img_tf(x,(image_size,image_size)))(model_inputs)
        preprocessed_inputs = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.dtypes.float32))(resized_inputs[0])
        preprocessed_inputs = tf.keras.layers.Lambda(lambda x: preprocess.normalize(x))(preprocessed_inputs)

        model_outputs = model(preprocessed_inputs,training=False)
        cls_out_list, box_out_list = model_outputs
        cls_outputs, box_outputs = {}, {}
        for i in range(model_args.min_level, model_args.max_level + 1):
            cls_outputs[i] = cls_out_list[i - model_args.min_level]
            box_outputs[i] = box_out_list[i - model_args.min_level]
        if args.nms == 'hard_nms_tf':
            nms_boxes, nms_scores, nms_classes, nms_num_valid = postprocess.postprocess(
                args, cls_outputs, box_outputs,tf.cast([image_size,image_size],tf.dtypes.float32))
            nms_boxes = (nms_boxes-tf.cast(tf.tile(resized_inputs[2],[2]),tf.dtypes.float32))/resized_inputs[1]

        else:
            raise ValueError('Unsupported nms type {}'.format(args.postprocess.nms))

        model = tf.keras.Model(inputs=model_inputs, outputs=[nms_boxes, nms_scores, nms_classes, nms_num_valid])
        return model




