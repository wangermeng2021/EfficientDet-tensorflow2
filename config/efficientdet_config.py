
EFFICIENTDET_CFG ={
    'network':
        {
            'efficientdet-d0':
                dict(
                    name='efficientdet-d0',
                    backbone_name='efficientnet-b0',
                    image_size=512,
                    fpn_num_filters=64,
                    fpn_cell_repeats=3,
                    box_class_repeats=3,
                    anchor_scale=4.0,
                    max_level=7,
                    fpn_weight_method='fastattn',
                ),
            'efficientdet-d1':
                dict(
                    name='efficientdet-d1',
                    backbone_name='efficientnet-b1',
                    image_size=640,
                    fpn_num_filters=88,
                    fpn_cell_repeats=4,
                    box_class_repeats=3,
                    anchor_scale=4.0,
                    max_level=7,
                    fpn_weight_method='fastattn',
                ),
            'efficientdet-d2':
                dict(
                    name='efficientdet-d2',
                    backbone_name='efficientnet-b2',
                    image_size=768,
                    fpn_num_filters=112,
                    fpn_cell_repeats=5,
                    box_class_repeats=3,
                    anchor_scale=4.0,
                    max_level=7,
                    fpn_weight_method='fastattn',
                ),
            'efficientdet-d3':
                dict(
                    name='efficientdet-d3',
                    backbone_name='efficientnet-b3',
                    image_size=896,
                    fpn_num_filters=160,
                    fpn_cell_repeats=6,
                    box_class_repeats=4,
                    anchor_scale=4.0,
                    max_level=7,
                    fpn_weight_method='fastattn',
                ),
            'efficientdet-d4':
                dict(
                    name='efficientdet-d4',
                    backbone_name='efficientnet-b4',
                    image_size=1024,
                    fpn_num_filters=224,
                    fpn_cell_repeats=7,
                    box_class_repeats=4,
                    anchor_scale=4.0,
                    max_level=7,
                    fpn_weight_method='fastattn',
                ),
            'efficientdet-d5':
                dict(
                    name='efficientdet-d5',
                    backbone_name='efficientnet-b5',
                    image_size=1280,
                    fpn_num_filters=288,
                    fpn_cell_repeats=7,
                    box_class_repeats=4,
                    anchor_scale=4.0,
                    max_level=7,
                    fpn_weight_method='fastattn',

                ),
            'efficientdet-d6':
                dict(
                    name='efficientdet-d6',
                    backbone_name='efficientnet-b6',
                    image_size=1280,
                    fpn_num_filters=384,
                    fpn_cell_repeats=8,
                    box_class_repeats=5,
                    anchor_scale=4.0,
                    max_level=7,
                    fpn_weight_method='sum',  # Use unweighted sum for stability.
                ),
            'efficientdet-d7':
                dict(
                    name='efficientdet-d7',
                    backbone_name='efficientnet-b6',
                    image_size=1536,
                    fpn_num_filters=384,
                    fpn_cell_repeats=8,
                    box_class_repeats=5,
                    anchor_scale=5.0,
                    max_level=7,
                    fpn_weight_method='sum',  # Use unweighted sum for stability.
                ),
            'efficientdet-d7x':
                dict(
                    name='efficientdet-d7x',
                    backbone_name='efficientnet-b7',
                    image_size=1536,
                    fpn_num_filters=384,
                    fpn_cell_repeats=8,
                    box_class_repeats=5,
                    anchor_scale=4.0,
                    max_level=8,
                    fpn_weight_method='sum',  # Use unweighted sum for stability.
                ),
        }
}
from utils.struct_config import Config
def get_struct_args(args):

    model_name = args.model_name+'-'+args.model_type
    EFFICIENTDET_CFG['name']=EFFICIENTDET_CFG['network'][model_name]['name']
    EFFICIENTDET_CFG['backbone_name'] = EFFICIENTDET_CFG['network'][model_name]['backbone_name']
    EFFICIENTDET_CFG['image_size'] = EFFICIENTDET_CFG['network'][model_name]['image_size']
    EFFICIENTDET_CFG['fpn_num_filters'] = EFFICIENTDET_CFG['network'][model_name]['fpn_num_filters']
    EFFICIENTDET_CFG['fpn_cell_repeats'] = EFFICIENTDET_CFG['network'][model_name]['fpn_cell_repeats']
    EFFICIENTDET_CFG['box_class_repeats'] = EFFICIENTDET_CFG['network'][model_name]['box_class_repeats']
    EFFICIENTDET_CFG['anchor_scale'] = EFFICIENTDET_CFG['network'][model_name]['anchor_scale']
    EFFICIENTDET_CFG['max_level'] = EFFICIENTDET_CFG['network'][model_name]['max_level']
    EFFICIENTDET_CFG['fpn_weight_method'] = EFFICIENTDET_CFG['network'][model_name]['fpn_weight_method']
    EFFICIENTDET_CFG['num_classes'] = args.num_classes

    EFFICIENTDET_CFG['min_level'] = args.min_level
    EFFICIENTDET_CFG['max_level'] = args.max_level
    EFFICIENTDET_CFG['num_scales'] = args.num_scales
    EFFICIENTDET_CFG['aspect_ratios'] = args.aspect_ratios
    EFFICIENTDET_CFG['anchor_scale'] = args.anchor_scale

    args = Config(EFFICIENTDET_CFG)
    return args
