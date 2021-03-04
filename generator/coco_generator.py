
import os
import numpy as np
from pycocotools.coco import COCO
from generator.generator import Generator

'''
    voc dataset directory:
        VOC2007
                Annotations
                ImageSets
                JPEGImages
        VOC2012
                Annotations
                ImageSets
                JPEGImages
    coco dataset directory:
        annotations/instances_train2017.json
        annotations/instances_val2017.json
        images/train2017
        images/val2017
    '''
class CocoGenerator(Generator):

    def __init__(self, args, mode='train'):
        super(CocoGenerator, self).__init__(args, mode)

    def get_data(self,args, mode):
        if mode == 'train':
            sets_name = args.coco_train_set
        else:
            sets_name = args.coco_val_set

        self.coco      = COCO(os.path.join(args.dataset, 'annotations', 'instances_' + sets_name + '.json'))
        self.image_ids = self.coco.getImgIds()
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
        self.classes = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels_inverse[c['id']] = len(self.coco_labels_inverse)
        img_path_list = []
        boxes_and_labels = []
        for id in self.image_ids:
            image_info = self.coco.imgs[id]
            img_path_list.append(os.path.join(args.dataset, 'images', sets_name, image_info['file_name']))
        for image_index in self.image_ids:
            boxes_and_labels.append(self.parse_json(image_index))
        return img_path_list,boxes_and_labels


    def parse_json(self,image_index):
        """ Load annotations for an image_index.
        """
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=image_index, iscrowd=False)
        labels = np.empty((0,))
        boxes = np.empty((0, 4))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return np.empty((0, 5))
        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            labels = np.concatenate([labels, [self.coco_labels_inverse[a['category_id']]]], axis=0)
            boxes = np.concatenate([boxes, [[
                a['bbox'][0],
                a['bbox'][1],
                a['bbox'][0] + a['bbox'][2],
                a['bbox'][1] + a['bbox'][3],
            ]]], axis=0)
        labels = np.expand_dims(labels, axis=-1)
        return np.concatenate([boxes, labels], axis=-1)


