
import tensorflow as tf

from utils.optimizers import get_optimizers
from utils.eager_coco_map import EagerCocoMap
from generator.generator_builder import get_generator
from model.model_builder import get_model
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint,TensorBoard
import os
from tqdm import tqdm
from tensorboard import program
import webbrowser
import logging
from utils.lr_scheduler import get_lr_scheduler
from utils.fit_coco_map import CocoMapCallback
from losses.loss_builder import get_loss
import time
import argparse
logging.getLogger().setLevel(logging.ERROR)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for using EfficientDet.')

    parser.add_argument('--model-type', default='d0', help="choices=['d0','d1','d2',...,'d7x']")

    parser.add_argument('--train-mode', default='fit', help="choices=['fit','eager']")
    parser.add_argument('--model-name', default='efficientdet', help="choices=['efficientdet']")
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--start-eval-epoch', default=100, type=int)
    parser.add_argument('--eval-epoch-interval', default=1, type=int)
    parser.add_argument('--use-pretrain', default=True, type=bool)
    parser.add_argument('--export-dir', default='./export')
    parser.add_argument('--checkpoints-dir', default='./checkpoints',help="Directory to store  checkpoints of model during training.")

    #dataset
    parser.add_argument('--dataset-type', default='voc', help="voc,coco")
    parser.add_argument('--num-classes', default=1, type=int)
    parser.add_argument('--class-names', default='dataset/pothole.names', help="voc.names,coco.names")
    parser.add_argument('--dataset', default='dataset/pothole_voc')#
    #voc data format setting
    parser.add_argument('--voc-train-set', default='dataset_1,train')
    parser.add_argument('--voc-val-set', default='dataset_1,val')
    parser.add_argument('--voc-skip-difficult', default=True)
    #coco data format setting
    parser.add_argument('--coco-train-set', default='train2017')
    parser.add_argument('--coco-valid-set', default='val2017')
    #agumentation
    parser.add_argument('--augment', default='ssd_random_crop',help="choices=[None,'only_flip_left_right','ssd_random_crop','mosaic']")
    parser.add_argument('--max-box-num-per-image', default=100, type=int)

    #loss
    parser.add_argument('--box-loss', default='huber',help="")
    parser.add_argument('--cls-loss', default='focal', help="")
    parser.add_argument('--focal-alpha', default= 0.25)
    parser.add_argument('--focal-gamma', default=1.5)
    # parser.add_argument('--moving-average-decay', default=0.9998)

    #optimizer
    parser.add_argument('--optimizer', default='adam', help="choices=[adam,sgd]")
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--nesterov', default=True)
    parser.add_argument('--weight-decay', default=5e-4)
    #lr scheduler
    parser.add_argument('--lr-scheduler', default='cosine', type=str, help="choices=['step','warmup_cosinedecay']")
    parser.add_argument('--init-lr', default=1e-3, type=float)
    parser.add_argument('--lr-decay', default=0.1, type=float)
    parser.add_argument('--lr-decay-epoch', default=[160, 180])
    parser.add_argument('--warmup-epochs', default=10, type=int)
    parser.add_argument('--warmup-lr', default=1e-6, type=float)
    #postprocess
    parser.add_argument('--nms', default='hard_nms_tf', help="choices=['hard_nms_tf']")
    parser.add_argument('--nms-max-box-num', default=300)
    parser.add_argument('--nms-iou-threshold', default=0.5, type=float)
    parser.add_argument('--nms-score-threshold', default=0.05, type=float)
    #anchor
    parser.add_argument('--anchor-match-type', default='wh_ratio',help="choices=['iou','wh_ratio']")
    parser.add_argument('--anchor-match-iou_thr', default=0.2, type=float)
    parser.add_argument('--anchor-match-wh-ratio-thr', default=4.0, type=float)

    parser.add_argument('--label-smooth', default=0.0, type=float)
    parser.add_argument('--accumulated-gradient-num', default=1, type=int)

    parser.add_argument('--min-level', default=3, type=int)
    parser.add_argument('--max-level', default=7, type=int)
    parser.add_argument('--num-scales', default=3, type=int)
    parser.add_argument('--aspect-ratios', default=[1.0, 2.0, 0.5])
    parser.add_argument('--anchor-scale', default=4., type=float)

    return parser.parse_args(args)

def main(args):
    #create dataset
    train_generator, val_dataset, pred_generator = get_generator(args)
    #create model
    model = get_model(args,training=True)
    #create loss
    loss_fun = get_loss(args)
    #create learning rate scheduler
    lr_scheduler = get_lr_scheduler(args)
    #create optimizer
    optimizer = get_optimizers(args)
    best_weight_path = ''
    #tensorboard
    open_tensorboard_url = False
    os.system('rm -rf ./logs/')
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'logs','--reload_interval','15'])
    url = tb.launch()
    print("Tensorboard engine is running at {}".format(url))
    if args.train_mode == 'fit':
        if open_tensorboard_url:
            webbrowser.open(url, new=1)
        mAP_writer = tf.summary.create_file_writer("logs/mAP")
        coco_map_callback = CocoMapCallback(pred_generator,model,args,mAP_writer)
        callbacks = [
            tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
            coco_map_callback,
            # ReduceLROnPlateau(verbose=1),
            # EarlyStopping(patience=3, verbose=1),
            TensorBoard(log_dir='logs')
        ]
        model.compile(optimizer=optimizer,loss=loss_fun,run_eagerly=False)
        model.fit(train_generator,epochs=args.epochs,
                            callbacks=callbacks,
                            # validation_data=val_dataset,
                            max_queue_size=10,
                            workers=8,
                            use_multiprocessing=False
                            )
        best_weight_path = coco_map_callback.best_weight_path
    else:
        print("loading dataset...")
        start_time = time.perf_counter()
        coco_map = EagerCocoMap(pred_generator, model, args)
        max_coco_map = -1
        max_coco_map_epoch = -1
        accumulate_num = args.accumulated_gradient_num
        accumulate_index = 0
        accum_gradient = [tf.Variable(tf.zeros_like(this_var)) for this_var in model.trainable_variables]

        train_writer = tf.summary.create_file_writer("logs/train")
        mAP_writer = tf.summary.create_file_writer("logs/mAP")

        for epoch in range(int(args.epochs)):
            lr = lr_scheduler(epoch)
            optimizer.learning_rate.assign(lr)
            remaining_epoches = args.epochs - epoch - 1
            epoch_start_time = time.perf_counter()
            train_loss = 0
            train_generator_tqdm = tqdm(enumerate(train_generator), total=len(train_generator))
            for batch_index, (batch_imgs, batch_labels)  in train_generator_tqdm:
                s1 = time.time()
                if args.model_name == "efficientdet":
                    with tf.GradientTape() as tape:
                        model_outputs = model(batch_imgs, training=True)
                        num_level = args.max_level - args.min_level + 1
                        cls_loss,box_loss = 0,0
                        for level in range(num_level):
                            cls_loss += loss_fun[0][level](batch_labels[0][level],model_outputs[0][level])
                            box_loss += loss_fun[1][level](batch_labels[1][level], model_outputs[1][level])
                        data_loss = cls_loss+box_loss
                        # data_loss = loss_fun(batch_labels,model_outputs)

                        total_loss = data_loss + args.weight_decay * tf.add_n(
                            [tf.nn.l2_loss(v) for v in model.trainable_variables if
                             'batch_normalization' not in v.name])
                else:
                    raise ValueError('unsupported model type {}'.format(args.model_name))

                grads = tape.gradient(total_loss, model.trainable_variables)
                accum_gradient = [acum_grad.assign_add(grad) for acum_grad, grad in zip(accum_gradient, grads)]

                accumulate_index += 1
                if accumulate_index == accumulate_num:
                    optimizer.apply_gradients(zip(accum_gradient, model.trainable_variables))
                    accum_gradient = [ grad.assign_sub(grad) for grad in accum_gradient]
                    accumulate_index = 0
                train_loss += total_loss
                train_generator_tqdm.set_description(
                    "epoch:{}/{},train_loss:{:.4f},lr:{:.6f}".format(epoch, args.epochs,
                                                                                     train_loss/(batch_index+1),
                                                                                     optimizer.learning_rate.numpy()))
            train_generator.on_epoch_end()

            with train_writer.as_default():
                tf.summary.scalar("train_loss", train_loss/len(train_generator), step=epoch)
                train_writer.flush()

            #evaluation
            if epoch >= args.start_eval_epoch:
                if epoch % args.eval_epoch_interval == 0:
                    summary_metrics = coco_map.eval()
                    if summary_metrics['Precision/mAP@.50IOU'] > max_coco_map:
                        max_coco_map = summary_metrics['Precision/mAP@.50IOU']
                        max_coco_map_epoch = epoch
                        best_weight_path = os.path.join(args.checkpoints_dir, 'best_weight_{}_{}_{:.3f}'.format(args.model_name+"_"+args.model_type,max_coco_map_epoch, max_coco_map))
                        model.save_weights(best_weight_path)

                    print("max_coco_map:{},epoch:{}".format(max_coco_map,max_coco_map_epoch))
                    with mAP_writer.as_default():
                        tf.summary.scalar("mAP@0.5", summary_metrics['Precision/mAP@.50IOU'], step=epoch)
                        mAP_writer.flush()

            cur_time = time.perf_counter()
            one_epoch_time = cur_time - epoch_start_time
            print("time elapsed: {:.3f} hour, time left: {:.3f} hour".format((cur_time-start_time)/3600,remaining_epoches*one_epoch_time/3600))

            if epoch>0 and not open_tensorboard_url:
                open_tensorboard_url = True
                webbrowser.open(url,new=1)

    print("Training is finished!")
    #save model
    print("Exporting model...")
    if args.export_dir and best_weight_path:
        tf.keras.backend.clear_session()
        pred_model = get_model(args, training=False)
        pred_model.load_weights(best_weight_path)
        best_model_path = os.path.join(args.export_dir,best_weight_path.split('/')[-1].replace('weight','model'),'1')
        tf.saved_model.save(pred_model, best_model_path)
import sys
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
