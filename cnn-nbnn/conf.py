
{ 'params':
  {
    'train_images': '/idiap/resource/database/imagenet/ILSVRC2012/ILSVRC2012_images_train/*/*.JPEG',
    'val_images': '/idiap/resource/database/imagenet/ILSVRC2012/ILSVRC2012_images_val/*/*.JPEG',
    'ilsvrc_meta': '/idiap/resource/database/imagenet/ILSVRC2012/ILSVRC2012_devkit_t12/data/meta.mat',
    'ilsvrc_validation_labels': '/idiap/resource/database/imagenet/ILSVRC2012/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt',
    'train_target': '/idiap/temp/ikuzbor/ilsvrc_caffe_patches/train',
    'val_target': '/idiap/temp/ikuzbor/ilsvrc_caffe_patches/val',

    # Objects/places hybrid model
    # 'caffe_model': '/idiap/temp/bcaputo/fabio_data/network/hybridCNN_iter_700000_upgraded.caffemodel',
    # 'caffe_proto': '/idiap/temp/bcaputo/fabio_data/network/hybridCNN_deploy_no_relu_upgraded.prototxt',
    # 'caffe_mean': '/idiap/temp/bcaputo/fabio_data/network/hybrid_mean.npy',

    # Objects-only model
    # 'caffe_model': '/idiap/temp/ikuzbor/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel',
    # 'caffe_proto': '/idiap/temp/ikuzbor/caffe/models/bvlc_alexnet/deploy.prototxt',
    # 'caffe_mean': '/idiap/temp/ikuzbor/caffe/data/ilsvrc12/imagenet_mean.binaryproto',


    'caffe_model': '/remote/lustre/2/temp/ikuzbor/fabio_data/network/hybridCNN_iter_700000_upgraded.caffemodel',
    'caffe_proto': '/remote/lustre/2/temp/ikuzbor/fabio_data/network/hybridCNN_deploy_upgraded.prototxt',
    'caffe_mean': '/remote/lustre/2/temp/ikuzbor/fabio_data/network/hybrid_mean.npy',

    # SmoothML3 models
    'model_path': '/home/fmcarlucci/data/models',
    'save_model_every_t': 12512,

    'layer_name': '7relu',
    'dim': 4096,
    'minibatch_size': 256,
    'num_patches': 100,
    'patch_size': 32,
    'levels': 3,
    'oversample': False,
    'image_dim': 227,
    'extraction_method': 'extra'
  }
}
