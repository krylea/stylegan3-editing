

dataset_paths = {
	'face': '../setgan2/datasets/vggface/train/',
    'imagenet': '/scratch/hdd001/datasets/imagenet/train',
	'celeba-src': "/scratch/ssd002/datasets/celeba/Img/img_align_celeba",
	'celeba-ident': "/scratch/ssd002/datasets/celeba/Anno/identity_CelebA.txt",
	'celeba-attr': "/scratch/ssd002/datasets/celeba/Anno/list_attr_celeba.txt",
    'cifar-train': '../setgan2/datasets/cifar100/train',
    'cifar-test': '../setgan2/datasets/cifar100/test',
    'mini-train': '../setgan2/datasets/mini-imagenet/train',
    'mini-test': '../setgan2/datasets/mini-imagenet/test',
}


model_paths = {
	# models for backbones and losses
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'curricular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',

    # encoders and decoders
    'stylegan_xl_face_256_encoder': 'experiments/test-ffhq-1/checkpoint.pt',
    'stylegan_xl_face_256': '../stylegan-xl/pretrained_models/ffhq256.pkl',

}