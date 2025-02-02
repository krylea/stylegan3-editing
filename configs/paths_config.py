from pathlib import Path

dataset_paths = {
	'celeba_train': Path(''),
	'celeba_test': Path(''),

    'vggface_train': '../setgan2/datasets/vggface/train/',
    'vggface_test': '../setgan2/datasets/vggface/test/',
    'animalfaces_train': '../setgan2/datasets/animalfaces/train',
    'animalfaces_test': '../setgan2/datasets/animalfaces/test',
    'flowers_train': '../setgan2/datasets/flowers/train',
    'flowers_test': '../setgan2/datasets/flowers/test',
    'cifar_train': '../setgan2/datasets/cifar100/train',
    'cifar_test': '../setgan2/datasets/cifar100/test',
    'mini_train': '../setgan2/datasets/mini-imagenet/train',
    'mini_test': '../setgan2/datasets/mini-imagenet/test',

	'ffhq': '/ssd003/projects/ffhq/images1024x1024/',
	'ffhq_unaligned': Path('/ssd003/projects/ffhq/images1024x1024/'),

    'imagenet': '/scratch/hdd001/datasets/imagenet/train',
    
	'celeba': Path('/scratch/ssd002/datasets/celeba/Img/img_align_celeba')
}

model_paths = {
	# models for backbones and losses
	'ir_se50': Path('pretrained_models/model_ir_se50.pth'),
	# stylegan3 generators
    'stylegan_xl_ffhq_256': '../stylegan-xl/pretrained_models/ffhq256.pkl',
    'stylegan_xl_ffhq_1024': Path('../stylegan-xl/pretrained_models/ffhq1024.pkl'),
	'stylegan3_ffhq': Path('pretrained_models/stylegan3-r-ffhq-1024x1024.pkl'),
	'stylegan3_ffhq_pt': Path('pretrained_models/sg3-r-ffhq-1024.pt'),
	'stylegan3_ffhq_unaligned': Path('pretrained_models/stylegan3-r-ffhqu-1024x1024.pkl'),
	'stylegan3_ffhq_unaligned_pt': Path('pretrained_models/sg3-r-ffhqu-1024.pt'),
	# model for face alignment
	'shape_predictor': Path('pretrained_models/shape_predictor_68_face_landmarks.dat'),
	# models for ID similarity computation
	'curricular_face': Path('pretrained_models/CurricularFace_Backbone.pth'),
	'mtcnn_pnet': Path('pretrained_models/mtcnn/pnet.npy'),
	'mtcnn_rnet': Path('pretrained_models/mtcnn/rnet.npy'),
	'mtcnn_onet': Path('pretrained_models/mtcnn/onet.npy'),
	# classifiers used for interfacegan training
	'age_estimator': Path('pretrained_models/dex_age_classifier.pth'),
	'pose_estimator': Path('pretrained_models/hopenet_robust_alpha1.pkl'),

    # encoders
    'stylegan_xl_ffhq_256_encoder': Path('experiments/test-ffhq-1/checkpoint.pt')
}

styleclip_directions = {
	"ffhq": {
		'delta_i_c': Path('editing/styleclip_global_directions/sg3-r-ffhq-1024/delta_i_c.npy'),
		's_statistics': Path('editing/styleclip_global_directions/sg3-r-ffhq-1024/s_stats'),
	},
	'templates': Path('editing/styleclip_global_directions/templates.txt')
}

interfacegan_aligned_edit_paths = {
	'age': Path('editing/interfacegan/boundaries/ffhq/age_boundary.npy'),
	'smile': Path('editing/interfacegan/boundaries/ffhq/Smiling_boundary.npy'),
	'pose': Path('editing/interfacegan/boundaries/ffhq/pose_boundary.npy'),
	'Male': Path('editing/interfacegan/boundaries/ffhq/Male_boundary.npy'),
}

interfacegan_unaligned_edit_paths = {
	'age': Path('editing/interfacegan/boundaries/ffhqu/age_boundary.npy'),
	'smile': Path('editing/interfacegan/boundaries/ffhqu/Smiling_boundary.npy'),
	'pose': Path('editing/interfacegan/boundaries/ffhqu/pose_boundary.npy'),
	'Male': Path('editing/interfacegan/boundaries/ffhqu/Male_boundary.npy'),
}
