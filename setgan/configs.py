

dataset_paths = {
	'face': '../setgan2/datasets/vggface/train/',
    'imagenet': '/scratch/hdd001/datasets/imagenet/train'
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