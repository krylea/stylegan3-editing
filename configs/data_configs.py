from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test']
	},
    'animalfaces_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['animalfaces_train'],
		'train_target_root': dataset_paths['animalfaces_train'],
		'test_source_root': dataset_paths['animalfaces_test'],
		'test_target_root': dataset_paths['animalfaces_test']
	},
    'flowers_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['flowers_train'],
		'train_target_root': dataset_paths['flowers_train'],
		'test_source_root': dataset_paths['flowers_test'],
		'test_target_root': dataset_paths['flowers_test']
	},
    'vggface_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['vggface_train'],
		'train_target_root': dataset_paths['vggface_train'],
		'test_source_root': dataset_paths['vggface_test'],
		'test_target_root': dataset_paths['vggface_test']
	}
}