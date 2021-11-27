import fiftyone as fo
import fiftyone.zoo as foz

def load_coco_dataset(dataset_name : str, split = "validation") -> fo.Dataset:
    if not fo.dataset_exists(dataset_name):
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split=split,
            dataset_name=dataset_name,
            dataset_dir='/data/datasets'
        )

        dataset.persistent = True

    else:
        dataset = fo.load_dataset(dataset_name)
    
    return dataset