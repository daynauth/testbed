import fiftyone as fo
import fiftyone.zoo as foz

if not fo.dataset_exists('efficientdet-coco'):
    print("dataset does not exist")
    # dataset = foz.load_zoo_dataset(
    #     "coco-2017",
    #     split="validation",
    #     dataset_name="efficientdet-coco"
    # )

    # dataset.persistent = True