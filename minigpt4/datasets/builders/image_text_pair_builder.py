import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.laion_dataset import LaionDataset
from minigpt4.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset
from minigpt4.datasets.datasets.coco_dataset import ReferCOCODataset
from minigpt4.datasets.datasets.gqa_datasets import GQADataset
from minigpt4.datasets.datasets.aok_vqa_datasets import AOKVQADataset
from minigpt4.datasets.datasets.coco_vqa_datasets import COCOVQADataset
from minigpt4.datasets.datasets.coco_caption import COCOCapDataset

from minigpt4.datasets.datasets.mer_datasets import MER2023Dataset, CAERDataset, DFEWDataset, MCEIUDataset, IEMOCAPDataset, E3Dataset, MAFWDataset, BOLDDataset, CHSIMEV2SDataset, CMUMOSEISDataset, CMUMOSISDataset, MELDDataset
from minigpt4.datasets.datasets.mer_datasets_ft import MER2023DatasetFT, CAERDatasetFT, DFEWDatasetFT, MCEIUDatasetFT, IEMOCAPDatasetFT, E3DatasetFT, MAFWDatasetFT, BOLDDatasetFT, CHSIMEV2SDatasetFT, CMUMOSEIDatasetFT, CMUMOSIDatasetFT, MELDDatasetFT

from minigpt4.datasets.datasets.memo_bench import MEMOBenchDataset


class AllRefCOCOBuilder(BaseDatasetBuilder):

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        image_path = build_info.image_path
        ann_path = build_info.ann_path

        datasets = dict()

        if not os.path.exists(image_path):
            warnings.warn("image path {} does not exist.".format(image_path))
        if not os.path.exists(ann_path):
            warnings.warn("ann path {} does not exist.".format(ann_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=ann_path,
            vis_root=image_path,
            dataset=build_info.dataset,
            splitBy=build_info.splitBy
        )

        return datasets
    

@registry.register_builder("refcoco")
class RefCOCOBuilder(AllRefCOCOBuilder):
    train_dataset_cls = ReferCOCODataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco_bbox/refcoco.yaml",
    }

@registry.register_builder("refcocop")
class RefCOCOPBuilder(AllRefCOCOBuilder):
    train_dataset_cls = ReferCOCODataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco_bbox/refcocop.yaml",
    }


@registry.register_builder("refcocog")
class RefCOCOGBuilder(AllRefCOCOBuilder):
    train_dataset_cls = ReferCOCODataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco_bbox/refcocog.yaml",
    }


@registry.register_builder("coco_vqa")
class COCOVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_vqa.yaml",
    }

@registry.register_builder("ok_vqa")
class OKVQABuilder(COCOVQABuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/okvqa/defaults.yaml",
    }


@registry.register_builder("aok_vqa")
class AOKVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQADataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/aokvqa/defaults.yaml"}


@registry.register_builder("gqa")
class GQABuilder(BaseDatasetBuilder):
    train_dataset_cls = GQADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gqa/balanced_val.yaml",
    }



class DocumentVQABuilder(BaseDatasetBuilder):
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.image_path,
            ann_path=build_info.ann_path
        )
        return datasets


@registry.register_builder("cc_sbu")
class CCSBUBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/cc_sbu/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("laion")
class LaionBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets



@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/caption.yaml",
    }



@registry.register_builder("cc_sbu_align")
class CCSBUAlignBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUAlignDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cc_sbu/align.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'filter_cap.json')],
            vis_root=os.path.join(storage_path, 'image'),
        )

        return datasets


######################### MER Datasets ############################################
# MER2023Dataset
class MERDatasetBuilder(BaseDatasetBuilder):    
    def _download_ann(self):
        pass
    def _download_vis(self):
        pass
    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )
        return datasets  
    
@registry.register_builder("mer2023")
class MER2023Builder(MERDatasetBuilder):
    train_dataset_cls = MER2023Dataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer/mer2023.yaml",
    }
    
    
@registry.register_builder("caer")
class CAERBuilder(MERDatasetBuilder):
    train_dataset_cls = CAERDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer/caer.yaml",
    }
    
@registry.register_builder("dfew")
class DFEWBuilder(MERDatasetBuilder):
    train_dataset_cls = DFEWDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer/dfew.yaml",
    }
    
@registry.register_builder("mc_eiu")
class MCEIUBuilder(MERDatasetBuilder):
    train_dataset_cls = MCEIUDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer/mc_eiu.yaml",
    }
    
@registry.register_builder("e3")
class E3Builder(MERDatasetBuilder):
    train_dataset_cls = E3Dataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer/e3.yaml",
    }
    
@registry.register_builder("iemocap")
class IEMOCAPBuilder(MERDatasetBuilder):
    train_dataset_cls = IEMOCAPDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer/iemocap.yaml",
    }
    
@registry.register_builder("mafw")
class MAFWBuilder(MERDatasetBuilder):
    train_dataset_cls = MAFWDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer/mafw.yaml",
    }
    
    
@registry.register_builder("bold")
class BOLDBuilder(MERDatasetBuilder):
    train_dataset_cls = BOLDDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer/bold.yaml",
    }
    
# CH-SIMS_v2_s
@registry.register_builder("ch_sims_v2_s")
class CHSIMEV2SBuilder(MERDatasetBuilder):
    train_dataset_cls = CHSIMEV2SDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer/ch_sims_v2_s.yaml",
    }
    
# CMU-MOSEI
@registry.register_builder("cmu_mosei")
class CMUMOSEISSBuilder(MERDatasetBuilder):
    train_dataset_cls = CMUMOSEISDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer/cmu_mosei.yaml",
    }
    
# CMU-MOSI
@registry.register_builder("cmu_mosi")
class CMUMOSISSBuilder(MERDatasetBuilder):
    train_dataset_cls = CMUMOSISDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer/cmu_mosi.yaml",
    }

@registry.register_builder("meld")
class MELDBuilder(MERDatasetBuilder):
    train_dataset_cls = MELDDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer/meld.yaml",
    } 
    
    
######################### MER Datasets Finetune ############################################ 
@registry.register_builder("mer2023_ft")
class MER2023Builder(MERDatasetBuilder):
    train_dataset_cls = MER2023DatasetFT
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer_ft/mer2023.yaml",
    }
    
@registry.register_builder("caer_ft")
class CAERBuilder(MERDatasetBuilder):
    train_dataset_cls = CAERDatasetFT
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer_ft/caer.yaml",
    }
    
@registry.register_builder("dfew_ft")
class DFEWBuilder(MERDatasetBuilder):
    train_dataset_cls = DFEWDatasetFT
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer_ft/dfew.yaml",
    }
    
@registry.register_builder("mc_eiu_ft")
class MCEIUBuilder(MERDatasetBuilder):
    train_dataset_cls = MCEIUDatasetFT
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer_ft/mc_eiu.yaml",
    }
    
@registry.register_builder("e3_ft")
class E3Builder(MERDatasetBuilder):
    train_dataset_cls = E3DatasetFT
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer_ft/e3.yaml",
    }
    
@registry.register_builder("iemocap_ft")
class IEMOCAPBuilder(MERDatasetBuilder):
    train_dataset_cls = IEMOCAPDatasetFT
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer_ft/iemocap.yaml",
    }
    
@registry.register_builder("mafw_ft")
class MAFWBuilder(MERDatasetBuilder):
    train_dataset_cls = MAFWDatasetFT
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer_ft/mafw.yaml",
    }
    
    
@registry.register_builder("bold_ft")
class BOLDBuilder(MERDatasetBuilder):
    train_dataset_cls = BOLDDatasetFT
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer_ft/bold.yaml",
    }
    
# CH-SIMS_v2_s
@registry.register_builder("ch_sims_v2_s_ft")
class CHSIMEV2SBuilder(MERDatasetBuilder):
    train_dataset_cls = CHSIMEV2SDatasetFT
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer_ft/ch_sims_v2_s.yaml",
    }
    
# CMU-MOSEI
@registry.register_builder("cmu_mosei_ft")
class CMUMOSEISSBuilder(MERDatasetBuilder):
    train_dataset_cls = CMUMOSEIDatasetFT
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer_ft/cmu_mosei.yaml",
    }
    
# CMU-MOSI
@registry.register_builder("cmu_mosi_ft")
class CMUMOSISSBuilder(MERDatasetBuilder):
    train_dataset_cls = CMUMOSIDatasetFT
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer_ft/cmu_mosi.yaml",
    }

@registry.register_builder("meld_ft")
class MELDBuilder(MERDatasetBuilder):
    train_dataset_cls = MELDDatasetFT
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mer_ft/meld.yaml",
    } 
    
    
@registry.register_builder("memo_bench")
class MEMOBenchBuilder(BaseDatasetBuilder):
    train_dataset_cls = MEMOBenchDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/memo_bench/label.yaml",
    }

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()
        split = "train"

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets