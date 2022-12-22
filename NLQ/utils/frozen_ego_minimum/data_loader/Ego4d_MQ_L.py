import sys
sys.path.append("/apdcephfs/private_qinghonglin/video_codebase/frozen-in-time-main")

from base.base_dataset import TextVideoDataset
import pandas as pd
import os
import json
try:
    sys.path.append("/apdcephfs/private_qinghonglin/video_codebase/frozen-in-time-main/data_loader")
    from transforms import init_transform_dict, init_video_transform_dict
except:
    pass

class Ego4d_MQ_L(TextVideoDataset):
    def _load_metadata(self):
        metadata_dir = '/apdcephfs/private_qinghonglin/video_dataset/ego4d/benchmark_splits/mq'

        split_files = {
            'train': 'moments_train.json',
            'val': 'moments_val.json',            # there is no test
            'test': 'moments_val.json'
        }

        self.metadata = pd.DataFrame(columns=['video_uid', 'query', 'clip_uid',
                                              'video_start_sec', 'video_end_sec', 'video_start_frame',
                                              'video_end_frame'])

        for split in ['train', 'val']:
            target_split_fp = split_files[split]

            ann_file = os.path.join(metadata_dir, target_split_fp)
            with open(ann_file) as f:
                anno_json = json.load(f)

            for anno_video in anno_json["videos"]:
                for anno_clip in anno_video["clips"]:
                    clip_times = float(anno_clip["video_start_sec"]), float(
                        anno_clip["video_end_sec"]
                    )
                    for anno in anno_clip['annotations']:
                        for query in anno["labels"]:
                            clip_duration = clip_times[1] - clip_times[0]
                            if 'label' not in query.keys():
                                continue
                            if query['label'] is None:
                                continue
                            new = pd.DataFrame({
                                'video_uid': anno_video['video_uid'],
                                'query': query["label"],
                                'clip_uid': anno_clip['clip_uid'],
                                'video_start_sec': clip_times[0],
                                'video_end_sec': clip_times[1],
                                'video_start_frame': anno_clip["clip_start_frame"],
                                'video_end_frame': anno_clip["clip_end_frame"]}, index=[1])
                            self.metadata = self.metadata.append(new, ignore_index=True)

        self.transforms = init_video_transform_dict()['test']

    def _get_video_path(self, sample):
        rel_video_fp = sample[0]
        full_video_fp = os.path.join(self.data_dir, rel_video_fp + '.mp4')
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        caption = sample[1]
        return caption

    def __getitem__(self, item):
        sample = self.metadata.iloc [item]
        text = self._get_caption(sample)
        data = {'text': text}
        return data

if __name__ == "__main__":
    split = 'train'
    kwargs = dict(
        dataset_name="Ego4d_MQ_L",
        text_params={
            "input": "text"
        },
        video_params={
        "input_res": 224,
        "num_frames": 4,
        "loading": "lax"
        },
        data_dir="/apdcephfs/private_qinghonglin/video_dataset/ego4d_256/data",
        tsfms=init_video_transform_dict()['test'],
        reader='decord_ego',
        split=split,
    )
    dataset = Ego4d_MQ_L(**kwargs)