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

class Ego4d_MQ_V(TextVideoDataset):
    def _load_metadata(self):
        metadata_dir = '/apdcephfs/private_qinghonglin/video_dataset/ego4d/benchmark_splits/mq'

        split_files = {
            'train': 'moments_train.json',
            'val': 'moments_val.json',            # there is no test
            'test': 'moments_val.json'
        }
        target_split_fp = split_files[self.split]

        ann_file = os.path.join(metadata_dir, target_split_fp)
        with open(ann_file) as f:
            anno_json = json.load(f)

        self.metadata = pd.DataFrame(columns=['video_uid', 'clip_uid',
                                              'video_start_sec', 'video_end_sec',
                                              'query'])

        for anno_video in anno_json["videos"]:
            for anno_clip in anno_video["clips"]:
                clip_times = float(anno_clip["video_start_sec"]), float(
                    anno_clip["video_end_sec"]
                )  #
                clip_duration = clip_times[1] - clip_times[0]
                new = pd.DataFrame({
                    'video_uid': anno_video['video_uid'],
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
        caption = sample[4]
        return caption

    def __getitem__(self, item):
        sample = self.metadata.iloc [item]
        video_fp, rel_fp = self._get_video_path(sample)

        fps = 1.87
        try:
            imgs, idxs = self.video_reader(video_fp, sample[5], sample[6],
                                               (sample[3]-sample[2]) * fps * self.video_params['num_frames'])
        except:
            print(f"Warning: missing video file {video_fp}.")

        if self.transforms is not None:
            imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
            imgs = self.transforms(imgs)
            imgs = imgs.transpose(0, 1)  # recover

        meta_arr = {'video_uid': sample[0], 'clip_uid': sample[1], 'dataset': self.dataset_name}
        data = {'video': imgs, 'meta' : meta_arr}
        return data

if __name__ == "__main__":
    split = 'train'
    kwargs = dict(
        dataset_name="Ego4d_MQ_V",
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
        reader='cv2_nlq',
        split=split,
    )
    dataset = Ego4d_MQ_V(**kwargs)
    print(dataset[1])