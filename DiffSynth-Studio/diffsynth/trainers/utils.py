import imageio, os, torch, warnings, torchvision, argparse, json, re
from safetensors.torch import load_file
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from pdb import set_trace as stx
from typing import Callable, Optional
import random
from glob import glob
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import islice




class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        
        # stx()
        '''
            base_path = 'data/example_video_dataset'
            num_frames = 81
            time_division_factor = 4
            height_division_factor = 16, width_division_factor = 16
            data_file_keys = ['image', 'video']
            metadata_path = 'data/example_video_dataset/metadata.csv'
        '''

        # height 和 weight 是 crop 吗？
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
        
        # metadata 可以根据 base_path 来生成
        # 但是需要有同名的 video 和 txt 文件
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            '''
               当有成对数据的时候, 比如 video1.mp4 和 video1.txt, 那么 metadata 如下:
                       video                                             prompt
                    0  video1.mp4  from sunset to night, a small town, light, hou...
            '''
            # stx()
            print(f"{len(metadata)} lines in metadata.")
            # 将 metadata 按行逐条转换为 list of dict
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            '''
                self.data - list of dict
                self.data[0] - dict_keys(['video', 'prompt']) - {'video': 'video1.mp4', 'prompt': 'from sunset to night, a small town, light, house, river, cyh'}
            '''
            # stx()
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        
        # stx()
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        # 创建一个空的 Pandas 表格
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
    # 先用大的 scale factor 放大，然后再 center crop
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            # 如果 num_frames 不是 time_division_factor 的倍数，则减小到 time_division_factor 的倍数
            # 保证时间维度上的 shape 与模型结构对齐
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    

    def load_video(self, file_path):
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class VideoDataset_Physics(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
        extend=True
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            extend = args.extend
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        self.extend = extend
        
        # stx()
        '''
            base_path = 'data/example_video_dataset'
            num_frames = 81
            time_division_factor = 4
            height_division_factor = 16, width_division_factor = 16
            data_file_keys = ['image', 'video']
            metadata_path = 'data/example_video_dataset/metadata.csv'
        '''

        # height 和 weight 是 crop 吗？
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
        
        # metadata 可以根据 base_path 来生成
        # 但是需要有同名的 video 和 txt 文件
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            '''
               当有成对数据的时候, 比如 video1.mp4 和 video1.txt, 那么 metadata 如下:
                       video                                             prompt
                    0  video1.mp4  from sunset to night, a small town, light, hou...
            '''
            # stx()
            print(f"{len(metadata)} lines in metadata.")
            # 将 metadata 按行逐条转换为 list of dict
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            '''
                self.data - list of dict
                self.data[0] - dict_keys(['video', 'prompt']) - {'video': 'video1.mp4', 'prompt': 'from sunset to night, a small town, light, house, river, cyh'}
            '''
            # stx()
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
            '''
                self.data - list of dict
                self.data[0].keys() - dict_keys(['original', 'parse', 'reason', 'extended', 'physics_related_score', 'physics_label', 'vid'])
            '''
            # stx()
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        
        # stx()
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        # 创建一个空的 Pandas 表格
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
    # 先用大的 scale factor 放大，然后再 center crop
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            # 如果 num_frames 不是 time_division_factor 的倍数，则减小到 time_division_factor 的倍数
            # 保证时间维度上的 shape 与模型结构对齐
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    

    def load_video(self, file_path):
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        try:
            data_cur = self.data[data_id % len(self.data)]
            data = {}
            '''
                data_cur: dict_keys(['original', 'parse', 'reason', 'extended', 'physics_related_score', 'physics_label', 'vid'])
            '''
            video_data_path = os.path.join(self.base_path, f"{data_cur['vid']}.mp4")
            # print("video_data_path: ", video_data_path)
            data["video"] = self.load_data(video_data_path)
            # print(data["video"])
            if not self.extend:
                data["prompt"] = data_cur.get("original", None)
            else:
                data["prompt"] = data_cur.get("extended", None)
            if data.get("video", None) is None or data.get("prompt", None) is None:
                warnings.warn(f"cannot load file data[video] or data[prompt].")
                re_get_data_id = random.randint(0, len(self.data) * self.repeat - 1)
                return self.__getitem__(re_get_data_id)
            return data
        except Exception as e:
            print(f"Error in __getitem__: {e}")
            re_get_data_id = random.randint(0, len(self.data) * self.repeat - 1)
            return self.__getitem__(re_get_data_id)
    

    def __len__(self):
        return len(self.data) * self.repeat


class VideoDataset_Physics_DPO(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path_positive=None, base_path_negative=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
        extend=True
    ):
        if args is not None:
            base_path_positive = args.dataset_base_path_positive
            base_path_negative = args.dataset_base_path_negative
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            extend = args.extend
        
        self.base_path_positive = base_path_positive
        self.base_path_negative = base_path_negative
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        self.extend = extend
        
        # stx()
        '''
            base_path = 'data/example_video_dataset'
            num_frames = 81
            time_division_factor = 4
            height_division_factor = 16, width_division_factor = 16
            data_file_keys = ['image', 'video']
            metadata_path = 'data/example_video_dataset/metadata.csv'
        '''

        # height 和 weight 是 crop 吗？
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
        
        # metadata 可以根据 base_path 来生成
        # 但是需要有同名的 video 和 txt 文件
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path_positive)
            '''
               当有成对数据的时候, 比如 video1.mp4 和 video1.txt, 那么 metadata 如下:
                       video                                             prompt
                    0  video1.mp4  from sunset to night, a small town, light, hou...
            '''
            # stx()
            print(f"{len(metadata)} lines in metadata.")
            # 将 metadata 按行逐条转换为 list of dict
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            '''
                self.data - list of dict
                self.data[0] - dict_keys(['video', 'prompt']) - {'video': 'video1.mp4', 'prompt': 'from sunset to night, a small town, light, house, river, cyh'}
            '''
            # stx()
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
            '''
                self.data - list of dict
                self.data[0].keys() - dict_keys(['original', 'parse', 'reason', 'extended', 'physics_related_score', 'physics_label', 'vid'])
            '''
            # stx()
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        
        # stx()
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        # 创建一个空的 Pandas 表格
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
    # 先用大的 scale factor 放大，然后再 center crop
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            # 如果 num_frames 不是 time_division_factor 的倍数，则减小到 time_division_factor 的倍数
            # 保证时间维度上的 shape 与模型结构对齐
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    

    def load_video(self, file_path):
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        try:
            data_cur = self.data[data_id % len(self.data)]
            data = {}
            '''
                data_cur: dict_keys(['original', 'parse', 'reason', 'extended', 'physics_related_score', 'physics_label', 'vid'])
            '''
            video_data_positive_path = os.path.join(self.base_path_positive, f"{data_cur['vid']}.mp4")
            video_data_negative_path = os.path.join(self.base_path_negative, f"{data_cur['vid']}.mp4")
            # print("video_data_positive_path: ", video_data_positive_path)
            # print("video_data_negative_path: ", video_data_negative_path)
            data["video_positive"] = self.load_data(video_data_positive_path)
            data["video_negative"] = self.load_data(video_data_negative_path)
            # print(data["video"])
            if not self.extend:
                data["prompt"] = data_cur.get("original", None)
            else:
                data["prompt"] = data_cur.get("extended", None)
            if data.get("video_positive", None) is None or data.get("video_negative", None) is None or data.get("prompt", None) is None:
                warnings.warn(f"cannot load file data[video_positive] or data[video_negative] or data[prompt].")
                re_get_data_id = random.randint(0, len(self.data) * self.repeat - 1)
                return self.__getitem__(re_get_data_id)
            return data
        except Exception as e:
            print(f"Error in __getitem__: {e}")
            re_get_data_id = random.randint(0, len(self.data) * self.repeat - 1)
            return self.__getitem__(re_get_data_id)
    

    def __len__(self):
        return len(self.data) * self.repeat


class VideoDataset_Physics_DPO_Reward(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path_positive=None, base_path_negative=None, metadata_path=None, csv_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
        extend=True
    ):
        if args is not None:
            base_path_positive = args.dataset_base_path_positive
            base_path_negative = args.dataset_base_path_negative
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            extend = args.extend
        
        self.base_path_positive = base_path_positive
        self.base_path_negative = base_path_negative
        self.csv_path = csv_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        self.extend = extend
        
        # stx()
        '''
            base_path = 'data/example_video_dataset'
            num_frames = 81
            time_division_factor = 4
            height_division_factor = 16, width_division_factor = 16
            data_file_keys = ['image', 'video']
            metadata_path = 'data/example_video_dataset/metadata.csv'
        '''

        # height 和 weight 是 crop 吗？
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
        
        # metadata 可以根据 base_path 来生成
        # 但是需要有同名的 video 和 txt 文件
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path_positive)
            '''
               当有成对数据的时候, 比如 video1.mp4 和 video1.txt, 那么 metadata 如下:
                       video                                             prompt
                    0  video1.mp4  from sunset to night, a small town, light, hou...
            '''
            # stx()
            print(f"{len(metadata)} lines in metadata.")
            # 将 metadata 按行逐条转换为 list of dict
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            '''
                self.data - list of dict
                self.data[0] - dict_keys(['video', 'prompt']) - {'video': 'video1.mp4', 'prompt': 'from sunset to night, a small town, light, house, river, cyh'}
            '''
            # stx()
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
            '''
                self.data - list of dict
                self.data[0].keys() - dict_keys(['original', 'parse', 'reason', 'extended', 'physics_related_score', 'physics_label', 'vid'])
            '''
            # stx()
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        
        # stx()
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        # 创建一个空的 Pandas 表格
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
    # 先用大的 scale factor 放大，然后再 center crop
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            # 如果 num_frames 不是 time_division_factor 的倍数，则减小到 time_division_factor 的倍数
            # 保证时间维度上的 shape 与模型结构对齐
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    

    def load_video(self, file_path):
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        try:
            data_cur = self.data[data_id % len(self.data)]
            data = {}
            '''
                data_cur: dict_keys(['original', 'parse', 'reason', 'extended', 'physics_related_score', 'physics_label', 'vid'])
            '''
            video_data_positive_path = os.path.join(self.base_path_positive, f"{data_cur['orig_idx']}.mp4")
            video_data_negative_path = os.path.join(self.base_path_negative, f"{data_cur['orig_idx']}.mp4")
            # print("video_data_positive_path: ", video_data_positive_path)
            # print("video_data_negative_path: ", video_data_negative_path)
            data["video_positive"] = self.load_data(video_data_positive_path)
            data["video_negative"] = self.load_data(video_data_negative_path)
            # print(data["video"])
            data['sa_score'] = data_cur['sa_score']
            data['pc_score'] = data_cur['pc_score']
            data['joint_score'] = data_cur['joint_score']

            data["prompt"] = data_cur['caption']
            if data.get("video_positive", None) is None or data.get("video_negative", None) is None or data.get("prompt", None) is None:
                warnings.warn(f"cannot load file data[video_positive] or data[video_negative] or data[prompt].")
                re_get_data_id = random.randint(0, len(self.data) * self.repeat - 1)
                return self.__getitem__(re_get_data_id)
            return data
        except Exception as e:
            print(f"Error in __getitem__: {e}")
            re_get_data_id = random.randint(0, len(self.data) * self.repeat - 1)
            return self.__getitem__(re_get_data_id)
    

    def __len__(self):
        return len(self.data) * self.repeat


class VideoDataset_Physics_DPO_Reward_Group(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path_positive=None, base_path_negative=None, metadata_path=None, csv_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
        extend=True
    ):
        if args is not None:
            base_path_positive = args.dataset_base_path_positive
            base_path_negative = args.dataset_base_path_negative
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            extend = args.extend
        
        self.base_path_positive = base_path_positive
        self.base_path_negative = base_path_negative
        self.csv_path = csv_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        self.extend = extend
        
        # stx()
        '''
            base_path = 'data/example_video_dataset'
            num_frames = 81
            time_division_factor = 4
            height_division_factor = 16, width_division_factor = 16
            data_file_keys = ['image', 'video']
            metadata_path = 'data/example_video_dataset/metadata.csv'
        '''

        # height 和 weight 是 crop 吗？
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
        
        # metadata 可以根据 base_path 来生成
        # 但是需要有同名的 video 和 txt 文件
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path_positive)
            '''
               当有成对数据的时候, 比如 video1.mp4 和 video1.txt, 那么 metadata 如下:
                       video                                             prompt
                    0  video1.mp4  from sunset to night, a small town, light, hou...
            '''
            # stx()
            print(f"{len(metadata)} lines in metadata.")
            # 将 metadata 按行逐条转换为 list of dict
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            '''
                self.data - list of dict
                self.data[0] - dict_keys(['video', 'prompt']) - {'video': 'video1.mp4', 'prompt': 'from sunset to night, a small town, light, house, river, cyh'}
            '''
            # stx()
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
            '''
                self.data - list of dict
                self.data[0].keys() - dict_keys(['original', 'parse', 'reason', 'extended', 'physics_related_score', 'physics_label', 'vid'])
            '''
            # stx()
        else:
            seed_0_metadata_path = metadata_path
            seed_1_metadata_path = metadata_path.replace("seed_0", "seed_1")
            seed_3_metadata_path = metadata_path.replace("seed_0", "seed_3")
            metadata_0 = pd.read_csv(seed_0_metadata_path)
            metadata_1 = pd.read_csv(seed_1_metadata_path)
            metadata_3 = pd.read_csv(seed_3_metadata_path)
            data_0 = [metadata_0.iloc[i].to_dict() for i in range(len(metadata_0))]
            data_1 = [metadata_1.iloc[i].to_dict() for i in range(len(metadata_1))]
            data_3 = [metadata_3.iloc[i].to_dict() for i in range(len(metadata_3))]
            self.data = {
                "seed_0": data_0,
                "seed_1": data_1,
                "seed_3": data_3
            }
        
        # stx()
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        # 创建一个空的 Pandas 表格
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
    # 先用大的 scale factor 放大，然后再 center crop
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            # 如果 num_frames 不是 time_division_factor 的倍数，则减小到 time_division_factor 的倍数
            # 保证时间维度上的 shape 与模型结构对齐
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    

    def load_video(self, file_path):
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        try:
            seed_list = ['seed_0', 'seed_1', 'seed_3']
            seed = random.choice(seed_list)
            # print(f"seed: {seed}")
            data_cur = self.data[seed][data_id % len(self.data[seed])]
            data = {}
            video_data_positive_path = os.path.join(self.base_path_positive, f"{data_cur['orig_idx']}.mp4")
            video_data_negative_path = os.path.join(self.base_path_negative, seed, f"{data_cur['orig_idx']}.mp4")
            # print("video_data_positive_path: ", video_data_positive_path)
            # print("video_data_negative_path: ", video_data_negative_path)
            data["video_positive"] = self.load_data(video_data_positive_path)
            data["video_negative"] = self.load_data(video_data_negative_path)
            # print(data["video"])
            data['sa_score'] = data_cur['sa_score']
            data['pc_score'] = data_cur['pc_score']
            data['joint_score'] = data_cur['joint_score']

            data["prompt"] = data_cur['caption']
            if data.get("video_positive", None) is None or data.get("video_negative", None) is None or data.get("prompt", None) is None:
                warnings.warn(f"cannot load file data[video_positive] or data[video_negative] or data[prompt].")
                re_get_data_id = random.randint(0, len(self.data[seed]) * self.repeat - 1)
                return self.__getitem__(re_get_data_id)
            return data
        except Exception as e:
            print(f"Error in __getitem__: {e}")
            re_get_data_id = random.randint(0, len(self.data[seed]) * self.repeat - 1)
            return self.__getitem__(re_get_data_id)
    

    def __len__(self):
        return len(self.data['seed_0']) * self.repeat




class VideoDataset_Physics_DPO_Mix(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path_positive=None, base_path_negative_implicit=None, base_path_negative_explicit=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
        extend=True
    ):
        if args is not None:
            base_path_positive = args.dataset_base_path_positive
            base_path_negative_implicit = args.dataset_base_path_negative_implicit
            base_path_negative_explicit = args.dataset_base_path_negative_explicit
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            extend = args.extend
            mix_explicit_ratio = args.mix_explicit_ratio
        
        self.base_path_positive = base_path_positive
        self.base_path_negative_implicit = base_path_negative_implicit
        self.base_path_negative_explicit = base_path_negative_explicit
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        self.extend = extend
        self.mix_explicit_ratio = mix_explicit_ratio
        # stx()
        '''
            base_path = 'data/example_video_dataset'
            num_frames = 81
            time_division_factor = 4
            height_division_factor = 16, width_division_factor = 16
            data_file_keys = ['image', 'video']
            metadata_path = 'data/example_video_dataset/metadata.csv'
        '''

        # height 和 weight 是 crop 吗？
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
        
        # metadata 可以根据 base_path 来生成
        # 但是需要有同名的 video 和 txt 文件
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path_positive)
            '''
               当有成对数据的时候, 比如 video1.mp4 和 video1.txt, 那么 metadata 如下:
                       video                                             prompt
                    0  video1.mp4  from sunset to night, a small town, light, hou...
            '''
            # stx()
            print(f"{len(metadata)} lines in metadata.")
            # 将 metadata 按行逐条转换为 list of dict
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            '''
                self.data - list of dict
                self.data[0] - dict_keys(['video', 'prompt']) - {'video': 'video1.mp4', 'prompt': 'from sunset to night, a small town, light, house, river, cyh'}
            '''
            # stx()
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
            '''
                self.data - list of dict
                self.data[0].keys() - dict_keys(['original', 'parse', 'reason', 'extended', 'physics_related_score', 'physics_label', 'vid'])
            '''
            # stx()
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        
        # stx()
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        # 创建一个空的 Pandas 表格
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
    # 先用大的 scale factor 放大，然后再 center crop
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            # 如果 num_frames 不是 time_division_factor 的倍数，则减小到 time_division_factor 的倍数
            # 保证时间维度上的 shape 与模型结构对齐
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    

    def load_video(self, file_path):
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        try:
            data_cur = self.data[data_id % len(self.data)]
            data = {}
            '''
                data_cur: dict_keys(['original', 'parse', 'reason', 'extended', 'physics_related_score', 'physics_label', 'vid'])
            '''
            video_data_positive_path = os.path.join(self.base_path_positive, f"{data_cur['vid']}.mp4")
            video_data_negative_path_implicit = os.path.join(self.base_path_negative_implicit, f"{data_cur['vid']}.mp4")
            video_data_negative_path_explicit = os.path.join(self.base_path_negative_explicit, f"{data_cur['vid']}.mp4")
            data["video_positive"] = self.load_data(video_data_positive_path)

            if random.random() < self.mix_explicit_ratio:
                data["prompt"] = data_cur.get("extended", None)
                data["video_negative"] = self.load_data(video_data_negative_path_explicit)
                # print("using explicit prompt")
                # print("data[prompt]: ", data["prompt"])
                # print("video_data_positive_path: ", video_data_positive_path)
                # print("video_data_negative_path_explicit: ", video_data_negative_path_explicit)
            else:
                data["prompt"] = data_cur.get("original", None)
                data["video_negative"] = self.load_data(video_data_negative_path_implicit)
                # print("using implicit prompt")
                # print("data[prompt]: ", data["prompt"])
                # print("video_data_positive_path: ", video_data_positive_path)
                # print("video_data_negative_path_implicit: ", video_data_negative_path_implicit)

            if data.get("video_positive", None) is None or data.get("video_negative", None) is None or data.get("prompt", None) is None:
                warnings.warn(f"cannot load file data[video_positive] or data[video_negative] or data[prompt].")
                re_get_data_id = random.randint(0, len(self.data) * self.repeat - 1)
                return self.__getitem__(re_get_data_id)
            return data
        except Exception as e:
            print(f"Error in __getitem__: {e}")
            re_get_data_id = random.randint(0, len(self.data) * self.repeat - 1)
            return self.__getitem__(re_get_data_id)
    

    def __len__(self):
        return len(self.data) * self.repeat






class VideoDataset_Physics_DPO_dual(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path_positive=None, base_path_negative=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
        extend=True
    ):
        if args is not None:
            base_path_positive = args.dataset_base_path_positive
            base_path_negative_implicit = args.dataset_base_path_negative_implicit
            base_path_negative_explicit = args.dataset_base_path_negative_explicit
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            extend = args.extend
        
        self.base_path_positive = base_path_positive
        self.base_path_negative_implicit = base_path_negative_implicit
        self.base_path_negative_explicit = base_path_negative_explicit
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        self.extend = extend

        # height 和 weight 是 crop 吗？
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
        
        # metadata 可以根据 base_path 来生成
        # 但是需要有同名的 video 和 txt 文件
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path_positive)
            '''
               当有成对数据的时候, 比如 video1.mp4 和 video1.txt, 那么 metadata 如下:
                       video                                             prompt
                    0  video1.mp4  from sunset to night, a small town, light, hou...
            '''
            # stx()
            print(f"{len(metadata)} lines in metadata.")
            # 将 metadata 按行逐条转换为 list of dict
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            '''
                self.data - list of dict
                self.data[0] - dict_keys(['video', 'prompt']) - {'video': 'video1.mp4', 'prompt': 'from sunset to night, a small town, light, house, river, cyh'}
            '''
            # stx()
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
            '''
                self.data - list of dict
                self.data[0].keys() - dict_keys(['original', 'parse', 'reason', 'extended', 'physics_related_score', 'physics_label', 'vid'])
            '''
            # stx()
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        
        # stx()
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        # 创建一个空的 Pandas 表格
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
    # 先用大的 scale factor 放大，然后再 center crop
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            # 如果 num_frames 不是 time_division_factor 的倍数，则减小到 time_division_factor 的倍数
            # 保证时间维度上的 shape 与模型结构对齐
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    

    def load_video(self, file_path):
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        try:
            data_cur = self.data[data_id % len(self.data)]
            data = {}
            '''
                data_cur: dict_keys(['original', 'parse', 'reason', 'extended', 'physics_related_score', 'physics_label', 'vid'])
            '''
            video_data_positive_path = os.path.join(self.base_path_positive, f"{data_cur['vid']}.mp4")
            video_data_negative_implicit_path = os.path.join(self.base_path_negative_implicit, f"{data_cur['vid']}.mp4")
            video_data_negative_explicit_path = os.path.join(self.base_path_negative_explicit, f"{data_cur['vid']}.mp4")
            # print("video_data_positive_path: ", video_data_positive_path)
            # print("video_data_negative_implicit_path: ", video_data_negative_implicit_path)
            # print("video_data_negative_explicit_path: ", video_data_negative_explicit_path)
            data["video_positive"] = self.load_data(video_data_positive_path)
            data["video_negative_implicit"] = self.load_data(video_data_negative_implicit_path)
            data["video_negative_explicit"] = self.load_data(video_data_negative_explicit_path)

            data["prompt_implicit"] = data_cur["original"]
            data["prompt_explicit"] = data_cur["extended"]
            if data.get("video_positive", None) is None or data.get("video_negative_implicit", None) is None or data.get("video_negative_explicit", None) is None or data.get("prompt_implicit", None) is None or data.get("prompt_explicit", None) is None:
                warnings.warn(f"cannot load file data[video_positive] or data[video_negative_implicit] or data[video_negative_explicit] or data[prompt_implicit] or data[prompt_explicit].")
                re_get_data_id = random.randint(0, len(self.data) * self.repeat - 1)
                return self.__getitem__(re_get_data_id)
            return data
        except Exception as e:
            print(f"Error in __getitem__: {e}")
            re_get_data_id = random.randint(0, len(self.data) * self.repeat - 1)
            return self.__getitem__(re_get_data_id)
    

    def __len__(self):
        return len(self.data) * self.repeat






class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        return model

    def add_lora_to_model_dpo(self, model, target_modules, lora_rank, lora_alpha=None, adapter_name="dpo"):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model, adapter_name=adapter_name)
        return model
    
    
    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict



class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        
    
    def on_step_end(self, loss):
        pass
    
    
    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)



# 按照训练 steps 来保存模型，并支持 resume 训练
class ModelLogger_Steps_Resume:
    """
    Parameters
    ----------
    output_path : str
        目录下会生成：
            ├─ step-0000100.safetensors         # 仅可训练权重
            └─ state-step-0000100/              # optimizer / scheduler / RNG
    remove_prefix_in_ckpt : Optional[str]
        如果想把参数名前缀（如 'pipe.dit.'）去掉再保存，填这里。
    state_dict_converter : Callable
        额外对权重做转换的函数（默认原样返回）。
    save_every_steps : int
        每隔多少 global step 保存一次。
    """
    def __init__(self, output_path: str, *, remove_prefix_in_ckpt=None,
                 state_dict_converter=lambda x: x,
                 save_every_steps: int = 1_000,
                 log_every_steps:  int = 100):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.save_every_steps = save_every_steps
        self.log_every_steps  = log_every_steps
        os.makedirs(self.output_path, exist_ok=True)
        # 日志文件
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.output_path, f"train_{ts}.txt")
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write(f"logging training loss for every {self.log_every_steps} steps\n")

    # ------------ public hook ------------
    def on_step_end(self, accelerator, model, global_step: int, loss: torch.Tensor):
        # 1. 打印 / 写日志
        if global_step % self.log_every_steps == 0:
            if accelerator.is_main_process:
                msg = f"\nstep: {global_step},\tloss: {loss.item():.6f}"
                print(msg, flush=True)
                with open(self.log_file, "a") as f:
                    f.write(msg)

        # 2. 保存 checkpoint
        if global_step % self.save_every_steps == 0:
            self._save(accelerator, model, tag=f"step-{global_step:07d}")
        
        # 确保 save 完成, 进程同步
        accelerator.wait_for_everyone()
    

    # ------------ public hook ------------
    def on_step_end_dual(self, accelerator, model, global_step: int, loss_explicit: torch.Tensor, loss_implicit: torch.Tensor):
        # 1. 打印 / 写日志
        if global_step % self.log_every_steps == 0:
            if accelerator.is_main_process:
                msg = f"\nstep: {global_step},\tloss_explicit: {loss_explicit.item():.6f}, loss_implicit: {loss_implicit.item():.6f}"
                print(msg, flush=True)
                with open(self.log_file, "a") as f:
                    f.write(msg)

        # 2. 保存 checkpoint
        if global_step % self.save_every_steps == 0:
            self._save(accelerator, model, tag=f"step-{global_step:07d}")
        
        # 确保 save 完成, 进程同步
        accelerator.wait_for_everyone()

    # --------------------------------------------------------
    # internal helpers
    # --------------------------------------------------------
    def _save(self, accelerator: Accelerator, model: torch.nn.Module, tag: str):
        """保存权重 + 完整训练状态"""
        # ---- 1. 仅保存可训练权重（LoRA / adapter / full-finetune）----
        state_dict = accelerator.get_state_dict(model)
        state_dict = (
            accelerator.unwrap_model(model)
            .export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
        )
        state_dict = self.state_dict_converter(state_dict)
        accelerator.save(
            state_dict,
            os.path.join(self.output_path, f"{tag}.safetensors"),
            safe_serialization=True,
        )

        # ---- 2. 保存 optimizer / scheduler / GradScaler / RNG 等完整状态 ----
        accelerator.save_state(os.path.join(self.output_path, f"state-{tag}"))



def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0])
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(loss)
                scheduler.step()
        model_logger.on_epoch_end(accelerator, model, epoch_id)


def launch_data_process_task(model: DiffusionTrainingModule, dataset, output_path="./models"):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0])
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    os.makedirs(os.path.join(output_path, "data_cache"), exist_ok=True)
    for data_id, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs = model.forward_preprocess(data)
            inputs = {key: inputs[key] for key in model.model_input_keys if key in inputs}
            torch.save(inputs, os.path.join(output_path, "data_cache", f"{data_id}.pth"))




def _resume_if_possible(
    accelerator: Accelerator,
    model: torch.nn.Module,
    output_path: str,
) -> int:
    """
    若 output_path 下有 step‑xxx.safetensors，则：
        1. 加载权重到 `model`
        2. accelerator.load_state(...) 恢复 optimizer/scheduler/RNG
        3. 返回已经训练到的 global_step
    否则返回 0。
    """
    ckpts = sorted(glob(os.path.join(output_path, "step-*.safetensors")))
    if not ckpts:
        return 0

    latest_ckpt = ckpts[-1]
    global_step = int(re.search(r"step-(\d+)\.safetensors", latest_ckpt).group(1))

    # —— 加载权重（仅可训练层 & 自动补前缀）——
    unwrap_model = accelerator.unwrap_model(model)
    ckpt = load_file(latest_ckpt, device="cpu")

    # 如果保存时去掉了 'pipe.dit.'，这里自动加回来
    sample_key = next(iter(ckpt))
    if not sample_key.startswith("pipe."):
        prefix = "pipe.dit."          # ⇦ 和你 --remove_prefix_in_ckpt 保持一致
        ckpt = {prefix + k: v for k, v in ckpt.items()}
    unwrap_model.load_state_dict(ckpt, strict=False)

    # —— 加载完整训练状态 ——
    state_dir = os.path.join(output_path, f"state-step-{global_step:07d}")
    accelerator.load_state(state_dir)

    accelerator.print(f"✅  Resumed from step {global_step}")
    return global_step




def launch_training_task_steps_resume(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger_Steps_Resume,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    total_training_steps: int = 100000,
    gradient_accumulation_steps: int = 1,
    show_progress: bool = True,
):
    # ----- DataLoader & Accelerator -----
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=lambda batch: batch[0],  # 保持原逻辑
    )

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # ----- Resume -----
    global_step = _resume_if_possible(accelerator, model, model_logger.output_path)
    accelerator.print(f"[Resume] Resumed global_step = {global_step}")

    if global_step >= total_training_steps:
        accelerator.print(
            f"[Resume] Target total_steps={total_training_steps} already reached (global_step={global_step}). Nothing to do."
        )

    # ----- Progress bar -----
    pbar = (
        tqdm(total=total_training_steps, initial=global_step, dynamic_ncols=True)
        if (accelerator.is_local_main_process and show_progress)
        else None
    )

    # ----- Training Loop (step budget controlled) -----
    while global_step < total_training_steps:
        for data in dataloader:
            if global_step >= total_training_steps:
                break

            with accelerator.accumulate(model):
                optimizer.zero_grad(set_to_none=True)
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                global_step += 1
                model_logger.on_step_end(accelerator, model, global_step, loss)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str(f"loss={loss.item():.4f}")

        # dataloader will re-shuffle each epoch because shuffle=True

    if pbar is not None:
        pbar.close()

    accelerator.print(f"[Done] Finished at global_step={global_step}/{total_training_steps}.")




def launch_training_task_steps_resume_dual_dpo(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger_Steps_Resume,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    total_training_steps: int = 100000,
    gradient_accumulation_steps: int = 1,
    show_progress: bool = True,
):
    # ----- DataLoader & Accelerator -----
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=lambda batch: batch[0],  # 保持原逻辑
    )

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # ----- Resume -----
    global_step = _resume_if_possible(accelerator, model, model_logger.output_path)
    accelerator.print(f"[Resume] Resumed global_step = {global_step}")

    if global_step >= total_training_steps:
        accelerator.print(
            f"[Resume] Target total_steps={total_training_steps} already reached (global_step={global_step}). Nothing to do."
        )

    # ----- Progress bar -----
    pbar = (
        tqdm(total=total_training_steps, initial=global_step, dynamic_ncols=True)
        if (accelerator.is_local_main_process and show_progress)
        else None
    )

    # ----- Training Loop (step budget controlled) -----
    while global_step < total_training_steps:
        for data in dataloader:
            if global_step >= total_training_steps:
                break

            with accelerator.accumulate(model):
                optimizer.zero_grad(set_to_none=True)
                # stx()
                # data: dict_keys(['video_positive', 'video_negative', 'prompt'])
                data_implicit = {
                    'video_positive': data['video_positive'],
                    'video_negative': data['video_negative_implicit'],
                    'prompt': data['prompt_implicit']
                }
                data_explicit = {
                    'video_positive': data['video_positive'],
                    'video_negative': data['video_negative_explicit'],
                    'prompt': data['prompt_explicit']
                }
                we, wi = 0.5, 0.5
                loss_explicit = model(data_explicit)
                accelerator.backward(we * loss_explicit)
                loss_implicit = model(data_implicit)
                accelerator.backward(wi * loss_implicit)

                optimizer.step()
                scheduler.step()
                global_step += 1
                model_logger.on_step_end_dual(accelerator, model, global_step, loss_explicit, loss_implicit)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str(f"loss_explicit={loss_explicit.item():.4f}, loss_implicit={loss_implicit.item():.4f}")

        # dataloader will re-shuffle each epoch because shuffle=True

    if pbar is not None:
        pbar.close()

    accelerator.print(f"[Done] Finished at global_step={global_step}/{total_training_steps}.")





def launch_training_task_steps_resume_dual_dpo_self(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger_Steps_Resume,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    total_training_steps: int = 100000,
    gradient_accumulation_steps: int = 1,
    show_progress: bool = True,
):
    # ----- DataLoader & Accelerator -----
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=lambda batch: batch[0],  # 保持原逻辑
    )

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # ----- Resume -----
    global_step = _resume_if_possible(accelerator, model, model_logger.output_path)
    accelerator.print(f"[Resume] Resumed global_step = {global_step}")

    if global_step >= total_training_steps:
        accelerator.print(
            f"[Resume] Target total_steps={total_training_steps} already reached (global_step={global_step}). Nothing to do."
        )

    # ----- Progress bar -----
    pbar = (
        tqdm(total=total_training_steps, initial=global_step, dynamic_ncols=True)
        if (accelerator.is_local_main_process and show_progress)
        else None
    )

    # ----- Training Loop (step budget controlled) -----
    while global_step < total_training_steps:
        for data in dataloader:
            if global_step >= total_training_steps:
                break

            with accelerator.accumulate(model):
                # stx()
                # data: dict_keys(['video_positive', 'video_negative', 'prompt'])

                # 做两次参数更新，一次是 implicit to explicit，一次是 explicit to real
                optimizer.zero_grad(set_to_none=True)
                data_implicit = {
                    'video_positive': data['video_negative_explicit'],
                    'video_negative': data['video_negative_implicit'],
                    'prompt': data['prompt_implicit']
                }

                loss_implicit = model(data_implicit)
                accelerator.backward(loss_implicit)
                optimizer.step()


                # 第二次参数更新，explicit to real
                optimizer.zero_grad(set_to_none=True)
                data_explicit = {
                    'video_positive': data['video_positive'],
                    'video_negative': data['video_negative_explicit'],
                    'prompt': data['prompt_implicit']
                }
                loss_explicit = model(data_explicit)
                accelerator.backward(loss_explicit)
                optimizer.step()

                scheduler.step()
                global_step += 1
                model_logger.on_step_end_dual(accelerator, model, global_step, loss_explicit, loss_implicit)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str(f"loss_explicit={loss_explicit.item():.4f}, loss_implicit={loss_implicit.item():.4f}")

        # dataloader will re-shuffle each epoch because shuffle=True

    if pbar is not None:
        pbar.close()

    accelerator.print(f"[Done] Finished at global_step={global_step}/{total_training_steps}.")




def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--skip_download", default=True, action="store_true", help="Whether to skip downloading models. Set to True if models are already downloaded.")
    parser.add_argument("--extend", default=False, action="store_true", help="Whether to extend the prompt.")
    parser.add_argument("--save_every_steps", type=int, default=1_000, help="Save every steps.")
    parser.add_argument("--log_every_steps", type=int,  default=100, help="Write/print loss every N global steps.")
    parser.add_argument("--total_training_steps", type=int, default=100000, help="Total training steps.")
    parser.add_argument("--dataset_base_path_positive", type=str, default=None, help="Base path of the positive dataset for DPO training.")
    parser.add_argument("--dataset_base_path_negative", type=str, default=None, help="Base path of the negative dataset for DPO training.")
    parser.add_argument("--dataset_base_path_negative_implicit", type=str, default=None, help="Base path of the negative implicit dataset for dual DPO training.")
    parser.add_argument("--dataset_base_path_negative_explicit", type=str, default=None, help="Base path of the negative explicit dataset for dual DPO training.")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="Beta for DPO loss.")
    parser.add_argument("--mix_explicit_ratio", type=float, default=0.0, help="Mix explicit ratio for DPO training.")
    parser.add_argument("--alpha_min", type=float, default=0.5, help="Alpha min for DPO loss.")
    parser.add_argument("--k_alpha", type=float, default=5.0, help="K alpha for DPO loss.")
    parser.add_argument("--b_alpha", type=float, default=0.5, help="B alpha for DPO loss.")
    parser.add_argument("--k_gamma", type=float, default=3.0, help="K gamma for DPO loss.")
    parser.add_argument("--b_gamma", type=float, default=0.5, help="B gamma for DPO loss.")
    parser.add_argument("--lambda_gamma", type=float, default=0.5, help="Lambda gamma for DPO loss.")
    return parser



def flux_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    return parser