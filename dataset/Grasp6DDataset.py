import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from pytorchse3.se3 import se3_log_map
import zipfile
import multiprocessing as mp
from utils import IOStream
import time
import sys

MAX_WIDTH = 0.202

class Grasp6DDataset():
    @staticmethod
    def create_sample_mappings(dataset_path, log_dir, is_train, num_workers):
        logger = IOStream(os.path.join(log_dir, "run.log"))
        
        try:
            pc_zip = zipfile.ZipFile(f'{dataset_path}/pc.zip')
        except Exception as e:
            logger.cprint(f'error in opening pc zip file: {e}')
            sys.exit(1)
        
        filenames = sorted(pc_zip.namelist())
        if is_train:
            filenames = filenames[:int(len(filenames)*4/5)]
        else:
            filenames = filenames[int(len(filenames)*4/5):]
        filenames_split = np.array_split(filenames, num_workers)
        pc_zip.close()

        
        processes = []
        queue = mp.Queue()
        for i in range(num_workers):
            p = mp.Process(
                target=Grasp6DDataset.create_sample_mappings_worker,
                args=(
                    i,
                    queue,
                    log_dir,
                    dataset_path,
                    filenames_split[i].tolist()
                )
            )
            p.start()
            processes.append(p)
            logger.cprint(f'process {i} started mapping dataset') 
            
        mappings = []
        for i in range(num_workers):
            worker_id, data = queue.get()
            if len(data) == 0:
                logger.cprint(f'worker {worker_id} failed')
                sys.exit(0)
            mappings.extend(data)
            logger.cprint(f'received mappings from worker {worker_id}')
            processes[worker_id].join()
            logger.cprint(f'worker {worker_id} joined')
        return mappings

    @staticmethod
    def create_sample_mappings_worker(worker_id, queue, log_dir, dataset_path, filenames):
        logger = IOStream(os.path.join(log_dir, "run.log"))
        
        try:
            pc_zip = zipfile.ZipFile(f'{dataset_path}/pc.zip')
            grasp_prompt_zip = zipfile.ZipFile(f'{dataset_path}/grasp_prompt.zip')
            grasp_zip = zipfile.ZipFile(f'{dataset_path}/grasp.zip')
        except Exception as e:
            logger.cprint(f'error in opening zip files: {e}')
            queue.put((worker_id, []))
            return
                    
        mappings = []
        
        try:
            for filename in filenames:
                filename = os.path.basename(filename)
                scene, _ = os.path.splitext(filename)
                
                try:
                    pc_zip.getinfo(f'pc/{scene}.npy')
                    with grasp_prompt_zip.open(f'grasp_prompt/{scene}.pkl') as f:
                        prompts = pickle.load(f)
                    num_objects = len(prompts)
                    for obj_id in range(num_objects):
                        with grasp_zip.open(f'grasp/{scene}_{obj_id}') as f:
                            Rts, _ = pickle.load(f)
                        mappings.extend([(scene, obj_id, grasp_id) for grasp_id in range(len(Rts))])
                except Exception as e:
                    continue
        except Exception as e:
            logger.cprint(f'error in mapping files: {e}')
        finally:
            pc_zip.close()
            grasp_prompt_zip.close()
            grasp_zip.close()

        queue.put((worker_id, mappings))

class Grasp6DDataset_Train(Dataset):
    """
    data loading class for training without unzipping
    """
    def __init__(self, dataset_path:str, log_dir:str, samples_mapping, num_neg_prompts=4):
        """
        dataset_path (str): path to the dataset
        num_neg_prompts: number of negative prompts used in training
        """
        super().__init__()
        self.num_neg_prompts = num_neg_prompts
        self.logger = IOStream(os.path.join(log_dir, "run.log"))
        self.samples_mappings = samples_mapping
        
        try:
            self.pc_zip = zipfile.ZipFile(f'{dataset_path}/pc.zip')
            self.grasp_prompt_zip = zipfile.ZipFile(f'{dataset_path}/grasp_prompt.zip')
            self.grasp_zip = zipfile.ZipFile(f'{dataset_path}/grasp.zip')
        except Exception as e:
            self.logger.cprint(f'error in opening zip files: {e}')
            sys.exit(1)
            
    def __del__(self):
        self.pc_zip.close()
        self.grasp_prompt_zip.close()
        self.grasp_zip.close()
        self.logger.close()
        
    def __getitem__(self, index):
        """
        index (int): the element index
        """
        scene, obj_id, grasp_id = self.samples_mappings[index]
        
        with self.pc_zip.open(f'pc/{scene}.npy') as f:
            pc = np.load(f)
        with self.grasp_prompt_zip.open(f'grasp_prompt/{scene}.pkl') as f:
            prompts = pickle.load(f)
        with self.grasp_zip.open(f"grasp/{scene}_{obj_id}") as f:
            Rts, ws = pickle.load(f)
        
        pos_prompt = prompts[obj_id]
        neg_prompts = prompts[:obj_id] + prompts[obj_id+1:]
        real_num_neg_prompts = len(neg_prompts)
        if 0 < real_num_neg_prompts < self.num_neg_prompts:
            neg_prompts = neg_prompts + [neg_prompts[-1]] * (self.num_neg_prompts - real_num_neg_prompts)
        elif real_num_neg_prompts == 0:
            neg_prompts = [""] * self.num_neg_prompts
        else:
            neg_prompts = neg_prompts[:self.num_neg_prompts]
        
        return scene, pc, pos_prompt, neg_prompts, Rts[grasp_id], ws[grasp_id]
    
    def __len__(self):
        return len(self.samples_mappings)
    
def Grasp6DDataset_Test(Dataset):
    pass



# class Grasp6DDataset_Test(Dataset):
#     """
#     Data loading class for testing.
#     """
#     def __init__(self, dataset_path: str):
#         """
#         dataset_path (str): path to the dataset
#         """
#         super().__init__()
#         self.dataset_path = dataset_path
#         self._load()

#     def _load(self):
#         self.all_data = []
#         filenames = sorted(os.listdir(f"{self.dataset_path}/pc"))
        
#         print("Processing dataset for testing!")
#         filenames = filenames[int(len(filenames)*4/5):]   # 20% scenes for testing
            
#         for filename in filenames:
#             scene, _ = os.path.splitext(filename)
#             pc = np.load(f"{self.dataset_path}/pc/{scene}.npy")
#             try: 
#                 with open(f"{self.dataset_path}/grasp_prompt/{scene}.pkl", "rb") as f:
#                     prompts = pickle.load(f)
#             except:
#                 continue
#             num_objects = len(prompts)
#             for i in range(num_objects):
#                 try:
#                     with open(f"{self.dataset_path}/grasp/{scene}_{i}.pkl", "rb") as f:
#                         Rts, ws = pickle.load(f)
#                 except:
#                     continue
#                 pos_prompt = prompts[i] # positive prompt
#                 gs = np.concatenate((se3_log_map(torch.from_numpy(Rts)).numpy(), 2*ws[:, None]/MAX_WIDTH-1.0), axis=-1)
#                 self.all_data.append({"scene": scene, "pc": pc, "pos_prompt": pos_prompt, "gs": gs})
        
#         return self.all_data
            
#     def __getitem__(self, index):
#         """
#         index (int): the element index
#         """
#         element = self.all_data[index]
#         return element["scene"], element["pc"], element["pos_prompt"] , element["gs"]      

#     def __len__(self):
#         return len(self.all_data)