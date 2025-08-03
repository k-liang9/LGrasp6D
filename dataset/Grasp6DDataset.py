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

MAX_WIDTH = 0.202   # maximum width of gripper 2F-140


class Grasp6DDataset_Train(Dataset):
    """
    Data loading class for training.
    """
    def __init__(self, dataset_path: str, num_neg_prompts=4):
        """
        dataset_path (str): path to the dataset
        num_neg_prompts: number of negative prompts used in training
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.num_neg_prompts = num_neg_prompts
        self._load()

    def _load(self):
        self.all_data = []
        filenames = sorted(os.listdir(f"{self.dataset_path}/pc"))
        
        print("Processing dataset for training!")
        filenames = filenames[:int(len(filenames)*4/5)]   # 80% scenes for training
            
        for filename in filenames:
            scene, _ = os.path.splitext(filename)
            pc = np.load(f"{self.dataset_path}/pc/{scene}.npy")
            try: 
                with open(f"{self.dataset_path}/grasp_prompt/{scene}.pkl", "rb") as f:
                    prompts = pickle.load(f)
            except:
                continue
            num_objects = len(prompts)
            for i in range(num_objects):
                try:
                    with open(f"{self.dataset_path}/grasp/{scene}_{i}.pkl", "rb") as f:
                        Rts, ws = pickle.load(f)
                except:
                    continue
                pos_prompt = prompts[i] # positive prompt
                neg_prompts = prompts[:i] + prompts[i + 1:]    # negative prompts
                real_num_neg_prompts = len(neg_prompts)
                if 0 < real_num_neg_prompts < self.num_neg_prompts:
                    neg_prompts = neg_prompts + [neg_prompts[-1]] * (self.num_neg_prompts - real_num_neg_prompts)    # pad with last negative prompt
                elif real_num_neg_prompts == 0: # if no negative prompt
                    neg_prompts = [""] * self.num_neg_prompts   # then use empty strings
                else:   # if the real number of negative prompts exceeeds self.num_neg_prompts
                    neg_text = neg_text[:self.num_neg_text]
                
                self.all_data.extend([{"scene": scene, "pc": pc, "pos_prompt": pos_prompt, "neg_prompts": neg_prompts,\
                    "Rt": Rt, "w": 2*w/MAX_WIDTH-1.0} for Rt, w in zip(Rts, ws)])
        
        return self.all_data
            
    def __getitem__(self, index):
        """
        index (int): the element index
        """
        element = self.all_data[index]
        return element["scene"], element["pc"], element["pos_prompt"] , element["neg_prompts"], element["Rt"], element["w"]      

    def __len__(self):
        return len(self.all_data)
    
    
class Grasp6DDataset_Test(Dataset):
    """
    Data loading class for testing.
    """
    def __init__(self, dataset_path: str):
        """
        dataset_path (str): path to the dataset
        """
        super().__init__()
        self.dataset_path = dataset_path
        self._load()

    def _load(self):
        self.all_data = []
        filenames = sorted(os.listdir(f"{self.dataset_path}/pc"))
        
        print("Processing dataset for testing!")
        filenames = filenames[int(len(filenames)*4/5):]   # 20% scenes for testing
            
        for filename in filenames:
            scene, _ = os.path.splitext(filename)
            pc = np.load(f"{self.dataset_path}/pc/{scene}.npy")
            try: 
                with open(f"{self.dataset_path}/grasp_prompt/{scene}.pkl", "rb") as f:
                    prompts = pickle.load(f)
            except:
                continue
            num_objects = len(prompts)
            for i in range(num_objects):
                try:
                    with open(f"{self.dataset_path}/grasp/{scene}_{i}.pkl", "rb") as f:
                        Rts, ws = pickle.load(f)
                except:
                    continue
                pos_prompt = prompts[i] # positive prompt
                gs = np.concatenate((se3_log_map(torch.from_numpy(Rts)).numpy(), 2*ws[:, None]/MAX_WIDTH-1.0), axis=-1)
                self.all_data.append({"scene": scene, "pc": pc, "pos_prompt": pos_prompt, "gs": gs})
        
        return self.all_data
            
    def __getitem__(self, index):
        """
        index (int): the element index
        """
        element = self.all_data[index]
        return element["scene"], element["pc"], element["pos_prompt"] , element["gs"]      

    def __len__(self):
        return len(self.all_data)
    
    
class ZippedGrasp6DDataset_Train(Dataset):
    """
    data loading class for training without unzipping
    """
    def __init__(self, dataset_path:str, log_dir:str, num_neg_prompts=4):
        """
        dataset_path (str): path to the dataset
        num_neg_prompts: number of negative prompts used in training
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.num_neg_prompts = num_neg_prompts
        self.log_dir = log_dir
        self.logger = IOStream(os.path.join(log_dir, "run.log"))
        self._load()
        
    def _load(self):
        self.all_data = []
        
        try:
            pc_zip = zipfile.ZipFile(f'{self.dataset_path}/pc.zip')
        except Exception as e:
            self.logger.cprint(f'error with opening pc zip file: {e}')
            return

        num_processes = len(os.sched_getaffinity(0))
        self.logger.cprint(f'number of cores: {num_processes}')
        filenames = sorted(pc_zip.namelist())
        filenames = filenames[:int(len(filenames)*4/5)]
        filenames_chunks = np.array_split(np.array(filenames), num_processes)
        pc_zip.close()
        
        self.logger.cprint(f'number of files: {len(filenames)}')
        self.logger.cprint("processing dataset for training!")
        
        try:
            processes = []
            queue = mp.Queue()
            filecount_lock = mp.Lock()
            filecount = mp.Value('i', 0)
            # logger_lock = mp.Lock()
            for i in range(num_processes):
                p = mp.Process(
                    target=load_dataset_worker,
                    args=(
                        filenames_chunks[i].tolist(),
                        self.dataset_path,
                        self.num_neg_prompts,
                        self.log_dir,
                        i,
                        (filecount, filecount_lock),
                        queue
                    )
                )
                p.start()
                processes.append(p)
                self.logger.cprint(f'process {i} started')
            
            all_data = [[] for _ in range(num_processes)]
            for i in range(num_processes):
                process_id, data = queue.get()
                if data == []:
                    self.logger.cprint(f'process {i} failed')
                    return
                all_data[process_id] = data
                self.logger.cprint(f'extracted data from process {process_id}')
            
            for i in range(num_processes):
                self.all_data.extend(all_data[i])
                processes[i].join()
                
        except Exception as e:
            self.logger.cprint(f'error in multiprocessing: {e}')
        
        return self.all_data
        
    def __getitem__(self, index):
        """
        index (int): the element index
        """
        element = self.all_data[index]
        return element["scene"], element["pc"], element["pos_prompt"], element["neg_prompts"], element["Rt"], element["w"]
    
    def __len__(self):
        return len(self.all_data)

def log(logger, logger_lock, msg):
    try:
        with logger_lock:
            try:
                logger.cprint(msg)
            except Exception as log_exc:
                print(f"Logging failed: {log_exc}")
    except Exception as lock_exc:
        print(f"Logger lock failed: {lock_exc}")

def load_dataset_worker(filenames, dataset_path, num_neg_prompts, log_dir, worker_id, shared_resources, queue):
    #TODO: investigate purpose of pc_mask
    logger = IOStream(os.path.join(log_dir, "run.log"))
    filecount, filecount_lock = shared_resources

    try:
        pc_zip = zipfile.ZipFile(f'{dataset_path}/pc.zip')
        grasp_prompt_zip = zipfile.ZipFile(f'{dataset_path}/grasp_prompt.zip')
        grasp_zip = zipfile.ZipFile(f'{dataset_path}/grasp.zip')
    except Exception as e:
        logger.cprint(f'error with loading zip files on process {worker_id}: {e}')
        queue.put((worker_id, []))
        return
    
    data = []
    try:
        count = 0
        for filename in filenames:
            filename = os.path.basename(filename)
            scene, _ = os.path.splitext(filename)
            
            try:
                with pc_zip.open(f'pc/{scene}.npy') as f:
                    pc = np.load(f)
            except Exception as e:
                continue

            try:
                with grasp_prompt_zip.open(f'grasp_prompt/{scene}.pkl') as f:
                    prompts = pickle.load(f)
            except Exception as e:
                continue
            
            num_objects = len(prompts)
            for i in range(num_objects):
                try:
                    with grasp_zip.open(f"grasp/{scene}_{i}") as f:
                        Rts, ws = pickle.load(f)
                except Exception as e:
                    continue
                
                pos_prompt = prompts[i]
                neg_prompts = prompts[:i] + prompts[i+1:]
                real_num_neg_prompts = len(neg_prompts)
                if 0 < real_num_neg_prompts < num_neg_prompts:
                    neg_prompts = neg_prompts + [neg_prompts[-1]] * (num_neg_prompts - real_num_neg_prompts)
                elif real_num_neg_prompts == 0:
                    neg_prompts = [""] * num_neg_prompts
                else:
                    neg_prompts = neg_prompts[:num_neg_prompts]
                
                data.extend([{"scene": scene, "pc": pc, "pos_prompt": pos_prompt, "neg_prompts": neg_prompts, "Rt": Rt, "w": 2*w/MAX_WIDTH-1.0} for Rt, w in zip(Rts, ws)])
            
            count += 1
            if count == 10000:
                logger.cprint(f'processed {filecount.value} scenes')
                with filecount_lock:
                    filecount.value += 10000
                count = 0
    finally:
        pc_zip.close()
        grasp_prompt_zip.close()
        grasp_zip.close()

    queue.put((worker_id, data))