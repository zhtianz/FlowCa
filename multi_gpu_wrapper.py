import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
from typing import List
from collections import namedtuple
import os

# Currently hard-coded, can change later
if os.environ.get("MASTER_ADDR") is None:
    os.environ["MASTER_ADDR"] = "localhost"
if os.environ.get("MASTER_PORT") is None:
    os.environ["MASTER_PORT"] = "29500"

InferenceData = namedtuple(
    "InferenceData",
    ["data", "status", "num_steps"]  
)


def is_iterable(object) -> bool:
    try:
        iter(object)
        return True
    except:
        return False


def send_dict_to_device(dictionary, device: str):
    for key in dictionary:  
        if torch.is_tensor(dictionary[key]):
            dictionary[key] = dictionary[key].to(device)
        elif is_iterable(dictionary[key]):
            dictionary[key] = [x.to(device) for x in dictionary[key]]
    return dictionary


def cast_dict_to_half(dictionary):
    for key in dictionary:
        if torch.is_tensor(dictionary[key]):
            if dictionary[key].dtype == torch.float32:
                dictionary[key] = dictionary[key].to(torch.float16)
        elif is_iterable(dictionary[key]):
            dictionary[key] = [x.to(torch.float16) for x in dictionary[key] if x.dtype == torch.float32]
    return dictionary


def cast_dict_to_full(dictionary):
    for key in dictionary:
        if torch.is_tensor(dictionary[key]):
            if dictionary[key].dtype == torch.float16:
                dictionary[key] = dictionary[key].to(torch.float32)
        elif is_iterable(dictionary[key]):
            dictionary[key] = [x.to(torch.float32) for x in dictionary[key] if x.dtype == torch.float16]
    return dictionary

def init_model(model_path, state_dict_path: str, device: str) -> nn.Module:
    model = model_path.eval()
    checkpoint = torch.load(state_dict_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        model.load_state_dict(checkpoint)
    else:
        state_dict = checkpoint['model_state_dict'] 
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict) 
    model.to(device)
    return model

def run_inference(
        rank_id: int,
        model_path,
        state_dict_path: str,
        devices: List[str],
        world_size: int,
        input_queues: List[mp.Queue],
        output_queues: List[mp.Queue],
        dtype: torch.dtype = torch.float32,
        num_steps: int = 1,  
):
    device = devices[rank_id]
    input_queue = input_queues[rank_id]
    output_queue = output_queues[rank_id]
    model = init_model(model_path, state_dict_path, device)

    dist.init_process_group("gloo", rank=rank_id, world_size=world_size)
    filter_useless_warnings()

    while True:
        with torch.no_grad():
            try:
                inference_data = input_queue.get()
                if inference_data.status != 1:
                    break
                
                density_data = inference_data.data
                steps = inference_data.num_steps if hasattr(inference_data, 'num_steps') else num_steps
                
                with torch.cuda.amp.autocast(dtype=dtype):
                    probability_map, trajectory, trajectory_times = model.sample(
                        density_condition=density_data, 
                        num_steps=steps
                    )
                    
                    output = probability_map
                
                output = output.to("cpu").to(torch.float32)
                output_queue.put(output)
            except Exception as e:
                output_queue.put(None)
                raise e


class MultiGPUWrapper(nn.Module):
    def __init__(
            self,
            model_path,
            state_dict_path: str,
            devices: List[str],
            fp16: bool = False,
            num_steps: int = 1, 
    ):
        super().__init__()
        self.proc_ctx = None
        self.input_queues = []
        self.output_queues = []
        self.world_size = len(devices)
        self.devices = devices
        self.dtype = torch.float32 if not fp16 else torch.float16
        self.num_steps = num_steps 

        if self.world_size > 1:
            torch.multiprocessing.set_start_method('spawn', force=True)

            self.input_queues, self.output_queues = [], []
            for _ in range(self.world_size):
                self.input_queues.append(mp.Queue())
                self.output_queues.append(mp.Queue())

            self.proc_ctx = mp.spawn(
                run_inference,
                args=(
                    model_path,
                    state_dict_path,
                    devices,
                    self.world_size,
                    self.input_queues,
                    self.output_queues,
                    self.dtype,
                    self.num_steps, 
                ),
                nprocs=self.world_size,
                join=False,
            )
        else:
            self.model = init_model(model_path, state_dict_path, devices[0])

    def forward(self, data_list: List[torch.Tensor]) -> List[torch.Tensor]:
        
        output_list = []
        
        for i, data_batch in enumerate(data_list):
            # data_batch: [batch_size, 1, D, H, W]
            device = self.devices[i]
            
            if self.dtype == torch.float16:
                data_batch = data_batch.to(torch.float16)
            
            if self.world_size > 1:
                input_queue = self.input_queues[i]
                input_queue.put(
                    InferenceData(
                        data=data_batch.to(device), 
                        status=1,
                        num_steps=self.num_steps 
                    )
                )
            else:
               
                with torch.cuda.amp.autocast(dtype=self.dtype), torch.no_grad():
                    
                    probability_map, trajectory, trajectory_times = self.model.sample(
                        density_condition=data_batch.to(device),
                        num_steps=self.num_steps
                    )
                    output_list.append(probability_map.to("cpu").to(torch.float32))
        
        
        if self.world_size > 1:
            for output_queue in self.output_queues[:len(data_list)]:
                output_list.append(output_queue.get())
        
        return output_list

    def __del__(self):
        if self.world_size > 1:
            for input_queue in self.input_queues:
                try:
                    input_queue.put(InferenceData(data=None, status=0, num_steps=0))
                except:
                    pass
            for input_queue, output_queue in zip(self.input_queues, self.output_queues):
                input_queue.close()
                input_queue.join_thread()
                output_queue.close()
                output_queue.join_thread()
            self.proc_ctx.join()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.__del__()
