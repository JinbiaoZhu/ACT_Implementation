import numpy as np
import torch
from h5py import File, Group, Dataset
from torch.utils.data.sampler import Sampler

TARGET_KEY_TO_SOURCE_KEY = {
    'states': 'env_states',
    'observations': 'obs',
    'success': 'success',
    'next_observations': 'obs',
    # 'dones': 'dones',
    # 'rewards': 'rewards',
    'actions': 'actions',
}


def load_content_from_h5_file(file):
    # 如果是 HDF5 文件或组，则递归加载所有子项
    if isinstance(file, (File, Group)):
        return {key: load_content_from_h5_file(file[key]) for key in list(file.keys())}
    elif isinstance(file, Dataset):
        # 如果是数据集，返回数据本身
        return file[()]
    else:
        # 不支持的类型，抛出异常
        raise NotImplementedError(f"Unspported h5 file type: {type(file)}")


def load_traj_hdf5(path, num_traj=None):
    print("载入 HDF5 文件", path)
    file = File(path, 'r')
    keys = list(file.keys())
    if num_traj is not None:
        assert num_traj <= len(keys), f"num_traj: {num_traj} > len(keys): {len(keys)}"
        keys = sorted(keys, key=lambda x: int(x.split('_')[-1]))
        keys = keys[:num_traj]
    ret = {
        key: load_content_from_h5_file(file[key]) for key in keys
    }
    file.close()
    print("数据集已载入")
    return ret


def load_demo_dataset(path, keys=None, num_traj=None, concat=True):
    if keys is None:
        keys = ['observations', 'actions']
    # 加载 HDF5 文件, 返回按轨迹组织的数据
    raw_data = load_traj_hdf5(path, num_traj)

    # 选择一个示例轨迹, 确保所有目标键在数据中
    _traj = raw_data['traj_0']
    for key in keys:
        source_key = TARGET_KEY_TO_SOURCE_KEY[key]
        assert source_key in _traj, f"key: {source_key} not in traj_0: {_traj.keys()}"

    # 从所有轨迹中提取对应的数据
    dataset = {}
    for target_key in keys:
        source_key = TARGET_KEY_TO_SOURCE_KEY[target_key]
        dataset[target_key] = [raw_data[idx][source_key] for idx in raw_data]

        # 处理 numpy 数组类型的数据
        if isinstance(dataset[target_key][0], np.ndarray) and concat:
            if target_key in ['observations', 'states'] and len(dataset[target_key][0]) > len(
                    raw_data['traj_0']['actions']):
                # 观测值和状态比动作多一个时间步，去掉最后一个时间步以对齐
                dataset[target_key] = np.concatenate(
                    [
                        t[:-1] for t in dataset[target_key]
                    ], axis=0
                )
            elif target_key in ['next_observations', 'next_states'] and len(dataset[target_key][0]) > len(
                    raw_data['traj_0']['actions']):
                # 下一个观测值/状态比动作晚一个时间步，去掉第一个时间步以对齐
                dataset[target_key] = np.concatenate(
                    [
                        t[1:] for t in dataset[target_key]
                    ], axis=0
                )
            else:
                # 直接拼接所有轨迹
                dataset[target_key] = np.concatenate(dataset[target_key], axis=0)

            print("载入", target_key, dataset[target_key].shape)

        else:
            # 如果数据不是 NumPy 数组或不需要拼接，则保留原始轨迹列表格式
            print("载入", target_key, len(dataset[target_key]), type(dataset[target_key][0]))

    return dataset


def worker_init_fn(worker_id, base_seed=None):
    """
    The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.
    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    if base_seed is None:
        base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)


class IterationBasedBatchSampler(Sampler):
    """
    Wraps a BatchSampler.
    Resampling from it until a specified number of iterations have been sampled
    References:
        https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                yield batch
                iteration += 1
                if iteration >= self.num_iterations:
                    break

    def __len__(self):
        return self.num_iterations - self.start_iter


if __name__ == "__main__":
    dtst = load_demo_dataset(
        "/home/zjb/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5")
    print(dtst)
