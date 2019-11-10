# coding:utf-8
import os
import ctypes
import numpy as np

"""
Train and Test data loader
Modified from OpenKE-PyTorch: https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/data
"""


class TrainDataSampler(object):
    def __init__(self, nbatches, datasampler):
        self.nbatches = nbatches
        self.datasampler = datasampler
        self.batch = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.batch += 1
        if self.batch > self.nbatches:
            raise StopIteration()
        return self.datasampler()

    def __len__(self):
        return self.nbatches


class TrainDataLoader(object):

    def __init__(self, in_path="./", batch_size=None, nbatches=None, threads=8, sampling_mode="normal", bern_flag=0,
                 filter_flag=1, neg_ent=1, neg_rel=0):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "./release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        """argtypes"""
        self.lib.sampling.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64
        ]
        """set essential parameters"""
        self.in_path = in_path
        self.work_threads = threads
        self.nbatches = nbatches
        self.batch_size = batch_size
        self.bern = bern_flag
        self.filter = filter_flag
        self.negative_ent = neg_ent
        self.negative_rel = neg_rel
        self.sampling_mode = sampling_mode
        self.cross_sampling_flag = 0
        self.read()

    def read(self):
        self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
        self.lib.setBern(self.bern)
        self.lib.setWorkThreads(self.work_threads)
        self.lib.randReset()
        self.lib.importTrainFiles()
        self.relTotal = self.lib.getRelationTotal()
        self.entTotal = self.lib.getEntityTotal()
        self.tripleTotal = self.lib.getTrainTotal()

        if self.batch_size == None:
            self.batch_size = self.tripleTotal // self.nbatches
        if self.nbatches == None:
            self.nbatches = self.tripleTotal // self.batch_size
        self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)

        self.batch_h = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_t = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_r = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_y = np.zeros(self.batch_seq_size, dtype=np.float32)
        self.batch_h_addr = self.batch_h.__array_interface__["data"][0]
        self.batch_t_addr = self.batch_t.__array_interface__["data"][0]
        self.batch_r_addr = self.batch_r.__array_interface__["data"][0]
        self.batch_y_addr = self.batch_y.__array_interface__["data"][0]

    def sampling(self):
        self.lib.sampling(
            self.batch_h_addr,
            self.batch_t_addr,
            self.batch_r_addr,
            self.batch_y_addr,
            self.batch_size,
            self.negative_ent,
            self.negative_rel,
            0,
            self.filter,
            0,
            0
        )
        return {
            "batch_h": self.batch_h,
            "batch_t": self.batch_t,
            "batch_r": self.batch_r,
            "batch_y": self.batch_y,
            "mode": "normal"
        }

    def sampling_head(self):
        self.lib.sampling(
            self.batch_h_addr,
            self.batch_t_addr,
            self.batch_r_addr,
            self.batch_y_addr,
            self.batch_size,
            self.negative_ent,
            self.negative_rel,
            -1,
            self.filter,
            0,
            0
        )
        return {
            "batch_h": self.batch_h,
            "batch_t": self.batch_t[:self.batch_size],
            "batch_r": self.batch_r[:self.batch_size],
            "batch_y": self.batch_y,
            "mode": "head_batch"
        }

    def sampling_tail(self):
        self.lib.sampling(
            self.batch_h_addr,
            self.batch_t_addr,
            self.batch_r_addr,
            self.batch_y_addr,
            self.batch_size,
            self.negative_ent,
            self.negative_rel,
            1,
            self.filter,
            0,
            0
        )
        return {
            "batch_h": self.batch_h[:self.batch_size],
            "batch_t": self.batch_t,
            "batch_r": self.batch_r[:self.batch_size],
            "batch_y": self.batch_y,
            "mode": "tail_batch"
        }

    def cross_sampling(self):
        self.cross_sampling_flag = 1 - self.cross_sampling_flag
        # self.cross_sampling_flag = 0 #haha
        if self.cross_sampling_flag == 0:
            return self.sampling_head()
        else:
            return self.sampling_tail()

    """interfaces to set essential parameters"""

    def set_work_threads(self, work_threads):
        self.work_threads = work_threads

    def set_in_path(self, in_path):
        self.in_path = in_path

    def set_nbatches(self, nbatches):
        self.nbatches = nbatches

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.nbatches = self.tripleTotal // self.batch_size

    def set_ent_neg_rate(self, rate):
        self.negative_ent = rate

    def set_rel_neg_rate(self, rate):
        self.negative_rel = rate

    def set_bern_flag(self, bern):
        self.bern = bern

    def set_filter_flag(self, filter):
        self.filter = filter

    """interfaces to get essential parameters"""

    def get_batch_size(self):
        return self.batch_size

    def get_ent_tot(self):
        return self.entTotal

    def get_rel_tot(self):
        return self.relTotal

    def get_triple_tot(self):
        return self.tripleTotal

    def __iter__(self):
        if self.sampling_mode == "normal":
            return TrainDataSampler(self.nbatches, self.sampling)
        else:
            return TrainDataSampler(self.nbatches, self.cross_sampling)

    def __len__(self):
        return self.nbatches


class TestDataSampler(object):

    def __init__(self, data_total, data_sampler):
        self.data_total = data_total
        self.data_sampler = data_sampler
        self.total = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.total += 1
        if self.total > self.data_total:
            raise StopIteration()
        return self.data_sampler()

    def __len__(self):
        return self.data_total


class TestDataLoader(object):

    def __init__(self, in_path="./", sampling_mode='link'):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "./release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        """for link prediction"""
        self.lib.getHeadBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.getTailBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        """for triple classification"""
        self.lib.getTestBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        """set essential parameters"""
        self.in_path = in_path
        self.sampling_mode = sampling_mode
        self.read()

    def read(self):
        self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
        self.lib.randReset()
        self.lib.importTestFiles()
        self.relTotal = self.lib.getRelationTotal()
        self.entTotal = self.lib.getEntityTotal()
        self.testTotal = self.lib.getTestTotal()

        self.test_h = np.zeros(self.entTotal, dtype=np.int64)
        self.test_t = np.zeros(self.entTotal, dtype=np.int64)
        self.test_r = np.zeros(self.entTotal, dtype=np.int64)
        self.test_h_addr = self.test_h.__array_interface__["data"][0]
        self.test_t_addr = self.test_t.__array_interface__["data"][0]
        self.test_r_addr = self.test_r.__array_interface__["data"][0]

        self.test_pos_h = np.zeros(self.testTotal, dtype=np.int64)
        self.test_pos_t = np.zeros(self.testTotal, dtype=np.int64)
        self.test_pos_r = np.zeros(self.testTotal, dtype=np.int64)
        self.test_pos_h_addr = self.test_pos_h.__array_interface__["data"][0]
        self.test_pos_t_addr = self.test_pos_t.__array_interface__["data"][0]
        self.test_pos_r_addr = self.test_pos_r.__array_interface__["data"][0]
        self.test_neg_h = np.zeros(self.testTotal, dtype=np.int64)
        self.test_neg_t = np.zeros(self.testTotal, dtype=np.int64)
        self.test_neg_r = np.zeros(self.testTotal, dtype=np.int64)
        self.test_neg_h_addr = self.test_neg_h.__array_interface__["data"][0]
        self.test_neg_t_addr = self.test_neg_t.__array_interface__["data"][0]
        self.test_neg_r_addr = self.test_neg_r.__array_interface__["data"][0]

    def sampling_lp(self):
        res = []
        self.lib.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
        res.append({
            "batch_h": self.test_h.copy(),
            "batch_t": self.test_t[:1].copy(),
            "batch_r": self.test_r[:1].copy(),
            "mode": "head_batch"
        })
        self.lib.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
        res.append({
            "batch_h": self.test_h[:1],
            "batch_t": self.test_t,
            "batch_r": self.test_r[:1],
            "mode": "tail_batch"
        })
        return res

    def sampling_tc(self):
        self.lib.getTestBatch(
            self.test_pos_h_addr,
            self.test_pos_t_addr,
            self.test_pos_r_addr,
            self.test_neg_h_addr,
            self.test_neg_t_addr,
            self.test_neg_r_addr,
        )
        return [
            {
                'batch_h': self.test_pos_h,
                'batch_t': self.test_pos_t,
                'batch_r': self.test_pos_r,
                "mode": "normal"
            },
            {
                'batch_h': self.test_neg_h,
                'batch_t': self.test_neg_t,
                'batch_r': self.test_neg_r,
                "mode": "normal"
            }
        ]

    """interfaces to get essential parameters"""

    def get_ent_tot(self):
        return self.entTotal

    def get_rel_tot(self):
        return self.relTotal

    def get_triple_tot(self):
        return self.testTotal

    def set_sampling_mode(self, sampling_mode):
        self.sampling_mode = sampling_mode

    def __len__(self):
        return self.testTotal

    def __iter__(self):
        if self.sampling_mode == "link":
            self.lib.initTest()
            return TestDataSampler(self.testTotal, self.sampling_lp)
        else:
            self.lib.initTest()
            return TestDataSampler(1, self.sampling_tc)
