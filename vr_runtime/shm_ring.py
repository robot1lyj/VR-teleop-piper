"""SharedMemory 环形缓冲（ring buffer）协议。

目标：把“实时控制链路”(90Hz IK/MIT 下发) 与 “录制/写盘链路”(30Hz 相机+数据集) 彻底解耦。

核心原则：
1) 只在 SharedMemory 中传输“小数据”（关节/夹爪/时间戳/握持状态），禁止传图像等大 ndarray。
2) 读写采用 seqlock（写入时 seq 为奇数，完成后改为偶数），读端无锁重试，保证不会读到撕裂数据。
3) 单调时间戳统一使用 ``time.monotonic_ns()``（纳秒），用于严格对齐“图像采集时间”与“控制/测量时间”。

本文件是协议的唯一真相（single source of truth）：Control Daemon 写入，Qt Recorder 只读。
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

MAGIC_RING = b"PVRRING1"  # 8 bytes
MAGIC_STATUS = b"PVRSTAT1"  # 8 bytes
VERSION = 1


def _next_odd(value: int) -> int:
    """返回 >= value 的下一个奇数，用于 seqlock 写入开始。"""

    return value + 1 if value % 2 == 0 else value + 2


@dataclass(frozen=True)
class RingLayout:
    """环形缓冲的内存布局（固定）。"""

    capacity: int

    # 记录的 payload：sample_seq(uint64) + t_ns(uint64) + meta(uint32) + pad(uint32) + q(7*float32)
    _payload_struct: struct.Struct = struct.Struct("<QQII7f")

    # slot：seq(uint32)+pad(uint32)+payload
    _slot_prefix_struct: struct.Struct = struct.Struct("<II")

    # header：seq(uint32)+pad(uint32)+magic(8)+version(uint32)+capacity(uint32)+record_size(uint32)+write_index(uint32)+r1(uint32)+pad2(uint32)+write_seq(uint64)+r2(16)
    _header_struct: struct.Struct = struct.Struct("<II8sIIIIIIQ16s")

    @property
    def header_size(self) -> int:
        return self._header_struct.size  # 64

    @property
    def payload_size(self) -> int:
        return self._payload_struct.size  # 52

    @property
    def slot_size(self) -> int:
        return self._slot_prefix_struct.size + self.payload_size  # 8 + 52 = 60

    @property
    def shm_size(self) -> int:
        return self.header_size + self.capacity * self.slot_size


class ShmRingWriter:
    """SharedMemory ring writer（单写多读）。"""

    def __init__(self, shm: SharedMemory, layout: RingLayout):
        self.shm = shm
        self.layout = layout
        self.buf = shm.buf

    @classmethod
    def create_or_replace(cls, name: str, capacity: int) -> "ShmRingWriter":
        """创建（或替换）一个 ring 共享内存段。"""

        layout = RingLayout(capacity=capacity)
        try:
            old = SharedMemory(name=name, create=False)
        except FileNotFoundError:
            old = None
        if old is not None:
            try:
                old.close()
            finally:
                try:
                    old.unlink()
                except FileNotFoundError:
                    pass

        shm = SharedMemory(name=name, create=True, size=layout.shm_size)
        writer = cls(shm=shm, layout=layout)
        writer._init_header()
        writer._zero_slots()
        return writer

    def _init_header(self) -> None:
        # 初始化 header（seq 置 0，magic/version 固定）
        self.layout._header_struct.pack_into(
            self.buf,
            0,
            0,  # seq
            0,  # pad
            MAGIC_RING,
            VERSION,
            int(self.layout.capacity),
            int(self.layout.slot_size),
            0,  # write_index
            0,  # r1
            0,  # pad2
            0,  # write_seq
            b"\x00" * 16,
        )

    def _zero_slots(self) -> None:
        start = self.layout.header_size
        self.buf[start : start + self.layout.capacity * self.layout.slot_size] = b"\x00" * (
            self.layout.capacity * self.layout.slot_size
        )

    def close(self) -> None:
        self.shm.close()

    def unlink(self) -> None:
        self.shm.unlink()

    def _read_header(self) -> Tuple[int, int]:
        """读 header，返回 (write_index, write_seq)。"""

        header = self.layout._header_struct
        while True:
            seq1 = struct.unpack_from("<I", self.buf, 0)[0]
            if seq1 % 2 == 1:
                continue
            fields = header.unpack_from(self.buf, 0)
            seq2 = struct.unpack_from("<I", self.buf, 0)[0]
            if seq1 == seq2 and seq2 % 2 == 0:
                magic = fields[2]
                version = int(fields[3])
                if magic != MAGIC_RING or version != VERSION:
                    raise RuntimeError(f"SharedMemory ring header 不匹配: {magic!r} v{version}")
                return int(fields[6]), int(fields[9])

    def _write_header(self, write_index: int, write_seq: int) -> None:
        hdr_seq = struct.unpack_from("<I", self.buf, 0)[0]
        start = _next_odd(int(hdr_seq))
        end = start + 1

        # 写入开始（odd）
        struct.pack_into("<I", self.buf, 0, start)
        # 只更新动态字段：write_index/write_seq
        struct.pack_into("<I", self.buf, 28, int(write_index))  # write_index offset
        struct.pack_into("<Q", self.buf, 48, int(write_seq))  # write_seq offset（8 对齐）
        # 写入结束（even）
        struct.pack_into("<I", self.buf, 0, end)

    def append(self, t_ns: int, q: Sequence[float], meta: int = 1) -> None:
        """追加一条记录。

        Args:
            t_ns: 单调时钟时间戳（纳秒）。
            q: 7 维（6 轴 + 夹爪）的 float 序列。
            meta: 元信息位（bit0: valid）。
        """

        if len(q) != 7:
            raise ValueError("q 必须是长度为 7 的序列（6 轴 + 夹爪）")

        write_index, write_seq = self._read_header()
        next_index = (write_index + 1) % self.layout.capacity
        next_seq = write_seq + 1

        slot_off = self.layout.header_size + next_index * self.layout.slot_size
        slot_seq = struct.unpack_from("<I", self.buf, slot_off)[0]
        start = _next_odd(int(slot_seq))
        end = start + 1

        # 1) 标记写入开始：seq=odd
        struct.pack_into("<I", self.buf, slot_off, start)
        # 2) 清 pad
        struct.pack_into("<I", self.buf, slot_off + 4, 0)
        # 3) 写 payload
        self.layout._payload_struct.pack_into(
            self.buf,
            slot_off + 8,
            int(next_seq),
            int(t_ns),
            int(meta),
            0,
            *[float(v) for v in q],
        )
        # 4) 标记写入完成：seq=even
        struct.pack_into("<I", self.buf, slot_off, end)

        # 更新 header 指针
        self._write_header(next_index, next_seq)


class ShmRingReader:
    """SharedMemory ring reader（只读）。"""

    def __init__(self, shm: SharedMemory, layout: RingLayout):
        self.shm = shm
        self.layout = layout
        self.buf = shm.buf

    @classmethod
    def attach(cls, name: str) -> "ShmRingReader":
        shm = SharedMemory(name=name, create=False)
        # 读一次 header 拿到 capacity/slot_size，校验 magic/version
        header_struct = RingLayout(capacity=1)._header_struct
        fields = header_struct.unpack_from(shm.buf, 0)
        magic = fields[2]
        version = int(fields[3])
        if magic != MAGIC_RING or version != VERSION:
            shm.close()
            raise RuntimeError(f"SharedMemory ring header 不匹配: {magic!r} v{version}")
        capacity = int(fields[4])
        slot_size = int(fields[5])
        layout = RingLayout(capacity=capacity)
        if slot_size != layout.slot_size:
            shm.close()
            raise RuntimeError(f"SharedMemory slot_size 不匹配: {slot_size} != {layout.slot_size}")
        return cls(shm=shm, layout=layout)

    def close(self) -> None:
        self.shm.close()

    def _read_header(self) -> Tuple[int, int]:
        header = self.layout._header_struct
        while True:
            seq1 = struct.unpack_from("<I", self.buf, 0)[0]
            if seq1 % 2 == 1:
                continue
            fields = header.unpack_from(self.buf, 0)
            seq2 = struct.unpack_from("<I", self.buf, 0)[0]
            if seq1 == seq2 and seq2 % 2 == 0:
                return int(fields[6]), int(fields[9])

    def _read_slot(self, index: int) -> Optional[Tuple[int, int, int, np.ndarray]]:
        """读取一个 slot，返回 (sample_seq, t_ns, meta, q[7])，失败返回 None。"""

        slot_off = self.layout.header_size + index * self.layout.slot_size
        payload_struct = self.layout._payload_struct
        while True:
            seq1 = struct.unpack_from("<I", self.buf, slot_off)[0]
            if seq1 % 2 == 1:
                return None
            # 复制 payload，避免 writer 写入时读到撕裂 float
            payload_bytes = bytes(self.buf[slot_off + 8 : slot_off + 8 + payload_struct.size])
            seq2 = struct.unpack_from("<I", self.buf, slot_off)[0]
            if seq1 == seq2 and seq2 % 2 == 0:
                sample_seq, t_ns, meta, _, *q_vals = payload_struct.unpack(payload_bytes)
                q_arr = np.asarray(q_vals, dtype=np.float32)
                return int(sample_seq), int(t_ns), int(meta), q_arr

    def get_latest(self) -> Optional[Tuple[int, int, int, np.ndarray]]:
        write_index, _ = self._read_header()
        return self._read_slot(write_index)

    def find_nearest(
        self,
        target_ns: int,
        *,
        max_delta_ns: int,
        max_scan: int = 64,
    ) -> Optional[Tuple[int, int, int, np.ndarray]]:
        """从最新往回扫描，找与 target_ns 最近的记录（绝对误差最小）。"""

        write_index, _ = self._read_header()
        best = None
        best_abs = None
        cap = self.layout.capacity
        for i in range(max_scan):
            idx = (write_index - i) % cap
            item = self._read_slot(idx)
            if item is None:
                continue
            sample_seq, t_ns, meta, q = item
            dt = int(t_ns) - int(target_ns)
            abs_dt = abs(dt)
            if abs_dt <= max_delta_ns and (best_abs is None or abs_dt < best_abs):
                best = item
                best_abs = abs_dt
            # t_ns 只会越来越旧，若已早于 target-max_delta，继续扫描只会更差
            if int(t_ns) < int(target_ns) - int(max_delta_ns):
                break
        return best

    def find_latest_before(
        self,
        target_ns: int,
        *,
        max_lag_ns: int,
        max_scan: int = 64,
    ) -> Optional[Tuple[int, int, int, np.ndarray]]:
        """从最新往回扫描，找“不晚于 target_ns”的最新记录。

        这是录制对齐最常用的策略：在任意时刻，真实系统“生效”的 command 是最后一次下发的 command，
        因此应当选择 ``t_ns <= target_ns`` 的最新样本，避免把“未来的 command”错配给过去的相机帧。
        """

        write_index, _ = self._read_header()
        cap = self.layout.capacity
        for i in range(max_scan):
            idx = (write_index - i) % cap
            item = self._read_slot(idx)
            if item is None:
                continue
            sample_seq, t_ns, meta, q = item
            if int(t_ns) <= int(target_ns):
                lag = int(target_ns) - int(t_ns)
                if lag <= int(max_lag_ns):
                    return sample_seq, t_ns, meta, q
                # 已经早于 target-max_lag，再往前只会更旧
                break
        return None


class ShmStatusWriter:
    """共享内存：发布“当前握持状态”。"""

    _struct = struct.Struct("<II8sIQI32s")  # 64 bytes

    def __init__(self, shm: SharedMemory):
        self.shm = shm
        self.buf = shm.buf

    @classmethod
    def create_or_replace(cls, name: str) -> "ShmStatusWriter":
        try:
            old = SharedMemory(name=name, create=False)
        except FileNotFoundError:
            old = None
        if old is not None:
            try:
                old.close()
            finally:
                try:
                    old.unlink()
                except FileNotFoundError:
                    pass
        shm = SharedMemory(name=name, create=True, size=cls._struct.size)
        writer = cls(shm)
        writer._struct.pack_into(shm.buf, 0, 0, 0, MAGIC_STATUS, VERSION, 0, 0, b"\x00" * 32)
        return writer

    def close(self) -> None:
        self.shm.close()

    def unlink(self) -> None:
        self.shm.unlink()

    def update(self, grip_mask: int, t_ns: Optional[int] = None) -> None:
        """更新握持掩码（bit0=right, bit1=left）。"""

        if t_ns is None:
            t_ns = time.monotonic_ns()

        seq = struct.unpack_from("<I", self.buf, 0)[0]
        start = _next_odd(int(seq))
        end = start + 1

        struct.pack_into("<I", self.buf, 0, start)
        # t_ns offset: 20, grip offset: 28
        struct.pack_into("<Q", self.buf, 20, int(t_ns))
        struct.pack_into("<I", self.buf, 28, int(grip_mask))
        struct.pack_into("<I", self.buf, 0, end)


class ShmStatusReader:
    _struct = ShmStatusWriter._struct

    def __init__(self, shm: SharedMemory):
        self.shm = shm
        self.buf = shm.buf

    @classmethod
    def attach(cls, name: str) -> "ShmStatusReader":
        shm = SharedMemory(name=name, create=False)
        fields = cls._struct.unpack_from(shm.buf, 0)
        magic = fields[2]
        version = int(fields[3])
        if magic != MAGIC_STATUS or version != VERSION:
            shm.close()
            raise RuntimeError(f"SharedMemory status header 不匹配: {magic!r} v{version}")
        return cls(shm)

    def close(self) -> None:
        self.shm.close()

    def read(self) -> Tuple[int, int]:
        """返回 (t_ns, grip_mask)。"""

        while True:
            seq1 = struct.unpack_from("<I", self.buf, 0)[0]
            if seq1 % 2 == 1:
                continue
            _, _, _, _, t_ns, grip_mask, _ = self._struct.unpack_from(self.buf, 0)
            seq2 = struct.unpack_from("<I", self.buf, 0)[0]
            if seq1 == seq2 and seq2 % 2 == 0:
                return int(t_ns), int(grip_mask)


def pack_q7(right_q: Sequence[float], left_q: Sequence[float]) -> np.ndarray:
    """将双臂 q 拼成 (14,) float32（右臂在前，左臂在后）。"""

    if len(right_q) != 7 or len(left_q) != 7:
        raise ValueError("right_q/left_q 必须长度为 7")
    return np.asarray([*right_q, *left_q], dtype=np.float32)
