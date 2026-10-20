# MP Mooncake HIXL_CS Host 映射交接（实验）

本适配基于 LMCache `0.5.6rc4`、LMCache-Ascend `v0.5.6rc4`，
目标为传统 Mooncake TE `0.3.11.post1` 的 A3 HIXL_CS + RoCE H2H 路径。
它不是新的 HIXL TransferChannel，也不修改 inprocess 的 HCCL/HIXL 实现。

## 已验证事实与当前边界

此前服务器注册对照中，2 GiB 普通分配成功，保留 LMCache 预映射失败，
撤销预映射后成功；400 GiB 普通分配、撤销预映射均成功；
普通分配路径在 1000 GiB 注册成功，1024 GiB 失败在分配阶段。
这些是独立注册程序的结果，不代表本适配已通过端到端测试。

MP 的 `lmcache_driven` 路径使用原始 Host 地址执行异步 H2D/D2H，
再由 kernel 处理 NPU 临时 buffer。第一版保留 `alloc_pinned_ptr`，
在 TE 注册前撤销其预映射，不恢复 LMCache 的 Device 映射记录。
最终由 TE 注销通信内存，再由 allocator 释放 Host 内存。

## 启用方式

显式使用专用入口，接收原有 `lmcache server` 的全部参数：

```bash
export LMCACHE_ASCEND_MP_TE_UNMAP_HOST=1
export MC_TE_PROTOCOL=ascend
export MC_TE_DEVICE=""
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
export ASCEND_BUFFER_POOL=0:0
export HCCL_INTRA_ROCE_ENABLE=1
unset MC_USE_TENT MC_USE_TEV1 MC_FORCE_TCP MC_FORCE_HCA
unset ASCEND_USE_ASYNC_TRANSFER ASCEND_ENABLE_USE_FABRIC_MEM

# 其余地址、端口、L1、coordinator 等参数沿用本机 start-lmcache.sh。
python -m lmcache_ascend.v1.distributed.mooncake_cs_server --help
```

启动器在 LMCache/plugin 完成导入后安装补丁，避免循环导入。
只有实际 MP 配置是 `lmcache_driven`、默认引擎、启用了 `mooncake_te`
P2P 时才允许交接；不支持 `auto`、`engine_driven`、Blend、TENT、
Fabric、中转池或 TE 原生异步模式。未开启环境变量时，原路径不变。
开启开关但仍使用普通 `lmcache server` 时，分配器会拒绝启动并提示正确入口。
不要向 vLLM 的 inprocess 流程传入这个开关。

补丁仅替换 MP Mooncake 实现中的 TE 构造入口。原 `mooncake.engine`
和其他 channel 不变。现有 Python 接口足够，无需为本修改重编 C++ 扩展；
若原来不是 editable 安装，则需要安装更新后的包。

## 软件测试

在本 checkout 中直接运行：

```bash
cd /mnt/sdb/csy/mp_p2p_dev/LMCache-Ascend
python tests/standalone/test_mp_mooncake_registration.py
```

这是标准库 unittest，使用内存/引擎替身验证注册冲突、错误传播、设备上下文、
释放保护、模式约束及安装入口，不需要 NPU。
故意不经过现有 `tests/conftest.py`：该 bootstrap 会尝试切换 LMCache 依赖版本。
这些测试不能替代真实 CANN/TE 数据搬运测试。

## 注册后本机拷贝与正常释放

在原测试容器、相同 Python 环境运行；先停止测试服务，避免资源与端口竞争。
该程序将测试原生 MP `lmcache_memcpy_async`，在池首、中部非对齐位置和末尾
执行 H2D/D2H，逐字节校验，并在成功时走正常注销与释放。

```bash
cd /mnt/sdb/csy/mp_p2p_dev/LMCache-Ascend
python tests/standalone/probe_mp_mooncake_handoff.py \
  --host 192.168.0.110 --nic enp23s0f3 --size-mib 64 --devices 0
```

单卡成功后验证当前 TP8 所用设备：

```bash
cd /mnt/sdb/csy/mp_p2p_dev/LMCache-Ascend
python tests/standalone/probe_mp_mooncake_handoff.py \
  --host 192.168.0.110 --nic enp23s0f3 --size-mib 2048 \
  --devices 0 1 2 3 4 5 6 7
```

再按需将 `--size-mib` 改为 `409600` 验证 400 GiB。
它只拷贝少量样本，不复制整个池，不进行跨机传输，不是带宽基准。
node1 使用 `--host 192.168.0.50` 和实际网卡。

成功输出应包含：

```text
MP Host TE registration succeeded
各 device/offset 的 status: pass
MP Host TE unregistered
{"status": "pass", "stage": "done", ..., "normal_cleanup": true}
```

还需确认进程正常退出、退出码为 0，最终结果之后没有析构错误。
失败时打印具体阶段并结束独立进程，不尝试对未知注册状态进行释放。

## 端到端验收

1. 两端更新相同分支提交，核对 `lmcache_ascend.__file__` 和
   `lmcache_ascend.c_ops.__file__`，避免加载另一份源码/扩展。
2. 使用更新后的 A3 两端 `start-lmcache.sh`，确认 `Host handoff enabled`
   与 `TE registration succeeded` 日志；失败应导致启动失败。
3. 先小池、短前缀，验证 node0 store、node1 远端命中、P2P 完成及正常生成。
   生成成功本身不证明命中，必须结合远端读取与预取日志/指标。
4. 本机拷贝和小规模双机通过后，恢复 400 GiB、TP8 和原 benchmark 负载。
5. 测试结束先停请求并等待两端预取、store/retrieve 完成，再关闭服务，
   检查注销日志和正常退出。

## 生命周期限制

记录注册结果，不把失败当成功；尚未注册成功或注销失败的池禁止 allocator
释放。外层异步 READ 会跟踪 batch，存在未完成或失败/超时的 READ 时，
拒绝注销，避免将“失败状态”当成“DMA 已停止”。失败后不支持原地恢复，
需停止流量、收集日志并重启进程。

**本适配无法观察对端对本机的入站 READ，也没有实现通用 cancel/drain 协议。**
不能声称它支持在有流量时安全热关闭/热重启。首轮仅验证两端停止流量后的正常关闭。
现有 MP 模块关闭顺序也未在本补丁中重排。

TE 成功注销后会删除本适配的通信所有权；LMCache 映射表此前已经移除，
`free_pinned_ptr` 不会再次注销同一份 LMCache 映射。若撤销原映射失败，
即使旧 C++ 接口删除了记录，仍根据非零返回码终止并保留内存。
软件状态不能证明驱动状态，错误路径不自动重试注册/恢复映射。
