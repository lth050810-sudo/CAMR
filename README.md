# WikiBigEdit 150K：最新版 CAMR K8 全参数实验

采用用户指定的八个原生时间段，每段 18,750 条，合计 150,000 条。沿用旧 100K 的确定性 tag/relation 分层抽样、token 有效性检查、去重和源行排序。优先选择 new/update；不足时使用该时间段 QA 完整但原始 tag 为空的记录补足，保留其空标签。第二段在 token 过滤前仅有 18,369 条有效唯一标注记录，故需要此补足规则。旧协议对早期标注源事实的去重规则保留，另排除此前已选中的无标签事实。数量及具体代码差异见 `source_changes.json`、`protocol/data_audit.json`。

方法冻结为 2026-10-08 已验证的 CAMR V5 `more_learned`：K8 历史预算，MERGE 最多六条学习选择、至少两条均匀历史，候选不足时均匀补齐；ADD 均匀历史。IGNORE 关闭。单原问法视图，每条到达一步，role_mean 当前/历史 20%/80%，lr=2e-6，seed=42。Qwen3-8B-Base 的全部 8,190,735,360 个参数可训练，FP32 主参数及 AdamW 状态，BF16 计算。

8 张 H100 80GB 共同训练一个模型，FSDP SHARD_GRAD_OP，按事实视图分配逻辑批次，梯度按完整逻辑损失精确加权。训练步数和 K8 不因设备数量改变。关闭 activation checkpointing，评估批量 512，使用 KV cache。

独立压力测试覆盖长样本、八个阶段和 512 条评估批量。正式任务在 128 步保存状态、重启并恢复同一条训练轨迹，验证权重指纹衔接。随后每 25K 保存一次完整 checkpoint，包括模型、优化器、策略、记忆、随机状态。128、25K、50K、75K、100K、125K、150K 均保留，不自动清理。

checkpoint 根目录：`/mnt/yy2607004_hs/engram_checkpoints/wikibigedit_camr_k8_150k_20261009/`。

训练先提交 150K checkpoint，随后独立评估 Original、Rephrase、Persona、Locality。主表沿用旧 100K 的歧义/冲突排除名单和每阶段 64 条 Locality 面板；同时保存全部有效配对行统计，报告 Match、Exact、正确数和有效分母。Locality 是对原 loc_ans 的 gold-answer Match，不是编辑前后输出一致率。评测 QA 不参与训练、记忆检索或奖励。

`status.json` 为总状态；`training_*/progress.json` 为训练进度；`evaluation/progress.json` 为评估进度；`gpu_telemetry.jsonl` 记录每张卡的显存、利用率与功耗。日志在 `logs/`。评估按批原子保存，可从已完成批次继续。结束时自动生成 `RESULTS.md`、`results_150k.json`、`results_150k.csv`，原始逐题预测位于 `evaluation/<axis>/batch*.json`。

本配置曾使用前 5K 开发数据选择；150K 流包含这些数据。150K 的实际结果不能称为完全独立的验证结果。数据来源、哈希、八阶段数量、缺失评测行及 locality 重叠均记录在 `data_audit.json`。

启动入口：`.venv-qwen3/bin/python runs/wikibigedit_camr_k8_150k_20261009/supervise.py`。监督进程持有八张卡的协作锁。失败保留日志和 checkpoint；正常结束且四项结果通过校验后才标记整个实验 complete。
