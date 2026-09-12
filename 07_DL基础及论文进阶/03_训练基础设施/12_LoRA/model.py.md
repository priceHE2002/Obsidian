---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# LoRA (Low-Rank Adaptation) 完整实现 - 基于 [[LoRA]] (Hu et al., ICLR 2022) - 代码实现

> 本文档包含 `model.py` 的 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
LoRA (Low-Rank Adaptation) 完整实现 - 基于 [[LoRA]] (Hu et al., ICLR 2022)

冻结预训练权重，在 Q/K/V 投影旁路注入可训练的低秩分解矩阵 B×A。
初始化策略: A 用 Kaiming 初始化，B 为零初始化，使训练从 ΔW=0 开始。
支持 merge_and_unload（推理合并）和多 LoRA 切换。

核心公式: h = W_0 x + (α/r) · BA x
其中 B ∈ R^{d×r}, A ∈ R^{r×k}, r ≪ min(d,k)

与 [[Adapter]] 的关键区别: LoRA 推理时可合并进入原权重，实现零延迟。
与 [[QLoRA]] 的关系: QLoRA = LoRA + NF4 量化 + 双重量化。

参考:
- [[LoRA]] - 原始论文 (ICLR 2022)
- [[QLoRA]] - 量化扩展 (NeurIPS 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """
    LoRA 线性层：在冻结的原有权重旁路注入低秩分解矩阵。

    训练时:  output = W_0 @ x + (α/r) * (B @ A @ x)
    推理时(合并后): output = (W_0 + (α/r) * B @ A) @ x  (零延迟)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
    ):
        """
        Args:
            in_features: 输入维度 k
            out_features: 输出维度 d
            r: LoRA 秩（低秩分解的中间维度）
            lora_alpha: 缩放因子 α，调 α ≈ 调学习率
            lora_dropout: LoRA 路径的 dropout
            merge_weights: 是否在前向时自动合并
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r  # 缩放因子，使 α 变更时无需重调学习率

        # ---- 冻结的原始权重 ----
        self.linear = nn.Linear(in_features, out_features, bias=False)
        # 冻结原有权重，不接收梯度
        self.linear.weight.requires_grad = False

        # ---- 可训练的 LoRA 矩阵 ----
        # A ∈ R^{r × k}: 下投影 (Kaiming 初始化，打破对称性)
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        # B ∈ R^{d × r}: 上投影 (零初始化，保证初始 ΔW = 0)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.merged = False  # 跟踪当前是否已合并

        self._init_lora_weights()

    def _init_lora_weights(self):
        """
        LoRA 初始化策略:
        - A 用 Kaiming 均匀分布初始化 → 保证梯度传播
        - B 零初始化 → 训练开始时 ΔW = B@A = 0，不影响预训练模型输出
        这与 [[Adapter]] 的近零初始化有本质区别：LoRA 严格从零开始。
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., in_features]
        Returns:
            [..., out_features]
        """
        # 原始权重的前向（冻结，无梯度）
        result = self.linear(x)

        if self.merged or self.r == 0:
            return result

        # LoRA 旁路: dropout → A·x → B·(A·x)
        x_drop = self.lora_dropout(x)
        lora_out = F.linear(x_drop, self.lora_A)  # [..., r]
        lora_out = F.linear(lora_out, self.lora_B)  # [..., out_features]

        return result + lora_out * self.scaling

    def merge(self):
        """
        将 LoRA 权重合并入原始权重: W = W_0 + (α/r) * B@A。

        合并后推理与全量微调模型计算图完全相同，零额外延迟。
        切换任务: W ← W - B1@A1 + B2@A2
        """
        if not self.merged:
            # 计算合并权重
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.linear.weight.data += delta_w
            self.merged = True

    def unmerge(self):
        """
        从原始权重中减去 LoRA 贡献，用于多 LoRA 切换。
        先 unmerge 当前 LoRA，再 merge 新 LoRA。
        """
        if self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.linear.weight.data -= delta_w
            self.merged = False

    def merge_and_unload(self):
        """
        合并 LoRA 权重并返回一个新的 nn.Linear（丢弃 LoRA 参数）。
        用于部署场景——最小化推理时的额外开销。
        """
        self.merge()
        merged_linear = nn.Linear(self.in_features, self.out_features, bias=False)
        merged_linear.weight.data = self.linear.weight.data.clone()
        return merged_linear


class MultiHeadLoRA(nn.Module):
    """
    对 Transformer 的 Q/K/V 投影同时应用 LoRA。

    论文关键发现:
    - 对 Q 和 V 应用 LoRA (r=4) 是性价比最高的配置
    - 仅对 Q 应用会损失性能
    - 宁可减小 r 也要覆盖更多权重类型

    应用于 LLaMA 等模型时，Q/K/V 通常合并为一个 Linear(d_model, 3*d_model)，
    此处的实现将其拆分为独立的 Q/K/V LoRA 模块。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        apply_q: bool = True,
        apply_k: bool = True,
        apply_v: bool = True,
    ):
        """
        Args:
            d_model: 模型隐藏维度
            num_heads: 注意力头数
            r: LoRA 秩
            lora_alpha: 缩放因子
            lora_dropout: dropout
            apply_q/k/v: 是否对对应投影应用 LoRA
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        head_dim = d_model // num_heads
        self.qkv_dim = num_heads * head_dim

        self.q_lora = None
        self.k_lora = None
        self.v_lora = None

        if apply_q:
            self.q_lora = LoRALinear(d_model, self.qkv_dim, r, lora_alpha, lora_dropout)
        if apply_k:
            self.k_lora = LoRALinear(d_model, self.qkv_dim, r, lora_alpha, lora_dropout)
        if apply_v:
            self.v_lora = LoRALinear(d_model, self.qkv_dim, r, lora_alpha, lora_dropout)

    def forward(self, x: torch.Tensor, q_proj, k_proj, v_proj):
        """
        前向传播：基模型投影 + LoRA 旁路。

        Args:
            x: [batch, seq, d_model]
            q_proj, k_proj, v_proj: 冻结的原始 Q/K/V 投影
        Returns:
            q, k, v: 各有 LoRA 修正的投影结果
        """
        q = q_proj(x)
        k = k_proj(x)
        v = v_proj(x)

        if self.q_lora is not None:
            q = q + self.q_lora(x)
        if self.k_lora is not None:
            k = k + self.k_lora(x)
        if self.v_lora is not None:
            v = v + self.v_lora(x)

        return q, k, v


class LoRASwitcher:
    """
    多 LoRA 管理器：支持在多个任务的 LoRA 权重之间热切换。

    典型工作流:
    - 任务 A 训练完后保存 lora_A, lora_B
    - 任务 B 训练时加载新的 lora_A, lora_B
    - 推理时通过 switcher 在 A/B 之间切换

    这对 VLA 多机器人场景特别有用：每个机器人形态有独立 LoRA 模块。
    """

    def __init__(self, lora_modules: dict):
        """
        Args:
            lora_modules: {name: LoRALinear} 字典，包含所有 LoRA 层
        """
        self.lora_modules = lora_modules
        # 缓存不同任务的 LoRA 权重: {task_name: {module_name: (A, B)}}
        self.task_cache: dict[str, dict[str, tuple]] = {}

    def save_task(self, task_name: str):
        """保存当前 LoRA 权重到缓存"""
        self.task_cache[task_name] = {}
        for name, module in self.lora_modules.items():
            self.task_cache[task_name][name] = (
                module.lora_A.data.clone(),
                module.lora_B.data.clone(),
            )

    def load_task(self, task_name: str):
        """从缓存加载指定任务的 LoRA 权重"""
        if task_name not in self.task_cache:
            raise KeyError(f"任务 '{task_name}' 未缓存。可用: {list(self.task_cache.keys())}")
        for name, module in self.lora_modules.items():
            a, b = self.task_cache[task_name][name]
            module.lora_A.data.copy_(a)
            module.lora_B.data.copy_(b)

    def list_tasks(self) -> list:
        return list(self.task_cache.keys())


# ============================================================
# 演示代码
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("LoRA 实现演示")
    print("参考: Hu et al. ICLR 2022")
    print("=" * 70)

    batch, seq, d_model, num_heads, r = 2, 16, 768, 12, 8
    head_dim = d_model // num_heads

    # --- 单层 LoRA ---
    print("\n[1] LoRA 线性层基础演示")
    x = torch.randn(batch, seq, d_model)
    lora_linear = LoRALinear(d_model, d_model, r=r, lora_alpha=16.0)

    # 测试前向
    y = lora_linear(x)
    print(f"    输入形状: {x.shape}")
    print(f"    输出形状: {y.shape}")
    print(f"    原始权重冻结: {not lora_linear.linear.weight.requires_grad}")
    print(f"    LoRA A 可训练: {lora_linear.lora_A.requires_grad}")
    print(f"    LoRA B 可训练: {lora_linear.lora_B.requires_grad}")
    lora_params = lora_linear.lora_A.numel() + lora_linear.lora_B.numel()
    orig_params = d_model * d_model
    print(f"    LoRA 参数占比: {lora_params/orig_params*100:.2f}%")

    # 测试初始时 ΔW = 0（因为 B 零初始化）
    delta_w = lora_linear.lora_B @ lora_linear.lora_A
    print(f"    ΔW 初始范数: {delta_w.norm().item():.8f} (应为 0)")

    # --- Merge/Unmerge ---
    print("\n[2] Merge / Unmerge 演示")
    # 先给 LoRA 一些非零值模拟训练
    with torch.no_grad():
        lora_linear.lora_A.data = torch.randn_like(lora_linear.lora_A) * 0.01
        lora_linear.lora_B.data = torch.randn_like(lora_linear.lora_B) * 0.01

    y_before = lora_linear(x)
    lora_linear.merge()
    y_merged = lora_linear(x)
    print(f"    合并前/后输出一致: {torch.allclose(y_before, y_merged, atol=1e-6)}")

    lora_linear.unmerge()
    y_unmerged = lora_linear(x)
    print(f"    unmerge 后输出恢复: {torch.allclose(y_before, y_unmerged, atol=1e-6)}")

    # --- merge_and_unload ---
    print("\n[3] merge_and_unload 演示")
    merged_nn = lora_linear.merge_and_unload()
    y_simple = merged_nn(x)
    print(f"    简化层输出形状: {y_simple.shape}")
    print(f"    简化层类型: {type(merged_nn).__name__} (普通 nn.Linear)")

    # --- 多 LoRA 切换 ---
    print("\n[4] 多任务 LoRA 切换")
    modules = {
        "q_proj": LoRALinear(d_model, d_model, r=8),
        "v_proj": LoRALinear(d_model, d_model, r=8),
    }
    switcher = LoRASwitcher(modules)

    # 模拟任务 A 训练
    with torch.no_grad():
        modules["q_proj"].lora_A.data = torch.randn_like(modules["q_proj"].lora_A) * 0.1
    switcher.save_task("task_a")
    output_a = modules["q_proj"](x)

    # 模拟任务 B 训练
    with torch.no_grad():
        modules["q_proj"].lora_A.data = torch.randn_like(modules["q_proj"].lora_A) * 0.2
    switcher.save_task("task_b")
    output_b = modules["q_proj"](x)

    # 切换回任务 A
    switcher.load_task("task_a")
    output_a_restored = modules["q_proj"](x)
    print(f"    可用任务: {switcher.list_tasks()}")
    print(f"    任务 A 输出恢复一致: {torch.allclose(output_a, output_a_restored)}")

    print("\n" + "=" * 70)
    print("LoRA 关键公式: h = W_0 x + (α/r) · B A x")
    print("实际配置: r=8, alpha=16 → scaling=2.0")
    print("与 Adapter 对比: LoRA 推理零延迟 (可 merge), Adapter 有 ~20% 延迟增加")
    print("=" * 70)

```
