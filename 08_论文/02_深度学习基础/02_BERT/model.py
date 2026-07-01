"""
BERT: Pre-training of Deep Bidirectional Transformers
====================================================
论文: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
      (Devlin et al., NAACL 2019)
核心贡献: 提出 Masked Language Model (MLM) + Next Sentence Prediction (NSP)，
         预训练+微调范式，双向 Transformer Encoder。
架构: 仅 Encoder（双向 Self-Attention），BERT-base: L=12, H=768, A=12, 110M参数
代码结构:
  1. BERTEmbedding - Token + Segment + Position 三种嵌入之和
  2. PreLNEncoderLayer / PreLNEncoder - Pre-LN Transformer Encoder
  3. MLMHead - 遮盖语言模型预测头
  4. NSPHead - 下一句预测头
  5. BERT - 完整预训练模型

关键设计:
  - [CLS] token: 借鉴原 Transformer，最终隐藏状态作为序列聚合表示
  - [SEP] token: 分隔不同句子
  - Segment Embeddings: 区分 A/B 句，后被多模态模型继承区分图像/文本 token
  - 80%-10%-10% mask策略: 缓解预训练-微调 mismatch

与 [[../01_Attention_Is_All_You_Need/Attention Is All You Need|Transformer]] 的关系:
  BERT 只用 Encoder 部分，使用双向注意力（无 causal mask）
与 [[../03_GPT/GPT|GPT]] 的区别: BERT 双向 vs GPT 单向因果注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# 1. BERT 输入嵌入 —— Token + Segment + Position
# ==============================================================================
class BERTEmbedding(nn.Module):
    """
    BERT 的输入表示由三种嵌入求和构成:

    输入序列格式: [CLS] token1 token2 ... [SEP] sentence_B tokens [SEP]

    1. Token Embedding: WordPiece 分词，词表30,000
    2. Segment Embedding: 可学习的分段嵌入，区分 A/B 句
       - A句所有token用 segment_id=0，B句用 segment_id=1
       - 这个设计后被多模态模型继承，用于区分图像/文本 token
    3. Position Embedding: 可学习的位置编码，max_len=512
       （BERT 使用可学习位置编码而非正弦编码，与ViT一致）
    加入 [CLS] token（借鉴原始 Transformer），其最终隐藏状态作为分类表示。
    """

    def __init__(self, vocab_size: int = 30000, hidden_size: int = 768,
                 max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.segment_embed = nn.Embedding(2, hidden_size)  # 0 = A句, 1 = B句
        self.position_embed = nn.Embedding(max_len, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self,
                input_ids: torch.Tensor,        # (batch, seq_len)
                segment_ids: torch.Tensor = None  # (batch, seq_len)
                ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # Token 嵌入
        token_emb = self.token_embed(input_ids)  # (batch, seq_len, hidden)

        # Segment 嵌入
        if segment_ids is None:
            segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long,
                                      device=input_ids.device)
        segment_emb = self.segment_embed(segment_ids)

        # Position 嵌入（可学习）
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_emb = self.position_embed(positions)

        # 三种嵌入求和 + LayerNorm + Dropout
        embeddings = token_emb + segment_emb + position_emb
        return self.dropout(self.norm(embeddings))


# ==============================================================================
# 2. Pre-LN Transformer Encoder Layer
# ==============================================================================
class PreLNEncoderLayer(nn.Module):
    """
    Pre-LN 设计的 Encoder 层（BERT 实际使用的是 Pre-LN）

    与 [[../01_Attention_Is_All_You_Need/Attention Is All You Need|原始 Transformer]] 不同:
    - 原始 Transformer 使用 Post-LN: LayerNorm(x + Sublayer(x))
    - BERT/现代 Transformer 使用 Pre-LN: x + Sublayer(LayerNorm(x))
    - Pre-LN 的优势: 梯度可在残差路径上自由流动，训练更稳定

    子层顺序:
      x → LayerNorm → Multi-Head Self-Attn → Dropout → + x (残差)
      x → LayerNorm → FFN               → Dropout → + x (残差)
    """

    def __init__(self, hidden_size: int = 768, num_heads: int = 12,
                 ff_size: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)  # Pre-LN: 在注意力之前
        self.norm2 = nn.LayerNorm(hidden_size)  # Pre-LN: 在 FFN 之前

        # FFN: d_model → d_ff (4x) → d_model, 使用 GELU 激活
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.GELU(),  # BERT 使用 GELU（而非 Transformer 的 ReLU）
            nn.Linear(ff_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-LN Self-Attention
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=attention_mask)
        x = residual + self.dropout(attn_out)

        # Pre-LN FFN
        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + self.dropout(ffn_out)

        return x


# ==============================================================================
# 3. Pre-LN Encoder (堆叠多层)
# ==============================================================================
class PreLNEncoder(nn.Module):
    """堆叠 L 层 PreLNEncoderLayer（BERT-base: L=12）"""

    def __init__(self, num_layers: int = 12, hidden_size: int = 768,
                 num_heads: int = 12, ff_size: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            PreLNEncoderLayer(hidden_size, num_heads, ff_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


# ==============================================================================
# 4. Masked Language Model Head
# ==============================================================================
class MLMHead(nn.Module):
    """
    Masked Language Model 预测头

    为什么选 15% mask？
    - 太低(5%): 预训练信号不足，收敛慢
    - 太高(25%+): 破坏太多上下文信息，难以学习有效表示

    80%-10%-10% 策略: 缓解预训练-微调 mismatch
    - 80% 替换为 [MASK] — 让模型学习预测缺失词
    - 10% 替换为随机词 — 避免模型只依赖 [MASK] 标记
    - 10% 保持不变 — 保持对正常输入的理解

    原因: [MASK] token 在微调阶段不存在，如果预训练时只用 [MASK]，
    模型会对它产生特定响应模式，造成分布不匹配。
    """

    def __init__(self, hidden_size: int = 768, vocab_size: int = 30000):
        super().__init__()
        # 简单的线性层 + GELU + LayerNorm + 输出投影
        self.transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        # 通常将 decoder 的权重与 token embedding 绑定
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, seq_len, hidden_size)
        x = self.transform(hidden_states)
        # 输出 logits: (batch, seq_len, vocab_size)
        return self.decoder(x) + self.bias


# ==============================================================================
# 5. Next Sentence Prediction Head
# ==============================================================================
class NSPHead(nn.Module):
    """
    Next Sentence Prediction 预测头

    目标: 判断句子 B 是否是句子 A 的下一句
    - 50% 正例: 实际相邻的句子对
    - 50% 负例: 随机抽取的句子对

    使用 [CLS] token 的最终隐藏状态进行分类（2 分类）。

    为何后来被废弃？
    RoBERTa 发现移除 NSP 后性能反而提升:
    1. NSP 任务太简单——可通过话题词重叠等表面特征完成
    2. 负例（随机抽取）判断的是"话题是否一致"而非"逻辑是否连续"
    3. ALBERT 的 SOP (Sentence Order Prediction) 是更好的替代
    """

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 2)  # 二分类: IsNext / NotNext

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        # cls_hidden: (batch, hidden_size) — 即 [CLS] token 的最终隐藏状态
        return self.classifier(cls_hidden)


# ==============================================================================
# 6. 完整 BERT 模型
# ==============================================================================
class BERT(nn.Module):
    """
    完整 BERT 预训练模型

    BERT-base: L=12, H=768, A=12, FFN=3072, 总参数约 110M
    BERT-large: L=24, H=1024, A=16, FFN=4096, 总参数约 340M
    """

    def __init__(self,
                 vocab_size: int = 30000,
                 hidden_size: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 ff_size: int = 3072,
                 max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, hidden_size, max_len, dropout)
        self.encoder = PreLNEncoder(num_layers, hidden_size, num_heads, ff_size, dropout)
        self.mlm_head = MLMHead(hidden_size, vocab_size)
        self.nsp_head = NSPHead(hidden_size)

    def forward(self,
                input_ids: torch.Tensor,
                segment_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
          mlm_logits: (batch, seq_len, vocab_size)  — MLM 预测
          nsp_logits: (batch, 2)                      — NSP 预测
        """
        # 嵌入
        x = self.embedding(input_ids, segment_ids)

        # Encoder
        x = self.encoder(x, attention_mask)

        # MLM 预测: 所有位置都预测（损失只在 masked 位置计算）
        mlm_logits = self.mlm_head(x)

        # NSP 预测: 仅用 [CLS] token
        nsp_logits = self.nsp_head(x[:, 0, :])  # [CLS] 在位置 0

        return mlm_logits, nsp_logits


# ==============================================================================
# 演示代码
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BERT 架构演示 (bidirectional Transformer Encoder)")
    print("=" * 60)

    # 超参数 (BERT-base)
    vocab_size = 30000
    hidden_size = 768
    num_layers = 3      # 演示用 3 层（实际 base 为 12）
    num_heads = 12
    ff_size = 3072

    model = BERT(vocab_size, hidden_size, num_layers, num_heads, ff_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"BERT 参数量: {total_params:,}")

    # 模拟输入: [CLS] A句 [SEP] B句 [SEP]
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    input_ids[:, 0] = 3    # [CLS] token
    input_ids[:, 32] = 4   # [SEP] token

    # Segment IDs: A句=0, B句=1
    segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    segment_ids[:, 33:] = 1  # B句部分

    # 前向传播
    mlm_logits, nsp_logits = model(input_ids, segment_ids)
    print(f"\nMLM logits 形状: {mlm_logits.shape}")
    print(f"  (batch={batch_size}, seq_len={seq_len}, vocab_size={vocab_size})")
    print(f"NSP logits 形状: {nsp_logits.shape}")
    print(f"  (batch={batch_size}, 2) — IsNext/NotNext 二分类")

    # 演示 MLM 训练损失计算
    print("\n--- BERT 预训练任务 ---")
    print("1. MLM (Masked Language Model):")
    print("   - 随机 mask 15% token，预测被 mask 的词")
    print("   - 80%→[MASK], 10%→随机词, 10%→保持不变")
    print("   - 原因: 缓解预训练-微调 mismatch")
    print("2. NSP (Next Sentence Prediction):")
    print("   - 判断句子B是否是句子A的下一句")
    print("   - 50%正例/50%负例")
    print("   - 后续被 RoBERTa 移除，ALBERT 用 SOP 替代")

    print("\n--- BERT vs GPT vs Transformer ---")
    print("  BERT:  Encoder-only, 双向注意力, 预训练+微调, 适合理解任务")
    print("  GPT:   Decoder-only, 因果注意力, 自回归生成, 适合生成任务")
    print("  Transformer: Encoder-Decoder, 完整架构, 适合序列转换(翻译)")
