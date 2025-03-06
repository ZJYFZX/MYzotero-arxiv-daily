import numpy as np
from sentence_transformers import SentenceTransformer
from paper import ArxivPaper
from datetime import datetime

def rerank_paper(candidate: list[ArxivPaper], corpus: list[dict], model: str = 'avsolatorio/GIST-small-Embedding-v0') -> list[ArxivPaper]:
    encoder = SentenceTransformer(model)
    # 调试打印语料库结构
    print("\n[Debug] 当前语料库内容：")
    print(f"语料库条目数量：{len(corpus)}")
    
    # 打印前3条语料（避免过多输出）
    for i, paper in enumerate(corpus[:3]):
        print(f"\n--- 语料条目 {i+1} ---")
        print("完整结构:", paper)
        
        # 检查关键字段是否存在
        if 'data' in paper:
            print("data字段类型:", type(paper['data']))
            if 'dateAdded' in paper['data']:
                print("添加时间:", paper['data']['dateAdded'])
            else:
                print("警告: data字段缺少dateAdded键")
            if 'abstractNote' in paper['data']:
                print("摘要长度:", len(paper['data']['abstractNote']))
            else:
                print("警告: data字段缺少abstractNote键")
        else:
            print("错误: 条目缺少data字段")
    # 处理空语料库（首次运行可能触发）
    if not corpus:
        print("Warning: 语料库为空，返回顺序。")
        
        for paper in candidate:
            paper.score = np.random.uniform(0, 10)  # 生成 0-10 的随机分
        return sorted(candidate, key=lambda x: x.score, reverse=True)
        
        # return candidate
    
    # 按时间排序语料库
    corpus = sorted(corpus, key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'), reverse=True)
    
    # 时间衰减权重
    time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
    time_decay_weight = time_decay_weight / time_decay_weight.sum()
    
    # 编码特征（强制对齐维度）
    try:
        
        corpus_feature = encoder.encode([paper['data']['abstractNote'] for paper in corpus])
        candidate_feature = encoder.encode([paper.summary for paper in candidate])
        
        # 强制修正维度（关键步骤）
        if corpus_feature.ndim == 1:
            corpus_feature = corpus_feature.reshape(1, -1)  # 单样本升维
        corpus_feature = corpus_feature.reshape(len(corpus), -1)  # 强制为 [N, D]
        candidate_feature = candidate_feature.reshape(len(candidate), -1)  # 强制为 [M, D]
        
        # 维度对齐验证
        if corpus_feature.shape[1] != candidate_feature.shape[1]:
            # 强制填充或截断（确保可运行）
            target_dim = candidate_feature.shape[1]
            corpus_feature = np.pad(corpus_feature, 
                                   ((0, 0), (0, target_dim - corpus_feature.shape[1])), 
                                   mode='constant')  # 填充0到目标维度
            
    except Exception as e:
        print(f"Error: 特征编码失败（{e}），使用随机相似度。")
        sim = np.random.rand(len(candidate), len(corpus))  # 随机矩阵
    else:
        # 正常计算相似度
        sim = encoder.similarity(candidate_feature, corpus_feature)
    
    # 计算得分并排序
    scores = (sim * time_decay_weight).sum(axis=1) * 10
    for s, c in zip(scores, candidate):
        c.score = s.item()
    return sorted(candidate, key=lambda x: x.score, reverse=True)
