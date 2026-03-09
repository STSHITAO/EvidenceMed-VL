SYSTEM_PROMPT = """
你是医学影像辅助分析系统中的专业助手。
请严格依据输入影像与检索到的医学证据进行回答，禁止编造文献或指南内容。
输出时请遵循：
1) 先给出影像所见与初步分析；
2) 再给出与证据一致的医学解释；
3) 最后给出风险提示与下一步建议（检查/会诊方向）。
若证据不足，请明确说明“不足以支持确定性结论”。
""".strip()


def build_user_prompt(question: str, evidence_blocks: list[str]) -> str:
    evidence_text = "\n\n".join(
        [f"[证据{i + 1}]\n{block}" for i, block in enumerate(evidence_blocks)]
    )
    return (
        f"用户问题：{question}\n\n"
        f"可用医学证据如下，请在回答中显式参考其要点：\n{evidence_text}\n\n"
        "请基于影像与证据进行联合推理并给出专业回答。"
    )
