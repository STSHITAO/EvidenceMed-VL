from __future__ import annotations

from typing import Tuple

import gradio as gr

from medical_rag.config import Settings
from medical_rag.pipeline import MedicalRAGPipeline


def _format_evidence_md(answer) -> str:
    rows = ["### 检索证据"]
    for i, item in enumerate(answer.evidence, start=1):
        rows.append(
            f"{i}. **来源**: `{item.source}` | **chunk**: `{item.chunk_index}` | **score**: `{item.score:.4f}`\n"
            f"   {item.content[:450]}{'...' if len(item.content) > 450 else ''}"
        )
    return "\n\n".join(rows)


def build_demo(settings: Settings) -> gr.Blocks:
    pipeline = MedicalRAGPipeline(settings)

    def infer(image_path: str, question: str) -> Tuple[str, str]:
        if not image_path:
            return "请先上传医学影像。", ""
        if not question.strip():
            return "请先输入问题。", ""

        result = pipeline.ask(image_path=image_path, question=question)
        return result.answer, _format_evidence_md(result)

    with gr.Blocks(title="Medical-RAG") as demo:
        gr.Markdown(
            """
            # Medical-RAG: 医疗影像循证问答系统
            上传医学影像并提问，系统将先检索医学指南，再进行图文联合推理。
            """
        )

        with gr.Row():
            image = gr.Image(type="filepath", label="上传医学影像")
            question = gr.Textbox(lines=5, label="请输入问题", placeholder="例如：该影像中可疑病灶特征是什么？下一步建议检查是什么？")

        with gr.Row():
            submit = gr.Button("开始分析", variant="primary")
            clear = gr.Button("清空")

        answer_box = gr.Textbox(label="系统回答", lines=12)
        evidence_box = gr.Markdown(label="证据")

        submit.click(infer, inputs=[image, question], outputs=[answer_box, evidence_box])
        clear.click(lambda: (None, "", "", ""), outputs=[image, question, answer_box, evidence_box])

        gr.Markdown(
            """
            **免责声明**
            - 本系统仅用于科研与教学演示，不替代执业医生诊断。
            - 高风险病例请结合临床表现、化验检查与专科会诊综合判断。
            """
        )

    return demo
