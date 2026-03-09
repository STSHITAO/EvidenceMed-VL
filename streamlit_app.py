#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
import streamlit as st


def _post_ask(api_base: str, question: str, image_name: str, image_bytes: bytes, image_type: str) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/ask"
    files = {"image": (image_name, image_bytes, image_type or "application/octet-stream")}
    data = {"question": question}
    resp = requests.post(url, data=data, files=files, timeout=600)
    resp.raise_for_status()
    return resp.json()


def _health(api_base: str) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/health"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


st.set_page_config(page_title="Medical-RAG", page_icon=":stethoscope:", layout="wide")
st.title("Medical-RAG 医疗影像循证问答")
st.caption("FastAPI + Streamlit 交互界面")

default_api = os.getenv("MEDICAL_RAG_API_BASE", "http://127.0.0.1:9000")
api_base = st.sidebar.text_input("FastAPI 地址", value=default_api)

if st.sidebar.button("健康检查"):
    try:
        health_data = _health(api_base)
        st.sidebar.success("后端可达")
        st.sidebar.json(health_data)
    except Exception as e:
        st.sidebar.error(f"后端不可达: {e}")

left, right = st.columns([1, 1])
with left:
    image_file = st.file_uploader(
        "上传医学影像",
        type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"],
        accept_multiple_files=False,
    )
with right:
    question = st.text_area(
        "输入医学问题",
        value="请结合影像和检索证据，分析可疑征象并给出下一步建议。",
        height=180,
    )

run = st.button("开始分析", type="primary")
if run:
    if image_file is None:
        st.warning("请先上传医学影像。")
    elif not question.strip():
        st.warning("请先输入问题。")
    else:
        with st.spinner("正在调用后端推理，请稍候..."):
            try:
                payload = _post_ask(
                    api_base=api_base,
                    question=question.strip(),
                    image_name=image_file.name,
                    image_bytes=image_file.getvalue(),
                    image_type=image_file.type or "image/png",
                )
            except Exception as e:
                st.error(f"推理失败: {e}")
                payload = {}

        if payload:
            st.image(image_file, caption="输入影像", use_container_width=True)
            st.subheader("系统回答")
            st.write(payload.get("answer", ""))

            evidence: List[Dict[str, Any]] = payload.get("evidence", [])
            st.subheader("检索证据")
            if evidence:
                rows: List[Dict[str, Any]] = []
                for idx, item in enumerate(evidence, start=1):
                    rows.append(
                        {
                            "rank": idx,
                            "score": round(float(item.get("score", 0.0)), 4),
                            "source": item.get("source", ""),
                            "chunk_index": item.get("chunk_index", -1),
                            "content_preview": str(item.get("content", ""))[:240],
                        }
                    )
                st.dataframe(rows, use_container_width=True, hide_index=True)
                with st.expander("查看完整返回 JSON"):
                    st.json(payload)
            else:
                st.info("未返回检索证据。")

st.markdown("---")
st.caption("免责声明：本系统仅用于科研与教学演示，不替代临床诊断。")

