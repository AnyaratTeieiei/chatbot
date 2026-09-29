import base64
import logging
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
import time
from gemini_service import create_chat, create_client, embedding_function
from rag import build_history, generate_answer, NOT_FOUND, CONTEXT_STATE_KEY, clean_question
from curriculum_rag import (build_retriever, retrieve, build_rag_prompt,
                            fallback_answer, DATASET_NAME, EMBEDDINGS_NAME,
                            METADATA_NAME)

LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="PELEK Chatbot | ครุศาสตร์ไฟฟ้า",
    page_icon="⚡",
    layout="centered",
    # Streamlit opens it on wide screens and collapses it on narrow screens.
    initial_sidebar_state="auto",
)

APP_DIR = Path(__file__).resolve().parent
BOT_AVATAR_PATH = APP_DIR / "pelek_chatbot_profile.png"


def image_to_data_uri(image_path):
    """แปลงรูปโปรไฟล์ในเครื่องให้แสดงในส่วนหัวของหน้าแอปได้"""
    image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{image_data}"


BOT_AVATAR_DATA_URI = image_to_data_uri(BOT_AVATAR_PATH)

st.markdown(
    """
    <style>
        :root {
            --pelek-orange: #f97316;
            --pelek-orange-dark: #d9530b;
            --pelek-gold: #ffb020;
            --pelek-cream: #fff8ef;
            --pelek-ink: #172033;
            --pelek-muted: #667085;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 4%, rgba(255, 176, 32, 0.22), transparent 28rem),
                radial-gradient(circle at 92% 18%, rgba(249, 115, 22, 0.12), transparent 24rem),
                linear-gradient(180deg, #fffdf9 0%, #fff7ec 100%);
            color: var(--pelek-ink);
            font-family: "Noto Sans Thai", "Leelawadee UI", Tahoma, sans-serif;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 920px;
            padding-top: 2rem;
            padding-bottom: 8rem;
        }

        .pelek-hero {
            position: relative;
            display: flex;
            align-items: center;
            gap: 1.5rem;
            overflow: hidden;
            margin-bottom: 1.35rem;
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(249, 115, 22, 0.2);
            border-radius: 28px;
            background: linear-gradient(125deg, rgba(255, 255, 255, 0.98), rgba(255, 244, 225, 0.96));
            box-shadow: 0 18px 48px rgba(151, 72, 12, 0.12);
        }

        .pelek-hero::after {
            content: "⚡";
            position: absolute;
            right: -1rem;
            bottom: -3.6rem;
            color: rgba(249, 115, 22, 0.08);
            font-size: 10rem;
            transform: rotate(8deg);
        }

        .pelek-avatar-wrap {
            position: relative;
            z-index: 1;
            flex: 0 0 132px;
            width: 132px;
            height: 132px;
            padding: 5px;
            border-radius: 50%;
            background: linear-gradient(145deg, var(--pelek-gold), var(--pelek-orange));
            box-shadow: 0 12px 28px rgba(249, 115, 22, 0.25);
        }

        .pelek-avatar-wrap img {
            display: block;
            width: 100%;
            height: 100%;
            border: 4px solid white;
            border-radius: 50%;
            background: white;
            object-fit: cover;
        }

        .pelek-hero-content {
            position: relative;
            z-index: 1;
        }

        .pelek-kicker {
            margin-bottom: 0.25rem;
            color: var(--pelek-orange-dark);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .pelek-hero h1 {
            margin: 0;
            color: var(--pelek-ink);
            font-size: clamp(1.75rem, 4vw, 2.65rem);
            font-weight: 850;
            line-height: 1.15;
        }

        .pelek-hero h1 span {
            color: var(--pelek-orange);
        }

        .pelek-hero p {
            max-width: 560px;
            margin: 0.5rem 0 0.8rem;
            color: var(--pelek-muted);
            font-size: 0.98rem;
            line-height: 1.65;
        }

        .pelek-status {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.82);
            color: #475467;
            font-size: 0.78rem;
            font-weight: 700;
            box-shadow: inset 0 0 0 1px rgba(249, 115, 22, 0.14);
        }

        .pelek-status-dot {
            width: 0.55rem;
            height: 0.55rem;
            border-radius: 50%;
            background: #22c55e;
            box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.12);
        }

        .pelek-section-label {
            margin: 0 0 0.65rem 0.35rem;
            color: #9a4b13;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.02em;
        }

        [data-testid="stChatMessage"] {
            margin-bottom: 0.8rem;
            padding: 1rem 1.1rem;
            border: 1px solid rgba(249, 115, 22, 0.13);
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.9);
            box-shadow: 0 8px 24px rgba(86, 49, 20, 0.06);
        }

        [data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] {
            overflow: hidden;
            border: 2px solid rgba(249, 115, 22, 0.34);
            border-radius: 50%;
            background: #fff7ed;
            box-shadow: 0 4px 12px rgba(249, 115, 22, 0.12);
        }

        [data-testid="stChatMessageContent"] p {
            color: #344054;
            line-height: 1.75;
        }

        [data-testid="stChatInput"] {
            border: 1px solid rgba(249, 115, 22, 0.32);
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.98);
            box-shadow: 0 12px 35px rgba(139, 70, 18, 0.14);
        }

        [data-testid="stChatInput"]:focus-within {
            border-color: var(--pelek-orange);
            box-shadow: 0 0 0 4px rgba(249, 115, 22, 0.1), 0 12px 35px rgba(139, 70, 18, 0.14);
        }

        [data-testid="stChatInput"] button {
            color: var(--pelek-orange) !important;
        }

        /* ให้แถบป้อนข้อความกว้างเท่าพื้นที่แชท และลบพื้นขาวด้านซ้าย-ขวา */
        [data-testid="stBottom"] {
            background: linear-gradient(
                180deg,
                rgba(255, 248, 239, 0) 0%,
                rgba(255, 248, 239, 0.96) 30%,
                #fff8ef 100%
            ) !important;
        }

        [data-testid="stBottom"] > div {
            background: transparent !important;
        }

        [data-testid="stBottomBlockContainer"] {
            width: calc(100% - 2rem) !important;
            max-width: 920px !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            background: transparent !important;
        }

        [data-testid="stBottomBlockContainer"] [data-testid="stChatInput"] {
            width: 100% !important;
        }

        [data-testid="stExpander"] {
            overflow: hidden;
            border: 1px solid rgba(249, 115, 22, 0.16);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.82);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(165deg, #c94b08 0%, #f97316 58%, #ffad1f 100%);
        }

        [data-testid="stSidebar"] * {
            color: white;
        }

        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            border: 1px solid rgba(255, 255, 255, 0.55);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.16);
            color: white;
            font-weight: 750;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: white;
            background: white;
            color: var(--pelek-orange-dark);
        }

        @media (max-width: 640px) {
            .block-container {
                padding-top: 1rem;
            }

            [data-testid="stBottomBlockContainer"] {
                width: calc(100% - 1rem) !important;
            }

            .pelek-hero {
                align-items: flex-start;
                gap: 1rem;
                padding: 1.1rem;
                border-radius: 22px;
            }

            .pelek-avatar-wrap {
                flex-basis: 88px;
                width: 88px;
                height: 88px;
            }

            .pelek-hero p {
                font-size: 0.88rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Minimal conversational layout inspired by modern AI chat products. This only
# restyles features the app already supports; it does not add imitation controls.
st.markdown(
    """
    <style>
        :root {
            --chat-accent: #a66a18;
            --chat-accent-dark: #70461f;
            --chat-accent-soft: #f4e6c7;
            --chat-text: #332820;
            --chat-muted: #786b5f;
            --chat-line: #e5d9c9;
            --chat-sidebar: #f5f0e8;
            --chat-user: #efe4d2;
            --chat-surface: #fffdf8;
            --chat-surface-soft: #faf5ec;
        }

        .stApp {
            background:
                radial-gradient(circle at 76% 0%, rgba(209, 158, 61, 0.1), transparent 28rem),
                var(--chat-surface);
            color: var(--chat-text);
            font-family: "Noto Sans Thai", "Leelawadee UI", Arial, sans-serif;
        }

        header[data-testid="stHeader"] {
            background: rgba(255, 253, 248, 0.9);
            backdrop-filter: blur(12px);
        }

        .block-container {
            max-width: 860px;
            padding-top: 0.9rem;
            padding-bottom: 8.5rem;
        }

        .pelek-appbar {
            position: sticky;
            top: 0.35rem;
            z-index: 5;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1.4rem;
            padding: 0.55rem 0.2rem 0.8rem;
            border-bottom: 1px solid var(--chat-line);
            background: rgba(255, 253, 248, 0.94);
            backdrop-filter: blur(14px);
        }

        .pelek-appbar-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            min-width: 0;
        }

        .pelek-appbar-avatar {
            width: 38px;
            height: 38px;
            border: 1px solid var(--chat-line);
            border-radius: 50%;
            background: var(--chat-surface);
            object-fit: cover;
        }

        .pelek-appbar-copy {
            min-width: 0;
        }

        .pelek-appbar-title {
            color: var(--chat-text);
            font-size: 1rem;
            font-weight: 700;
            line-height: 1.25;
        }

        .pelek-appbar-subtitle {
            overflow: hidden;
            color: var(--chat-muted);
            font-size: 0.76rem;
            line-height: 1.35;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .pelek-appbar-status {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            flex: 0 0 auto;
            color: var(--chat-muted);
            font-size: 0.76rem;
        }

        .pelek-appbar-status::before {
            width: 0.48rem;
            height: 0.48rem;
            border-radius: 50%;
            background: #b98726;
            content: "";
        }

        [data-testid="stChatMessage"] {
            width: 100%;
            margin: 0 0 0.35rem;
            padding: 0.85rem 0;
            border: 0;
            border-radius: 0;
            background: transparent;
            box-shadow: none;
        }

        [data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] {
            width: 2rem;
            height: 2rem;
            border: 1px solid var(--chat-line);
            background: var(--chat-surface);
            box-shadow: none;
        }

        [data-testid="stChatMessageContent"] {
            max-width: 720px;
            min-width: 0;
            overflow-wrap: anywhere;
            word-break: break-word;
        }

        [data-testid="stChatMessageContent"] p,
        [data-testid="stChatMessageContent"] li {
            color: var(--chat-text);
            font-size: 0.98rem;
            line-height: 1.75;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            box-sizing: border-box;
            width: fit-content;
            max-width: min(72%, 620px);
            margin: 0.45rem 2.75rem 0.7rem auto;
            min-height: 4.15rem;
            padding: 0.82rem 1.3rem;
            border: 1px solid #e3d3b9;
            border-radius: 1.35rem;
            background: var(--chat-user);
            box-shadow: 0 2px 8px rgba(91, 62, 32, 0.05);
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
        [data-testid="stChatMessageAvatarUser"] {
            display: none;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
        [data-testid="stChatMessageContent"] {
            padding: 0;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
        [data-testid="stChatMessageContent"] p {
            margin: 0;
            color: var(--chat-text);
            font-family: "Noto Sans Thai", "Leelawadee UI", Arial, sans-serif;
            font-size: 1.06rem;
            font-weight: 400 !important;
            line-height: 1.7;
            letter-spacing: 0;
            text-align: left;
        }

        [data-testid="stBottom"] {
            background: linear-gradient(
                180deg,
                rgba(255, 253, 248, 0) 0%,
                rgba(255, 253, 248, 0.96) 34%,
                var(--chat-surface) 100%
            ) !important;
        }

        [data-testid="stBottom"] > div {
            background: transparent !important;
        }

        [data-testid="stBottomBlockContainer"] {
            width: calc(100% - 2rem) !important;
            max-width: 820px !important;
            padding: 1rem 0 1.25rem !important;
            background: transparent !important;
        }

        [data-testid="stChatInput"] {
            width: 100% !important;
            min-height: 3.7rem;
            border: 1px solid #ddcfbb;
            border-radius: 1.8rem;
            background: var(--chat-surface-soft);
            box-shadow: 0 4px 18px rgba(83, 58, 33, 0.11);
        }

        [data-testid="stChatInput"]:focus-within {
            border-color: #c39952;
            box-shadow: 0 0 0 3px rgba(196, 151, 76, 0.12), 0 4px 18px rgba(83, 58, 33, 0.12);
        }

        [data-testid="stChatInput"] button {
            color: var(--chat-accent) !important;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid var(--chat-line);
            background:
                radial-gradient(circle at 18% 5%, rgba(213, 166, 80, 0.2), transparent 13rem),
                linear-gradient(165deg, #f9f5ed 0%, var(--chat-sidebar) 48%, #eee3d3 100%);
            box-shadow: 9px 0 26px rgba(73, 50, 28, 0.12);
        }

        [data-testid="stSidebar"] * {
            color: var(--chat-text);
        }

        .pelek-sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            margin: 0.15rem 0 1.15rem;
            padding: 0.72rem;
            border: 1px solid rgba(205, 180, 140, 0.55);
            border-radius: 1rem;
            background: rgba(255, 253, 248, 0.72);
            box-shadow:
                0 7px 18px rgba(83, 58, 33, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(8px);
        }

        .pelek-sidebar-brand img {
            width: 34px;
            height: 34px;
            border: 1px solid var(--chat-line);
            border-radius: 50%;
            background: var(--chat-surface);
            object-fit: cover;
        }

        .pelek-sidebar-brand strong {
            display: block;
            font-size: 1rem;
            line-height: 1.2;
        }

        .pelek-sidebar-brand span {
            color: var(--chat-muted) !important;
            font-size: 0.72rem;
        }

        [data-testid="stSidebar"] .stButton > button {
            justify-content: flex-start;
            width: 100%;
            min-height: 2.8rem;
            padding: 0 0.9rem;
            border: 1px solid #ddcfbb;
            border-radius: 0.85rem;
            background: var(--chat-surface);
            color: var(--chat-text);
            font-weight: 600;
            box-shadow:
                0 5px 12px rgba(83, 58, 33, 0.09),
                inset 0 1px 0 rgba(255, 255, 255, 0.85);
            transition: transform 150ms ease, box-shadow 150ms ease, background 150ms ease;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: #c9aa76;
            background: var(--chat-accent-soft);
            color: var(--chat-accent-dark);
            box-shadow: 0 8px 17px rgba(83, 58, 33, 0.13);
            transform: translateY(-1px);
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] {
            border: 1px solid rgba(205, 180, 140, 0.5);
            border-radius: 0.75rem;
            background: rgba(255, 253, 248, 0.58);
            box-shadow:
                0 4px 12px rgba(83, 58, 33, 0.07),
                inset 0 1px 0 rgba(255, 255, 255, 0.8);
        }

        [data-testid="stExpander"] {
            overflow: hidden;
            border: 1px solid var(--chat-line);
            border-radius: 0.85rem;
            background: var(--chat-surface-soft);
        }

        @media (max-width: 768px) {
            .block-container {
                padding: 0.65rem 1rem 8rem;
            }

            .pelek-appbar-subtitle,
            .pelek-appbar-status {
                display: none;
            }

            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
                max-width: 88%;
                margin-right: 0;
            }

            [data-testid="stBottomBlockContainer"] {
                width: calc(100% - 1rem) !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)



load_dotenv(APP_DIR / ".env")
api_key = os.getenv("GEMINI_API_KEY_INSURVERSE")

if not api_key:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY_INSURVERSE")
    except (FileNotFoundError, KeyError):
        pass

@st.cache_resource(show_spinner=False)
def cached_gemini_client(key):
    return create_client(key)


client = cached_gemini_client(api_key) if api_key else None
embed_content = embedding_function(client)


def clear_history():
    st.session_state["messages"] = [
        {
            "role": "model",
            "content": "สวัสดีค่ะ สอบถามข้อมูลจากเล่มหลักสูตรวิศวกรรมไฟฟ้าและการศึกษา พ.ศ. 2566 ได้เลยค่ะ เช่น แผนการเรียน รายวิชา หน่วยกิต หรือการฝึกปฏิบัติ หากระบุแขนงและชั้นปี จะช่วยค้นหาได้ตรงขึ้นค่ะ",
        }
    ]
    st.session_state.pop(CONTEXT_STATE_KEY, None)
    st.session_state.pop("embedding_retry_after", None)
    st.rerun()


with st.sidebar:
    st.markdown(
        f"""
        <div class="pelek-sidebar-brand">
            <img src="{BOT_AVATAR_DATA_URI}" alt="PELEK">
            <div>
                <strong>PELEK</strong>
                <span>Electrical Education Assistant</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("＋ แชตใหม่", use_container_width=True):
        clear_history()
    st.caption("ผู้ช่วยข้อมูลจากเล่มหลักสูตรวิศวกรรมไฟฟ้าและการศึกษา พ.ศ. 2566")

st.markdown(
    f"""
    <section class="pelek-appbar">
        <div class="pelek-appbar-brand">
            <img class="pelek-appbar-avatar" src="{BOT_AVATAR_DATA_URI}" alt="PELEK">
            <div class="pelek-appbar-copy">
                <div class="pelek-appbar-title">PELEK</div>
                <div class="pelek-appbar-subtitle">ผู้ช่วยข้อมูลหลักสูตรครุศาสตร์อุตสาหกรรม · วิศวกรรมไฟฟ้า</div>
            </div>
        </div>
        <div class="pelek-appbar-status">พร้อมตอบคำถาม</div>
    </section>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "model",
            "content": "สวัสดีค่ะ สอบถามข้อมูลจากเล่มหลักสูตรวิศวกรรมไฟฟ้าและการศึกษา พ.ศ. 2566 ได้เลยค่ะ เช่น แผนการเรียน รายวิชา หน่วยกิต หรือการฝึกปฏิบัติ หากระบุแขนงและชั้นปี จะช่วยค้นหาได้ตรงขึ้นค่ะ",
        }
    ]

file_path = APP_DIR / DATASET_NAME


@st.cache_resource
def cached_retriever(source_bytes, embedding_version, vocabulary_version):
    return build_retriever(source_bytes, APP_DIR)


try:
    embedding_version = tuple(
        (path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else None
        for path in (APP_DIR / EMBEDDINGS_NAME, APP_DIR / METADATA_NAME)
    )
    retriever = cached_retriever(file_path.read_bytes(), embedding_version,
                                 (APP_DIR / "synonyms.json").read_bytes())
except (OSError, ValueError, UnicodeError):
    st.error("ไม่สามารถอ่านชุดข้อมูลหลักสูตรได้ กรุณาตรวจสอบไฟล์ TEE66_Curriculum.md")
    st.stop()

with st.sidebar:
    with st.expander("ข้อมูลระบบสำหรับผู้ดูแล"):
        if retriever["warning"]:
            st.warning(retriever["warning"])
        else:
            st.caption("โหลด embeddings ที่ตรงกับเล่มหลักสูตรแล้ว")
        if not api_key:
            st.warning("ไม่พบ GEMINI_API_KEY_INSURVERSE: ระบบจะแสดงข้อความจากเล่มหลักสูตรที่ค้นพบ")

for msg in st.session_state["messages"]:
    if msg["role"] == "model":
        st.chat_message("assistant", avatar=str(BOT_AVATAR_PATH)).write(msg["content"])
    else:
        st.chat_message("user").write(msg["content"])

if prompt := st.chat_input("พิมพ์คำถามเกี่ยวกับภาควิชาที่นี่..."):
    prompt = clean_question(prompt)
    if not prompt:
        st.warning("กรุณาพิมพ์คำถามก่อนส่งค่ะ")
        st.stop()
    started = time.perf_counter()
    history = build_history(st.session_state["messages"])
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    result = None
    with st.chat_message("assistant", avatar=str(BOT_AVATAR_PATH)):
        placeholder = st.empty()
        grounded_answer = False
        try:
            result = retrieve(prompt, retriever, st.session_state, embed_content)
            answer = result["answer"]
            if answer is None:
                if not api_key:
                    answer = fallback_answer(result, retriever) or NOT_FOUND
                    grounded_answer = answer != NOT_FOUND
                else:
                    generation_started = time.perf_counter()
                    try:
                        # The Gemini SDK applies bounded exponential backoff.
                        # If it still fails, use the sourced curriculum fallback.
                        chat = create_chat(client, history)
                        answer, streaming = generate_answer(
                            chat, build_rag_prompt(result["context"], prompt), placeholder.markdown)
                        result["streaming"] = streaming
                        grounded_answer = answer.strip() != NOT_FOUND
                    finally:
                        result["timings"]["Gemini generation (ms)"] = (time.perf_counter() - generation_started) * 1000
            else:
                grounded_answer = answer != NOT_FOUND
            placeholder.markdown(answer)
            if grounded_answer:
                st.session_state[CONTEXT_STATE_KEY] = result["next_context"]
        except Exception:
            # Replace partial streams and never expose SDK errors or credentials.
            LOGGER.exception("Answer generation failed")
            answer = fallback_answer(result, retriever)
            if answer:
                st.session_state[CONTEXT_STATE_KEY] = result["next_context"]
            else:
                answer = "ระบบยังไม่สามารถสร้างคำตอบได้ในขณะนี้ กรุณาลองใหม่อีกครั้งค่ะ"
            placeholder.markdown(answer)
        st.session_state["messages"].append({"role": "model", "content": answer})
        if result:
            result["timings"]["Total answer time (ms)"] = (time.perf_counter() - started) * 1000
            if result["matches"]:
                with st.expander("ดูแหล่งข้อมูลจากเล่มหลักสูตร"):
                    for index, _ in result["matches"]:
                        st.markdown(retriever["chunks"][index])
                        st.divider()
