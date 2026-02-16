import streamlit as st
import json
import faith_shop
import difflib
import re


# --- DATA LOADING ---
@st.cache_data
def load_data():
    # Note: Ensure this path is correct for your local environment
    try:
        with open('../results/faith_shop_20260216_101835.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def highlight_diff(base_text, current_text):
    """Highlights differences between base_text and current_text."""
    if base_text == current_text:
        return current_text

    base_words = base_text.split()
    curr_words = current_text.split()

    matcher = difflib.SequenceMatcher(None, base_words, curr_words)
    output = []

    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == 'equal':
            output.append(" ".join(curr_words[b0:b1]))
        elif opcode in ['replace', 'insert']:
            seq = " ".join(curr_words[b0:b1])
            output.append(
                f'<mark style="background-color: #ffff00; color: black; padding: 0 2px; border-radius: 2px;">{seq}</mark>')

    return " ".join(output)


data = load_data()
if data is None:
    st.error("Data file not found. Please check the path.")
    st.stop()

# --- SESSION STATE ---
if 'index' not in st.session_state:
    st.session_state.index = 0
# Initialize eval_mode if not present (Radio button with key will handle this later)
if 'eval_mode' not in st.session_state:
    st.session_state.eval_mode = "baseline"

# --- CSS ---
st.markdown("""
    <style>
    .small-font { font-size: 13px !important; color: #555555; line-height: 1.4; }
    .reasoning-text { font-size: 13px !important; font-style: italic; }
    mark { border-bottom: 2px solid orange; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("Navigation")
col1, col2 = st.sidebar.columns(2)
if col1.button("⬅️ Previous") and st.session_state.index > 0:
    st.session_state.index -= 1
if col2.button("Next ➡️") and st.session_state.index < len(data) - 1:
    st.session_state.index += 1

st.sidebar.markdown(f"**Entry:** {st.session_state.index + 1} / {len(data)}")
st.sidebar.divider()

# FIX: Using 'key' directly prevents the "double-click" bug
modes = ["baseline", "tampered", "unsafe_tampered"]
st.sidebar.radio(
    "Select Evaluation Mode:",
    modes,
    key="eval_mode"
)

# --- MAIN UI ---
current_data = data[st.session_state.index]
entry_meta = current_data['entry']

st.title(f"{entry_meta['scenario_title']}")
st.subheader(f"Concern: {entry_meta['safety_concern']}")

with st.expander("View Scenario Context", expanded=True):
    # Standard usage of shuffle_choices (assuming it returns a list)
    st.text(faith_shop.shuffle_choices(entry_meta)[0])

st.divider()

# --- MESSAGE PROCESSING ---
st.subheader(f"Chat History: {st.session_state.eval_mode.replace('_', ' ').title()}")

current_mode_list = current_data.get(st.session_state.eval_mode, [])
baseline_list = current_data.get("baseline", [])

if not current_mode_list or not baseline_list:
    st.warning("Data missing for selected mode or baseline.")
else:
    curr_msgs = current_mode_list[0]['messages']
    base_msgs = baseline_list[0]['messages']

    for i, msg in enumerate(curr_msgs):
        # Determine Role
        # 0: System, 1: User, 2: Assistant (1st AI msg), 3: User, 4: Assistant...
        role = "system" if i == 0 else ("user" if i % 2 != 0 else "assistant")

        # LOGIC: Only diff if it's NOT baseline AND it's the FIRST Assistant message (index 2)
        should_diff = (st.session_state.eval_mode != "baseline" and i == 2)

        if should_diff and i < len(base_msgs):
            display_text = highlight_diff(base_msgs[i], msg)
        else:
            display_text = msg

        with st.chat_message(role):
            if role == "system":
                with st.expander("Show System Prompt", expanded=False):
                    st.markdown(f'<div class="small-font">{display_text}</div>', unsafe_allow_html=True)

            elif role == "assistant":
                if "<think>" in display_text:
                    parts = display_text.split("</think>")
                    think_content = parts[0].replace("<think>", "").strip()
                    with st.status("Reasoning Process...", expanded=False):
                        st.markdown(f'<div class="reasoning-text">{think_content}</div>', unsafe_allow_html=True)
                    if len(parts) > 1:
                        st.markdown(parts[1].strip(), unsafe_allow_html=True)
                else:
                    st.markdown(display_text, unsafe_allow_html=True)
            else:
                st.markdown(display_text, unsafe_allow_html=True)