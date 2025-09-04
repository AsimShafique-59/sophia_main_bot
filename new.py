import os
import re
from typing import List, Dict, Optional
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


def _guard_against_local_openai_shadowing():
    names = {n.lower() for n in os.listdir(os.getcwd())}
    if "openai.py" in names or ("openai" in names and os.path.isdir("openai")):
        st.error(
            "❌ A local `openai.py` file or `openai/` folder is shadowing the real package.\n"
            "Rename/delete it (e.g., `openai.py` → `groq_client_demo.py`), "
            "remove `__pycache__/`, then restart."
        )
        st.stop()

_guard_against_local_openai_shadowing()


load_dotenv()
st.set_page_config(page_title="Sophia Bot", page_icon="💊")
st.title("Sophia Bot - OHC Pharmacy Assistant")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
if not GROQ_API_KEY:
    st.warning("Missing GROQ_API_KEY in .env")

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

#  Prompts 
GENERAL_PROMPT = """You are Sophia, the Master Assistant for OHC Pharmacy.
Answer general pharmacy questions in a friendly, concise way.
"""

CGM_PROMPT = """You are the CGM Specialist for OHC Pharmacy.
Strict flow (one step at a time):
1) Greeting & ID → Ask for full name and DOB.
2) Insurance → Ask about insurance (Medicare/Medicaid/commercial). If none, explain cash-pay.
3) Clinical → diabetes dx, insulin use, testing frequency, hypoglycemia, last A1c, doctor name.
4) Rx & Medical Necessity → ask for photos; if not available, offer doctor outreach / telehealth link.
5) Expectations → Rx + medical necessity needed before compliance packet & shipment.
6) Delivery → ask for address & phone.
7) If hesitant → offer call/appointment.
Tone: empathetic, supportive, short confirmations.
"""

WEIGHT_PROMPT = """You are the Weight Loss Specialist.
Strict flow (stepwise, concise):
1) Prior use → Ask if patient used semaglutide or tirzepatide.
2) Program & Cost → cash-pay starts at $149; telehealth visit is free.
3) Rx status → If Rx, ask for photo or doctor send. If no Rx, share telehealth link: https://landing.xpedicare.com/#/widget/d6t4
4) Telehealth steps → choose med (injection/sublingual), create account, questions, upload ID + full-body photo, doctor review.
5) After approval → pickup (free) or delivery ($20/$30).
6) If hesitant → offer call/appointment.
7) Common Qs → sema vs tirze, B12, injections vs sublingual, ~5min telehealth, uploads required.
Tone: friendly, simple, conversational.
"""

DME_PROMPT = """You are the General DME Specialist (Texas-compliant).
Strict flow (concise):
1) Greeting & ID → ask for name and DOB.
2) Identify item → ask what equipment is needed.
3) Insurance → ask if they have insurance; if yes collect details; if no explain cash-pay.
4) Prescription rules (Texas):
   - Always Rx: CPAP, oxygen, CGM, power/custom wheelchairs, hospital beds, spinal braces.
   - Insurance needs Rx; cash doesn’t: walkers, canes, shower chairs, off-the-shelf braces, compression 20–30mmHg.
   - Cash only: OTC supplies, comfort aids, low-compression stockings.
   If Rx missing → offer telehealth.
5) Clinical → condition, doctor, prior use, mobility aids.
6) Expectations → insurance requires Rx + medical necessity before compliance packet & shipment.
7) Delivery → ask for address and phone.
8) If hesitant → offer call/appointment.
Tone: empathetic, supportive, clear.
"""

# ── Session State 
# Global chat history used for UI AND for giving the LLM context (last 1 turn)
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

# Boolean router flags; start in general
if "route" not in st.session_state:
    st.session_state.route = {"cgm": False, "weight": False, "dme": False, "general": True}

def set_active(agent: str):
    for k in st.session_state.route.keys():
        st.session_state.route[k] = (k == agent)

def get_active() -> Optional[str]:
    for k, v in st.session_state.route.items():
        if v:
            return k
    return None

def reset_route():
    for k in st.session_state.route.keys():
        st.session_state.route[k] = False
    st.session_state.route["general"] = True


RGX_CGM = re.compile(
    r"\b(cgms?|continuous\s+glucose\s+monitor(s)?|glucose\s+monitor(s)?|blood\s+sugar|dexcom|freestyle\s*-?\s*libre)\b",
    re.I,
)

RGX_WEIGHT = re.compile(
    r"\b(weight[-\s]*loss|semaglutide|tirzepatide|ozempic|wegovy)\b",
    re.I,
)

RGX_DME = re.compile(
    r"\b("
    r"blood\s*pressure\s*monitor(s)?|bp\s*monitor(s)?|"
    r"walker(s)?|wheelchair(s)?|"
    r"shower\s*chair(s)?|bath\s*-?\s*bench(es)?|commode(s)?|"
    r"hospital\s*bed(s)?|"
    r"dme|medical\s*equipment"
    r")\b",
    re.I,
)

RGX_GREET = re.compile(r"^\s*(hi|hello|hey|salam|salaam|assalamualaikum|as-?salamu ?alaykum)\s*!*\.?$", re.I)

def classify_intent(text: str) -> str:
    if RGX_GREET.match(text or ""):
        return "greet"
    if RGX_CGM.search(text):
        return "cgm"
    if RGX_WEIGHT.search(text):
        return "weight"
    if RGX_DME.search(text):
        return "dme"
    return "general"


def chat_complete(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()

def last_n_turns_from_global(n_turns: int = 1) -> List[Dict[str, str]]:
    """
    Return the last 2*n_turns messages (user+assistant pairs) from global chat,
    EXCLUDING the most recent user message (we add it explicitly later).
    """
    msgs = st.session_state.messages[:]
    if msgs and msgs[-1]["role"] == "user":
        msgs = msgs[:-1]  # drop current user so we don't duplicate
    # Take last 2*n_turns messages
    keep = max(0, len(msgs) - 2 * n_turns)
    return [{"role": m["role"], "content": m["content"]} for m in msgs[keep:]]

def specialist_turn(system_prompt: str, user_text: str, temp: float = 0.2) -> str:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    msgs.extend(last_n_turns_from_global(1))  # ← only last 1 turn
    msgs.append({"role": "user", "content": user_text})
    return chat_complete(msgs, temperature=temp) or "⚠️ No response."


def general_reply(query: str) -> str:
    msgs = [{"role": "system", "content": GENERAL_PROMPT}]
    msgs.extend(last_n_turns_from_global(1))
    msgs.append({"role": "user", "content": query})
    return chat_complete(msgs, temperature=0.6)

def cgm_reply(query: str) -> str:
    return specialist_turn(CGM_PROMPT, query)

def weight_reply(query: str) -> str:
    return specialist_turn(WEIGHT_PROMPT, query)

def dme_reply(query: str) -> str:
    return specialist_turn(DME_PROMPT, query)

# ── Router 
def _dispatch(agent: str, query: str) -> str:
    if agent == "cgm":
        return cgm_reply(query)
    if agent == "weight":
        return weight_reply(query)
    if agent == "dme":
        return dme_reply(query)
    return general_reply(query)

def route_message(user_text: str) -> str:
    txt = (user_text or "").strip()

    # Reset commands
    if txt.lower() in {"reset", "exit", "start over"}:
        reset_route()
        return "Okay, I've reset the conversation. How can I help you today?"

    intent = classify_intent(txt)
    active = get_active()

    # Greeting → Sophia
    if intent == "greet":
        reset_route()
        return "Hi! I'm Sophia, the Master Assistant for OHC Pharmacy. How can I help you today?"

    # If active agent exists
    if active:
        if intent == "general":
            set_active("general")
            return general_reply(txt)
        if intent != active:
            set_active(intent)
            return _dispatch(intent, txt)
        return _dispatch(active, txt)

    # Always start with Sophia (general)
    set_active("general")
    return general_reply(txt)

# Render existing chat 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input + routing (flags update first) 
user_input = st.chat_input("Say hi or ask about CGM, Weight Loss, DME… (type 'reset' to start over)")
if user_input:
    # show user bubble
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # route + reply
    reply = route_message(user_input)
    with st.chat_message("assistant"):
        st.markdown(reply)

    # store assistant bubble for global history (LLM context), not per-router
    st.session_state.messages.append({"role": "assistant", "content": reply})

# ── Sidebar AFTER routing so you see the latest flags 
with st.sidebar:
    st.subheader("Routing Status")
    st.markdown(
        f"""**Active:** `{get_active() or "None"}`
- CGM: `{st.session_state.route['cgm']}`
- Weight: `{st.session_state.route['weight']}`
- DME: `{st.session_state.route['dme']}`
- General: `{st.session_state.route['general']}`"""
    )
    if st.button("Reset conversation"):
        reset_route()
        st.success("Routing reset.")
