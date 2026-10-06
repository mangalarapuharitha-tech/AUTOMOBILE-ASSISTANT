from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AutoGuide AI",
    page_icon="🚗",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #0b0f14;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #11161d;
        border-right: 1px solid #252c35;
    }

    .main-title {
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 0px;
    color: white;
}
    }

    .subtitle {
        color: #aab4c0;
        font-size: 18px;
        margin-bottom: 25px;
    }

    /* Dashboard cards */
    .dashboard-card {
        background: linear-gradient(145deg, #151b23, #10151b);
        border: 1px solid #29313b;
        border-radius: 16px;
        padding: 20px;
        min-height: 125px;
        box-shadow: 0px 5px 20px rgba(0,0,0,0.25);
    }

    .card-title {
        color: #9da8b5;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .card-value {
        font-size: 27px;
        font-weight: 700;
        color: white;
    }

    .card-status {
        color: #58d68d;
        font-size: 13px;
        margin-top: 6px;
    }

    /* Section headings */
    .section-title {
        font-size: 23px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Info banner */
    .info-box {
        background: linear-gradient(
            90deg,
            rgba(255,75,75,0.12),
            rgba(255,138,101,0.05)
        );
        border: 1px solid rgba(255,75,75,0.3);
        border-radius: 15px;
        padding: 18px;
        margin-bottom: 20px;
    }

    .info-title {
        font-size: 19px;
        font-weight: 700;
        color: white;
    }

    .info-text {
        color: #aeb8c4;
        margin-top: 5px;
    }

    /* Knowledge base */
    .knowledge-box {
        background: #121820;
        border: 1px solid #29323d;
        border-radius: 14px;
        padding: 15px;
    }

    /* Hide Streamlit default menu/footer */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_context" not in st.session_state:
    st.session_state.last_context = ""

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = ""


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🚗🏍️ AutoGuide AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your Intelligent Car & Bike Assistant'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🚗🏍️ AutoGuide AI")

    st.caption("Intelligent Automobile Knowledge Assistant")

    st.divider()

    # Vehicle
    vehicle_type = st.selectbox(
        "🚘 Select Vehicle",
        ["🚗 Car", "🏍️ Bike"]
    )

    # Personality
    personalities = {

        "🔧 Mechanic":
        """
        Act like an experienced automobile mechanic.
        Explain vehicle problems, possible causes and
        basic maintenance clearly.
        Use simple practical language.
        """,

        "🧑‍🔧 Expert":
        """
        Act as an automobile expert.
        Give technically accurate answers about cars
        and bikes using correct automobile terminology.
        Keep explanations clear and useful.
        """
    }

    personality = st.selectbox(
        "🤖 Assistant Personality",
        list(personalities.keys())
    )

    # Top K
    top_k = st.slider(
        "🔎 Knowledge Results",
        min_value=1,
        max_value=5,
        value=3
    )

    st.divider()

    # Upload
    st.markdown("### 📄 Vehicle Knowledge")

    uploaded_file = st.file_uploader(
        "Upload your vehicle file",
        type=["txt"],
        help="Upload a TXT file containing vehicle information."
    )

    if uploaded_file:

        st.success("✅ Knowledge file loaded")

        st.caption(
            f"📄 {uploaded_file.name}"
        )

    st.divider()

    st.caption("🧠 RAG Engine: Active")
    st.caption("🗄️ ChromaDB: Active")
    st.caption("🤖 Llama 3.2: Active")


# =========================================================
# CHROMADB
# Separate knowledge base for Car and Bike
# =========================================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

if vehicle_type == "🚗 Car":
    collection_name = "AutoGuide_Car"
else:
    collection_name = "AutoGuide_Bike"

collection = client.get_or_create_collection(
    name=collection_name
)


# =========================================================
# FILE PROCESSING
# =========================================================

if uploaded_file:

    text = uploaded_file.read().decode(
        "utf-8",
        errors="ignore"
    )

    st.session_state.uploaded_name = uploaded_file.name

    # Preview
    with st.sidebar.expander("👁️ Preview Knowledge"):

        st.text(
            text[:2500]
        )

    # -----------------------------------------------------
    # CHUNKING
    # -----------------------------------------------------

    chunks = []

    chunk_size = 500
    chunk_overlap = 100

    step = chunk_size - chunk_overlap

    for i in range(
        0,
        len(text),
        step
    ):

        chunk = text[i:i + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

    # -----------------------------------------------------
    # EMBEDDINGS
    # -----------------------------------------------------

    embeddings = model.encode(chunks)

    # -----------------------------------------------------
    # IDS
    # -----------------------------------------------------

    ids = []

    for i in range(len(chunks)):

        ids.append(
            f"{uploaded_file.name}_{i}"
        )

    # -----------------------------------------------------
    # AVOID DUPLICATES
    # -----------------------------------------------------

    existing_data = collection.get(
        ids=ids
    )

    existing_ids = existing_data["ids"]

    new_chunks = []
    new_embeddings = []
    new_ids = []

    for i in range(len(ids)):

        if ids[i] not in existing_ids:

            new_chunks.append(
                chunks[i]
            )

            new_embeddings.append(
                embeddings[i].tolist()
            )

            new_ids.append(
                ids[i]
            )

    # -----------------------------------------------------
    # ADD TO CHROMADB
    # -----------------------------------------------------

    if new_chunks:

        collection.add(

            documents=new_chunks,

            embeddings=new_embeddings,

            ids=new_ids
        )

        st.sidebar.success(
            "✅ Knowledge added to RAG database!"
        )


# =========================================================
# WELCOME INFORMATION
# =========================================================

st.markdown("""
<div class="info-box">

<div class="info-title">
🤖 Welcome to AutoGuide AI
</div>

<div class="info-text">
Ask questions about vehicle maintenance, engines, batteries,
brakes, tyres, fuel and other automobile information.
Upload your vehicle knowledge file to activate RAG-based answers.
</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# VEHICLE DASHBOARD
# =========================================================

st.markdown(
    '<div class="section-title">📊 Vehicle Dashboard</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div class="dashboard-card">
        <div class="card-title">⚙️ ENGINE</div>
        <div class="card-value">INFO</div>
        <div class="card-status">Available</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="dashboard-card">
        <div class="card-title">🔋 BATTERY</div>
        <div class="card-value">INFO</div>
        <div class="card-status">Available</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="dashboard-card">
        <div class="card-title">🛞 TYRES</div>
        <div class="card-value">INFO</div>
        <div class="card-status">Available</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="dashboard-card">
        <div class="card-title">🛑 BRAKES</div>
        <div class="card-value">INFO</div>
        <div class="card-status">Available</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="dashboard-card">
        <div class="card-title">🔧 SERVICE</div>
        <div class="card-value">INFO</div>
        <div class="card-status">Available</div>
    </div>
    """, unsafe_allow_html=True)

st.caption("Vehicle information categories — not live sensor readings.")


# =========================================================
# KNOWLEDGE BASE
# =========================================================



st.markdown(
    '<div class="section-title">🧠 Knowledge Base</div>',
    unsafe_allow_html=True
)

k1, k2, k3 = st.columns(3)

with k1:

    st.metric(
        "Vehicle",
        vehicle_type
    )

with k2:

    st.metric(
        "Knowledge Chunks",
        collection.count()
    )

with k3:

    if uploaded_file:
        st.metric(
            "File Status",
            "Loaded"
        )
    else:
        st.metric(
            "File Status",
            "Waiting"
        )


st.markdown(
    '<div class="section-title">💬 AI Assistant</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)


with c1:

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.last_context = ""

        st.rerun()


with c2:

    st.info(
        f"Selected: {vehicle_type}"
    )


with c3:

    st.info(
        f"Mode: {personality}"
    )


# =========================================================
# CHAT HISTORY
# =========================================================

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.write(
            msg["content"]
        )


question = st.chat_input(
    f"Ask AutoGuide AI about your {vehicle_type}..."
)


if question:


    with st.chat_message("user"):

        st.write(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # =====================================================
    # RAG MODE
    # =====================================================

    if uploaded_file:

        with st.spinner(
            "🔎 Searching vehicle knowledge..."
        ):

            q_embedding = model.encode(
                question
            )

            results = collection.query(

                query_embeddings=[
                    q_embedding.tolist()
                ],

                n_results=top_k
            )

            retrieved_chunks = results["documents"][0]

            context = "\n\n".join(
                retrieved_chunks
            )

            st.session_state.last_context = context



        prompt = f"""

You are AutoGuide AI, an intelligent automobile assistant.

Selected Vehicle:
{vehicle_type}

Assistant Personality:
{personalities[personality]}

User Question:
{question}

Retrieved Vehicle Knowledge:
{context}

Instructions:

1. Answer specifically for the selected vehicle.
2. Use the retrieved knowledge as the main source.
3. Do not invent specifications.
4. Keep the explanation clear and useful.
5. If the retrieved knowledge does not contain the answer,
   clearly say that the information is not available.
6. For serious vehicle problems, recommend consulting
   a qualified mechanic.
7. Do not claim that demo dashboard values are real sensor data.

Final Answer:
"""


        # -------------------------------------------------
        # OLLAMA
        # -------------------------------------------------

        with st.spinner(
            "🤖 AutoGuide AI is thinking..."
        ):

            response = ollama.chat(

                model="llama3.2:3b",

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )


        answer = response[
            "message"
        ][
            "content"
        ]


    # =====================================================
    # NORMAL AI CHAT
    # =====================================================

    else:

        with st.spinner(
            "🤖 AutoGuide AI is thinking..."
        ):

            response = ollama.chat(

                model="llama3.2:3b",

                messages=[

                    {
                        "role": "system",

                        "content": f"""

You are AutoGuide AI.

Selected Vehicle:
{vehicle_type}

Assistant Personality:
{personalities[personality]}

Answer automobile questions clearly and safely.

If the user asks for serious mechanical repair instructions,
recommend consulting a qualified mechanic.
"""
                    }

                ] + st.session_state.messages
            )


        answer = response[
            "message"
        ][
            "content"
        ]


    # =====================================================
    # SAVE ANSWER
    # =====================================================

    st.session_state.messages.append(

        {
            "role": "assistant",
            "content": answer
        }

    )


    # =====================================================
    # DISPLAY ANSWER
    # =====================================================

    with st.chat_message(
        "assistant"
    ):

        st.write(answer)


# =========================================================
# RAG SOURCES
# =========================================================

if st.session_state.last_context:

    with st.expander(
        "🔍 View Retrieved Knowledge"
    ):

        st.write(
            st.session_state.last_context
        )