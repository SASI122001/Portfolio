import streamlit as st
import streamlit.components.v1 as components
import time
import os
import requests
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

st.set_page_config(page_title="Sasi Kiran Boyapati Portfolio", layout="wide", page_icon="🤖")


groq_key = ""

# Theme selector
selected_theme = st.sidebar.radio("🎨 Select Theme", ["Dark", "Light"])
dark_mode = selected_theme == "Dark"

# Theme colors
if dark_mode:
    bg_color = "#0d1117"
    text_color = "#ffffff"
    card_bg = "linear-gradient(145deg, #1c1f26, #111319)"
    title_color = "#00c6ff"
    highlight_color = "#00bfff"
else:
    bg_color = "#f4f6fb"
    text_color = "#212529"
    card_bg = "#ffffff"
    title_color = "#003366"
    highlight_color = "#0078d7"

# Custom styles with 3D effect
st.markdown(f"""
<style>
body {{
  background-color: {bg_color}; color: {text_color}; font-family: 'Segoe UI', sans-serif; overflow-x: hidden;
}}
.main-title {{
  font-size: 3em; font-weight: 700; color: {title_color}; text-align: center; margin-bottom: 20px;
}}
.section {{
  background: {card_bg}; border: 1px solid rgba(0,0,0,0.08); border-radius: 15px;
  padding: 1.8rem; margin-bottom: 2rem;
  box-shadow: 8px 8px 15px rgba(0,0,0,0.15), -8px -8px 15px rgba(255,255,255,0.1);
  transform: perspective(1000px) rotateX(1deg) rotateY(1deg);
  transition: transform 0.4s ease, box-shadow 0.4s ease;
}}
.section:hover {{
  transform: perspective(1000px) rotateX(0deg) rotateY(0deg);
  box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}}
h3 {{
  font-size: 1.6em; font-weight: bold; color: {highlight_color}; margin-bottom: 1rem;
}}
p, li {{
  font-size: 1.05em; line-height: 1.7; color: {text_color};
}}
.stDownloadButton>button {{
  background: {highlight_color}; color: white; border: none;
  padding: 10px 20px; font-weight: bold; border-radius: 10px;
  transition: all 0.3s ease;
}}
.stDownloadButton>button:hover {{
  transform: scale(1.05); background-color: {title_color};
}}
</style>
""", unsafe_allow_html=True)

# Floating particles background
components.html("""
<canvas id='particleCanvas' style='position:fixed; top:0; left:0; width:100vw; height:100vh; z-index:-1'></canvas>
<script>
const canvas = document.getElementById('particleCanvas');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
const particles = Array.from({length: 90}, () => ({
  x: Math.random()*canvas.width,
  y: Math.random()*canvas.height,
  r: Math.random()*2+1,
  dx: (Math.random()-0.5)*0.8,
  dy: (Math.random()-0.5)*0.8
}));
function animate() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let p of particles) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.fill();
    p.x += p.dx; p.y += p.dy;
    if (p.x < 0 || p.x > canvas.width) p.dx *= -1;
    if (p.y < 0 || p.y > canvas.height) p.dy *= -1;
  }
  requestAnimationFrame(animate);
}
animate();
</script>
""", height=0)

# PDF to TXT
if not os.path.exists("data/resume.txt") and os.path.exists("assets/resume.pdf"):
    with open("data/resume.txt", "w", encoding="utf-8") as f:
        for page in reader.pages:
            text = page.extract_text()
            if text:
                f.write(text + "\n")

@st.cache_resource
def build_chatbot():
    docs = []
    for txt_file in Path("data").glob("*.txt"):
        loader = TextLoader(str(txt_file))
        docs.extend(loader.load())
    chunks = CharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
    vectorstore = Chroma.from_documents(chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), persist_directory="chroma_store")
    vectorstore.persist()
    return RetrievalQA.from_chain_type(
        llm=ChatGroq(groq_api_key=groq_key, model_name="llama3-8b-8192"),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )

def show_section(title, filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            st.markdown(f"""
                <div class='section'>
                    <h3>{title}</h3>
                    <div>{f.read()}</div>
                </div>
            """, unsafe_allow_html=True)

st.markdown("<div class='main-title'>🤖 Sasi Kiran Boyapati Portfolio</div>", unsafe_allow_html=True)

# Resume download
if os.path.exists("assets/resume.pdf"):
    with open("assets/resume.pdf", "rb") as f:
        st.download_button("📄 Download Resume", f, "SasiKiran_Resume.pdf", mime="application/pdf")

# Show sections
for title, path in [
    ("📜 About Me", "data/about.txt"),
    ("🎓 Certifications", "data/certifications.txt"),
    ("🚀 Projects", "data/projects.txt"),
    ("💼 Experience", "data/experience.txt"),
    ("🎓 Education", "data/education.txt")
]:
    show_section(title, path)

# 📬 Contact form
with st.form("contact_form"):
    st.markdown("<div class='section'><h3>📬 Contact Me</h3></div>", unsafe_allow_html=True)
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Your Message")
    if st.form_submit_button("Send"):
        if name and email and message:
            r = requests.post("https://formspree.io/f/xzzgbnpz", data={"name": name, "email": email, "message": message}, headers={"Accept": "application/json"})
            if r.status_code == 200:
                st.success("✅ Message sent to Sasi Kiran successfully!")
            else:
                st.error("❌ Failed to send message. Please try again later.")
        else:
            st.warning("⚠️ Please fill out all fields before submitting.")

# 💬 Chatbot
st.sidebar.title("💬 SasiBot - Ask Me Anything")
st.sidebar.caption("🧠 This chatbot uses only Sasi's resume & data.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.sidebar.text_input("Ask about Sasi:", key="chat_input")
if user_input:
    try:
        result = build_chatbot()({"query": user_input})
        st.session_state.chat_history.append((user_input, result["result"]))
    except Exception as e:
        st.session_state.chat_history.append((user_input, f"❌ Error: {e}"))

for q, a in reversed(st.session_state.chat_history):
    with st.sidebar.expander(f"🧑 You: {q}", expanded=False):
        typed_text = ""
        response_placeholder = st.empty()
        for char in a:
            typed_text += char
            response_placeholder.markdown(f"**🤖 SasiBot:** {typed_text}")
            time.sleep(0.01)

# 🌐 Connect with Me
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤝 Connect with Me")

st.sidebar.markdown(
    "[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/boyapatisasi/)"
)

st.sidebar.markdown(
    "[![GitHub](https://img.shields.io/badge/GitHub-Code-black?style=for-the-badge&logo=github)](https://github.com/SASI122001)"
)
