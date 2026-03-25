---
title: Fed Moderation
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.32.2"
python_version: "3.11"
app_file: app.py
pinned: false
---



# 🛡️ Federated GenAI Content Moderation System

> Decentralized AI for Transparent and Unbiased Social Media Governance

Built by **Vaibhav Mishra** | RGIPT Amethi | Extension of Curtin University Research Internship

---

## 🏗️ Architecture

```
Social Media Post (text)
        ↓
FastAPI Gateway  (src/api/)
        ↓
Federated Learning Cluster  (src/federated/)
│  ├── Client 1 (Opacus DP)
│  ├── Client 2 (Opacus DP)
│  └── Client 3 (Opacus DP)
│        ↓ FedAvg aggregation
        ↓
GenAI Explainer  (src/genai/)   ← Gemini API
"Why was this flagged?"
        ↓
IPFS Audit Trail  (src/ipfs/)   ← Pinata
        ↓
Streamlit Dashboard  (src/dashboard/)
        ↓
Docker → GitHub Actions → HuggingFace Spaces
```

---

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone <your-repo>
cd fed-moderation

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp .env.example .env
# Fill in your API keys in .env

# 5. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 6. Run Step 2 - Prepare data
python src/federated/data_prep.py

# 7. Start FL Server (Terminal 1)
python src/federated/server.py

# 8. Start FL Clients (Terminal 2, 3, 4)
python src/federated/client.py --client-id 0
python src/federated/client.py --client-id 1
python src/federated/client.py --client-id 2

# 9. Start API (Terminal 5)
uvicorn src.api.main:app --reload

# 10. Start Dashboard (Terminal 6)
streamlit run src/dashboard/app.py
```

---

## 📁 Project Structure

```
fed-moderation/
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Cleaned, split per client
├── src/
│   ├── federated/        # FL server, clients, model
│   ├── genai/            # Gemini explainer
│   ├── ipfs/             # Pinata audit trail
│   ├── api/              # FastAPI backend
│   └── dashboard/        # Streamlit UI
├── mlflow_tracking/      # Experiment logs
├── docker/               # Dockerfile, compose
├── tests/                # Unit tests
├── .github/workflows/    # CI/CD
├── requirements.txt
└── .env.example
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Federated Learning | Flower (flwr), FedAvg |
| Privacy | Opacus (Differential Privacy) |
| Model | PyTorch + HuggingFace |
| GenAI | Google Gemini API + LangChain |
| Storage | Pinata / IPFS |
| API | FastAPI |
| Dashboard | Streamlit + Plotly |
| MLOps | MLflow + DVC |
| Deployment | Docker + GitHub Actions + HuggingFace Spaces |
