# Eye-O — Product Idea Oracle

**Live Demo:** [https://eyeo.app](https://eyeo.app)

Eye-O turns rough product ideas into investor-ready briefs in 60 seconds. 
Get a viability score, brand kit, risk map, competitor scan, domains, and a launch plan — instantly.

---

## ✨ Features
- Structured JSON analysis via OpenAI
- Names, tagline, color palette, mood keywords
- Risks → counter-moves, opportunities, similar products
- Domain/handle checks
- Export results (PDF, copy/share ready)

---

## 🛠️ Tech Stack
- FastAPI, Python 3.11+
- Jinja2 Templates
- OpenAI API
- HTML / CSS (static assets)

---

## 🚀 Local Development
```bash
git clone https://github.com/moody-creative/eye-o.git
cd eye-o

python -m venv venv
source venv/bin/activate   # or .\venv\Scripts\activate on Windows

pip install -r requirements.txt
cp .env.example .env       # Add your own OPENAI_API_KEY=sk-xxxx
uvicorn main:app --reload