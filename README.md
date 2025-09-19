# Eye-O — Product Idea Oracle

**Live:** https://eyeo.app 

Eye-O turns rough product ideas into investor-ready briefs: viability score, brand kit, risk map, competitor scan, domains, and a 30-day launch plan.

---

## Features
- Structured JSON analysis via OpenAI
- Names, tagline, color palette, mood keywords
- Risks → counter-moves, opportunities, similar products
- Domain/handle checks, print/share-ready results

---

## Tech
FastAPI, Jinja2, Python 3.11+, OpenAI API.

---

## Local Dev
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY=sk-...
uvicorn main:app --reload