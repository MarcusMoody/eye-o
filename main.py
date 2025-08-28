import os
import json
import re
import uuid
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import dns.resolver
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI

# -------------------------
# Setup & config
# -------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2:7b")
ENABLE_EXTERNAL_CHECKS = os.getenv("ENABLE_EXTERNAL_CHECKS", "true").lower() == "true"

print("OLLAMA_MODEL =", OLLAMA_MODEL)
print("OLLAMA_URL =", OLLAMA_URL)

client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

DB_PATH = os.getenv("EYE_O_DB", "eyeo.sqlite3")


# -------------------------
# DB helpers
# -------------------------
def db_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with db_conn() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            idea TEXT NOT NULL,
            analysis_json TEXT NOT NULL,
            score INTEGER,
            created_at TEXT NOT NULL
        )
        """)
        con.commit()

@app.on_event("startup")
def on_startup():
    init_db()


# -------------------------
# Utilities
# -------------------------
def balanced_json_or_all(text: str) -> str:
    """Extract the first top-level JSON object, or return all text."""
    start = text.find('{')
    if start == -1:
        return text
    
    brace_count = 0
    for i, char in enumerate(text[start:], start):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start:i+1]
    
    return text

def ensure_list(val: Any) -> List[Any]:
    if isinstance(val, list):
        return val
    if val is None:
        return []
    return [val]

def normalize_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure keys exist with sane defaults so templates never crash."""
    return {
        "viability_score": int(data.get("viability_score", 6)),
        "score_explanation": data.get("score_explanation", "Requires further market validation"),
        "positioning": data.get("positioning", "Clear, differentiated positioning still needed."),
        "target_users": data.get("target_users", "Early adopters and brand-conscious users"),
        "core_pain_points": ensure_list(data.get("core_pain_points") or ["Pain point 1", "Pain point 2", "Pain point 3"]),
        "product_names": ensure_list(data.get("product_names") or ["InnovatePro", "NextGen", "SmartConnect"]),
        "tagline": data.get("tagline", "Innovation made simple."),
        "brand_personality": data.get("brand_personality", "Modern, confident, user-centric"),
        "mood_keywords": ensure_list(data.get("mood_keywords") or ["innovative", "reliable", "modern", "bold", "clean"]),
        "color_palette": ensure_list(data.get("color_palette") or ["#667eea", "#764ba2", "#f093fb"]),
        "key_risks": ensure_list(data.get("key_risks") or ["Competition", "Adoption", "Execution"]),
        "counter_moves": ensure_list(data.get("counter_moves") or ["Differentiate on brand", "Nail onboarding", "Partner GTM"]),
        "opportunities": ensure_list(data.get("opportunities") or ["Growing creator tools", "SMB demand", "AI assist boom"]),
        "similar_products": ensure_list(data.get("similar_products") or ["Competitor A", "Competitor B", "Competitor C"]),
        "revenue_model": data.get("revenue_model", "Subscription with freemium"),
        "gtm_channels": ensure_list(data.get("gtm_channels") or ["TikTok UGC", "Founder-led LinkedIn", "Community partnerships"]),
        "launch_30_day_plan": ensure_list(data.get("launch_30_day_plan") or [
            "Define ICP + pain points", "Ship MVP", "Seed 10 users", "Iterate weekly", "Share results publicly"
        ]),
        "tamsam_som": data.get("tamsam_som", "TAM/SAM/SOM rough: early-stage brand tooling for solopreneurs and SMBs."),
        "next_steps": ensure_list(data.get("next_steps") or ["Market research", "MVP build", "User testing", "Beta launch", "Iterate GTM"]),
    }


# -------------------------
# Prompt builders
# -------------------------
def build_oracle_prompt(idea: str) -> str:
    return f"""
You are the Moody Maestro Oracle: witty, precise, street-smart strategist.
Return ONLY valid JSON with these keys:
viability_score (1-10), score_explanation, positioning, target_users,
core_pain_points (3), product_names (3), tagline, brand_personality,
mood_keywords (5), color_palette (3 hex), key_risks (3),
counter_moves (3), opportunities (3), similar_products (3),
revenue_model, gtm_channels (3), launch_30_day_plan (5 steps),
tamsam_som (rough text), next_steps (5).

Use concise, concrete language. No hype.

Product Idea: "{idea}"

Return ONLY a single JSON object. No explanations, no prose, no backticks.\n\n
""".strip()


# -------------------------
# AI calls (OpenAI and Local/Ollama)
# -------------------------
def analyze_idea_with_openai(idea: str) -> Dict[str, Any]:
    if not client:
        raise RuntimeError("OPENAI_API_KEY is not set")
    prompt = build_oracle_prompt(idea)
    try:
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1200,
        )
        content = resp.choices[0].message.content.strip()
        json_str = balanced_json_or_all(content)
        data = json.loads(json_str)
        return normalize_analysis(data)
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return normalize_analysis({})

def analyze_idea_with_local(idea: str) -> Dict[str, Any]:
    prompt = build_oracle_prompt(idea)
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json", "options": { "num_predict": 800}},
            timeout=120,
        )
        r.raise_for_status()
        
        raw = r.json().get("response", "").strip()
        print("\n--- OLLAMA RAW (first 300) ---\n", raw[:300], "\n")
        
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            
            json_str = balanced_json_or_all(raw)
            data = json.loads(json_str)
        return normalize_analysis(data)
    except Exception as e:
        
        try: print("Ollama raw (truncated):", r.text[:500])
        except: 
            pass
        print(f"Ollama error: {e}")
        return normalize_analysis({})


# -------------------------
# Domain & handle checks
# -------------------------
def domain_available(name: str, ext: str) -> bool:
    """
    Simple availability heuristic:
    - If DNS A/NS records resolve -> taken
    - If NXDOMAIN -> likely available
    - On error/timeouts -> treat as unknown => False (i.e., show as taken)
    """
    domain = f"{name.lower().replace(' ', '').replace('-', '')}{ext}"
    try:
        # Try A record first
        dns.resolver.resolve(domain, "A")
        return False  # resolves => taken
    except dns.resolver.NXDOMAIN:
        return True   # doesn't exist => likely available
    except Exception:
        # Try NS as a secondary
        try:
            dns.resolver.resolve(domain, "NS")
            return False
        except dns.resolver.NXDOMAIN:
            return True
        except Exception:
            return False  # unknown => conservative

def check_domains(names: List[str], extensions=None) -> Dict[str, Dict[str, bool]]:
    if extensions is None:
        extensions = [".com", ".io", ".ai"]
    availability: Dict[str, Dict[str, bool]] = {}
    for name in names:
        availability[name] = {}
        for ext in extensions:
            if ENABLE_EXTERNAL_CHECKS:
                availability[name][ext] = domain_available(name, ext)
            else:
                availability[name][ext] = True  # mock
    return availability

def check_social_handles(names: List[str]) -> Dict[str, Dict[str, Optional[bool]]]:
    """
    Quick 'availability' check by requesting profile pages.
    - True => 404 (not found) => likely available
    - False => 200/302 => likely taken
    - None => unknown/error
    """
    platforms = {
        "x": "https://x.com/{}",
        "instagram": "https://www.instagram.com/{}/",
        "tiktok": "https://www.tiktok.com/@{}",
    }
    out: Dict[str, Dict[str, Optional[bool]]] = {}
    headers = {"User-Agent": "Mozilla/5.0 (Eye-O)"}
    for raw in names:
        handle = raw.lower().replace(" ", "").replace("-", "")
        out[raw] = {}
        for p, url_tpl in platforms.items():
            if not ENABLE_EXTERNAL_CHECKS:
                out[raw][p] = None
                continue
            try:
                # HEAD is often blocked; use GET with small timeout
                r = requests.get(url_tpl.format(handle), headers=headers, timeout=6, allow_redirects=True)
                if r.status_code == 404:
                    out[raw][p] = True
                elif 200 <= r.status_code < 400:
                    out[raw][p] = False
                else:
                    out[raw][p] = None
            except Exception:
                out[raw][p] = None
    return out


# -------------------------
# Routes
# -------------------------
@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    idea: str = Form(...),
    use_local: str = Form(None),
):
    if not idea.strip():
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Please enter a product idea to analyze."},
        )
    try:
        if use_local == "1":
            analysis = analyze_idea_with_local(idea)
        else:
            analysis = analyze_idea_with_openai(idea)

        domains = check_domains(analysis.get("product_names", []))
        handles = check_social_handles(analysis.get("product_names", []))

        # Save to DB
        rec_id = str(uuid.uuid4())
        with db_conn() as con:
            con.execute(
                "INSERT INTO analyses (id, idea, analysis_json, score, created_at) VALUES (?, ?, ?, ?, ?)",
                (rec_id, idea, json.dumps(analysis), int(analysis["viability_score"]), datetime.utcnow().isoformat()),
            )
            con.commit()

        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "idea": idea,
                "analysis": analysis,
                "analysis_json": json.dumps(analysis, indent=2),
                "domains": domains,
                "handles": handles,
                "record_id": rec_id,
            },
        )
    except Exception as e:
        print(f"Error in /analyze: {e}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"An error occurred: {str(e)}"},
        )

@app.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    with db_conn() as con:
        rows = con.execute(
            "SELECT id, idea, score, created_at FROM analyses ORDER BY datetime(created_at) DESC LIMIT 50"
        ).fetchall()
    items = [{"id": r[0], "idea": r[1], "score": r[2], "created_at": r[3]} for r in rows]
    return templates.TemplateResponse("history.html", {"request": request, "items": items})

@app.get("/view/{rec_id}", response_class=HTMLResponse)
async def view_record(request: Request, rec_id: str):
    with db_conn() as con:
        row = con.execute(
            "SELECT idea, analysis_json FROM analyses WHERE id = ?", (rec_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Record not found")
    idea = row[0]
    analysis = json.loads(row[1])
    domains = check_domains(analysis.get("product_names", []))
    handles = check_social_handles(analysis.get("product_names", []))
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "idea": idea,
            "analysis": analysis,
            "analysis_json": json.dumps(analysis, indent=2),
            "domains": domains,
            "handles": handles,
            "record_id": rec_id,
        },
    )

@app.get("/download/{rec_id}.json")
async def download_json(rec_id: str):
    with db_conn() as con:
        row = con.execute(
            "SELECT analysis_json FROM analyses WHERE id = ?", (rec_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    data = json.loads(row[0])
    return JSONResponse(content=data)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# -------------------------
# Local dev entrypoint
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)