import os
import json
import re
import uuid
import sqlite3
import io
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
from prompt_testing import log_prompt_test
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from fastapi.responses import FileResponse

# -------------------------
# Setup & config
# -------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2:7b")
ENABLE_EXTERNAL_CHECKS = os.getenv("ENABLE_EXTERNAL_CHECKS", "true").lower() == "true"
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
NAMECHEAP_API_KEY = os.getenv("NAMECHEAP_API_KEY")
NAMECHEAP_USERNAME = os.getenv("NAMECHEAP_USERNAME")

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

def get_score_label(score: int) -> str:
    """Convert numeric score to contextual label"""
    if score >= 9: return "Lock in now!"
    elif score >= 7: return "Strong potential"
    elif score >= 5: return "Good foundation" 
    elif score >= 3: return "Needs work"
    else: return "High risk"

def normalize_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Restructured for new Product/Target sections"""
    # Select best name from the three options
    product_names = ensure_list(data.get("product_names") or ["InnovatePro", "NextGen", "SmartConnect"])
    selected_name = product_names[0] if product_names else "InnovatePro"
    
    return {
        # Scoring with contextual labels
        "viability_score": int(data.get("viability_score", 6)),
        "score_label": get_score_label(int(data.get("viability_score", 6))),
        "score_explanation": data.get("score_explanation", "Requires further market validation"),
        
        # Product section (renamed from Positioning)
        "product": {
            "selected_name": selected_name,
            "all_names": product_names,
            "positioning": data.get("positioning", "Clear, differentiated positioning needed"),
            "tagline": data.get("tagline", "Innovation made simple")
        },
        
        # Target section (consolidated)
        "target": {
            "users": data.get("target_users", "Early adopters and tech-savvy consumers"),
            "how": ensure_list(data.get("gtm_channels") or ["Social media", "Content marketing", "Partnerships"]),
            "my_focus": data.get("tamsam_som", "Early-stage market validation needed")
        },
        
        # Keep existing structure for other sections
        "core_pain_points": ensure_list(data.get("core_pain_points") or ["Pain point 1", "Pain point 2", "Pain point 3"]),
        "brand_personality": data.get("brand_personality", "Modern, confident, user-centric"),
        "mood_keywords": ensure_list(data.get("mood_keywords") or ["innovative", "reliable", "modern"]),
        "color_palette": ensure_list(data.get("color_palette") or ["#667eea", "#764ba2", "#f093fb"]),
        "key_risks": ensure_list(data.get("key_risks") or ["Competition", "Adoption", "Execution"]),
        "counter_moves": ensure_list(data.get("counter_moves") or ["Differentiate", "Optimize onboarding", "Partner strategy"]),
        "opportunities": ensure_list(data.get("opportunities") or ["Growing market", "Technology advancement", "User demand"]),
        "similar_products": ensure_list(data.get("similar_products") or ["Competitor A", "Competitor B", "Competitor C"]),
        "revenue_model": data.get("revenue_model", "Subscription with freemium"),
        "launch_30_day_plan": ensure_list(data.get("launch_30_day_plan") or ["Research", "Build MVP", "Test", "Launch", "Iterate", "Scale"]),
        "next_steps": ensure_list(data.get("next_steps") or ["Market research", "MVP build", "User testing", "Beta launch", "Scale"]),
        "mood_images": data.get("mood_images", [])
    }


# -------------------------
# API integrations
# -------------------------
def get_mood_images(keywords: List[str]) -> List[str]:
    """Based on this specific product idea, Get mood board images from Unsplash"""
    if not UNSPLASH_ACCESS_KEY or not ENABLE_EXTERNAL_CHECKS:
        return []
    
    images = []
    for keyword in keywords[:3]:  # Limit to 3 keywords
        try:
            response = requests.get(
                f"https://api.unsplash.com/search/photos",
                params={"query": keyword, "per_page": 1, "orientation": "landscape"},
                headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    images.append(data["results"][0]["urls"]["small"])
        except Exception as e:
            print(f"Unsplash error for {keyword}: {e}")
    
    return images

def check_uspto_trademark(name: str) -> Dict[str, Any]:
    """Check trademark status via USPTO API"""
    try:
        # Simple check - in production, implement full USPTO TESS API
        url = f"https://tmsapi.uspto.gov/api/v1/trademark/search"
        # For now, return mock data
        return {"status": "available", "conflicts": 0}
    except Exception as e:
        return {"status": "unknown", "error": str(e)}


# -------------------------
# Prompt builders
# -------------------------
def build_oracle_prompt(idea: str) -> str:
    return f"""
You are a strategic product analyst. Analyze this product idea and return ONLY valid JSON.

Product Idea: "{idea}"

Return JSON with this exact structure:
{{
    "viability_score": [Based on this specific product idea, give a comprehensive 1-10 score of the business idea taking into accountability originality, it solving a real-world problem, growth potential, and all criteria of a potentially successful business idea],
    "score_explanation": "[Based on the specific product idea, give a brief explanation of the score]",
    "positioning": "[Based on this specific product idea, give a clear business market positioning statement]",
    "target_users": "[Based on this specific product idea, give specific, short yet highly descriptive user demographics and characteristics]",
    "product_names": [Give 3 original, memorable, innovative, relative and immediately usable product names that relate to the specific product and business. For a meme app: "MeMe (pronounced Mimi)", For a memory palace generator AI app: "Memory's Past". Be as creative as possible.],
    "tagline": "[catchy, original, and memorable tagline under 8 words]",
    "brand_personality": "[based on this specific product idea, give a viable brand character and tone description]",
    "mood_keywords": [Give 5 words that describe the specific aesthetic, vibe, feeling, style, and energy of the specific product],
    "color_palette": [Give 3 beautiful, stylistically and aesthetically pleasing colors and their hexcodes],
    "core_pain_points": ["[specific_pain1]", "[specific_pain2]", "[specific_pain3]"],
    "key_risks": [Give top 3 riskiest assumptions about the specific product],
    "counter_moves": [Give 3 strategic, tactical, and defensive counters to the products top 3 riskiest assumptions],
    "opportunities": [Based on this specific product idea, list 3 opportunities in the realm of marketing, growth, and competition failures],
    "similar_products": [Research and name 3 REAL competing products/companies that actually exist, not placeholders],
    "revenue_model": "[Based on this specific product, give a tailored, logical, and specific revenue approach. Nothing simple and basic like "freemium mode, premium features". Be specific to the product and think of revenue for the product in ways that may not have been pursued before, but will increase revenue],
    "gtm_channels": [Based on this specific product idea, list 3 concrete marketing channels with platform names. Be specific - instead of "social media" say "TikTok videos" or "LinkedIn posts". Instead of "partnerships" say "partnership with [specific type of company]". For a meme app: "TikTok creator partnerships", "Reddit r/memes community", "Discord meme servers". For B2B: "LinkedIn cold outreach", "Product Hunt launch", "Y Combinator Slack". Be this specific.],
    "tamsam_som": "Based on this specific product idea, take a realistic market size assessment with context, and briefly explain how to aquire a products SOM. Please explain in terms that a high school graduate would understand],
    "launch_30_day_plan": [Based on this specific product idea, Create 6 actionable steps specific to THIS product type, not a generic startup checklist],
    "next_steps": [Based on this specific product idea, list 6 steps with more concrete details for users to get the product from idea to launch. An example of an immediate first step would be "Render Working Code" and the link should be to a site where the user can render starter code. For an AI-powered, legal marketing tool, steps should include specific details to product like: specific legal conferences to target and exact LinkedIn ad strategies]
}}

Requirements:
- Product names must be creative, brandable, NOT generic (avoid "Pro", "Solution", "Platform"), innovative
- Use real competitor/company names when possible
- Color palette must be valid hex codes starting with #
- GTM channels should be specific platforms/methods, not general categories
- All arrays must have the exact number of items specified
- Be concrete and actionable, avoid vague business speak
- Return ONLY the JSON object, no other text
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
            max_tokens=1500,
        )
        content = resp.choices[0].message.content.strip()
        json_str = balanced_json_or_all(content)
        data = json.loads(json_str)
        data["idea"] = idea
        
        # Add mood images based on keywords
        mood_keywords = data.get("mood_keywords", [])
        data["mood_images"] = get_mood_images(mood_keywords)
        
        # After generating analysis
        log_prompt_test("v1.0", idea, data)
        
        return normalize_analysis(data)
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return normalize_analysis({})

def analyze_idea_with_local(idea: str) -> Dict[str, Any]:
    prompt = build_oracle_prompt(idea)
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json", "options": { "num_predict": 1000}},
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
            
        # Add mood images for local model too
        mood_keywords = data.get("mood_keywords", [])
        data["mood_images"] = get_mood_images(mood_keywords)
        
        return normalize_analysis(data)
    except Exception as e:
        try: 
            print("Ollama raw (truncated):", r.text[:500])
        except: 
            pass
        print(f"Ollama error: {e}")
        return normalize_analysis({})


# -------------------------
# Domain & handle checks
# -------------------------
def domain_available(name: str, ext: str) -> bool:
    """Simple DNS-based domain availability check"""
    domain = f"{name.lower().replace(' ', '').replace('-', '')}{ext}"
    try:
        dns.resolver.resolve(domain, "A")
        return False  # resolves => taken
    except dns.resolver.NXDOMAIN:
        return True   # doesn't exist => likely available
    except Exception:
        try:
            dns.resolver.resolve(domain, "NS")
            return False
        except dns.resolver.NXDOMAIN:
            return True
        except Exception:
            return False  # unknown => conservative

def check_domains(names: List[str], extensions=None) -> Dict[str, Dict[str, bool]]:
    if extensions is None:
        extensions = [".com", ".io", ".ai", ".net", ".co"]
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
    """Quick availability check by requesting profile pages"""
    platforms = {
        "x": "https://x.com/{}",
        "instagram": "https://www.instagram.com/{}/",
        "tiktok": "https://www.tiktok.com/@{}",
    }
    out: Dict[str, Dict[str, Optional[bool]]] = {}
    headers = {"User-Agent": "Mozilla/5.0 (Eye-O Bot)"}
    
    for raw in names:
        handle = raw.lower().replace(" ", "").replace("-", "")
        out[raw] = {}
        for p, url_tpl in platforms.items():
            if not ENABLE_EXTERNAL_CHECKS:
                out[raw][p] = None
                continue
            try:
                r = requests.get(url_tpl.format(handle), headers=headers, timeout=6, allow_redirects=True)
                if r.status_code == 404:
                    out[raw][p] = True  # Available
                elif 200 <= r.status_code < 400:
                    out[raw][p] = False  # Taken
                else:
                    out[raw][p] = None   # Unknown
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

        # Check domains for the selected product name
        selected_name = analysis.get("product", {}).get("selected_name", "")
        domains = check_domains([selected_name]) if selected_name else {}
        handles = check_social_handles([selected_name]) if selected_name else {}

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
    
    # Re-check domains/handles for this record
    selected_name = analysis.get("product", {}).get("selected_name", "")
    domains = check_domains([selected_name]) if selected_name else {}
    handles = check_social_handles([selected_name]) if selected_name else {}
    
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


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0"}

@app.get("/logs")
def get_logs():
    """View recent prompt testing logs"""
    try:
        with open("prompt_tests.jsonl", "r") as f:
            logs = [json.loads(line) for line in f.readlines()]
        return {"logs": logs[-50:], "total": len(logs)}
    except FileNotFoundError:
        return {"logs": [], "total": 0}

@app.get("/logs/analysis")
def log_analysis():
    """Quick analysis of prompt performance"""
    try:
        with open("prompt_tests.jsonl", "r") as f:
            logs = [json.loads(line) for line in f.readlines()]
        
        if not logs:
            return {"message": "No logs found"}
        
        scores = [log.get("viability_score", 0) for log in logs]
        return {
            "total_tests": len(logs),
            "avg_score": sum(scores) / len(scores),
            "score_distribution": {
                "high (7-10)": len([s for s in scores if s >= 7]),
                "medium (4-6)": len([s for s in scores if 4 <= s < 7]),
                "low (1-3)": len([s for s in scores if s < 4])
            }
        }
    except:
        return {"error": "Could not analyze logs"}


@app.get("/download/{record_id}")
async def download_pdf(record_id: str):
    # Get analysis data from database
    with db_conn() as con:
        row = con.execute(
            "SELECT * FROM analyses WHERE id = ?", (record_id,)
        ).fetchone()
        
        if not row:
            raise HTTPException(404, "Analysis not found")
        
        analysis_dict = json.loads(row[2])  # Assuming analysis data is in column 2
        idea = row[1]  # Assuming idea is in column 1
    
    # Create PDF
    filename = f"eye-o-analysis-{record_id}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
    )
    story.append(Paragraph("Eye-O Analysis Report", title_style))
    story.append(Spacer(1, 12))
    
    # Idea
    story.append(Paragraph(f"<b>Idea:</b> {idea}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Score
    score = analysis_dict.get("viability_score", "N/A")
    story.append(Paragraph(f"<b>Viability Score:</b> {score}/10", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Product Name
    product_name = analysis_dict.get("product", {}).get("selected_name", "N/A")
    story.append(Paragraph(f"<b>Product Name:</b> {product_name}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Add more sections as needed
    sections = [
        ("Positioning", analysis_dict.get("product", {}).get("positioning", "")),
        ("Tagline", analysis_dict.get("product", {}).get("tagline", "")),
        ("Target Users", analysis_dict.get("target", {}).get("users", "")),
        ("Revenue Model", analysis_dict.get("revenue_model", "")),
        ("Key Risks", analysis_dict.get("key_risks", [])),
        ("Counter Moves", analysis_dict.get("counter_moves", [])),
        ("Opportunities", analysis_dict.get("opportunities", [])),
        ("Launch Plan", analysis_dict.get("launch_30_day_plan", [])),
        ("Next Steps", analysis_dict.get("next_steps", [])),
    ]
    
    for title, content in sections:
        if content:
            story.append(Paragraph(f"<b>{title}:</b>", styles['Heading2']))
        
        # Handle lists vs strings
        if isinstance(content, list):
            content_text = "<br/>".join([f"• {item}" for item in content])
        else:
            content_text = str(content)
            
        story.append(Paragraph(content_text, styles['Normal']))
        story.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(story)
    
    return FileResponse(
        filename, 
        media_type='application/pdf',
        filename=f"eye-o-analysis-{record_id}.pdf"
    )
    
@app.get("/view/{rec_id}")
async def view_record(rec_id: str, request: Request):
    with db_conn() as con:
        row = con.execute(
            "SELECT * FROM analyses WHERE id = ?", (rec_id,)
        ).fetchone()
        
        if not row:
            raise HTTPException(404, "Analysis not found")
        
        # Convert the JSON string back to dict, then to object
        idea = row[1]
        analysis_dict = json.loads(row[2])  # Assuming analysis data is in column 2
        
        # Convert dict to object with dot notation
        class DictAsAttr:
            def __init__(self, d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        setattr(self, k, DictAsAttr(v))
                    else:
                        setattr(self, k, v)
        
        analysis = DictAsAttr(analysis_dict)
        
        return templates.TemplateResponse("results.html", {
            "request": request,
            "analysis": analysis,
            "idea": idea,
            "record_id": rec_id,
            "analysis_json": analysis_dict,
        })

@app.get("/share/{record_id}")
async def get_share_link(record_id: str, request: Request):
    return {"share_url": f"{request.base_url}view/{record_id}"}

@app.get("/debug/db")
def debug_db():
    with db_conn() as con:
        cursor = con.execute("PRAGMA table_info(analyses)")
        columns = cursor.fetchall()
        
        # Also get a sample row
        sample = con.execute("SELECT * FROM analyses LIMIT 1").fetchone()
        
        return {
            "columns": columns,
            "sample_row": sample
        }


# -------------------------
# Local dev entrypoint
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)