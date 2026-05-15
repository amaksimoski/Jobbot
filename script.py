"""
Daily job hunter for Aleksander.
- Pulls jobs from JSearch API (Indeed + LinkedIn + Glassdoor)
- Filters by GTA + Remote Canada + target keywords
- Scores fit 1-10 using Claude
- For 8+ matches: tailors resume + cover letter, emails as PDF attachments
- Generates an HTML dashboard published to GitHub Pages
- Sends short Telegram digest with link to dashboard
"""

import os
import json
import smtplib
import re
import html
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from pathlib import Path

import requests
from anthropic import Anthropic
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT

# --- Required secrets ---
BOT_TOKEN = os.environ["BOT_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]
RAPIDAPI_KEY = os.environ["RAPIDAPI_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
GMAIL_USER = os.environ["GMAIL_USER"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
EMAIL_TO = os.environ.get("EMAIL_TO", GMAIL_USER)
PHONE = os.environ.get("PHONE", "")            # kept out of public repo
SITE_URL = os.environ.get("SITE_URL", "")      # set after first Pages deploy

# --- Config ---
STATE_FILE = "state.json"
MATCH_THRESHOLD = 8
MAX_JOBS_PER_QUERY = 10
MAX_NEW_JOBS_TO_PROCESS = 12
DASHBOARD_HISTORY_DAYS = 7

SEARCHES = [
    {"query": "network engineer in Toronto, Canada", "remote_only": False},
    {"query": "NOC engineer in Toronto, Canada", "remote_only": False},
    {"query": "ISP network operations in Canada", "remote_only": True},
    {"query": "sales engineer in Toronto, Canada", "remote_only": False},
    {"query": "technical sales engineer in Canada", "remote_only": True},
    {"query": "junior trader in Toronto, Canada", "remote_only": False},
    {"query": "ETF analyst in Toronto, Canada", "remote_only": False},
    {"query": "brokerage analyst in Toronto, Canada", "remote_only": False},
]

GTA_CITIES = {
    "toronto", "mississauga", "brampton", "oakville", "burlington",
    "milton", "georgetown", "halton hills", "vaughan", "markham",
    "richmond hill", "north york", "etobicoke", "scarborough",
    "ajax", "pickering", "whitby", "oshawa", "newmarket", "aurora",
    "concord", "thornhill", "woodbridge",
}

CONTACT_LINE = (
    f"Georgetown, ON | {PHONE} | alek.mak12@gmail.com"
    if PHONE else "Georgetown, ON | alek.mak12@gmail.com"
)

RESUME_BASE = {
    "name": "Aleksander Maksimoski",
    "contact": CONTACT_LINE,
    "summary": "Network Operations Engineer with 3+ years maintaining 99.99%+ availability on Bell Canada's nationwide core. Hands-on with BGP, MPLS, CGNAT, and L2/L3 distributed systems. Strong incident-response track record on Nokia and Juniper platforms.",
    "experience": [
        {
            "title": "Network Engineer",
            "company": "Bell Canada",
            "dates": "2023 – Present",
            "bullets": [
                "Maintain maximum service availability for nationwide core networks; monitor and support day-to-day systems.",
                "Provide L2 technical support for mission-critical protocols including BGP and MPLS.",
                "Troubleshoot and remediate incidents on ISP-level hardware: Nokia 7750, Juniper MX960, Cisco 3750.",
                "Lead outage bridges and coordinate across engineering and operations teams on customer-impacting events.",
                "Support network stability and maintenance for critical support units.",
            ],
        }
    ],
    "education": [
        "Master of Applied Science (MASc), Computer Networking — Toronto Metropolitan University, 2023",
        "Humber Real Estate Program — Course 4 (in progress)",
    ],
    "skills": [
        "Networking: BGP, MPLS, CGNAT, L2/L3 distributed systems",
        "Hardware: Nokia 7750, Juniper MX960, Cisco 3750",
        "Incident response: outage bridges, root cause analysis, shift support",
        "Research: Deep Learning for Botnet Detection (Master's thesis)",
    ],
}

client = Anthropic(api_key=ANTHROPIC_API_KEY)


# ─────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────

def load_state():
    if Path(STATE_FILE).exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"seen_ids": [], "history": [], "last_run": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ─────────────────────────────────────────────────────────────
# Job fetching
# ─────────────────────────────────────────────────────────────

def fetch_jobs(query, remote_only):
    url = "https://jsearch.p.rapidapi.com/search"
    params = {
        "query": query,
        "page": "1",
        "num_pages": "1",
        "date_posted": "3days",
        "country": "ca",
    }
    if remote_only:
        params["work_from_home"] = "true"

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "jsearch.p.rapidapi.com",
    }

    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        data = r.json().get("data") or []
    except Exception as e:
        print(f"  [!] fetch failed for '{query}': {e}")
        return []

    jobs = []
    for j in data[:MAX_JOBS_PER_QUERY]:
        jobs.append({
            "id": j.get("job_id"),
            "title": j.get("job_title", ""),
            "company": j.get("employer_name", ""),
            "city": (j.get("job_city") or "").lower(),
            "country": j.get("job_country", ""),
            "is_remote": bool(j.get("job_is_remote")),
            "description": j.get("job_description", "")[:4000],
            "apply_link": j.get("job_apply_link") or j.get("job_google_link", ""),
            "posted": j.get("job_posted_at_datetime_utc", ""),
            "employment_type": j.get("job_employment_type", ""),
        })
    return jobs


def passes_location_filter(job):
    if job["is_remote"] and job["country"] == "CA":
        return True
    if job["city"] in GTA_CITIES:
        return True
    return False


# ─────────────────────────────────────────────────────────────
# Claude: scoring + tailoring
# ─────────────────────────────────────────────────────────────

def score_job(job):
    prompt = f"""Rate how well this job fits the candidate. Output ONLY two lines:
SCORE: <integer 1-10>
REASON: <one short sentence, no fluff>

Candidate:
- Network Engineer at Bell Canada, 3+ years (BGP, MPLS, CGNAT, Nokia/Juniper/Cisco)
- MASc Computer Networking, TMU 2023
- Master's thesis on Deep Learning for Botnet Detection
- Currently studying Humber Real Estate Course 4 (side interest)
- Open to: network ops/engineering, sales engineer / technical sales, junior trading or ETF/brokerage analyst (beginner in finance)
- Location: GTA or remote Canada

Job:
Title: {job['title']}
Company: {job['company']}
Location: {job['city']}, {job['country']} (remote: {job['is_remote']})
Description: {job['description'][:2500]}

Scoring guidance:
- 9-10: Direct fit (network engineer, NOC, sales engineer with networking, ISP role)
- 7-8: Adjacent fit (cloud/sysadmin requiring network skills, junior trading/ETF for finance interest, sales eng without networking)
- 5-6: Weak fit (general IT, unrelated sales, senior finance role)
- 1-4: Poor fit (frontend dev, marketing, unrelated)
"""
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text
        score_match = re.search(r"SCORE:\s*(\d+)", text)
        reason_match = re.search(r"REASON:\s*(.+)", text)
        score = int(score_match.group(1)) if score_match else 0
        reason = reason_match.group(1).strip() if reason_match else ""
        return score, reason
    except Exception as e:
        print(f"  [!] score failed: {e}")
        return 0, "scoring failed"


def tailor_resume_and_letter(job):
    prompt = f"""Tailor a resume summary, 5 resume bullets, and a cover letter for this job. The candidate's master resume is below.

CRITICAL voice rules — the output MUST NOT sound AI-generated:
- NO em-dashes (—). Use periods or commas.
- NO tricolons ("efficient, scalable, and robust"; "fast, reliable, and secure")
- NO openers like "I am excited to" / "I am writing to express my interest" / "I am passionate about"
- NO words: leverage, synergy, robust, cutting-edge, dynamic, proven track record, results-driven, detail-oriented, team player, passionate
- Match the candidate's existing tone: clipped, technical, concrete numbers, active verbs (Maintain, Provide, Execute, Lead, Support)
- Vary sentence lengths. Mix short (5-8 words) with medium (12-18 words). No two consecutive sentences the same length.
- Cover letter must reference something specific from the job description (a tool, protocol, product, or company detail). No generic "your company is great."
- Cover letter ~180-240 words, three short paragraphs.

Candidate master resume:
{json.dumps(RESUME_BASE, indent=2)}

Job:
Title: {job['title']}
Company: {job['company']}
Description: {job['description'][:3500]}

Output ONLY valid JSON in this exact schema, no preamble, no markdown fences:
{{
  "summary": "2-3 sentences tailored to this role",
  "bullets": ["bullet 1", "bullet 2", "bullet 3", "bullet 4", "bullet 5"],
  "cover_letter": "Three short paragraphs separated by \\n\\n. Start with hook tied to the job. Middle paragraph: relevant experience with one concrete number. Close: short, no 'I look forward to hearing from you' cliche."
}}"""
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)
    except Exception as e:
        print(f"  [!] tailoring failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# PDF generation
# ─────────────────────────────────────────────────────────────

def build_resume_pdf(tailored, job, out_path):
    doc = SimpleDocTemplate(
        out_path, pagesize=letter,
        leftMargin=0.7*inch, rightMargin=0.7*inch,
        topMargin=0.6*inch, bottomMargin=0.6*inch,
    )
    styles = getSampleStyleSheet()
    name_style = ParagraphStyle("Name", parent=styles["Heading1"],
                                fontSize=16, spaceAfter=2, alignment=TA_LEFT)
    contact_style = ParagraphStyle("Contact", parent=styles["Normal"],
                                   fontSize=9, spaceAfter=10)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                        fontSize=11, spaceBefore=8, spaceAfter=4,
                        textColor="#222222")
    body = ParagraphStyle("Body", parent=styles["Normal"],
                          fontSize=10, leading=13, spaceAfter=3)

    flow = []
    flow.append(Paragraph(RESUME_BASE["name"], name_style))
    flow.append(Paragraph(RESUME_BASE["contact"], contact_style))

    flow.append(Paragraph("PROFESSIONAL SUMMARY", h2))
    flow.append(Paragraph(tailored["summary"], body))

    flow.append(Paragraph("EXPERIENCE", h2))
    exp = RESUME_BASE["experience"][0]
    flow.append(Paragraph(f"<b>{exp['company']}</b> &nbsp;|&nbsp; {exp['title']} &nbsp;|&nbsp; {exp['dates']}", body))
    for b in tailored["bullets"]:
        flow.append(Paragraph(f"• {b}", body))

    flow.append(Paragraph("EDUCATION", h2))
    for e in RESUME_BASE["education"]:
        flow.append(Paragraph(f"• {e}", body))

    flow.append(Paragraph("TECHNICAL SKILLS", h2))
    for s in RESUME_BASE["skills"]:
        flow.append(Paragraph(f"• {s}", body))

    doc.build(flow)


def build_cover_letter_pdf(tailored, job, out_path):
    doc = SimpleDocTemplate(
        out_path, pagesize=letter,
        leftMargin=1*inch, rightMargin=1*inch,
        topMargin=1*inch, bottomMargin=1*inch,
    )
    styles = getSampleStyleSheet()
    header = ParagraphStyle("Hd", parent=styles["Normal"], fontSize=10, spaceAfter=12)
    body = ParagraphStyle("Bd", parent=styles["Normal"],
                          fontSize=11, leading=15, spaceAfter=11)

    flow = []
    flow.append(Paragraph(RESUME_BASE["name"], header))
    flow.append(Paragraph(RESUME_BASE["contact"], header))
    flow.append(Spacer(1, 12))
    flow.append(Paragraph(datetime.now().strftime("%B %d, %Y"), body))
    flow.append(Paragraph(f"Re: {job['title']} — {job['company']}", body))
    flow.append(Spacer(1, 8))

    for para in tailored["cover_letter"].split("\n\n"):
        flow.append(Paragraph(para.strip(), body))

    flow.append(Spacer(1, 8))
    flow.append(Paragraph("Sincerely,", body))
    flow.append(Paragraph(RESUME_BASE["name"], body))

    doc.build(flow)


# ─────────────────────────────────────────────────────────────
# Email
# ─────────────────────────────────────────────────────────────

def send_email_with_attachments(job, score, reason, resume_path, letter_path):
    msg = MIMEMultipart()
    msg["From"] = GMAIL_USER
    msg["To"] = EMAIL_TO
    msg["Subject"] = f"[Job {score}/10] {job['title']} @ {job['company']}"

    body = f"""Match score: {score}/10
Why: {reason}

Apply: {job['apply_link']}

Resume and cover letter attached. Review before sending.
"""
    msg.attach(MIMEText(body, "plain"))

    for path, name in [(resume_path, "resume.pdf"), (letter_path, "cover_letter.pdf")]:
        with open(path, "rb") as f:
            att = MIMEApplication(f.read(), _subtype="pdf")
        att.add_header("Content-Disposition", "attachment", filename=name)
        msg.attach(att)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        s.send_message(msg)


# ─────────────────────────────────────────────────────────────
# HTML dashboard
# ─────────────────────────────────────────────────────────────

def build_dashboard(today_scored, history, generated_at):
    today_str = generated_at.strftime("%A, %B %d, %Y")
    high = [j for j in today_scored if j["score"] >= MATCH_THRESHOLD]
    mid = [j for j in today_scored if 5 <= j["score"] < MATCH_THRESHOLD]

    def card(j):
        score = j["score"]
        tier = "high" if score >= 8 else ("mid" if score >= 5 else "low")
        emailed_badge = '<span class="badge-emailed">resume sent</span>' if j.get("emailed") else ""
        remote_badge = '<span class="badge-remote">remote</span>' if j.get("is_remote") else ""
        city = html.escape(j.get("city", "").title() or "—")
        reason = html.escape(j.get("reason", ""))
        return f"""
        <article class="card card-{tier}">
            <div class="card-head">
                <div class="score score-{tier}">{score}<span>/10</span></div>
                <div class="card-meta">
                    <div class="company">{html.escape(j['company'])}</div>
                    <div class="location">{city}{remote_badge}</div>
                </div>
                {emailed_badge}
            </div>
            <h3 class="title"><a href="{html.escape(j['apply_link'])}" target="_blank" rel="noopener">{html.escape(j['title'])}</a></h3>
            <p class="reason">{reason}</p>
            <a class="apply" href="{html.escape(j['apply_link'])}" target="_blank" rel="noopener">Apply →</a>
        </article>"""

    def section(title, jobs, empty_msg):
        if not jobs:
            return f'<section class="section"><h2>{title}</h2><p class="empty">{empty_msg}</p></section>'
        cards_html = "\n".join(card(j) for j in jobs)
        return f'<section class="section"><h2>{title} <span class="count">{len(jobs)}</span></h2><div class="cards">{cards_html}</div></section>'

    history_html = ""
    if history:
        items = []
        for day in history[-DASHBOARD_HISTORY_DAYS:][::-1]:
            jobs = day.get("jobs", [])
            high_n = sum(1 for j in jobs if j["score"] >= MATCH_THRESHOLD)
            mid_n = sum(1 for j in jobs if 5 <= j["score"] < MATCH_THRESHOLD)
            items.append(
                f'<li><span class="hist-date">{html.escape(day["date"])}</span>'
                f'<span class="hist-stats">{high_n} high · {mid_n} mid · {len(jobs)} total</span></li>'
            )
        history_html = f"""
        <section class="section history">
            <h2>Last 7 days</h2>
            <ul class="history-list">{"".join(items)}</ul>
        </section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Job Hunt — {today_str}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=JetBrains+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
    --bg: #faf8f3;
    --bg-card: #ffffff;
    --bg-card-hover: #fdfcf7;
    --border: #e8e4d8;
    --border-strong: #d4cfc0;
    --text: #1a1816;
    --text-dim: #5a564f;
    --text-faint: #8b857a;
    --accent: #b8654a;
    --high: #6b8e6b;
    --mid: #c8983a;
    --low: #a8a29e;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', system-ui, sans-serif;
    line-height: 1.5;
    padding: 40px 20px 60px;
}}
.container {{ max-width: 1100px; margin: 0 auto; }}
header {{ border-bottom: 1px solid var(--border); padding-bottom: 28px; margin-bottom: 36px; }}
.eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--accent);
    margin-bottom: 8px;
}}
h1 {{
    font-family: 'Fraunces', serif;
    font-weight: 500;
    font-size: 44px;
    line-height: 1.1;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
}}
h1 em {{ font-style: italic; color: var(--accent); font-weight: 400; }}
.subtitle {{ color: var(--text-dim); font-size: 15px; }}
.stats {{
    display: flex;
    gap: 24px;
    margin-top: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: var(--text-dim);
}}
.stats strong {{ color: var(--text); font-weight: 500; }}
.section {{ margin-bottom: 44px; }}
.section h2 {{
    font-family: 'Fraunces', serif;
    font-weight: 500;
    font-size: 24px;
    margin-bottom: 18px;
    letter-spacing: -0.01em;
}}
.section h2 .count {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: var(--text-faint);
    font-weight: 400;
    margin-left: 8px;
}}
.empty {{ color: var(--text-faint); font-style: italic; }}
.cards {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 16px;
}}
.card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 20px;
    position: relative;
    transition: background 0.15s, border-color 0.15s;
    animation: fadeIn 0.4s ease-out backwards;
}}
.card:hover {{ background: var(--bg-card-hover); border-color: var(--border-strong); }}
.card::before {{
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 4px 0 0 4px;
}}
.card-high::before {{ background: var(--high); }}
.card-mid::before {{ background: var(--mid); }}
.card-low::before {{ background: var(--low); }}
.card-head {{
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin-bottom: 12px;
}}
.score {{
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 32px;
    line-height: 1;
    letter-spacing: -0.02em;
}}
.score span {{ font-size: 13px; font-weight: 400; color: var(--text-faint); }}
.score-high {{ color: var(--high); }}
.score-mid {{ color: var(--mid); }}
.score-low {{ color: var(--low); }}
.card-meta {{ flex: 1; min-width: 0; }}
.company {{
    font-weight: 600;
    font-size: 14px;
    color: var(--text);
    margin-bottom: 2px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}
.location {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-faint);
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
.badge-remote, .badge-emailed {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 2px 6px;
    margin-left: 6px;
    border-radius: 2px;
}}
.badge-remote {{ background: rgba(184,101,74,0.1); color: var(--accent); }}
.badge-emailed {{
    background: var(--high);
    color: white;
    margin-left: auto;
    align-self: flex-start;
}}
.title {{
    font-family: 'Fraunces', serif;
    font-weight: 500;
    font-size: 18px;
    line-height: 1.25;
    margin-bottom: 10px;
    letter-spacing: -0.01em;
}}
.title a {{ color: var(--text); text-decoration: none; }}
.title a:hover {{ color: var(--accent); }}
.reason {{
    font-size: 13px;
    color: var(--text-dim);
    margin-bottom: 16px;
    line-height: 1.5;
}}
.apply {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--accent);
    text-decoration: none;
    border-bottom: 1px solid currentColor;
    padding-bottom: 1px;
    transition: color 0.15s;
}}
.apply:hover {{ color: var(--text); }}
.history-list {{
    list-style: none;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px 0;
}}
.history-list li {{
    display: flex;
    justify-content: space-between;
    padding: 10px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    border-bottom: 1px solid var(--border);
}}
.history-list li:last-child {{ border-bottom: none; }}
.hist-date {{ color: var(--text); }}
.hist-stats {{ color: var(--text-faint); }}
footer {{
    margin-top: 60px;
    padding-top: 24px;
    border-top: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-faint);
    text-align: center;
}}
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(4px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
.card:nth-child(1) {{ animation-delay: 0.0s; }}
.card:nth-child(2) {{ animation-delay: 0.05s; }}
.card:nth-child(3) {{ animation-delay: 0.1s; }}
.card:nth-child(4) {{ animation-delay: 0.15s; }}
.card:nth-child(5) {{ animation-delay: 0.2s; }}
.card:nth-child(6) {{ animation-delay: 0.25s; }}
.card:nth-child(n+7) {{ animation-delay: 0.3s; }}
@media (max-width: 600px) {{
    body {{ padding: 24px 14px 40px; }}
    h1 {{ font-size: 32px; }}
    .cards {{ grid-template-columns: 1fr; }}
    .stats {{ flex-direction: column; gap: 8px; }}
}}
</style>
</head>
<body>
<div class="container">
    <header>
        <div class="eyebrow">Daily brief</div>
        <h1>Job <em>hunt</em></h1>
        <p class="subtitle">{today_str}</p>
        <div class="stats">
            <span><strong>{len(high)}</strong> high match</span>
            <span><strong>{len(mid)}</strong> worth a look</span>
            <span><strong>{len(today_scored)}</strong> new scored</span>
        </div>
    </header>

    {section("High match — resume sent", high, "No high matches today.")}
    {section("Worth a look", mid, "No mid matches today.")}
    {history_html}

    <footer>
        Generated {generated_at.strftime('%Y-%m-%d %H:%M UTC')} · Data: JSearch (Indeed + LinkedIn + Glassdoor)
    </footer>
</div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────
# Telegram
# ─────────────────────────────────────────────────────────────

def send_telegram(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        r = requests.post(url, json={
            "chat_id": CHAT_ID,
            "parse_mode": "HTML",
            "text": chunk,
            "disable_web_page_preview": False,
        }, timeout=30)
        r.raise_for_status()


def format_digest(scored_jobs, emailed_count):
    today = datetime.now().strftime("%a %b %d")
    high = [j for j in scored_jobs if j["score"] >= MATCH_THRESHOLD]
    mid = [j for j in scored_jobs if 5 <= j["score"] < MATCH_THRESHOLD]

    lines = [f"<b>🎯 Job Hunt — {today}</b>", ""]
    if high or mid:
        lines.append(f"🔥 {len(high)} high match · 👀 {len(mid)} worth a look")
        if emailed_count:
            lines.append(f"📧 {emailed_count} resume(s) emailed")
    else:
        lines.append("Nothing matched today.")

    if SITE_URL:
        lines.append("")
        lines.append(f'👉 <a href="{SITE_URL}">Open dashboard</a>')

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    now = datetime.now(timezone.utc)
    print(f"Run: {now.isoformat()}")
    state = load_state()
    seen = set(state["seen_ids"])

    # 1. Fetch
    all_jobs = []
    for s in SEARCHES:
        print(f"  Fetching: {s['query']} (remote={s['remote_only']})")
        all_jobs.extend(fetch_jobs(s["query"], s["remote_only"]))

    # 2. Dedupe + filter
    by_id = {}
    for j in all_jobs:
        if not j["id"] or j["id"] in seen:
            continue
        if not passes_location_filter(j):
            continue
        by_id[j["id"]] = j

    new_jobs = list(by_id.values())[:MAX_NEW_JOBS_TO_PROCESS]
    print(f"  {len(new_jobs)} new jobs after dedupe + filters")

    scored = []
    emailed = 0

    if new_jobs:
        # 3. Score
        for j in new_jobs:
            print(f"  Scoring: {j['title']} @ {j['company']}")
            score, reason = score_job(j)
            scored.append({**j, "score": score, "reason": reason, "emailed": False})

        scored.sort(key=lambda x: -x["score"])

        # 4. Tailor + email high matches
        Path("/tmp/job-bot").mkdir(exist_ok=True)
        for j in scored:
            if j["score"] < MATCH_THRESHOLD:
                continue
            print(f"  Tailoring: {j['title']} @ {j['company']}")
            tailored = tailor_resume_and_letter(j)
            if not tailored:
                continue
            safe_co = re.sub(r"[^A-Za-z0-9]+", "_", j["company"])[:30]
            resume_pdf = f"/tmp/job-bot/resume_{safe_co}_{j['id'][:8]}.pdf"
            letter_pdf = f"/tmp/job-bot/letter_{safe_co}_{j['id'][:8]}.pdf"
            try:
                build_resume_pdf(tailored, j, resume_pdf)
                build_cover_letter_pdf(tailored, j, letter_pdf)
                send_email_with_attachments(j, j["score"], j["reason"], resume_pdf, letter_pdf)
                j["emailed"] = True
                emailed += 1
                print(f"    ✓ emailed")
            except Exception as e:
                print(f"    [!] email/PDF failed: {e}")

    # 5. Update history
    history = state.get("history", [])
    today_key = now.strftime("%Y-%m-%d")
    history = [h for h in history if h.get("date") != today_key]
    history.append({
        "date": today_key,
        "jobs": [
            {"score": j["score"], "title": j["title"], "company": j["company"]}
            for j in scored
        ],
    })
    history = history[-DASHBOARD_HISTORY_DAYS:]

    # 6. Build dashboard (always; even if 0 new jobs, dashboard reflects state)
    os.makedirs("site", exist_ok=True)
    # pass history excluding today (shown above as today's cards)
    html_doc = build_dashboard(scored, history[:-1] if history else [], now)
    with open("site/index.html", "w", encoding="utf-8") as f:
        f.write(html_doc)
    print("  Dashboard written to site/index.html")

    # 7. Telegram
    digest = format_digest(scored, emailed)
    send_telegram(digest)

    # 8. Save state
    for j in scored:
        seen.add(j["id"])
    state["seen_ids"] = list(seen)[-1000:]
    state["history"] = history
    state["last_run"] = now.isoformat()
    save_state(state)
    print("Done.")


if __name__ == "__main__":
    main()
