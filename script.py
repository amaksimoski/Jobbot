"""
Daily job hunter for Aleksander.
- Pulls jobs from JSearch API (Indeed + LinkedIn + Glassdoor)
- Filters by GTA + Remote Canada + target keywords
- Scores fit 1-10 using Claude
- Generates an HTML dashboard published to GitHub Pages
- Sends short Telegram digest with link to dashboard
"""

import os
import json
import re
import html
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from anthropic import Anthropic

# --- Required secrets ---
BOT_TOKEN = os.environ["BOT_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]
RAPIDAPI_KEY = os.environ["RAPIDAPI_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
SITE_URL = os.environ.get("SITE_URL", "")    # set after first Pages deploy

# --- Config ---
STATE_FILE = "state.json"
MATCH_THRESHOLD = 8
MAX_JOBS_PER_QUERY = 10
MAX_NEW_JOBS_TO_PROCESS = 12
ROLLING_WINDOW_DAYS = 14   # how long a job stays on the dashboard after first seen

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

client = Anthropic(api_key=ANTHROPIC_API_KEY)


# ─────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────

def load_state():
    """State schema:
    {
      "active_jobs": { "job_id": {full job dict with score, reason, first_seen}, ... },
      "daily_counts": [ {"date": "YYYY-MM-DD", "new": N, "high": N, "mid": N}, ... ],
      "last_run": "ISO timestamp"
    }
    """
    if Path(STATE_FILE).exists():
        with open(STATE_FILE) as f:
            data = json.load(f)
        # Migrate old schema if needed (seen_ids -> active_jobs)
        if "active_jobs" not in data:
            data["active_jobs"] = {}
        if "daily_counts" not in data:
            data["daily_counts"] = []
        return data
    return {"active_jobs": {}, "daily_counts": [], "last_run": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def prune_old_jobs(active_jobs, now):
    """Remove jobs older than ROLLING_WINDOW_DAYS."""
    cutoff = now - timedelta(days=ROLLING_WINDOW_DAYS)
    kept = {}
    for jid, job in active_jobs.items():
        try:
            first_seen = datetime.fromisoformat(job["first_seen"])
            if first_seen >= cutoff:
                kept[jid] = job
        except (KeyError, ValueError):
            # Malformed entry, drop it
            pass
    return kept


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
# Claude: scoring
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


# ─────────────────────────────────────────────────────────────
# HTML dashboard
# ─────────────────────────────────────────────────────────────

def build_dashboard(active_jobs, daily_counts, generated_at, today_new_ids):
    """active_jobs: dict of job_id -> full job dict (incl. score, reason, first_seen)
    daily_counts: list of {date, new, high, mid}
    today_new_ids: set of job IDs first seen on this run
    """
    today_str = generated_at.strftime("%A, %B %d, %Y")
    today_date = generated_at.strftime("%Y-%m-%d")

    # Sort all active jobs by score desc, then by first_seen desc (newest first within tier)
    all_jobs = sorted(
        active_jobs.values(),
        key=lambda j: (-int(j.get("score", 0)), j.get("first_seen", "")),
    )

    # Split into today vs earlier, but only within the visible tiers (>=5)
    visible = [j for j in all_jobs if j.get("score", 0) >= 5]
    today_jobs = [j for j in visible if j["id"] in today_new_ids]
    earlier_jobs = [j for j in visible if j["id"] not in today_new_ids]

    high_today = [j for j in today_jobs if j["score"] >= MATCH_THRESHOLD]
    mid_today = [j for j in today_jobs if 5 <= j["score"] < MATCH_THRESHOLD]
    high_earlier = [j for j in earlier_jobs if j["score"] >= MATCH_THRESHOLD]
    mid_earlier = [j for j in earlier_jobs if 5 <= j["score"] < MATCH_THRESHOLD]

    def days_ago(iso_str):
        try:
            seen = datetime.fromisoformat(iso_str)
            delta = (generated_at - seen).days
            if delta == 0:
                return "today"
            if delta == 1:
                return "1 day ago"
            return f"{delta} days ago"
        except Exception:
            return ""

    def card(j):
        score = j["score"]
        tier = "high" if score >= 8 else ("mid" if score >= 5 else "low")
        remote_badge = '<span class="badge-remote">remote</span>' if j.get("is_remote") else ""
        new_badge = '<span class="badge-new">new</span>' if j["id"] in today_new_ids else ""
        city = html.escape(j.get("city", "").title() or "—")
        reason = html.escape(j.get("reason", ""))
        age = days_ago(j.get("first_seen", ""))
        age_html = f'<span class="age">{age}</span>' if age else ""
        return f"""
        <article class="card card-{tier}">
            <div class="card-head">
                <div class="score score-{tier}">{score}<span>/10</span></div>
                <div class="card-meta">
                    <div class="company">{html.escape(j['company'])}{new_badge}</div>
                    <div class="location">{city}{remote_badge}</div>
                </div>
            </div>
            <h3 class="title"><a href="{html.escape(j['apply_link'])}" target="_blank" rel="noopener">{html.escape(j['title'])}</a></h3>
            <p class="reason">{reason}</p>
            <div class="card-foot">
                <a class="apply" href="{html.escape(j['apply_link'])}" target="_blank" rel="noopener">Apply →</a>
                {age_html}
            </div>
        </article>"""

    def section(title, jobs, empty_msg=None):
        if not jobs:
            if empty_msg:
                return f'<section class="section"><h2>{title}</h2><p class="empty">{empty_msg}</p></section>'
            return ""
        cards_html = "\n".join(card(j) for j in jobs)
        return f'<section class="section"><h2>{title} <span class="count">{len(jobs)}</span></h2><div class="cards">{cards_html}</div></section>'

    # Daily history rail (last 14 days of counts)
    history_html = ""
    if daily_counts:
        items = []
        for day in daily_counts[-ROLLING_WINDOW_DAYS:][::-1]:
            items.append(
                f'<li><span class="hist-date">{html.escape(day["date"])}</span>'
                f'<span class="hist-stats">{day.get("high", 0)} high · {day.get("mid", 0)} mid · {day.get("new", 0)} new</span></li>'
            )
        history_html = f"""
        <section class="section history">
            <h2>Daily activity</h2>
            <ul class="history-list">{"".join(items)}</ul>
        </section>"""

    # Build sections in order
    sections_html = ""
    if today_jobs:
        sections_html += section(f"New today — high match", high_today)
        sections_html += section(f"New today — worth a look", mid_today)
    else:
        sections_html += '<section class="section"><h2>New today</h2><p class="empty">No new matches today. Earlier jobs below.</p></section>'

    if high_earlier:
        sections_html += section(f"Still open — high match", high_earlier)
    if mid_earlier:
        sections_html += section(f"Still open — worth a look", mid_earlier)

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
.badge-remote {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 2px 6px;
    margin-left: 6px;
    border-radius: 2px;
    background: rgba(184,101,74,0.1);
    color: var(--accent);
}}
.badge-new {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 2px 6px;
    margin-left: 6px;
    border-radius: 2px;
    background: var(--high);
    color: white;
    vertical-align: 1px;
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
.card-foot {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
}}
.age {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-faint);
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
        <p class="subtitle">{today_str} · showing last {ROLLING_WINDOW_DAYS} days</p>
        <div class="stats">
            <span><strong>{len(today_jobs)}</strong> new today</span>
            <span><strong>{len(high_today) + len(high_earlier)}</strong> high match open</span>
            <span><strong>{len(mid_today) + len(mid_earlier)}</strong> worth a look</span>
            <span><strong>{len(visible)}</strong> total active</span>
        </div>
    </header>

    {sections_html}
    {history_html}

    <footer>
        Generated {generated_at.strftime('%Y-%m-%d %H:%M UTC')} · Data: JSearch (Indeed + LinkedIn + Glassdoor) · Window: {ROLLING_WINDOW_DAYS} days
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


def format_digest(new_today, high_today, mid_today, total_active):
    today = datetime.now().strftime("%a %b %d")
    lines = [f"<b>🎯 Job Hunt — {today}</b>", ""]
    if new_today > 0:
        lines.append(f"🔥 {high_today} new high match · 👀 {mid_today} new worth a look")
    else:
        lines.append("No new matches today.")

    if total_active > 0:
        lines.append(f"📋 {total_active} jobs still open on the board")

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

    # Prune old jobs first (anything beyond 14 days drops out)
    active_jobs = prune_old_jobs(state.get("active_jobs", {}), now)
    pruned_count = len(state.get("active_jobs", {})) - len(active_jobs)
    if pruned_count:
        print(f"  Pruned {pruned_count} job(s) older than {ROLLING_WINDOW_DAYS} days")

    seen_ids = set(active_jobs.keys())

    # 1. Fetch
    all_fetched = []
    for s in SEARCHES:
        print(f"  Fetching: {s['query']} (remote={s['remote_only']})")
        all_fetched.extend(fetch_jobs(s["query"], s["remote_only"]))

    # 2. Dedupe + filter (skip jobs we've already shown in the active window)
    by_id = {}
    for j in all_fetched:
        if not j["id"] or j["id"] in seen_ids:
            continue
        if not passes_location_filter(j):
            continue
        by_id[j["id"]] = j

    new_jobs = list(by_id.values())[:MAX_NEW_JOBS_TO_PROCESS]
    print(f"  {len(new_jobs)} new jobs after dedupe + filters")

    today_new_ids = set()
    high_today_count = 0
    mid_today_count = 0

    if new_jobs:
        # 3. Score and add to active_jobs
        for j in new_jobs:
            print(f"  Scoring: {j['title']} @ {j['company']}")
            score, reason = score_job(j)
            job_record = {
                **j,
                "score": score,
                "reason": reason,
                "first_seen": now.isoformat(),
            }
            active_jobs[j["id"]] = job_record
            today_new_ids.add(j["id"])
            if score >= MATCH_THRESHOLD:
                high_today_count += 1
            elif score >= 5:
                mid_today_count += 1

    # 4. Update daily counts
    daily_counts = state.get("daily_counts", [])
    today_key = now.strftime("%Y-%m-%d")
    daily_counts = [d for d in daily_counts if d.get("date") != today_key]
    daily_counts.append({
        "date": today_key,
        "new": len(today_new_ids),
        "high": high_today_count,
        "mid": mid_today_count,
    })
    daily_counts = daily_counts[-ROLLING_WINDOW_DAYS:]

    # 5. Build dashboard from full active_jobs window
    os.makedirs("site", exist_ok=True)
    html_doc = build_dashboard(active_jobs, daily_counts, now, today_new_ids)
    with open("site/index.html", "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"  Dashboard written: {len(active_jobs)} active jobs in window")

    # 6. Telegram
    digest = format_digest(
        new_today=len(today_new_ids),
        high_today=high_today_count,
        mid_today=mid_today_count,
        total_active=len([j for j in active_jobs.values() if j.get("score", 0) >= 5]),
    )
    send_telegram(digest)

    # 7. Save state
    state["active_jobs"] = active_jobs
    state["daily_counts"] = daily_counts
    state["last_run"] = now.isoformat()
    save_state(state)
    print("Done.")


if __name__ == "__main__":
    main()
