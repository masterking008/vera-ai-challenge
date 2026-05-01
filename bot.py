"""
Vera Bot — Lean v3, optimized for speed and 45+/50 scores.
Faster LLM calls, better prompt engineering, no timeouts.

Run: uvicorn bot_lean_v3:app --host 0.0.0.0 --port 8080
Env: OPENAI_API_KEY=your_key
"""

import os
import time
import re
import json
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
import openai

app = FastAPI()
START = time.time()

contexts: dict[tuple[str, str], dict] = {}
conversations: dict[str, list] = {}
suppressed: set[str] = set()
closed_conversations: set[str] = set()
auto_reply_counts: dict[str, int] = {}

_client: Optional[openai.OpenAI] = None

def get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _client

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZED SYSTEM PROMPT FOR SPEED + 45+/50
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Vera, magicpin's merchant AI assistant on WhatsApp.

HARD RULES: Body ≤320 chars, no URLs, one CTA, no fabrication.

TO HIT 45+/50 — AGGRESSIVE SPECIFICITY + TRIGGER RELEVANCE + ENGAGEMENT:

1. SPECIFICITY (9-10/10): MUST cite TWO concrete facts:
   - Fact 1: A number (CTR%, views, calls, price, count, days, trial_n, rating)
   - Fact 2: A source (JIDA, DCI, peer median, date, merchant name, competitor name, category avg)
   Example: "CTR 6.2% (peer median 4.1%)" or "views 7200 (up 30% vs last month)" or "calls down 50% (vs 3-month avg)"

2. CATEGORY FIT (9-10/10): Match voice exactly — Dentists: "Dr. [name]" + clinical vocab. Salons: warm + first-name. Restaurants: operator-to-operator. Gyms: coaching. Pharmacies: trustworthy.

3. MERCHANT FIT (9-10/10): Owner/customer name in first 3 words + one specific merchant stat (their CTR, lapsed count, offer price) + if Hindi: 4-5 Hindi phrases throughout ("aapke", "mein", "ke liye", "hai", "ho gaye", "se pehle", "abhi", "zaroori")

4. TRIGGER RELEVANCE (9-10/10): WHY NOW signal MUST appear in first 10 words:
   - "Dr. X — JIDA just published:"
   - "DCI deadline Dec 15:"
   - "Hi Y, 6 mahine ho gaye:"
   - "Competitor opened 1.3 km away:"
   - "Festival season is here:"
   - "Your appointment is tomorrow:"
   Generic openings like "I wanted to share" or "Curious about" score 5-6/10.

5. ENGAGEMENT (9-10/10): Binary CTA + consequence (deadline, scarcity, or benefit):
   - "Reply YES — [specific benefit] by [date/time]"
   - "Reply 1 for [slot1], 2 for [slot2]"
   - "Book now — [benefit] + [deadline]"
   Open-ended CTAs like "Book now" or "Let me know" score 7/10.

OUTPUT: {"body": "...", "cta": "binary_yes_no"|"multi_choice_slot"|"open_ended"|"none", "send_as": "vera"|"merchant_on_behalf", "suppression_key": "...", "rationale": "..."}
"""

def compose_message(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    conversation_history: Optional[list] = None,
    is_first_message: bool = True,
) -> dict:
    """Fast, grounded composer."""
    history = conversation_history or []
    trigger_kind = trigger.get("kind", "generic")
    
    # Extract facts
    merchant_id = merchant.get("merchant_id", "")
    merchant_name = merchant.get("identity", {}).get("name", "")
    owner_name = merchant.get("identity", {}).get("owner_first_name", "")
    city = merchant.get("identity", {}).get("city", "")
    languages = merchant.get("identity", {}).get("languages", ["en"])
    hindi = "hi" in languages
    
    category_slug = category.get("slug", "")
    
    # Build compact context
    context_lines = [
        f"MERCHANT: {merchant_name} ({owner_name}), {city}",
        f"CATEGORY: {category_slug}",
        f"LANGUAGES: {languages}"
    ]
    
    # Performance
    perf = merchant.get("performance", {})
    if perf:
        ctr = perf.get("ctr", 0)
        views = perf.get("views", 0)
        calls = perf.get("calls", 0)
        if ctr > 0:
            context_lines.append(f"CTR: {ctr:.1%}")
        if views > 0:
            context_lines.append(f"Views 30d: {views}")
        if calls > 0:
            context_lines.append(f"Calls 30d: {calls}")
        
        peer_stats = category.get("peer_stats", {})
        if peer_stats and ctr > 0:
            peer_ctr = peer_stats.get("avg_ctr", 0)
            if peer_ctr > 0:
                gap = round((ctr - peer_ctr) / peer_ctr * 100)
                context_lines.append(f"CTR vs peer: {gap:+d}%")
    
    # Customer aggregate
    cust_agg = merchant.get("customer_aggregate", {})
    if cust_agg:
        total = cust_agg.get("total_unique_ytd", 0)
        lapsed = cust_agg.get("lapsed_180d_plus", 0)
        if total > 0 and lapsed > 0:
            context_lines.append(f"Lapsed: {lapsed}/{total}")
        high_risk = cust_agg.get("high_risk_adult_count", 0)
        if high_risk > 0:
            context_lines.append(f"High-risk: {high_risk}")
    
    # Offers
    offers = merchant.get("offers", [])
    active_offers = [o for o in offers if o.get("status") == "active"]
    if active_offers:
        offer_strs = [f"{o.get('title')} @ Rs{o.get('price', '?')}" for o in active_offers[:2]]
        context_lines.append(f"Offers: {', '.join(offer_strs)}")
    
    # Trigger
    trigger_payload = trigger.get("payload", {})
    context_lines.append(f"TRIGGER: {trigger_kind}")
    
    # Trigger facts
    if trigger_kind == "research_digest":
        digest_items = category.get("digest", [])
        if digest_items:
            item = digest_items[0]
            context_lines.append(f"Research: {item.get('title', '')[:80]}")
            context_lines.append(f"Source: {item.get('source', '')}")
            if item.get("trial_n"):
                context_lines.append(f"Trial N: {item.get('trial_n')}")
    
    elif trigger_kind == "regulation_change":
        digest_items = category.get("digest", [])
        if digest_items:
            item = digest_items[0]
            context_lines.append(f"Regulation: {item.get('title', '')[:80]}")
            context_lines.append(f"Source: {item.get('source', '')}")
    
    elif trigger_kind in ("perf_dip", "perf_spike"):
        metric = trigger_payload.get("metric", "views")
        delta = trigger_payload.get("delta_pct", 0)
        context_lines.append(f"Metric: {metric}, Change: {delta:+.0%}")
    
    elif trigger_kind == "recall_due":
        context_lines.append(f"Last visit: {trigger_payload.get('last_visit', '')}")
        slots = trigger_payload.get("available_slots", [])
        if slots:
            slot_strs = [s.get("label", "") for s in slots[:2]]
            context_lines.append(f"Slots: {', '.join(slot_strs)}")
    
    elif trigger_kind == "festival_upcoming":
        context_lines.append(f"Festival: {trigger_payload.get('name', '')}, Days: {trigger_payload.get('days_until', '')}")
    
    elif trigger_kind == "competitor_opened":
        context_lines.append(f"Competitor: {trigger_payload.get('distance_km', '')}km away, Rating: {trigger_payload.get('rating', '')}")
    
    elif trigger_kind == "renewal_due":
        context_lines.append(f"Days remaining: {trigger_payload.get('days_until', '')}")
    
    elif trigger_kind == "review_theme_emerged":
        context_lines.append(f"Theme: {trigger_payload.get('theme', '')}, Count: {trigger_payload.get('occurrences', '')}, Sentiment: {trigger_payload.get('sentiment', '')}")
    
    elif trigger_kind == "milestone_reached":
        context_lines.append(f"Milestone: {trigger_payload.get('metric', '')} = {trigger_payload.get('value', '')}")
    
    elif trigger_kind in ("customer_lapsed_soft", "customer_lapsed_hard"):
        context_lines.append(f"Days since visit: {trigger_payload.get('days_since_last_visit', '')}")
    
    elif trigger_kind == "trial_followup":
        slots = trigger_payload.get("next_session_options", [])
        if slots:
            slot_strs = [s.get("label", "") for s in slots[:2]]
            context_lines.append(f"Next slots: {', '.join(slot_strs)}")
    
    # Customer
    if customer:
        cust_id = customer.get("identity", {})
        cust_name = cust_id.get("name", "")
        context_lines.append(f"CUSTOMER: {cust_name}")
    
    context_str = "\n".join(context_lines)
    
    # Compact user prompt with aggressive specificity + trigger relevance + engagement
    user_prompt = f"""Compose WhatsApp message.

{context_str}

AGGRESSIVE RULES FOR 45+/50:
1. SPECIFICITY: Cite TWO facts — a number + a source. Examples: "CTR 6.2% (peer 4.1%)" or "views 7200 (up 30%)" or "calls down 50% (vs avg)"
2. TRIGGER RELEVANCE: WHY NOW in first 10 words. Examples: "Dr. X — JIDA just published:", "6 mahine ho gaye:", "Competitor 1.3 km away:", "Festival season:", "Tomorrow appointment:"
3. ENGAGEMENT: Binary CTA + consequence. Examples: "Reply YES — [benefit] by [date]" or "Reply 1 for [slot], 2 for [slot]"
4. VOICE: First 3 words = "Dr. [owner]" or customer name. Match category tone.
5. LENGTH: ≤320 chars. No URLs. If Hindi: 4-5 phrases throughout.

JSON: {{"body": "...", "cta": "binary_yes_no"|"multi_choice_slot"|"open_ended"|"none", "send_as": "vera"|"merchant_on_behalf", "suppression_key": "...", "rationale": "..."}}
"""
    
    try:
        response = get_client().chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=300,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            timeout=8
        )
        
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
        
        # Enforce constraints
        body = result.get("body", "")
        if len(body) > 320:
            result["body"] = body[:317] + "..."
        result["body"] = re.sub(r"https?://\S+", "", result["body"]).strip()
        
        if not result.get("suppression_key"):
            result["suppression_key"] = trigger.get("suppression_key", f"{trigger_kind}:{merchant_id}")
        
        return result
        
    except Exception as e:
        owner = owner_name or merchant_name[:20]
        return {
            "body": f"Hi {owner}, quick update on your profile — want me to share what I found?",
            "cta": "binary_yes_no",
            "send_as": "vera" if not customer else "merchant_on_behalf",
            "suppression_key": trigger.get("suppression_key", f"{trigger_kind}:{merchant_id}"),
            "rationale": f"Fallback: {str(e)[:40]}"
        }

# ─────────────────────────────────────────────────────────────────────────────
# Reply handler
# ─────────────────────────────────────────────────────────────────────────────

AUTO_REPLY_PHRASES = [
    "thank you for contacting", "our team will respond", "automated response",
    "i am an automated", "this is an automated", "aapki jaankari ke liye bahut-bahut shukriya",
    "main ek automated assistant hoon", "unable to respond right now",
]

OPT_OUT_PHRASES = [
    "stop messaging", "stop sending", "not interested", "please stop",
    "band karo", "mat bhejo", "leave me alone", "do not contact", "unsubscribe",
]

def is_auto_reply(msg: str) -> bool:
    return any(p in msg.lower() for p in AUTO_REPLY_PHRASES)

def is_opt_out(msg: str) -> bool:
    return any(p in msg.lower() for p in OPT_OUT_PHRASES)

def compose_reply(merchant_message: str, conversation_history: list, merchant: dict, category: dict, turn_number: int, merchant_id: str = "") -> dict:
    """Fast reply handler."""
    
    if is_opt_out(merchant_message):
        return {"action": "end", "rationale": "Opted out"}
    
    if is_auto_reply(merchant_message):
        key = merchant_id or ""
        count = auto_reply_counts.get(key, 0) + 1
        auto_reply_counts[key] = count
        
        if count >= 3:
            return {"action": "end", "rationale": "Auto-reply 3x"}
        elif count == 2:
            return {"action": "wait", "wait_seconds": 86400, "rationale": "Auto-reply 2x"}
        else:
            return {
                "action": "send",
                "body": "Looks like auto-reply 🙏 Reply YES when you see this.",
                "cta": "binary_yes_no",
                "rationale": "Auto-reply 1x"
            }
    
    merchant_name = merchant.get("identity", {}).get("name", "")
    category_slug = category.get("slug", "")
    
    history_text = "\n".join(
        f"[{t.get('from', '?').upper()}]: {t.get('body', t.get('msg', ''))[:80]}"
        for t in conversation_history[-4:]
    )
    
    reply_prompt = f"""Merchant: {merchant_name} ({category_slug}), Turn {turn_number}
Message: "{merchant_message}"
History: {history_text}

Reply briefly (≤200 chars). Confirm action or redirect.
JSON: {{"action": "send"|"end", "body": "...", "cta": "...", "rationale": "..."}}
"""
    
    try:
        response = get_client().chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=150,
            temperature=0,
            messages=[{"role": "user", "content": reply_prompt}],
            timeout=5
        )
        
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
        
    except Exception:
        return {
            "action": "send",
            "body": "Got it! What next?",
            "cta": "open_ended",
            "rationale": "Fallback"
        }

# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START),
        "contexts_loaded": counts
    }

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera Challenger",
        "team_members": ["Dinesh Sahu"],
        "model": "GPT-4o-mini (lean v3, optimized for speed + 45+/50)",
        "approach": "Fast, grounded, adaptive",
        "contact_email": "dinesh@dins.in",
        "version": "3.2.0-lean-v3",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "bot_url": "https://api.dins.in",
    }

class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str

@app.post("/v1/context")
async def push_context(body: CtxBody):
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": datetime.now(timezone.utc).isoformat()}

class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []

@app.post("/v1/tick")
async def tick(body: TickBody):
    eligible = []
    
    for trg_id in body.available_triggers:
        if len(eligible) >= 20:
            break
        
        trg_entry = contexts.get(("trigger", trg_id))
        if not trg_entry:
            continue
        
        trigger = trg_entry["payload"]
        sup_key = trigger.get("suppression_key", "")
        if sup_key and sup_key in suppressed:
            continue
        
        merchant_id = trigger.get("merchant_id") or trigger.get("payload", {}).get("merchant_id")
        if not merchant_id:
            continue
        
        merch_entry = contexts.get(("merchant", merchant_id))
        if not merch_entry:
            continue
        
        merchant = merch_entry["payload"]
        cat_entry = contexts.get(("category", merchant.get("category_slug")))
        if not cat_entry:
            continue
        
        category = cat_entry["payload"]
        
        customer = None
        customer_id = trigger.get("customer_id") or trigger.get("payload", {}).get("customer_id")
        if customer_id:
            ce = contexts.get(("customer", customer_id))
            if ce:
                customer = ce["payload"]
        
        conv_id = f"conv_{merchant_id}_{trg_id}"
        if conv_id in closed_conversations:
            continue
        
        is_first = conv_id not in conversations
        history = conversations.get(conv_id, [])
        
        eligible.append((trg_id, category, merchant, trigger, customer, merchant_id, conv_id, is_first, history, sup_key))
    
    if not eligible:
        return {"actions": []}
    
    actions = []
    for trg_id, category, merchant, trigger, customer, merchant_id, conv_id, is_first, history, sup_key in eligible:
        composed = compose_message(
            category=category,
            merchant=merchant,
            trigger=trigger,
            customer=customer,
            conversation_history=history,
            is_first_message=is_first
        )
        
        key = composed.get("suppression_key") or sup_key
        if key:
            suppressed.add(key)
        
        conversations.setdefault(conv_id, []).append({
            "from": "bot",
            "body": composed["body"],
            "ts": body.now
        })
        
        merchant_name = merchant.get("identity", {}).get("name", "")
        
        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": trigger.get("customer_id"),
            "send_as": composed.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
            "template_params": [merchant_name, composed["body"][:100], composed.get("cta", "")],
            "body": composed["body"],
            "cta": composed.get("cta", "open_ended"),
            "suppression_key": key,
            "rationale": composed.get("rationale", ""),
        })
    
    return {"actions": actions}

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_id = body.conversation_id
    conversations.setdefault(conv_id, []).append({
        "from": body.from_role,
        "msg": body.message,
        "ts": body.received_at
    })
    
    merchant: dict = {}
    category: dict = {}
    
    if body.merchant_id:
        m = contexts.get(("merchant", body.merchant_id))
        if m:
            merchant = m["payload"]
            cat = contexts.get(("category", merchant.get("category_slug", "")))
            if cat:
                category = cat["payload"]
    
    result = compose_reply(
        body.message,
        conversations[conv_id],
        merchant,
        category,
        body.turn_number,
        body.merchant_id or ""
    )
    
    if result.get("action") == "send":
        conversations[conv_id].append({
            "from": "bot",
            "body": result.get("body", ""),
            "ts": body.received_at
        })
    elif result.get("action") == "end":
        closed_conversations.add(conv_id)
    
    return result

@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    suppressed.clear()
    closed_conversations.clear()
    auto_reply_counts.clear()
    return {"status": "wiped"}
