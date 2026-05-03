"""
Vera Bot — context-grounded composer for the magicpin AI Challenge.

Run: uvicorn bot:app --host 0.0.0.0 --port 8080
Env: OPENAI_API_KEY=...
"""

import os
import re
import json
import time
import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from openai import AsyncOpenAI

app = FastAPI()
START = time.time()

contexts: dict[tuple[str, str], dict] = {}
conversations: dict[str, list] = {}
suppressed: set[str] = set()
closed_conversations: set[str] = set()
auto_reply_counts: dict[str, int] = {}
last_merchant_messages: dict[str, list[str]] = {}

COMPOSE_MODEL = os.environ.get("VERA_COMPOSE_MODEL", "gpt-4o")
REPLY_MODEL = os.environ.get("VERA_REPLY_MODEL", "gpt-4o-mini")
COMPOSE_TIMEOUT_S = float(os.environ.get("VERA_COMPOSE_TIMEOUT_S", "13"))
REPLY_TIMEOUT_S = float(os.environ.get("VERA_REPLY_TIMEOUT_S", "8"))
TICK_BUDGET_S = float(os.environ.get("VERA_TICK_BUDGET_S", "14"))

_client: Optional[AsyncOpenAI] = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# Composer
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Vera, magicpin's WhatsApp assistant for Indian local merchants.

You compose ONE message at a time from four context layers:
- category: the kind of business (voice, peer benchmarks, current research/compliance digest, content library)
- merchant: the specific business (identity, performance, offers, signals, review themes, customer aggregates)
- trigger: the concrete event prompting THIS message right now (kind + payload, sometimes pointing at a digest item by id)
- customer (optional): when sending on the merchant's behalf to one of their customers

Every message must satisfy ALL of these:

1. WHY NOW comes first — and the FACT comes before the greeting. Open with the concrete trigger fact, then the addressee. Examples of strong openers: "57 din ho gaye Rashmi ki last visit ko —", "Heads up Ramesh — your stock of metformin runs out 28 Apr.", "Worth knowing, Dr. Meera — Smile Studio opened 1.3 km away offering Cleaning @ ₹199.", "DCI's new radiograph dose limit (1.0 mSv) kicks in 15 Dec, Dr. Meera." Avoid weak openers: "Hi {name}, …", "I wanted to share…", "Festival season is here…". If the trigger references a digest item, its headline + source appear in the first sentence.

2. Cite verifiable facts only from the contexts. Numbers (CTR, peer median, views delta, days, prices), names (merchant, owner, competitor, journal/source), dates. Never invent a stat, study, competitor, slot, or price that is not in the contexts.

3. Anchor on ONE merchant-specific fact (besides the name). Quote a signal verbatim (e.g., "stale_posts:22d", "ctr_below_peer_median", "high_risk_adult_cohort"), or a review_theme + occurrences/common_quote, or an active offer title, or a customer_aggregate number, or a peer_stats comparison vs their actual performance.

4. Match the category voice from category.voice — its tone, register, allowed vocabulary, salutation pattern. Avoid every word in vocab_taboo. Dentists/doctors get clinical-peer tone; pharmacies get trustworthy-precise; salons get warm-practical; restaurants get operator-to-operator; gyms get coach tone.

5. Honor language. If merchant.identity.languages includes 'hi' (or customer.identity.language_pref mentions hi/te/ta/mr/kn/bn), code-mix Hindi/English naturally — only as much as a real Indian would. Pure English if 'hi' is absent. Don't bolt Hindi phrases onto an English sentence; either commit or don't.

6. Address by the OWNER'S FIRST NAME (or "Dr. {owner_first_name}" for dentists/doctors). Customer-facing: use customer.identity.name. Never open with the business name as the salutation — the business name belongs in context, not as the greeting. The name appears AFTER the opening fact, not before it.

7. End with ONE binary CTA that bundles a concrete consequence — never a bare "Reply YES". Patterns that work:
   - "Reply YES and I'll draft 3 posts and send by tonight."
   - "Reply 1 for Wed 6pm, 2 for Thu 5pm — I'll block the slot."
   - "Reply YES — I'll renew + lock in the ₹4999 Pro plan now."
   - "Reply YES and I'll pull the full abstract + a patient-ed WhatsApp you can share."
   For action triggers (recall_due, refill_due, renewal_due, trial_followup, booking) use multi_choice_slot with payload slot labels OR binary_yes_no with the slot in the body. For information triggers use binary_yes_no. Use open_ended ONLY for curious_ask_due.

8. Be tight. 2–4 sentences, ≤320 chars. No URLs. No multi-CTA. No re-introducing yourself. No "I hope you're well" preambles.

9. send_as = "merchant_on_behalf" when a customer context is present; otherwise "vera".

GOLD-STANDARD EXAMPLES (for tone, structure, density — do not copy verbatim):

A) research_digest, merchant=Dr. Meera (dentist), digest=JIDA Oct fluoride study, signal=high_risk_adult_cohort:
"Dr. Meera, JIDA's Oct issue landed. One item is relevant to your high-risk adult cohort — a 2,100-patient trial shows 3-mo fluoride recall cuts caries recurrence 38% better than 6-mo. Reply YES and I'll pull the abstract + draft a patient-ed WhatsApp you can share. — JIDA Oct 2026 p.14"
Why: named source + month, ties to merchant's actual signal, trial_n + effect size, binary CTA bundling 2 concrete deliverables.

B) customer_lapsed_hard, customer=Rashmi (gym, weight_loss focus, 57 days lapsed, 22 prior visits), merchant=PowerHouse Gym:
"57 din ho gaye Rashmi — your weight-loss program at PowerHouse was 22 sessions strong. Apke liye ek 2-week comeback plan ready hai (₹0 reactivation, evening 7am slot pe). Reply YES — main aaj hi reserve kar deti hoon."
Why: leads with the days_since fact + name, names previous_focus + visits_total, frames a low-friction comeback ("2-week, ₹0 reactivation"), binary CTA with effort-externalization.

C) trial_followup, customer=Karthik (kids yoga, trial 22 Apr, slot Sat 3 May 8am), merchant=Zen Yoga:
"Karthik's trial on 22 Apr was the right age-fit, Sumitra. Sat 3 May 8am ka slot reserved hai — 4 kids confirmed already, 2 spots bache hain. Reply 1 to lock Karthik's spot, 2 if Sat doesn't work and I'll find another slot."
Why: names trial_date + parent (whatsapp_via_parent), creates social_proof (4 kids confirmed) + scarcity (2 spots left), multi-choice CTA with both paths handled.

Across all examples: opener anchors a fact, body weaves a merchant-specific data point, CTA bundles a deliverable. NEVER open with "Hi {name}, I wanted to share..." — that loses 2-3 points immediately.

Output ONLY this JSON:
{"body": "...", "cta": "binary_yes_no" | "multi_choice_slot" | "open_ended" | "none", "send_as": "vera" | "merchant_on_behalf", "suppression_key": "...", "rationale": "<one short line: why this message + what fact anchors it>"}
"""


# Per-trigger framing nudges. Short hints, NOT canned phrases.
TRIGGER_FRAMING = {
    "research_digest": "Lead with the digest item's headline and source. Tie it to the merchant's situation (their signals / customer cohort).",
    "regulation_change": "Lead with the regulation, the deadline, and what specifically the merchant must do.",
    "perf_dip": "Lead with the metric + delta + window. Offer one concrete next step (draft posts, audit listing, fix offers).",
    "perf_spike": "Lead with the metric + delta. Suggest amplifying the likely_driver while it's hot.",
    "seasonal_perf_dip": "Acknowledge the seasonality up front so it doesn't sound like an alarm. Suggest a category-appropriate counter-move.",
    "recall_due": "Lead with the service due + when their last visit was. Offer the actual slot labels from payload.",
    "festival_upcoming": "If days_until <= 14, lead with name + days_until + a specific category-relevant beat. If days_until > 14, frame as 'planning ahead' and propose ONE concrete prep action (early-bird slot block, menu development, festive-package post draft) — never just 'festival season is coming'.",
    "competitor_opened": "Name the competitor + distance + their offer. Frame as 'worth knowing', not panic.",
    "renewal_due": "State days_remaining + plan + amount. Offer one-tap renew.",
    "review_theme_emerged": "Cite the theme + occurrences + the actual quote. Suggest a 1-step fix.",
    "milestone_reached": "Name the metric + value_now + how close to milestone_value. Suggest celebrating publicly (a GBP post / customer thank-you).",
    "customer_lapsed_soft": "Name days_since_last_visit. Light-touch reach-out.",
    "customer_lapsed_hard": "Name days_since_last_visit + previous_focus. Specific winback hook.",
    "trial_followup": "Reference the trial_date and offer the next_session_options as 1/2 multi-choice.",
    "supply_alert": "Lead with molecule + affected batches. Tell merchant the exact action + deadline.",
    "chronic_refill_due": "Name the molecules + that stock runs out on stock_runs_out_iso. Offer free home delivery if delivery_address_saved.",
    "category_seasonal": "Name the season + the top trend (e.g., '+45% antifungal demand'). Suggest one shelf action.",
    "winback_eligible": "Name days_since_expiry + lapsed_customers_added. Offer a renew + winback combo.",
    "dormant_with_vera": "Reopen on the merchant's last_topic from the payload — be specific about that topic, not generic. Add one concrete merchant fact (a signal or review theme) so it lands as 'I noticed X about your account'. Binary CTA with a deliverable.",
    "ipl_match_today": "Name the match + venue + match_time_iso. Suggest a category-specific play (combo, late hours, delivery push).",
    "wedding_package_followup": "Name the wedding_date + days_to_wedding + the next_step_window_open. Soft, warm.",
    "curious_ask_due": "Open with a peer-style observation that anchors WHY you're asking (a signal, peer_stat, or trend_signal from category), then the specific question matching ask_template. cta = open_ended. No pitch, but make the observation merchant-specific so it doesn't feel like a survey.",
    "active_planning_intent": "Continue the merchant's intent_topic from where their merchant_last_message left off. Action-mode, not qualification.",
    "cde_opportunity": "Name the digest_item_id event + date + credits + fee. Binary RSVP.",
    "gbp_unverified": "Name verification_path + estimated_uplift_pct. One-tap kick-off.",
}


def _resolve_digest_item(category: dict, trigger: dict) -> Optional[dict]:
    """Look up the specific digest item the trigger payload points at."""
    payload = trigger.get("payload", {}) or {}
    item_id = (
        payload.get("top_item_id")
        or payload.get("digest_item_id")
        or payload.get("alert_id")
    )
    if not item_id:
        return None
    for item in category.get("digest", []) or []:
        if item.get("id") == item_id:
            return item
    return None


def _slim_merchant(merchant: dict) -> dict:
    """Project the merchant fields the composer actually needs."""
    return {
        "merchant_id": merchant.get("merchant_id"),
        "identity": merchant.get("identity", {}),
        "subscription": merchant.get("subscription", {}),
        "performance": merchant.get("performance", {}),
        "active_offers": [
            {"title": o.get("title"), "id": o.get("id")}
            for o in merchant.get("offers", []) or []
            if o.get("status") == "active"
        ],
        "expired_offers": [
            o.get("title")
            for o in merchant.get("offers", []) or []
            if o.get("status") == "expired"
        ][:2],
        "customer_aggregate": merchant.get("customer_aggregate", {}),
        "signals": merchant.get("signals", []),
        "review_themes": merchant.get("review_themes", []),
        "recent_history": (merchant.get("conversation_history") or [])[-2:],
    }


def _slim_category(category: dict, digest_item: Optional[dict]) -> dict:
    """Project the category fields the composer actually needs."""
    voice = category.get("voice", {}) or {}
    return {
        "slug": category.get("slug"),
        "voice": {
            "tone": voice.get("tone"),
            "register": voice.get("register"),
            "code_mix": voice.get("code_mix"),
            "salutation_examples": voice.get("salutation_examples", [])[:2],
            "vocab_allowed": voice.get("vocab_allowed", [])[:8],
            "vocab_taboo": voice.get("vocab_taboo", [])[:6],
            "tone_examples": voice.get("tone_examples", [])[:2],
        },
        "peer_stats": category.get("peer_stats", {}),
        "trigger_digest_item": digest_item,
        "seasonal_beats": (category.get("seasonal_beats") or [])[:2],
        "trend_signals": (category.get("trend_signals") or [])[:2],
    }


def _slim_customer(customer: Optional[dict]) -> Optional[dict]:
    if not customer:
        return None
    return {
        "customer_id": customer.get("customer_id"),
        "identity": customer.get("identity", {}),
        "relationship": customer.get("relationship", {}),
        "state": customer.get("state"),
        "preferences": customer.get("preferences", {}),
        "consent_scope": (customer.get("consent") or {}).get("scope", []),
    }


def _build_user_prompt(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict],
) -> str:
    digest_item = _resolve_digest_item(category, trigger)
    framing = TRIGGER_FRAMING.get(
        trigger.get("kind", ""),
        "Lead with the trigger payload's most concrete fact. Name the event clearly.",
    )

    blob = {
        "trigger": {
            "id": trigger.get("id"),
            "kind": trigger.get("kind"),
            "scope": trigger.get("scope"),
            "source": trigger.get("source"),
            "urgency": trigger.get("urgency"),
            "payload": trigger.get("payload", {}),
        },
        "category": _slim_category(category, digest_item),
        "merchant": _slim_merchant(merchant),
        "customer": _slim_customer(customer),
    }

    return (
        f"FRAMING FOR THIS TRIGGER KIND: {framing}\n\n"
        f"CONTEXTS (use only facts visible here; do not invent):\n"
        f"```json\n{json.dumps(blob, ensure_ascii=False, default=str)}\n```\n\n"
        f"Compose the message now. Output JSON only."
    )


async def compose_message(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
) -> dict:
    merchant_id = merchant.get("merchant_id", "")
    trigger_kind = trigger.get("kind", "generic")
    fallback_sup_key = trigger.get("suppression_key", f"{trigger_kind}:{merchant_id}")

    user_prompt = _build_user_prompt(category, merchant, trigger, customer)

    try:
        response = await asyncio.wait_for(
            get_client().chat.completions.create(
                model=COMPOSE_MODEL,
                temperature=0,
                seed=42,
                max_tokens=300,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            ),
            timeout=COMPOSE_TIMEOUT_S,
        )
        raw = response.choices[0].message.content or ""
        result = json.loads(raw)
    except Exception as e:
        owner = (
            merchant.get("identity", {}).get("owner_first_name")
            or merchant.get("identity", {}).get("name", "there")
        )
        return {
            "body": f"Hi {owner}, quick check-in on something specific to your account — want the detail?",
            "cta": "binary_yes_no",
            "send_as": "merchant_on_behalf" if customer else "vera",
            "suppression_key": fallback_sup_key,
            "rationale": f"Fallback ({type(e).__name__})",
        }

    body = (result.get("body") or "").strip()
    body = re.sub(r"https?://\S+", "", body).strip()
    if len(body) > 320:
        body = body[:317].rstrip() + "..."
    result["body"] = body

    if result.get("cta") not in {"binary_yes_no", "multi_choice_slot", "open_ended", "none"}:
        result["cta"] = "binary_yes_no"
    if result.get("send_as") not in {"vera", "merchant_on_behalf"}:
        result["send_as"] = "merchant_on_behalf" if customer else "vera"
    if not result.get("suppression_key"):
        result["suppression_key"] = fallback_sup_key
    if not result.get("rationale"):
        result["rationale"] = f"Composed for {trigger_kind}"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reply handling
# ─────────────────────────────────────────────────────────────────────────────

AUTO_REPLY_PHRASES = [
    "thank you for contacting", "our team will respond", "automated response",
    "i am an automated", "this is an automated", "automated assistant",
    "main ek automated", "aapki jaankari ke liye bahut-bahut shukriya",
    "team tak pahuncha", "unable to respond right now", "out of office",
    "currently unavailable", "will get back to you",
]

OPT_OUT_PHRASES = [
    "stop messaging", "stop sending", "not interested", "please stop",
    "band karo", "mat bhejo", "leave me alone", "do not contact",
    "unsubscribe", "useless spam", "spam mat", "block me",
]

# Strong commitment phrases — switch to action mode immediately.
COMMITMENT_PHRASES = [
    "lets do it", "let's do it", "let us do it", "lets go", "let's go",
    "go ahead", "proceed", "send it", "yes do it", "do it",
    "ok please", "ok do it", "ok go", "okay do",
    "sure go ahead", "yes please", "haan kar do", "kar do", "shuru karo",
    "haan bhej do", "bhej do", "theek hai bhejo", "theek hai go",
    "main ready hoon", "i am ready", "ready", "agreed", "confirm",
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def is_auto_reply_phrase(msg: str) -> bool:
    n = _norm(msg)
    return any(p in n for p in AUTO_REPLY_PHRASES)


def is_opt_out(msg: str) -> bool:
    n = _norm(msg)
    return any(p in n for p in OPT_OUT_PHRASES)


def is_commitment(msg: str) -> bool:
    n = _norm(msg)
    if len(n) > 80:  # commitments are short
        return False
    return any(p in n for p in COMMITMENT_PHRASES) or n in {"yes", "ok", "okay", "haan", "yep"}


def detect_repeat_auto_reply(conv_id: str, msg: str) -> int:
    """Track verbatim repeats per conversation. Returns count of times seen (incl. this)."""
    n = _norm(msg)
    if not n:
        return 0
    arr = last_merchant_messages.setdefault(conv_id, [])
    arr.append(n)
    return arr.count(n)


REPLY_SYSTEM = """You are Vera continuing a WhatsApp conversation with an Indian local merchant (or one of their customers).

Rules:
- Reply in ≤2 sentences, ≤200 chars.
- Match the language they used (English / Hindi-English mix).
- If they accepted, switched to action-mode, or asked "what's next" — give them the next concrete step. Do NOT ask another qualifying question.
- If they asked a clarifying question — answer briefly and forward-move.
- If they said something off-topic or hostile — acknowledge once, exit politely.
- Don't re-introduce yourself. Don't repeat your previous message verbatim.

Output ONLY this JSON:
{"action": "send" | "wait" | "end", "body": "...", "cta": "binary_yes_no" | "open_ended" | "none", "wait_seconds": <int, only if action=wait>, "rationale": "..."}
"""


async def compose_reply(
    merchant_message: str,
    conversation_history: list,
    merchant: dict,
    category: dict,
    customer: Optional[dict],
    turn_number: int,
) -> dict:
    merchant_name = merchant.get("identity", {}).get("name", "")
    owner = merchant.get("identity", {}).get("owner_first_name", "")
    cat_slug = category.get("slug", "")
    languages = merchant.get("identity", {}).get("languages", ["en"])

    history_text = "\n".join(
        f"[{(t.get('from') or '?').upper()}]: {(t.get('body') or t.get('msg') or '')[:120]}"
        for t in conversation_history[-6:]
    )

    user_prompt = (
        f"Merchant: {merchant_name} (owner: {owner}, category: {cat_slug}, languages: {languages})\n"
        f"Customer-facing: {bool(customer)}\n"
        f"Turn: {turn_number}\n"
        f"Latest from them: \"{merchant_message}\"\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"Compose the next reply. Output JSON only."
    )

    try:
        response = await asyncio.wait_for(
            get_client().chat.completions.create(
                model=REPLY_MODEL,
                temperature=0,
                seed=42,
                max_tokens=200,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": REPLY_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
            ),
            timeout=REPLY_TIMEOUT_S,
        )
        raw = response.choices[0].message.content or ""
        out = json.loads(raw)
    except Exception:
        return {
            "action": "send",
            "body": "Got it — I'll set that up. Reply YES to confirm.",
            "cta": "binary_yes_no",
            "rationale": "Reply LLM fallback",
        }

    body = (out.get("body") or "").strip()
    body = re.sub(r"https?://\S+", "", body).strip()
    if len(body) > 320:
        body = body[:317].rstrip() + "..."
    if body:
        out["body"] = body
    if out.get("action") not in {"send", "wait", "end"}:
        out["action"] = "send"
    return out


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
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera Challenger",
        "team_members": ["Dinesh Sahu"],
        "model": f"compose:{COMPOSE_MODEL} | reply:{REPLY_MODEL}",
        "approach": "Context-grounded composer: trigger-resolved digest items, category voice, named compulsion lever, parallel LLM calls, intent-aware reply.",
        "contact_email": "dinesh@dins.in",
        "version": "4.0.0",
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
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": cur["version"],
        }
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


def _gather_eligible(available_triggers: list[str]) -> list[tuple]:
    eligible = []
    seen_pairs: set[tuple[str, str]] = set()
    for trg_id in available_triggers:
        if len(eligible) >= 20:
            break
        trg_entry = contexts.get(("trigger", trg_id))
        if not trg_entry:
            continue
        trigger = trg_entry["payload"]
        sup_key = trigger.get("suppression_key", "")
        if sup_key and sup_key in suppressed:
            continue

        merchant_id = (
            trigger.get("merchant_id")
            or (trigger.get("payload") or {}).get("merchant_id")
        )
        if not merchant_id:
            continue
        merch_entry = contexts.get(("merchant", merchant_id))
        if not merch_entry:
            continue
        merchant = merch_entry["payload"]

        cat_slug = merchant.get("category_slug")
        cat_entry = contexts.get(("category", cat_slug))
        if not cat_entry:
            continue
        category = cat_entry["payload"]

        customer = None
        customer_id = (
            trigger.get("customer_id")
            or (trigger.get("payload") or {}).get("customer_id")
        )
        if customer_id:
            ce = contexts.get(("customer", customer_id))
            if ce:
                customer = ce["payload"]

        conv_id = f"conv_{merchant_id}_{trg_id}"
        if conv_id in closed_conversations:
            continue

        # one action per (merchant, conv) per tick
        pair = (merchant_id, conv_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        eligible.append(
            (trg_id, category, merchant, trigger, customer, merchant_id, conv_id, sup_key)
        )
    return eligible


@app.post("/v1/tick")
async def tick(body: TickBody):
    eligible = _gather_eligible(body.available_triggers)
    if not eligible:
        return {"actions": []}

    tasks = [
        compose_message(category, merchant, trigger, customer)
        for (_, category, merchant, trigger, customer, _, _, _) in eligible
    ]

    try:
        composed_list = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=TICK_BUDGET_S,
        )
    except asyncio.TimeoutError:
        return {"actions": []}

    actions = []
    for (trg_id, _cat, merchant, trigger, customer, merchant_id, conv_id, sup_key), composed in zip(eligible, composed_list):
        if isinstance(composed, Exception) or not isinstance(composed, dict):
            continue
        body_text = composed.get("body", "")
        if not body_text:
            continue

        key = composed.get("suppression_key") or sup_key
        if key:
            suppressed.add(key)

        conversations.setdefault(conv_id, []).append(
            {"from": "bot", "body": body_text, "ts": body.now}
        )

        merchant_name = merchant.get("identity", {}).get("name", "")
        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": (customer or {}).get("customer_id") if customer else trigger.get("customer_id"),
            "send_as": composed.get("send_as", "merchant_on_behalf" if customer else "vera"),
            "trigger_id": trg_id,
            "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
            "template_params": [merchant_name, body_text[:100], composed.get("cta", "")],
            "body": body_text,
            "cta": composed.get("cta", "binary_yes_no"),
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
    msg = body.message or ""

    conversations.setdefault(conv_id, []).append(
        {"from": body.from_role, "msg": msg, "ts": body.received_at}
    )

    # 1) Hard exit on opt-out / hostile.
    if is_opt_out(msg):
        closed_conversations.add(conv_id)
        return {"action": "end", "rationale": "Opted out / hostile — graceful exit"}

    # 2) Auto-reply: explicit phrases OR same verbatim message repeated.
    repeat_count = detect_repeat_auto_reply(conv_id, msg)
    is_auto = is_auto_reply_phrase(msg) or repeat_count >= 2
    if is_auto:
        key = body.merchant_id or conv_id
        c = auto_reply_counts.get(key, 0) + 1
        auto_reply_counts[key] = c
        if c >= 2 or repeat_count >= 3:
            closed_conversations.add(conv_id)
            return {
                "action": "end",
                "rationale": f"Auto-reply pattern ({c}x phrase, {repeat_count}x verbatim) — exiting",
            }
        return {
            "action": "send",
            "body": "Looks like an auto-reply 🙏 When you see this, reply YES and I'll continue.",
            "cta": "binary_yes_no",
            "rationale": "Auto-reply detected once; one polite re-poke before exit",
        }

    merchant: dict = {}
    category: dict = {}
    customer: Optional[dict] = None
    if body.merchant_id:
        m = contexts.get(("merchant", body.merchant_id))
        if m:
            merchant = m["payload"]
            cat = contexts.get(("category", merchant.get("category_slug", "")))
            if cat:
                category = cat["payload"]
    if body.customer_id:
        c = contexts.get(("customer", body.customer_id))
        if c:
            customer = c["payload"]

    # 3) Commitment / intent transition — short-circuit, never re-qualify.
    if is_commitment(msg):
        owner = merchant.get("identity", {}).get("owner_first_name", "")
        prefix = f"{owner}, " if owner else ""
        return {
            "action": "send",
            "body": f"{prefix}done — drafting now and sending you the preview to confirm. 2 mins.",
            "cta": "binary_yes_no",
            "rationale": "Detected commitment; switched to action mode",
        }

    # 4) Default: LLM reply.
    out = await compose_reply(
        msg,
        conversations[conv_id],
        merchant,
        category,
        customer,
        body.turn_number,
    )

    if out.get("action") == "send" and out.get("body"):
        conversations[conv_id].append(
            {"from": "bot", "body": out["body"], "ts": body.received_at}
        )
    elif out.get("action") == "end":
        closed_conversations.add(conv_id)

    return out


@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    suppressed.clear()
    closed_conversations.clear()
    auto_reply_counts.clear()
    last_merchant_messages.clear()
    return {"status": "wiped"}
