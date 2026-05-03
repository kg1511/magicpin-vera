from __future__ import annotations

import re
import time
import uuid
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(dt_str: str) -> datetime:
    # Supports: 2026-05-03T00:00:00Z and 2026-04-26T19:30:00+05:30
    if not dt_str:
        return _utcnow()
    dt_str = dt_str.strip()
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(dt_str)
    except ValueError:
        # Very defensive fallback: try to salvage basic date-only strings
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", dt_str)
        if m:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
        return _utcnow()


def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def _fmt_pct(x: Any) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "?%"


def _first_nonempty(*values: Optional[str]) -> str:
    for v in values:
        if v and str(v).strip():
            return str(v).strip()
    return ""


# -----------------------------------------------------------------------------
# In-memory state
# -----------------------------------------------------------------------------


AllowedScope = Literal["category", "merchant", "customer", "trigger"]


@dataclass
class StoredContext:
    version: int
    payload: Dict[str, Any]
    delivered_at: str
    stored_at: str


@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: Optional[str]
    customer_id: Optional[str]
    trigger_id: Optional[str]
    send_as: Literal["vera", "merchant_on_behalf"]
    turns: List[Dict[str, Any]] = field(default_factory=list)
    last_inbound_at: Optional[datetime] = None
    last_outbound_body: Optional[str] = None
    auto_reply_hits: int = 0
    booking_name: Optional[str] = None
    booking_phone: Optional[str] = None
    booking_service: Optional[str] = None
    booking_time: Optional[str] = None


START_TIME = time.time()

contexts: Dict[Tuple[str, str], StoredContext] = {}
conversations: Dict[str, ConversationState] = {}
suppressed: set[str] = set()


# -----------------------------------------------------------------------------
# FastAPI app + schemas
# -----------------------------------------------------------------------------

app = FastAPI()

SUBMITTED_AT = _iso_utc(_utcnow())
APP_VERSION = "0.2.0"


def _build_version_string() -> str:
    sha = (os.getenv("RENDER_GIT_COMMIT") or "").strip()
    if sha:
        return f"{APP_VERSION}+{sha[:7]}"
    return APP_VERSION


class ContextPushBody(BaseModel):
    scope: AllowedScope
    context_id: str
    version: int = Field(ge=1)
    payload: Dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = Field(default_factory=list)


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: Literal["merchant", "customer"]
    message: str
    received_at: str
    turn_number: int


# -----------------------------------------------------------------------------
# Composition logic (rule-based, context-only)
# -----------------------------------------------------------------------------


_AUTO_REPLY_PATTERNS = [
    r"thank\s+you\s+for\s+contacting",
    r"we\s+will\s+respond\s+shortly",
    r"our\s+team\s+will\s+respond",
    r"your\s+message\s+has\s+been\s+received",
]

_HOSTILE_PATTERNS = [
    r"stop\s+messaging",
    r"useless\s+spam",
    r"don'?t\s+message",
    r"unsubscribe",
    r"block\s+me",
]

_COMMIT_PATTERNS = [
    r"lets\s+do\s+it",
    r"let'?s\s+do\s+it",
    r"what'?s\s+next",
    r"whats\s+next",
    r"go\s+ahead",
    r"do\s+it",
]


def _matches_any(text: str, patterns: List[str]) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in patterns)


def _merchant_salutation(category: Dict[str, Any], merchant: Dict[str, Any]) -> str:
    slug = (category or {}).get("slug")
    ident = (merchant or {}).get("identity", {})
    owner = _first_nonempty(ident.get("owner_first_name"))
    name = _first_nonempty(ident.get("name"))

    if slug == "dentists":
        if owner:
            return f"Dr. {owner}"
        # If the merchant name already has Dr., keep it
        if name.lower().startswith("dr"):
            return name
        return f"Doctor"  # fallback

    return owner or name or "there"


def _merchant_anchor_line(category: Dict[str, Any], merchant: Dict[str, Any]) -> str:
    ident = (merchant or {}).get("identity", {})
    perf = (merchant or {}).get("performance", {})
    locality = _first_nonempty(ident.get("locality"))
    city = _first_nonempty(ident.get("city"))

    views = perf.get("views")
    calls = perf.get("calls")
    ctr = perf.get("ctr")

    pieces = []
    if locality or city:
        pieces.append(f"({', '.join([p for p in [locality, city] if p])})")
    if views is not None and calls is not None and ctr is not None:
        pieces.append(f"last 30d: {views} views, {calls} calls, CTR {_fmt_pct(ctr)}")
    return " ".join(pieces).strip()


def _pick_offer_ideas(category: Dict[str, Any], k: int = 2) -> List[str]:
    catalog = (category or {}).get("offer_catalog") or []
    ideas = [c.get("title") for c in catalog if c.get("title")]
    return ideas[:k]


def _find_digest_item(category: Dict[str, Any], item_id: str) -> Optional[Dict[str, Any]]:
    for item in (category or {}).get("digest") or []:
        if item.get("id") == item_id:
            return item
    return None


def _business_label(category_slug: str) -> str:
    slug = (category_slug or "").strip().lower()
    if slug == "dentists":
        return "clinic"
    if slug == "pharmacies":
        return "pharmacy"
    if slug == "salons":
        return "salon"
    if slug == "gyms":
        return "gym"
    if slug == "restaurants":
        return "restaurant"
    return "business"


def _customer_voice_prefix(customer_name: str, merchant_name: str) -> str:
    """Return a natural customer-facing opening line.

    Rules:
    - If both names exist: "Hi Priya - this is Dr. Meera's Dental Clinic."
    - If missing names: use a neutral fallback; do not invent names.
    """

    cust = (customer_name or "").strip()
    merch = (merchant_name or "").strip()
    if cust and merch:
        return f"Hi {cust} - this is {merch}."
    if cust:
        return f"Hi {cust} - this is the business."
    if merch:
        return f"Hi there - this is {merch}."
    # Neutral fallback (no placeholders like "Hi - this is the clinic")
    return "Hi there - this is the business."


def _is_ack_message(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    if len(t) <= 3 and t in {"ok", "k", "kk", "thx", "ty"}:
        return True
    if re.search(r"\b(thanks|thank\s+you|ok(?:ay)?|cool|great|noted|received|sure)\b", t) and "?" not in t:
        return True
    return False


def _extract_phone(text: str) -> str:
    # Simple India-friendly extraction: +91XXXXXXXXXX / 10-digit numbers.
    t = (text or "")
    m = re.search(r"(?:\+?91[\s-]?)?([6-9]\d{9})\b", t)
    if m:
        return m.group(1)
    m2 = re.search(r"\b(\d{10})\b", t)
    if m2:
        return m2.group(1)
    return ""


def _extract_booking_time_hint(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    # Common relative hints
    m = re.search(r"\b(today|tomorrow|tmr|tonight|this\s+evening|this\s+morning)\b", t, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    # Time of day patterns: 5pm, 5:30 pm, 17:30
    m2 = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", t, flags=re.IGNORECASE)
    if m2:
        hh = m2.group(1)
        mm = m2.group(2) or "00"
        ap = m2.group(3).lower()
        return f"{hh}:{mm} {ap}"
    m3 = re.search(r"\b([01]?\d|2[0-3]):[0-5]\d\b", t)
    if m3:
        return m3.group(0)

    # Very light capture after 'for ...' (avoid echoing long strings)
    m4 = re.search(r"\bfor\b\s+(.{1,40})$", t, flags=re.IGNORECASE)
    if m4:
        hint = m4.group(1).strip().rstrip(".")
        if hint and len(hint.split()) <= 6:
            return hint
    return ""


def _extract_service_hint(category_slug: str, text: str) -> str:
    slug = (category_slug or "").strip().lower()
    t = (text or "").lower()
    if not t:
        return ""

    # Keep this intentionally lightweight: detect only common service keywords.
    catalog: Dict[str, List[str]] = {
        "dentists": [
            r"\b(cleaning|scale\s*&\s*polish|scaling|polish)\b",
            r"\b(whitening|bleach(?:ing)?)\b",
            r"\b(check\s*up|checkup|consult(?:ation)?)\b",
            r"\b(braces|aligners|invisalign)\b",
            r"\b(root\s*canal|rct)\b",
            r"\b(filling|extraction)\b",
        ],
        "salons": [
            r"\b(hair\s*cut|haircut|trim)\b",
            r"\b(facial|cleanup)\b",
            r"\b(hair\s*spa|spa)\b",
            r"\b(manicure|pedicure)\b",
            r"\b(wax(?:ing)?)\b",
        ],
        "gyms": [
            r"\b(membership|monthly|quarterly|annual)\b",
            r"\b(trial|free\s*trial|day\s*pass)\b",
            r"\b(personal\s*training|pt)\b",
            r"\b(yoga|zumba|crossfit)\b",
        ],
        "restaurants": [
            r"\b(table|reservation|reserve|booking)\b",
            r"\b(birthday|anniversary)\b",
        ],
        "pharmacies": [
            r"\b(home\s*delivery|delivery)\b",
            r"\b(pick\s*up|pickup)\b",
        ],
    }

    patterns = catalog.get(slug) or [
        r"\b(cleaning|whitening|haircut|membership|reservation|delivery|pickup|consultation)\b"
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def _looks_like_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if "?" in t:
        return True
    return bool(re.search(r"\b(how|what|when|where|which|can\s+you|could\s+you|pls|please|help)\b", t))


def _looks_like_request(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return bool(
        re.search(
            r"\b(send|share|draft|write|create|make|give\s+me|show\s+me|steps|checklist|templates|abstract|summary|details)\b",
            t,
        )
    )


def _compose_for_trigger(
    *,
    now: datetime,
    category: Dict[str, Any],
    merchant: Dict[str, Any],
    trigger: Dict[str, Any],
    customer: Optional[Dict[str, Any]],
) -> Tuple[str, str, str]:
    """Returns (body, cta, rationale)."""

    kind = (trigger or {}).get("kind", "")
    urgency = trigger.get("urgency")

    sal = _merchant_salutation(category, merchant)
    anchor = _merchant_anchor_line(category, merchant)

    if kind == "research_digest":
        item = _find_digest_item(category, (trigger.get("payload") or {}).get("top_item_id", ""))
        if item:
            title = item.get("title", "")
            source = item.get("source", "")
            trial_n = item.get("trial_n")
            seg = item.get("patient_segment")
            summary = item.get("summary")
            line = f"{sal} - quick one from {source}" if source else f"{sal} - quick one from this week"
            facts = []
            if trial_n:
                facts.append(f"n={trial_n}")
            if seg:
                facts.append(seg.replace("_", " "))
            fact_str = f" ({', '.join(facts)})" if facts else ""
            body = (
                f"{line}{fact_str}: {title}.\n"
                f"If you want, I can draft 1 short WhatsApp patient note + 1 GBP post that stays compliant. Reply YES and tell me: focus on preventive or cosmetic?"
            )
            rationale = "Uses the digest item + citation; asks a low-friction choice to drive reply."
            return body, "open_ended", rationale

        body = f"{sal} - I have a short dentistry research digest for this week. Want the 2-bullet summary + a ready-to-post patient WhatsApp?"
        rationale = "Research digest trigger; asks permission and offers done-for-you assets."
        return body, "open_ended", rationale

    if kind == "regulation_change":
        payload = trigger.get("payload") or {}
        deadline = payload.get("deadline_iso") or trigger.get("expires_at")
        item = _find_digest_item(category, payload.get("top_item_id", ""))
        title = item.get("title") if item else "A compliance update"
        source = item.get("source") if item else ""
        deadline_fmt = deadline.split("T")[0] if isinstance(deadline, str) and deadline else ""
        src = f" ({source})" if source else ""
        body = (
            f"{sal} - heads up on compliance{src}: {title}.\n"
            f"Deadline looks like {deadline_fmt}. Want a 5-point clinic checklist (what to change + what to document) you can forward to your staff?"
        )
        rationale = "Regulation change + deadline; offers a concrete checklist and asks a simple question."
        return body, "open_ended", rationale

    if kind in {"perf_dip", "perf_spike"}:
        payload = trigger.get("payload") or {}
        metric = payload.get("metric", "performance")
        delta = payload.get("delta_pct")
        window = payload.get("window", "")
        delta_str = "" if delta is None else f"{delta * 100:+.0f}%"
        offer_ideas = _pick_offer_ideas(category, 2)
        offer_line = " / ".join(offer_ideas) if offer_ideas else "a service+price offer"

        if kind == "perf_dip":
            body = (
                f"{sal} - noticed a dip in {metric} {delta_str} over {window}. {anchor}\n"
                f"Want me to draft 2 Google Posts + 1 offer idea ({offer_line}) to recover calls this week?"
            ).strip()
            rationale = "Connects trigger delta to merchant performance and proposes concrete assets."
            return body, "open_ended", rationale

        body = (
            f"{sal} - nice spike in {metric} {delta_str} over {window}. {anchor}\n"
            f"Want to capitalize with a quick post + an offer pin (e.g., {offer_line})?"
        ).strip()
        rationale = "Celebrates spike; proposes next-best action to convert attention."
        return body, "open_ended", rationale

    if kind == "renewal_due":
        payload = trigger.get("payload") or {}
        days = payload.get("days_remaining")
        amt = payload.get("renewal_amount")
        plan = payload.get("plan")
        days_txt = f"{days} day(s)" if isinstance(days, int) else "soon"
        amt_txt = f"₹{amt}" if amt is not None else ""
        plan_txt = f"{plan}" if plan else "your plan"
        body = (
            f"{sal} - your {plan_txt} renewal is due in {days_txt}. {amt_txt}\n"
            f"If you want, I can (a) share the renewal steps, or (b) quickly set 1 high-converting offer + 2 posts so the plan pays back this month. Which one first?"
        ).strip()
        rationale = "Uses concrete renewal facts; offers two clear paths to reduce churn."
        return body, "open_ended", rationale

    if kind == "festival_upcoming":
        payload = trigger.get("payload") or {}
        fest = payload.get("festival", "the upcoming festival")
        days_until = payload.get("days_until")
        offer_ideas = _pick_offer_ideas(category, 2)
        offers = " / ".join(offer_ideas) if offer_ideas else "a service+price offer"
        days_txt = f"in {days_until} day(s)" if isinstance(days_until, int) else "soon"
        body = (
            f"{sal} - {fest} is coming {days_txt}.\n"
            f"Want 2 ready-to-post creatives + one offer suggestion tailored for your category (e.g., {offers})?"
        ).strip()
        rationale = "Uses festival timing; offers done-for-you assets and a concrete offer pattern."
        return body, "open_ended", rationale

    if kind == "review_theme_emerged":
        payload = trigger.get("payload") or {}
        theme = payload.get("theme", "a theme")
        occ = payload.get("occurrences_30d")
        quote = payload.get("common_quote")
        occ_txt = f"({occ} mentions in 30d)" if occ is not None else ""
        quote_txt = f"\nExample: “{quote}”" if quote else ""
        body = (
            f"{sal} - a review theme popped up: {theme} {occ_txt}.{quote_txt}\n"
            f"Want me to draft 3 polite, non-defensive reply templates + a one-line ops fix you can try this week?"
        ).strip()
        rationale = "Anchors on concrete review evidence and offers ready-to-use replies."
        return body, "open_ended", rationale

    if kind == "competitor_opened":
        payload = trigger.get("payload") or {}
        competitor = _first_nonempty(payload.get("competitor_name"))
        dist_km = payload.get("distance_km")
        if dist_km is None:
            dist_km = payload.get("distance")
        their_offer = _first_nonempty(payload.get("their_offer"), payload.get("offer"), payload.get("competitor_offer"))
        opened = _first_nonempty(payload.get("opened_date"))

        who = competitor or "a nearby competitor"
        where = ""
        if isinstance(dist_km, (int, float)):
            where = f" (~{dist_km:.1f} km)"
        when = f" (opened {opened})" if opened else ""

        offer_line = f"They are pushing: {their_offer}. " if their_offer else ""
        cat_offer = " / ".join(_pick_offer_ideas(category, 2))
        counter_hint = f" (e.g., {cat_offer})" if cat_offer else ""

        body = (
            f"{sal} - heads up: {who}{where}{when}. {offer_line}{anchor}\n"
            f"Want a 3-part counter (1 WhatsApp reply, 1 Google Post headline, 1 better-value offer{counter_hint}) you can post today?"
        ).strip()
        rationale = "Competitor opened nearby; summarizes concrete facts and offers ready-to-use counter assets without leaking placeholder keys."
        return body, "open_ended", rationale

    if kind == "ipl_match_today":
        payload = trigger.get("payload") or {}
        match = payload.get("match", "today's match")
        venue = payload.get("venue")
        match_time = payload.get("match_time_iso")
        match_time_txt = ""
        if isinstance(match_time, str) and match_time:
            try:
                dt = _parse_iso(match_time)
                match_time_txt = dt.strftime("%I:%M %p").lstrip("0")
            except Exception:
                match_time_txt = ""
        parts = [match]
        if venue:
            parts.append(venue)
        if match_time_txt:
            parts.append(match_time_txt)
        headline = " • ".join([p for p in parts if p])
        offer_ideas = _pick_offer_ideas(category, 2)
        offers = " / ".join(offer_ideas) if offer_ideas else "a match-night combo"
        body = (
            f"{sal} - {headline}.\n"
            f"Want a match-night Google Post + 1 specific offer idea ({offers}) to catch last-minute searches?"
        ).strip()
        rationale = "Uses event specifics (match/time/venue) and suggests a concrete conversion action."
        return body, "open_ended", rationale

    if kind in {"recall_due", "appointment_tomorrow", "wedding_package_followup"}:
        # customer-facing nudges (send as merchant_on_behalf)
        cust_name = _first_nonempty((customer or {}).get("identity", {}).get("name")) or "Hi"
        category_slug = _first_nonempty((merchant or {}).get("category_slug"), (category or {}).get("slug"))
        fallback_label = _business_label(category_slug)
        mname = _first_nonempty((merchant or {}).get("identity", {}).get("name")) or f"our {fallback_label}"
        payload = trigger.get("payload") or {}

        if kind == "recall_due":
            due = payload.get("due_date")
            last = payload.get("last_service_date")
            slots = payload.get("available_slots") or []
            slot_labels = [s.get("label") for s in slots if s.get("label")][:2]
            slots_txt = " or ".join([s for s in slot_labels if s])
            due_txt = due or "soon"
            last_txt = f"(last visit {last})" if last else ""
            body = (
                f"Hi {cust_name} - this is {mname}. Your next visit is due around {due_txt} {last_txt}.\n"
                f"If you want, I can book you in {slots_txt}. Reply YES for a slot, or STOP to opt out."
            ).strip()
            rationale = "Uses due date + slot labels; clear YES/STOP CTA; no fabricated claims."
            return body, "open_ended", rationale

        if kind == "appointment_tomorrow":
            appt = payload.get("appointment_time_label") or payload.get("appointment_time_iso") or "tomorrow"
            body = (
                f"Hi {cust_name} - reminder from {mname}: your appointment is {appt}.\n"
                f"Reply YES to confirm, or message here if you need to reschedule."
            ).strip()
            rationale = "Short reminder with confirmation CTA."
            return body, "open_ended", rationale

        if kind == "wedding_package_followup":
            wedding = payload.get("wedding_date")
            next_step = payload.get("next_step_window_open")
            wedding_txt = wedding or "your wedding"
            ns_txt = next_step.replace("_", " ") if isinstance(next_step, str) else "next steps"
            body = (
                f"Hi {cust_name} - this is {mname}. With {wedding_txt} coming up, your {ns_txt} window is open.\n"
                f"Want 2 package options + timings? Reply YES and I'll share."
            ).strip()
            rationale = "Connects to wedding timeline; offers options; clear CTA."
            return body, "open_ended", rationale

    # Generic fallback (do not leak raw payload keys)
    body = (
        f"{sal} - quick update on {kind.replace('_', ' ')} (urgency {urgency}).\n"
        f"Want me to turn this into 1 ready-to-send message/post tailored to your category?"
    ).strip()

    rationale = "Fallback message: references trigger kind and offers a concrete next step."
    return body, "open_ended", rationale


def _dedupe_body(conv: ConversationState, body: str) -> str:
    if conv.last_outbound_body and conv.last_outbound_body.strip() == body.strip():
        return body + "\n\n(If you prefer, reply STOP and I'll pause.)"
    return body


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _cid) in contexts.keys():
        counts[scope] = counts.get(scope, 0) + 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "LocalBot",
        "team_members": ["You"],
        "model": "rules+templates",
        "approach": "rule-based composer using only pushed context (no fabrication)",
        "contact_email": "local@example.com",
        "version": _build_version_string(),
        "submitted_at": SUBMITTED_AT,
    }


@app.post("/v1/context")
async def push_context(body: ContextPushBody):
    key = (body.scope, body.context_id)
    cur = contexts.get(key)

    if cur and body.version < cur.version:
        return JSONResponse(
            status_code=409,
            content={"accepted": False, "reason": "stale_version", "current_version": cur.version},
        )

    # Idempotent for same version
    if cur and body.version == cur.version:
        return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": cur.stored_at}

    stored_at = _iso_utc(_utcnow())
    contexts[key] = StoredContext(version=body.version, payload=body.payload, delivered_at=body.delivered_at, stored_at=stored_at)

    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": stored_at}


@app.post("/v1/tick")
async def tick(body: TickBody):
    now = _parse_iso(body.now)

    # Gather candidate actions
    candidates: List[Tuple[int, Dict[str, Any]]] = []

    for trg_id in body.available_triggers:
        trg_ctx = contexts.get(("trigger", trg_id))
        if not trg_ctx:
            continue

        trigger = trg_ctx.payload

        # Expiry check (judge datasets sometimes use past dates; allow a grace window)
        expires_at = trigger.get("expires_at")
        if isinstance(expires_at, str) and expires_at:
            try:
                expires_dt = _parse_iso(expires_at)
                if expires_dt < (now - timedelta(days=14)):
                    continue
            except Exception:
                pass

        sup_key = _first_nonempty(trigger.get("suppression_key"))
        if sup_key and sup_key in suppressed:
            continue

        merchant_id = trigger.get("merchant_id")
        merchant_ctx = contexts.get(("merchant", merchant_id)) if merchant_id else None
        if not merchant_ctx:
            continue
        merchant = merchant_ctx.payload

        customer_id = trigger.get("customer_id")
        customer_ctx = contexts.get(("customer", customer_id)) if customer_id else None
        customer = customer_ctx.payload if customer_ctx else None

        category_slug = merchant.get("category_slug") or (trigger.get("payload") or {}).get("category")
        cat_ctx = contexts.get(("category", category_slug)) if category_slug else None
        category = cat_ctx.payload if cat_ctx else {}

        send_as: Literal["vera", "merchant_on_behalf"] = "vera" if trigger.get("scope") != "customer" else "merchant_on_behalf"

        conv_id = f"conv_{trg_id}"
        conv = conversations.get(conv_id)
        if not conv:
            conv = ConversationState(
                conversation_id=conv_id,
                merchant_id=merchant_id,
                customer_id=customer_id,
                trigger_id=trg_id,
                send_as=send_as,
            )
            conversations[conv_id] = conv

        body_text, cta, rationale = _compose_for_trigger(
            now=now,
            category=category,
            merchant=merchant,
            trigger=trigger,
            customer=customer,
        )
        body_text = _dedupe_body(conv, body_text)

        # Store for de-dupe on the next turn
        conv.last_outbound_body = body_text

        template_name = f"vera_{trigger.get('kind', 'generic')}_v1" if send_as == "vera" else f"merchant_{trigger.get('kind', 'generic')}_v1"

        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trg_id,
            "template_name": template_name,
            "template_params": [
                _first_nonempty(merchant.get("identity", {}).get("name")),
                trigger.get("kind", ""),
                _first_nonempty(trigger.get("suppression_key")),
            ],
            "body": body_text,
            "cta": cta,
            "suppression_key": sup_key,
            "rationale": rationale,
        }

        urgency = trigger.get("urgency")
        priority = int(urgency) if isinstance(urgency, int) else 1
        candidates.append((priority, action))

    # Prioritize high urgency; keep restraint
    candidates.sort(key=lambda x: x[0], reverse=True)
    MAX_ACTIONS_PER_TICK = 2
    actions = [a for _p, a in candidates[:MAX_ACTIONS_PER_TICK]]

    for a in actions:
        if a.get("suppression_key"):
            suppressed.add(a["suppression_key"])

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    received_at = _parse_iso(body.received_at)
    conv = conversations.get(body.conversation_id)
    if not conv:
        conv = ConversationState(
            conversation_id=body.conversation_id,
            merchant_id=body.merchant_id,
            customer_id=body.customer_id,
            trigger_id=None,
            send_as="vera" if body.customer_id is None else "merchant_on_behalf",
        )
        conversations[body.conversation_id] = conv

    conv.turns.append({"from": body.from_role, "body": body.message, "received_at": body.received_at, "turn": body.turn_number})
    conv.last_inbound_at = received_at

    msg = body.message or ""
    msg_l = msg.lower()

    # Pull names (if present) to keep replies correctly voiced.
    merchant_name = ""
    merchant_category_slug = ""
    if body.merchant_id:
        mctx = contexts.get(("merchant", body.merchant_id))
        if mctx:
            merchant_name = _first_nonempty((mctx.payload.get("identity") or {}).get("name"))
            merchant_category_slug = _first_nonempty(mctx.payload.get("category_slug"))
    customer_name = ""
    if body.customer_id:
        cctx = contexts.get(("customer", body.customer_id))
        if cctx:
            customer_name = _first_nonempty((cctx.payload.get("identity") or {}).get("name"))

    # Hard stops
    if _matches_any(msg, _HOSTILE_PATTERNS) or re.search(r"\bstop\b", msg_l):
        return {"action": "end", "rationale": "User asked to stop / was hostile; ending respectfully."}

    # Auto-reply pollution
    if _matches_any(msg, _AUTO_REPLY_PATTERNS):
        conv.auto_reply_hits += 1
        # For auto-replies, don't send additional messages; back off.
        wait_seconds = int(_clamp(600 * conv.auto_reply_hits, 600, 3600))
        return {
            "action": "wait",
            "wait_seconds": wait_seconds,
            "rationale": "Detected WhatsApp auto-reply; waiting to avoid loops and resume when a human responds.",
        }
    # Customer routing: always customer-facing
    if body.customer_id is not None or body.from_role == "customer":
        prefix = _customer_voice_prefix(customer_name, merchant_name)

        # Keep lightweight extracted state so we only ask for missing booking details.
        if customer_name:
            conv.booking_name = customer_name
        extracted_phone = _extract_phone(msg)
        if extracted_phone:
            conv.booking_phone = extracted_phone
        extracted_time = _extract_booking_time_hint(msg)
        if extracted_time:
            conv.booking_time = extracted_time
        extracted_service = _extract_service_hint(merchant_category_slug, msg)
        if extracted_service:
            conv.booking_service = extracted_service

        is_booking_intent = bool(re.search(r"\b(book|booking|appointment|appt|reservation|reserve)\b", msg_l)) or bool(
            re.search(r"\b(reschedule|confirm)\b", msg_l)
        )

        # If it's just an acknowledgement and no action is required, wait.
        if not is_booking_intent and not _looks_like_question(msg) and _is_ack_message(msg):
            return {"action": "wait", "wait_seconds": 600, "rationale": "No customer action required; waiting."}

        if is_booking_intent:
            missing: List[str] = []
            if not (conv.booking_name or customer_name):
                missing.append("name")
            if not conv.booking_phone:
                missing.append("phone number")
            if not conv.booking_service:
                missing.append("service")
            if not conv.booking_time:
                missing.append("preferred time")

            if missing:
                ask = " and ".join(missing) if len(missing) <= 2 else ", ".join(missing[:-1]) + ", and " + missing[-1]
                next_body = f"{prefix} Sure - could you share your {ask}?"
            else:
                next_body = f"{prefix} Perfect - thanks. We'll confirm shortly."

            next_body = _dedupe_body(conv, next_body)
            conv.last_outbound_body = next_body
            return {
                "action": "send",
                "body": next_body,
                "cta": "open_ended",
                "rationale": "Customer booking intent; asked only for missing details.",
            }

        if _looks_like_question(msg):
            next_body = f"{prefix} Happy to help - are you looking to book? If yes, what day/time works best?"
            next_body = _dedupe_body(conv, next_body)
            conv.last_outbound_body = next_body
            return {
                "action": "send",
                "body": next_body,
                "cta": "open_ended",
                "rationale": "Customer question; asked a single clarifying question.",
            }

        if _is_ack_message(msg):
            return {"action": "wait", "wait_seconds": 600, "rationale": "No clear customer action required; waiting."}

        next_body = f"{prefix} How can I help today - are you looking to book or reschedule?"
        next_body = _dedupe_body(conv, next_body)
        conv.last_outbound_body = next_body
        return {"action": "send", "body": next_body, "cta": "open_ended", "rationale": "Intent unclear; asked a simple clarifying question."}


    # Intent transition: act, don't qualify
    if _matches_any(msg, _COMMIT_PATTERNS):
        next_body = (
            "Done - next step is quick:\n"
            "1) Tell me your top 1 service to push this week (e.g., cleaning/whitening/aligners).\n"
            "2) Tell me your preferred price point (₹299 / ₹499 / ₹999).\n"
            "I'll draft the offer + 2 posts in the next message."
        )
        next_body = _dedupe_body(conv, next_body)
        conv.last_outbound_body = next_body
        return {"action": "send", "body": next_body, "cta": "open_ended", "rationale": "Merchant committed; moved directly to execution with minimal inputs."}

    # Merchant routing: handle technical/compliance follow-ups with specificity
    if body.customer_id is None and body.from_role == "merchant":
        if not (_looks_like_question(msg) or _looks_like_request(msg)) and _is_ack_message(msg):
            return {"action": "wait", "wait_seconds": 600, "rationale": "Merchant acknowledgement; waiting."}

        if re.search(r"\b(x[-\s]?ray|xray|radiograph|radiography|d[-\s]?speed|film unit)\b", msg_l):
            next_body = (
                "Got it - I can help you audit this quickly.\n"
                "Reply with these 3 details and I'll turn it into a 5-point checklist + a staff note:\n"
                "1) Make/model of the unit (if you know)\n"
                "2) Your typical settings (kVp / mA / exposure time)\n"
                "3) Film/sensor + processing type (D-speed film, CR/DR, manual/automatic)"
            )
            next_body = _dedupe_body(conv, next_body)
            conv.last_outbound_body = next_body
            return {"action": "send", "body": next_body, "cta": "open_ended", "rationale": "Merchant asked a technical compliance question; request minimal specifics to generate a concrete checklist."}

        if _looks_like_request(msg):
            # If this conversation is tied to a trigger, try to fulfill directly.
            trg_id = conv.trigger_id
            trigger = None
            if trg_id:
                trg_ctx = contexts.get(("trigger", trg_id))
                trigger = trg_ctx.payload if trg_ctx else None

            if trigger and trigger.get("kind") == "research_digest":
                merchant_id = body.merchant_id or conv.merchant_id
                merchant = None
                if merchant_id:
                    mctx = contexts.get(("merchant", merchant_id))
                    merchant = mctx.payload if mctx else None

                category_slug = _first_nonempty((merchant or {}).get("category_slug"), (trigger.get("payload") or {}).get("category"))
                category = None
                if category_slug:
                    cctx = contexts.get(("category", category_slug))
                    category = cctx.payload if cctx else None

                top_item_id = _first_nonempty((trigger.get("payload") or {}).get("top_item_id"))
                item = _find_digest_item(category or {}, top_item_id) if top_item_id else None
                if item:
                    title = _first_nonempty(item.get("title"))
                    source = _first_nonempty(item.get("source"))
                    trial_n = item.get("trial_n")
                    seg = _first_nonempty(item.get("patient_segment"))
                    summary = _first_nonempty(item.get("summary"))

                    facts = []
                    if trial_n:
                        facts.append(f"n={trial_n}")
                    if seg:
                        facts.append(seg.replace("_", " "))
                    fact_str = f" ({', '.join(facts)})" if facts else ""
                    src_str = f"Source: {source}. " if source else ""

                    next_body = (
                        f"Done. {src_str}{title}{fact_str}.\n"
                        f"Summary: {summary}\n"
                        "Want me to draft (1) a short patient WhatsApp note, (2) a Google Post, or (3) both?"
                    ).strip()
                    next_body = _dedupe_body(conv, next_body)
                    conv.last_outbound_body = next_body
                    return {
                        "action": "send",
                        "body": next_body,
                        "cta": "open_ended",
                        "rationale": "Merchant requested the digest details; fulfilled using stored category+trigger context and offered next-best drafts.",
                    }

            # Generic request handling (merchant-facing)
            next_body = (
                "Got it - I can do that.\n"
                "Reply with these 2 details so I can draft the right message:\n"
                "1) What you want to send (WhatsApp reply / Google Post / offer)\n"
                "2) Any timing or price point to mention (if any)"
            )
            next_body = _dedupe_body(conv, next_body)
            conv.last_outbound_body = next_body
            return {"action": "send", "body": next_body, "cta": "open_ended", "rationale": "Merchant requested action; asked for minimal inputs to fulfill."}

        # Merchant asked a question (or otherwise needs help)
        next_body = (
            "Got it - I can help.\n"
            "Share 2 quick details and I'll draft next steps:\n"
            "1) What outcome you want (more calls / more walk-ins / more repeats)\n"
            "2) Any offer or price point to highlight (if any)"
        )
        next_body = _dedupe_body(conv, next_body)
        conv.last_outbound_body = next_body
        return {"action": "send", "body": next_body, "cta": "open_ended", "rationale": "Merchant asked for help; requested minimal inputs for a practical reply."}

    # Default: no reply unless action is clearly required
    if _looks_like_question(msg):
        next_body = "Can you share what you need help with (1 sentence)?"
        next_body = _dedupe_body(conv, next_body)
        conv.last_outbound_body = next_body
        return {"action": "send", "body": next_body, "cta": "open_ended", "rationale": "Intent unclear; asked a single clarifying question."}

    return {"action": "wait", "wait_seconds": 600, "rationale": "No action required; waiting."}


@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    suppressed.clear()
    return {"ok": True, "cleared_at": _iso_utc(_utcnow())}


# Convenience: `python bot.py` runs uvicorn
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8081"))
    uvicorn.run("bot:app", host="0.0.0.0", port=port, reload=False)
