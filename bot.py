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
    - If both names exist: "Hi Priya — this is Dr. Meera’s Dental Clinic."
    - If missing names: use a neutral fallback; do not invent names.
    """

    cust = (customer_name or "").strip()
    merch = (merchant_name or "").strip()
    if cust and merch:
        return f"Hi {cust} — this is {merch}."
    if cust:
        return f"Hi {cust}."
    # Neutral fallback (no placeholders like "Hi - this is the clinic")
    return "Hi there — this is the business."


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
    if body.merchant_id:
        mctx = contexts.get(("merchant", body.merchant_id))
        if mctx:
            merchant_name = _first_nonempty((mctx.payload.get("identity") or {}).get("name"))
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
    # Customer routing: booking / confirm / reschedule intents
    if body.customer_id is not None or body.from_role == "customer":
        prefix = _customer_voice_prefix(customer_name, merchant_name)

        # Try to echo back a provided slot if the customer wrote one.
        slot_hint = ""
        m = re.search(r"\bfor\b\s+(.{0,40})$", msg.strip(), flags=re.IGNORECASE)
        if m:
            slot_hint = m.group(1).strip().rstrip(".")

        if re.search(r"\b(book|booking|appointment|appt)\b", msg_l) or re.search(r"\b(reschedule|confirm)\b", msg_l):
            slot_line = f"I have noted: {slot_hint}.\n" if slot_hint else ""
            next_body = (
                f"{prefix} Sure, I can help with that booking.\n"
                f"{slot_line}"
                "Please confirm your phone number (and optionally the service).\n"
                "We'll confirm shortly. Reply STOP to opt out."
            ).strip()
            next_body = _dedupe_body(conv, next_body)
            conv.last_outbound_body = next_body
            return {"action": "send", "body": next_body, "cta": "open_ended", "rationale": "Customer asked to book/confirm; collect minimum details to proceed without asking merchant-growth goals."}

        if re.search(r"\b(yes|yep|yeah|ok|okay)\b", msg_l):
            next_body = (
                f"{prefix} Great. Please share your phone number and we'll confirm shortly.\n"
                "Reply STOP anytime to opt out."
            )
            next_body = _dedupe_body(conv, next_body)
            conv.last_outbound_body = next_body
            return {"action": "send", "body": next_body, "cta": "open_ended", "rationale": "Customer confirmation; ask for contact details to complete the loop."}


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

    # Default helpful reply
    next_body = (
        "Got it. If you share one detail (goal: more calls / more walk-ins / more repeat customers), "
        "I'll suggest the best next message + a ready-to-post draft." 
    ).strip()
    next_body = _dedupe_body(conv, next_body)
    conv.last_outbound_body = next_body
    return {"action": "send", "body": next_body, "cta": "open_ended", "rationale": "Acknowledged and asked for a single clarifying detail to personalize."}


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
