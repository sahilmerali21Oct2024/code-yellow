import os
import smtplib
import ssl
import urllib.parse
from email.message import EmailMessage

import psycopg2
import psycopg2.extras
import pandas as pd
from datetime import datetime, timezone, timedelta
from databricks.sdk.core import Config

import dash
from dash import dcc, html, callback, Input, Output, State, ALL, no_update, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
LAKEBASE_PROJECT = "code-yellow"
LAKEBASE_ENDPOINT = f"projects/{LAKEBASE_PROJECT}/branches/production/endpoints/primary"
LAKEBASE_DB = "databricks_postgres"

# Email / paging configuration
PAGE_EMAIL_TO = os.getenv("PAGE_EMAIL_TO", "sahil.merali@databricks.com")
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER or "code-yellow@databricks.com")

# Akron Children's Hospital brand palette
PURPLE = "#5C2D91"
PURPLE_DARK = "#421F69"
PURPLE_LIGHT = "#EDE6F5"
GREEN = "#7AB648"
TEAL = "#00A8A8"
TEAL_DARK = "#007A7A"
BG_PAGE = "#F4F6F8"
BG_CARD = "#FFFFFF"
TEXT_PRIMARY = "#1F2937"
TEXT_MUTED = "#6B7280"
BORDER_SUBTLE = "#E5E7EB"
RED_ALERT = "#DC3545"
ORANGE_ALERT = "#FD7E14"
GRAY_INACTIVE = "#ADB5BD"

HOSPITAL_UNITS = [
    {"id": "nicu", "name": "NICU"},
    {"id": "picu", "name": "PICU"},
    {"id": "ed", "name": "ED"},
    {"id": "or", "name": "OR"},
    {"id": "pharmacy", "name": "Pharmacy"},
    {"id": "radiology", "name": "Radiology"},
    {"id": "4east", "name": "4 East"},
    {"id": "4west", "name": "4 West"},
    {"id": "5east", "name": "5 East"},
    {"id": "5west", "name": "5 West"},
]
UNIT_NAME_BY_ID = {u["id"]: u["name"] for u in HOSPITAL_UNITS}

LOCATION_TO_UNIT = {
    "nicu": "nicu", "neonatal intensive care": "nicu",
    "picu": "picu", "pediatric intensive care": "picu",
    "emergency": "ed", "emergency department": "ed", "ed": "ed",
    "operating room": "or", "surgery": "or", "or": "or",
    "pharmacy": "pharmacy", "pharmacy dispensing": "pharmacy",
    "radiology": "radiology", "pacs": "radiology", "imaging": "radiology",
    "4 east": "4east", "4east": "4east", "floor 4 east": "4east",
    "4 west": "4west", "4west": "4west", "floor 4 west": "4west",
    "5 east": "5east", "5east": "5east", "floor 5 east": "5east",
    "5 west": "5west", "5west": "5west", "floor 5 west": "5west",
}


# ─────────────────────────────────────────────────────────────────────────────
# LAKEBASE CONNECTION (OAuth-based for Autoscale)
# ─────────────────────────────────────────────────────────────────────────────
_cached_host = None
_cached_token = None
_token_expiry = None


def get_lakebase_connection():
    global _cached_host, _cached_token, _token_expiry
    import requests

    cfg = Config()
    now = datetime.now(timezone.utc)
    if _cached_host is None or _token_expiry is None or now >= _token_expiry:
        host = cfg.host.rstrip("/")
        auth_headers = cfg.authenticate()
        auth_headers["Content-Type"] = "application/json"

        ep_resp = requests.get(
            f"{host}/api/2.0/postgres/{LAKEBASE_ENDPOINT}",
            headers=auth_headers,
        )
        ep_resp.raise_for_status()
        _cached_host = ep_resp.json()["status"]["hosts"]["host"]

        cred_resp = requests.post(
            f"{host}/api/2.0/postgres/credentials",
            headers=auth_headers,
            json={"endpoint": LAKEBASE_ENDPOINT},
        )
        cred_resp.raise_for_status()
        _cached_token = cred_resp.json()["token"]
        _token_expiry = now + timedelta(minutes=45)

    username = os.getenv("DATABRICKS_CLIENT_ID") or os.getenv("USER", "unknown")
    return psycopg2.connect(
        host=_cached_host,
        database=LAKEBASE_DB,
        user=username,
        password=_cached_token,
        port=5432,
        sslmode="require",
    )


def query_db(sql, params=None):
    conn = get_lakebase_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()


def execute_db(sql, params=None):
    conn = get_lakebase_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL IMPACT CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
def classify_clinical_impact(row):
    priority = row.get("priority")
    patient_safety = row.get("u_patient_safety_impact")
    impact = row.get("impact")
    is_clinical_ci = row.get("is_clinical")

    if priority == 1:
        return "life-safety"
    if patient_safety and str(patient_safety).lower() in ("true", "1", "yes"):
        return "life-safety"
    if is_clinical_ci and str(is_clinical_ci).lower() in ("true", "1", "yes") and impact == 1:
        return "life-safety"
    if priority == 2:
        return "care-delivery"
    if is_clinical_ci and str(is_clinical_ci).lower() in ("true", "1", "yes"):
        return "care-delivery"
    return "administrative"


def get_unit_from_location(location_name):
    if not location_name:
        return None
    loc_lower = location_name.lower().strip()
    for key, unit_id in LOCATION_TO_UNIT.items():
        if key in loc_lower:
            return unit_id
    return None


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
def get_active_incidents():
    sql = """
        SELECT
            i.sys_id, i.number, i.short_description, i.category,
            i.priority, i.impact, i.urgency, i.opened_at,
            i.assignment_group, i.u_clinical_impact, i.u_patient_safety_impact,
            c.name AS ci_name, c.is_clinical, c.service_tier,
            l.name AS location_name
        FROM service_now.synced_incident i
        LEFT JOIN service_now.synced_cmdb_ci c ON i.cmdb_ci = c.sys_id
        LEFT JOIN service_now.synced_cmn_location l ON i.location = l.sys_id
        WHERE i.active = true
        ORDER BY i.priority ASC, i.opened_at ASC
    """
    rows = query_db(sql)
    for row in rows:
        row["clinical_impact_tier"] = classify_clinical_impact(row)
        row["affected_unit"] = get_unit_from_location(row.get("location_name"))
        if row.get("opened_at"):
            opened = row["opened_at"]
            if hasattr(opened, "tzinfo") and opened.tzinfo is None:
                opened = opened.replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - opened
            hours = delta.total_seconds() / 3600
            if hours < 1:
                row["time_open"] = f"{int(delta.total_seconds() / 60)}m"
            elif hours < 24:
                row["time_open"] = f"{hours:.1f}h"
            else:
                row["time_open"] = f"{delta.days}d {int(hours % 24)}h"
        else:
            row["time_open"] = "N/A"
    return rows


def get_mttr_by_unit():
    sql = """
        SELECT
            l.name AS location_name,
            DATE(i.resolved_at) AS resolve_date,
            AVG(EXTRACT(EPOCH FROM (i.resolved_at - i.opened_at)) / 3600) AS mttr_hours
        FROM service_now.synced_incident i
        LEFT JOIN service_now.synced_cmn_location l ON i.location = l.sys_id
        WHERE i.resolved_at IS NOT NULL
          AND i.resolved_at >= NOW() - INTERVAL '30 days'
          AND l.name IS NOT NULL
        GROUP BY l.name, DATE(i.resolved_at)
        ORDER BY resolve_date ASC
    """
    return query_db(sql)


def get_assignment_groups():
    sql = "SELECT sys_id, name FROM service_now.synced_sys_user_group WHERE active = true ORDER BY name"
    return query_db(sql)


PAGE_TYPE_LABELS = {
    "page_team":     "Page Team",
    "bridge_call":   "Bridge Call",
    "status_update": "Status Update",
}


def build_page_email(incident, form):
    """Construct (subject, body, mailto_url) for an incident page email.

    incident: dict from incidents-store (has number, short_description,
              clinical_impact_tier, affected_unit, and we resolve more via DB)
    form: {'page_type', 'group', 'unit', 'message', 'priority', 'description'}
    """
    inc_number   = incident.get("number") or "[INC number]"
    priority     = form.get("priority")
    priority_str = f"P{priority}" if priority not in (None, "", "?") else "[P1 / P2 / P3]"
    group        = form.get("group") or "[team/group being paged]"
    unit_id      = form.get("unit")
    unit_label   = UNIT_NAME_BY_ID.get(unit_id, unit_id) if unit_id else "[system, application, or service impacted]"
    message      = form.get("message") or "[free text description of the issue, impact, and any immediate actions needed]"
    action_label = PAGE_TYPE_LABELS.get(form.get("page_type"), "Page Team")

    body_lines = [
        "**INCIDENT PAGE**",
        "",
        f"Incident: {inc_number}",
        f"Page Type: {priority_str}",
        f"Target Group: {group}",
        f"Affected Unit: {unit_label}",
        "",
        "Message:",
        message,
        "",
        "—",
        f"Page action: {action_label}",
        f"Sent by Code Yellow at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
    ]
    body = "\n".join(body_lines)
    subject = f"[{priority_str}] INCIDENT PAGE — {inc_number} — {action_label}"

    mailto = (
        f"mailto:{PAGE_EMAIL_TO}"
        f"?subject={urllib.parse.quote(subject)}"
        f"&body={urllib.parse.quote(body)}"
    )
    return subject, body, mailto


def send_page_email(subject, body):
    """Attempt to send email via SMTP. Returns (sent: bool, detail: str)."""
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD:
        return False, "SMTP not configured — using mailto fallback."

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = PAGE_EMAIL_TO
    msg.set_content(body)

    try:
        ctx_ssl = ssl.create_default_context()
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx_ssl, timeout=15) as s:
                s.login(SMTP_USER, SMTP_PASSWORD)
                s.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as s:
                s.ehlo()
                s.starttls(context=ctx_ssl)
                s.ehlo()
                s.login(SMTP_USER, SMTP_PASSWORD)
                s.send_message(msg)
        return True, f"Email sent to {PAGE_EMAIL_TO}."
    except Exception as e:
        return False, f"SMTP error: {str(e)[:120]}"


def get_recent_pages():
    sql = """
        SELECT incident_number, paged_group, page_type, message,
               clinical_impact_tier, affected_unit, created_at
        FROM public.paged_actions ORDER BY created_at DESC LIMIT 10
    """
    try:
        return query_db(sql)
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# CLICKABLE FLOOR MAP (CSS grid of tile buttons)
# ─────────────────────────────────────────────────────────────────────────────
TIER_COLORS = {
    "life-safety": RED_ALERT,
    "care-delivery": ORANGE_ALERT,
    "administrative": GREEN,
}
TIER_SEVERITY = {"life-safety": 3, "care-delivery": 2, "administrative": 1}


def build_unit_status(incidents):
    """Return {unit_id: {'tier': ..., 'count': int, 'life_safety': int}}."""
    status = {}
    for inc in incidents:
        unit = inc.get("affected_unit")
        if not unit:
            continue
        tier = inc["clinical_impact_tier"]
        entry = status.setdefault(unit, {"tier": None, "severity": 0, "count": 0, "life_safety": 0})
        entry["count"] += 1
        if tier == "life-safety":
            entry["life_safety"] += 1
        if TIER_SEVERITY.get(tier, 0) > entry["severity"]:
            entry["severity"] = TIER_SEVERITY[tier]
            entry["tier"] = tier
    return status


def build_floor_map(incidents, selected_unit):
    status = build_unit_status(incidents)
    tiles = []
    for unit in HOSPITAL_UNITS:
        uid = unit["id"]
        info = status.get(uid)
        tier = info["tier"] if info else None
        count = info["count"] if info else 0
        bg = TIER_COLORS.get(tier, GRAY_INACTIVE) if tier else "#FFFFFF"
        text_color = "white" if tier else TEXT_PRIMARY
        is_pulse = tier == "life-safety"
        is_selected = selected_unit == uid

        tile_class = "floor-tile"
        if is_pulse:
            tile_class += " floor-tile--pulse"
        if is_selected:
            tile_class += " floor-tile--selected"

        badge = None
        if count > 0:
            badge = html.Span(
                str(count),
                className="floor-tile__badge",
                style={"background": "white", "color": TIER_COLORS.get(tier, TEXT_PRIMARY)},
            )

        tiles.append(
            html.Button(
                [
                    html.Div(unit["name"], className="floor-tile__name"),
                    html.Div(
                        f"{count} active" if count else "All clear",
                        className="floor-tile__meta",
                    ),
                    badge,
                ],
                id={"type": "unit-tile", "unit": uid},
                n_clicks=0,
                className=tile_class,
                style={
                    "background": bg,
                    "color": text_color,
                    "borderColor": PURPLE if is_selected else (TIER_COLORS.get(tier, BORDER_SUBTLE) if tier else BORDER_SUBTLE),
                },
            )
        )

    return html.Div(tiles, className="floor-grid")


# ─────────────────────────────────────────────────────────────────────────────
# DASH APP SETUP
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME,
        "https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700;900&display=swap",
    ],
    title="Code Yellow | Clinical Impact Command Center",
    suppress_callback_exceptions=True,
)

CUSTOM_CSS = f"""
:root {{
  --ach-purple: {PURPLE};
  --ach-purple-dark: {PURPLE_DARK};
  --ach-purple-light: {PURPLE_LIGHT};
  --ach-teal: {TEAL};
  --ach-green: {GREEN};
  --bg-page: {BG_PAGE};
  --bg-card: {BG_CARD};
  --text-primary: {TEXT_PRIMARY};
  --text-muted: {TEXT_MUTED};
  --border-subtle: {BORDER_SUBTLE};
}}

* {{ box-sizing: border-box; }}
html, body {{
  font-family: 'Lato', 'Helvetica Neue', Helvetica, Arial, sans-serif;
  background: var(--bg-page);
  color: var(--text-primary);
  -webkit-font-smoothing: antialiased;
  margin: 0;
}}

/* ── HEADER ───────────────────────────────────────────── */
.app-header {{
  background: var(--bg-card);
  border-bottom: 4px solid var(--ach-purple);
  padding: 18px 28px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.app-header__brand {{ display: flex; align-items: center; gap: 14px; }}
.app-header__logo {{
  width: 44px; height: 44px; border-radius: 10px;
  background: linear-gradient(135deg, var(--ach-purple) 0%, var(--ach-teal) 100%);
  display: inline-flex; align-items: center; justify-content: center;
  color: white; font-size: 1.3rem;
}}
.app-header__title {{
  font-size: 1.35rem; font-weight: 900;
  letter-spacing: 0.02em; color: var(--ach-purple); line-height: 1.1;
}}
.app-header__subtitle {{
  font-size: 0.85rem; font-weight: 400; color: var(--text-muted); margin-top: 2px;
}}
.app-header__meta {{ display: flex; align-items: center; gap: 16px; }}
.app-header__time {{ font-size: 0.85rem; color: var(--text-muted); }}

/* ── LEGEND ───────────────────────────────────────────── */
.legend-bar {{
  background: var(--bg-card);
  padding: 10px 28px;
  border-bottom: 1px solid var(--border-subtle);
  display: flex; flex-wrap: wrap; gap: 18px; align-items: center;
  font-size: 0.82rem; color: var(--text-muted);
}}
.legend-item {{ display: inline-flex; align-items: center; gap: 6px; }}
.legend-dot {{
  width: 12px; height: 12px; border-radius: 3px; display: inline-block;
}}

/* ── CARDS ────────────────────────────────────────────── */
.page-body {{ padding: 24px 28px; }}
.panel {{
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  box-shadow: 0 1px 2px rgba(16,24,40,0.04), 0 1px 3px rgba(16,24,40,0.05);
  overflow: hidden;
  height: 100%;
}}
.panel__header {{
  padding: 14px 18px;
  border-bottom: 1px solid var(--border-subtle);
  display: flex; align-items: center; justify-content: space-between;
  background: linear-gradient(180deg, #FAFAFC 0%, #FFFFFF 100%);
}}
.panel__title {{
  font-size: 0.95rem; font-weight: 700; color: var(--ach-purple);
  margin: 0; letter-spacing: 0.01em;
}}
.panel__body {{ padding: 16px 18px; }}

/* ── FLOOR MAP TILES ─────────────────────────────────── */
.floor-grid {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  padding: 4px;
}}
.floor-tile {{
  position: relative;
  font-family: inherit;
  border: 2px solid var(--border-subtle);
  border-radius: 10px;
  padding: 14px 12px;
  min-height: 90px;
  display: flex; flex-direction: column; justify-content: center;
  text-align: center;
  cursor: pointer;
  transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease;
  outline: none;
}}
.floor-tile:hover {{
  transform: translateY(-2px);
  box-shadow: 0 6px 14px rgba(92,45,145,0.18);
  filter: brightness(1.05);
}}
.floor-tile:focus-visible {{ box-shadow: 0 0 0 3px rgba(92,45,145,0.35); }}
.floor-tile--selected {{
  box-shadow: 0 0 0 3px var(--ach-purple), 0 6px 14px rgba(92,45,145,0.25);
  transform: translateY(-1px);
}}
.floor-tile__name {{
  font-size: 1.05rem; font-weight: 900; letter-spacing: 0.03em;
}}
.floor-tile__meta {{
  font-size: 0.72rem; font-weight: 400; opacity: 0.9; margin-top: 4px;
}}
.floor-tile__badge {{
  position: absolute; top: 6px; right: 8px;
  font-size: 0.72rem; font-weight: 900;
  padding: 2px 7px; border-radius: 999px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.15);
}}
@keyframes pulseTile {{
  0%, 100% {{ box-shadow: 0 0 0 0 rgba(220,53,69,0.55); }}
  50%      {{ box-shadow: 0 0 0 10px rgba(220,53,69,0); }}
}}
.floor-tile--pulse {{ animation: pulseTile 1.6s ease-in-out infinite; }}

/* ── BADGES ──────────────────────────────────────────── */
.tier-pill {{
  display: inline-block;
  padding: 2px 9px; border-radius: 999px;
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.04em;
  text-transform: uppercase; color: white;
}}
.tier-pill--life-safety   {{ background: {RED_ALERT}; }}
.tier-pill--care-delivery {{ background: {ORANGE_ALERT}; }}
.tier-pill--administrative {{ background: {GREEN}; }}

.count-pill {{
  background: var(--ach-purple); color: white;
  padding: 4px 12px; border-radius: 999px;
  font-size: 0.8rem; font-weight: 700;
}}
.count-pill--alert {{ background: {RED_ALERT}; }}
.count-pill--warn  {{ background: {ORANGE_ALERT}; }}
.count-pill--ok    {{ background: var(--ach-green); }}

/* ── FILTER BANNER ───────────────────────────────────── */
.filter-banner {{
  display: flex; align-items: center; justify-content: space-between;
  background: var(--ach-purple-light);
  border: 1px solid #D8C7EE;
  color: var(--ach-purple-dark);
  padding: 8px 12px; border-radius: 8px;
  margin: 0 0 12px 0;
  font-size: 0.85rem;
}}
.filter-banner__clear {{
  background: transparent; border: 1px solid var(--ach-purple);
  color: var(--ach-purple); font-weight: 700; font-size: 0.78rem;
  padding: 3px 10px; border-radius: 6px; cursor: pointer;
}}
.filter-banner__clear:hover {{ background: var(--ach-purple); color: white; }}

/* ── INCIDENT CARDS (expandable) ─────────────────────── */
.inc-list {{ display: flex; flex-direction: column; gap: 6px; }}
.inc-card {{
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  background: var(--bg-card);
  transition: box-shadow 120ms ease, border-color 120ms ease;
  overflow: hidden;
}}
.inc-card:hover {{
  border-color: #D8C7EE;
  box-shadow: 0 2px 6px rgba(92,45,145,0.08);
}}
.inc-card[open] {{
  border-color: var(--ach-purple);
  box-shadow: 0 4px 14px rgba(92,45,145,0.10);
}}
.inc-card__summary {{
  list-style: none;
  cursor: pointer;
  padding: 10px 12px;
  display: grid;
  grid-template-columns: 18px 110px 110px 1fr 40px 70px;
  align-items: center;
  gap: 10px;
  font-size: 0.85rem;
  user-select: none;
}}
.inc-card__summary::-webkit-details-marker {{ display: none; }}
.inc-card__chevron {{
  display: inline-block;
  color: var(--ach-purple);
  font-size: 0.9rem;
  transition: transform 150ms ease;
}}
.inc-card[open] .inc-card__chevron {{ transform: rotate(90deg); }}
.inc-card__num  {{ font-weight: 700; color: var(--ach-purple); }}
.inc-card__pill {{ justify-self: start; }}
.inc-card__desc {{
  color: var(--text-primary);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.inc-card__pri  {{ font-weight: 700; color: var(--text-muted); text-align: center; }}
.inc-card__time {{ color: var(--text-muted); font-size: 0.8rem; text-align: right; }}

.inc-detail__grid {{
  border-top: 1px dashed var(--border-subtle);
  background: #FAFBFD;
  padding: 12px 14px;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  column-gap: 18px;
  row-gap: 6px;
  font-size: 0.82rem;
}}
.inc-detail__row {{
  display: grid;
  grid-template-columns: 150px 1fr;
  gap: 8px;
  padding: 3px 0;
  border-bottom: 1px solid #EEF0F3;
  align-items: start;
}}
.inc-detail__label {{
  color: var(--text-muted);
  font-size: 0.72rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  font-weight: 700;
  padding-top: 1px;
}}
.inc-detail__value {{
  color: var(--text-primary);
  word-break: break-word;
}}
@media (max-width: 1200px) {{
  .inc-detail__grid {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 700px) {{
  .inc-card__summary {{ grid-template-columns: 18px 1fr 60px; gap: 6px; }}
  .inc-card__pill, .inc-card__pri, .inc-card__time {{ display: none; }}
}}

.empty-state {{
  text-align: center; padding: 36px 12px; color: var(--text-muted); font-size: 0.9rem;
}}

/* ── BUTTONS ─────────────────────────────────────────── */
.btn-primary-ach {{
  background: var(--ach-purple); border-color: var(--ach-purple);
  font-weight: 700;
}}
.btn-primary-ach:hover {{ background: var(--ach-purple-dark); border-color: var(--ach-purple-dark); }}

/* ── PAGES FEED ──────────────────────────────────────── */
.page-feed__item {{
  padding: 10px 0; border-bottom: 1px solid #F1F3F5;
}}
.page-feed__item:last-child {{ border-bottom: none; }}
.page-feed__title {{ font-weight: 700; color: var(--text-primary); font-size: 0.88rem; }}
.page-feed__meta  {{ color: var(--text-muted); font-size: 0.78rem; margin-top: 2px; }}

@media (max-width: 900px) {{
  .floor-grid {{ grid-template-columns: repeat(2, 1fr); }}
  .page-body {{ padding: 16px; }}
}}
"""

app.index_string = (
    '<!DOCTYPE html>\n<html>\n<head>\n'
    '    {%metas%}\n    <title>{%title%}</title>\n    {%favicon%}\n    {%css%}\n'
    '    <style>' + CUSTOM_CSS + '</style>\n'
    '</head>\n<body>\n    {%app_entry%}\n'
    '    <footer>{%config%}{%scripts%}{%renderer%}</footer>\n'
    '</body>\n</html>'
)


def header():
    return html.Div(
        [
            html.Div([
                html.Div(html.I(className="fas fa-heart-pulse"), className="app-header__logo"),
                html.Div([
                    html.Div("CODE YELLOW", className="app-header__title"),
                    html.Div("Clinical Impact Command Center", className="app-header__subtitle"),
                ]),
            ], className="app-header__brand"),
            html.Div([
                html.Span(id="active-count-badge"),
                html.Span(id="current-time", className="app-header__time"),
            ], className="app-header__meta"),
        ],
        className="app-header",
    )


def legend():
    return html.Div(
        [
            html.Span([html.Span(className="legend-dot", style={"background": RED_ALERT}), "Life-Safety"], className="legend-item"),
            html.Span([html.Span(className="legend-dot", style={"background": ORANGE_ALERT}), "Care-Delivery"], className="legend-item"),
            html.Span([html.Span(className="legend-dot", style={"background": GREEN}), "Administrative"], className="legend-item"),
            html.Span([html.Span(className="legend-dot", style={"background": GRAY_INACTIVE}), "No Active Incidents"], className="legend-item"),
            html.Span("Tip: click any unit on the floor map to filter the incident list.",
                      style={"marginLeft": "auto", "fontStyle": "italic", "color": TEAL_DARK}),
        ],
        className="legend-bar",
    )


app.layout = html.Div([
    dcc.Interval(id="interval-refresh", interval=30_000, n_intervals=0),
    dcc.Store(id="incidents-store"),
    dcc.Store(id="selected-unit-store", data=None),

    header(),
    legend(),

    html.Div([
        dbc.Row([
            dbc.Col(html.Div([
                html.Div([
                    html.H6("Hospital Floor Map", className="panel__title"),
                    html.Span("Click a unit to filter", style={"fontSize": "0.78rem", "color": TEXT_MUTED}),
                ], className="panel__header"),
                html.Div(html.Div(id="floor-map-container"), className="panel__body"),
            ], className="panel"), md=6, className="mb-3"),

            dbc.Col(html.Div([
                html.Div([
                    html.H6("Active Incidents", className="panel__title"),
                    html.Div([
                        html.Button("Clear filter", id="clear-filter-btn", n_clicks=0,
                                    className="filter-banner__clear me-2",
                                    style={"display": "none"}),
                        dbc.Button([html.I(className="fas fa-bullhorn me-1"), "Page Team"],
                                   id="open-page-modal", color="danger", size="sm",
                                   className="btn-primary-ach"),
                    ]),
                ], className="panel__header"),
                html.Div([
                    html.Div(id="filter-banner-container"),
                    html.Div(id="incidents-table-container", style={"maxHeight": "360px", "overflowY": "auto"}),
                ], className="panel__body"),
            ], className="panel"), md=6, className="mb-3"),
        ]),

        dbc.Row([
            dbc.Col(html.Div([
                html.Div(html.H6("Mean Time To Resolve (MTTR) — by Unit · Last 30 Days",
                                 id="mttr-title", className="panel__title"),
                         className="panel__header"),
                html.Div(dcc.Graph(id="mttr-chart", config={"displayModeBar": False}), className="panel__body"),
            ], className="panel"), md=7, className="mb-3"),

            dbc.Col(html.Div([
                html.Div(html.H6("Recent Page Actions", className="panel__title"), className="panel__header"),
                html.Div(html.Div(id="recent-pages-container", style={"maxHeight": "300px", "overflowY": "auto"}),
                         className="panel__body"),
            ], className="panel"), md=5, className="mb-3"),
        ]),
    ], className="page-body"),

    # Page modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle([
            html.I(className="fas fa-bullhorn me-2", style={"color": RED_ALERT}),
            "Page the Right Team",
        ])),
        dbc.ModalBody([dbc.Form([
            dbc.Row([
                dbc.Col([dbc.Label("Incident"),
                    dcc.Dropdown(id="page-incident-select", placeholder="Select incident...")], md=6),
                dbc.Col([dbc.Label("Page Type"),
                    dcc.Dropdown(id="page-type-select", options=[
                        {"label": "Page Team", "value": "page_team"},
                        {"label": "Bridge Call", "value": "bridge_call"},
                        {"label": "Status Update", "value": "status_update"},
                    ], value="page_team")], md=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([dbc.Label("Target Group"),
                    dcc.Dropdown(id="page-group-select", placeholder="Select team...")], md=6),
                dbc.Col([dbc.Label("Affected Unit"),
                    dcc.Dropdown(id="page-unit-select",
                        options=[{"label": u["name"], "value": u["id"]} for u in HOSPITAL_UNITS],
                        placeholder="Select unit...")], md=6),
            ], className="mb-3"),
            dbc.Row([dbc.Col([dbc.Label("Message"),
                dbc.Textarea(id="page-message",
                    placeholder="e.g., Workaround in place on 4 East...", rows=3)])],
                className="mb-3"),
        ])]),
        dbc.ModalFooter([
            html.Div(id="page-feedback"),
            dbc.Button("Cancel", id="close-page-modal", className="me-2", outline=True, color="secondary"),
            dbc.Button([html.I(className="fas fa-paper-plane me-1"), "Send Page"],
                       id="submit-page", color="danger"),
        ]),
    ], id="page-modal", is_open=False, size="lg"),
])


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
def _fmt(value, dash="\u2014"):
    if value is None or value == "":
        return dash
    return str(value)


def _fmt_dt(value, dash="\u2014"):
    if not value:
        return dash
    try:
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass
    return str(value)


def render_incident_card(inc):
    tier = inc["clinical_impact_tier"]
    detail_rows = [
        ("Incident #",          _fmt(inc.get("number"))),
        ("Short description",   _fmt(inc.get("short_description"))),
        ("Clinical impact",     tier.replace("-", " ").title()),
        ("Priority",            f"P{_fmt(inc.get('priority'))}"),
        ("Impact",              _fmt(inc.get("impact"))),
        ("Urgency",             _fmt(inc.get("urgency"))),
        ("Category",            _fmt(inc.get("category"))),
        ("Assignment group",    _fmt(inc.get("assignment_group"))),
        ("Location",            _fmt(inc.get("location_name"))),
        ("Mapped unit",         _fmt(UNIT_NAME_BY_ID.get(inc.get("affected_unit")) if inc.get("affected_unit") else None)),
        ("CI name",             _fmt(inc.get("ci_name"))),
        ("Clinical CI",         _fmt(inc.get("is_clinical"))),
        ("Service tier",        _fmt(inc.get("service_tier"))),
        ("Patient-safety flag", _fmt(inc.get("u_patient_safety_impact"))),
        ("Clinical-impact tag", _fmt(inc.get("u_clinical_impact"))),
        ("Opened",              _fmt_dt(inc.get("opened_at"))),
        ("Time open",           _fmt(inc.get("time_open"))),
        ("sys_id",              _fmt(inc.get("sys_id"))),
    ]
    details = html.Div(
        [html.Div([html.Span(label, className="inc-detail__label"),
                   html.Span(value, className="inc-detail__value")],
                  className="inc-detail__row") for label, value in detail_rows],
        className="inc-detail__grid",
    )
    summary = html.Summary([
        html.Span("\u25B8", className="inc-card__chevron"),
        html.Span(inc.get("number") or "—", className="inc-card__num"),
        html.Span(tier.replace("-", " ").title(), className=f"tier-pill tier-pill--{tier} inc-card__pill"),
        html.Span((inc.get("short_description") or "")[:70], className="inc-card__desc"),
        html.Span(f"P{_fmt(inc.get('priority'))}", className="inc-card__pri"),
        html.Span(inc.get("time_open") or "—", className="inc-card__time"),
    ], className="inc-card__summary")
    return html.Details([summary, details], className="inc-card")


def render_incidents_list(incidents):
    if not incidents:
        return html.Div("No active incidents.", className="empty-state")
    return html.Div(
        [render_incident_card(i) for i in incidents],
        className="inc-list",
    )


def render_filter_banner(selected_unit, total_visible, total_overall):
    """Returns (banner_children, clear_btn_style)."""
    if not selected_unit:
        return None, {"display": "none"}
    name = UNIT_NAME_BY_ID.get(selected_unit, selected_unit)
    banner = html.Div([
        html.Span([
            html.I(className="fas fa-filter me-2"),
            html.Span("Filtered by unit: "),
            html.Strong(name),
            html.Span(f"  ·  {total_visible} of {total_overall} incidents",
                      style={"marginLeft": "8px", "color": TEXT_MUTED}),
        ]),
        html.Span("Click the tile again or use Clear to reset.",
                  style={"fontSize": "0.78rem", "color": TEXT_MUTED, "fontStyle": "italic"}),
    ], className="filter-banner")
    return banner, {"display": "inline-block"}


@callback(
    [Output("floor-map-container", "children"),
     Output("incidents-table-container", "children"),
     Output("filter-banner-container", "children"),
     Output("clear-filter-btn", "style"),
     Output("active-count-badge", "children"),
     Output("current-time", "children"),
     Output("incidents-store", "data"),
     Output("recent-pages-container", "children")],
    [Input("interval-refresh", "n_intervals"),
     Input("selected-unit-store", "data")],
)
def refresh_data(_n, selected_unit):
    try:
        incidents = get_active_incidents()
    except Exception as e:
        incidents = []
        print(f"Error fetching incidents: {e}")

    floor_map = build_floor_map(incidents, selected_unit)

    visible_incidents = [i for i in incidents if (not selected_unit or i.get("affected_unit") == selected_unit)]
    if visible_incidents:
        table = render_incidents_list(visible_incidents)
    elif selected_unit:
        table = html.Div(f"No active incidents for {UNIT_NAME_BY_ID.get(selected_unit, selected_unit)}.",
                         className="empty-state")
    else:
        table = html.Div("No active incidents.", className="empty-state")

    banner, clear_btn_style = render_filter_banner(selected_unit, len(visible_incidents), len(incidents))

    life_safety = sum(1 for i in incidents if i["clinical_impact_tier"] == "life-safety")
    total = len(incidents)
    pill_class = "count-pill"
    if life_safety > 0:
        pill_class += " count-pill--alert"
    elif total > 0:
        pill_class += " count-pill--warn"
    else:
        pill_class += " count-pill--ok"
    count_badge = html.Span(
        f"{total} ACTIVE  ·  {life_safety} LIFE-SAFETY",
        className=pill_class,
    )

    now_str = datetime.now(timezone.utc).strftime("%a %b %d  ·  %H:%M UTC")

    store_data = [{
        "sys_id": i["sys_id"], "number": i["number"],
        "short_description": i.get("short_description", ""),
        "clinical_impact_tier": i["clinical_impact_tier"],
        "affected_unit": i.get("affected_unit"),
        "priority": i.get("priority"),
        "location_name": i.get("location_name"),
    } for i in incidents]

    try:
        pages = get_recent_pages()
        if pages:
            recent_pages = html.Div([
                html.Div([
                    html.Div([
                        html.Span(p.get("incident_number", ""), className="page-feed__title"),
                        html.Span(f"  →  {p.get('paged_group', '')}",
                                 style={"color": TEXT_MUTED, "fontSize": "0.85rem", "marginLeft": "4px"}),
                    ]),
                    html.Div([
                        html.Span(p.get("page_type", "").replace("_", " ").title(),
                                  className="tier-pill",
                                  style={"background": TEAL, "marginRight": "6px"}),
                        html.Span((p.get("message") or "")[:80]),
                    ], className="page-feed__meta"),
                ], className="page-feed__item") for p in pages
            ])
        else:
            recent_pages = html.Div("No recent pages", className="empty-state")
    except Exception:
        recent_pages = html.Div("No recent pages", className="empty-state")

    return floor_map, table, banner, clear_btn_style, count_badge, now_str, store_data, recent_pages


@callback(
    Output("selected-unit-store", "data"),
    [Input({"type": "unit-tile", "unit": ALL}, "n_clicks"),
     Input("clear-filter-btn", "n_clicks")],
    [State("selected-unit-store", "data")],
    prevent_initial_call=True,
)
def update_selected_unit(_tile_clicks, _clear_clicks, current):
    # The floor map tiles are rebuilt every refresh interval, which sets their
    # n_clicks back to 0 and fires this callback with value=0. We must ignore
    # those "phantom" triggers and only act on real user clicks (n_clicks > 0).
    if not ctx.triggered:
        return no_update

    import json as _json
    real_trigger = next((t for t in ctx.triggered if t.get("value")), None)
    if not real_trigger:
        return no_update

    prop_id = real_trigger["prop_id"]
    if prop_id.startswith("clear-filter-btn"):
        return None
    if "unit-tile" in prop_id:
        id_part = prop_id.rsplit(".", 1)[0]
        try:
            tile_id = _json.loads(id_part)
        except Exception:
            return no_update
        clicked = tile_id.get("unit")
        if clicked == current:
            return None
        return clicked
    return no_update


def _empty_mttr_fig(message):
    fig = go.Figure()
    fig.update_layout(
        height=280, margin=dict(l=40, r=20, t=12, b=40),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        annotations=[{"text": message, "xref": "paper", "yref": "paper",
                      "x": 0.5, "y": 0.5, "showarrow": False,
                      "font": {"color": TEXT_MUTED, "family": "Lato, sans-serif", "size": 13}}],
    )
    return fig


@callback(
    [Output("mttr-chart", "figure"),
     Output("mttr-title", "children")],
    [Input("interval-refresh", "n_intervals"),
     Input("selected-unit-store", "data")],
)
def update_mttr_chart(_n, selected_unit):
    base_title = "Mean Time To Resolve (MTTR) — by Unit · Last 30 Days"
    try:
        data = get_mttr_by_unit()
    except Exception as e:
        print(f"MTTR chart error: {e}")
        return _empty_mttr_fig("No MTTR data available"), base_title

    if not data:
        return _empty_mttr_fig("No MTTR data available"), base_title

    df = pd.DataFrame(data)
    df["unit_id"] = df["location_name"].apply(get_unit_from_location)

    title = base_title
    if selected_unit:
        unit_name = UNIT_NAME_BY_ID.get(selected_unit, selected_unit)
        title = f"Mean Time To Resolve (MTTR) — {unit_name} · Last 30 Days"
        df = df[df["unit_id"] == selected_unit]
        if df.empty:
            return _empty_mttr_fig(f"No resolved incidents for {unit_name} in the last 30 days"), title

    color_col = "location_name"
    fig = px.line(
        df, x="resolve_date", y="mttr_hours", color=color_col,
        labels={"resolve_date": "Date",
                "mttr_hours": "Mean Time To Resolve (hours)",
                "location_name": "Location"},
        template="plotly_white",
        color_discrete_sequence=[PURPLE, TEAL, GREEN, ORANGE_ALERT, RED_ALERT,
                                  "#6F42C1", "#0D6EFD", "#20C997", "#E83E8C", "#FD7E14"],
    )
    fig.update_layout(
        margin=dict(l=40, r=20, t=12, b=40), height=280,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, x=0.5, xanchor="center",
                    font=dict(size=10)),
        font=dict(family="Lato, system-ui, sans-serif"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    fig.update_traces(line=dict(width=2.5), mode="lines+markers", marker=dict(size=5))
    return fig, title


@callback(
    [Output("page-incident-select", "options"),
     Output("page-group-select", "options"),
     Output("page-unit-select", "value")],
    [Input("page-modal", "is_open")],
    [State("incidents-store", "data"),
     State("selected-unit-store", "data")],
)
def populate_page_form(is_open, incidents_data, selected_unit):
    if not is_open:
        return no_update, no_update, no_update
    inc_options = [{"label": f"{i['number']} - {i.get('short_description','')[:40]}",
                    "value": i["sys_id"]} for i in (incidents_data or [])]
    try:
        groups = get_assignment_groups()
        group_options = [{"label": g["name"], "value": g["name"]} for g in groups]
    except Exception:
        group_options = []
    return inc_options, group_options, (selected_unit or no_update)


@callback(
    Output("page-modal", "is_open"),
    [Input("open-page-modal", "n_clicks"),
     Input("close-page-modal", "n_clicks"),
     Input("submit-page", "n_clicks")],
    [State("page-modal", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal(_open, _close, _submit, _is_open):
    if ctx.triggered_id == "open-page-modal":
        return True
    return False


@callback(
    Output("page-feedback", "children"),
    [Input("submit-page", "n_clicks")],
    [State("page-incident-select", "value"), State("page-type-select", "value"),
     State("page-group-select", "value"), State("page-unit-select", "value"),
     State("page-message", "value"), State("incidents-store", "data")],
    prevent_initial_call=True,
)
def submit_page_action(_n_clicks, incident_id, page_type, group, unit, message, incidents_data):
    if not incident_id or not page_type or not group:
        return dbc.Alert("Please select an incident, page type, and target group.",
                         color="warning", className="mb-0 py-1")

    incident = next((i for i in (incidents_data or []) if i["sys_id"] == incident_id), None)
    if not incident:
        return dbc.Alert("Couldn't resolve the selected incident — refresh and retry.",
                         color="warning", className="mb-0 py-1")

    inc_number    = incident.get("number", "")
    clinical_tier = incident.get("clinical_impact_tier", "")

    # 1) Log the page action to the DB (existing behavior)
    db_msg = None
    try:
        execute_db(
            """INSERT INTO public.paged_actions
               (incident_sys_id, incident_number, paged_group, paged_by,
                page_type, clinical_impact_tier, affected_unit, message)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (incident_id, inc_number, group, "app_user", page_type, clinical_tier, unit, message),
        )
    except Exception as e:
        db_msg = f"DB log error: {str(e)[:100]}"

    # 2) Build templated email per user spec
    form = {
        "page_type":   page_type,
        "group":       group,
        "unit":        unit,
        "message":     message,
        "priority":    incident.get("priority"),
        "description": incident.get("short_description", ""),
    }
    subject, body, mailto_url = build_page_email(incident, form)

    # 3) Try real SMTP send; fall back to mailto link
    sent, detail = send_page_email(subject, body)

    parts = []
    if sent:
        parts.append(html.Div(
            [html.I(className="fas fa-check-circle me-2"),
             html.Strong("Email sent"), f" to {PAGE_EMAIL_TO}. Page also logged for {inc_number}."],
            style={"color": "#0F5132"}))
    else:
        parts.append(html.Div(
            [html.I(className="fas fa-envelope me-2"),
             html.Strong("Open in your mail client"),
             f" to send to {PAGE_EMAIL_TO}. ({detail}) Page logged for {inc_number}."],
            style={"color": TEXT_PRIMARY}))
        parts.append(html.Div(
            html.A([html.I(className="fas fa-paper-plane me-1"),
                    f"Send page email to {PAGE_EMAIL_TO}"],
                   href=mailto_url, target="_blank",
                   className="btn btn-sm",
                   style={"background": PURPLE, "color": "white", "marginTop": "6px",
                          "fontWeight": "700"}),
        ))
    if db_msg:
        parts.append(html.Div(db_msg, style={"color": ORANGE_ALERT, "fontSize": "0.78rem", "marginTop": "4px"}))

    # 4) Show the rendered email body so the user can review/copy
    parts.append(html.Details([
        html.Summary("Preview email body", style={"cursor": "pointer", "color": PURPLE,
                                                   "fontSize": "0.8rem", "marginTop": "6px"}),
        html.Pre(body, style={"background": "#F4F6F8", "border": f"1px solid {BORDER_SUBTLE}",
                              "borderRadius": "6px", "padding": "10px", "fontSize": "0.78rem",
                              "whiteSpace": "pre-wrap", "marginTop": "6px"}),
    ]))

    color = "success" if sent else "info"
    return dbc.Alert(parts, color=color, className="mb-0 py-2", style={"width": "100%"})


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("DATABRICKS_APP_PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
