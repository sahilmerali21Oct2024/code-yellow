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
import dash_cytoscape as cyto
import plotly.express as px
import plotly.graph_objects as go

cyto.load_extra_layouts()

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

# Databricks Job that sends the page email (Option 1 architecture)
PAGE_NOTIFIER_JOB_ID = os.getenv("PAGE_NOTIFIER_JOB_ID", "")
PAGE_SECRET_SCOPE = os.getenv("PAGE_SECRET_SCOPE", "code-yellow")

# Code Yellow - Dark Command Center palette (inspired by major-incident dashboards)
PURPLE = "#8B5CF6"        # brighter purple for dark bg
PURPLE_DARK = "#6D28D9"
PURPLE_LIGHT = "#312249"
TEAL = "#14B8A6"
TEAL_DARK = "#0E8F84"
GREEN = "#22C55E"
BLUE = "#3B82F6"
AMBER = "#F59E0B"
RED_ALERT = "#EF4444"
ORANGE_ALERT = "#F97316"
GRAY_INACTIVE = "#3F4F6E"

# Surfaces
BG_PAGE = "#0A1428"       # deep navy page background
BG_HEADER = "#0F1B33"
BG_PANEL = "#13213C"      # panel surface
BG_PANEL_HEAD = "#172847"
BG_TILE = "#1A2B4A"
BORDER_SUBTLE = "#23365E"
BORDER_BRIGHT = "#2E4978"

# Text
TEXT_PRIMARY = "#E6EEFB"
TEXT_SECONDARY = "#B7C3D6"
TEXT_MUTED = "#7C8DA8"
BG_CARD = BG_PANEL

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


def execute_db_returning(sql, params=None):
    """INSERT/UPDATE with RETURNING. Returns the first row as dict, or None."""
    conn = get_lakebase_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        conn.commit()
        return dict(row) if row else None
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
    """Return one row per (location, resolve_date, derived tier) so we can
    filter the MTTR chart by the same impact tier the rest of the app uses."""
    sql = """
        SELECT
            l.name AS location_name,
            DATE(i.resolved_at) AS resolve_date,
            i.priority, i.impact, i.u_patient_safety_impact,
            c.is_clinical,
            EXTRACT(EPOCH FROM (i.resolved_at - i.opened_at)) / 3600 AS mttr_hours
        FROM service_now.synced_incident i
        LEFT JOIN service_now.synced_cmn_location l ON i.location = l.sys_id
        LEFT JOIN service_now.synced_cmdb_ci      c ON i.cmdb_ci  = c.sys_id
        WHERE i.resolved_at IS NOT NULL
          AND i.resolved_at >= NOW() - INTERVAL '30 days'
          AND l.name IS NOT NULL
    """
    rows = query_db(sql)
    for r in rows:
        r["clinical_impact_tier"] = classify_clinical_impact(r)
    return rows


# ── DEPENDENCY BLAST RADIUS ──────────────────────────────────────────────────
# Healthcare topology rules — derive CI → CI edges from category/subcategory,
# since the ServiceNow synced CMDB doesn't include cmdb_rel_ci. These rules
# encode how a real hospital service map would look:
#   foundation: AD, SSO, network, storage, virt, integration engine
#   clinical apps depend on identity + integration + storage + network
#   imaging depends on PACS + storage + integration
#   pharmacy ADCs depend on Epic Willow + AD + integration
#   medical devices depend on Epic + integration + network
DEP_RULES = [
    # (predicate on source row, predicate on target row, weight)
    # Everything Epic depends on the SSO + integration engine + AD + storage.
    (lambda c: c["name"].startswith("Epic"),
     lambda c: c["name"] in ("Imprivata OneSign (SSO)", "Rhapsody Integration Engine",
                              "Active Directory (Corp)", "Pure Storage FlashArray",
                              "VMware vCenter (Prod)"), 0.9),
    # PACS depends on storage, integration, SSO, network
    (lambda c: c["subcategory"] == "Imaging",
     lambda c: c["name"] in ("Sectra VNA", "Pure Storage FlashArray",
                              "Rhapsody Integration Engine", "Imprivata OneSign (SSO)",
                              "Core Network Switch - Riverside"), 0.85),
    # LIS depends on Epic Beaker + integration
    (lambda c: c["subcategory"] == "Laboratory" and "Sunquest" in c["name"],
     lambda c: c["name"] in ("Epic Beaker Lab", "Rhapsody Integration Engine"), 0.8),
    # Pharmacy ADC depends on Epic Willow + AD + integration
    (lambda c: c["subcategory"] == "ADC",
     lambda c: c["name"] in ("Epic Willow Inpatient", "Active Directory (Corp)",
                              "Rhapsody Integration Engine"), 0.85),
    # Medical devices depend on Epic + integration + network
    (lambda c: c["category"] == "Medical Device",
     lambda c: c["name"] in ("Epic ClinDoc", "Epic Hyperspace - Production",
                              "Rhapsody Integration Engine",
                              "Meraki Wireless (Hospital WiFi)"), 0.75),
    # Voice / dictation depends on Epic + AD
    (lambda c: c["subcategory"] == "Voice Recognition",
     lambda c: c["name"] in ("Epic Hyperspace - Production", "Active Directory (Corp)"), 0.7),
    # Communications (Vocera, IP phones) depend on AD + network
    (lambda c: c["category"] == "Communications",
     lambda c: c["name"] in ("Active Directory (Corp)",
                              "Core Network Switch - Riverside",
                              "Meraki Wireless (Hospital WiFi)"), 0.6),
    # Collaboration depends on AD + Office 365 + network
    (lambda c: c["category"] == "Collaboration",
     lambda c: c["name"] in ("Active Directory (Corp)", "Office 365 (Exchange Online)",
                              "Palo Alto Firewall (Perimeter)"), 0.55),
    # HR / Productivity / Supply Chain / IT Tools depend on AD
    (lambda c: c["category"] in ("HR", "Productivity", "Supply Chain", "IT Tools", "Revenue Cycle"),
     lambda c: c["name"] == "Active Directory (Corp)", 0.6),
    # MyChart / patient SSO chain
    (lambda c: "MyChart" in c["name"],
     lambda c: c["name"] in ("Okta (Patient & Partner SSO)", "F5 Load Balancers",
                              "Palo Alto Firewall (Perimeter)"), 0.85),
    # VPN/Remote Access feeds Identity
    (lambda c: c["subcategory"] == "Remote Access",
     lambda c: c["name"] in ("Active Directory (Corp)", "Palo Alto Firewall (Perimeter)"), 0.7),
    # Network hierarchy
    (lambda c: c["category"] == "Network" and c["subcategory"] in ("Wireless", "Switch", "Load Balancer"),
     lambda c: c["name"] == "Palo Alto Firewall (Perimeter)", 0.65),
    # Backup / monitoring depends on storage + virt
    (lambda c: c["subcategory"] in ("Backup", "SIEM"),
     lambda c: c["name"] in ("Pure Storage FlashArray", "VMware vCenter (Prod)"), 0.55),
]


# Coarser groupings shown as chip filters in the UI.
DEP_CATEGORY_BUCKETS = {
    "ehr":           ("EHR",          lambda r: r["subcategory"] == "EHR"),
    "imaging":       ("Imaging",      lambda r: r["subcategory"] == "Imaging"),
    "identity":      ("Identity",     lambda r: r["category"] == "Identity"),
    "interfaces":    ("Interfaces",   lambda r: r["category"] == "Integration"),
    "devices":       ("Devices",      lambda r: r["category"] == "Medical Device"),
    "network":       ("Network",      lambda r: r["category"] == "Network"),
    "infrastructure":("Infrastructure", lambda r: r["category"] == "Infrastructure"),
    "pharmacy":      ("Pharmacy",     lambda r: r["category"] == "Pharmacy Automation"),
    "comm":          ("Comms",        lambda r: r["category"] in ("Communications", "Collaboration")),
    "business":      ("Business",     lambda r: r["category"] in ("HR", "Productivity", "Supply Chain",
                                                                    "IT Tools", "Revenue Cycle", "Security")),
}


def _ci_bucket(row):
    for bid, (_, pred) in DEP_CATEGORY_BUCKETS.items():
        if pred(row):
            return bid
    return "business"


def get_dependency_blast_radius():
    """Build the CMDB blast-radius graph: nodes (each CI w/ incident pressure
    + risk score) and edges (derived from healthcare topology rules)."""
    ci_sql = """
        SELECT c.sys_id, c.name, c.ci_class, c.category, c.subcategory,
               c.business_criticality, c.service_tier, c.is_clinical,
               c.support_group,
               COALESCE(SUM(CASE WHEN i.active THEN 1 ELSE 0 END), 0)                AS incident_count,
               COALESCE(SUM(CASE WHEN i.active AND i.priority = 1 THEN 1 ELSE 0 END), 0) AS p1_count,
               COALESCE(SUM(CASE WHEN i.active AND i.priority IN (1,2) THEN 1 ELSE 0 END), 0) AS major_count,
               COALESCE(SUM(CASE WHEN i.active AND i.u_patient_safety_impact THEN 1 ELSE 0 END), 0) AS patient_safety
        FROM service_now.synced_cmdb_ci c
        LEFT JOIN service_now.synced_incident i ON i.cmdb_ci = c.sys_id
        GROUP BY c.sys_id, c.name, c.ci_class, c.category, c.subcategory,
                 c.business_criticality, c.service_tier, c.is_clinical, c.support_group
        ORDER BY c.name
    """
    rows = query_db(ci_sql)

    def risk(r):
        if r["patient_safety"] > 0: return 100
        if r["p1_count"] > 0:       return 85
        if r["incident_count"] >= 10: return 65
        if r["incident_count"] > 0: return 35
        return 5

    def status(score):
        if score >= 80: return "critical"
        if score >= 35: return "warning"
        return "healthy"

    nodes = []
    by_name = {r["name"]: r for r in rows}
    for r in rows:
        r["risk_score"] = risk(r)
        r["status"] = status(r["risk_score"])
        r["bucket"] = _ci_bucket(r)
        crit_n = 1 if (r["business_criticality"] or "").startswith("1") else (
            2 if (r["business_criticality"] or "").startswith("2") else 3
        )
        nodes.append({
            "data": {
                "id": r["sys_id"],
                "name": r["name"],
                "label": r["name"],
                "category": r["category"] or "Other",
                "subcategory": r["subcategory"] or "",
                "bucket": r["bucket"],
                "criticality": crit_n,
                "service_tier": r["service_tier"] or "",
                "is_clinical": bool(r["is_clinical"]),
                "incident_count": int(r["incident_count"]),
                "p1_count": int(r["p1_count"]),
                "major_count": int(r["major_count"]),
                "patient_safety": int(r["patient_safety"]),
                "risk_score": int(r["risk_score"]),
                "status": r["status"],
                "has_p1": bool(r["p1_count"] > 0 or r["patient_safety"] > 0),
            },
            "classes": f"node-{r['status']} bucket-{r['bucket']}"
                       + (" has-p1" if r["p1_count"] > 0 or r["patient_safety"] > 0 else ""),
        })

    edges = []
    seen = set()
    for src in rows:
        for src_pred, tgt_pred, weight in DEP_RULES:
            try:
                if not src_pred(src): continue
            except Exception: continue
            for tgt in rows:
                if tgt["sys_id"] == src["sys_id"]: continue
                try:
                    if not tgt_pred(tgt): continue
                except Exception: continue
                key = (src["sys_id"], tgt["sys_id"])
                if key in seen: continue
                seen.add(key)
                shared = min(src["incident_count"], tgt["incident_count"])
                edges.append({
                    "data": {
                        "id": f"{src['sys_id']}__{tgt['sys_id']}",
                        "source": src["sys_id"],
                        "target": tgt["sys_id"],
                        "weight": weight,
                        "shared_incidents": int(shared),
                        "thickness": 1.0 + weight * 2.0 + min(shared, 6) * 0.4,
                    },
                })

    return nodes, edges


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
    """In-process SMTP send. Used only as a fallback when the Job route is
    unavailable. Returns (sent: bool, detail: str)."""
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD:
        return False, "Local SMTP not configured."

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


def trigger_page_notifier_job(incident, form, lakebase_row):
    """Fire the Databricks Job that sends the email and syncs to UC.
    Returns (run_id, detail)."""
    if not PAGE_NOTIFIER_JOB_ID:
        return None, "PAGE_NOTIFIER_JOB_ID not configured on the app."
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        unit_id = form.get("unit") or ""
        unit_label = UNIT_NAME_BY_ID.get(unit_id, unit_id) if unit_id else ""
        created_at = lakebase_row.get("created_at") if lakebase_row else None
        created_at_iso = created_at.isoformat() if created_at else ""
        params = {
            "inc_number":     str(incident.get("number") or ""),
            "priority":       str(incident.get("priority") or ""),
            "group":          str(form.get("group") or ""),
            "unit":           str(unit_id),
            "unit_label":     str(unit_label),
            "page_type":      str(form.get("page_type") or "page_team"),
            "message":        str(form.get("message") or ""),
            "page_to":        PAGE_EMAIL_TO,
            "secret_scope":   PAGE_SECRET_SCOPE,
            # UC sync params (the same row we just inserted into Lakebase)
            "action_id":      str(lakebase_row.get("action_id") or "") if lakebase_row else "",
            "incident_sys_id": str(incident.get("sys_id") or ""),
            "paged_by":       "app_user",
            "clinical_impact_tier": str(incident.get("clinical_impact_tier") or ""),
            "created_at_iso": created_at_iso,
            "uc_target":      "sahil_merali.service_now.paged_actions",
        }
        run = w.jobs.run_now(job_id=int(PAGE_NOTIFIER_JOB_ID), job_parameters=params)
        return run.run_id, f"Job triggered (run_id={run.run_id})."
    except Exception as e:
        return None, f"Job trigger error: {str(e)[:200]}"


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
        bg = TIER_COLORS.get(tier, GRAY_INACTIVE) if tier else BG_TILE
        text_color = "white" if tier else TEXT_PRIMARY
        is_pulse = tier == "life-safety"
        is_selected = selected_unit == uid

        tile_class = "floor-tile"
        if info:
            tile_class += " floor-tile--has-incidents"
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
  --c-purple: {PURPLE};
  --c-purple-dark: {PURPLE_DARK};
  --c-purple-light: {PURPLE_LIGHT};
  --c-teal: {TEAL};
  --c-green: {GREEN};
  --c-blue: {BLUE};
  --c-amber: {AMBER};
  --bg-page: {BG_PAGE};
  --bg-header: {BG_HEADER};
  --bg-panel: {BG_PANEL};
  --bg-panel-head: {BG_PANEL_HEAD};
  --bg-tile: {BG_TILE};
  --text-primary: {TEXT_PRIMARY};
  --text-secondary: {TEXT_SECONDARY};
  --text-muted: {TEXT_MUTED};
  --border-subtle: {BORDER_SUBTLE};
  --border-bright: {BORDER_BRIGHT};
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
  background: linear-gradient(180deg, {BG_HEADER} 0%, {BG_PAGE} 100%);
  border-bottom: 1px solid var(--border-subtle);
  padding: 14px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}}
.app-header__brand {{ display: flex; align-items: center; gap: 14px; }}
.app-header__numbox {{
  width: 38px; height: 38px; border-radius: 8px;
  background: var(--c-purple);
  display: inline-flex; align-items: center; justify-content: center;
  color: white; font-size: 1.15rem; font-weight: 900;
  box-shadow: 0 0 0 1px rgba(139,92,246,0.4);
}}
.app-header__title {{
  font-size: 1.15rem; font-weight: 700;
  letter-spacing: 0.01em; color: var(--text-primary); line-height: 1.1;
}}
.app-header__subtitle {{
  font-size: 0.78rem; font-weight: 400; color: var(--text-muted); margin-top: 3px;
}}
.app-header__meta {{ display: flex; align-items: center; gap: 16px; }}
.app-header__time {{
  font-size: 0.78rem; color: var(--text-muted);
  padding: 4px 10px; border: 1px solid var(--border-subtle);
  border-radius: 6px; background: var(--bg-panel);
}}

/* ── LEGEND ───────────────────────────────────────────── */
.legend-bar {{
  background: var(--bg-header);
  padding: 8px 24px;
  border-bottom: 1px solid var(--border-subtle);
  display: flex; flex-wrap: wrap; gap: 18px; align-items: center;
  font-size: 0.78rem; color: var(--text-secondary);
}}
.legend-item {{ display: inline-flex; align-items: center; gap: 6px; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}

/* ── CARDS ────────────────────────────────────────────── */
.page-body {{ padding: 18px 24px; background: var(--bg-page); }}
.panel {{
  background: var(--bg-panel);
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  overflow: hidden;
  height: 100%;
  box-shadow: 0 1px 0 rgba(255,255,255,0.02) inset, 0 4px 16px rgba(0,0,0,0.25);
}}
.panel__header {{
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-subtle);
  display: flex; align-items: center; justify-content: space-between;
  background: var(--bg-panel-head);
}}
.panel__title {{
  font-size: 0.82rem; font-weight: 700; color: var(--text-primary);
  margin: 0; letter-spacing: 0.06em; text-transform: uppercase;
}}
.panel__body {{ padding: 14px 16px; }}

/* ── FLOOR MAP TILES ─────────────────────────────────── */
.floor-grid {{
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 6px;
  padding: 2px;
}}
.floor-tile {{
  position: relative;
  font-family: inherit;
  border: 1px solid var(--border-bright);
  border-radius: 6px;
  padding: 8px 6px;
  min-height: 52px;
  display: flex; flex-direction: column; justify-content: center;
  text-align: center;
  cursor: pointer;
  background: var(--bg-tile);
  color: var(--text-primary);
  transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease, border-color 120ms ease;
  outline: none;
}}
.floor-tile:hover {{
  transform: translateY(-1px);
  border-color: var(--c-purple);
  box-shadow: 0 6px 14px rgba(139,92,246,0.25);
}}
.floor-tile:focus-visible {{ box-shadow: 0 0 0 2px rgba(139,92,246,0.55); }}
.floor-tile--selected {{
  box-shadow: 0 0 0 2px var(--c-purple), 0 6px 14px rgba(139,92,246,0.30);
}}
.floor-tile__name {{
  font-size: 0.78rem; font-weight: 800; letter-spacing: 0.04em;
}}
.floor-tile__meta {{
  font-size: 0.62rem; font-weight: 400; opacity: 0.85; margin-top: 2px;
  color: var(--text-secondary);
}}
.floor-tile--has-incidents .floor-tile__meta {{ color: rgba(255,255,255,0.95); }}
.floor-tile__badge {{
  position: absolute; top: 3px; right: 4px;
  font-size: 0.6rem; font-weight: 900;
  padding: 1px 5px; border-radius: 999px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.4);
}}
@keyframes pulseTile {{
  0%, 100% {{ box-shadow: 0 0 0 0 rgba(239,68,68,0.55); }}
  50%      {{ box-shadow: 0 0 0 10px rgba(239,68,68,0); }}
}}
.floor-tile--pulse {{ animation: pulseTile 1.6s ease-in-out infinite; }}

/* ── BADGES ──────────────────────────────────────────── */
.tier-pill {{
  display: inline-block;
  padding: 2px 8px; border-radius: 4px;
  font-size: 0.68rem; font-weight: 800; letter-spacing: 0.05em;
  text-transform: uppercase; color: white;
}}
.tier-pill--life-safety   {{ background: {RED_ALERT}; }}
.tier-pill--care-delivery {{ background: {ORANGE_ALERT}; }}
.tier-pill--administrative {{ background: {GREEN}; }}

/* ── TIER FILTER CHIPS (toggle row above the incidents list) ── */
.tier-chip-bar {{
  display: flex; flex-wrap: wrap; gap: 6px;
  padding: 8px 2px 10px;
  border-bottom: 1px solid var(--border-subtle);
  margin-bottom: 6px;
  align-items: center;
}}
.tier-chip-bar__label {{
  font-size: 0.68rem; font-weight: 800; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--text-muted); margin-right: 4px;
}}
.tier-chip {{
  border: 1px solid var(--border-bright);
  background: var(--bg-tile);
  color: var(--text-secondary);
  padding: 3px 10px 3px 8px; border-radius: 999px;
  font-size: 0.74rem; font-weight: 700;
  display: inline-flex; align-items: center; gap: 6px;
  cursor: pointer; transition: all 120ms ease;
  font-family: inherit;
}}
.tier-chip:hover {{ border-color: var(--c-purple); color: var(--text-primary); }}
.tier-chip__dot {{
  width: 8px; height: 8px; border-radius: 50%; display: inline-block;
}}
.tier-chip__count {{
  margin-left: 4px; padding: 0 6px; border-radius: 999px;
  background: rgba(255,255,255,0.06); color: var(--text-muted);
  font-size: 0.68rem; font-weight: 800;
}}
.tier-chip--active {{
  color: white; border-color: transparent;
}}
.tier-chip--active .tier-chip__count {{
  background: rgba(255,255,255,0.22); color: white;
}}
.tier-chip--life-safety.tier-chip--active    {{ background: {RED_ALERT}; }}
.tier-chip--care-delivery.tier-chip--active  {{ background: {ORANGE_ALERT}; }}
.tier-chip--administrative.tier-chip--active {{ background: {GREEN}; }}

/* ── BLAST-RADIUS PANEL ─────────────────────────────────── */
.blast-toolbar {{
  display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
  padding: 4px 2px 10px; border-bottom: 1px solid var(--border-subtle);
  margin-bottom: 8px;
}}
.blast-toolbar__label {{
  font-size: 0.68rem; font-weight: 800; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--text-muted);
}}
.blast-toolbar .Select-control {{ min-height: 30px; }}
.blast-anchor-wrap {{ min-width: 260px; flex: 0 0 280px; }}
.blast-chips {{ display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }}
.cat-chip {{
  border: 1px solid var(--border-bright);
  background: var(--bg-tile); color: var(--text-secondary);
  padding: 2px 8px; border-radius: 999px;
  font-size: 0.7rem; font-weight: 700;
  cursor: pointer; font-family: inherit;
  transition: all 120ms ease;
}}
.cat-chip:hover {{ border-color: var(--c-purple); color: var(--text-primary); }}
.cat-chip--active {{
  background: var(--c-purple); color: white; border-color: transparent;
}}
.risk-toggle {{ display: inline-flex; gap: 0; border: 1px solid var(--border-bright);
                border-radius: 6px; overflow: hidden; }}
.risk-toggle__btn {{
  background: var(--bg-tile); color: var(--text-secondary);
  border: 0; padding: 4px 10px; font-size: 0.72rem; font-weight: 700;
  cursor: pointer; font-family: inherit;
}}
.risk-toggle__btn--active {{ background: var(--c-purple); color: white; }}
.blast-legend {{ display: flex; gap: 12px; font-size: 0.72rem; color: var(--text-muted); }}
.blast-legend__sw {{
  display: inline-block; width: 10px; height: 10px; border-radius: 50%;
  margin-right: 4px; vertical-align: middle;
}}
.blast-graph-wrap {{
  position: relative;
  border: 1px solid var(--border-subtle); border-radius: 8px;
  background: radial-gradient(circle at 50% 45%, rgba(124,92,255,0.10) 0%, rgba(11,18,41,0) 60%),
              var(--bg-tile);
}}
.blast-inspector {{
  position: absolute; bottom: 10px; right: 10px; max-width: 290px;
  background: rgba(11,18,41,0.92); border: 1px solid var(--border-bright);
  border-radius: 8px; padding: 10px 12px; font-size: 0.78rem;
  color: var(--text-primary); box-shadow: 0 6px 18px rgba(0,0,0,0.45);
  backdrop-filter: blur(6px);
}}
.blast-inspector__title {{ font-weight: 800; color: var(--c-teal); margin-bottom: 4px; }}
.blast-inspector__row  {{ display: flex; justify-content: space-between; gap: 12px;
                          padding: 1px 0; color: var(--text-secondary); }}
.blast-inspector__row b {{ color: var(--text-primary); }}
.blast-inspector__pill {{
  display: inline-block; padding: 1px 7px; border-radius: 999px;
  font-size: 0.66rem; font-weight: 800; text-transform: uppercase;
  letter-spacing: 0.05em;
}}
.blast-inspector__pill--critical {{ background: {RED_ALERT}; color: white; }}
.blast-inspector__pill--warning  {{ background: {ORANGE_ALERT}; color: white; }}
.blast-inspector__pill--healthy  {{ background: {GREEN}; color: white; }}
.blast-empty {{ position: absolute; inset: 0; display: flex; align-items: center;
                justify-content: center; color: var(--text-muted); font-size: 0.85rem; }}

.count-pill {{
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--c-purple); color: white;
  padding: 5px 12px; border-radius: 6px;
  font-size: 0.78rem; font-weight: 800; letter-spacing: 0.04em;
  text-transform: uppercase;
}}
.count-pill--alert {{ background: {RED_ALERT}; }}
.count-pill--warn  {{ background: {ORANGE_ALERT}; }}
.count-pill--ok    {{ background: var(--c-green); }}

/* ── FILTER BANNER ───────────────────────────────────── */
.filter-banner {{
  display: flex; align-items: center; justify-content: space-between;
  background: var(--c-purple-light);
  border: 1px solid var(--c-purple);
  color: var(--text-primary);
  padding: 8px 12px; border-radius: 6px;
  margin: 0 0 12px 0;
  font-size: 0.82rem;
}}
.filter-banner__clear {{
  background: transparent; border: 1px solid var(--c-purple);
  color: var(--text-primary); font-weight: 700; font-size: 0.74rem;
  padding: 4px 10px; border-radius: 5px; cursor: pointer;
}}
.filter-banner__clear:hover {{ background: var(--c-purple); color: white; }}

/* ── INCIDENT CARDS (expandable) ─────────────────────── */
.inc-list {{ display: flex; flex-direction: column; gap: 6px; }}
.inc-list__header {{
  display: grid;
  grid-template-columns: 18px 100px 105px 1fr 76px 40px 64px;
  align-items: center;
  gap: 10px;
  padding: 6px 12px;
  font-size: 0.68rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border-subtle);
  margin-bottom: 2px;
}}
.inc-list__header span:nth-child(5) {{ text-align: center; }}
.inc-list__header span:nth-child(6) {{ text-align: center; }}
.inc-list__header span:nth-child(7) {{ text-align: right; }}
.inc-card {{
  border: 1px solid var(--border-subtle);
  border-radius: 6px;
  background: var(--bg-tile);
  transition: box-shadow 120ms ease, border-color 120ms ease;
  overflow: hidden;
}}
.inc-card:hover {{
  border-color: var(--c-purple);
}}
.inc-card[open] {{
  border-color: var(--c-purple);
  box-shadow: 0 4px 14px rgba(139,92,246,0.18);
}}
.inc-card__summary {{
  list-style: none;
  cursor: pointer;
  padding: 9px 12px;
  display: grid;
  grid-template-columns: 18px 100px 105px 1fr 76px 40px 64px;
  align-items: center;
  gap: 10px;
  font-size: 0.82rem;
  user-select: none;
  color: var(--text-primary);
}}
.inc-card__unit {{
  font-weight: 700;
  color: var(--c-teal);
  text-align: center;
  font-size: 0.74rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  background: rgba(20,184,166,0.12);
  padding: 2px 6px;
  border-radius: 4px;
}}
.inc-card__summary::-webkit-details-marker {{ display: none; }}
.inc-card__chevron {{
  display: inline-block;
  color: var(--c-purple);
  font-size: 0.9rem;
  transition: transform 150ms ease;
}}
.inc-card[open] .inc-card__chevron {{ transform: rotate(90deg); }}
.inc-card__num  {{ font-weight: 800; color: var(--c-teal); }}
.inc-card__pill {{ justify-self: start; }}
.inc-card__desc {{
  color: var(--text-primary);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.inc-card__pri  {{ font-weight: 800; color: var(--c-amber); text-align: center; }}
.inc-card__time {{ color: var(--text-muted); font-size: 0.78rem; text-align: right; }}

.inc-detail__grid {{
  border-top: 1px dashed var(--border-subtle);
  background: rgba(0,0,0,0.18);
  padding: 12px 14px;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  column-gap: 18px;
  row-gap: 4px;
  font-size: 0.8rem;
}}
.inc-detail__row {{
  display: grid;
  grid-template-columns: 150px 1fr;
  gap: 8px;
  padding: 3px 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  align-items: start;
}}
.inc-detail__label {{
  color: var(--text-muted);
  font-size: 0.7rem;
  letter-spacing: 0.05em;
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
  .inc-card__pill, .inc-card__unit, .inc-card__pri, .inc-card__time {{ display: none; }}
}}

.empty-state {{
  text-align: center; padding: 36px 12px; color: var(--text-muted); font-size: 0.88rem;
}}

/* ── BUTTONS ─────────────────────────────────────────── */
.btn-primary-ach {{
  background: var(--c-purple); border-color: var(--c-purple);
  font-weight: 700; color: white;
}}
.btn-primary-ach:hover {{ background: var(--c-purple-dark); border-color: var(--c-purple-dark); }}

/* ── PAGES FEED ──────────────────────────────────────── */
.page-feed__item {{
  padding: 10px 0; border-bottom: 1px solid var(--border-subtle);
}}
.page-feed__item:last-child {{ border-bottom: none; }}
.page-feed__title {{ font-weight: 800; color: var(--c-teal); font-size: 0.86rem; }}
.page-feed__meta  {{ color: var(--text-muted); font-size: 0.76rem; margin-top: 2px; }}

/* ── MODAL OVERRIDES (dark theme) ────────────────────── */
.modal-content {{ background: var(--bg-panel); color: var(--text-primary); border: 1px solid var(--border-subtle); }}
.modal-header, .modal-footer {{ border-color: var(--border-subtle); }}
.modal-title {{ color: var(--text-primary); }}
.form-label {{ color: var(--text-secondary); font-weight: 700; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }}
.form-control, .form-select, .Select-control, textarea.form-control {{
  background: var(--bg-tile) !important; color: var(--text-primary) !important;
  border-color: var(--border-bright) !important;
}}
.form-control::placeholder, textarea.form-control::placeholder {{ color: var(--text-muted); }}
/* react-select (dcc.Dropdown) — force light text on dark tile so values are readable */
.Select-control, .Select-control .Select-value, .Select-control .Select-value-label,
.Select-control .Select-input > input, .Select-control .Select-input input,
.Select--single > .Select-control .Select-value, .Select-placeholder,
.has-value.Select--single > .Select-control .Select-value .Select-value-label,
.has-value.is-pseudo-focused.Select--single > .Select-control .Select-value .Select-value-label {{
  color: var(--text-primary) !important;
}}
.Select-placeholder {{ color: var(--text-muted) !important; }}
.Select-menu-outer {{
  background: var(--bg-panel) !important; color: var(--text-primary) !important;
  border: 1px solid var(--border-bright) !important;
}}
.Select-option {{ background: var(--bg-panel) !important; color: var(--text-primary) !important; }}
.Select-option.is-focused, .VirtualizedSelectFocusedOption {{
  background: rgba(124,92,255,0.18) !important; color: #FFFFFF !important;
}}
.Select-option.is-selected {{ background: rgba(124,92,255,0.32) !important; color: #FFFFFF !important; }}
.Select-arrow {{ border-color: var(--text-secondary) transparent transparent !important; }}
.Select-clear {{ color: var(--text-secondary) !important; }}
.Select.is-focused:not(.is-open) > .Select-control {{
  border-color: var(--c-purple) !important;
  box-shadow: 0 0 0 2px rgba(124,92,255,0.25) !important;
}}
.btn-close {{ filter: invert(1) opacity(0.7); }}

/* Plotly chart dark tweaks */
.js-plotly-plot .plotly .modebar {{ display: none !important; }}

@media (max-width: 900px) {{
  .floor-grid {{ grid-template-columns: repeat(2, 1fr); }}
  .page-body {{ padding: 14px; }}
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


HOSPITAL_NAME = "Akron Children's Hospital"


def header():
    return html.Div(
        [
            html.Div([
                html.Div("1", className="app-header__numbox"),
                html.Div([
                    html.Div("Code Yellow — Clinical Impact Command Center",
                             className="app-header__title"),
                    html.Div(f"{HOSPITAL_NAME}  ·  Real-time coordination for critical incidents",
                             className="app-header__subtitle"),
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


BLAST_CYTOSCAPE_STYLESHEET = [
    # Defaults
    {"selector": "node", "style": {
        "label": "data(label)",
        "color": "#E6EEFB", "font-size": "10px", "font-family": "Lato, sans-serif",
        "text-valign": "bottom", "text-halign": "center", "text-margin-y": 6,
        "text-outline-color": "#0B1229", "text-outline-width": 2,
        "background-color": "#7C5CFF",
        "width":  "mapData(criticality, 1, 3, 44, 22)",
        "height": "mapData(criticality, 1, 3, 44, 22)",
        "border-width": 2, "border-color": "rgba(255,255,255,0.25)",
        "transition-property": "background-color, border-color, opacity, width, height",
        "transition-duration": "180ms",
    }},
    # Risk status colors
    {"selector": "node.node-critical", "style": {
        "background-color": "#EF4444", "border-color": "#FCA5A5",
    }},
    {"selector": "node.node-warning", "style": {
        "background-color": "#F97316", "border-color": "#FED7AA",
    }},
    {"selector": "node.node-healthy", "style": {
        "background-color": "#22C55E", "border-color": "rgba(255,255,255,0.35)",
    }},
    # Anchor node — larger + pulsing purple ring
    {"selector": "node.anchor", "style": {
        "border-width": 6, "border-color": "#7C5CFF",
        "width":  "mapData(criticality, 1, 3, 64, 40)",
        "height": "mapData(criticality, 1, 3, 64, 40)",
        "font-size": "12px",
        "color": "#FFFFFF",
    }},
    # P1 / patient-safety halo
    {"selector": "node.has-p1", "style": {
        "shadow-blur": 18, "shadow-color": "#EF4444", "shadow-opacity": 0.9,
    }},
    # Dim faded nodes when hovering / focusing
    {"selector": "node.faded", "style": {"opacity": 0.12}},
    {"selector": "edge.faded", "style": {"opacity": 0.04}},
    # 1-hop emphasis
    {"selector": "node.hop1", "style": {
        "border-width": 4, "border-color": "#14B8A6",
    }},
    {"selector": "node.hop2", "style": {
        "border-width": 3, "border-color": "rgba(20,184,166,0.55)",
    }},
    # Edges
    {"selector": "edge", "style": {
        "curve-style": "bezier",
        "width": "data(thickness)",
        "line-color": "rgba(124,92,255,0.35)",
        "target-arrow-color": "rgba(124,92,255,0.55)",
        "target-arrow-shape": "triangle",
        "arrow-scale": 0.8,
        "opacity": 0.85,
        "transition-property": "line-color, opacity, width",
        "transition-duration": "180ms",
    }},
    {"selector": "edge.hop1", "style": {
        "line-color": "#14B8A6", "target-arrow-color": "#14B8A6", "opacity": 1.0,
    }},
    {"selector": "edge.hop2", "style": {
        "line-color": "rgba(20,184,166,0.45)", "target-arrow-color": "rgba(20,184,166,0.55)",
    }},
]


app.layout = html.Div([
    dcc.Interval(id="interval-refresh", interval=30_000, n_intervals=0),
    dcc.Store(id="incidents-store"),
    dcc.Store(id="selected-unit-store", data=None),
    dcc.Store(id="selected-tier-store", data=None),
    dcc.Store(id="blast-anchor-store", data=None),
    dcc.Store(id="blast-categories-store", data=[]),
    dcc.Store(id="blast-risk-mode-store", data="incident_load"),

    header(),
    legend(),

    html.Div([
        # Row 1: compact floor map + wide active incidents
        dbc.Row([
            dbc.Col(html.Div([
                html.Div([
                    html.H6("Floor Map", className="panel__title"),
                    html.Span("Click to filter", style={"fontSize": "0.72rem", "color": TEXT_MUTED}),
                ], className="panel__header"),
                html.Div(html.Div(id="floor-map-container"), className="panel__body"),
            ], className="panel"), md=3, className="mb-3"),

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
                    html.Div(id="tier-filter-bar", className="tier-chip-bar"),
                    html.Div(id="filter-banner-container"),
                    html.Div(id="incidents-table-container", style={"maxHeight": "360px", "overflowY": "auto"}),
                ], className="panel__body"),
            ], className="panel"), md=9, className="mb-3"),
        ]),

        # Row 2: MTTR + recent pages
        dbc.Row([
            dbc.Col(html.Div([
                html.Div(html.H6("Mean Time To Resolve (MTTR) — by Floor Unit · Last 30 Days",
                                 id="mttr-title", className="panel__title"),
                         className="panel__header"),
                html.Div(dcc.Graph(id="mttr-chart", config={"displayModeBar": False}), className="panel__body"),
            ], className="panel"), md=8, className="mb-3"),

            dbc.Col(html.Div([
                html.Div(html.H6("Recent Page Actions", className="panel__title"), className="panel__header"),
                html.Div(html.Div(id="recent-pages-container", style={"maxHeight": "300px", "overflowY": "auto"}),
                         className="panel__body"),
            ], className="panel"), md=4, className="mb-3"),
        ]),

        # Row 3: hero blast-radius graph (full width, at the bottom)
        dbc.Row([
            dbc.Col(html.Div([
                html.Div([
                    html.H6("System Blast Radius — CMDB Dependency Graph",
                            className="panel__title"),
                    html.Span("Hover dims · click to recenter · double-click to clear",
                              style={"fontSize": "0.72rem", "color": TEXT_MUTED}),
                ], className="panel__header"),
                html.Div([
                    html.Div([
                        html.Span("Anchor", className="blast-toolbar__label"),
                        html.Div(dcc.Dropdown(id="blast-anchor-dropdown",
                                              placeholder="Center on a system...",
                                              clearable=True),
                                 className="blast-anchor-wrap"),
                        html.Span("Category", className="blast-toolbar__label"),
                        html.Div(id="blast-category-chips", className="blast-chips"),
                        html.Span("Risk mode", className="blast-toolbar__label"),
                        html.Div([
                            html.Button("Incident load",   id={"type":"risk-mode","mode":"incident_load"},
                                        n_clicks=0, className="risk-toggle__btn risk-toggle__btn--active"),
                            html.Button("Patient safety",  id={"type":"risk-mode","mode":"patient_safety"},
                                        n_clicks=0, className="risk-toggle__btn"),
                            html.Button("P1 / major",      id={"type":"risk-mode","mode":"p1_major"},
                                        n_clicks=0, className="risk-toggle__btn"),
                        ], className="risk-toggle"),
                        html.Div([
                            html.Span([html.Span(className="blast-legend__sw",
                                                 style={"background": RED_ALERT}), "Critical"]),
                            html.Span([html.Span(className="blast-legend__sw",
                                                 style={"background": ORANGE_ALERT}), "Warning"]),
                            html.Span([html.Span(className="blast-legend__sw",
                                                 style={"background": GREEN}), "Healthy"]),
                        ], className="blast-legend", style={"marginLeft":"auto"}),
                    ], className="blast-toolbar"),

                    html.Div([
                        cyto.Cytoscape(
                            id="blast-cyto",
                            elements=[],
                            stylesheet=BLAST_CYTOSCAPE_STYLESHEET,
                            layout={"name": "cose-bilkent", "animate": False,
                                    "idealEdgeLength": 110, "nodeRepulsion": 8500,
                                    "edgeElasticity": 0.45, "gravity": 0.25,
                                    "numIter": 2500, "padding": 30,
                                    "randomize": False, "fit": True},
                            style={"width": "100%", "height": "560px"},
                            minZoom=0.3, maxZoom=2.2,
                            wheelSensitivity=0.18,
                            autoungrabify=False,
                        ),
                        html.Div(id="blast-inspector", className="blast-inspector",
                                 children="Click a node for details."),
                    ], className="blast-graph-wrap"),
                ], className="panel__body"),
            ], className="panel"), md=12, className="mb-3"),
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
    unit_name = UNIT_NAME_BY_ID.get(inc.get("affected_unit")) if inc.get("affected_unit") else "Other"
    detail_rows = [
        ("Incident #",          _fmt(inc.get("number"))),
        ("Short description",   _fmt(inc.get("short_description"))),
        ("Clinical impact",     tier.replace("-", " ").title()),
        ("Priority",            f"P{_fmt(inc.get('priority'))}"),
        ("Impact",              _fmt(inc.get("impact"))),
        ("Urgency",             _fmt(inc.get("urgency"))),
        ("Category",            _fmt(inc.get("category"))),
        ("Assignment group",    _fmt(inc.get("assignment_group"))),
        ("Floor unit",          unit_name),
        ("CI name",             _fmt(inc.get("ci_name"))),
        ("Clinical CI",         _fmt(inc.get("is_clinical"))),
        ("Service tier",        _fmt(inc.get("service_tier"))),
        ("Patient-safety flag", _fmt(inc.get("u_patient_safety_impact"))),
        ("Clinical-impact tag", _fmt(inc.get("u_clinical_impact"))),
        ("Opened",              _fmt_dt(inc.get("opened_at"))),
        ("Time open",           _fmt(inc.get("time_open"))),
        ("Raw location",        _fmt(inc.get("location_name"))),
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
        html.Span(unit_name, className="inc-card__unit"),
        html.Span(f"P{_fmt(inc.get('priority'))}", className="inc-card__pri"),
        html.Span(inc.get("time_open") or "—", className="inc-card__time"),
    ], className="inc-card__summary")
    return html.Details([summary, details], className="inc-card")


def render_incidents_list(incidents):
    if not incidents:
        return html.Div("No active incidents.", className="empty-state")
    header = html.Div([
        html.Span(""),            # chevron column
        html.Span("Incident #"),
        html.Span("Impact tier"),
        html.Span("Description"),
        html.Span("Unit"),
        html.Span("Pri"),
        html.Span("Open"),
    ], className="inc-list__header")
    return html.Div(
        [header] + [render_incident_card(i) for i in incidents],
        className="inc-list",
    )


TIER_ORDER = [
    ("life-safety",   "Life-Safety"),
    ("care-delivery", "Care-Delivery"),
    ("administrative","Administrative"),
]


def render_tier_chip_bar(tier_counts, selected_tier):
    """Render a strip of tier-filter chips with live counts. Returns a list of
    children for the tier-filter-bar Div."""
    chips = [html.Span("Filter:", className="tier-chip-bar__label")]
    for tier_id, label in TIER_ORDER:
        cnt = tier_counts.get(tier_id, 0)
        cls = f"tier-chip tier-chip--{tier_id}"
        if selected_tier == tier_id:
            cls += " tier-chip--active"
        chips.append(html.Button(
            [
                html.Span(className=f"tier-chip__dot tier-pill--{tier_id}"),
                html.Span(label),
                html.Span(str(cnt), className="tier-chip__count"),
            ],
            id={"type": "tier-chip", "tier": tier_id},
            n_clicks=0,
            className=cls,
            title=f"Click to show only {label} incidents — click again to clear.",
        ))
    return chips


def render_filter_banner(selected_unit, selected_tier, total_visible, total_overall):
    """Returns (banner_children, clear_btn_style)."""
    if not selected_unit and not selected_tier:
        return None, {"display": "none"}
    chips = []
    if selected_unit:
        chips.append(html.Span([
            html.I(className="fas fa-map-marker-alt me-1"),
            "Unit: ", html.Strong(UNIT_NAME_BY_ID.get(selected_unit, selected_unit)),
        ], style={"marginRight": "10px"}))
    if selected_tier:
        tier_label = dict(TIER_ORDER).get(selected_tier, selected_tier)
        chips.append(html.Span([
            html.I(className="fas fa-bolt me-1"),
            "Impact: ", html.Strong(tier_label),
        ], style={"marginRight": "10px"}))
    banner = html.Div([
        html.Span([
            html.I(className="fas fa-filter me-2"),
            *chips,
            html.Span(f"·  {total_visible} of {total_overall} incidents",
                      style={"marginLeft": "4px", "color": TEXT_MUTED}),
        ]),
        html.Span("Click an active chip/tile again or use Clear to reset.",
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
     Output("recent-pages-container", "children"),
     Output("tier-filter-bar", "children")],
    [Input("interval-refresh", "n_intervals"),
     Input("selected-unit-store", "data"),
     Input("selected-tier-store", "data")],
)
def refresh_data(_n, selected_unit, selected_tier):
    try:
        all_incidents = get_active_incidents()
    except Exception as e:
        all_incidents = []
        print(f"Error fetching incidents: {e}")

    # Scope the entire dashboard to incidents that map to a tile on the
    # hospital floor map. Infrastructure / unmapped incidents are excluded
    # from counts, list, floor-tile badges, and the MTTR chart.
    incidents = [i for i in all_incidents if i.get("affected_unit") in UNIT_NAME_BY_ID]

    floor_map = build_floor_map(incidents, selected_unit)

    # Tier counts are computed on the unit-filtered set so the chip counts
    # reflect what's actually relevant when a unit is also selected.
    unit_scoped = [i for i in incidents if (not selected_unit or i.get("affected_unit") == selected_unit)]
    tier_counts = {tid: 0 for tid, _ in TIER_ORDER}
    for i in unit_scoped:
        t = i.get("clinical_impact_tier")
        if t in tier_counts:
            tier_counts[t] += 1

    visible_incidents = [
        i for i in unit_scoped
        if (not selected_tier or i.get("clinical_impact_tier") == selected_tier)
    ]
    if visible_incidents:
        table = render_incidents_list(visible_incidents)
    elif selected_unit or selected_tier:
        scope_bits = []
        if selected_unit:
            scope_bits.append(UNIT_NAME_BY_ID.get(selected_unit, selected_unit))
        if selected_tier:
            scope_bits.append(dict(TIER_ORDER).get(selected_tier, selected_tier))
        table = html.Div(f"No active incidents for {' · '.join(scope_bits)}.",
                         className="empty-state")
    else:
        table = html.Div("No active incidents.", className="empty-state")

    banner, clear_btn_style = render_filter_banner(
        selected_unit, selected_tier, len(visible_incidents), len(incidents)
    )
    tier_chip_bar = render_tier_chip_bar(tier_counts, selected_tier)

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

    return (floor_map, table, banner, clear_btn_style, count_badge, now_str,
            store_data, recent_pages, tier_chip_bar)


@callback(
    [Output("selected-unit-store", "data"),
     Output("selected-tier-store", "data")],
    [Input({"type": "unit-tile", "unit": ALL}, "n_clicks"),
     Input({"type": "tier-chip", "tier": ALL}, "n_clicks"),
     Input("clear-filter-btn", "n_clicks")],
    [State("selected-unit-store", "data"),
     State("selected-tier-store", "data")],
    prevent_initial_call=True,
)
def update_selected_filters(_tile_clicks, _tier_clicks, _clear_clicks,
                            current_unit, current_tier):
    # The floor map tiles are rebuilt every refresh interval, which sets their
    # n_clicks back to 0 and fires this callback with value=0. We must ignore
    # those "phantom" triggers and only act on real user clicks (n_clicks > 0).
    if not ctx.triggered:
        return no_update, no_update

    import json as _json
    real_trigger = next((t for t in ctx.triggered if t.get("value")), None)
    if not real_trigger:
        return no_update, no_update

    prop_id = real_trigger["prop_id"]
    if prop_id.startswith("clear-filter-btn"):
        return None, None
    if "unit-tile" in prop_id:
        id_part = prop_id.rsplit(".", 1)[0]
        try:
            tile_id = _json.loads(id_part)
        except Exception:
            return no_update, no_update
        clicked = tile_id.get("unit")
        return (None if clicked == current_unit else clicked), no_update
    if "tier-chip" in prop_id:
        id_part = prop_id.rsplit(".", 1)[0]
        try:
            chip_id = _json.loads(id_part)
        except Exception:
            return no_update, no_update
        clicked = chip_id.get("tier")
        return no_update, (None if clicked == current_tier else clicked)
    return no_update, no_update


# ─────────────────────────────────────────────────────────────────────────────
# BLAST RADIUS GRAPH — callbacks
# ─────────────────────────────────────────────────────────────────────────────
_BLAST_CACHE = {"nodes": None, "edges": None, "ts": None}


def _load_blast(force=False):
    now = datetime.now(timezone.utc)
    if not force and _BLAST_CACHE["nodes"] is not None and _BLAST_CACHE["ts"] and \
            (now - _BLAST_CACHE["ts"]).total_seconds() < 60:
        return _BLAST_CACHE["nodes"], _BLAST_CACHE["edges"]
    try:
        nodes, edges = get_dependency_blast_radius()
    except Exception as e:
        print(f"Blast radius load error: {e}")
        nodes, edges = [], []
    _BLAST_CACHE["nodes"], _BLAST_CACHE["edges"], _BLAST_CACHE["ts"] = nodes, edges, now
    return nodes, edges


def _default_anchor_id(nodes):
    for n in nodes:
        if n["data"]["name"] == "Epic Hyperspace - Production":
            return n["data"]["id"]
    return nodes[0]["data"]["id"] if nodes else None


@callback(
    [Output("blast-anchor-dropdown", "options"),
     Output("blast-category-chips", "children")],
    [Input("interval-refresh", "n_intervals"),
     Input("blast-categories-store", "data")],
)
def populate_blast_controls(_n, categories):
    nodes, _ = _load_blast()
    sorted_nodes = sorted(
        nodes,
        key=lambda n: (n["data"]["criticality"], -n["data"]["incident_count"], n["data"]["name"]),
    )
    options = [
        {"label": f"{n['data']['name']}  ·  {n['data']['category']}",
         "value": n["data"]["id"]}
        for n in sorted_nodes
    ]
    active_cats = set(categories or [])
    chips = []
    for bid, (label, _pred) in DEP_CATEGORY_BUCKETS.items():
        cls = "cat-chip" + (" cat-chip--active" if bid in active_cats else "")
        chips.append(html.Button(
            label, id={"type": "blast-cat", "bucket": bid}, n_clicks=0, className=cls,
        ))
    chips.append(html.Button(
        "Reset", id={"type": "blast-cat", "bucket": "__reset__"}, n_clicks=0,
        className="cat-chip", style={"marginLeft": "6px", "fontStyle": "italic"},
    ))
    return options, chips


@callback(
    Output("blast-categories-store", "data"),
    Input({"type": "blast-cat", "bucket": ALL}, "n_clicks"),
    State("blast-categories-store", "data"),
    prevent_initial_call=True,
)
def toggle_blast_category(_clicks, current):
    import json as _json
    if not ctx.triggered:
        return no_update
    real = next((t for t in ctx.triggered if t.get("value")), None)
    if not real:
        return no_update
    try:
        cid = _json.loads(real["prop_id"].rsplit(".", 1)[0])
    except Exception:
        return no_update
    bucket = cid.get("bucket")
    if bucket == "__reset__":
        return []
    current = list(current or [])
    if bucket in current:
        current.remove(bucket)
    else:
        current.append(bucket)
    return current


@callback(
    Output("blast-risk-mode-store", "data"),
    Input({"type": "risk-mode", "mode": ALL}, "n_clicks"),
    State("blast-risk-mode-store", "data"),
    prevent_initial_call=True,
)
def toggle_risk_mode(_clicks, current):
    import json as _json
    if not ctx.triggered:
        return no_update
    real = next((t for t in ctx.triggered if t.get("value")), None)
    if not real:
        return no_update
    try:
        rid = _json.loads(real["prop_id"].rsplit(".", 1)[0])
    except Exception:
        return no_update
    return rid.get("mode") or current


@callback(
    Output({"type": "risk-mode", "mode": ALL}, "className"),
    Input("blast-risk-mode-store", "data"),
    State({"type": "risk-mode", "mode": ALL}, "id"),
)
def style_risk_mode_buttons(mode, ids):
    out = []
    for i in ids or []:
        cls = "risk-toggle__btn"
        if (i or {}).get("mode") == (mode or "incident_load"):
            cls += " risk-toggle__btn--active"
        out.append(cls)
    return out


@callback(
    Output("blast-anchor-store", "data"),
    [Input("blast-anchor-dropdown", "value"),
     Input("blast-cyto", "tapNodeData")],
    prevent_initial_call=True,
)
def update_blast_anchor(dropdown_val, tap_node):
    # whichever fired last wins (Dash uses ctx)
    if not ctx.triggered:
        return no_update
    trig = ctx.triggered[0]["prop_id"]
    if trig.startswith("blast-cyto"):
        return (tap_node or {}).get("id") or no_update
    return dropdown_val


def _restyle_for_risk(node, mode):
    """Recompute status class based on the selected risk mode without
    rebuilding the whole node list."""
    d = node["data"]
    if mode == "patient_safety":
        score = 100 if d["patient_safety"] > 0 else (40 if d["incident_count"] > 0 else 0)
    elif mode == "p1_major":
        score = 100 if d["p1_count"] > 0 else (60 if d["major_count"] > 0 else (20 if d["incident_count"] > 0 else 0))
    else:  # incident_load (default)
        score = d["risk_score"]
    if score >= 80: status = "critical"
    elif score >= 35: status = "warning"
    else: status = "healthy"
    classes = []
    base = (node.get("classes") or "").split()
    for c in base:
        if not c.startswith("node-"):
            classes.append(c)
    classes.append(f"node-{status}")
    new = {"data": dict(d), "classes": " ".join(classes)}
    new["data"]["status"] = status
    return new


@callback(
    Output("blast-cyto", "elements"),
    [Input("interval-refresh", "n_intervals"),
     Input("blast-anchor-store", "data"),
     Input("blast-categories-store", "data"),
     Input("blast-risk-mode-store", "data")],
)
def render_blast(_n, anchor, categories, risk_mode):
    nodes_all, edges_all = _load_blast()
    if not nodes_all:
        return []

    # Fall back to Epic Hyperspace as the default anchor.
    if anchor is None:
        anchor = _default_anchor_id(nodes_all)

    # Category filter: when one or more buckets are selected, hide nodes not
    # in those buckets (but always keep the anchor visible for context).
    active = set(categories or [])
    if active:
        visible_ids = {n["data"]["id"] for n in nodes_all if n["data"]["bucket"] in active}
        if anchor:
            visible_ids.add(anchor)
    else:
        visible_ids = {n["data"]["id"] for n in nodes_all}

    nodes_vis = [_restyle_for_risk(n, risk_mode) for n in nodes_all if n["data"]["id"] in visible_ids]
    edges_vis = [e for e in edges_all
                 if e["data"]["source"] in visible_ids and e["data"]["target"] in visible_ids]

    if anchor:
        # Mark the anchor + collect hops
        for n in nodes_vis:
            if n["data"]["id"] == anchor:
                n["classes"] = (n.get("classes") or "") + " anchor"
        hop1 = set()
        for e in edges_vis:
            if e["data"]["source"] == anchor: hop1.add(e["data"]["target"])
            if e["data"]["target"] == anchor: hop1.add(e["data"]["source"])
        hop2 = set()
        for e in edges_vis:
            if (e["data"]["source"] in hop1 and e["data"]["target"] != anchor
                    and e["data"]["target"] not in hop1):
                hop2.add(e["data"]["target"])
            if (e["data"]["target"] in hop1 and e["data"]["source"] != anchor
                    and e["data"]["source"] not in hop1):
                hop2.add(e["data"]["source"])
        for n in nodes_vis:
            nid = n["data"]["id"]
            if nid == anchor: continue
            if nid in hop1:
                n["classes"] = (n.get("classes") or "") + " hop1"
            elif nid in hop2:
                n["classes"] = (n.get("classes") or "") + " hop2"
            else:
                n["classes"] = (n.get("classes") or "") + " faded"
        for e in edges_vis:
            src, tgt = e["data"]["source"], e["data"]["target"]
            if anchor in (src, tgt):
                e["classes"] = "hop1"
            elif src in hop1 and tgt in hop1:
                e["classes"] = "hop2"
            elif src in hop1 or tgt in hop1:
                e["classes"] = "hop2"
            else:
                e["classes"] = "faded"

    return nodes_vis + edges_vis


@callback(
    Output("blast-inspector", "children"),
    [Input("blast-cyto", "tapNodeData"),
     Input("blast-anchor-store", "data")],
)
def render_inspector(tap, anchor):
    nodes, edges = _load_blast()
    target_id = (tap or {}).get("id") or anchor
    if not target_id:
        return "Click a node for details."
    node = next((n for n in nodes if n["data"]["id"] == target_id), None)
    if not node:
        return "Click a node for details."
    d = node["data"]
    pill_cls = f"blast-inspector__pill blast-inspector__pill--{d['status']}"
    upstream = [e for e in edges if e["data"]["target"] == d["id"]]
    downstream = [e for e in edges if e["data"]["source"] == d["id"]]
    return html.Div([
        html.Div(d["name"], className="blast-inspector__title"),
        html.Div([
            html.Span(d["status"].upper(), className=pill_cls),
            html.Span(d["category"], style={"marginLeft": "6px", "color": TEXT_MUTED}),
        ], style={"marginBottom": "6px"}),
        html.Div([html.Span("Active incidents"), html.B(str(d["incident_count"]))],
                 className="blast-inspector__row"),
        html.Div([html.Span("P1 active"), html.B(str(d["p1_count"]))],
                 className="blast-inspector__row"),
        html.Div([html.Span("Patient-safety"), html.B(str(d["patient_safety"]))],
                 className="blast-inspector__row"),
        html.Div([html.Span("Criticality"), html.B(d.get("service_tier") or "—")],
                 className="blast-inspector__row"),
        html.Div([html.Span("Depends on"), html.B(str(len(downstream)))],
                 className="blast-inspector__row"),
        html.Div([html.Span("Depended on by"), html.B(str(len(upstream)))],
                 className="blast-inspector__row"),
    ])


DARK_PLOT_LAYOUT = dict(
    plot_bgcolor=BG_PANEL,
    paper_bgcolor=BG_PANEL,
    font=dict(family="Lato, system-ui, sans-serif", color=TEXT_SECONDARY),
    xaxis=dict(gridcolor=BORDER_SUBTLE, zerolinecolor=BORDER_SUBTLE, color=TEXT_MUTED),
    yaxis=dict(gridcolor=BORDER_SUBTLE, zerolinecolor=BORDER_SUBTLE, color=TEXT_MUTED),
)


def _empty_mttr_fig(message):
    fig = go.Figure()
    fig.update_layout(
        height=280, margin=dict(l=40, r=20, t=12, b=40),
        **DARK_PLOT_LAYOUT,
        annotations=[{"text": message, "xref": "paper", "yref": "paper",
                      "x": 0.5, "y": 0.5, "showarrow": False,
                      "font": {"color": TEXT_MUTED, "family": "Lato, sans-serif", "size": 13}}],
    )
    return fig


@callback(
    [Output("mttr-chart", "figure"),
     Output("mttr-title", "children")],
    [Input("interval-refresh", "n_intervals"),
     Input("selected-unit-store", "data"),
     Input("selected-tier-store", "data")],
)
def update_mttr_chart(_n, selected_unit, selected_tier):
    base_title = "Mean Time To Resolve (MTTR) — by Floor Unit · Last 30 Days"
    try:
        data = get_mttr_by_unit()
    except Exception as e:
        print(f"MTTR chart error: {e}")
        return _empty_mttr_fig("No MTTR data available"), base_title

    if not data:
        return _empty_mttr_fig("No MTTR data available"), base_title

    df = pd.DataFrame(data)
    df["unit_id"] = df["location_name"].apply(get_unit_from_location)
    # Scope to incidents that map to a tile on the hospital floor map. Drop
    # unmapped infrastructure incidents entirely (no "Other" line).
    df = df[df["unit_id"].isin(UNIT_NAME_BY_ID.keys())]
    if df.empty:
        return _empty_mttr_fig("No MTTR data available"), base_title
    df["unit"] = df["unit_id"].map(UNIT_NAME_BY_ID)

    # Apply the same tier filter as the rest of the dashboard.
    if selected_tier:
        df = df[df["clinical_impact_tier"] == selected_tier]
        if df.empty:
            tier_label = dict(TIER_ORDER).get(selected_tier, selected_tier)
            return _empty_mttr_fig(f"No resolved {tier_label} incidents in the last 30 days"), base_title

    # Re-aggregate to per-(unit, date) so we get one line per floor unit
    # rather than per raw location_name.
    agg = (
        df.groupby(["unit", "resolve_date"], as_index=False)
          .agg(mttr_hours=("mttr_hours", "mean"))
    )

    title = base_title
    title_bits = []
    if selected_unit:
        title_bits.append(UNIT_NAME_BY_ID.get(selected_unit, selected_unit))
    if selected_tier:
        title_bits.append(dict(TIER_ORDER).get(selected_tier, selected_tier))
    if title_bits:
        title = f"Mean Time To Resolve (MTTR) — {' · '.join(title_bits)} · Last 30 Days"
    if selected_unit:
        unit_name = UNIT_NAME_BY_ID.get(selected_unit, selected_unit)
        agg = agg[agg["unit"] == unit_name]
        if agg.empty:
            return _empty_mttr_fig(f"No resolved incidents for {unit_name} in the last 30 days"), title

    # Stable color order matching floor-map tile order, so each unit gets the
    # same color in the chart that it has on the map.
    unit_order = [u["name"] for u in HOSPITAL_UNITS]
    unit_palette = [PURPLE, TEAL, GREEN, ORANGE_ALERT, RED_ALERT,
                    BLUE, AMBER, "#EC4899", "#06B6D4", "#A78BFA"]
    color_map = {u: unit_palette[i % len(unit_palette)] for i, u in enumerate(unit_order)}

    fig = px.line(
        agg, x="resolve_date", y="mttr_hours", color="unit",
        labels={"resolve_date": "Date",
                "mttr_hours": "Mean Time To Resolve (hours)",
                "unit": "Unit"},
        category_orders={"unit": unit_order},
        color_discrete_map=color_map,
    )
    fig.update_layout(
        margin=dict(l=40, r=20, t=12, b=50), height=280,
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0.5, xanchor="center",
                    font=dict(size=10, color=TEXT_SECONDARY)),
        **DARK_PLOT_LAYOUT,
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

    # 1) Log the page action to Lakebase, capture the generated action_id +
    #    created_at so we can MERGE the same row into UC from the Job.
    db_msg = None
    lakebase_row = None
    try:
        lakebase_row = execute_db_returning(
            """INSERT INTO public.paged_actions
               (incident_sys_id, incident_number, paged_group, paged_by,
                page_type, clinical_impact_tier, affected_unit, message)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
               RETURNING action_id, created_at""",
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

    # 3) Trigger the Databricks Job that sends email + syncs the row to UC
    #    (Option 1). Fall back to in-process SMTP, then to a mailto: link.
    run_id, job_detail = trigger_page_notifier_job(incident, form, lakebase_row)
    job_route = run_id is not None
    sent = False
    detail = job_detail
    if not job_route:
        sent, detail = send_page_email(subject, body)

    parts = []
    if job_route:
        parts.append(html.Div(
            [html.I(className="fas fa-check-circle me-2"),
             html.Strong("Page sent successfully")],
            style={"color": GREEN}))
    elif sent:
        parts.append(html.Div(
            [html.I(className="fas fa-check-circle me-2"),
             html.Strong("Page sent successfully")],
            style={"color": GREEN}))
    else:
        parts.append(html.Div(
            [html.I(className="fas fa-envelope me-2"),
             html.Strong("Open in your mail client"),
             f" to send to {PAGE_EMAIL_TO}.  Page logged for {inc_number}."],
            style={"color": TEXT_PRIMARY}))
        parts.append(html.Div(detail, style={"color": TEXT_MUTED, "fontSize": "0.76rem", "marginTop": "2px"}))
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

    color = "success" if (job_route or sent) else "info"
    return dbc.Alert(parts, color=color, className="mb-0 py-2", style={"width": "100%"})


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("DATABRICKS_APP_PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
