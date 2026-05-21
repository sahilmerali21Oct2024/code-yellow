# Databricks notebook source
# MAGIC %md
# MAGIC # Code Yellow — Send Page Email
# MAGIC
# MAGIC Job-triggered notebook that sends a templated "INCIDENT PAGE" email.
# MAGIC
# MAGIC **Triggered by:** the Code Yellow Databricks App, via `jobs.run_now` with parameters.
# MAGIC
# MAGIC **Email transport priority:**
# MAGIC 1. **Resend** — if secret `code-yellow/resend_api_key` is set
# MAGIC 2. **SMTP** — if `code-yellow/smtp_password` + `smtp_user` are set
# MAGIC 3. **Log-only** — if neither is configured (notebook still exits success so the app shows a clean status)
# MAGIC
# MAGIC See README in the repo for setup commands.

# COMMAND ----------

import json
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage

import urllib.request
import urllib.error

# COMMAND ----------
# MAGIC %md ## Parameters
# COMMAND ----------

dbutils.widgets.text("inc_number", "", "Incident number")
dbutils.widgets.text("priority", "", "Priority (1-5)")
dbutils.widgets.text("group", "", "Target group")
dbutils.widgets.text("unit", "", "Affected unit (id)")
dbutils.widgets.text("unit_label", "", "Affected unit (label)")
dbutils.widgets.text("page_type", "page_team", "Page action type")
dbutils.widgets.text("message", "", "Free-text message")
dbutils.widgets.text("page_to", "sahil.merali@databricks.com", "Email recipient")
dbutils.widgets.text("secret_scope", "code-yellow", "Databricks secret scope")

inc_number   = dbutils.widgets.get("inc_number") or "[INC number]"
priority     = dbutils.widgets.get("priority") or ""
group        = dbutils.widgets.get("group") or "[team/group being paged]"
unit_id      = dbutils.widgets.get("unit")
unit_label   = dbutils.widgets.get("unit_label") or unit_id or "[system, application, or service impacted]"
page_type    = dbutils.widgets.get("page_type") or "page_team"
message_body = dbutils.widgets.get("message") or "[free text description of the issue, impact, and any immediate actions needed]"
page_to      = dbutils.widgets.get("page_to") or "sahil.merali@databricks.com"
secret_scope = dbutils.widgets.get("secret_scope") or "code-yellow"

PAGE_TYPE_LABELS = {
    "page_team":     "Page Team",
    "bridge_call":   "Bridge Call",
    "status_update": "Status Update",
}
action_label = PAGE_TYPE_LABELS.get(page_type, "Page Team")
priority_str = f"P{priority}" if priority and priority not in ("None", "?") else "[P1 / P2 / P3]"

# COMMAND ----------
# MAGIC %md ## Build the email
# COMMAND ----------

body_lines = [
    "**INCIDENT PAGE**",
    "",
    f"Incident: {inc_number}",
    f"Page Type: {priority_str}",
    f"Target Group: {group}",
    f"Affected Unit: {unit_label}",
    "",
    "Message:",
    message_body,
    "",
    "—",
    f"Page action: {action_label}",
    f"Sent by Code Yellow job at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
]
body = "\n".join(body_lines)
subject = f"[{priority_str}] INCIDENT PAGE — {inc_number} — {action_label}"

print("Composed email:")
print("Subject:", subject)
print("To:", page_to)
print("---")
print(body)

# COMMAND ----------
# MAGIC %md ## Secret lookup helpers
# COMMAND ----------

def _safe_secret(scope, key):
    try:
        v = dbutils.secrets.get(scope=scope, key=key)
        return v if v else None
    except Exception:
        return None

resend_key    = _safe_secret(secret_scope, "resend_api_key")
smtp_password = _safe_secret(secret_scope, "smtp_password")
smtp_user     = _safe_secret(secret_scope, "smtp_user")
smtp_from     = _safe_secret(secret_scope, "smtp_from") or smtp_user
smtp_host     = _safe_secret(secret_scope, "smtp_host") or "smtp.gmail.com"
smtp_port_str = _safe_secret(secret_scope, "smtp_port") or "587"
try:
    smtp_port = int(smtp_port_str)
except ValueError:
    smtp_port = 587

resend_from = _safe_secret(secret_scope, "resend_from") or "Code Yellow <onboarding@resend.dev>"

# COMMAND ----------
# MAGIC %md ## Send via Resend (preferred)
# COMMAND ----------

def send_via_resend(subject, body, api_key, to_addr, from_addr):
    payload = json.dumps({
        "from": from_addr,
        "to": [to_addr],
        "subject": subject,
        "text": body,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8")
            return True, f"Resend OK: {data[:200]}"
    except urllib.error.HTTPError as e:
        return False, f"Resend HTTPError {e.code}: {e.read().decode('utf-8', errors='ignore')[:200]}"
    except Exception as e:
        return False, f"Resend error: {str(e)[:200]}"

# COMMAND ----------
# MAGIC %md ## Send via SMTP (fallback)
# COMMAND ----------

def send_via_smtp(subject, body, host, port, user, password, sender, recipient):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(body)
    ctx_ssl = ssl.create_default_context()
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, context=ctx_ssl, timeout=15) as s:
                s.login(user, password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=15) as s:
                s.ehlo()
                s.starttls(context=ctx_ssl)
                s.ehlo()
                s.login(user, password)
                s.send_message(msg)
        return True, f"SMTP OK via {host}:{port}"
    except Exception as e:
        return False, f"SMTP error: {str(e)[:200]}"

# COMMAND ----------
# MAGIC %md ## Dispatch
# COMMAND ----------

transport = None
detail = None

if resend_key:
    transport = "resend"
    sent, detail = send_via_resend(subject, body, resend_key, page_to, resend_from)
elif smtp_password and smtp_user:
    transport = "smtp"
    sent, detail = send_via_smtp(subject, body, smtp_host, smtp_port,
                                  smtp_user, smtp_password,
                                  smtp_from or smtp_user, page_to)
else:
    transport = "log-only"
    sent = True
    detail = (
        "No transport configured. Set 'resend_api_key' OR "
        "('smtp_user' + 'smtp_password') in secret scope 'code-yellow' to enable real sends."
    )

print(f"\nTransport: {transport}")
print("Result:", detail)

result = {
    "transport": transport,
    "sent": bool(sent),
    "detail": detail,
    "to": page_to,
    "subject": subject,
    "inc_number": inc_number,
}

# Always exit success — the app shouldn't get a hard failure just because the
# transport isn't configured yet. The job's email_notifications.on_failure
# only fires on actual unhandled exceptions.
dbutils.notebook.exit(json.dumps(result))
