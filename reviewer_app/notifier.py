"""
notifier.py - Email Notification Sender

Sends TWO separate emails to the reviewer:
1. Notification email — contains the case ID and secure dashboard link
2. PIN email — contains the 6-digit verification PIN

This two-email approach ensures that even if someone else
clicks the link, they cannot access the dashboard without
the PIN which arrives in a separate email.

This matches real industry practice (e.g. LexisNexis RiskNarrative,
banking portals) where link + OTP/PIN = two-factor access.
"""

import os
import random
import secrets
from flask import Flask
from flask_mail import Mail, Message
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def create_mail_app():
    """
    Creates and configures a minimal Flask app for sending emails.
    Returns the app and mail objects.
    """
    app = Flask(__name__)

    # Gmail SMTP configuration
    app.config['MAIL_SERVER'] = 'smtp.gmail.com'    # Gmail's SMTP server
    app.config['MAIL_PORT'] = 587                    # Port for TLS encryption
    app.config['MAIL_USE_TLS'] = True                # Use TLS (secure connection)
    app.config['MAIL_USE_SSL'] = False               # Don't use SSL (TLS is enough)
    app.config['MAIL_USERNAME'] = os.getenv('GMAIL_SENDER')
    app.config['MAIL_PASSWORD'] = os.getenv('GMAIL_APP_PASSWORD')
    app.config['MAIL_DEFAULT_SENDER'] = os.getenv('GMAIL_SENDER')

    mail = Mail(app)

    return app, mail


def generate_token() -> str:
    """
    Generates a cryptographically secure random token for the dashboard URL.
    This token is impossible to guess — 32 random characters.
    Example: "x7k9m2p4q8r1s3t6w0y5z..."

    Returns:
        A secure random URL-safe token string
    """
    return secrets.token_urlsafe(32)


def generate_pin() -> str:
    """
    Generates a random 6-digit PIN for dashboard access verification.
    This PIN is sent in a SEPARATE email from the dashboard link,
    so even if someone else clicks the link, they cannot proceed
    without the PIN.

    Returns:
        A 6-digit PIN as a string (e.g. "847291")
    """
    return str(random.randint(100000, 999999))


def send_edd_notification(case_id: str,
                          customer_name: str,
                          dashboard_url: str,
                          pin: str) -> bool:
    """
    Sends TWO emails for EDD confirmation:

    Email 1 — Notification: case ID + secure dashboard link
    Email 2 — PIN: 6-digit verification PIN (sent separately)

    Args:
        case_id: The unique case identifier (e.g. CASE-2026-315)
        customer_name: The customer's full name
        dashboard_url: The secure URL containing the token
        pin: The 6-digit PIN generated for this review session

    Returns:
        True if both emails sent successfully, False if either failed
    """

    app, mail = create_mail_app()
    reviewer_email = os.getenv('REVIEWER_EMAIL')

    # ---------------------------------------------------------------
    # Email 1 — Notification email (link only, no PIN)
    # ---------------------------------------------------------------
    notification_subject = f"[ACTION REQUIRED] EDD Confirmation Needed — {case_id}"

    notification_body = f"""
Dear Reviewer,

A new case has been flagged and requires your Enhanced Due Diligence
(EDD) confirmation before the pipeline can continue.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Case ID       : {case_id}
Customer      : {customer_name}
Action Needed : EDD Completion Confirmation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Please click the link below to access the secure review dashboard:

{dashboard_url}

You will be asked to enter a verification PIN when you open the link.
Your PIN will arrive in a separate email shortly.

This link expires in 24 hours.
The pipeline is paused and waiting for your response.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LexisNexis Agentic Identity Risk Management System
This is an automated notification. Do not reply to this email.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    # ---------------------------------------------------------------
    # Email 2 — PIN email (PIN only, no link)
    # ---------------------------------------------------------------
    pin_subject = f"[SECURE PIN] Your Verification PIN — {case_id}"

    pin_body = f"""
Dear Reviewer,

Your secure verification PIN for case {case_id} is:

        ┌─────────────────┐
        │   PIN: {pin}   │
        └─────────────────┘

Enter this PIN when prompted on the review dashboard.

IMPORTANT SECURITY NOTICE:
- This PIN is valid for 24 hours only
- Do not share this PIN with anyone
- If you did not request this PIN, please contact your administrator

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LexisNexis Agentic Identity Risk Management System
This is an automated notification. Do not reply to this email.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    with app.app_context():
        try:
            # Send Email 1 — Notification with link
            msg1 = Message(
                subject=notification_subject,
                recipients=[reviewer_email],
                body=notification_body
            )
            mail.send(msg1)
            print(f"[NOTIFIER] EDD notification email sent to "
                  f"{reviewer_email} for {case_id}")

            # Send Email 2 — PIN separately
            msg2 = Message(
                subject=pin_subject,
                recipients=[reviewer_email],
                body=pin_body
            )
            mail.send(msg2)
            print(f"[NOTIFIER] PIN email sent to "
                  f"{reviewer_email} for {case_id}")

            return True

        except Exception as e:
            print(f"[NOTIFIER] Failed to send EDD notification: {e}")
            return False


def send_human_review_notification(case_id: str,
                                   customer_name: str,
                                   dashboard_url: str,
                                   pin: str) -> bool:
    """
    Sends TWO emails for human review confirmation:

    Email 1 — Notification: case ID + secure dashboard link
    Email 2 — PIN: 6-digit verification PIN (sent separately)

    Args:
        case_id: The unique case identifier
        customer_name: The customer's full name
        dashboard_url: The secure URL containing the token
        pin: The 6-digit PIN generated for this review session

    Returns:
        True if both emails sent successfully, False if either failed
    """

    app, mail = create_mail_app()
    reviewer_email = os.getenv('REVIEWER_EMAIL')

    # ---------------------------------------------------------------
    # Email 1 — Notification email (link only, no PIN)
    # ---------------------------------------------------------------
    notification_subject = f"[ACTION REQUIRED] Human Review Needed — {case_id}"

    notification_body = f"""
Dear Reviewer,

A new case has been flagged and requires your human review before
the pipeline can continue. This case has a low confidence score
and requires analyst verification under GDPR Article 22C.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Case ID       : {case_id}
Customer      : {customer_name}
Action Needed : Human Review & Acknowledgement
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Please click the link below to access the secure review dashboard:

{dashboard_url}

You will be asked to enter a verification PIN when you open the link.
Your PIN will arrive in a separate email shortly.

This link expires in 24 hours.
The pipeline is paused and waiting for your acknowledgement.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LexisNexis Agentic Identity Risk Management System
This is an automated notification. Do not reply to this email.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    # ---------------------------------------------------------------
    # Email 2 — PIN email (PIN only, no link)
    # ---------------------------------------------------------------
    pin_subject = f"[SECURE PIN] Your Verification PIN — {case_id}"

    pin_body = f"""
Dear Reviewer,

Your secure verification PIN for case {case_id} is:

        ┌─────────────────┐
        │   PIN: {pin}   │
        └─────────────────┘

Enter this PIN when prompted on the review dashboard.

IMPORTANT SECURITY NOTICE:
- This PIN is valid for 24 hours only
- Do not share this PIN with anyone
- If you did not request this PIN, please contact your administrator

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LexisNexis Agentic Identity Risk Management System
This is an automated notification. Do not reply to this email.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    with app.app_context():
        try:
            # Send Email 1 — Notification with link
            msg1 = Message(
                subject=notification_subject,
                recipients=[reviewer_email],
                body=notification_body
            )
            mail.send(msg1)
            print(f"[NOTIFIER] Human review notification email sent to "
                  f"{reviewer_email} for {case_id}")

            # Send Email 2 — PIN separately
            msg2 = Message(
                subject=pin_subject,
                recipients=[reviewer_email],
                body=pin_body
            )
            mail.send(msg2)
            print(f"[NOTIFIER] PIN email sent to "
                  f"{reviewer_email} for {case_id}")

            return True

        except Exception as e:
            print(f"[NOTIFIER] Failed to send human review notification: {e}")
            return False