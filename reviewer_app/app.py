# Flask web dashboard

"""
reviewer_app/app.py - Secure Reviewer Dashboard

This is the Flask web application that serves as the secure case
review dashboard. It handles:

1. Token validation    — verifies the link is genuine and not expired
2. PIN verification    — ensures only the authorised reviewer can access
3. Case dashboard      — displays full case details to the reviewer
4. Decision handling   — receives and records the reviewer's decision
5. Pipeline signalling — communicates the decision back to the pipeline

The dashboard runs in a background thread alongside the main pipeline,
so both can operate simultaneously without blocking each other.
"""

import threading
from datetime import datetime, timedelta
from flask import Flask, request, render_template, redirect, url_for
from schemas import RiskAssessment

# -----------------------------------------------------------------------
# Flask app initialisation
# -----------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = 'lexisnexis_reviewer_secret_key_2026'  
# Secret key is required by Flask for session management.
# In production this would be a proper secret stored in .env

# -----------------------------------------------------------------------
# In-memory storage for pending reviews
#
# This dictionary stores all active review sessions.
# Key   = the secure token sent in the email link
# Value = dictionary containing all review details
#
# Structure:
# {
#     "token_abc123": {
#         "case_id": "CASE-2026-315",
#         "customer_name": "Marcus Whitfield",
#         "review_type": "edd",           # or "human_review"
#         "pin": "847291",
#         "expires_at": datetime object,
#         "risk_assessment": RiskAssessment object,
#         "decision": None,               # filled when reviewer responds
#         "decided_at": None              # timestamp of decision
#     }
# }
# -----------------------------------------------------------------------
pending_reviews = {}

# -----------------------------------------------------------------------
# Threading event — this is how the dashboard signals the pipeline
#
# Think of it like a traffic light:
# - RED   (not set) = pipeline is waiting, reviewer hasn't decided yet
# - GREEN (set)     = reviewer has made a decision, pipeline can continue
#
# The pipeline calls event.wait() — it stops and waits until the event
# is set. The dashboard calls event.set() when the reviewer clicks a
# button — this immediately wakes up the waiting pipeline.
# -----------------------------------------------------------------------
review_complete_event = threading.Event()


def register_review(token: str,
                    case_id: str,
                    customer_name: str,
                    review_type: str,
                    pin: str,
                    risk_assessment: RiskAssessment,
                    expiry_hours: int = 24):
    """
    Registers a new pending review session in the in-memory store.
    Called by the pipeline BEFORE sending the notification email.

    Args:
        token: The secure random token generated for this review
        case_id: The unique case identifier
        customer_name: The customer's full name
        review_type: Either "edd" or "human_review"
        pin: The 6-digit PIN for this review session
        risk_assessment: The full RiskAssessment object from Agent 1
        expiry_hours: How long the link remains valid (default 24 hours)
    """
    pending_reviews[token] = {
        "case_id": case_id,
        "customer_name": customer_name,
        "review_type": review_type,
        "pin": pin,
        "expires_at": datetime.now() + timedelta(hours=expiry_hours),
        "risk_assessment": risk_assessment,
        "decision": None,       # will be filled when reviewer responds
        "decided_at": None      # will be filled when reviewer responds
    }
    print(f"[DASHBOARD] Review session registered for {case_id} "
          f"(type: {review_type}, expires in {expiry_hours}h)")


def get_decision(token: str):
    """
    Returns the reviewer's decision for a given token.
    Called by the pipeline after the event is set, to retrieve
    what the reviewer actually decided.

    Args:
        token: The secure token for this review session

    Returns:
        The decision string, or None if not yet decided
    """
    review = pending_reviews.get(token)
    if review:
        return review.get("decision")
    return None


# -----------------------------------------------------------------------
# Flask Routes
# -----------------------------------------------------------------------

@app.route('/review', methods=['GET'])
def review_landing():
    """
    ROUTE 1: Landing page — first page the reviewer sees after
    clicking the link in the notification email.

    Validates the token from the URL:
    - If token missing    → show error page
    - If token not found  → show error page
    - If token expired    → show expired page
    - If token valid      → show PIN entry form

    URL format: http://localhost:5000/review?token=abc123...
    """

    # Get the token from the URL query parameter
    token = request.args.get('token')

    # Check token exists in the URL
    if not token:
        return render_template('error.html',
                               message="Invalid link — no token provided."), 400

    # Check token exists in our pending reviews store
    review = pending_reviews.get(token)
    if not review:
        return render_template('error.html',
                               message="Invalid or already used link."), 404

    # Check token has not expired
    if datetime.now() > review['expires_at']:
        # Clean up expired token
        del pending_reviews[token]
        return render_template('error.html',
                               message="This link has expired. "
                                       "Please contact your administrator."), 410

    # Token is valid — show PIN entry page
    # Pass case_id and review_type so the PIN page can display
    # basic context (but NOT the full case details — PIN not verified yet)
    return render_template('pin_entry.html',
                           token=token,
                           case_id=review['case_id'],
                           review_type=review['review_type'])


@app.route('/verify-pin', methods=['POST'])
def verify_pin():
    """
    ROUTE 2: PIN verification — handles the PIN form submission.

    Called when the reviewer submits the PIN entry form.
    - If PIN correct   → redirect to case dashboard
    - If PIN incorrect → show PIN entry page again with error message
    """

    # Get token and entered PIN from the form
    token = request.form.get('token')
    entered_pin = request.form.get('pin')

    # Validate token still exists and is not expired
    review = pending_reviews.get(token)
    if not review or datetime.now() > review['expires_at']:
        return render_template('error.html',
                               message="Session expired. Please use "
                                       "a fresh link."), 410

    # Check PIN matches
    if entered_pin != review['pin']:
        # Wrong PIN — show entry page again with error
        return render_template('pin_entry.html',
                               token=token,
                               case_id=review['case_id'],
                               review_type=review['review_type'],
                               error="Incorrect PIN. Please try again.")

    # PIN correct — redirect to the case dashboard
    # We pass the token as a URL parameter so the dashboard
    # knows which review session to display
    return redirect(url_for('case_dashboard', token=token))


@app.route('/dashboard', methods=['GET'])
def case_dashboard():
    """
    ROUTE 3: Case dashboard — shown after successful PIN verification.

    Displays the FULL case details from the RiskAssessment object,
    plus the appropriate action buttons based on review_type:

    - EDD review      → 3 buttons: Confirmed Complete / Not Complete / Unknown
    - Human review    → 1 button:  I Have Reviewed This Case
    """

    token = request.args.get('token')

    # Validate token
    review = pending_reviews.get(token)
    if not review or datetime.now() > review['expires_at']:
        return render_template('error.html',
                               message="Session expired or invalid."), 410

    # Get the full risk assessment for display
    risk_assessment = review['risk_assessment']

    return render_template('dashboard.html',
                           token=token,
                           review=review,
                           risk_assessment=risk_assessment)


@app.route('/submit-decision', methods=['POST'])
def submit_decision():
    """
    ROUTE 4: Decision handler — called when reviewer clicks an action button.

    Records the reviewer's decision in the pending_reviews store,
    then sets the threading event to wake up the waiting pipeline.

    Decision values:
    - EDD cases:        "yes" / "no" / "skip"
    - Human review:     "acknowledged"
    """

    token = request.form.get('token')
    decision = request.form.get('decision')

    # Validate token
    review = pending_reviews.get(token)
    if not review:
        return render_template('error.html',
                               message="Invalid session."), 404

    # Record the decision and timestamp
    review['decision'] = decision
    review['decided_at'] = datetime.now().isoformat()

    print(f"[DASHBOARD] Decision received for {review['case_id']}: "
          f"{decision} at {review['decided_at']}")

    # Save these for the confirmation page
    case_id = review['case_id']
    review_type = review['review_type']

    # ---------------------------------------------------------------
    # SIGNAL THE PIPELINE
    # This is the key moment — setting the event wakes up the pipeline
    # which has been waiting with event.wait()
    # ---------------------------------------------------------------
    review_complete_event.set()

    # Show confirmation page to the reviewer
    return render_template('confirmed.html',
                           case_id=case_id,
                           decision=decision,
                           review_type=review_type)


# -----------------------------------------------------------------------
# Dashboard server management
# -----------------------------------------------------------------------

_server_thread = None  # keeps track of the background server thread


def start_dashboard(port: int = 5000):
    """
    Starts the Flask dashboard in a BACKGROUND THREAD.

    This means Flask runs alongside the pipeline simultaneously —
    the pipeline does not have to stop and wait for Flask to finish.
    Flask just sits in the background, ready to serve web requests,
    while the pipeline continues its own work (waiting for the event).

    Args:
        port: The port number for the Flask server (default 5000)
    """
    global _server_thread

    # Only start if not already running
    if _server_thread is not None and _server_thread.is_alive():
        print("[DASHBOARD] Server already running.")
        return

    def run_flask():
        """Inner function that actually starts Flask."""
        # use_reloader=False is essential when running in a thread —
        # the reloader would try to start a second thread and cause conflicts
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

    # Create and start the background thread
    _server_thread = threading.Thread(target=run_flask, daemon=True)
    # daemon=True means the thread automatically stops when the main
    # program exits — no need to manually shut it down
    _server_thread.start()
    print(f"[DASHBOARD] Review dashboard started at http://localhost:{port}")


def wait_for_decision(timeout_seconds: int = 86400) -> bool:
    """
    Blocks the pipeline until the reviewer makes a decision,
    or until the timeout expires.

    This is called by the pipeline after sending the notification.
    The pipeline stops here and waits — but Flask keeps running
    in its background thread, ready to receive the reviewer's click.

    Args:
        timeout_seconds: Maximum wait time in seconds (default 24 hours)

    Returns:
        True if reviewer responded within timeout
        False if timeout expired with no response
    """
    print(f"[DASHBOARD] Pipeline paused — waiting for reviewer decision "
          f"(timeout: {timeout_seconds//3600}h)...")

    # Clear any previous event state before waiting
    review_complete_event.clear()

    # Wait for the event to be set by submit_decision()
    responded = review_complete_event.wait(timeout=timeout_seconds)

    if responded:
        print("[DASHBOARD] Reviewer has made a decision — pipeline resuming.")
    else:
        print("[DASHBOARD] Timeout expired — no reviewer response received.")

    return responded