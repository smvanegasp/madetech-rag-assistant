"""Contact/feedback email service using Resend.

Sends an email to CONTACT_EMAIL when a user submits feedback or a contact request.
Errors are raised so the /api/contact endpoint can return a meaningful HTTP error.
"""

import os
from html import escape

import resend


def send_contact_email(
    contact_type: str,
    name: str,
    email: str,
    message: str,
) -> None:
    """Send a feedback or contact email via Resend.

    Args:
        contact_type: "feedback" or "contact"
        name: Sender's name
        email: Sender's email address
        message: The message body
    """
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        raise ValueError("RESEND_API_KEY environment variable not set")

    to_email = os.getenv("CONTACT_EMAIL")
    if not to_email:
        raise ValueError("CONTACT_EMAIL environment variable not set")

    resend.api_key = api_key

    subject_prefix = "Feedback" if contact_type == "feedback" else "Get in Touch"
    # A visitor's name must not add lines to the subject.
    one_line_name = " ".join(name.split())
    subject = f"[MadeTech RAG Assistant] {subject_prefix} from {one_line_name}"

    # Everything a visitor typed is escaped, so it shows as text and can't inject HTML or
    # links into the email.
    e_subject, e_type = escape(subject), escape(contact_type.title())
    e_name, e_email, e_message = escape(name), escape(email), escape(message)

    html_body = f"""
    <div style="font-family: sans-serif; max-width: 600px;">
      <h2 style="color: #10b981;">{e_subject}</h2>
      <table style="border-collapse: collapse; width: 100%;">
        <tr>
          <td style="padding: 8px 0; font-weight: bold; color: #6b7280; width: 80px;">Type</td>
          <td style="padding: 8px 0;">{e_type}</td>
        </tr>
        <tr>
          <td style="padding: 8px 0; font-weight: bold; color: #6b7280;">Name</td>
          <td style="padding: 8px 0;">{e_name}</td>
        </tr>
        <tr>
          <td style="padding: 8px 0; font-weight: bold; color: #6b7280;">Email</td>
          <td style="padding: 8px 0;"><a href="mailto:{e_email}">{e_email}</a></td>
        </tr>
      </table>
      <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 16px 0;" />
      <p style="white-space: pre-wrap; color: #374151;">{e_message}</p>
    </div>
    """

    resend.Emails.send(
        {
            "from": "onboarding@resend.dev",
            "to": [to_email.strip()],
            "subject": subject,
            "html": html_body,
            "reply_to": [email],
        }
    )
