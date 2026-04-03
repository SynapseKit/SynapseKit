"""EmailLoader — load emails from IMAP servers (Gmail, etc.) via IMAP."""

from __future__ import annotations

import contextlib
import email
import imaplib
from email.message import Message

from .base import Document


class EmailLoader:
    """Load emails from IMAP servers (Gmail, etc.) into Documents.

    Example:
        >>> loader = EmailLoader(
        ...     imap_server="imap.gmail.com",
        ...     email_address="user@gmail.com",
        ...     password="app_password",
        ...     folder="INBOX",
        ...     search='SINCE "01-Jan-2024"',
        ...     limit=10
        ... )
        >>> docs = loader.load()
    """

    def __init__(
        self,
        imap_server: str,
        email_address: str,
        password: str,
        folder: str = "INBOX",
        search: str = "ALL",
        limit: int | None = None,
    ) -> None:
        self.imap_server = imap_server
        self.email_address = email_address
        self.password = password
        self.folder = folder
        self.search = search
        self.limit = limit

    def load(self) -> list[Document]:
        """Connect to IMAP server and load emails as Documents."""
        mail = self._connect()
        try:
            mail.select(self.folder)
            email_ids = self._search(mail)
            docs = []
            for email_id in email_ids:
                doc = self._fetch_email(mail, email_id)
                if doc:
                    docs.append(doc)
            return docs
        finally:
            with contextlib.suppress(Exception):
                mail.logout()

    def _connect(self) -> imaplib.IMAP4_SSL:
        """Connect and login to IMAP server."""
        mail = imaplib.IMAP4_SSL(self.imap_server)
        mail.login(self.email_address, self.password)
        return mail

    def _search(self, mail: imaplib.IMAP4_SSL) -> list[bytes]:
        """Search for emails matching query and return IDs."""
        status, messages = mail.search(None, self.search)
        if status != "OK" or not messages[0]:
            return []

        email_ids: list[bytes] = messages[0].split()
        if self.limit is not None:
            email_ids = email_ids[-self.limit :]
        return email_ids

    def _fetch_email(self, mail: imaplib.IMAP4_SSL, email_id: bytes) -> Document | None:
        """Fetch and parse a single email into a Document."""
        status, msg_data = mail.fetch(email_id.decode(), "(RFC822)")
        if status != "OK" or not msg_data or not msg_data[0]:
            return None

        raw_email = msg_data[0][1]
        if not isinstance(raw_email, bytes):
            return None

        msg = email.message_from_bytes(raw_email)

        subject = msg.get("Subject", "")
        sender = msg.get("From", "")
        date = msg.get("Date", "")
        body = self._extract_body(msg)

        return Document(
            text=body,
            metadata={
                "source": "email",
                "subject": subject,
                "from": sender,
                "date": date,
                "folder": self.folder,
                "email_id": email_id.decode(errors="ignore"),
            },
        )

    def _extract_body(self, msg: Message) -> str:
        """Extract plain text body from email message."""
        body = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload and isinstance(payload, bytes):
                        body = payload.decode(errors="ignore")
                        break
        else:
            payload = msg.get_payload(decode=True)
            if payload and isinstance(payload, bytes):
                body = payload.decode(errors="ignore")

        return body.strip()
