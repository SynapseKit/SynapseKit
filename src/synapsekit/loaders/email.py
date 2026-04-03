"""EmailLoader — load emails from IMAP servers (Gmail, etc.) via IMAP."""

from __future__ import annotations

import asyncio
import contextlib
import email
import imaplib
from email.message import Message

from .base import Document


class EmailLoader:
    """Load emails from an IMAP mailbox into Documents.

    Uses only stdlib — no extra dependencies required.

    Example::

        loader = EmailLoader(
            imap_server="imap.gmail.com",
            email_address="user@gmail.com",
            password="app_password",
            folder="INBOX",
            search='SINCE "01-Jan-2024"',
            limit=10,
        )
        docs = loader.load()
        docs = await loader.aload()
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
        self._imap_server = imap_server
        self._email_address = email_address
        self._password = password
        self._folder = folder
        self._search = search
        self._limit = limit

    def load(self) -> list[Document]:
        """Connect to the IMAP server and return emails as Documents."""
        mail = self._connect()
        try:
            mail.select(self._folder)
            email_ids = self._search_ids(mail)
            docs = []
            for email_id in email_ids:
                doc = self._fetch_email(mail, email_id)
                if doc:
                    docs.append(doc)
            return docs
        finally:
            with contextlib.suppress(Exception):
                mail.logout()

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _connect(self) -> imaplib.IMAP4_SSL:
        mail = imaplib.IMAP4_SSL(self._imap_server)
        mail.login(self._email_address, self._password)
        return mail

    def _search_ids(self, mail: imaplib.IMAP4_SSL) -> list[bytes]:
        status, messages = mail.search(None, self._search)
        if status != "OK" or not messages[0]:
            return []

        email_ids: list[bytes] = messages[0].split()
        if self._limit is not None:
            email_ids = email_ids[-self._limit :]
        return email_ids

    def _fetch_email(self, mail: imaplib.IMAP4_SSL, email_id: bytes) -> Document | None:
        status, msg_data = mail.fetch(email_id.decode(), "(RFC822)")
        if status != "OK" or not msg_data or not msg_data[0]:
            return None

        raw_email = msg_data[0][1]
        if not isinstance(raw_email, bytes):
            return None

        msg = email.message_from_bytes(raw_email)

        return Document(
            text=self._extract_body(msg),
            metadata={
                "source": "email",
                "subject": msg.get("Subject", ""),
                "from": msg.get("From", ""),
                "date": msg.get("Date", ""),
                "folder": self._folder,
                "email_id": email_id.decode(errors="ignore"),
            },
        )

    def _extract_body(self, msg: Message) -> str:
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload and isinstance(payload, bytes):
                        body = payload.decode(errors="ignore")
                        break
        else:
            payload = msg.get_payload(decode=True)
            if payload and isinstance(payload, bytes):
                body = payload.decode(errors="ignore")
        return body.strip()
