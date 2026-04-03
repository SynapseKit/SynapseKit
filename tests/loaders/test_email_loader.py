"""Tests for EmailLoader."""

from __future__ import annotations

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders.email import EmailLoader


class TestEmailLoader:
    """Test EmailLoader with mocked IMAP connection."""

    def test_basic_email_load(self):
        """Test loading a simple plain text email."""
        # Create a simple email
        msg = MIMEText("This is the email body")
        msg["Subject"] = "Test Subject"
        msg["From"] = "sender@example.com"
        msg["Date"] = "Mon, 1 Jan 2024 12:00:00 +0000"

        # Mock IMAP connection
        mock_mail = MagicMock()
        mock_mail.select.return_value = ("OK", [])
        mock_mail.search.return_value = ("OK", [b"1"])
        mock_mail.fetch.return_value = ("OK", [(None, msg.as_bytes())])
        mock_mail.logout.return_value = ("BYE", [])

        with patch("imaplib.IMAP4_SSL") as mock_imap:
            mock_imap.return_value = mock_mail

            loader = EmailLoader(
                imap_server="imap.gmail.com",
                email_address="test@example.com",
                password="password123",
            )
            docs = loader.load()

        assert len(docs) == 1
        assert docs[0].text == "This is the email body"
        assert docs[0].metadata["subject"] == "Test Subject"
        assert docs[0].metadata["from"] == "sender@example.com"
        assert docs[0].metadata["source"] == "email"
        assert docs[0].metadata["folder"] == "INBOX"
        assert "email_id" in docs[0].metadata

    def test_multipart_email_extracts_plain_text(self):
        """Test extracting plain text from multipart email."""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Multipart Test"
        msg["From"] = "sender@example.com"
        msg["Date"] = "Mon, 1 Jan 2024 12:00:00 +0000"

        # Add plain text and HTML parts
        text_part = MIMEText("Plain text body", "plain")
        html_part = MIMEText("<html><body>HTML body</body></html>", "html")
        msg.attach(text_part)
        msg.attach(html_part)

        mock_mail = MagicMock()
        mock_mail.select.return_value = ("OK", [])
        mock_mail.search.return_value = ("OK", [b"1"])
        mock_mail.fetch.return_value = ("OK", [(None, msg.as_bytes())])
        mock_mail.logout.return_value = ("BYE", [])

        with patch("imaplib.IMAP4_SSL") as mock_imap:
            mock_imap.return_value = mock_mail

            loader = EmailLoader(
                imap_server="imap.gmail.com",
                email_address="test@example.com",
                password="password123",
            )
            docs = loader.load()

        assert len(docs) == 1
        assert docs[0].text == "Plain text body"
        assert "<html>" not in docs[0].text

    def test_multiple_emails(self):
        """Test loading multiple emails."""
        msg1 = MIMEText("Email 1 body")
        msg1["Subject"] = "Subject 1"
        msg1["From"] = "sender1@example.com"
        msg1["Date"] = "Mon, 1 Jan 2024 12:00:00 +0000"

        msg2 = MIMEText("Email 2 body")
        msg2["Subject"] = "Subject 2"
        msg2["From"] = "sender2@example.com"
        msg2["Date"] = "Tue, 2 Jan 2024 12:00:00 +0000"

        mock_mail = MagicMock()
        mock_mail.select.return_value = ("OK", [])
        mock_mail.search.return_value = ("OK", [b"1 2"])
        mock_mail.fetch.side_effect = [
            ("OK", [(None, msg1.as_bytes())]),
            ("OK", [(None, msg2.as_bytes())]),
        ]
        mock_mail.logout.return_value = ("BYE", [])

        with patch("imaplib.IMAP4_SSL") as mock_imap:
            mock_imap.return_value = mock_mail

            loader = EmailLoader(
                imap_server="imap.gmail.com",
                email_address="test@example.com",
                password="password123",
            )
            docs = loader.load()

        assert len(docs) == 2
        assert docs[0].text == "Email 1 body"
        assert docs[1].text == "Email 2 body"
        assert docs[0].metadata["subject"] == "Subject 1"
        assert docs[1].metadata["subject"] == "Subject 2"

    def test_empty_mailbox(self):
        """Test loading from empty mailbox."""
        mock_mail = MagicMock()
        mock_mail.select.return_value = ("OK", [])
        mock_mail.search.return_value = ("OK", [b""])
        mock_mail.logout.return_value = ("BYE", [])

        with patch("imaplib.IMAP4_SSL") as mock_imap:
            mock_imap.return_value = mock_mail

            loader = EmailLoader(
                imap_server="imap.gmail.com",
                email_address="test@example.com",
                password="password123",
            )
            docs = loader.load()

        assert docs == []

    def test_custom_folder(self):
        """Test loading from custom folder."""
        msg = MIMEText("Test body")
        msg["Subject"] = "Test"
        msg["From"] = "sender@example.com"
        msg["Date"] = "Mon, 1 Jan 2024 12:00:00 +0000"

        mock_mail = MagicMock()
        mock_mail.select.return_value = ("OK", [])
        mock_mail.search.return_value = ("OK", [b"1"])
        mock_mail.fetch.return_value = ("OK", [(None, msg.as_bytes())])
        mock_mail.logout.return_value = ("BYE", [])

        with patch("imaplib.IMAP4_SSL") as mock_imap:
            mock_imap.return_value = mock_mail

            loader = EmailLoader(
                imap_server="imap.gmail.com",
                email_address="test@example.com",
                password="password123",
                folder="Sent",
            )
            docs = loader.load()

        mock_mail.select.assert_called_once_with("Sent")
        assert docs[0].metadata["folder"] == "Sent"

    def test_imap_search_query(self):
        """Test custom IMAP search query."""
        msg = MIMEText("Test body")
        msg["Subject"] = "Test"
        msg["From"] = "sender@example.com"
        msg["Date"] = "Mon, 1 Jan 2024 12:00:00 +0000"

        mock_mail = MagicMock()
        mock_mail.select.return_value = ("OK", [])
        mock_mail.search.return_value = ("OK", [b"1"])
        mock_mail.fetch.return_value = ("OK", [(None, msg.as_bytes())])
        mock_mail.logout.return_value = ("BYE", [])

        with patch("imaplib.IMAP4_SSL") as mock_imap:
            mock_imap.return_value = mock_mail

            loader = EmailLoader(
                imap_server="imap.gmail.com",
                email_address="test@example.com",
                password="password123",
                search='SINCE "01-Jan-2024"',
            )
            docs = loader.load()

        mock_mail.search.assert_called_once_with(None, 'SINCE "01-Jan-2024"')
        assert len(docs) == 1

    def test_limit_parameter(self):
        """Test limiting number of emails loaded."""
        mock_mail = MagicMock()
        mock_mail.select.return_value = ("OK", [])
        # Return 5 email IDs
        mock_mail.search.return_value = ("OK", [b"1 2 3 4 5"])
        mock_mail.logout.return_value = ("BYE", [])

        # Create simple message for fetch
        msg = MIMEText("Test body")
        msg["Subject"] = "Test"
        msg["From"] = "sender@example.com"
        msg["Date"] = "Mon, 1 Jan 2024 12:00:00 +0000"
        mock_mail.fetch.return_value = ("OK", [(None, msg.as_bytes())])

        with patch("imaplib.IMAP4_SSL") as mock_imap:
            mock_imap.return_value = mock_mail

            loader = EmailLoader(
                imap_server="imap.gmail.com",
                email_address="test@example.com",
                password="password123",
                limit=2,
            )
            docs = loader.load()

        # Should only fetch the last 2 emails (most recent)
        assert len(docs) == 2
        assert mock_mail.fetch.call_count == 2

    def test_missing_headers(self):
        """Test handling emails with missing headers."""
        msg = MIMEText("Body only")
        # Don't set any headers

        mock_mail = MagicMock()
        mock_mail.select.return_value = ("OK", [])
        mock_mail.search.return_value = ("OK", [b"1"])
        mock_mail.fetch.return_value = ("OK", [(None, msg.as_bytes())])
        mock_mail.logout.return_value = ("BYE", [])

        with patch("imaplib.IMAP4_SSL") as mock_imap:
            mock_imap.return_value = mock_mail

            loader = EmailLoader(
                imap_server="imap.gmail.com",
                email_address="test@example.com",
                password="password123",
            )
            docs = loader.load()

        assert len(docs) == 1
        assert docs[0].text == "Body only"
        assert docs[0].metadata["subject"] == ""
        assert docs[0].metadata["from"] == ""
        assert docs[0].metadata["date"] == ""

    def test_failed_search_returns_empty(self):
        """Test handling of failed search."""
        mock_mail = MagicMock()
        mock_mail.select.return_value = ("OK", [])
        mock_mail.search.return_value = ("NO", [])
        mock_mail.logout.return_value = ("BYE", [])

        with patch("imaplib.IMAP4_SSL") as mock_imap:
            mock_imap.return_value = mock_mail

            loader = EmailLoader(
                imap_server="imap.gmail.com",
                email_address="test@example.com",
                password="password123",
            )
            docs = loader.load()

        assert docs == []

    def test_failed_fetch_skips_email(self):
        """Test that failed fetch is handled gracefully."""
        msg = MIMEText("Valid email")
        msg["Subject"] = "Valid"
        msg["From"] = "sender@example.com"
        msg["Date"] = "Mon, 1 Jan 2024 12:00:00 +0000"

        mock_mail = MagicMock()
        mock_mail.select.return_value = ("OK", [])
        mock_mail.search.return_value = ("OK", [b"1 2"])
        # First fetch fails, second succeeds
        mock_mail.fetch.side_effect = [
            ("NO", []),
            ("OK", [(None, msg.as_bytes())]),
        ]
        mock_mail.logout.return_value = ("BYE", [])

        with patch("imaplib.IMAP4_SSL") as mock_imap:
            mock_imap.return_value = mock_mail

            loader = EmailLoader(
                imap_server="imap.gmail.com",
                email_address="test@example.com",
                password="password123",
            )
            docs = loader.load()

        # Should only get the successful one
        assert len(docs) == 1
        assert docs[0].metadata["subject"] == "Valid"

    def test_connection_parameters(self):
        """Test that connection uses correct parameters."""
        mock_mail = MagicMock()
        mock_mail.select.return_value = ("OK", [])
        mock_mail.search.return_value = ("OK", [b""])
        mock_mail.logout.return_value = ("BYE", [])

        with patch("imaplib.IMAP4_SSL") as mock_imap:
            mock_imap.return_value = mock_mail

            loader = EmailLoader(
                imap_server="imap.gmail.com",
                email_address="user@gmail.com",
                password="app_password",
            )
            loader.load()

        mock_imap.assert_called_once_with("imap.gmail.com")
        mock_mail.login.assert_called_once_with("user@gmail.com", "app_password")

    def test_logout_called_on_success(self):
        """Test that logout is called even on success."""
        mock_mail = MagicMock()
        mock_mail.select.return_value = ("OK", [])
        mock_mail.search.return_value = ("OK", [b""])
        mock_mail.logout.return_value = ("BYE", [])

        with patch("imaplib.IMAP4_SSL") as mock_imap:
            mock_imap.return_value = mock_mail

            loader = EmailLoader(
                imap_server="imap.gmail.com",
                email_address="test@example.com",
                password="password123",
            )
            loader.load()

        mock_mail.logout.assert_called_once()

    def test_logout_called_on_exception(self):
        """Test that logout is attempted even if an error occurs."""
        mock_mail = MagicMock()
        mock_mail.select.side_effect = Exception("Connection error")
        mock_mail.logout.return_value = ("BYE", [])

        with patch("imaplib.IMAP4_SSL") as mock_imap:
            mock_imap.return_value = mock_mail

            loader = EmailLoader(
                imap_server="imap.gmail.com",
                email_address="test@example.com",
                password="password123",
            )

            with pytest.raises(Exception, match="Connection error"):
                loader.load()

        # Logout should still be called
        mock_mail.logout.assert_called_once()
