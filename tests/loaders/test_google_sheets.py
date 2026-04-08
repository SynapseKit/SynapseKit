from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders.google_sheets import GoogleSheetsLoader


def test_import_error_missing_api_client():
    with patch.dict("sys.modules", {"googleapiclient": None, "google.auth": MagicMock()}):
        loader = GoogleSheetsLoader("123", credentials_path="dummy.json")
        with pytest.raises(ImportError, match="google-api-python-client required"):
            loader.load()


def test_import_error_missing_auth():
    with patch.dict("sys.modules", {"google.auth": None, "googleapiclient.discovery": MagicMock()}):
        loader = GoogleSheetsLoader("123", credentials_path="dummy.json")
        with pytest.raises(ImportError, match="google-auth required"):
            loader.load()


def test_credentials_file_not_found():
    pytest.importorskip("google.oauth2.service_account")
    loader = GoogleSheetsLoader("123", credentials_path="/path/that/does/not/exist.json")
    with pytest.raises(FileNotFoundError, match="Credentials file not found"):
        loader._get_credentials()


@patch("os.path.exists", return_value=True)
def test_load_sheet_data(mock_exists):
    pytest.importorskip("googleapiclient")
    pytest.importorskip("google.oauth2.service_account")

    with (
        patch("google.oauth2.service_account.Credentials.from_service_account_file"),
        patch("googleapiclient.discovery.build") as mock_build,
    ):
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        mock_sheets_api = mock_service.spreadsheets.return_value

        mock_values_get = mock_sheets_api.values.return_value.get.return_value
        mock_values_get.execute.return_value = {
            "values": [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]
        }

        loader = GoogleSheetsLoader("test_id", sheet_name="Sheet1")
        docs = loader.load()

        assert len(docs) == 2
        assert docs[0].text == "Name: Alice, Age: 30"
        assert docs[0].metadata["row"] == 2
        assert docs[0].metadata["sheet"] == "Sheet1"
        assert docs[1].text == "Name: Bob, Age: 25"
        assert docs[1].metadata["row"] == 3


@patch("os.path.exists", return_value=True)
def test_load_default_sheet(mock_exists):
    pytest.importorskip("googleapiclient")
    pytest.importorskip("google.oauth2.service_account")

    with (
        patch("google.oauth2.service_account.Credentials.from_service_account_file"),
        patch("googleapiclient.discovery.build") as mock_build,
    ):
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        mock_sheets_api = mock_service.spreadsheets.return_value

        mock_get = mock_sheets_api.get.return_value
        mock_get.execute.return_value = {"sheets": [{"properties": {"title": "DefaultSheet"}}]}

        mock_values_get = mock_sheets_api.values.return_value.get.return_value
        mock_values_get.execute.return_value = {"values": [["Col1"], ["Val1"]]}

        loader = GoogleSheetsLoader("test_id")
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].metadata["sheet"] == "DefaultSheet"


@patch("os.path.exists", return_value=True)
def test_load_empty_sheet(mock_exists):
    pytest.importorskip("googleapiclient")
    pytest.importorskip("google.oauth2.service_account")

    with (
        patch("google.oauth2.service_account.Credentials.from_service_account_file"),
        patch("googleapiclient.discovery.build") as mock_build,
    ):
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        mock_sheets_api = mock_service.spreadsheets.return_value

        mock_values_get = mock_sheets_api.values.return_value.get.return_value
        mock_values_get.execute.return_value = {"values": []}

        loader = GoogleSheetsLoader("test_id", sheet_name="Empty")
        docs = loader.load()

        assert len(docs) == 0
