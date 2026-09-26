```python
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path
import pandas as pd
import numpy as np

# Import the module under test
from backend.agents.data_scientist import DataScientistAgent, ModelMetrics, EDAReport


@pytest.fixture
def agent(tmp_path):
    """Fixture to create a DataScientistAgent instance with a temporary artifacts directory."""
    return DataScientistAgent(artifacts_dir=tmp_path, random_state=42, test_size=0.2)


@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample DataFrame mimicking the HR dataset structure."""
    data = {
        "enrollee_id": [1, 2, 3, 4, 5],
        "city": ["city_1", "city_2", "city_1", "city_3", "city_2"],
        "city_development_index": [0.9, 0.8, 0.9, 0.7, 0.8],
        "gender": ["Male", "Female", "Male", "Other", "Female"],
        "relevent_experience": ["Has relevent experience", "No relevent experience", "Has relevent experience", "Has relevent experience", "No relevent experience"],
        "enrolled_university": ["Full time course", "Part time course", "no_enrollment", "Full time course", "Part time course"],
        "education_level": ["Graduate", "Masters", "High School", "Graduate", "Phd"],
        "major_discipline": ["STEM", "Business Degree", "Arts", "STEM", "STEM"],
        "experience": [">20", "10", "5", "3", "1"],
        "company_size": ["10000+", "50-99", "100-500", "1000-4999", "10-49"],
        "company_type": ["Pvt Ltd", "Public Sector", "Early Stage Startup", "Pvt Ltd", "NGO"],
        "last_new_job": [">4", "3", "2", "1", "never"],
        "training_hours": [100, 80, 60, 40, 20],
        "target": [0, 1, 0, 1, 0],
    }
    return pd.DataFrame(data)


class TestDataScientistAgentInitialization:
    """Tests for the __init__ method and default attributes."""

    def test_initialization_defaults(self, tmp_path):
        agent = DataScientistAgent(artifacts_dir=tmp_path)
        assert agent.artifacts_dir == tmp_path
        assert agent.random_state == 42
        assert agent.test_size == 0.2
        assert agent.data is None
        assert agent.X_train is None
        assert agent.X_test is None
        assert agent.y_train is None
        assert agent.y_test is None
        assert agent.preprocessor is None
        assert agent.models == {}
        assert agent.metrics == {}
        assert agent.best_model_name is None
        assert agent.eda_report is None
        assert tmp_path.exists()

    def test_initialization_custom_params(self, tmp_path):
        agent = DataScientistAgent(artifacts_dir=tmp_path, random_state=123, test_size=0.3)
        assert agent.random_state == 123
        assert agent.test_size == 0.3


class TestDownloadData:
    """Tests for the download_data method."""

    @patch("backend.agents.data_scientist.kagglehub.dataset_download")
    def test_download_data_success(self, mock_kagglehub, agent):
        mock_path = "/fake/kaggle/path"
        mock_kagglehub.return_value = mock_path

        result = agent.download_data()

        mock_kagglehub.assert_called_once_with(DataScientistAgent.DATASET_NAME)
        assert result == Path(mock_path)

    @patch("backend.agents.data_scientist.kagglehub.dataset_download")
    def test_download_data_failure(self, mock_kagglehub, agent):
        mock_kagglehub.side_effect = Exception("Network error")

        with pytest.raises(RuntimeError, match="Kagglehub download failed"):
            agent.download_data()


class TestLoadData:
    """Tests for the load_data method."""

    @patch.object(DataScientistAgent, "download_data")
    def test_load_data_auto_download(self, mock_download, agent, sample_dataframe, tmp_path):
        # Setup mock download directory with a CSV file
        mock_download_dir = tmp_path / "kaggle_download"
        mock_download_dir.mkdir()
        csv_file = mock_download_dir / "hr_data.csv"
        sample_dataframe.to_csv(csv_file, index=False)
        mock_download.return_value = mock_download_dir

        # Mock pd.read_csv to return our sample dataframe
        with patch("backend.agents.data_scientist.pd.read_csv", return_value=sample_dataframe) as mock_read_csv:
            df = agent.load_data()

        mock_download.assert_called_once()
        mock_read_csv.assert_called_once_with(csv_file)
        assert df.equals(sample_dataframe)
        assert agent.data is not None
        assert agent.data.equals(sample_dataframe)

    def test_load_data_explicit_path(self, agent, sample_dataframe, tmp_path):
        csv_file = tmp_path / "explicit_data.csv"
        sample_dataframe.to_csv(csv_file, index=False)

        with patch("backend.agents.data_scientist.pd.read_csv", return_value=sample_dataframe) as mock_read_csv:
            df = agent.load_data(data_path=csv_file)

        mock_read_csv.assert_called_once_with(csv_file)
        assert df.equals(sample_dataframe)

    @patch.object(DataScientistAgent, "download_data")
    def test_load_data_no_csv_found(self, mock_download, agent, tmp_path):
        mock_download_dir = tmp_path / "empty_dir"
        mock_download_dir.mkdir()
        mock_download.return_value = mock_download_dir

        with pytest.raises(FileNotFoundError, match="No CSV file found"):
            agent.load_data()


class TestPerformEDA:
    """Tests for the perform_eda method (partial implementation)."""

    def test_perform_eda_no_data_raises_error(self, agent):
        with pytest.raises(ValueError, match="Data not loaded. Call load_data\\(\\) first."):
            agent.perform_eda()

    @patch("backend.agents.data_scientist.logger")
    def test_perform_eda_basic_structure(self, mock_logger, agent, sample_dataframe):
        agent.data = sample_dataframe.copy()

        # Since the method is cut off in the provided code, we can only test
        # that it starts and returns an EDAReport-like object if completed.
        # Here we test the initial validation and logging.
        with pytest.raises(AttributeError):  # The method is incomplete in the snippet
            agent.perform_eda()

        mock_logger.info.assert_any_call("Starting Exploratory Data Analysis...")


class TestDataclasses:
    """Tests for the dataclass structures."""

    def test_model_metrics_creation(self):
        metrics = ModelMetrics(
            model_name="TestModel",
            accuracy=0.9,
            precision=0.85,
            recall=0.8,
            f1_score=0.82,
            roc_auc=0.95,
            cv_mean=0.88,