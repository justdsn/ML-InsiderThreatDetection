import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from typing import Optional
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(file_path: str, chunksize: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Load CSV file with error handling, supporting chunked reading."""
    try:
        if chunksize:
            chunks = []
            for chunk in tqdm(pd.read_csv(file_path, chunksize=chunksize), desc=f"Loading {file_path}"):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(file_path)
        logger.info(f"Loaded {file_path} with shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File {file_path} not found.")
        return None
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return None


def preprocess_email() -> pd.DataFrame:
    """Preprocess email data."""
    email = load_data("email.csv", chunksize=100000)
    if email is None:
        raise FileNotFoundError("email.csv is required.")

    # Log columns and sample user_id values
    logger.info(f"Columns in email.csv: {list(email.columns)}")
    logger.info(f"Sample user_id values (user column): {email['user'].unique()[:5].tolist()}")
    logger.info(f"Total unique user_id values: {email['user'].nunique()}")

    # Clean and rename columns
    email = email.rename(columns={"id": "email_id", "user": "user_id"})
    email["user_id"] = email["user_id"].str.strip().str.upper()

    # Ensure required columns exist
    required_cols = ["user_id", "date", "pc", "size", "attachments"]
    if not all(col in email.columns for col in required_cols):
        logger.error(f"email.csv must contain columns: {required_cols}")
        raise ValueError("Missing required columns in email.csv")

    # Parse dates
    try:
        email["date"] = pd.to_datetime(email["date"])
    except Exception as e:
        logger.error(f"Failed to parse 'date' column: {str(e)}")
        raise

    email = email.dropna(subset=["user_id", "date", "pc"])
    logger.info(f"Preprocessed email shape: {email.shape}")
    return email


def feature_engineering(email: pd.DataFrame) -> pd.DataFrame:
    """Engineer email-based features."""
    features = email.groupby("user_id").agg({
        "email_id": "count",
        "size": "mean",
        "attachments": "sum"
    }).reset_index()
    features.columns = ["user_id", "email_count", "avg_email_size", "total_attachments"]

    # Handle missing values
    for col in ["avg_email_size", "total_attachments"]:
        if col not in features.columns or features[col].isna().all():
            features[col] = 0
            logger.warning(f"{col} column not found or all NaN; using placeholder value 0.")

    features = features.fillna(features.mean(numeric_only=True))
    logger.info(f"Engineered features shape: {features.shape}")
    return features


def anomaly_detection(features: pd.DataFrame) -> pd.DataFrame:
    """Perform anomaly detection using Isolation Forest."""
    X = features[["email_count", "avg_email_size", "total_attachments"]]
    forest = IsolationForest(
        bootstrap=False, contamination=0.1, max_features=1.0,
        max_samples="auto", n_estimators=100, n_jobs=-1, random_state=42
    )
    forest.fit(X)
    ascore = forest.decision_function(X)
    y_pred = forest.predict(X)

    result = features[["user_id"]].copy()
    result["email_count"] = X["email_count"]
    result["avg_email_size"] = X["avg_email_size"]
    result["total_attachments"] = X["total_attachments"]
    result["ascore"] = ascore
    result["Anomaly"] = np.where(y_pred == -1, -1, 1)
    logger.info(f"Detected {sum(result['Anomaly'] == -1)} anomalies")
    return result


def plot_anomalies_scatter(result: pd.DataFrame) -> None:
    """Generate scatterplot of anomalies."""
    try:
        sns.set(rc={"figure.figsize": (12, 10)})
        plot = sns.scatterplot(
            data=result,
            x="email_count",
            y="avg_email_size",
            s=125,
            hue="Anomaly",
            palette=["red", "green"]
        )
        plt.xlabel("Email Frequency (Count)", fontsize=18)
        plt.ylabel("Average Email Size", fontsize=18)
        plt.legend(bbox_to_anchor=(1.01, 0.5), borderaxespad=0, title="Anomaly")
        plt.title("Anomaly Scatterplot (Email Features)", fontsize=20)
        plt.tight_layout()
        plt.savefig("anomaly_scatterplot_email.png")
        logger.info("Saved scatterplot to anomaly_scatterplot_email.png")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot scatterplot: {str(e)}")
        plt.close()


def plot_email_timeseries(email: pd.DataFrame) -> None:
    """Generate time-series plot of email activity."""
    try:
        # Aggregate email count by user and week
        email["week"] = email["date"].dt.to_period("W").astype(str)
        timeseries_data = email.groupby(["user_id", "week"]).size().reset_index(name="email_count")

        # Select top 5 users by total email count
        top_users = timeseries_data.groupby("user_id")["email_count"].sum().nlargest(5).index
        timeseries_data = timeseries_data[timeseries_data["user_id"].isin(top_users)]

        plt.figure(figsize=(15, 8))
        sns.lineplot(data=timeseries_data, x="week", y="email_count", hue="user_id", marker="o")
        plt.title("Weekly Email Activity for Top 5 Users", fontsize=20)
        plt.xlabel("Week", fontsize=15)
        plt.ylabel("Email Count", fontsize=15)
        plt.xticks(rotation=45, fontsize=10)
        plt.tight_layout()
        plt.savefig("email_timeseries.png")
        logger.info("Saved time-series plot to email_timeseries.png")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot time-series: {str(e)}")
        plt.close()


def main():
    """Main function for core insider threat detection."""
    try:
        # Preprocess email data
        email = preprocess_email()

        # Feature engineering
        features = feature_engineering(email)

        # Anomaly detection
        result = anomaly_detection(features)

        # Plot anomalies
        plot_anomalies_scatter(result)
        plot_email_timeseries(email)

        # Save results
        result.to_csv("anomaly_results_email_core.csv", index=False)
        logger.info("Saved results to anomaly_results_email_core.csv")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()