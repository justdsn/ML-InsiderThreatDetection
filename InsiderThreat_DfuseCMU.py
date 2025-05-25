import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import logging
from typing import Optional, Tuple
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


def preprocess_psychometric() -> pd.DataFrame:
    """Preprocess psychometric data."""
    psychometric = load_data("psychometric.csv")
    if psychometric is None:
        raise FileNotFoundError("psychometric.csv is required.")

    # Log columns and sample user_id values
    logger.info(f"Columns in psychometric.csv: {list(psychometric.columns)}")
    logger.info(f"Sample user_id values: {psychometric['user_id'].unique()[:5].tolist()}")
    logger.info(f"Total unique user_id values: {psychometric['user_id'].nunique()}")

    # Clean user_id
    psychometric["user_id"] = psychometric["user_id"].str.strip().str.upper()

    # Ensure required columns exist
    required_cols = ["user_id", "employee_name", "O", "C", "E", "A", "N"]
    if not all(col in psychometric.columns for col in required_cols):
        logger.error(f"psychometric.csv must contain columns: {required_cols}")
        raise ValueError("Missing required columns in psychometric.csv")

    psychometric = psychometric[required_cols]
    psychometric = psychometric.dropna()
    logger.info(f"Preprocessed psychometric shape: {psychometric.shape}")
    return psychometric


def check_user_id_overlap(email: pd.DataFrame, psychometric: pd.DataFrame) -> None:
    """Check for overlapping user_id values between email and psychometric data."""
    email_users = set(email["user_id"].unique())
    psych_users = set(psychometric["user_id"].unique())
    common_users = email_users.intersection(psych_users)
    logger.info(f"Number of common user_id values: {len(common_users)}")
    logger.info(f"Sample common user_id values (first 5): {list(common_users)[:5]}")
    if not common_users:
        logger.warning("No overlapping user_id values found. Merge will be empty.")


def create_bipartite_graph(email: pd.DataFrame) -> nx.Graph:
    """Create a bipartite graph of user-PC interactions."""
    try:
        graph_data = email.groupby(["user_id", "pc"]).size().reset_index(name="weight")

        users_nodes = list(graph_data["user_id"].unique())
        pc_nodes = list(graph_data["pc"].unique())
        weighted_edges = [(row["user_id"], row["pc"], row["weight"]) for _, row in graph_data.iterrows()]

        graph = nx.Graph()
        graph.add_nodes_from(users_nodes, bipartite=0)  # Users
        graph.add_nodes_from(pc_nodes, bipartite=1)  # PCs
        graph.add_weighted_edges_from(weighted_edges)

        logger.info(f"Bipartite graph created with {len(users_nodes)} users and {len(pc_nodes)} PCs")
        return graph
    except Exception as e:
        logger.error(f"Failed to create bipartite graph: {str(e)}")
        return nx.Graph()


def plot_bipartite_graph(graph: nx.Graph) -> None:
    """Plot the bipartite graph with specified styling."""
    try:
        plt.figure(figsize=(20, 20))  # Reduced size for stability
        nx.draw_networkx(
            graph,
            node_size=500,
            font_size=12,
            node_color="darkorange",
            font_color="darkblue"
        )
        plt.savefig("bipartite_user_pc_graph.png")
        logger.info("Saved bipartite graph to bipartite_user_pc_graph.png")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot bipartite graph: {str(e)}")
        plt.close()


def merge_datasets(email: pd.DataFrame, psychometric: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Merge email and psychometric data on user_id."""
    try:
        merged = pd.merge(
            email,
            psychometric[["user_id", "O", "C", "E", "A", "N"]],
            on="user_id",
            how="inner"
        )
        if merged.empty:
            logger.error("Merged dataset is empty. Check user_id values for mismatches.")
            logger.error("Run check_user_ids.py to compare user_id values.")
            return None
        logger.info(f"Merged dataset shape: {merged.shape}")
        return merged
    except Exception as e:
        logger.error(f"Merge failed: {str(e)}")
        return None


def feature_engineering(email: pd.DataFrame, psychometric: Optional[pd.DataFrame], graph: nx.Graph) -> pd.DataFrame:
    """Engineer features from email, psychometric, and graph data."""
    # Email features
    email_agg = email.groupby("user_id").agg({
        "email_id": "count",
        "size": "mean",
        "attachments": "sum"
    }).reset_index()
    email_agg.columns = ["user_id", "email_count", "avg_email_size", "total_attachments"]

    # Initialize features DataFrame
    features = email_agg.copy()

    # Add psychometric features if available
    if psychometric is not None:
        psych_cols = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        psych_agg = psychometric[["user_id", "O", "C", "E", "A", "N"]].rename(columns={
            "O": "openness", "C": "conscientiousness", "E": "extraversion",
            "A": "agreeableness", "N": "neuroticism"
        })
        features = pd.merge(features, psych_agg, on="user_id", how="left")
    else:
        for col in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            features[col] = 0
        logger.warning("No psychometric data available; setting psychometric features to 0.")

    # Add graph-based feature (user degree)
    degrees = dict(graph.degree())
    features["graph_degree"] = features["user_id"].map(degrees).fillna(0)

    # Handle missing values
    for col in ["avg_email_size", "total_attachments", "graph_degree"]:
        if col not in features.columns or features[col].isna().all():
            features[col] = 0
            logger.warning(f"{col} column not found or all NaN; using placeholder value 0.")

    features = features.fillna(features.mean(numeric_only=True))
    logger.info(f"Engineered features shape: {features.shape}")
    return features


def anomaly_detection(features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform anomaly detection for different feature sets."""
    # Email-based features
    X_email = features[["email_count", "avg_email_size", "total_attachments"]]
    forest_email = IsolationForest(
        bootstrap=False, contamination=0.1, max_features=1.0,
        max_samples="auto", n_estimators=100, n_jobs=-1, random_state=42
    )
    forest_email.fit(X_email)
    ascore_email = forest_email.decision_function(X_email)
    y_pred_email = forest_email.predict(X_email)

    # Psychometric-based features
    X_psych = features[["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]]
    forest_psych = IsolationForest(
        bootstrap=False, contamination=0.1, max_features=1.0,
        max_samples="auto", n_estimators=100, n_jobs=-1, random_state=42
    )
    forest_psych.fit(X_psych)
    ascore_psych = forest_psych.decision_function(X_psych)
    y_pred_psych = forest_psych.predict(X_psych)

    # Graph-based features
    X_graph = features[["graph_degree"]]
    forest_graph = IsolationForest(
        bootstrap=False, contamination=0.1, max_features=1.0,
        max_samples="auto", n_estimators=100, n_jobs=-1, random_state=42
    )
    forest_graph.fit(X_graph)
    ascore_graph = forest_graph.decision_function(X_graph)
    y_pred_graph = forest_graph.predict(X_graph)

    # Combine results
    result_email = features[["user_id"]].copy()
    result_email["email_count"] = features["email_count"]
    result_email["neuroticism"] = features["neuroticism"]
    result_email["ascore"] = ascore_email
    result_email["Anomaly"] = np.where(y_pred_email == -1, -1, 1)
    result_email["Feature_Set"] = "Email"

    result_psych = features[["user_id"]].copy()
    result_psych["ascore"] = ascore_psych
    result_psych["Anomaly"] = np.where(y_pred_psych == -1, -1, 1)
    result_psych["Feature_Set"] = "Psychometric"

    result_graph = features[["user_id"]].copy()
    result_graph["ascore"] = ascore_graph
    result_graph["Anomaly"] = np.where(y_pred_graph == -1, -1, 1)
    result_graph["Feature_Set"] = "Graph"

    logger.info(f"Detected anomalies: Email={sum(result_email['Anomaly'] == -1)}, "
                f"Psychometric={sum(result_psych['Anomaly'] == -1)}, "
                f"Graph={sum(result_graph['Anomaly'] == -1)}")
    return result_email, result_psych, result_graph


def plot_anomalies_pointplot(result_email: pd.DataFrame, result_psych: pd.DataFrame,
                             result_graph: pd.DataFrame) -> None:
    """Generate multi-line point plot of anomaly scores."""
    try:
        # Combine results
        all_results = pd.concat([result_email, result_psych, result_graph], ignore_index=True)

        # Limit to top 20 users by average ascore to avoid clutter
        top_users = all_results.groupby("user_id")["ascore"].mean().nlargest(20).index
        all_results = all_results[all_results["user_id"].isin(top_users)]

        plt.figure(figsize=(15, 8))
        sns.set_theme(style="darkgrid")

        # Plot each feature set
        sns.pointplot(x="user_id", y="ascore", data=all_results[all_results["Feature_Set"] == "Email"],
                      color="purple", label="Email")
        sns.pointplot(x="user_id", y="ascore", data=all_results[all_results["Feature_Set"] == "Psychometric"],
                      color="brown", label="Psychometric")
        sns.pointplot(x="user_id", y="ascore", data=all_results[all_results["Feature_Set"] == "Graph"],
                      color="darkorange", label="Graph")

        plt.legend(title="Feature Set")
        plt.axhline(0, ls="-")
        plt.title("Anomaly Score for Different Feature Sets", size=20)
        plt.xlabel("User ID", fontsize=15)
        plt.ylabel("Anomaly Score", fontsize=15)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig("anomaly_pointplot_feature_sets.png")
        logger.info("Saved point plot to anomaly_pointplot_feature_sets.png")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot point plot: {str(e)}")
        plt.close()


def plot_anomalies_scatter(result_email: pd.DataFrame, psychometric_available: bool) -> None:
    """Generate scatterplot of anomalies (email features)."""
    try:
        sns.set(rc={"figure.figsize": (12, 10)})
        if psychometric_available and result_email["neuroticism"].var() > 0:
            plot = sns.scatterplot(
                data=result_email,
                x="email_count",
                y="neuroticism",
                s=125,
                hue="Anomaly",
                palette=["red", "green"]
            )
            plt.ylabel("Neuroticism Score", fontsize=18)
        else:
            plot = sns.scatterplot(
                data=result_email,
                x="email_count",
                y="avg_email_size",
                s=125,
                hue="Anomaly",
                palette=["red", "green"]
            )
            plt.ylabel("Average Email Size", fontsize=18)
        plt.xlabel("Email Frequency (Count)", fontsize=18)
        plt.legend(bbox_to_anchor=(1.01, 0.5), borderaxespad=0, title="Anomaly")
        plt.title("Anomaly Scatterplot", fontsize=20)
        plt.tight_layout()
        plt.savefig("anomaly_scatterplot_email_psychometric.png")
        logger.info("Saved scatterplot to anomaly_scatterplot_email_psychometric.png")
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


def plot_feature_heatmap(features: pd.DataFrame) -> None:
    """Generate heatmap of feature correlations."""
    try:
        correlation_matrix = features[["email_count", "avg_email_size", "total_attachments",
                                       "openness", "conscientiousness", "extraversion",
                                       "agreeableness", retened, "neuroticism", "graph_degree"]].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap", fontsize=20)
        plt.tight_layout()
        plt.savefig("feature_correlation_heatmap.png")
        logger.info("Saved heatmap to feature_correlation_heatmap.png")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot heatmap: {str(e)}")
        plt.close()


def plot_anomaly_boxplot(result_email: pd.DataFrame, result_psych: pd.DataFrame, result_graph: pd.DataFrame) -> None:
    """Generate box plot of anomaly scores by feature set."""
    try:
        all_results = pd.concat([result_email, result_psych, result_graph], ignore_index=True)

        plt.figure(figsize=(10, 8))
        sns.boxplot(x="Feature_Set", y="ascore", data=all_results)
        plt.title("Distribution of Anomaly Scores by Feature Set", fontsize=20)
        plt.xlabel("Feature Set", fontsize=15)
        plt.ylabel("Anomaly Score", fontsize=15)
        plt.tight_layout()
        plt.savefig("anomaly_boxplot.png")
        logger.info("Saved box plot to anomaly_boxplot.png")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot box plot: {str(e)}")
        plt.close()


def main():
    """Main function to run the insider threat detection pipeline."""
    try:
        # Preprocess datasets
        email = preprocess_email()
        psychometric = preprocess_psychometric()

        # Check user_id overlap
        check_user_id_overlap(email, psychometric)

        # Create and plot bipartite graph
        graph = create_bipartite_graph(email)
        plot_bipartite_graph(graph)

        # Merge datasets
        merged = merge_datasets(email, psychometric)
        psychometric_available = merged is not None

        # Feature engineering
        features = feature_engineering(email, psychometric if psychometric_available else None, graph)

        # Anomaly detection
        result_email, result_psych, result_graph = anomaly_detection(features)

        # Plot anomalies
        plot_anomalies_pointplot(result_email, result_psych, result_graph)
        plot_anomalies_scatter(result_email, psychometric_available)
        plot_email_timeseries(email)
        plot_feature_heatmap(features)
        plot_anomaly_boxplot(result_email, result_psych, result_graph)

        # Save results
        result_email.to_csv("anomaly_results_email.csv", index=False)
        result_psych.to_csv("anomaly_results_psychometric.csv", index=False)
        result_graph.to_csv("anomaly_results_graph.csv", index=False)
        logger.info("Saved anomaly results to CSV files")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()