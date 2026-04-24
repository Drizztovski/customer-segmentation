"""
segmentation_engine.py — Customer Segmentation Engine
==================================================================================
Segments customers based on purchasing behavior using RFM analysis and K-means
clustering, then uses Google Gemini to generate persona descriptions.

Usage:
    from segmentation_engine import SegmentationEngine
    engine = SegmentationEngine('data/online_retail_sample.csv')
    engine.run_segmentation(n_clusters=4)
    engine.generate_personas()

API Key:
    Set GOOGLE_API_KEY in a .env file (see .env.example) or pass api_key= directly.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — that's fine, just use env vars directly


# ============================================================
# PART A: Data Loading & Cleaning
# ============================================================

def load_and_clean(filepath):
    """
    Load and clean the raw Online Retail transaction CSV.

    Drops rows missing a CustomerID, removes returns and zero-price entries,
    and adds a TotalPrice column (Quantity * UnitPrice).

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned transaction DataFrame.
    """
    df = pd.read_csv(filepath, encoding='latin-1')

    # Parse InvoiceDate as datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Drop rows with missing CustomerID
    df = df.dropna(subset=['CustomerID'])

    # Remove returns/adjustments (Quantity and UnitPrice must be positive)
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]

    # Add TotalPrice column
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    return df


# ============================================================
# PART B: RFM Feature Engineering
# ============================================================

def build_rfm(df, reference_date=None):
    """
    Build an RFM feature table from cleaned transaction data.

    Groups by CustomerID and computes Recency (days since last purchase),
    Frequency (number of distinct orders), and Monetary (total spend).

    Args:
        df (pd.DataFrame): Cleaned transaction DataFrame from load_and_clean().
        reference_date (str or None): Anchor date for recency calculation.
            Defaults to one day after the latest transaction in the dataset.

    Returns:
        pd.DataFrame: One row per customer with columns Recency, Frequency, Monetary.
    """
    if reference_date is None:
        reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    else:
        reference_date = pd.to_datetime(reference_date)

    rfm_df = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()

    return rfm_df


# ============================================================
# PART C: Feature Scaling
# ============================================================

def scale_features(rfm_df):
    """
    Scale RFM features using StandardScaler.

    K-Means is distance-based, so all features must be on the same scale.
    Without scaling, Monetary (large dollar values) would dominate Recency
    and Frequency, producing distorted clusters.

    Args:
        rfm_df (pd.DataFrame): RFM DataFrame from build_rfm().

    Returns:
        tuple: (scaled_data np.ndarray, fitted StandardScaler instance)
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

    return scaled_data, scaler


# ============================================================
# PART D: K-Means Clustering
# ============================================================

def find_optimal_k(scaled_data, k_range=range(2, 9)):
    """
    Evaluate K-Means across a range of K values to help identify the optimal number of clusters.

    Computes inertia (elbow method) and silhouette score for each K.
    Inertia measures within-cluster tightness — lower is better but always
    decreases with K. Silhouette measures cluster separation — higher is better.
    Use both together to find the point of diminishing returns.

    Args:
        scaled_data (np.ndarray): Scaled RFM features from scale_features().
        k_range (range): Range of K values to evaluate. Defaults to 2-8.

    Returns:
        pd.DataFrame: Columns K, Inertia, and Silhouette for each value tested.
    """
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled_data)
        results.append({
            'K': k,
            'Inertia': km.inertia_,
            'Silhouette': silhouette_score(scaled_data, km.labels_)
        })
    return pd.DataFrame(results)


def run_kmeans(scaled_data, n_clusters):
    """
    Fit a K-Means model to the scaled RFM data.

    Args:
        scaled_data (np.ndarray): Scaled RFM features from scale_features().
        n_clusters (int): Number of clusters to fit.

    Returns:
        KMeans: Fitted scikit-learn KMeans model with labels_ and cluster_centers_.
    """
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(scaled_data)
    return km


# ============================================================
# PART E: Visualization Helpers
# ============================================================

def plot_elbow_and_silhouette(results_df):
    """
    Plot elbow curve and silhouette scores side-by-side to aid K selection.

    The elbow curve shows where inertia stops dropping sharply. The silhouette
    plot shows where cluster separation peaks. Use both together to pick K.

    Args:
        results_df (pd.DataFrame): Output from find_optimal_k() with columns
            K, Inertia, and Silhouette.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(results_df['K'], results_df['Inertia'], marker='o', color='steelblue')
    axes[0].set_title('Elbow Curve')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Inertia')

    axes[1].plot(results_df['K'], results_df['Silhouette'], marker='o', color='darkorange')
    axes[1].set_title('Silhouette Score')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')

    plt.tight_layout()
    plt.show()


def plot_clusters_2d(scaled_data, labels, rfm_df):
    """
    Visualize customer clusters in 2D using PCA projection.

    Reduces the 3D RFM feature space to 2 principal components for plotting.
    Axis labels include the percentage of variance captured by each component.

    Args:
        scaled_data (np.ndarray): Scaled RFM features from scale_features().
        labels (np.ndarray): Cluster label assignments from KMeans.labels_.
        rfm_df (pd.DataFrame): RFM DataFrame, used for reference (not plotted directly).
    """
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_data)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(components[:, 0], components[:, 1],
                          c=labels, cmap='tab10', alpha=0.6, edgecolors='k', linewidths=0.3)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Customer Segments — PCA Projection')
    plt.tight_layout()
    plt.show()


# ============================================================
# PART F: AI-Powered Persona Generator
# ============================================================

def generate_persona(cluster_stats, cluster_id, client, model='gemini-2.5-flash', total_customers=None, median_rfm=None):
    """
    Generate an AI-powered persona description for a customer cluster.
    
    Args:
        cluster_stats: Series or dict with RFM statistics for the cluster
        cluster_id: Cluster identifier (int)
        client: Google Gemini API client
        model: Gemini model name to use
        total_customers: Total number of customers across all clusters (for percentage)
        median_rfm: Dict with median values {'recency': x, 'frequency': y, 'monetary': z}
    
    Returns:
        str: AI-generated persona description
    """
    # Support both plain column names (notebook test) and _Mean suffix (engine class)
    recency = cluster_stats.get('Recency_Mean', cluster_stats.get('Recency'))
    frequency = cluster_stats.get('Frequency_Mean', cluster_stats.get('Frequency'))
    monetary = cluster_stats.get('Monetary_Mean', cluster_stats.get('Monetary'))
    count = cluster_stats.get('Count', 'N/A')
    
    # Calculate percentage if total provided
    if total_customers and count != 'N/A':
        pct_of_base = f"{count/total_customers*100:.1f}%"
    else:
        pct_of_base = "N/A"
    
    # Generate RFM hint if median values provided
    rfm_hint = ""
    if median_rfm:
        is_recent = recency < median_rfm['recency']
        is_frequent = frequency > median_rfm['frequency']
        is_high_value = monetary > median_rfm['monetary']
        
        # Determine segment type based on RFM pattern
        if is_recent and is_frequent and is_high_value:
            hint = "Champions or VIP customers (best segment)"
        elif is_recent and is_high_value and not is_frequent:
            hint = "Big Spenders (high value, lower frequency)"
        elif is_recent and is_frequent and not is_high_value:
            hint = "Loyal Regulars (engaged but lower spend)"
        elif not is_recent and is_high_value:
            hint = "At-Risk High-Value customers (used to spend big, now dormant)"
        elif not is_recent and is_frequent:
            hint = "Slipping Loyalists (were frequent, now fading)"
        elif not is_recent and not is_frequent and not is_high_value:
            hint = "Lost Low-Value customers (hibernating or churned)"
        elif is_frequent and not is_high_value:
            hint = "Bargain Hunters (frequent but low-spend)"
        else:
            hint = "Promising segment (mixed signals)"
        
        rfm_hint = f"\n**RFM Pattern:** {hint}\n"
    
    # Build median context if provided
    median_context = ""
    if median_rfm:
        median_context = f" (median: {median_rfm['recency']:.1f} | lower = more recent)"
        freq_context = f" (median: {median_rfm['frequency']:.1f} | higher = more loyal)"
        mon_context = f" (median: ${median_rfm['monetary']:.2f} | higher = more valuable)"
    else:
        median_context = " (lower = more recent purchase)"
        freq_context = " (higher = more loyal)"
        mon_context = " (higher = more valuable)"

    prompt = f"""You are a marketing strategist analyzing customer segments based on RFM analysis.

**Cluster {cluster_id} Profile:**
- **Recency:** {recency:.1f} days{median_context}
- **Frequency:** {frequency:.1f} orders{freq_context}
- **Monetary:** ${monetary:.2f} total spend{mon_context}
- **Segment Size:** {count:,} customers ({pct_of_base} of customer base)
{rfm_hint}
Create a persona for this segment:

1. **Persona Name** (2-4 words, descriptive + memorable)
   Examples: "Big Spender Drifters" (high monetary, high recency), "Bargain Hunters" (low monetary, high frequency), "VIP Champions" (high on all three)

2. **Behavioral Profile** (2 sentences maximum)
   Focus on: purchase patterns, value to business, engagement level, and likely motivations.

3. **Marketing Strategy** (3 tactical recommendations)
   Be specific about:
   - Communication channels (email, SMS, app push, direct mail)
   - Offer types (discounts, loyalty rewards, exclusive access, bundles)
   - Timing and frequency of outreach
   - Key messaging angles

Keep it concise, actionable, and tailored to the RFM scores above."""

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    return response.text


# ============================================================
# PART G: The Complete Segmentation Engine
# ============================================================

class SegmentationEngine:
    """
    End-to-end customer segmentation pipeline.

    Parameters
    ----------
    filepath : str
        Path to the Online Retail CSV.
    api_key : str
        Google Gemini API key.
    model : str
        Gemini model name.
    """

    def __init__(self, filepath, api_key=None, model='gemini-2.5-flash'):
        self.filepath = filepath
        self.model = model
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY')

        # These get populated as the pipeline runs
        self.raw_df = None
        self.clean_df = None
        self.rfm_df = None
        self.scaled_data = None
        self.scaler = None
        self.kmeans_model = None
        self.k_results = None
        self.personas = {}

        # Initialize Gemini client if key provided
        self.client = None
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Could not initialize Gemini client: {e}")

    def run_segmentation(self, n_clusters=None):
        """
        Run the full segmentation pipeline end-to-end.

        Executes all steps in order: load and clean data, build RFM features,
        scale, optionally find optimal K, fit K-Means, and attach cluster labels
        to the RFM DataFrame. Results are stored as instance attributes for use
        by the Streamlit app and generate_personas().

        Args:
            n_clusters (int or None): Number of clusters to use. If None, optimal
                K is selected automatically using the highest silhouette score.
        """
        # 1. Load and clean the data
        print("Loading and cleaning data...")
        self.clean_df = load_and_clean(self.filepath)

        # 2. Build RFM features
        print("Building RFM features...")
        self.rfm_df = build_rfm(self.clean_df)

        # 3. Scale features
        print("Scaling features...")
        self.scaled_data, self.scaler = scale_features(self.rfm_df)

        # 4. Find optimal K if not provided
        if n_clusters is None:
            print("Finding optimal K...")
            self.k_results = find_optimal_k(self.scaled_data)
            n_clusters = int(self.k_results.loc[
                self.k_results['Silhouette'].idxmax(), 'K'])
            print(f"Optimal K selected: {n_clusters}")

        # 5. Run K-Means
        print(f"Running K-Means with K={n_clusters}...")
        self.kmeans_model = run_kmeans(self.scaled_data, n_clusters)

        # 6. Add cluster labels to RFM dataframe
        self.rfm_df['Cluster'] = self.kmeans_model.labels_

        # 7. Print summary
        print("\nSegmentation complete!")
        print(self.get_cluster_summary())

    def generate_personas(self):
        """Generate AI personas for each cluster with enhanced RFM context."""
        if self.rfm_df is None or 'Cluster' not in self.rfm_df.columns:
            print("Run segmentation first!")
            return {}

        if self.client is None:
            print("Gemini API key required for persona generation.")
            return {}

        # Calculate cluster statistics
        cluster_summary = self.rfm_df.groupby('Cluster').agg(
            Recency_Mean=('Recency', 'mean'),
            Frequency_Mean=('Frequency', 'mean'),
            Monetary_Mean=('Monetary', 'mean'),
            Count=('Recency', 'count')
        ).round(1)
        
        # Calculate medians across all customers for context
        median_rfm = {
            'recency': self.rfm_df['Recency'].median(),
            'frequency': self.rfm_df['Frequency'].median(),
            'monetary': self.rfm_df['Monetary'].median()
        }
        
        total_customers = len(self.rfm_df)

        print("\nGenerating personas...\n")
        for cluster_id, stats in cluster_summary.iterrows():
            persona = generate_persona(
                stats, 
                cluster_id, 
                self.client, 
                self.model,
                total_customers=total_customers,
                median_rfm=median_rfm
            )
            self.personas[cluster_id] = persona
            print(f"--- Cluster {cluster_id} ({stats['Count']:.0f} customers) ---")
            print(persona)
            print()

        return self.personas

    def get_cluster_summary(self):
        """Return a summary DataFrame of cluster statistics."""
        if self.rfm_df is None or 'Cluster' not in self.rfm_df.columns:
            print("Run segmentation first!")
            return None
        return self.rfm_df.groupby('Cluster').agg(
            Count=('Recency', 'count'),
            Recency_Mean=('Recency', 'mean'),
            Frequency_Mean=('Frequency', 'mean'),
            Monetary_Mean=('Monetary', 'mean'),
            Monetary_Total=('Monetary', 'sum')
        ).round(1)

    def plot_results(self):
        """Plot elbow/silhouette curves and 2D cluster scatter."""
        if self.k_results is not None:
            plot_elbow_and_silhouette(self.k_results)
        if self.scaled_data is not None and self.kmeans_model is not None:
            plot_clusters_2d(self.scaled_data, self.kmeans_model.labels_, self.rfm_df)
