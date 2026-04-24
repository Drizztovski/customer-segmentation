"""
segmentation_app.py — Streamlit Dashboard for Customer Segmentation
====================================================================
Interactive web interface for running RFM-based customer segmentation,
exploring cluster results, and generating AI-powered marketing personas
powered by Google Gemini.

Run with:  streamlit run segmentation_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Make sure we can import from the current directory
sys.path.insert(0, os.path.dirname(__file__))

from segmentation_engine import SegmentationEngine


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<div style="
    background-color: #1E2A3A;
    padding: 1.5rem 2rem;
    border-radius: 10px;
    border-left: 5px solid #4FC3F7;
    margin-bottom: 1rem;
">
    <h1 style="color: #FAFAFA; margin: 0; font-size: 2rem;">👥 Customer Segmentation Dashboard</h1>
    <p style="color: #4FC3F7; margin: 0.4rem 0 0 0; font-size: 1rem;">
        RFM Analysis · K-Means Clustering · AI-Powered Personas
    </p>
    <p style="color: #8BA3B8; margin: 0.3rem 0 0 0; font-size: 0.85rem;">
        Built by AJ Amatrudo · Data Analytics Bootcamp Portfolio Project
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR: Configuration
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Gemini API Key (optional)",
        type="password",
        help="Needed for AI persona generation only",
    )

    data_path = st.text_input(
        "Data file path",
        value="data/online_retail_sample.csv",
    )

    n_clusters = st.slider(
        "Number of clusters (K)",
        min_value=2, max_value=8, value=4,
        help="Try the elbow method to find the best K"
    )

    run_button = st.button("🚀 Run Segmentation", type="primary", use_container_width=True)

    st.divider()

    if st.session_state.get('segmentation_run'):
        st.subheader("🔄 Try Different K")
        new_k = st.slider(
            "Re-run with a new K value",
            min_value=2, max_value=8, value=4, key="rerun_k"
        )
        if st.button("Re-run Segmentation", use_container_width=True):
            with st.spinner(f"Re-running with K={new_k}..."):
                try:
                    st.session_state.engine.run_segmentation(n_clusters=new_k)
                    st.session_state.engine.personas = {}
                    st.toast(f"Re-segmented with K={new_k}!", icon="🔄")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        st.divider()
        st.subheader("📊 Cluster Overview")
        _engine = st.session_state.engine
        _pca = PCA(n_components=2)
        _components = _pca.fit_transform(_engine.scaled_data)
        _fig, _ax = plt.subplots(figsize=(4, 3))
        _fig.patch.set_facecolor('#1A2332')
        _ax.set_facecolor('#1A2332')
        _ax.scatter(_components[:, 0], _components[:, 1],
                    c=_engine.kmeans_model.labels_,
                    cmap='tab10', alpha=0.6, s=10, edgecolors='none')
        _ax.tick_params(colors='#8BA3B8', labelsize=7)
        for spine in _ax.spines.values():
            spine.set_edgecolor('#8BA3B8')
        _ax.set_xlabel('PC1', color='#8BA3B8', fontsize=7)
        _ax.set_ylabel('PC2', color='#8BA3B8', fontsize=7)
        plt.tight_layout()
        st.pyplot(_fig)
        plt.close(_fig)


    st.divider()


# ============================================================
# SESSION STATE
# ============================================================

if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'segmentation_run' not in st.session_state:
    st.session_state.segmentation_run = False


# ============================================================
# RUN SEGMENTATION
# ============================================================

if run_button:
    with st.spinner("Running segmentation..."):
        try:
            engine = SegmentationEngine(
                filepath=data_path,
                api_key=api_key if api_key else None
            )
            engine.run_segmentation(n_clusters=n_clusters)
            st.session_state.engine = engine
            st.session_state.segmentation_run = True
            st.toast("Segmentation complete!", icon="✅")
        except Exception as e:
            st.error(f"Error running segmentation: {e}")


# ============================================================
# MAIN AREA: Results Display
# ============================================================

if st.session_state.segmentation_run:
    engine = st.session_state.engine
    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🔵 Cluster Results", "🤖 Personas"])

    with tab1:
        st.subheader("Dataset Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", f"{engine.rfm_df.shape[0]:,}")
        col2.metric("Total Transactions", f"{engine.clean_df.shape[0]:,}")
        col3.metric("Date Range", (
            f"{engine.clean_df['InvoiceDate'].min().strftime('%b %Y')} – "
            f"{engine.clean_df['InvoiceDate'].max().strftime('%b %Y')}"
        ))

        st.subheader("RFM Distributions")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, col, color in zip(axes,
                                   ['Recency', 'Frequency', 'Monetary'],
                                   ['steelblue', 'darkorange', 'seagreen']):
            ax.hist(engine.rfm_df[col], bins=30, color=color, edgecolor='white')
            ax.set_title(col)
            ax.set_xlabel(col)
            ax.set_ylabel('Customers')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.subheader("⬇️ Download Segmented Data")
        csv = engine.rfm_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download RFM + Cluster Labels as CSV",
            data=csv,
            file_name="customer_segments.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tab2:
        st.subheader("Cluster Summary")
        summary = engine.get_cluster_summary()
        st.dataframe(summary, use_container_width=True)

        st.subheader("🔍 Compare Two Clusters")
        cluster_ids = sorted(engine.rfm_df['Cluster'].unique().tolist())
        col_cmp_l, col_cmp_r = st.columns(2)
        with col_cmp_l:
            cluster_a = st.selectbox("Cluster A", cluster_ids, index=0, key="cmp_a")
        with col_cmp_r:
            cluster_b = st.selectbox("Cluster B", cluster_ids, index=1, key="cmp_b")
        if cluster_a != cluster_b:
            cmp_df = summary.loc[[cluster_a, cluster_b]].T
            cmp_df.columns = [f"Cluster {cluster_a}", f"Cluster {cluster_b}"]
            st.dataframe(cmp_df, use_container_width=True)
        else:
            st.warning("Select two different clusters to compare.")

        st.subheader("Customer Segments — PCA Projection")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        pca = PCA(n_components=2)
        components = pca.fit_transform(engine.scaled_data)
        scatter = ax2.scatter(components[:, 0], components[:, 1],
                              c=engine.kmeans_model.labels_,
                              cmap='tab10', alpha=0.6, edgecolors='k', linewidths=0.3)
        plt.colorbar(scatter, ax=ax2, label='Cluster')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        st.subheader("Cluster Profiles")
        col_a, col_b = st.columns(2)

        with col_a:
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            cluster_means = engine.rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
            cluster_means.plot(kind='bar', ax=ax3, colormap='Set2', edgecolor='white')
            ax3.set_title('Mean RFM per Cluster')
            ax3.set_xlabel('Cluster')
            ax3.tick_params(axis='x', rotation=0)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

        with col_b:
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            counts = engine.rfm_df['Cluster'].value_counts().sort_index()
            ax4.bar(counts.index.astype(str), counts.values, color='steelblue', edgecolor='white')
            ax4.set_title('Customers per Cluster')
            ax4.set_xlabel('Cluster')
            ax4.set_ylabel('Count')
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)

    with tab3:
        st.subheader("AI-Generated Customer Personas")

        if not engine.client:
            st.warning("No Gemini API key provided. Enter your key in the sidebar to generate personas.")
        else:
            if st.button("🤖 Generate Personas", type="primary"):
                with st.spinner("Generating personas for each cluster..."):
                    try:
                        engine.generate_personas()
                        st.toast("Personas generated!", icon="🤖")
                    except Exception as e:
                        st.error(f"Error generating personas: {e}")

            if engine.personas:
                for cluster_id, persona_text in engine.personas.items():
                    cluster_size = engine.rfm_df[engine.rfm_df['Cluster'] == cluster_id].shape[0]
                    with st.expander(f"Cluster {cluster_id} — {cluster_size} customers", expanded=True):
                        st.markdown(persona_text)
            else:
                st.info("Click **Generate Personas** above to get AI-powered marketing personas for each cluster.")

else:
    st.info("Configure settings in the sidebar and click **🚀 Run Segmentation** to get started.")
