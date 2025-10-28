# streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# --- App Title ---
st.title("Chapter 4: Results and Discussion")
st.subheader("Bridging Gaps: A Comprehensive Approach to Indigenous Knowledge Preservation in Bunawan, Agusan del Sur")

st.info("This Streamlit app shows analytical results based on Chapter 3 methods — Decision Tree Classification and Clustering Analysis.")

# --- Upload Dataset ---
uploaded_file = st.file_uploader("📤 Upload your preprocessed CSV dataset", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset loaded successfully! ({df.shape[0]} rows, {df.shape[1]} columns)")
    st.write("### Preview of Data")
    st.dataframe(df.head())

    # --- Descriptive Statistics ---
    st.write("## Descriptive Statistics")
    st.write(df.describe(include='all'))

    # --- Encode categorical data ---
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    # --- Correlation Heatmap ---
    st.write("## Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(), cmap='coolwarm', annot=True, fmt=".2f", ax=ax)
    st.pyplot(fig)

    # --- Decision Tree Classification ---
    st.write("## Decision Tree Classification Result")
    target_col = st.selectbox("Select Target Column (e.g., Recognized)", df.columns)
    feature_cols = st.multiselect("Select Feature Columns", [c for c in df.columns if c != target_col])

    if st.button("Run Decision Tree Classification"):
        X = df[feature_cols]
        y = df[target_col]

        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X, y)
        acc = model.score(X, y)
        st.write(f"### ✅ Model Accuracy: {acc*100:.2f}%")

        # Plot the Decision Tree
        fig, ax = plt.subplots(figsize=(12,6))
        class_labels = [str(c) for c in sorted(y.unique())]  # auto detect class names
        plot_tree(model, feature_names=feature_cols, class_names=class_labels, filled=True, ax=ax)


    # --- K-Means Clustering ---
    st.write("## K-Means Clustering (Generational Gap Analysis)")
    num_clusters = st.slider("Select number of clusters (e.g., 3 for G1, G2, G3)", 2, 6, 3)
    cluster_features = st.multiselect("Select features for clustering", df.columns)

    if st.button("Run Clustering"):
        X = df[cluster_features]
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)
        st.success("✅ Clustering complete!")

        # Cluster visualization
        fig, ax = plt.subplots(figsize=(8,6))
        sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,1], hue=df['Cluster'], palette='tab10', s=80)
        plt.title("Generational Clusters Showing Knowledge Gaps")
        st.pyplot(fig)

        st.write("### Cluster Distribution")
        st.bar_chart(df['Cluster'].value_counts())

    # --- Recognition Gap Table ---
    st.write("## Recognition Gap Analysis")
    if 'Elders_Score' in df.columns and 'Youth_Score' in df.columns:
        df['Gap(%)'] = (df['Elders_Score'] - df['Youth_Score']).abs()
        st.write("### Recognition Gap per Cultural Item")
        st.dataframe(df[['Cultural_Item', 'Elders_Score', 'Youth_Score', 'Gap(%)']])
        st.write("### Items with ≥50% Gap (Critical Cultural Loss)")
        st.dataframe(df[df['Gap(%)'] >= 50])

        # Heatmap
        fig, ax = plt.subplots(figsize=(8,5))
        pivot = df.pivot_table(index='Cultural_Item', values='Gap(%)')
        sns.heatmap(pivot, cmap='coolwarm', annot=True, ax=ax)
        plt.title("Recognition Gap Heatmap")
        st.pyplot(fig)

else:
    st.warning("Please upload your Chapter 3 processed dataset (.csv) to view Chapter 4 results.")
