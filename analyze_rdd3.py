import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def clean_numeric(x):
    if isinstance(x, str):
        x = x.replace(',', '.').strip()
        if x == '?' or x == '':
            return np.nan
        # Handle cases like ">6" or "<5" by taking the number
        if '>' in x:
            x = x.replace('>', '')
        if '<' in x:
            x = x.replace('<', '')
        if '(' in x: # Handle cases like "5> (1?)"
             x = x.split('(')[0].strip().replace('>', '')
        try:
            return float(x)
        except ValueError:
            return np.nan
    return x

def analyze_rdd3():
    file_path = 'RDD3 Thibault  - Feuille 1.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    # --- Preprocessing ---
    print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns.")

    # Target variable
    target_col = 'resultat'
    
    # 1. Handle "Heure" BEFORE numeric cleaning (since it is text)
    if 'heure' in df.columns:
        # Normalize text
        df['heure'] = df['heure'].astype(str).str.lower().str.strip()
        df['heure'] = df['heure'].str.replace('aprèm', 'aprem').str.replace('aprème', 'aprem')
        # Fix possible typos or extra spaces
        
        # Create Dummy Variables
        # We want meaningful cols like 'heure_soir', 'heure_aprem', 'heure_matin'
        # We add them to the dataframe so they get picked up by numeric selection later
        dummies = pd.get_dummies(df['heure'], prefix='heure')
        # Convert bool to int for correlation
        dummies = dummies.astype(int)
        df = pd.concat([df, dummies], axis=1)
        print(f"Processed 'heure': added columns {dummies.columns.tolist()}")

    # 2. Handle Numeric Columns
    potential_numeric_cols = [
        'hb', 'age', 'body', 'cout', 'mois', 'resultat', 'note', 'nbr', 
        'temporaire', 'béquille', 'footing /sport avant ', 't shirt blanc', 
        'percée ', 'tatouée ', 'pull', 'pourcentage de sexu', 
        "nombre de jours après l'open", 'sextos par sms'
    ]
    # Removed 'heure' from potential_numeric_cols as we handled it separately

    for col in potential_numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)

    # 3. Remove Outliers
    open_col = "nombre de jours après l'open"
    if open_col in df.columns:
        # User requested to remove the single huge outlier (661). 
        # We'll filter sensible range, e.g., < 300. 
        # Most data is small.
        initial_count = len(df)
        
        # Proper filter:
        mask = (df[open_col] < 300) | (df[open_col].isna())
        df = df[mask]
        print(f"Removed outliers in '{open_col}': {initial_count} -> {len(df)} rows.")

    # Fill NaN in result with 0 if appropraite, or drop? 
    # Looking at the data, result seems to be 0 or 1.
    df = df.dropna(subset=[target_col])
    
    # --- Correlation Analysis (Numeric) ---
    print("\n--- Correlations with 'resultat' (Numeric) ---")
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlations and p-values
    correlations_data = []
    from scipy import stats
    
    for col in numeric_df.columns:
        if col == target_col:
            continue
        # Drop NaNs for the pair calculation
        valid_data = numeric_df[[col, target_col]].dropna()
        if len(valid_data) < 2:
            continue
            
        corr, p_value = stats.pearsonr(valid_data[col], valid_data[target_col])
        correlations_data.append({
            'Feature': col,
            'Correlation': corr,
            'P-Value': p_value
        })
    
    corr_df = pd.DataFrame(correlations_data).sort_values(by='Correlation', ascending=False, key=abs)
    
    print(corr_df.to_string(index=False))
    
    if not corr_df.empty:
        best_feature = corr_df.iloc[0]
        print(f"\nStrongest numeric predictor: {best_feature['Feature']} (Corr: {best_feature['Correlation']:.4f}, p={best_feature['P-Value']:.4e})")

    # Use the correlation series for plotting code below (keep compatibility)
    correlations_no_target = corr_df.set_index('Feature')['Correlation']
    correlations = correlations_no_target # For the heatmap selection logic below

    # --- Visualization: Correlation Matrix with Significance ---
    # User requested ALL variables (cout, age, body, mois, etc.)
    # We use all numeric columns available
    matrix_df = numeric_df
    
    corr_matrix = matrix_df.corr()
    
    # Calculate p-values for the entire matrix
    pval_matrix = pd.DataFrame(index=matrix_df.columns, columns=matrix_df.columns)
    
    for c1 in matrix_df.columns:
        for c2 in matrix_df.columns:
            if c1 == c2:
                pval_matrix.loc[c1, c2] = 0.0
            else:
                # dropna for the pair
                valid = matrix_df[[c1, c2]].dropna()
                if len(valid) > 2:
                    _, p = stats.pearsonr(valid[c1], valid[c2])
                    pval_matrix.loc[c1, c2] = p
                else:
                    pval_matrix.loc[c1, c2] = 1.0 # Not sig
    
    pval_matrix = pval_matrix.astype(float)
    
    plt.figure(figsize=(12, 10))
    # Create mask for non-significant values (p > 0.05)
    # also mask diagonal? usually keep it or auto-masked by seaborn? No, seaborn doesn't auto mask diag unless asked.
    # We will just mask non-sig
    mask = pval_matrix > 0.05
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", mask=mask, cbar_kws={'label': 'Correlation (Significant only)'})
    plt.title('Correlation Matrix (p < 0.05 only)')
    plt.tight_layout()
    plt.savefig('correlation_matrix_sig.png')
    print("Saved correlation_matrix_sig.png")

    # --- Visualization: Top Correlations ---
    plt.figure(figsize=(10, 6))
    colors = ['green' if x > 0 else 'red' for x in correlations_no_target.values]
    correlations_no_target.head(10).plot(kind='bar', color=colors)
    plt.title('Top Numeric Correlations with "resultat"')
    plt.ylabel('Correlation Coefficient')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('top_correlations.png')
    print("Saved top_correlations.png")


    # --- Over-representation Analysis (Lift) ---
    print("\n--- Over-representation Analysis (Lift) ---")
    global_success_rate = df[target_col].mean()
    print(f"Global Sucess Rate: {global_success_rate:.2%}")
    
    significant_findings = []

    for col in df.columns:
        if col == target_col:
            continue
            
        # Treat as categorical if object or low cardinality numeric (e.g. days of week, ratings)
        # Even if it has a linear correlation, we want to see if specific values are over-represented.
        is_categorical = df[col].dtype == 'object' or (df[col].nunique() < 15)
        
        if is_categorical:
            # fillna to handle missing values as a category
            series = df[col].fillna('Unknown')
            
            # Calculate success rate per group
            group_stats = df.groupby(series)[target_col].agg(['mean', 'count'])
            group_stats = group_stats[group_stats['count'] > 2] # Filter out very small groups
            
            for category, row in group_stats.iterrows():
                lift = row['mean'] / global_success_rate if global_success_rate > 0 else 0
                if lift > 1.3: # Lower threshold slightly for plot
                    significant_findings.append({
                        'Feature': col,
                        'Value': category,
                        'Success Rate': row['mean'],
                        'Lift': lift,
                        'Count': row['count']
                    })

    # Sort findings by Lift
    significant_findings.sort(key=lambda x: x['Lift'], reverse=True)
    
    print(f"\nTop Over-represented features (where success rate > 1.3x global rate, min sample 3):")
    for item in significant_findings[:10]: # Top 10
        print(f" - {item['Feature']} = {item['Value']}: Success Rate {item['Success Rate']:.2%} (Lift: {item['Lift']:.2f}, n={item['Count']})")

    # --- Visualization: Significant Lifts ---
    if significant_findings:
        top_lifts = significant_findings[:8]
        labels = [f"{x['Feature']}={x['Value']}" for x in top_lifts]
        lifts = [x['Lift'] for x in top_lifts]
        
        plt.figure(figsize=(12, 6))
        plt.bar(labels, lifts, color='purple')
        plt.axhline(1.0, color='gray', linestyle='--', label='Average (1.0)')
        plt.title('Top Over-represented Segments (Lift > 1.0)')
        plt.ylabel('Lift (vs Global Average)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('significant_lifts.png')
        print("Saved significant_lifts.png")

    # --- Specific Analysis: Footing/Sport ---
    print("\n--- Detail: Footing / Sport Avant ---")
    sport_col = 'footing /sport avant '
    if sport_col in df.columns:
        # 0 = pas de sport, 1=musculation avant, 2 = footing avant.
        sport_mapping = {0: 'Pas de sport', 1: 'Musculation', 2: 'Footing'}
        df['sport_label'] = df[sport_col].map(sport_mapping)
        
        sport_stats = df.groupby('sport_label')[target_col].agg(['mean', 'count', 'sum'])
        sport_stats['mean'] = sport_stats['mean'] * 100
        print(sport_stats)
    else:
        print(f"Column '{sport_col}' not found.")

    # --- Clustering (Ideal Types) ---
    print("\n--- Clustering (Ideal Types - Profile Based) ---")
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer

    # 1. Feature Selection
    # CRITICAL UPDATE via User Input:
    # note = post-date feeling (Outcome) -> Exclude from clustering
    # pull = brought home (Outcome/Proxy) -> Exclude from clustering
    # We want clusters of PROFILES (The Girl), not outcomes.
    
    # Input Features (The Girl & The Setup)
    # hb = beauty
    # body = body count (historic)
    # nbr = date number
    # temporaire = temporary resident
    # percée, tatouée = style
    # classe, ethnie = demographics
    # age = demographics
    # sextos par sms = interaction style (pre-date mostly)
    
    cluster_numeric = ['age', 'hb', 'body', 'sextos par sms', 'pourcentage de sexu'] 
    cluster_cat = ['classe', 'ethnie', 'temporaire'] 
    cluster_binary = ['tatouée ', 'percée ', 'béquille'] 

    # Prepare data for clustering
    cluster_df = df.copy()
    
    final_cols = []
    
    # Handle Numeric
    for col in cluster_numeric:
        if col in cluster_df.columns:
            final_cols.append(col)

    # Handle Categorical
    for col in cluster_cat:
        if col in cluster_df.columns:
            # Convert numeric-coded cats (like temporaire 0/1) to string for treating as categorical if needed, 
            # OR just treat temporaire as numeric/binary. 
            # Temporaire is 0/1, so treat as numeric/binary is fine.
            if df[col].dtype == 'object' or df[col].nunique() < 5:
                le = LabelEncoder()
                cluster_df[col] = cluster_df[col].astype(str)
                cluster_df[col] = le.fit_transform(cluster_df[col])
                final_cols.append(col)
            else:
                 final_cols.append(col)

    # Add binary style cols if exist
    for col in cluster_binary:
         if col in cluster_df.columns:
             final_cols.append(col)

    print(f"Clustering based on features: {final_cols}")

    # Select and Impute
    X = cluster_df[final_cols]
    imputer = SimpleImputer(strategy='mean') # Mean imputation for missing values (e.g. missing body count)
    X_imputed = imputer.fit_transform(X)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # K-Means
    k = 3 
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters

    # Description
    print(f"\nIdentified {k} Clusters (Profile Types):")
    for i in range(k):
        print(f"\n--- Cluster {i} ({sum(clusters==i)} profiles) ---")
        cluster_subset = df[df['Cluster'] == i]
        
        # Numeric Stats
        stats_cols = [c for c in ['age', 'hb', 'body', 'sextos par sms'] if c in df.columns]
        print(cluster_subset[stats_cols].mean().to_string())
        
        # Categorical Dominance
        for col in ['classe', 'ethnie', 'temporaire']:
            if col in df.columns:
                top_val = cluster_subset[col].mode()
                if not top_val.empty:
                    print(f"Top {col}: {top_val.iloc[0]}")
        
        # Style Indicators
        for col in ['tatouée ', 'percée ', 'béquille']:
             if col in df.columns:
                 rate = cluster_subset[col].mean()
                 if rate > 0.5:
                     print(f"Mostly {col.strip()} ({rate:.0%})")
        
        # Result rate
        success_rate = cluster_subset[target_col].mean()
        
        # Detailed Pull Analysis
        if 'pull' in df.columns:
            pull_counts = cluster_subset['pull'].value_counts(dropna=False)
            pull_yes = pull_counts.get(1.0, 0)
            pull_no = pull_counts.get(0.0, 0)
            pull_nan = pull_counts.get(np.nan, 0)
            pull_total_known = pull_yes + pull_no
            pull_rate_known = (pull_yes / pull_total_known) if pull_total_known > 0 else 0
        
            print(f"-> Success Rate: {success_rate:.2%}")
            print(f"-> Pull Stats: {pull_yes} Yes, {pull_no} No, {pull_nan} Unknown/NaN")
            print(f"-> Pull Rate (Known Only): {pull_rate_known:.2%} (based on {pull_total_known} samples)")
        
        avg_note = cluster_subset['note'].mean() if 'note' in df.columns else 0
        print(f"-> Avg Note (Feeling): {avg_note:.2f}")

    # Visualization: PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(components[:, 0], components[:, 1], c=clusters, cmap='viridis', s=100, alpha=0.7)
    plt.title('Profiles Clustering (Girl Features Only)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter, label='Cluster ID')
    plt.tight_layout()
    plt.savefig('cluster_pca.png')
    print("Saved cluster_pca.png")

if __name__ == "__main__":
    analyze_rdd3()
