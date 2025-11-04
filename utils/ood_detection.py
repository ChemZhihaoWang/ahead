import cv2
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class OODDetector:
    def __init__(self, in_dist_folder, ood_folder):
        """
        Initialize OOD detector
        
        Parameters:
            in_dist_folder: Path to in-distribution data folder
            ood_folder: Path to OOD data folder to be detected
        """
        self.in_dist_folder = in_dist_folder
        self.ood_folder = ood_folder
        self.in_dist_features = None
        self.ood_features = None
        self.scaler = StandardScaler()
        self.histogram_data = None
        
    def extract_features(self, image_path):
        """
        Extract multi-dimensional features from a single image
        
        Returns feature vector
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            features = []
            
            # 1. Statistical features
            features.extend([
                np.mean(gray),
                np.std(gray),
                np.var(gray),
                np.median(gray),
                stats.skew(gray.flatten()),
                stats.kurtosis(gray.flatten()),
            ])
            
            # 2. Color histogram features (RGB channels)
            if len(img.shape) == 3:
                for channel in range(3):
                    hist = cv2.calcHist([img], [channel], None, [32], [0, 256])
                    features.extend(hist.flatten()[:16])  # Take first 16 bins
            
            # 3. Texture features (using LBP-like statistics)
            # Calculate local variance
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            local_var = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
            features.extend([
                np.mean(local_var),
                np.std(local_var),
            ])
            
            # 4. Edge features
            edges = cv2.Canny(gray, 50, 150)
            features.extend([
                np.mean(edges),
                np.sum(edges > 0) / (gray.shape[0] * gray.shape[1]),  # Edge pixel ratio
            ])
            
            # 5. Image size and shape features
            features.extend([
                gray.shape[0],
                gray.shape[1],
                gray.shape[0] / gray.shape[1],  # Aspect ratio
            ])
            
            # 6. Percentile features
            features.extend([
                np.percentile(gray, 25),
                np.percentile(gray, 75),
                np.percentile(gray, 90),
                np.percentile(gray, 95),
            ])
            
            return np.array(features)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def load_and_extract_features(self, folder_path, description="Extracting features"):
        """
        Load all images from folder and extract features
        """
        features_list = []
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\n{description}: {len(image_files)} images")
        
        for filename in tqdm(image_files, desc=description):
            image_path = os.path.join(folder_path, filename)
            features = self.extract_features(image_path)
            if features is not None:
                features_list.append(features)
        
        return np.array(features_list)
    
    def detect_with_one_class_svm(self):
        """
        Detect OOD using One-Class SVM method
        """
        print("\n=== OOD Detection using One-Class SVM ===")
        
        # Standardize features
        X_in = self.scaler.fit_transform(self.in_dist_features)
        X_ood = self.scaler.transform(self.ood_features)
        
        # Train One-Class SVM
        clf = OneClassSVM(nu=0.05, gamma='scale', kernel='rbf')
        clf.fit(X_in)
        
        # Predict
        in_pred = clf.predict(X_in)
        ood_pred = clf.predict(X_ood)
        
        # Calculate scores (distance to hyperplane)
        in_scores = clf.score_samples(X_in)
        ood_scores = clf.score_samples(X_ood)
        
        print(f"In-distribution anomaly detection rate: {(in_pred == -1).sum() / len(in_pred) * 100:.2f}%")
        print(f"OOD data anomaly detection rate: {(ood_pred == -1).sum() / len(ood_pred) * 100:.2f}%")
        print(f"\nIn-distribution average score: {in_scores.mean():.4f}")
        print(f"OOD data average score: {ood_scores.mean():.4f}")
        print(f"Score difference: {abs(in_scores.mean() - ood_scores.mean()):.4f}")
        
        return {
            'method': 'One-Class SVM',
            'in_anomaly_rate': (in_pred == -1).sum() / len(in_pred),
            'ood_anomaly_rate': (ood_pred == -1).sum() / len(ood_pred),
            'in_scores': in_scores,
            'ood_scores': ood_scores
        }
    
    def statistical_test(self):
        """
        Compare feature distributions using statistical tests
        """
        print("\n=== Statistical Test: Comparing Feature Distributions ===")
        
        # Perform t-test for each feature dimension
        significant_diffs = 0
        total_features = self.in_dist_features.shape[1]
        
        print(f"Performing t-test on {total_features} feature dimensions...")
        
        for i in range(total_features):
            in_feat = self.in_dist_features[:, i]
            ood_feat = self.ood_features[:, i]
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(in_feat, ood_feat)
            
            if p_value < 0.05:  # Significance level 0.05
                significant_diffs += 1
        
        print(f"Significantly different feature dimensions: {significant_diffs} / {total_features} ({significant_diffs/total_features*100:.2f}%)")
        
        # Kolmogorov-Smirnov test
        ks_significant = 0
        for i in range(min(20, total_features)):  # Only test first 20 features to avoid excessive computation
            in_feat = self.in_dist_features[:, i]
            ood_feat = self.ood_features[:, i]
            
            ks_stat, ks_p_value = stats.ks_2samp(in_feat, ood_feat)
            
            if ks_p_value < 0.05:
                ks_significant += 1
        
        print(f"KS test significantly different features (first 20): {ks_significant} / {min(20, total_features)}")
        
        return {
            'method': 'Statistical Test',
            't_test_significant_features': significant_diffs,
            'total_features': total_features,
            'ks_significant_features': ks_significant
        }
    
    def save_histogram_data_for_origin(self, svm_result, base_path):
        """
        Save histogram data (bin centers and densities) for Origin plotting
        Uses the exact bin edges from the visualization
        """
        if self.histogram_data is None:
            # Fallback: calculate manually if not available
            bins = 50
            all_scores = np.concatenate([svm_result['in_scores'], svm_result['ood_scores']])
            min_score = all_scores.min()
            max_score = all_scores.max()
            bin_edges = np.linspace(min_score, max_score, bins + 1)
            id_counts, _ = np.histogram(svm_result['in_scores'], bins=bin_edges, density=True)
            ood_counts, _ = np.histogram(svm_result['ood_scores'], bins=bin_edges, density=True)
        else:
            # Use the exact bin edges from visualization
            bin_edges = self.histogram_data['bin_edges']
            id_counts = self.histogram_data['id_counts']
            ood_counts = self.histogram_data['ood_counts']
        
        # Calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = np.diff(bin_edges)
        
        # Create DataFrame for Origin
        # For Origin column plot, use Bin_Left_Edge as X-axis to match matplotlib's alignment
        # matplotlib hist aligns bars to the left edge of each bin
        histogram_data = {
            'Bin_Left_Edge': bin_edges[:-1],  # Use left edge for X-axis to match matplotlib
            'Bin_Center': bin_centers,  # Also include for reference
            'Bin_Right_Edge': bin_edges[1:],
            'Bin_Width': bin_widths,
            'ID_Density': id_counts,
            'OOD_Density': ood_counts
        }
        
        histogram_df = pd.DataFrame(histogram_data)
        histogram_csv_path = os.path.join(base_path, 'origin_histogram_data.csv')
        histogram_df.to_csv(histogram_csv_path, index=False)
        print(f"Histogram data for Origin saved to: {histogram_csv_path}")
        print(f"  - Total bins: {len(bin_centers)}")
        print(f"  - Score range: {bin_edges[0]:.4f} to {bin_edges[-1]:.4f}")
        print(f"  - Bin width: {bin_widths[0]:.4f}")
        print(f"  - ID density range: {id_counts.min():.4f} to {id_counts.max():.4f}")
        print(f"  - OOD density range: {ood_counts.min():.4f} to {ood_counts.max():.4f}")
        print(f"\n*** IMPORTANT for Origin plotting ***")
        print(f"  - Use 'Bin_Left_Edge' as X-axis (not Bin_Center)")
        print(f"  - Set column width to match Bin_Width ({bin_widths[0]:.4f})")
        print(f"  - This ensures bars align exactly like matplotlib's hist plot")
    
    def save_barchart_data_for_origin(self, svm_result, base_path):
        """
        Save bar chart data for Origin plotting
        """
        bar_data = {
            'Method': ['One-Class SVM'],
            'ID_Anomaly_Rate_Percent': [svm_result['in_anomaly_rate'] * 100],
            'OOD_Anomaly_Rate_Percent': [svm_result['ood_anomaly_rate'] * 100]
        }
        
        bar_df = pd.DataFrame(bar_data)
        bar_csv_path = os.path.join(base_path, 'origin_barchart_data.csv')
        bar_df.to_csv(bar_csv_path, index=False)
        print(f"Bar chart data for Origin saved to: {bar_csv_path}")
    
    def save_results_to_csv(self, svm_result):
        """
        Save detection results and scores to CSV files
        """
        print("\n=== Saving Data to CSV Files ===")
        
        base_path = os.path.dirname(self.in_dist_folder)
        
        # 1. Save score data (for histogram)
        score_data = {
            'Sample_Index': [],
            'Score': [],
            'Data_Type': []
        }
        
        # Add in-distribution scores
        for idx, score in enumerate(svm_result['in_scores']):
            score_data['Sample_Index'].append(f'IN_{idx+1}')
            score_data['Score'].append(score)
            score_data['Data_Type'].append('In-distribution')
        
        # Add OOD scores
        for idx, score in enumerate(svm_result['ood_scores']):
            score_data['Sample_Index'].append(f'OOD_{idx+1}')
            score_data['Score'].append(score)
            score_data['Data_Type'].append('OOD')
        
        score_df = pd.DataFrame(score_data)
        score_csv_path = os.path.join(base_path, 'ood_detection_scores.csv')
        score_df.to_csv(score_csv_path, index=False)
        print(f"Score data saved to: {score_csv_path}")
        
        # 2. Save detection results summary
        summary_data = {
            'Metric': [
                'In-distribution_Anomaly_Detection_Rate',
                'OOD_Anomaly_Detection_Rate',
                'In-distribution_Average_Score',
                'OOD_Average_Score',
                'Score_Difference',
                'In-distribution_Min_Score',
                'In-distribution_Max_Score',
                'OOD_Min_Score',
                'OOD_Max_Score',
                'In-distribution_Std_Score',
                'OOD_Std_Score',
                'In-distribution_Sample_Count',
                'OOD_Sample_Count'
            ],
            'Value': [
                svm_result['in_anomaly_rate'] * 100,
                svm_result['ood_anomaly_rate'] * 100,
                svm_result['in_scores'].mean(),
                svm_result['ood_scores'].mean(),
                abs(svm_result['in_scores'].mean() - svm_result['ood_scores'].mean()),
                svm_result['in_scores'].min(),
                svm_result['in_scores'].max(),
                svm_result['ood_scores'].min(),
                svm_result['ood_scores'].max(),
                svm_result['in_scores'].std(),
                svm_result['ood_scores'].std(),
                len(svm_result['in_scores']),
                len(svm_result['ood_scores'])
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(base_path, 'ood_detection_summary.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Summary data saved to: {summary_csv_path}")
        
        # 3. Save detailed statistics by data type
        stats_data = {
            'Data_Type': ['In-distribution', 'OOD'],
            'Count': [len(svm_result['in_scores']), len(svm_result['ood_scores'])],
            'Mean_Score': [svm_result['in_scores'].mean(), svm_result['ood_scores'].mean()],
            'Std_Score': [svm_result['in_scores'].std(), svm_result['ood_scores'].std()],
            'Min_Score': [svm_result['in_scores'].min(), svm_result['ood_scores'].min()],
            'Max_Score': [svm_result['in_scores'].max(), svm_result['ood_scores'].max()],
            'Median_Score': [np.median(svm_result['in_scores']), np.median(svm_result['ood_scores'])],
            'Anomaly_Rate_Percent': [
                svm_result['in_anomaly_rate'] * 100,
                svm_result['ood_anomaly_rate'] * 100
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        stats_csv_path = os.path.join(base_path, 'ood_detection_statistics.csv')
        stats_df.to_csv(stats_csv_path, index=False)
        print(f"Statistics data saved to: {stats_csv_path}")
        
        # 4. Save histogram data for Origin (bin centers and densities)
        self.save_histogram_data_for_origin(svm_result, base_path)
        
        # 5. Save bar chart data for Origin
        self.save_barchart_data_for_origin(svm_result, base_path)
    
    def visualize_results(self, svm_result):
        """
        Visualize One-Class SVM detection results
        """
        print("\n=== Generating Visualization Results ===")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('OOD Detection Results Visualization (One-Class SVM)', fontsize=16, fontweight='bold')
        
        # 1. One-Class SVM score distribution
        ax = axes[0]
        # Determine unified bin range for both histograms
        bins = 50
        all_scores = np.concatenate([svm_result['in_scores'], svm_result['ood_scores']])
        min_score = all_scores.min()
        max_score = all_scores.max()
        # Use the same bin edges that matplotlib would use for this range
        bin_edges = np.linspace(min_score, max_score, bins + 1)
        
        # Plot histograms and get the ACTUAL values that matplotlib uses
        id_counts, id_bin_edges, id_patches = ax.hist(svm_result['in_scores'], bins=bin_edges, 
                alpha=0.7, label='ID', color='blue', density=True, 
                edgecolor='black', linewidth=0.5)
        ood_counts, ood_bin_edges, ood_patches = ax.hist(svm_result['ood_scores'], bins=bin_edges, 
                alpha=0.7, label='OOD', color='orange', density=True, 
                edgecolor='black', linewidth=0.5)
        
        # Verify bin edges are the same (they should be)
        if not np.array_equal(id_bin_edges, ood_bin_edges):
            print("Warning: ID and OOD bin edges are different! Using unified bin edges.")
            # Use unified bin edges and recalculate
            unified_bin_edges = id_bin_edges  # Use ID's bin edges as reference
            id_counts, _ = np.histogram(svm_result['in_scores'], bins=unified_bin_edges, density=True)
            ood_counts, _ = np.histogram(svm_result['ood_scores'], bins=unified_bin_edges, density=True)
            bin_edges = unified_bin_edges
        
        # Store histogram data for Origin export (using the ACTUAL values from matplotlib)
        self.histogram_data = {
            'bin_edges': id_bin_edges,  # Use the actual bin edges from matplotlib
            'id_counts': id_counts,
            'ood_counts': ood_counts
        }
        ax.set_xlabel('Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('One-Class SVM Score Distribution', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 2. Anomaly detection rate comparison
        ax = axes[1]
        methods = ['One-Class SVM']
        in_rates = [svm_result['in_anomaly_rate'] * 100]
        ood_rates = [svm_result['ood_anomaly_rate'] * 100]
        
        x = np.arange(len(methods))
        width = 0.35
        bars1 = ax.bar(x - width/2, in_rates, width, label='ID', 
                      alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, ood_rates, width, label='OOD', 
                      alpha=0.7, color='orange', edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Anomaly Detection Rate (%)', fontsize=12)
        ax.set_title('Anomaly Detection Rate Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=11)
        ax.set_ylim([0, max(max(in_rates), max(ood_rates)) * 1.2])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save image
        output_path = os.path.join(os.path.dirname(self.in_dist_folder), 'ood_detection_results.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization results saved to: {output_path}")
        plt.show()
        
        # Save data to CSV
        self.save_results_to_csv(svm_result)
    
    def run_detection(self):
        """
        Run complete OOD detection pipeline
        """
        print("=" * 60)
        print("OOD Detection Program Started")
        print("=" * 60)
        
        # 1. Extract features
        print("\nStep 1: Feature Extraction")
        self.in_dist_features = self.load_and_extract_features(
            self.in_dist_folder, "Extracting in-distribution features"
        )
        self.ood_features = self.load_and_extract_features(
            self.ood_folder, "Extracting OOD features"
        )
        
        print(f"\nFeature dimensions: {self.in_dist_features.shape[1]}")
        print(f"In-distribution samples: {self.in_dist_features.shape[0]}")
        print(f"OOD samples: {self.ood_features.shape[0]}")
        
        # 2. Statistical test (auxiliary information)
        self.statistical_test()
        
        # 3. One-Class SVM detection
        svm_result = self.detect_with_one_class_svm()
        
        # 4. Generate visualization
        self.visualize_results(svm_result)
        
        # 5. Summary
        print("\n" + "=" * 60)
        print("Detection Results Summary")
        print("=" * 60)
        print(f"\n✓ Successfully extracted {self.in_dist_features.shape[1]} dimensional features")
        print(f"✓ In-distribution data: {self.in_dist_features.shape[0]} images")
        print(f"✓ OOD data: {self.ood_features.shape[0]} images")
        
        print("\nOne-Class SVM Detection Results:")
        print(f"  - In-distribution anomaly detection rate: {svm_result['in_anomaly_rate']*100:.2f}%")
        print(f"  - OOD data anomaly detection rate: {svm_result['ood_anomaly_rate']*100:.2f}%")
        print(f"  - In-distribution average score: {svm_result['in_scores'].mean():.4f}")
        print(f"  - OOD data average score: {svm_result['ood_scores'].mean():.4f}")
        
        print("\n" + "=" * 60)
        print("Conclusion:")
        if svm_result['ood_anomaly_rate'] > svm_result['in_anomaly_rate'] * 2:
            print("✓ The anomaly detection rate of OOD data is significantly higher than in-distribution data,")
            print("  proving that OOD data has a different distribution compared to in-distribution data.")
        else:
            print("The distribution difference between OOD data and in-distribution data requires further analysis.")
        print("=" * 60)
        
        return svm_result


if __name__ == "__main__":
    # Set folder paths
    in_dist_folder = r'D:\Desktop\OOD\hydrofig'
    ood_folder = r'D:\Desktop\OOD\ood_fig'
    
    # Create detector and run
    detector = OODDetector(in_dist_folder, ood_folder)
    results = detector.run_detection()

