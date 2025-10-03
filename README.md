Clustering with K-Means


Objective:

The primary goal of this task was to perform **unsupervised machine learning** using the **K-Means clustering algorithm** to segment customers based on their Annual Income and Spending Score. This analysis is crucial for businesses to identify distinct customer groups for targeted marketing strategies.

------------------------------------------------------------------------------------------------------------------------

Dataset:

The project utilizes the **Mall Customer Segmentation Dataset** sourced from Kaggle, focusing on `Annual Income (k$)` and `Spending Score (1-100)`.

------------------------------------------------------------------------------------------------------------------------

Tools and Libraries Used:

* Python 3.x
* Pandas
* Scikit-learn (sklearn)
* Matplotlib / Seaborn

------------------------------------------------------------------------------------------------------------------------

Output:

**Elbow Method Plot** | Shows the **optimal number of clusters is $K=5$**, based on the point of diminishing returns for WCSS (Inertia). | [View Elbow Method Plot](https://github.com/NAVEENAPRIYA-06/Task-8-Clustering/blob/main/Screenshots/Elbow_method_plot.png) |
| **Customer Segments Plot** | The final visualization of the **5 color-coded customer clusters** based on Income vs. Spending Score. | [View Customer Segments Plot](https://github.com/NAVEENAPRIYA-06/Task-8-Clustering/blob/main/Screenshots/Customer_segment_plot.png) |
| **Terminal Output** | Snapshot of the final script run, showing the calculated **Silhouette Score** ($\approx 0.55$) and the **Cluster Summary** (mean values). | [View Terminal Output](https://github.com/NAVEENAPRIYA-06/Task-8-Clustering/blob/main/Screenshots/Terminal_output.png) |
