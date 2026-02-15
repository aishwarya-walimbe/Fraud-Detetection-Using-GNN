Financial fraud costs the global economy over $5 trillion annually, yet most detection systems analyse transactions in isolation — missing the bigger picture.
This project takes a network-first approach: every bank account is a node, every money transfer is an edge, and a Graph Neural Network learns to identify fraudulent accounts by studying not just their own behaviour, but the behaviour of everyone they're connected to.
Built on the PaySim dataset (6.3M synthetic mobile money transactions) and powered by GraphSAGE and Graph Attention Networks (GAT), this system tackles the extreme class imbalance (~0.13% fraud) using Focal Loss — the same technique used in state-of-the-art object detection — and automatically tunes its decision threshold on a validation set to maximise fraud recall without drowning analysts in false alarms.
The graph-based approach mirrors what production fraud systems at companies like PayPal, Stripe, and Mastercard use — making this not just an academic exercise, but a practical, industry-relevant architecture.

📌 Table of Contents

Why Graph Neural Networks?
Dataset
Project Structure
Architecture
Key Improvements Over Baseline
Results
How to Run
Visualisations
Future Work

Overview
Traditional fraud detection treats each transaction independently. This project takes a fundamentally different approach — it models the entire transaction network as a graph, where:

- Nodes = Bank accounts (senders & receivers)
- Edges = Transactions between accounts (directed)
- Node features = Behavioural statistics (12 features per account)
- Edge features = Per-transaction attributes (amount, type, time, balance changes)

By examining not just individual transactions but the patterns of connections between accounts, the GNN can detect fraud rings and suspicious behaviour that isolated transaction analysis would miss.

**Why Graph Neural Networks?**

 - Patterns across connected accounts
 - Relationships between accounts

**Dataset**
PaySim — A synthetic mobile money transaction simulator calibrated against real data from a mobile money service in Africa.
Property
- ValueTotal transactions ~6.3 million
- Features: 11 columns
- Fraud rate ~0.13%
- Fraud types TRANSFER, CASH_OUT only
- Source Kaggle — https://www.kaggle.com/datasets/mtalaltariq/paysim-data

**Project Structure**

fraud-detection-gnn/

├── README.md
├── preprocess.py
├── model.py
├── train.py
├── requirements.txt
├── outputs/
└── data/


**Architecture**
Two model variants are implemented:

1. ImprovedGraphSAGE (recommended for speed)
Input (12 features)
    │
   
SAGEConv Layer 1 → BatchNorm → ReLU → Dropout + skip connection (residual)
    │ 
    
SAGEConv Layer 2 → BatchNorm → ReLU → Dropout + residual
    │ 
    
SAGEConv Layer 3 → BatchNorm → ReLU → Dropout + residual
    │ 
    
Linear(128→64) → ReLU → Dropout → Linear(64→2)
    │
    
Fraud / Normal


3. GATFraudNet (recommended for best F1)

Input (12 node features + 5 edge features)
    │
    
GATConv (4 attention heads) → BatchNorm → ELU → Dropout
    │
    
GATConv (1 head)  → BatchNorm → ELU
    │
    
Linear(64→64) → ReLU → Dropout → Linear(64→2)
    │
    
Fraud / Normal


**Results**

| Metric        | This Project  |
| ------------- | ------------- |
| Accuracy      | ~ 97.5%       |
| F1 Score      | ~0.55-0.65    |
| ROC-AUC       | ~0.93-0.96    |
| Precision     | ~60-70%       |
| Recall        | ~ 50-60%      |

Results vary slightly per run due to random seeds in graph construction.
The improvement comes primarily from Focal Loss + edge features + threshold tuning.

**Training Process**
<img width="1800" height="600" alt="loss_f1_curve" src="https://github.com/user-attachments/assets/f807eafa-08b8-4b65-95d6-9ce370a85111" />

**Test Performance**
<img width="750" height="600" alt="confusion_matrix" src="https://github.com/user-attachments/assets/55ed1fdf-25e1-40d5-b28b-fb728e43c23b" />

<img width="900" height="750" alt="roc_curve" src="https://github.com/user-attachments/assets/3c4bbe73-c923-496b-99b9-35b9936b597e" />







