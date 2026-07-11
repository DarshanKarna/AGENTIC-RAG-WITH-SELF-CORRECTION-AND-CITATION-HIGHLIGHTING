# 📊 B.Tech AI Project: RAG Evaluation Report
## Comparative Metrics: Naive Generation (Before) vs Self-Correcting LangGraph (After)

This report presents the comparative metrics of the RAG system **Before** vs **After** the NLI-based sentence-level self-correction loops. Evaluated using a local Gemma `gemma` LLM judge via Ollama.

### 📈 Core Metrics Comparative Table

| Query | Initial Groundedness | Final Groundedness | Initial Relevance | Final Relevance | Improvement Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| What is the function of the BRCA1 gene? | **5/5** | **5/5** | **5/5** | **5/5** | ✅ Perfect Groundedness Maintained |
| What is the role of p53 protein in cell cycle control? | **5/5** | **5/5** | **5/5** | **5/5** | ✅ Perfect Groundedness Maintained |
| Who won the FIFA World Cup in 2022? | **5/5** | **5/5** | **1/5** | **1/5** | ✅ Perfect Groundedness Maintained |


### 🧬 Query-by-Query Comparative Detail

#### 1. Query: "What is the function of the BRCA1 gene?"

##### ❌ Initial Draft (Before NLI Self-Correction)
```text
The BRCA1 gene is a tumor suppressor gene.
The BRCA1 gene plays a role in regulating DNA repair and maintenance.
The BRCA1 gene is involved in cell cycle progression.
The BRCA1 gene is involved in transcription.
The BRCA1 gene is involved in cell survival/apoptosis.
The BRCA1 gene maintains genomic integrity.
The BRCA1 gene acts as a ubiquitin E3 ligase.
```
- **Groundedness**: 5/5
- **Relevance**: 5/5

##### ✅ Verified Final Answer (After NLI Self-Correction)
```text
The BRCA1 gene is a tumor suppressor gene.
The BRCA1 gene plays a role in regulating DNA repair and maintenance.
The BRCA1 gene acts as a ubiquitin E3 ligase.
Inherited mutations in the BRCA1 gene are associated with a higher risk of breast and ovarian cancer.
The BRCA1 protein has multiple roles in regulating DNA repair and maintenance, cell cycle progression, transcription, and cell survival/apoptosis.
```
- **Groundedness**: 5/5
- **Relevance**: 5/5
- **Critic Reasoning**: *The answer is fully supported by the retrieved context, which explicitly states the functions and roles of the BRCA1 gene, including its role as a tumor suppressor, its involvement in DNA repair and maintenance, and its association with breast and ovarian cancer. The answer directly addresses the user's question and provides a comprehensive overview of the BRCA1 gene's functions.*

---

#### 2. Query: "What is the role of p53 protein in cell cycle control?"

##### ❌ Initial Draft (Before NLI Self-Correction)
```text
The p53 protein has a key role as a transcription factor in limiting oncogenesis through cell cycle arrest. 
The p53 protein acts as a cell cycle checkpoint protein, halting the cell cycle upon DNA damage. 
The p53 protein initiates cell cycle arrest as one of its growth suppressive functions. 
The p53 protein maintains the homeostasis of cellular metabolism and redox balance in cells through regulation of cell cycle arrest. 
The p53 protein exerts its function in tumor suppression through its transcriptional regulation of target genes to initiate cell cycle arrest.
```
- **Groundedness**: 5/5
- **Relevance**: 5/5

##### ✅ Verified Final Answer (After NLI Self-Correction)
```text
The p53 protein has a role in limiting oncogenesis through several growth suppressive functions.
The p53 protein initiates apoptosis, senescence, or cell cycle arrest.
The p53 protein is involved in halting the cell cycle upon DNA damage as a cell cycle checkpoint protein.
The p53 protein exerts its function in tumor suppression through transcriptional regulation.
Cell cycle arrest is a function of the p53 protein.
```
- **Groundedness**: 5/5
- **Relevance**: 5/5
- **Critic Reasoning**: *The answer is fully supported by the retrieved context, which explicitly states the roles of p53 protein in cell cycle control, including limiting oncogenesis, initiating apoptosis, senescence, or cell cycle arrest, and halting the cell cycle upon DNA damage. The answer directly addresses the user's question and provides relevant information from the context.*

---

#### 3. Query: "Who won the FIFA World Cup in 2022?"

##### ❌ Initial Draft (Before NLI Self-Correction)
```text
I'm sorry, but I couldn't find any relevant biomedical information in the local database to answer your question.
```
- **Groundedness**: 5/5
- **Relevance**: 1/5

##### ✅ Verified Final Answer (After NLI Self-Correction)
```text
I'm sorry, but I couldn't find any relevant biomedical information in the local database to answer your question.
```
- **Groundedness**: 5/5
- **Relevance**: 1/5
- **Critic Reasoning**: *Fallback answer indicating database mismatch.*

---

