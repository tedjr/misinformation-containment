# Misinformation Containment Simulations  

This repository contains the simulation notebook developed for my MSc dissertation:  
**“Minimizing Misinformation Spread on Social Networks” (University of Leeds, 2025).**  

The notebook implements GPU-accelerated simulations of misinformation diffusion on the SNAP Facebook combined network, comparing multiple containment strategies under two canonical diffusion models.  

---

## 📘 Project Overview  

Misinformation propagates rapidly in social networks due to:  
- **Scale-free degree distributions** (hubs drive cascades)  
- **Small-world paths** (fast global reach)  
- **Community modularity** (local reinforcement)  

This project evaluates containment strategies under two diffusion models:  
- **Independent Cascade (IC)** – stochastic, opportunity-driven spreading  
- **Linear Threshold (LT)** – reinforcement-driven spreading  

Containment strategies tested:  
- **Node blocking** (degree-based / betweenness-based)  
- **Edge blocking** (betweenness-based / random baseline)  
- **Truth campaigns** (competitive diffusion with truth-priority tie-breaking)  

---

## ⚙️ Requirements  

Python 3.10+ and a CUDA-enabled GPU.  

Dependencies:  
- RAPIDS cuGraph  
- CuPy  
- NetworkX  
- Pandas / NumPy  
- Matplotlib / Seaborn  
- SciPy  

Install requirements with:  

```bash
pip install cupy-cuda12x cugraph-cu12 pandas numpy networkx matplotlib seaborn scipy

