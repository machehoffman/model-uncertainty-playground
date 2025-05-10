# model-uncertainty-playground 🎯

🔬 A hands-on playground for exploring **model calibration**, **uncertainty estimation**, and **meta-loss prediction** in deep learning classifiers.

This project includes:
- ✅ Inputting a trained model and dataset to evaluate calibration, uncertainty, and meta-loss prediction
- 📊 Evaluating calibration with **ECE** and **reliability diagrams**
- 🔍 Computing predictive uncertainty using:
  - Entropy & margin-based metrics
  - MC Dropout sampling
- 🧠 Training a secondary model to **predict loss/uncertainty**
- 🔁 Applying **post-hoc calibration techniques** (Temperature Scaling, Platt, etc.)


## Goals
- Understand when and why models are **overconfident**
- Apply and compare different uncertainty estimation methods
- Use meta-models to detect **difficult or novel inputs**

---

👨‍💻 *Built as a research-driven sandbox for trustworthy AI.*

## 🚀 Local Setup

```bash
git clone git@github.com:machehoffman/model-uncertainty-playground.git
cd model-uncertainty-playground
```

##  🐳 Docker Setup

```bash
docker build -t model-uncertainty-playground .
docker run -it model-uncertainty-playground
```
