<h1 align="center">Muhammad Abdullah Rasheed</h1>

<p align="center">
  <b>ML Research Engineer</b> · scientific computing, neural operators, and applied machine learning
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/abdullahrasheed-/"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn"></a>
  <a href="mailto:abdullahrasheed45@gmail.com"><img src="https://img.shields.io/badge/Email-333333?style=flat-square&logo=gmail&logoColor=white" alt="Email"></a>
  <a href="http://www.techvibes360.com"><img src="https://img.shields.io/badge/Website-techvibes360.com-1f6feb?style=flat-square&logo=googlechrome&logoColor=white" alt="Website"></a>
</p>

---

## About

ML Research Engineer at **Aeris UK**, based in London. MSc Data Science
(Distinction). I work at the intersection of **numerical methods and machine
learning** — reinforcement learning and simulation software, PDE solvers and
neural-operator surrogates, distributed training pipelines, and LLM-driven
agents. I care about code that is correct, tested, and honest about its
trade-offs: most of my recent projects ship with a validation suite, CI, and
benchmarks rather than just a notebook.

- Currently building **reinforcement-learning and simulation software** at Aeris UK.
- Research interest in **operator learning for PDEs** and high-performance numerical computing.
- Comfortable across the stack: C++/OpenMP kernels, PyTorch (DDP/FSDP), and full-stack deployment.
- **15+ ML models deployed** across research and production settings.
- Reach me at **abdullahrasheed45@gmail.com** or on [LinkedIn](https://www.linkedin.com/in/abdullahrasheed-/).

<!-- OSS-CONTRIBUTIONS:START -->

## Open source contributions

| Repo | Merged pull request(s) |
|---|---|
| [huggingface/transformers](https://github.com/huggingface/transformers) (163k⭐) | [#47663](https://github.com/huggingface/transformers/pull/47663) fix: correct text input validation logic in 8 multimodal processors (and → or) |

Also have 18 pull requests currently under review across 12 other repositories:

<details>
<summary>Open pull requests</summary>

| Repo | Pull request(s) |
|---|---|
| [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) (36k⭐) | [#10112](https://github.com/stanfordnlp/dspy/pull/10112) dedupe signatures by content · [#10085](https://github.com/stanfordnlp/dspy/pull/10085) roll nested usage-tracking scopes into enclosing tracker |
| [Lightning-AI/pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning) (31k⭐) | [#21866](https://github.com/Lightning-AI/pytorch-lightning/pull/21866) bump `rich` upper bound for v15 support |
| [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric) (24k⭐) | [#10762](https://github.com/pyg-team/pytorch_geometric/pull/10762) raise on numpy arrays stored as tensor attributes · [#10756](https://github.com/pyg-team/pytorch_geometric/pull/10756) verify TLS certificates when downloading datasets |
| [huggingface/datasets](https://github.com/huggingface/datasets) (22k⭐) | [#8368](https://github.com/huggingface/datasets/pull/8368) reject duplicate column names in `select_columns` |
| [huggingface/peft](https://github.com/huggingface/peft) (21k⭐) | [#3477](https://github.com/huggingface/peft/pull/3477) remove forward hooks when the wrapped block raises |
| [onnx/onnx](https://github.com/onnx/onnx) (21k⭐) | [#8215](https://github.com/onnx/onnx/pull/8215) fix node test directory count in `generate-data` |
| [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (13k⭐) | [#3971](https://github.com/EleutherAI/lm-evaluation-harness/pull/3971) & [#3970](https://github.com/EleutherAI/lm-evaluation-harness/pull/3970) fix `ValueError` on `batch_size="auto"` · [#3951](https://github.com/EleutherAI/lm-evaluation-harness/pull/3951) include `generation_kwargs` in request cache key |
| [huggingface/accelerate](https://github.com/huggingface/accelerate) (10k⭐) | [#4138](https://github.com/huggingface/accelerate/pull/4138) reject sharded-checkpoint paths that escape the checkpoint folder |
| [bentoml/BentoML](https://github.com/bentoml/BentoML) (9k⭐) | [#5676](https://github.com/bentoml/BentoML/pull/5676) keep every URL value of a multipart list field |
| [cvxpy/cvxpy](https://github.com/cvxpy/cvxpy) (6k⭐) | [#3470](https://github.com/cvxpy/cvxpy/pull/3470) stop internal parameter checks warning about sparse `.value` reads |
| [huggingface/lighteval](https://github.com/huggingface/lighteval) (2.5k⭐) | [#1314](https://github.com/huggingface/lighteval/pull/1314) fix always-true conditional hiding `DEFAULT_FORMAT` · [#1313](https://github.com/huggingface/lighteval/pull/1313) fix litellm judge sending `max_tokens` as a 1-tuple |
| [Lightning-AI/torchmetrics](https://github.com/Lightning-AI/torchmetrics) (2.5k⭐) | [#3441](https://github.com/Lightning-AI/torchmetrics/pull/3441) & [#3440](https://github.com/Lightning-AI/torchmetrics/pull/3440) fix argument validation across multiple metrics |

</details>

<!-- OSS-CONTRIBUTIONS:END -->

## Featured work

| Project | What it is | Stack |
|---------|------------|-------|
| **[fno-darcy-flow](https://github.com/AbdullahRasheed45/fno-darcy-flow)** | Fourier Neural Operator surrogate for 2D Darcy flow, with a from-scratch finite-volume data generator and 1-vs-N-GPU scaling benchmarks. Trains identically under **DDP** and **FSDP/ZeRO-3**. | PyTorch, DDP/FSDP, NumPy |
| **[darcy-cpp](https://github.com/AbdullahRasheed45/darcy-cpp)** | Matrix-free **multigrid** finite-volume solver for 2D Darcy flow. Grid-independent CG iteration count, OpenMP-parallel, with zero-copy Python bindings — up to 16× faster than a sparse direct solve. | C++17, OpenMP, pybind11 |
| **[Adaptive-Micro-Gesture-Recognition](https://github.com/AbdullahRasheed45/Adaptive-Micro-Gesture-Recognition)** | Master's thesis: a **3D CNN** that recognises subtle hand movements as computer commands to improve accessibility for motor-impaired users (94.6% accuracy). | TensorFlow, OpenCV, Flask |
| **[Grammify](https://github.com/AbdullahRasheed45/Grammify)** | A **T5 Seq2Seq** grammar-correction app with real-time error analysis, deployed on Hugging Face Spaces. | Transformers, FastAPI, Streamlit |
| **[crypto_trading_ai_agents](https://github.com/AbdullahRasheed45/crypto_trading_ai_agents)** | Agentic system that forecasts price movements and produces buy/sell signals. | Python, LLM agents |

## Tech stack

**Languages** · Python, C++, SQL, JavaScript/TypeScript, R

**ML / scientific** ·
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)

**Systems / tooling** ·
![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=flat-square&logo=cplusplus&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-064F8C?style=flat-square&logo=cmake&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?style=flat-square&logo=postgresql&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=flat-square&logo=amazonaws&logoColor=white)

**Web** ·
![React](https://img.shields.io/badge/React-20232A?style=flat-square&logo=react&logoColor=61DAFB)
![Next.js](https://img.shields.io/badge/Next.js-000000?style=flat-square&logo=nextdotjs&logoColor=white)
![Node.js](https://img.shields.io/badge/Node.js-339933?style=flat-square&logo=nodedotjs&logoColor=white)

## Experience & education

- **ML Research Engineer** — Aeris UK *(current)* · reinforcement learning and simulation software
- **Data Analyst** — Digital Pulse 360
- **Full-Stack Developer** — ATnR, Karachi
- **MSc Data Science** — Anglia Ruskin University, Cambridge *(Distinction)*
- **BSc Economics** — Institute of Business Administration (IBA), Karachi *(Dean's Honour List)*

**Certifications** · Google Advanced Data Analytics · Google AI Essentials · Google Data Analytics Professional · Tableau Specialization (UC Davis)
