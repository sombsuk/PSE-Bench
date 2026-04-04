# PSE-Bench

**Benchmark for evaluating Large Language Models across Process Systems Engineering domains**

## Paper

> Sukpancharoen, S. & Srinophakun, T.R. (2026). Benchmarking AI prompt performance of large language models across chemical process systems engineering domains using multi-judge evaluation.(under review).

## Overview

PSE-Bench is a benchmark of **200 open-ended questions** across four core domains of Process Systems Engineering (PSE), designed to evaluate LLM performance using a **multi-judge evaluation framework** with five independent AI judges.

| Domain | Abbr. | Questions | Topics |
|--------|-------|-----------|--------|
| Process Modeling & Simulation | MOD | 50 | Thermodynamic modeling, flash calculations, reactor modeling, dynamic simulation |
| Process Optimization | OPT | 50 | LP/NLP/MILP/MINLP, pinch analysis, real-time optimization, multi-objective optimization |
| Machine Learning for Chemical Processes | ML | 50 | Soft sensors, fault detection, PINNs, transfer learning |
| Process Design & Systems Engineering | DES | 50 | HAZOP, LOPA, process intensification, plantwide control, FEED |

## Models Evaluated

| Model | Version | Provider |
|-------|---------|----------|
| DeepSeek-V3 | deepseek-chat | DeepSeek |
| Claude Sonnet 4 | claude-sonnet-4-20250514 | Anthropic |
| Gemini 2.5 Flash | gemini-2.5-flash | Google |
| GPT-4o | gpt-4o-2024-08-06 | OpenAI |
| Llama 3.3 70B | llama-3.3-70b-versatile | Meta (via Groq) |

All responses collected on **February 23, 2026** under zero-shot conditions.

## Key Results

- **DeepSeek** achieved the highest performance (element coverage: 78.1%)
- Consistent domain difficulty pattern: **DES > ML > OPT > MOD**
- Human validation confirmed AI-human agreement (Spearman rs = 0.416, p < 0.001)
- Model rankings stable across all alternative scoring schemes (rs = 1.000)

## Repository Structure

```
PSE-Bench/
├── ChemEng_Bench_200_GroundTruth.xlsx   # Benchmark: 200 questions + ground truths + rubrics
├── score_benchmark_v3.py                 # Multi-judge evaluation script
│
├── *_responses.xlsx                      # Raw LLM responses (5 files)
│   ├── deepseek_responses.xlsx
│   ├── claude_responses.xlsx
│   ├── gemini_responses_v2.xlsx
│   ├── gpt4o_responses.xlsx
│   └── llama_responses.xlsx
│
├── *_evaluation_v3.xlsx                  # Multi-judge evaluation results (5 files)
│   ├── deepseek_evaluation_v3.xlsx
│   ├── claude_evaluation_v3.xlsx
│   ├── gemini_evaluation_v3.xlsx
│   ├── gpt4o_evaluation_v3.xlsx
│   └── llama_evaluation_v3.xlsx
│
├── ChemEng_Bench_Summary_v3.xlsx         # Summary statistics
├── Human_Validation_Final.xlsx           # Human expert validation data
└── README.md
```

## Evaluation Framework

Each response is scored by **5 independent AI judges** against a **7-element rubric**. The composite score is:

```
Overall = 0.15 × ROUGE-1 + 0.15 × ROUGE-L + 0.20 × Cosine + 0.50 × Element%
```

Grades: **Good** (≥ 0.50) | **Fair** (0.35–0.49) | **Poor** (< 0.35)

## Usage

To reproduce the evaluation:

1. Install dependencies:
```bash
pip install openai anthropic google-genai openpyxl
```

2. Add your API keys in `score_benchmark_v3.py`

3. Run:
```bash
python score_benchmark_v3.py
```

## Citation

If you use PSE-Bench in your research, please cite:

```bibtex
@article{sukpancharoen2026psebench,
  title={Benchmarking {AI} prompt performance of large language models across chemical process systems engineering domains using multi-judge evaluation},
  author={Sukpancharoen, Somboon and Srinophakun, Thongchai Rohitatisha},
  journal={Computers \& Chemical Engineering},
  year={2026},
  note={Under review}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Somboon Sukpancharoen** — Department of Agricultural Engineering, Khon Kaen University, Thailand
- Email: sombsuk@kku.ac.th
