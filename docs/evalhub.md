# EvalHub Community Suites (`synapsekit bench`)

EvalHub is the built-in community registry for benchmarking SynapseKit pipelines.

## Available suites

| Suite | Description | Cases | Community baseline |
|---|---|---:|---:|
| `community/customer-support` | Ticket resolution, tone, escalation | 20 | 0.74 |
| `community/code-generation` | Python, JS, SQL correctness | 30 | 0.71 |
| `community/rag-general` | Faithfulness, relevancy, groundedness | 25 | 0.76 |
| `community/summarization` | ROUGE-like, length, coverage | 15 | 0.73 |
| `community/qa-hotpotqa` | Multi-hop Q&A from HotpotQA-style tasks | 50 | 0.69 |

## List suites

```bash
synapsekit bench --list
```

## Run a suite against your pipeline/model

```bash
synapsekit bench --suite community/customer-support --model gpt-4o-mini
```

### Optional flags

- `--provider openai` (override provider auto-detection)
- `--api-key ...` (or use provider env var)
- `--pipeline my_module:run_pipeline` (custom callable pipeline)
- `--format json` (machine-readable output)
- `--limit N` (run first N cases)

## Publish your own suite

```bash
synapsekit bench --publish my_evals/ --name myorg/rag-finance
```

By default this command:

1. Packages `my_evals/` as a zip bundle
2. Clones the registry repository
3. Adds your suite under the registry directory
4. Opens a GitHub PR

Use `--no-submit` to package only:

```bash
synapsekit bench --publish my_evals/ --name myorg/rag-finance --no-submit
```
