#!/bin/bash
python3 baseline/runners/run_raptor_qa.py --dataset hotpotqa --llm_backend qwen --retrieval_only --content_risk_retries 0
