FROM python:3.11-slim

WORKDIR /SQLAgentProject

# CPU by default; override for CUDA: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

# Prevent HuggingFace tokenizer from forking background threads (crashes in fork-restricted containers)
ENV TOKENIZERS_PARALLELISM=false
# Limit OpenMP thread count so inference stays within the CPU quota
ENV OMP_NUM_THREADS=1

COPY src/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "torch==2.4.1" --index-url ${TORCH_INDEX} && \
    pip install --no-cache-dir "transformers==5.7.0"

COPY src/ .

# ml_gr.py resolves the model path as Path(__file__).parent.parent.parent / "data_science/..."
# With src/ copied flat into /SQLAgentProject, that traversal lands at filesystem root,
# so the model must live at /data_science/train_state/final_model/.
COPY data_science/train_state/final_model/ /data_science/train_state/final_model/

EXPOSE 8000

CMD ["python", "main.py"]
