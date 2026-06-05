# ModernColBERT Models on Hugging Face (2026)

This document lists the most popular and foundational **ModernColBERT** models hosted on Hugging Face as of 2026. These models combine the efficient, long-context capabilities of the ModernBERT encoder backbone with PyLate / ColBERT late-interaction (MaxSim) retrieval.

---

## 1. Foundational Baseline Models

*   **[lightonai/GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1)**
    *   **Description**: The original baseline model in the ModernColBERT family. Trained on MS MARCO via knowledge distillation, it serves as the foundational architecture for downstream fine-tuning.
    *   **Context Length**: 8192 tokens
    *   **Parameters**: ~149 million

---

## 2. Multilingual SauerkrautLM Series (VAGOsolutions)

*   **[VAGOsolutions/SauerkrautLM-Multi-ModernColBERT](https://huggingface.co/VAGOsolutions/SauerkrautLM-Multi-ModernColBERT)**
    *   **Description**: A multilingual variant based on `GTE-ModernColBERT-v1`, pre-trained on an additional 4.6 billion tokens. Optimised for German, English, French, Spanish, Italian, Portuguese, and Dutch.
    *   **Context Length**: 8192 tokens
    *   **Parameters**: ~149 million

*   **[VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT](https://huggingface.co/VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT)**
    *   **Description**: Advanced multilingual retriever utilizing knowledge distillation from large models (like Qwen3) and LaserRMT compression. Optimised for reasoning-heavy retrieval in European languages.
    *   **Context Length**: 8192 tokens
    *   **Parameters**: ~149 million

---

## 3. Specialized Task Encoders (LightonAI)

*   **[lightonai/Reason-ModernColBERT](https://huggingface.co/lightonai/Reason-ModernColBERT)**
    *   **Description**: Fine-tuned on the `reasonir-hq` dataset to perform well on reasoning-intensive retrieval tasks.
    *   **Context Length**: 8192 tokens
    *   **Parameters**: ~150 million

*   **[lightonai/Agent-ModernColBERT](https://huggingface.co/lightonai/Agent-ModernColBERT)**
    *   **Description**: Tailored for agentic workflows and search tools, incorporating reasoning traces directly to boost agent retrieval accuracy.
    *   **Context Length**: 8192 tokens
    *   **Parameters**: ~150 million

---

## 4. Smaller ColBERT Models (VAGOsolutions)

*   **[VAGOsolutions/SauerkrautLM-Multi-ColBERT-15m](https://huggingface.co/VAGOsolutions/SauerkrautLM-Multi-ColBERT-15m)**
    *   **Description**: Extremely lightweight multilingual ColBERT model optimized for low-resource environments and high-throughput use-cases.
    *   **Parameters**: ~15 million

*   **[VAGOsolutions/SauerkrautLM-Multi-ColBERT-33m](https://huggingface.co/VAGOsolutions/SauerkrautLM-Multi-ColBERT-33m)**
    *   **Description**: Compact multilingual ColBERT model offering a strong balance between parameter count/inference speed and retrieval quality.
    *   **Parameters**: ~33 million

