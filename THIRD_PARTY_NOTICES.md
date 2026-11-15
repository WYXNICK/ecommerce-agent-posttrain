# Third-party components and provenance

The project-specific contribution is the data preparation, shared agent execution, corrective-data workflow, training integration, controlled experiments, and evidence organization described in the project documentation. The simulator, base model, and training libraries are separate dependencies; their authorship is not claimed by this project.

| Component | Source / identity | Use |
|---|---|---|
| ShopSimulator | [ShopAgent-Team/ShopSimulator](https://github.com/ShopAgent-Team/ShopSimulator); exact embedded revision in [EMBEDDED_SOURCE.json](environments/ShopSimulator/EMBEDDED_SOURCE.json) | Environment, product data, tools and Reward v3 |
| Initial shopping training scaffold | [YYHDBL/shopping-grpo-longhorizon](https://github.com/YYHDBL/shopping-grpo-longhorizon) | Earlier code/data lineage; retained source history and 200-task assets |
| Qwen3.5-2B | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B); fixed revision in [execution.json](next/configs/execution.json) | Student starting model |
| veRL / vLLM / Transformers / PEFT / FLA | Versions in execution and training manifests | Policy loss, inference, model and adapter training, kernels |
| SDPO | [lasgroup/SDPO](https://github.com/lasgroup/SDPO) | Research reference for the independently documented experimental branch |

Repository branding and project documentation are independent of the initial scaffold. Historical source identities, factual provenance, copyright notices and artifact hashes remain intact.

No top-level license granting redistribution of the entire combined repository was found in the checked-in snapshot. This document does not grant such a license or relicense third-party code, model weights, or product data. Applicable upstream terms must be checked before public redistribution; repository visibility alone is not a license. Model/API access and private artifact storage are also separate from code availability.
